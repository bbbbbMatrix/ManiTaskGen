"""
custom_geometry_helper_new/convex_hull_processor.py
This module provides functionality to process convex hulls in 2D space.
In the pipline, because platforms are maintained as convexes, it is used to handle geometric operations such as refining receptacles. 
 


"""

from typing import List
import numpy as np
from typing import List
from shapely.geometry import Polygon, LinearRing, LineString, Point
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from .basic_geometries import Basic2DGeometry


class ConvexHullProcessor_2d:
    def __init__(self, vertices: List[List[float]], heading=(1, 0)):
        self.vertices = np.array(vertices)
        if self.vertices.shape[1] > 2:
            self.vertices = self.vertices[:, :2]
        # convexhull.vertices is the index of the vertices on the convex hull
        # for 2d hull, convexhull.volume is the area of the hull, convexhull.area is the perimeter of the hull

        self.convex_hull = ConvexHull(self.vertices, qhull_options="QJ")

        self.heading = np.array(heading)

    def __repr__(self):
        return f"ConvexHullProcessor_2d({self.vertices}),{self.heading}"

    def plot_convex_hull(self, ax):
        convex_hull_vertices = self.get_vertices_on_convex_hull()
        convex_hull_vertices = np.vstack(
            [convex_hull_vertices, convex_hull_vertices[0]]
        )
        ax.plot(convex_hull_vertices[:, 0], convex_hull_vertices[:, 1], "r-")
        ax.plot(self.vertices[:, 0], self.vertices[:, 1], "bo")
        ax.set_aspect("equal")
        plt.show()

    def get_vertices_on_convex_hull(self):
        return self.vertices[self.convex_hull.vertices]

    def get_vertices_on_convex_hull_with_indices(self):
        return self.vertices[self.convex_hull.vertices], self.convex_hull.vertices

    def get_indented_vertices_on_convex_hull(self, indent_ratio=0.2):
        center = np.mean(self.vertices[self.convex_hull.vertices], axis=0)
        convex_hull_vertices = self.get_vertices_on_convex_hull()
        sample_vertices = [
            (vertex - center) * (1 - indent_ratio) + center
            for vertex in convex_hull_vertices
        ]
        return np.array(sample_vertices)

    def __add__(self, offset_2d):
        offset = np.array(offset_2d)
        return ConvexHullProcessor_2d(self.vertices + offset)

    def get_area(self):
        return self.convex_hull.volume

    @staticmethod
    def get_headed_bbox(vertices, heading=(1, 0)):
        heading = np.array(heading)
        perp_heading = np.array([-heading[1], heading[0]])

        min_proj_heading = np.inf
        max_proj_heading = -np.inf
        min_proj_perp_heading = np.inf
        max_proj_perp_heading = -np.inf

        for vertex in vertices:
            proj_heading = float(np.dot(vertex, heading))
            proj_perp_heading = float(np.dot(vertex, perp_heading))

            if proj_heading < min_proj_heading:
                min_proj_heading = proj_heading
            if proj_heading > max_proj_heading:
                max_proj_heading = proj_heading
            if proj_perp_heading < min_proj_perp_heading:
                min_proj_perp_heading = proj_perp_heading
            if proj_perp_heading > max_proj_perp_heading:
                max_proj_perp_heading = proj_perp_heading

        bbox = [
            min_proj_heading * heading + min_proj_perp_heading * perp_heading,
            min_proj_heading * heading + max_proj_perp_heading * perp_heading,
            max_proj_heading * heading + max_proj_perp_heading * perp_heading,
            max_proj_heading * heading + min_proj_perp_heading * perp_heading,
        ]

        return np.array(bbox)

    def can_fit_in(self, rect, allow_rotate=False):
        rect = np.array(rect)
        rect_heading = rect[1] - rect[0]
        rect_width, rect_height = (rect[1] - rect[0]).dot(rect[1] - rect[0]), (
            rect[2] - rect[1]
        ).dot(rect[2] - rect[1])
        rect_heading = rect_heading / np.linalg.norm(rect_heading)
        bbox_in_rect = ConvexHullProcessor_2d.get_headed_bbox(
            self.vertices, rect_heading
        )
        bbox_width, bbox_height = (bbox_in_rect[1] - bbox_in_rect[0]).dot(
            bbox_in_rect[1] - bbox_in_rect[0]
        ), (bbox_in_rect[2] - bbox_in_rect[1]).dot(bbox_in_rect[2] - bbox_in_rect[1])
        return bbox_width <= rect_width and bbox_height <= rect_height

    def get_fit_in_translation(self, rect):
        rect = np.array(rect)
        rect_heading = rect[1] - rect[0]
        rect_width, rect_height = (rect[1] - rect[0]).dot(rect[1] - rect[0]), (
            rect[2] - rect[1]
        ).dot(rect[2] - rect[1])
        rect_heading = rect_heading / np.linalg.norm(rect_heading)
        bbox_in_rect = ConvexHullProcessor_2d.get_headed_bbox(
            self.vertices, rect_heading
        )
        bbox_width, bbox_height = (bbox_in_rect[1] - bbox_in_rect[0]).dot(
            bbox_in_rect[1] - bbox_in_rect[0]
        ), (bbox_in_rect[2] - bbox_in_rect[1]).dot(bbox_in_rect[2] - bbox_in_rect[1])

        if bbox_width > rect_width or bbox_height > rect_height:
            return None

        bbox_center = (bbox_in_rect[0] + bbox_in_rect[2]) / 2
        rect_center = (rect[0] + rect[2]) / 2
        if np.isnan(bbox_center).any() or np.isnan(rect_center).any():
            return None
        return rect_center - bbox_center

    def get_intersection_of_convex_hull(self, other: "ConvexHullProcessor_2d"):

        polygon1 = Polygon(self.vertices[self.convex_hull.vertices])
        polygon2 = Polygon(other.vertices[other.convex_hull.vertices])
        intersection = polygon1.intersection(polygon2)
        intersection_vertices = []
        if isinstance(intersection, Polygon):
            intersection_vertices = list(intersection.exterior.coords)
        elif (
            isinstance(intersection, LineString)
            or isinstance(intersection, Point)
            or isinstance(intersection, LinearRing)
        ):
            intersection_vertices = list(intersection.coords)
        else:
            return None
        if len(intersection_vertices) == 0:
            return None
        intersection = ConvexHullProcessor_2d(vertices=intersection_vertices)
        return intersection

    @staticmethod
    def intersection_of_2_convex_hulls(
        convex_hull1: "ConvexHullProcessor_2d", convex_hull2: "ConvexHullProcessor_2d"
    ):
        polygon1 = Polygon(convex_hull1.vertices[convex_hull1.convex_hull.vertices])
        polygon2 = Polygon(convex_hull2.vertices[convex_hull2.convex_hull.vertices])
        intersection = polygon1.intersection(polygon2)
        intersection_vertices = []
        if isinstance(intersection, Polygon):
            intersection_vertices = list(intersection.exterior.coords)
        elif (
            isinstance(intersection, LineString)
            or isinstance(intersection, Point)
            or isinstance(intersection, LinearRing)
        ):
            intersection_vertices = list(intersection.coords)
        else:
            return None
        if len(intersection_vertices) == 0:
            return None
        intersection = ConvexHullProcessor_2d(vertices=intersection_vertices)
        return intersection

    def get_bbox_instance(self):
        return ConvexHullProcessor_2d.get_bbox(self.vertices)

    def get_headed_bbox_instance(self):
        return ConvexHullProcessor_2d.get_headed_bbox(
            vertices=self.vertices, heading=self.heading
        )

    def get_headed_bbox_instance_with_heading(self, heading):
        return ConvexHullProcessor_2d.get_headed_bbox(
            vertices=self.vertices, heading=heading
        )

    def get_closest_point_to_line(self, segment, default_point=None):
        convex_hull_vertices = self.get_vertices_on_convex_hull()
        closest_point = default_point
        min_distance = (
            Basic2DGeometry.point_to_line_distance(closest_point, segment)
            if default_point is not None
            else 1e9
        )
        for vertex in convex_hull_vertices:
            distance = Basic2DGeometry.point_to_line_distance(vertex, segment)
            if distance < min_distance:
                min_distance = distance
                closest_point = vertex

        return closest_point

    def get_farthest_point_to_line(self, segment, default_point=None):
        convex_hull_vertices = self.get_vertices_on_convex_hull()
        farest_point = default_point
        max_distance = (
            Basic2DGeometry.point_to_line_distance(farest_point, segment)
            if default_point is not None
            else -1
        )
        for vertex in convex_hull_vertices:
            distance = Basic2DGeometry.point_to_line_distance(vertex, segment)
            if distance > max_distance:
                max_distance = distance
                farest_point = vertex

        return farest_point

    def cut_free_space_with_point_cloud(
        self, near_side, far_side, pivot_point, force=False
    ):
        freespace_rect = ConvexHullProcessor_2d(
            vertices=[near_side[0], near_side[1], far_side[0], far_side[1]]
        )
        intersection = freespace_rect.intersect_with_another_convex(self)
        if intersection is None:
            return near_side, far_side

        closest_point_between_sides, min_distance = far_side[0], (
            far_side[0] - pivot_point
        ).dot(far_side[0] - pivot_point)
        for vertex in intersection.vertices:
            if (
                not Basic2DGeometry.point_in_parallel_lines(vertex, near_side, far_side)
                and not force
            ):
                continue
            distance = (pivot_point - vertex).dot(pivot_point - vertex)
            if distance < min_distance:
                min_distance = distance
                closest_point_between_sides = vertex

        if min_distance > 1e9 or min_distance == np.inf:
            return near_side, far_side
        # print('!closest_point_between_sides!', closest_point_between_sides)
        closest_line = np.array(
            [
                closest_point_between_sides,
                near_side[1] - near_side[0] + closest_point_between_sides,
            ]
        )
        # print('!closest_line!', closest_line)
        new_segment = Basic2DGeometry.point_to_line(
            near_side[0], closest_line
        ), Basic2DGeometry.point_to_line(near_side[1], closest_line)
        # print('!new_segment!\n', new_segment,'\n\n')
        return near_side, new_segment

    def cut_free_space_with_convex(self, near_side, far_side, force=False):

        # self.heading = (near_side[1] - near_side[0]) / np.linalg.norm(near_side[1] - near_side[0])
        # headed_bbox = self.get_headed_bbox()

        freespace_rect = ConvexHullProcessor_2d(
            vertices=[near_side[0], near_side[1], far_side[0], far_side[1]]
        )
        intersection = freespace_rect.intersect_with_another_convex(self)
        if intersection is None:
            return near_side, far_side

        closest_point_near_side = intersection.get_closest_point_to_line(
            near_side, default_point=far_side[0]
        )
        farthest_point_near_side = intersection.get_farthest_point_to_line(
            near_side, default_point=far_side[0]
        )

        closest_line = far_side

        if Basic2DGeometry.point_in_parallel_lines(
            farthest_point_near_side, near_side, far_side
        ):
            closest_line = [
                farthest_point_near_side,
                near_side[1] - near_side[0] + farthest_point_near_side,
            ]

        if Basic2DGeometry.point_in_parallel_lines(
            closest_point_near_side, near_side, far_side
        ):
            closest_line = [
                closest_point_near_side,
                near_side[1] - near_side[0] + closest_point_near_side,
            ]

        # print('closest_line', closest_line)
        new_segment = Basic2DGeometry.point_to_line(
            near_side[0], closest_line
        ), Basic2DGeometry.point_to_line(near_side[1], closest_line)
        # if max(abs(new_segment)) > 100:
        #     print('new_segment', new_segment)

        return near_side, new_segment
        pass

    def intersect_with_another_convex(self, other):
        polygon1 = Polygon(self.vertices[self.convex_hull.vertices])
        polygon2 = Polygon(other.vertices[other.convex_hull.vertices])
        if polygon2.within(polygon1):
            return other
        if polygon1.within(polygon2):
            return self
        intersection = polygon1.intersection(polygon2)
        intersection_vertices = []
        if isinstance(intersection, Polygon):
            intersection_vertices = list(intersection.exterior.coords)
        elif (
            isinstance(intersection, LineString)
            or isinstance(intersection, Point)
            or isinstance(intersection, LinearRing)
        ):
            intersection_vertices = list(intersection.coords)

        if intersection is None or len(intersection_vertices) < 3:
            return None
        intersection_convex_hull = ConvexHullProcessor_2d(
            vertices=intersection_vertices
        )
        if intersection.area < 1e-4:
            return None
        return intersection_convex_hull

    def intersect_area_with_another_convex(self, other):
        polygon1 = Polygon(self.vertices[self.convex_hull.vertices])
        polygon2 = Polygon(other.vertices[other.convex_hull.vertices])
        intersection = polygon1.intersection(polygon2)
        return intersection.area

    def is_intersected_with_rectangle(self, rectangle):
        rectangle_polygon = Polygon(rectangle)
        convex_polygon = Polygon(self.vertices[self.convex_hull.vertices])
        if rectangle_polygon.intersects(convex_polygon) or rectangle_polygon.within(
            convex_polygon
        ):
            return True
        if convex_polygon.intersects(rectangle_polygon) or convex_polygon.within(
            rectangle_polygon
        ):
            return True
        convex_hull_vertices = self.get_vertices_on_convex_hull()
        for i in range(len(convex_hull_vertices)):
            v1 = convex_hull_vertices[i]
            v2 = convex_hull_vertices[(i + 1) % len(convex_hull_vertices)]
            line = (v1, v2)
            for j in range(4):
                r1 = rectangle[j]
                r2 = rectangle[(j + 1) % 4]
                rect_line = (r1, r2)
                if Basic2DGeometry.intersection_of_segment(line, rect_line) is not None:
                    return True
        return False
