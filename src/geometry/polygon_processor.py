"""
custom_geometry_helper_new/polygon_processor.py
This module provides functionality to process polygons, including creating polygons from rectangles.

It includes methods for plotting polygons, converting meshes to polygons, and performing geometric operations like intersection and union.

"""

from shapely.geometry import Polygon, LinearRing, LineString, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from typing import List
import glog


class PolygonProcessor:
    def __init__(self, polygons: List[Polygon]):
        self.polygons = polygons

    def from_2d_rectangle(self, rectangle):
        self.polygons = [Polygon(rectangle)]

    @staticmethod
    def from_2d_rectangle_list(rectangle_list):
        polygons = []
        for rectangle in rectangle_list:
            polygons.append(Polygon(rectangle))
        return polygons

    @staticmethod
    def from_2d_polygon(polygon):
        polygons = [Polygon(polygon)]
        return PolygonProcessor(polygons)

    @staticmethod
    def from_2d_polygon_list(polygon_list):
        polygons = []
        for polygon in polygon_list:
            polygons.append(Polygon(polygon))
        return PolygonProcessor(polygons)

    @staticmethod
    def from_vertices_and_faces(vertices, faces):
        polygons = PolygonProcessor.vertices_and_face_to_polygon(vertices, faces)
        return PolygonProcessor(polygons)

    def from_trimesh_geometry(geometry):
        pass

    def plot_polygons(self, title="Polygons"):
        polygons = self.polygons
        fig, ax = plt.subplots()
        for geom in polygons:
            if geom.is_empty:
                continue
            if isinstance(geom, Polygon):
                exterior = LinearRing(geom.exterior.coords)
                x, y = exterior.xy
                ax.plot(
                    x,
                    y,
                    color="blue",
                    alpha=0.7,
                    linewidth=2,
                    solid_capstyle="round",
                    zorder=2,
                )
                ax.fill(x, y, color="skyblue", alpha=0.4)
                # inner cycle
                for interior in geom.interiors:
                    ring = LinearRing(interior.coords)
                    x, y = ring.xy
                    ax.plot(
                        x,
                        y,
                        color="red",
                        alpha=0.7,
                        linewidth=2,
                        solid_capstyle="round",
                        zorder=2,
                    )
                    ax.fill(x, y, color="lightcoral", alpha=0.4)
            elif isinstance(geom, LineString):
                x, y = geom.xy
                ax.plot(
                    x,
                    y,
                    color="green",
                    alpha=0.7,
                    linewidth=2,
                    solid_capstyle="round",
                    zorder=2,
                )
            elif isinstance(geom, Point):
                x, y = geom.xy
                ax.plot(x, y, "o", color="purple", alpha=0.7, markersize=5, zorder=2)
            else:
                print(f"Unsupported geometry type: {type(geom)}")
        ax.set_title(title)
        plt.show()

    @staticmethod
    def mesh_to_polygon(mesh):
        vertices = mesh.vertices[:, :2]
        faces = mesh.faces
        polygons = []
        for face in faces:
            polygon = Polygon(vertices[face])
            polygons.append(polygon)
        return polygons

    @staticmethod
    def get_convex_hull(vertices):
        return Polygon(vertices).convex_hull

    @staticmethod
    def vertices_and_face_to_polygon(vertices, faces):
        vertices = vertices[:, :2]
        polygons = []
        for face in faces:
            polygon = Polygon(vertices[face])
            polygons.append(polygon)
        return polygons

    @staticmethod
    def intersect_two_polygons(polygon1, polygon2):
        intersection = []
        for poly1 in polygon1:
            for poly2 in polygon2:
                inter = poly1.intersection(poly2)
                if not inter.is_empty:
                    intersection.append(inter)
        return intersection

    @staticmethod
    def intersect_two_polygon_instances(polygon1, polygon2):
        return PolygonProcessor.intersect_two_polygons(
            polygon1.polygons, polygon2.polygons
        )

    @staticmethod
    def intersect_polygon_list(polygon_list):
        polygons = polygon_list[0]
        for i in range(1, len(polygon_list)):
            polygons = PolygonProcessor.intersect_two_polygon_instances(
                polygons, polygon_list[i]
            )
        if isinstance(polygons, list):
            return PolygonProcessor(polygons)
        return polygons

    @staticmethod
    def union_two_polygons(polygon1, polygon2):
        union = polygon1.union(polygon2)
        #  glog.info(f"Union of {polygon1} and {polygon2} is {union}")
        return union

    @staticmethod
    def union_polygon_list(polygon_list):
        if len(polygon_list) == 0:
            return Polygon()
        polygons = polygon_list[0]
        for i in range(1, len(polygon_list)):
            polygons = PolygonProcessor.union_two_polygons(polygons, polygon_list[i])
        return polygons

    @staticmethod
    def union_numpy_polygon_list(polygon_list):
        polygons = []
        for i in range(len(polygon_list)):
            polygons.append(Polygon(polygon_list[i]))
        return PolygonProcessor.union_polygon_list(polygons)

    @staticmethod
    def union_numpy_polygon_list_area(polygon_list):
        polygons = []
        for i in range(len(polygon_list)):
            polygons.append(Polygon(polygon_list[i]))
        return PolygonProcessor.union_polygon_list(polygons).area

    def intersect_polygon_instance(self, polygons):
        polygons1 = self.polygons
        polygons2 = polygons
        return PolygonProcessor.intersect_two_polygons(polygons1, polygons2)

    @staticmethod
    def are_polygons_connected(polygon_list):
        if len(polygon_list) <= 1:
            return True
        try:
            unary = unary_union(polygon_list)
            return unary.geom_type == "Polygon"
        except Exception as e:
            import glog

            glog.error(f"Error in are_polygons_connected: {e}")
            return False

    @staticmethod
    def are_rectangles_connected(rectangle_list):
        #   glog.info(f"Checking if rectangles are connected: {rectangle_list}")
        polygon_list = [Polygon(rectangle) for rectangle in rectangle_list]
        return PolygonProcessor.are_polygons_connected(polygon_list)

    @staticmethod
    def is_fitable_in_polygon_union(polygon_list, polygon):
        union = PolygonProcessor.union_polygon_list(polygon_list)
        return union.contains(polygon)

    @staticmethod
    def is_rectangle_fitable_in_rectangle_union(target_rectangle, rectangle_list):
        polygon = Polygon(target_rectangle)
        polygon_list = [Polygon(rectangle) for rectangle in rectangle_list]
        return PolygonProcessor.is_fitable_in_polygon_union(polygon_list, polygon)

    @staticmethod
    def intersection_area(polygon1_vertices, polygon2_vertices):
        polygon1 = Polygon(polygon1_vertices)
        polygon2 = Polygon(polygon2_vertices)
        return polygon1.intersection(polygon2).area

    def is_inside(self, other):
        return all([poly.within(other) for poly in self.polygons])

    def get_len(self):
        return len(self.polygons)

    def get_area(self):
        return sum([poly.area for poly in self.polygons])
