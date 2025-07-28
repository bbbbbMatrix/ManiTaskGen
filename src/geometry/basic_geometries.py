"""
custom_geometry_helper_new/basic_geometries.py

This module provides basic geometry functions for 2D and 3D geometries.

"""

from typing import List
import numpy as np
from scipy.spatial.transform import Rotation as R
from .polygon_processor import PolygonProcessor
from shapely import affinity
from src.utils.config_manager import get_basic_geometry_config
import glog


class Basic2DGeometry:
    def __init__(self):
        pass

    @property
    def EPS(self):
        """Get EPS value from current configuration"""
        return get_basic_geometry_config().EPS

    @classmethod
    def get_eps(cls):
        """Class method to get EPS value from current configuration"""
        return get_basic_geometry_config().EPS

    @staticmethod
    def rotate_point_counterclockwise(point, angle):
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return np.dot(point, R)

    @staticmethod
    def point_to_segment_distance(point, segment):
        start, end = segment[0], segment[1]
        if np.allclose(start, end):
            return np.linalg.norm(point - start)

        direction = end - start
        t = np.dot(point - start, direction) / np.dot(direction, direction)
        if t < 0:
            return np.linalg.norm(point - start)
        elif t > 1:
            return np.linalg.norm(point - end)
        else:
            projection = start + t * direction
            return np.linalg.norm(point - projection)

    @staticmethod
    def get_perpendicular_intersection(point, line):
        start, end = line[0], line[1]
        if np.allclose(start, end):
            return np.array([point[0], start[1]])

        direction = end - start
        direction = direction / np.linalg.norm(direction)

        point_to_line = point - start

        projection_length = np.dot(point_to_line, direction)
        projection = start + projection_length * direction

        return projection

    @staticmethod
    def is_inside_rectangle(point, rect, on_edge=False):

        signs = [
            (rect[i][0] - point[0]) * (rect[(i + 1) % 4][1] - point[1])
            - (rect[(i + 1) % 4][0] - point[0]) * (rect[i][1] - point[1])
            for i in range(4)
        ]
        eps = Basic2DGeometry.get_eps()
        return all(sign > eps if not on_edge else -eps for sign in signs) or all(
            sign < -eps if not on_edge else -eps for sign in signs
        )

    @staticmethod
    def is_on_segment(point, segment):
        start, end = segment[0], segment[1]
        if np.allclose(start, end):
            return np.allclose(point, start)

        direction = end - start
        t = np.dot(point - start, direction) / np.dot(direction, direction)
        return 0 <= t <= 1

    @staticmethod
    def intersection_of_line(line1, line2):
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        return np.array([px, py])

    @staticmethod
    def intersection_of_segment(segment1, segment2):
        intersection = Basic2DGeometry.intersection_of_line(segment1, segment2)
        if intersection is None:
            return None

        if Basic2DGeometry.is_on_segment(
            intersection, segment1
        ) and Basic2DGeometry.is_on_segment(intersection, segment2):
            return intersection
        return None

    @staticmethod
    def intersection_of_parallel_rectangle(rect1, rect2):
        """
        Calculate the intersection of two parallel rectangles

        Args:
            rect1: Four vertices of the first rectangle, numpy array shape=(4,2)
            rect2: Four vertices of the second rectangle, numpy array shape=(4,2)

        Returns:
            numpy.ndarray: Four vertices of the intersection rectangle, returns None if no intersection
        """
        if rect1 is None or rect2 is None:
            return None

        eps = Basic2DGeometry.get_eps()

        # 1. Calculate direction vectors of both rectangles
        dir1 = rect1[1] - rect1[0]
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = rect2[1] - rect2[0]
        dir2 = dir2 / np.linalg.norm(dir2)

        # 2. Check if they are parallel
        if (
            np.abs(np.atan2(dir1[1], dir1[0]) - np.atan2(dir2[1], dir2[0])) > eps
            and np.abs(np.atan2(dir1[1], dir1[0]) - np.atan2(-dir2[1], dir2[0])) > eps
        ):
            return None

        # 3. Project both rectangles onto parallel and perpendicular directions
        perp_dir = np.array([-dir1[1], dir1[0]])  # Perpendicular direction

        # Calculate projection ranges for each rectangle in both directions
        proj1_parallel = [np.dot(p, dir1) for p in rect1]
        proj1_perp = [np.dot(p, perp_dir) for p in rect1]
        proj2_parallel = [np.dot(p, dir1) for p in rect2]
        proj2_perp = [np.dot(p, perp_dir) for p in rect2]

        # 4. Calculate intersection of projections
        min_parallel = max(min(proj1_parallel), min(proj2_parallel))
        max_parallel = min(max(proj1_parallel), max(proj2_parallel))
        min_perp = max(min(proj1_perp), min(proj2_perp))
        max_perp = min(max(proj1_perp), max(proj2_perp))

        # 5. Check if there is an intersection
        if min_parallel >= max_parallel or min_perp >= max_perp:
            return None

        # 6. Build intersection rectangle
        corner = min_parallel * dir1 + min_perp * perp_dir
        width_vec = (max_parallel - min_parallel) * dir1
        height_vec = (max_perp - min_perp) * perp_dir

        intersection_rect = np.array(
            [
                corner,
                corner + width_vec,
                corner + width_vec + height_vec,
                corner + height_vec,
            ]
        )

        # 7. Check if intersection area is too small
        width = max_parallel - min_parallel
        height = max_perp - min_perp
        eps = Basic2DGeometry.get_eps()
        if (
            width * height < eps
        ):  # Area threshold, can be adjusted based on actual needs
            return None

        return intersection_rect

    @staticmethod
    def intersection_of_parallel_rectangle_old(rect1, rect2):
        if rect1 is None or rect2 is None:
            return None
        intersection_points = []
        for point in rect1:
            if Basic2DGeometry.is_inside_rectangle(point, rect2, on_edge=False):
                intersection_points.append(point)
        for point in rect2:
            if Basic2DGeometry.is_inside_rectangle(point, rect1, on_edge=False):
                intersection_points.append(point)

        for i in range(4):
            for j in range(4):
                intersection = Basic2DGeometry.intersection_of_segment(
                    [rect1[i], rect1[(i + 1) % 4]], [rect2[j], rect2[(j + 1) % 4]]
                )
                if intersection is not None:
                    intersection_points.append(intersection)

        if len(intersection_points) < 4:
            return None
        from .convex_hull_processor import ConvexHullProcessor_2d

        intersection_rect = ConvexHullProcessor_2d.get_headed_bbox(
            vertices=intersection_points,
            heading=(rect1[1] - rect1[0]) / np.linalg.norm(rect1[1] - rect1[0]),
        )

        if intersection_rect is None:
            return None

        return intersection_rect

    @staticmethod
    def intersection_area_of_parallel_rectangle(rect1, rect2):
        intersection = Basic2DGeometry.intersection_of_parallel_rectangle(rect1, rect2)
        if intersection is None:
            return 0
        return intersection.get_area()

    @staticmethod
    def intersection_area_of_rectangle(rect1, rect2):
        return PolygonProcessor.intersection_area(rect1, rect2)

    @staticmethod
    def find_random_placement_inside(outer_rect, inner_rect):
        """
        Find a random valid placement position for the inner rectangle within the outer rectangle

        Args:
            outer_rect: Outer rectangle, numpy array shape=(4,2)
            inner_rect: Inner rectangle, numpy array shape=(4,2)

        Returns:
            translation: Translation vector, returns None if placement is not possible
        """
        # 1. Calculate direction vectors of rectangles
        outer_dir = outer_rect[1] - outer_rect[0]
        outer_dir = outer_dir / np.linalg.norm(outer_dir)
        inner_dir = inner_rect[1] - inner_rect[0]
        inner_dir = inner_dir / np.linalg.norm(inner_dir)

        # 2. Check if they are parallel
        if not np.allclose(outer_dir, inner_dir) and not np.allclose(
            outer_dir, -inner_dir
        ):
            glog.warning("rectangle not parallel")
            return None

        # 3. Calculate dimensions of inner and outer rectangles
        outer_perp = np.array([-outer_dir[1], outer_dir[0]])
        inner_center = np.mean(inner_rect, axis=0)

        # Calculate projection ranges
        outer_parallel = [np.dot(p - outer_rect[0], outer_dir) for p in outer_rect]
        outer_perp_proj = [np.dot(p - outer_rect[0], outer_perp) for p in outer_rect]
        inner_parallel = [np.dot(p - inner_center, outer_dir) for p in inner_rect]
        inner_perp_proj = [np.dot(p - inner_center, outer_perp) for p in inner_rect]

        # Calculate placeable ranges
        outer_pmin, outer_pmax = min(outer_parallel), max(outer_parallel)
        outer_hmin, outer_hmax = min(outer_perp_proj), max(outer_perp_proj)
        inner_pmin, inner_pmax = min(inner_parallel), max(inner_parallel)
        inner_hmin, inner_hmax = min(inner_perp_proj), max(inner_perp_proj)

        # Calculate valid placement ranges
        valid_pmin = outer_pmin - inner_pmin
        valid_pmax = outer_pmax - inner_pmax
        valid_hmin = outer_hmin - inner_hmin
        valid_hmax = outer_hmax - inner_hmax

        if valid_pmax <= valid_pmin or valid_hmax <= valid_hmin:
            glog.warning("rectangle not fit")
            return None

        # 4. Randomly select a valid position
        random_p = np.random.uniform(valid_pmin, valid_pmax)
        random_h = np.random.uniform(valid_hmin, valid_hmax)

        # 5. Calculate translation vector
        translation = (
            outer_rect[0] + random_p * outer_dir + random_h * outer_perp - inner_center
        )

        return translation

    @staticmethod
    def point_side_of_line(point, line):
        return np.sign(np.cross(line[1] - line[0], point - line[0]))

    @staticmethod
    def closest_point_on_line(point, a, b, c):
        eps = Basic2DGeometry.get_eps()
        x = (b * (b * point[0] - a * point[1]) - a * c) / (a**2 + b**2 + eps)
        y = (a * (-b * point[0] + a * point[1]) - b * c) / (a**2 + b**2 + eps)
        return np.array([x, y])

    @staticmethod
    def point_to_line_distance(point, line):
        return np.linalg.norm(
            np.cross(line[1] - line[0], line[0] - point)
        ) / np.linalg.norm(line[1] - line[0] + 1e-8)

    @staticmethod
    def line2abc(line):
        start, end = line[0], line[1]
        a = end[1] - start[1]
        b = start[0] - end[0]
        c = -a * start[0] - b * start[1]
        return a, b, c

    @staticmethod
    def point_to_line(point, line):
        a, b, c = Basic2DGeometry.line2abc(line)
        return Basic2DGeometry.closest_point_on_line(point, a, b, c)

    @staticmethod
    def point_in_polygon(point, polygon):
        n = len(polygon)
        polygon.append(polygon[0])
        signs = [
            (polygon[i][0] - point[0]) * (polygon[i + 1][1] - point[1])
            - (polygon[i + 1][0] - point[0]) * (polygon[i][1] - point[1])
            for i in range(n)
        ]
        return all(sign > 0 for sign in signs) or all(sign < 0 for sign in signs)

    @staticmethod
    def point_in_parallel_lines(point, line1, line2):
        eps = Basic2DGeometry.get_eps()
        direction1 = (line1[1] - line1[0]) / np.linalg.norm(line1[1] - line1[0] + eps)
        direction2 = (line2[1] - line2[0]) / np.linalg.norm(line2[1] - line2[0] + eps)

        start1, start2 = line1[0], line2[0]

        perpendicular_direction1 = np.array([-direction1[1], direction1[0]])
        perpendicular_direction2 = np.array([-direction2[1], direction2[0]])

        distance_to_line1 = np.dot(point - start1, perpendicular_direction1)
        distance_to_line2 = np.dot(point - start2, perpendicular_direction2)

        return (distance_to_line1 >= -eps and distance_to_line2 <= eps) or (
            distance_to_line1 <= eps and distance_to_line2 >= -eps
        )

    @staticmethod
    def cal_vector_cos(vector1, vector2):
        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )

    @staticmethod
    def is_fitable_in_rectangle_union(target_rectangle, rectangle_list):
        return PolygonProcessor.is_rectangle_fitable_in_rectangle_union(
            target_rectangle, rectangle_list
        )

    @staticmethod
    def cal_rectangle_with_diagonal_line_n_heading(diagonal_line, heading=(1, 0)):

        heading = np.array(heading) / np.linalg.norm(heading)
        perpendicular_heading = np.array([-heading[1], heading[0]])
        start, end = diagonal_line[0], diagonal_line[1]
        if np.dot(end - start, heading) < 0:
            heading = -heading
        if np.dot(end - start, perpendicular_heading) < 0:
            perpendicular_heading = -perpendicular_heading

        width, height = np.dot(end - start, heading), np.dot(
            end - start, perpendicular_heading
        )

        return [
            start,
            start + heading * width,
            start + heading * width + perpendicular_heading * height,
            start + perpendicular_heading * height,
        ]

    @staticmethod
    def rectangle_affordable(rectangle_a, rectangle_b):
        # check if all points of rectangle_b are inside rectangle_a

        center_a = np.mean(rectangle_a, axis=0)
        center_b = np.mean(rectangle_b, axis=0)

        translated_rectangle_b = rectangle_b - center_b + center_a

        for point in translated_rectangle_b:
            if not Basic2DGeometry.is_inside_rectangle(
                point, rectangle_a, on_edge=True
            ):
                return False
        return True

    @staticmethod
    def align_rectangle_vertice(rectangle):
        # Calculate the center of the rectangle
        center = np.mean(rectangle, axis=0)

        # Define a function to determine the quadrant of a point relative to the center
        def quadrant(point):
            if point[0] <= center[0] and point[1] <= center[1]:
                return 0  # Rear-left
            elif point[0] <= center[0] and point[1] > center[1]:
                return 1  # Front-left
            elif point[0] > center[0] and point[1] > center[1]:
                return 2  # Front-right
            else:
                return 3  # Rear-right

        # Sort the vertices based on their quadrant
        sorted_rectangle = sorted(rectangle, key=quadrant)

        return np.array(sorted_rectangle)

    @staticmethod
    def normalize_rectangle(rectangle):
        edge1 = rectangle[3] - rectangle[0]
        angle = np.arctan2(edge1[1], edge1[0])
        return Basic2DGeometry.rotate_point_counterclockwise(rectangle, angle), angle

    @staticmethod
    def unnormalize_rectangle(rectangle, angle):
        # import ipdb
        # ipdb.set_trace()
        return Basic2DGeometry.rotate_point_counterclockwise(rectangle, -angle)


class Basic3DGeometry:

    def __init__(self):
        pass

    @staticmethod
    def point_to_line_distance(point, line):
        return np.linalg.norm(
            np.cross(line[1] - line[0], line[0] - point)
        ) / np.linalg.norm(line[1] - line[0])

    @staticmethod
    def point_to_line_segment_distance(point, segment):
        start, end = segment[0], segment[1]
        if np.allclose(start, end):
            return np.linalg.norm(point - start)

        direction = end - start
        t = np.dot(point - start, direction) / np.dot(direction, direction)
        if t < 0:
            return np.linalg.norm(point - start)
        elif t > 1:
            return np.linalg.norm(point - end)
        else:
            projection = start + t * direction
            return np.linalg.norm(point - projection)

    @staticmethod
    def point_to_plane_distance(point, plane):
        """
        Calculate the distance from a point to a plane defined by a point and a normal vector.

        Parameters:
        - point: The point [x, y, z]
        - plane: A tuple (plane_point, normal_vector) where:
            - plane_point: A point on the plane [x0, y0, z0]
            - normal_vector: The normal vector of the plane [nx, ny, nz]
        Returns:
        - The distance from the point to the plane
        """
        plane_point, normal_vector = plane
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        # Calculate the vector from the plane point to the point
        point_vector = point - plane_point

        # Calculate the distance using the dot product
        distance = np.abs(np.dot(point_vector, normal_vector))

        return distance

    @staticmethod
    def point_to_triangle_distance(point, triangle):
        """
        Calculate the distance from a point to a triangle defined by its vertices.

        Parameters:
        - point: The point [x, y, z]
        - triangle: A list of three vertices of the triangle [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
        Returns:
        - The distance from the point to the triangle
        """

        triangle = np.array(triangle)
        v0 = triangle[1] - triangle[0]
        v1 = triangle[2] - triangle[0]
        v2 = point - triangle[0]

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return np.linalg.norm(point - triangle[0])
        # Barycentric coordinates
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w
        # Check if the point is inside the triangle
        if u >= 0 and v >= 0 and w >= 0:
            # The point is inside the triangle, return the distance to the plane of the triangle
            plane_point = triangle[0]
            normal_vector = np.cross(v0, v1)
            return Basic3DGeometry.point_to_plane_distance(
                point, (plane_point, normal_vector)
            )
        else:
            # The point is outside the triangle, calculate the distance to the closest edge or vertex
            distances = []
            for i in range(3):
                edge_start = triangle[i]
                edge_end = triangle[(i + 1) % 3]
                distances.append(
                    Basic3DGeometry.point_to_line_segment_distance(
                        point, [edge_start, edge_end]
                    )
                )
            # Also consider the distance to the vertices
            distances.extend([np.linalg.norm(point - vertex) for vertex in triangle])
            return min(distances)

    @staticmethod
    def quaternion_between_vectors(v1, v2):
        """
        calculate the rotation quaternion from vector v1 to vector v2.

        Parameters:
        - v1:  [x1, y1, z1]
        - v2:  [x2, y2, z2]

        Returns:
        - quaternion [x, y, z, w]
        """
        # normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # calculate rotation axis
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)

        # special handling if vectors are parallel or anti-parallel
        if axis_norm == 0:
            if np.dot(v1, v2) > 0:
                return np.array([0, 0, 0, 1])  # parallel vectors, no rotation needed
            else:
                # anti-parallel vectors, choose any perpendicular vector as rotation axis
                orthogonal_vector = (
                    np.array([1, 0, 0])
                    if abs(v1[0]) < abs(v1[1])
                    else np.array([0, 1, 0])
                )
                axis = np.cross(v1, orthogonal_vector)
                axis = axis / np.linalg.norm(axis)
                return np.concatenate((axis, [0]))

        axis = axis / axis_norm

        # calculate rotation angle
        angle = np.arccos(np.dot(v1, v2))

        # calculate rotation quaternion
        quaternion = R.from_rotvec(angle * axis).as_quat()

        return quaternion

    @staticmethod
    def quaternion_from_vector(vector, angle):
        """
        Calculate the rotation quaternion from the rotation axis vector and angle.

        Parameters:
        - vector: rotation axis vector[x, y, z]
        - angle: rotation angle in radians

        Returns:
        -  [x, y, z, w]
        """
        # normalize vector
        vector = vector / np.linalg.norm(vector)

        # calculate rotation quaternion
        quaternion = R.from_rotvec(angle * vector).as_quat()

        return quaternion

    @staticmethod
    def vector_to_rpy(vector):

        # normalize vector
        vector = vector / np.linalg.norm(vector)

        # calculate rotation matrix
        z_axis = vector
        x_axis = np.array([1, 0, 0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

        # calculate rpy
        r = R.from_matrix(rotation_matrix)
        rpy = r.as_euler("xyz", degrees=True)

        return rpy

    @staticmethod
    def quaternion_between_vectors(v1, v2):
        """
        Calculate the rotation quaternion from vector v1 to vector v2.

        Parameters:
        - v1: The first vector [x1, y1, z1]
        - v2: The second vector [x2, y2, z2]

        Returns:
        - Rotation quaternion [x, y, z, w]
        """
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # Calculate rotation axis
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)

        # Special handling if vectors are parallel or anti-parallel
        if axis_norm == 0:
            if np.dot(v1, v2) > 0:
                # Parallel vectors, no rotation needed
                return np.array([0, 0, 0, 1])
            else:
                # Anti-parallel vectors, choose any perpendicular vector as rotation axis
                orthogonal_vector = (
                    np.array([1, 0, 0])
                    if abs(v1[0]) < abs(v1[1])
                    else np.array([0, 1, 0])
                )
                axis = np.cross(v1, orthogonal_vector)
                axis = axis / np.linalg.norm(axis)
                return np.concatenate((axis, [0]))

        axis = axis / axis_norm

        # Calculate rotation angle
        angle = np.arccos(np.dot(v1, v2))

        # Calculate rotation quaternion
        quaternion = R.from_rotvec(angle * axis).as_quat()

        return quaternion

    @staticmethod
    def ray_triangle_intersection(ray_origin, ray_direction, triangle, eps=1e-6):
        """

        Calculate the intersection point of a ray and a triangle.
        Parameters:
        - ray_origin: The origin of the ray [x, y, z]
        - ray_direction: The direction of the ray [dx, dy, dz]
        - triangle: The vertices of the triangle [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]

        Returns:
        - intersection_point: The intersection point [x, y, z] or None if no intersection

        """

        edge1, edge2 = triangle[1] - triangle[0], triangle[2] - triangle[0]
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)

        if -eps < a < eps:
            # This ray is parallel to this triangle.
            return None

        f = 1.0 / a
        s = ray_origin - triangle[0]
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * np.dot(edge2, q)

        if t > eps:  # ray intersection
            intersection_point = ray_origin + ray_direction * t
            return intersection_point
        else:  # This means that there is a line intersection but not a ray intersection.
            return None
