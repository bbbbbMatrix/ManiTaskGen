"""
custom_geometry_helper_new/object_mesh_processor.py

This module provides functionality to process 3D meshes, particularly for extracting affordable platforms from the mesh.
Specifically, 3D meshes are processed to extract affordable platforms with the following peculiarities:
- 2D convex hulls. If a platform is far from convex, custom_geometry_helper_new/concave_processor.py is used to decompose the platform into convex polygons.
- Height information. The height of the platform is defined as the difference.
- Whether visible from 4 directions. This is heuristic, and may not be accurate.


"""

import trimesh
import numpy as np
from scipy.spatial import cKDTree
from typing import List
from shapely.geometry import Polygon, LinearRing, LineString, Point, box
from .polygon_processor import PolygonProcessor
from . import convex_hull_processor, basic_geometries, concave_processor
from .concave_processor import ConcaveProcessor
import glog
from src.utils.config_manager import get_mesh_processor_config


class AffordablePlatform:
    def __init__(self, vertices=[], faces=[], available_height=1e9, name=""):
        # vertices and faces
        self.vertices = vertices
        self.faces = faces
        self.bbox = np.min(vertices, axis=0)[:2], np.max(vertices, axis=0)[:2]
        # available height of the platform's free space
        self.available_height = 1e9
        # derivative of the platform (not sure if it is useful)
        self.derivative = []
        # name of the platform. object_name + platform_index
        self.name = name
        # is top platform or not
        self.is_top_platform = False
        self.visible_directions = {}
        pass

    # get the top and bottom height of the platform
    def get_height(self):
        return self.vertices[:, 2].max(), self.vertices[:, 2].min()

    def get_convex_hull_2d(self):
        return convex_hull_processor.ConvexHullProcessor_2d(self.vertices, (1, 0))

    # get the platform as shapely.geometry.Polygon
    def get_polygon(self):
        return PolygonProcessor.vertices_and_face_to_polygon(self.vertices, self.faces)

    def get_area(self):
        polygons = self.get_polygon()
        return sum([polygon.area for polygon in polygons])

    def block_by_other_platform(self, other_platform, threshold=0.5):

        # convex = self.get_convex_hull_2d()
        # other_convex = other_platform.get_convex_hull_2d()
        polygon = PolygonProcessor.from_vertices_and_faces(self.vertices, self.faces)
        other_polygon = PolygonProcessor.from_vertices_and_faces(
            other_platform.vertices, other_platform.faces
        )
        intersection_polygon = PolygonProcessor.intersect_polygon_list(
            [polygon, other_polygon]
        )

        area_bottom = polygon.get_area()
        if (
            intersection_polygon is None
            or intersection_polygon.get_area() < MeshProcessor.get_config().EPS
        ):
            return False
        if area_bottom * threshold < intersection_polygon.get_area():
            return True

        return False


class MeshProcessor:
    # minimal size of the platform

    # mainly used to process the mesh
    def __init__(self, mesh=None, name=""):
        # mesh is a trimesh object
        self.mesh = mesh
        # name of the mesh
        self.name = name
        # list of platforms that are affordable. Each platform consists of a list of faces.
        self.affordable_platforms = []
        self.inverse_affordable_platforms = []

    @classmethod
    def get_config(cls):
        """Get the configuration for MeshProcessor"""

        return get_mesh_processor_config()

    def __repr__(self):
        return f"MeshProcessor(name={self.name})"

    def from_faces_and_vertices(faces, vertices):
        return MeshProcessor(trimesh.Trimesh(vertices=vertices, faces=faces))

    def get_bounding_box(self):
        return self.mesh.bounds

    def apply_offset(self, offset):
        """
        Apply an offset to the mesh vertices.
        Args:
            offset: a 3D vector in the format of [x, y, z]
        """
        if not isinstance(offset, np.ndarray):
            offset = np.array(offset)
        self.mesh.vertices += offset
        return self

    def calculate_normal_vector(self):

        vertices = self.mesh.vertices
        faces = self.mesh.faces
        res = []
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            face_normal /= np.linalg.norm(face_normal)
            res.append(face_normal)
        return res

    @staticmethod
    def repair_mesh(mesh):
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        return mesh

    def repair_mesh_instance(self):

        trimesh.repair.fix_inversion(self.mesh)
        trimesh.repair.fix_winding(self.mesh)
        trimesh.repair.fix_normals(self.mesh)
        trimesh.repair.fill_holes(self.mesh)
        return self

    def cal_orientation(self):

        vertices_2d = self.mesh.vertices[:, :2]
        vertices_2d += (
            np.random.rand(*vertices_2d.shape) * 1e-6
        )  # to avoid numerical issues(e.g. all vertices are inline)

        bounds_2d = trimesh.bounds.oriented_bounds_2D(vertices_2d)
        x_min = bounds_2d[1][0] * (-0.5)
        x_max = bounds_2d[1][0] * 0.5
        y_min = bounds_2d[1][1] * (-0.5)
        y_max = bounds_2d[1][1] * 0.5

        bounding_points = np.array(
            [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
        )
        bounding_points -= bounds_2d[0][2, :2]

        rotation_matrix_inv = np.linalg.inv(bounds_2d[0][:2, :2])
        bounding_points = bounding_points @ rotation_matrix_inv.T

        return bounding_points, (rotation_matrix_inv[0, 0], rotation_matrix_inv[1, 0])

    def cal_convex_hull_2d(self):
        vertices_2d = self.mesh.vertices[:, :2]
        convex_hull = convex_hull_processor.ConvexHullProcessor_2d(vertices_2d)
        return convex_hull

    def mesh_after_merge_close_vertices(self, tol=1e-6):
        # use CKDTree to deal with mesh with coincident vertices
        tree = cKDTree(self.mesh.vertices)
        unique_indices = tree.query_ball_tree(tree, tol)

        # create a map to merge vertices
        merge_map = {}
        for group in unique_indices:
            representative = group[0]
            for idx in group:
                merge_map[idx] = representative

        # create new vertices and faces
        new_vertices = []
        new_faces = []
        vertex_map = {}
        for _, new_idx in merge_map.items():
            if new_idx not in vertex_map:
                vertex_map[new_idx] = len(new_vertices)
                new_vertices.append(self.mesh.vertices[new_idx])

        for face in self.mesh.faces:
            new_face = [vertex_map[merge_map[idx]] for idx in face]
            new_faces.append(new_face)

        new_vertices = np.array(new_vertices)
        new_faces = np.array(new_faces)

        self.mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        return self

    def find_vertex_and_faces(mesh, target_vertex):
        matching_indices = np.where(
            np.all(np.isclose(mesh.vertices, target_vertex, atol=1e-2), axis=1)
        )[0]

        if len(matching_indices) == 0:
            print(f"No vertex found at position {target_vertex}")
            return

        for idx in matching_indices:
            print(f"Vertex found at index {idx}: {mesh.vertices[idx]}")
            faces_with_vertex = np.where(np.any(mesh.faces == idx, axis=1))[0]
            for face_idx in faces_with_vertex:
                print(f"Face {face_idx}: {mesh.faces[face_idx]}")

    def point_inside_mesh(self, point):
        """
        Check if a point is inside the mesh.
        Args:
            point: a 3D point in the format of [x, y, z]
        Returns:
            True if the point is inside the mesh, False otherwise.
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        return self.mesh.contains([point])[0]

    def point_2d_inside_mesh(self, point):
        """
        Check if a 2D point is inside the mesh's 2D projection.
        Args:
            point: a 2D point in the format of [x, y]
        Returns:
            True if the point is inside the mesh's 2D projection, False otherwise.
        """
        for face in self.mesh.faces:
            triangle = self.mesh.vertices[face]
            polygon = Polygon(triangle[:, :2])
            if polygon.contains(Point(point)):
                return True
        return False

    @staticmethod
    def create_cuboid_from_vertices(vertices):
        # ensure the order of vertices follows the right-hand rule
        vertices = np.array(vertices)

        # calculate the normal vectors of the bottom and top faces
        bottom_normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[1])
        top_normal = np.cross(vertices[5] - vertices[4], vertices[6] - vertices[5])

        # If the normal vectors of the bottom and top faces are opposite, swap the order of the top face vertices
        if np.dot(bottom_normal, top_normal) < 0:
            vertices[4], vertices[5], vertices[6], vertices[7] = (
                vertices[4],
                vertices[7],
                vertices[6],
                vertices[5],
            )

        # Define the 12 faces of the cuboid
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Bottom face
                [4, 6, 5],
                [4, 7, 6],  # Top face
                [0, 5, 1],
                [0, 4, 5],  # Front face
                [1, 6, 2],
                [1, 5, 6],  # Right face
                [2, 7, 3],
                [2, 6, 7],  # Back face
                [3, 4, 0],
                [3, 7, 4],  # Left face
            ]
        )

        # Create the cuboid mesh
        cuboid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if cuboid_mesh.volume < 0:
            faces = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],  # Bottom face
                    [4, 5, 6],
                    [4, 6, 7],  # Top face
                    [0, 1, 5],
                    [0, 5, 4],  # Front face
                    [1, 2, 6],
                    [1, 6, 5],  # Right face
                    [2, 3, 7],
                    [2, 7, 6],  # Back face
                    [3, 0, 4],
                    [3, 4, 7],  # Left face
                ]
            )
            cuboid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return MeshProcessor(cuboid_mesh)

    @staticmethod
    def create_cuboid_from_bbox(bbox_min, bbox_max):
        # Define the 8 vertices of the cuboid
        vertices = np.array(
            [
                [bbox_min[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_min[1], bbox_max[2]],
                [bbox_max[0], bbox_min[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_max[2]],
                [bbox_min[0], bbox_max[1], bbox_max[2]],
            ]
        )

        # Define the 12 faces of the cuboid
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Bottom face
                [4, 5, 6],
                [4, 6, 7],  # Top face
                [0, 1, 5],
                [0, 5, 4],  # Front face
                [1, 2, 6],
                [1, 6, 5],  # Right face
                [2, 3, 7],
                [2, 7, 6],  # Back face
                [3, 0, 4],
                [3, 4, 7],  # Left face
            ]
        )

        # Create the cuboid mesh
        cuboid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return MeshProcessor(cuboid_mesh)

    def intersection_with_cuboid(self, cuboid_mesh):

        cuboid_mesh.repair_mesh_instance()
        self.repair_mesh_instance()
        if self.mesh.is_volume == False:
            return None
        if cuboid_mesh.mesh.is_volume == False:
            return None
        intersection_mesh = trimesh.boolean.boolean_manifold(
            [cuboid_mesh.mesh, self.mesh], operation="intersection"
        )
        if intersection_mesh.vertices.shape[0] < 4:
            if intersection_mesh.vertices.shape[0] == 0:
                pass
            else:
                print(intersection_mesh.vertices, "intersection is not a volume!")
                if intersection_mesh.vertices.shape[0] == 3:
                    intersection_mesh = (
                        MeshProcessor.create_thin_cylinder_from_three_points(
                            intersection_mesh.vertices
                        )
                    )
                else:
                    return None

            return None
        intersection_mesh = MeshProcessor.repair_mesh(intersection_mesh)
        return MeshProcessor(intersection_mesh)

    @staticmethod
    def create_thin_cylinder_from_three_points(points, thickness=1e-2):

        # calculate the normal vector of the plane defined by the three points
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        # calculate the offset of the two circles
        offset = normal * thickness / 2
        vertices = np.vstack([points + offset, points - offset])

        # define the
        faces = np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
                [0, 1, 4],
                [0, 4, 3],
                [1, 2, 5],
                [1, 5, 4],
                [2, 0, 3],
                [2, 3, 5],
            ]
        )

        # Create an object.
        thin_cylinder = trimesh.Trimesh(vertices=vertices, faces=faces)

        return thin_cylinder

    @staticmethod
    def create_cylinder_from_polygon(polygon, z_bottom, z_top):
        meshes = []
        polygons = polygon.polygons
        for polygon in polygons:
            if isinstance(polygon, Polygon):
                trimesh_polygon = trimesh.creation.extrude_polygon(
                    polygon, height=z_top - z_bottom
                )
                trimesh_polygon.apply_translation([0, 0, z_bottom])
                meshes.append(trimesh_polygon)
            elif isinstance(polygon, LineString):
                coords = np.array(polygon.coords)
                for i in range(len(coords) - 1):
                    start = coords[i]
                    end = coords[i + 1]
                    length = np.linalg.norm(end - start)
                    direction = (end - start) / length
                    width = 0.1
                    height = z_top - z_bottom
                    rect = trimesh.creation.box(extents=[length, width, height])
                    # create a transformation matrix to move the rectangle to the right position
                    transform = np.eye(4)
                    transform[:2, 3] = (start + end) / 2
                    rect.apply_transform(transform)
                    rect.apply_translation([0, 0, z_bottom])
                    meshes.append(rect)
            elif isinstance(polygon, Point):
                # Turn the point into a sphere
                sphere = trimesh.creation.icosphere(radius=0.1)
                sphere.apply_translation([polygon.x, polygon.y, (z_bottom + z_top) / 2])
                meshes.append(sphere)
            else:
                print(f"Unsupported geometry type: {type(polygon)}")
                continue

        combined_mesh = trimesh.util.concatenate(meshes)
        # Rotate the mesh to swap the z and y coordinates
        # rotation_matrix = trimesh.transformations.rotation_matrix(3*np.pi / 2, [1, 0, 0])

        # combined_mesh.apply_transform(rotation_matrix)

        combined_mesh = MeshProcessor(combined_mesh).repair_mesh_instance()

        return combined_mesh

    def get_raw_affordable_platforms(
        self,
        base_thres=np.cos(np.deg2rad(20)),
        adj_thres=np.deg2rad(20),
        abs_thres=np.cos(np.deg2rad(40)),
        inverse=False,
    ):
        def reindex(vertex_indices, old_faces):
            new_faces = []
            for face in old_faces:
                new_face = [
                    np.where(vertex_indices == face[i])[0][0]
                    for i in range(face.shape[0])
                ]
                new_faces.append(new_face)
            return new_faces

        separate_face_list = []
        geometry = self.mesh

        normal_vector = self.calculate_normal_vector()
        vertical_vector = np.array([0, 0, 1])
        if inverse:
            vertical_vector = -vertical_vector
        n_face = geometry.faces.shape[0]
        adjacency_faces = geometry.face_adjacency
        adjacency_angles = geometry.face_adjacency_angles
        adjacency = [set() for _ in range(n_face)]

        for i in range(len(adjacency_faces)):
            face1, face2 = adjacency_faces[i]
            angle = adjacency_angles[i]
            adjacency[face1].add((face2, angle))
            adjacency[face2].add((face1, angle))

        # calculate the cos_value between normal_vector and vertical_vector
        cos_values = [
            basic_geometries.Basic2DGeometry.cal_vector_cos(
                normal_vector[i], vertical_vector
            )
            for i in range(n_face)
        ]
        affordable_face_indices = {
            i for i in range(n_face) if cos_values[i] > base_thres
        }

        bel = [i for i in range(n_face)]

        def getf(x):
            return x if bel[x] == x else getf(bel[x])

        while True:
            new_faces = set()
            for face_indice in affordable_face_indices:
                for adj_face_indice, angle in adjacency[face_indice]:
                    if (
                        cos_values[adj_face_indice] > abs_thres
                        and abs(angle) < adj_thres
                    ):
                        bel[getf(adj_face_indice)] = getf(face_indice)
                        if (
                            adj_face_indice not in affordable_face_indices
                            and adj_face_indice not in new_faces
                        ):
                            new_faces.add(adj_face_indice)
            if len(new_faces) == 0:
                break
            affordable_face_indices.update(new_faces)

        separate_faces = {}
        for face in affordable_face_indices:
            if getf(face) not in separate_faces:
                separate_faces[getf(face)] = []
            separate_faces[getf(face)].append(face)

        separate_face_id = 0
        for v in separate_faces.values():
            face_indices = v
            faces = [geometry.faces[i] for i in face_indices]
            vertices = np.unique(faces)
            new_faces = reindex(vertices, faces)
            new_vertices = geometry.vertices[vertices]
            tmp_affordable_platform = AffordablePlatform(
                vertices=new_vertices,
                faces=new_faces,
                name=self.name + "_" + str(separate_face_id),
            )
            if tmp_affordable_platform.get_area() > self.get_config().min_size:
                separate_face_id += 1
                separate_face_list.append(tmp_affordable_platform)

        convex_face_list = []

        for separate_face in separate_face_list:
            separate_face_division = ConcaveProcessor.decompose_concave_polygon(
                vertices_3d=separate_face.vertices,
                faces=separate_face.faces,
            )
            if len(separate_face_division) > 1:

                for i, (vertices, faces) in enumerate(separate_face_division):
                    tmp_affordable_platform = AffordablePlatform(
                        vertices=vertices,
                        faces=faces,
                        name=separate_face.name + "_division_" + str(i),
                    )
                    if tmp_affordable_platform.get_area() > self.get_config().min_size:
                        convex_face_list.append(tmp_affordable_platform)
            else:
                convex_face_list.append(separate_face)
            pass

        if not inverse:
            self.affordable_platforms = convex_face_list
        else:
            self.inverse_affordable_platforms = convex_face_list
        return convex_face_list

    def calculate_affordable_platforms(self, raw_platform=True, top_area=0):

        new_derivatives = []

        for i in range(len(self.affordable_platforms)):
            self.affordable_platforms[i].is_top_platform = True

        self.repair_mesh_instance()
        for i, bottom_platform in enumerate(self.affordable_platforms):
            bottom_convex = bottom_platform.get_convex_hull_2d()
            bottom_height = bottom_platform.get_height()[1]

            for j, other_platform in enumerate(self.affordable_platforms):
                if i == j:
                    continue

                other_convex = other_platform.get_convex_hull_2d()
                other_height = other_platform.get_height()[0]

                if (
                    other_height < bottom_height - MeshProcessor.get_config().EPS
                    or other_height
                    > bottom_height + self.affordable_platforms[i].available_height
                ):
                    continue

                intersection_convex = convex_hull_processor.ConvexHullProcessor_2d.intersection_of_2_convex_hulls(
                    bottom_convex, other_convex
                )
                area_bottom = bottom_convex.get_area()

                if (
                    intersection_convex is None
                    or intersection_convex.get_area() < MeshProcessor.get_config().EPS
                ):
                    continue
                if (
                    area_bottom * MeshProcessor.get_config().coverage_threshold
                    < intersection_convex.get_area()
                ):
                    self.affordable_platforms[i].available_height = min(
                        self.affordable_platforms[i].available_height,
                        other_height - bottom_height,
                    )
            for j, other_platform in enumerate(self.inverse_affordable_platforms):
                other_convex = other_platform.get_convex_hull_2d()
                other_height = other_platform.get_height()[0]
                if (
                    other_height < bottom_height - MeshProcessor.get_config().EPS
                    or other_height
                    > bottom_height + self.affordable_platforms[i].available_height
                ):
                    continue
                intersection_convex = convex_hull_processor.ConvexHullProcessor_2d.intersection_of_2_convex_hulls(
                    bottom_convex, other_convex
                )

                area_bottom = bottom_convex.get_area()
                # import ipdb
                # ipdb.set_trace()
                if (
                    intersection_convex is None
                    or intersection_convex.get_area() < MeshProcessor.get_config().EPS
                ):
                    continue
                if (
                    area_bottom * MeshProcessor.get_config().coverage_threshold
                    < intersection_convex.get_area()
                ):
                    self.affordable_platforms[i].available_height = min(
                        self.affordable_platforms[i].available_height,
                        other_height - bottom_height,
                    )

        self.clear_small_platforms()

        return new_derivatives

    def clear_small_platforms(self):
        if len(self.affordable_platforms) == 0:
            return self
        max_area = np.max(
            [platform.get_area() for platform in self.affordable_platforms]
        )
        self.affordable_platforms = [
            platform
            for platform in self.affordable_platforms
            if platform.get_area()
            > max_area * MeshProcessor.config().relative_size_ratio
            or "division" in platform.name
        ]
        return self

    def clear_too_low_platforms(self, height_threshold=None):
        height_threshold = (
            MeshProcessor.get_config().height_threshold
            if height_threshold is None
            else height_threshold
        )
        if len(self.affordable_platforms) == 0:
            return self
        self.affordable_platforms = [
            platform
            for platform in self.affordable_platforms
            if platform.available_height > height_threshold
        ]
        return self

    def cal_platform_avl_height(self):
        """

        check the height of the platform and calculate the available height of the platform.
        Args:
            height_steps: number of steps to sample the height
            width_steps: number of steps to sample the width

        """
        platform_list = self.affordable_platforms
        if len(platform_list) == 0:
            return
        for platform_id, platform in enumerate(platform_list):
            convex = platform.get_convex_hull_2d()
            height = platform.get_height()[1]
            indent_vertice_list = convex.get_indented_vertices_on_convex_hull()
            indent_vertice_list = [
                indent_vertice_list[i]
                for i in range(
                    0, len(indent_vertice_list), len(indent_vertice_list) // 10 + 1
                )
            ]
            mesh_triangle_list = []
            for platform_triangle in platform.faces:
                for mesh_triangle in self.mesh.triangles:
                    if not np.all(
                        np.isclose(
                            mesh_triangle,
                            platform.vertices[platform_triangle],
                            atol=1e-6,
                        )
                    ):
                        mesh_triangle_list.append(mesh_triangle)
            for vertex in indent_vertice_list:
                ray_direction = np.array([0, 0, 1])  # ray pointing upward
                ray_start_point = np.array([vertex[0], vertex[1], height + 1e-3])
                for triangle in mesh_triangle_list:

                    if min(triangle[:, 2]) > platform.available_height + height:
                        continue
                    intersection = (
                        basic_geometries.Basic3DGeometry.ray_triangle_intersection(
                            ray_start_point, ray_direction, triangle
                        )
                    )

                    if intersection is not None:
                        intersection_height = intersection[2]
                        if intersection_height < platform.available_height + height:
                            platform.available_height = min(
                                platform.available_height, intersection_height - height
                            )

        pass

    def check_platform_visability(
        self,
        heading=(1, 0),
        distance=0.25,
        height_steps=3,
        width_steps=3,
        height_range=(0.01, 0.35),
        vis_threshold=0.92,
    ):
        # glog.info(f"name: {self.name}, heading: {heading}, distance: {distance}, height_steps: {height_steps}, height_range: {height_range}")
        """
        check if the platform is visible from 4 sides of the heading.

        We select 4 observation directions: left, front, right, rear.
        For each direction, we sample the observation positions from the height range and check if most of the platform is visible from the observation positions.

        This is not a perfect method, but it is a good approximation with acceptable speed.


        Args:
            platform_list: list of platforms
            height_range: the height range of the platform
            samples_per_edge: number of samples per edge

        Returns:
            visible_platforms: list of platforms that are visible
        """

        vertices_2d = self.mesh.vertices[:, :2]

        platform_list = self.affordable_platforms

        bounds_2d = trimesh.bounds.oriented_bounds_2D(vertices_2d)

        rotation_matrix_inv = np.linalg.inv(bounds_2d[0][:2, :2])

        x_min = bounds_2d[1][0] * (-0.5)
        x_max = bounds_2d[1][0] * 0.5
        y_min = bounds_2d[1][1] * (-0.5)
        y_max = bounds_2d[1][1] * 0.5
        # define observation directions
        directions = {
            "left": np.array([0, -1]) @ rotation_matrix_inv.T,
            "front": np.array([-1, 0]) @ rotation_matrix_inv.T,
            "right": np.array([0, 1]) @ rotation_matrix_inv.T,
            "rear": np.array([1, 0]) @ rotation_matrix_inv.T,
        }

        # define observation places
        observation_positions = {
            "left": np.array(
                [
                    [
                        (x_min + x_max) / 2,
                        y_min - distance,
                        height_range[0]
                        + i * (height_range[1] - height_range[0]) / height_steps,
                    ]
                    for i in range(height_steps + 1)
                ]
            ),
            "front": np.array(
                [
                    [
                        x_min - distance,
                        (y_min + y_max) / 2,
                        height_range[0]
                        + i * (height_range[1] - height_range[0]) / height_steps,
                    ]
                    for i in range(height_steps + 1)
                ]
            ),
            "right": np.array(
                [
                    [
                        (x_min + x_max) / 2,
                        y_max + distance,
                        height_range[0]
                        + i * (height_range[1] - height_range[0]) / height_steps,
                    ]
                    for i in range(height_steps + 1)
                ]
            ),
            "rear": np.array(
                [
                    [
                        x_max + distance,
                        (y_min + y_max) / 2,
                        height_range[0]
                        + i * (height_range[1] - height_range[0]) / height_steps,
                    ]
                    for i in range(height_steps + 1)
                ]
            ),
        }

        fronting_point_positions = {
            "left": np.array(
                [
                    [
                        x_min + (x_max - x_min) * (i + 1) / (width_steps + 1),
                        y_min + (y_max - y_min) * 0.2,
                    ]
                    for i in range(width_steps + 1)
                ]
            ),
            "front": np.array(
                [
                    [
                        x_min + (x_max - x_min) * 0.2,
                        y_min + (y_max - y_min) * (i + 1) / (width_steps + 1),
                    ]
                    for i in range(height_steps + 1)
                ]
            ),
            "right": np.array(
                [
                    [
                        x_min + (x_max - x_min) * (i + 1) / (width_steps + 1),
                        y_min + (y_max - y_min) * 0.8,
                    ]
                    for i in range(width_steps + 1)
                ]
            ),
            "rear": np.array(
                [
                    [
                        x_min + (x_max - x_min) * 0.8,
                        y_min + (y_max - y_min) * (i + 1) / (width_steps + 1),
                    ]
                    for i in range(height_steps + 1)
                ]
            ),
        }

        mesh_triangles = self.mesh.triangles

        results = {}

        for direction_name, _ in directions.items():

            platform_visibility = []
            height_list = []
            for platform_id, platform in enumerate(platform_list):
                convex = platform.get_convex_hull_2d()
                height = platform.get_height()[1]
                height_list.append(height)
                platform_center = np.mean(convex.vertices, axis=0)
                platform_center_3d = np.array(
                    [platform_center[0], platform_center[1], height]
                )
                height_samples = np.linspace(
                    height_range[0], height_range[1], height_steps
                )

                position_visibility = []
                sample_platform_point_list = [platform_center_3d]

                convex_vertice_ids = np.random.choice(
                    len(convex.vertices), min(10, len(convex.vertices)), replace=False
                )

                convex_vertices = convex.vertices[convex_vertice_ids]

                # for vertex in convex_vertices:
                #     v = (vertex + platform_center * 0.5) / (0.5 + 1)
                #     sample_platform_point_list.append(np.array([v[0], v[1], height]))

                for fronting_point_pos in fronting_point_positions[direction_name]:
                    sample_platform_point_list.append(
                        np.array([fronting_point_pos[0], fronting_point_pos[1], height])
                    )

                for observer_pos in observation_positions[direction_name]:

                    observer_position = np.array(observer_pos)
                    observer_position[2] += height

                    view_vector = platform_center_3d - observer_position
                    view_distance = np.linalg.norm(view_vector)
                    visible_cnt = 0

                    for i, point in enumerate(sample_platform_point_list):

                        ray_vector = point - observer_position
                        ray_distance = np.linalg.norm(ray_vector)
                        ray_direction = ray_vector / ray_distance
                        hit = False
                        for triangle in mesh_triangles:
                            all_under = [vertex[2] < height for vertex in triangle]
                            all_upper = [
                                vertex[2] > observer_position[2] for vertex in triangle
                            ]
                            if all(all_under) or all(all_upper):
                                continue

                            intersection = basic_geometries.Basic3DGeometry.ray_triangle_intersection(
                                observer_position, ray_direction, triangle
                            )
                            if intersection is not None:
                                int_distance = np.linalg.norm(
                                    intersection - observer_position
                                )
                                #      if 'sofa' in self.name:
                                #           glog.info('height: {}, int_distance: {}, view_distance: {}'.format(height, int_distance, view_distance))
                                #          glog.info('intersection: {}, observer_position: {}, ray_direction: {}, triangle: {}'.format(intersection, observer_position, ray_direction, triangle))

                                if int_distance < view_distance * vis_threshold:
                                    hit = True
                                    break
                        if not hit:
                            visible_cnt += 1

                    if visible_cnt > 0:  # len(sample_platform_point_list) * 0.5:
                        position_visibility.append(True)
                        break
                    else:
                        position_visibility.append(False)

                platform_visibility.append(any(position_visibility))
                platform_list[platform_id].visible_directions[direction_name] = any(
                    position_visibility
                )

            results[direction_name] = {
                "platform_visibility": platform_visibility,
                "observation_positions": observation_positions[direction_name],
                "height": height_list,
            }

        return results, directions
