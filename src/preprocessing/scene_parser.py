import os
import trimesh
import numpy as np
from src.geometry.object_mesh_processor import MeshProcessor
from src.geometry.convex_hull_processor import ConvexHullProcessor_2d
from src.utils.config_manager import get_scene_element_config
from scipy.spatial.transform import Rotation as R
import glog

# import affordable_face


def reindex(vertex_indices, old_faces):
    new_faces = []
    for face in old_faces:
        new_face = [
            np.where(vertex_indices == face[i])[0][0] for i in range(face.shape[0])
        ]
        new_faces.append(new_face)
    return new_faces


# either a affordable platform or a object.
# mainly use to sort all the object and platform according to
class SceneElement:
    # contact_eps = 2.05e-2 for replica
    contact_eps = 5e-2
    bbox_eps = 4e-1
    # Height adjustment value to apply to objects that are incorrectly positioned below ground level
    ground_level_correction = 1e-4

    def __init__(
        self,
        name="",
        height=0,
        avl_height=0,
        top=0,
        bbox=[(0, 0), (1e9, 1e9)],
        bounding_points=[],
        convex_hull_2d=[],
        heading=(1, 0),
        instancetype="platform",
        visible_directions={},
        belong=None,
        mesh=None,
        centroid_translation=(0, 0, 0),
    ):
        # height of the platform
        self.name = name
        self.height = height
        self.heading = heading
        self.avl_height = avl_height
        self.top = top
        self.bounding_points = bounding_points
        self.centroid_translation = centroid_translation
        # self.bbox=bbox
        # remember to put it in here.
        self.convex_hull_2d = convex_hull_2d
        self.convex_hull_2d.heading = heading
        self.bbox = convex_hull_2d.get_headed_bbox_instance()
        self.visible_directions = visible_directions

        # type (object or platform)
        if instancetype == "platform" or instancetype == "object":
            self.type = instancetype
        else:
            print("Warning: The type of the platform is unexpected.")
        # belong = none for objects, = object_name for platforms

        # if mesh is not None(only when self is an object).
        self.mesh = mesh

        self.belong = belong

    @classmethod
    def get_config(cls):

        return get_scene_element_config()

    def __repr__(self):
        return f"""
            SceneElement(
            name={self.name},
            height={self.height}, 
            bbox={self.bbox}, 
            heading={self.heading}, 
            type={self.type}, 
            belong={self.belong})
            """

    def is_beyond_surface_bbox(self, other_object, heading=(1, 0)):

        box_left_side = [self.bbox[0], self.bbox[1]]
        box_right_side = [self.bbox[3], self.bbox[2]]
        box_front_side = [self.bbox[1], self.bbox[2]]
        box_rear_side = [self.bbox[0], self.bbox[3]]

        l, r, lr, u, d, ud = False, False, False, False, False, False

        for point in other_object.bbox:
            left = np.cross(
                box_left_side[1] - box_left_side[0], point - box_left_side[0]
            )
            right = np.cross(
                box_right_side[1] - box_right_side[0], point - box_right_side[0]
            )
            front = np.cross(
                box_front_side[1] - box_front_side[0], point - box_front_side[0]
            )
            rear = np.cross(
                box_rear_side[1] - box_rear_side[0], point - box_rear_side[0]
            )
            l |= left > 0 and right > 0
            lr |= left < 0 and right > 0
            r |= left < 0 and right < 0
            u |= front > 0 and rear > 0
            ud |= front < 0 and rear > 0
            d |= front < 0 and rear < 0

        at_directions = []

        if l and u:
            at_directions.append("front-left")
        if l and d:
            at_directions.append("rear-left")
        if r and u:
            at_directions.append("front-right")
        if r and d:
            at_directions.append("rear-right")

        # if l and u and d, the object is occupied the whole left side of the surface;
        # if l and ud, the object  is occupied the exact left side of the surface and/or the front-left / rear-left side of the surface
        if l and ((u and d) or ud):
            at_directions.append("left")
        if r and ((u and d) or ud):
            at_directions.append("right")
        if u and ((l and r) or lr):
            at_directions.append("front")
        if d and ((l and r) or lr):
            at_directions.append("rear")

        return at_directions

    @staticmethod
    def sort_platforms(platforms):
        # sort the platforms according to the height, and with the same height, let the platform be in front of the platform
        return sorted(platforms, key=lambda x: x.height)

    # Note that the logic like this cannot be use on the further stages such as generating the tasks.
    @staticmethod
    def calculate_contact_conditions(platforms):
        # sort platforms and objects, and use two pointers to find the contact conditions
        n = len(platforms)
        negative_height_adjustment = {}
        for i in range(n):
            if "GROUND" in platforms[i].name:
                continue
            if platforms[i].height < 0 and platforms[i].type == "object":
                negative_height_adjustment[platforms[i].name] = (
                    SceneElement.get_config().ground_level_correction
                    - platforms[i].height
                )
                platforms[i].height = SceneElement.get_config().ground_level_correction
            elif (
                platforms[i].type == "platform"
                and platforms[i].belong in negative_height_adjustment
            ):
                platforms[i].height += negative_height_adjustment[platforms[i].belong]
        platforms = SceneElement.sort_platforms(platforms)
        import ipdb

        ipdb.set_trace()
        # id_contacts should be a list of tuples, each tuple is a pair of indices(x,y), x is the index of the platform, y is the index of the object, and y is on top of x

        # Note:
        # For objects, we still use the whole bbox to calculate the contact conditions, but for platforms we use polygon to calculate the contact conditions.
        # It may cause some problems, in cases such as the object is a lamp with a huge head but a small base, and we may consider the head of the lamp as the contact condition.
        # But it can't be solved with simply calculate the platform with the lowest z, because there will be lanes and other platforms that are lower than the head of the lamp.
        id_contacts = []
        id_possible_contact = 0
        for i in range(n):
            if platforms[i].name == "GROUND" or i == 0:
                continue

            if platforms[i].type == "object":
                id_possible_contact = i - 1

                while True:
                    if i >= n or id_possible_contact >= n:
                        break
                    if (
                        id_possible_contact < 0
                        or platforms[i].height - platforms[id_possible_contact].height
                        > SceneElement.get_config().contact_eps
                    ):
                        id_possible_contact = i + 1

                    if (
                        i >= n
                        or id_possible_contact >= n
                        or (
                            platforms[id_possible_contact].height - platforms[i].height
                            > SceneElement.get_config().contact_eps
                            and id_possible_contact >= i
                        )
                    ):
                        break

                    if (
                        platforms[id_possible_contact].type == "object"
                        or platforms[id_possible_contact].belong == platforms[i].name
                    ):
                        id_possible_contact -= 1 if id_possible_contact < i else -1
                        continue
                    object_area = min(
                        platforms[i].convex_hull_2d.get_area(),
                        platforms[id_possible_contact].convex_hull_2d.get_area(),
                    )
                    bbox_intersect_area = platforms[
                        i
                    ].convex_hull_2d.intersect_area_with_another_convex(
                        platforms[id_possible_contact].convex_hull_2d
                    )
                    if (
                        bbox_intersect_area
                        > object_area * SceneElement.get_config().bbox_eps
                    ):
                        id_contacts.append((id_possible_contact, i))
                        break
                    else:
                        id_possible_contact -= 1 if id_possible_contact < i else -1
                        if bbox_intersect_area > 1e-6:
                            print(
                                "warning: Too small bbox_intersect_area",
                                bbox_intersect_area,
                                "object_area",
                                object_area,
                                "i",
                                platforms[i].name,
                                "id",
                                platforms[id_possible_contact].name,
                            )

        pass

        return platforms, id_contacts


class SceneObject:

    relative_threshold = 0.2

    def __init__(
        self,
        heading=(1, 0),
        geometries=[],
        derivatives=[],
        centroid_translation=(0, 0, 0),
        quaternion=(1, 0, 0, 0),
        rpy=(0, 0, 0),
        bbox=(0, 0, 0),
        name="",
        no_glb=False,
    ):
        # heading, should be a unit vector xy. Usually it should be either of 8 directions.
        self.heading = heading
        # list of geometries. TVstand has 3, other has 1 in replica_01.
        self.mesh = None

        self.eps = 1e-6

        self.derivatives = derivatives
        self.bounding_points = []
        # bbox center
        self.centroid_translation = centroid_translation
        # maybe it need to be deprecated
        self.quaternion = quaternion
        # roll, pitch, yaw
        self.rpy = rpy
        if rpy[0] != 0 or rpy[1] != 0 or rpy[2] != 0:
            self.quaternion = R.from_euler("xyz", rpy).as_quat()
        elif (
            quaternion[0] != 0
            or quaternion[1] != 0
            or quaternion[2] != 0
            or quaternion[3] != 0
        ):
            self.rpy = R.from_quat(quaternion).as_euler("xyz")
        else:
            print("Warning: No rotation information is provided.")

        self.bbox = self.get_bounding_box()
        self.height = self.bbox[0][2]
        self.convex_hull_2d = None

        # name
        self.name = name

        vertices = []

        vertices = np.array(vertices)
        if len(vertices):
            vertices += np.array(
                [
                    centroid_translation[0],
                    centroid_translation[1],
                    centroid_translation[2],
                ]
            )

            self.convex_hull_2d = ConvexHullProcessor_2d(vertices, heading)
            self.bbox = self.convex_hull_2d.get_headed_bbox_instance()
        else:
            self.convex_hull_2d = None
            self.bbox = [(0, 0, 0), (0, 0, 0)]

        # import ipdb
        # ipdb.set_trace()

        if len(geometries) > 1:
            # Find the geometry with the largest volume
            largest_volume = -1
            largest_idx = -1

            for i, geometry in enumerate(geometries):
                volume = geometry.volume
                if volume > largest_volume:
                    largest_volume = volume
                    largest_idx = i

            if largest_idx != -1:
                # Merge geometries with identical vertices
                main_geometry = geometries[largest_idx]
                main_vertices = main_geometry.vertices
                main_faces = main_geometry.faces.copy()

                for i, geometry in enumerate(geometries):
                    if i == largest_idx:
                        continue

                    # Check for identical vertices
                    other_vertices = geometry.vertices
                    other_faces = geometry.faces

                    # Use numpy's allclose to compare vertices (allowing for small floating-point errors)
                    if len(main_vertices) == len(other_vertices) and np.allclose(
                        main_vertices, other_vertices, atol=1e-6
                    ):
                        # If vertices are identical, merge the faces
                        main_faces = np.vstack([main_faces, other_faces])

                # Create the merged geometry
                merged_geometry = trimesh.Trimesh(
                    vertices=main_vertices, faces=main_faces
                )

                # Fix potential duplicate or invalid faces
                merged_geometry.remove_duplicate_faces()
                merged_geometry.remove_degenerate_faces()

                geometries = [merged_geometry]

        if len(geometries) > 0:
            self.mesh = MeshProcessor(geometries[0])

            vertices = geometries[0].vertices
            print(self.mesh.mesh)
        # for geometry in geometries:
        #     self.mesh.append(MeshProcessor(geometry))
        #     vertices.extend(geometry.vertices)

        # constants

    def is_point_in_object(self, point_2d):
        if self.mesh.point_2d_inside_mesh(point_2d):
            return True
        return False

    def set_ground(self):
        ground_vertices = np.array(
            [
                [-1e2, -1e2, 0],
                [1e2, -1e2, 0],
                [1e2, 1e2, 0],
                [-1e2, 1e2, 0],
                [-1e2, -1e2, -1e-4],
                [1e2, -1e2, -1e-4],
                [1e2, 1e2, -1e-4],
                [-1e2, 1e2, -1e-4],
            ]
        )

        ground_faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # bottom face
                [4, 5, 6],
                [4, 6, 7],  # top face
                [0, 1, 5],
                [0, 5, 4],  # front face
                [1, 2, 6],
                [1, 6, 5],  # right face
                [2, 3, 7],
                [2, 7, 6],  # back face
                [3, 0, 4],
                [3, 4, 7],  # left face
            ]
        )

        ground_mesh = trimesh.Trimesh(vertices=ground_vertices, faces=ground_faces)
        self.mesh = MeshProcessor(ground_mesh)
        self.convex_hull_2d = ConvexHullProcessor_2d(ground_vertices[:, :2], (1, 0))

        return None

    @property
    def base_threshold(self):
        return self.platform_base_threshold

    @property
    def absolute_threshold(self):
        return self.platform_absolute_threshold

    def get_bounding_box(self):
        cx, cy, cz = self.centroid_translation
        min_bbox = np.array([np.inf, np.inf, np.inf])
        max_bbox = np.array([-np.inf, -np.inf, -np.inf])
        mesh = self.mesh
        if mesh is None:
            return min_bbox, max_bbox
        bbox_min, bbox_max = mesh.get_bounding_box()
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
        min_bbox = np.minimum(vertices.min(axis=0), min_bbox)
        max_bbox = np.maximum(vertices.max(axis=0), max_bbox)
        min_bbox += np.array([cx, cy, cz])
        max_bbox += np.array([cx, cy, cz])
        return min_bbox, max_bbox

    def cal_heading(self):
        volume = -1
        geometry = self.mesh
        bounding_points, orientation = geometry.cal_orientation()
        if abs(orientation[0]) < 1e-4:
            orientation = (0, orientation[1])
        if abs(orientation[1]) < 1e-4:
            orientation = (orientation[0], 0)
        if volume < len(geometry.mesh.vertices):
            self.heading = orientation
            self.bounding_points = bounding_points  #

        if isinstance(self.bounding_points, list) and len(self.bounding_points) < 2:
            import ipdb

            ipdb.set_trace()

        self.bounding_points += np.array(
            [self.centroid_translation[0], self.centroid_translation[1]]
        )
        return None

    def cal_visible_direction(self):
        result_list, visible_direction = [], []
        geometry = self.mesh
        geometry.name = self.name
        glog.info(f"heading: {self.heading}")
        result, direction = geometry.check_platform_visability()
        if "sofa" in self.name or "cabinet" in self.name:
            glog.info(f"result: {result}")
            glog.info(f"direction: {direction}")
        result_list.append(result)
        visible_direction.append(direction)

    def cal_convex_hull_2d(self):
        vertices = []
        geometry = self.mesh
        vertices.extend(geometry.mesh.vertices)

        self.convex_hull_2d = ConvexHullProcessor_2d(vertices, self.heading)
        self.convex_hull_2d += np.array(
            [self.centroid_translation[0], self.centroid_translation[1]]
        )

    def repair_object(self):
        if self.mesh is None:
            return
        self.mesh = self.mesh.mesh_after_merge_close_vertices()
        self.mesh = self.mesh.repair_mesh_instance()

    def is_on_top_of_surface(self, other_object):

        convex_within = (
            self.convex_hull_2d.intersect_with_another_convex(
                other_object.convex_hull_2d
            )
            is not None
        )
        min_self, max_self = self.get_bounding_box()

        tz_top = max_self[2]
        oz_bottom = min_self[2]

        in_contact_z = abs(tz_top - oz_bottom) < self.eps

        return in_contact_z and convex_within

    def calculate_affordable_platforms(self):
        max_area = 0
        platform_list = []
        inverse_platform_list = []
        mesh = self.mesh

        mesh.get_raw_affordable_platforms(inverse=False)

        mesh.get_raw_affordable_platforms(inverse=True)
        # if 'StandardCounter' in self.name:
        #     import ipdb
        #     ipdb.set_trace()
        if len(mesh.affordable_platforms) == 0:
            return []
        max_area = max(
            max_area,
            np.max([platform.get_area() for platform in mesh.affordable_platforms]),
        )

        mesh.affordable_platforms = [
            platform
            for platform in mesh.affordable_platforms
            if platform.get_area() > max_area * SceneObject.relative_threshold
            or "division" in platform.name
        ]
        # mesh.inverse_affordable_platforms = [
        #         platform
        #         for platform in mesh.inverse_affordable_platforms
        #         if platform.get_area() > max_area * SceneObject.relative_threshold or 'division' in platform.name
        # ]
        inverse_platform_list.extend(mesh.inverse_affordable_platforms)
        platform_list.extend(mesh.affordable_platforms)

        for platform in mesh.affordable_platforms:
            for obstacle_platform in inverse_platform_list + platform_list:
                if np.allclose(
                    platform.get_height(),
                    obstacle_platform.get_height(),
                    atol=1e-6,
                ):
                    continue
                if (
                    obstacle_platform.get_height()[0] < platform.get_height()[0] + 1e-6
                    or obstacle_platform.get_height()[0]
                    > platform.get_height()[0] + platform.available_height
                ):
                    continue
                if platform.block_by_other_platform(obstacle_platform):
                    platform.available_height = min(
                        platform.available_height,
                        obstacle_platform.get_height()[0] - platform.get_height()[0],
                    )

        return []

    def get_affordable_platforms(self):
        res = []
        mesh = self.mesh
        res.extend(mesh.get_affordable_platforms())
        return res


def create_object_list_wo_mesh(object_mesh_list):
    pass


def create_room_envelope_mesh(mesh, wall_thickness=0.2, height_extension=0.5):
    """ """

    vertices_2d = mesh.vertices[:, :2]

    from scipy.spatial import ConvexHull

    hull_2d = ConvexHull(vertices_2d)
    hull_points = vertices_2d[hull_2d.vertices]

    z_min = mesh.vertices[:, 2].min() - height_extension
    z_max = mesh.vertices[:, 2].max() + height_extension

    center = np.mean(hull_points, axis=0)
    outer_points = []
    inner_points = []

    for point in hull_points:
        direction = point - center
        direction_norm = direction / np.linalg.norm(direction)

        outer_point = point + direction_norm * wall_thickness
        outer_points.append(outer_point)

        inner_points.append(point)

    outer_points = np.array(outer_points)
    inner_points = np.array(inner_points)

    vertices = []

    for point in inner_points:
        vertices.append([point[0], point[1], z_min])

    for point in outer_points:
        vertices.append([point[0], point[1], z_min])

    for point in inner_points:
        vertices.append([point[0], point[1], z_max])

    for point in outer_points:
        vertices.append([point[0], point[1], z_max])

    vertices = np.array(vertices)

    n_points = len(hull_points)
    faces = []

    for i in range(n_points):
        next_i = (i + 1) % n_points

        faces.append([i, n_points + i, n_points + next_i])
        faces.append([i, n_points + next_i, next_i])

    for i in range(n_points):
        next_i = (i + 1) % n_points

        base_idx = 2 * n_points
        faces.append(
            [base_idx + i, base_idx + n_points + next_i, base_idx + n_points + i]
        )
        faces.append([base_idx + i, base_idx + next_i, base_idx + n_points + next_i])

    for i in range(n_points):
        next_i = (i + 1) % n_points

        bottom_outer = n_points + i
        bottom_outer_next = n_points + next_i
        top_outer = 3 * n_points + i
        top_outer_next = 3 * n_points + next_i

        faces.append([bottom_outer, top_outer, top_outer_next])
        faces.append([bottom_outer, top_outer_next, bottom_outer_next])

    for i in range(n_points):
        next_i = (i + 1) % n_points

        bottom_inner = i
        bottom_inner_next = next_i
        top_inner = 2 * n_points + i
        top_inner_next = 2 * n_points + next_i

        faces.append([bottom_inner, top_inner_next, top_inner])
        faces.append([bottom_inner, bottom_inner_next, top_inner_next])

    faces = np.array(faces)

    envelope_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    envelope_mesh.remove_duplicate_faces()
    envelope_mesh.remove_degenerate_faces()
    envelope_mesh.fix_normals()

    return envelope_mesh


def create_object_list(object_mesh_list, calculate_affordable_platforms=True):
    objects = []
    idx = 0

    for mesh in object_mesh_list:
        if mesh.get("name") is not None:
            mesh["template_name"] = mesh.get("name")
        path_prefix = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "scene_graph"
        )
        if mesh is None:
            continue
        if (
            mesh.get("template_name") == "ground"
            or mesh.get("template_name") == "scene_background"
            or mesh.get("template_name") is None
        ):
            continue
        if "door" in mesh.get("template_name") or "drawer" in mesh.get("template_name"):
            continue

        if calculate_affordable_platforms == False and "tvstand" in mesh.get(
            "template_name"
        ):
            continue

        glb = trimesh.load(os.path.join(path_prefix, mesh["visual_path"]))

        geometries = []
        centroid_translation = (
            mesh["centroid_translation"]["x"],
            mesh["centroid_translation"]["y"],
            mesh["centroid_translation"]["z"],
        )

        quaternion = (
            mesh["quaternion"]["w"],
            mesh["quaternion"]["x"],
            mesh["quaternion"]["y"],
            mesh["quaternion"]["z"],
        )
        name = [node for node in glb.graph.nodes]
        for mesh_name, geometry in glb.geometry.items():
            name = glb.graph.geometry_nodes[mesh_name][0]
            """
            we need this if running replicaCAD.
            """
            # if calculate_affordable_platforms == False:
            #       if not ("wall" in name or "blind" in name):
            #           continue
            transform_matrix = np.array(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
            )
            global_matrix = np.array(glb.graph[name][0])

            # Only for replica_01, we need to scale the mesh
            if (
                "kitchenCupboard_01" in mesh["template_name"]
                or "chestOfDrawers_01" in mesh["template_name"]
            ):
                for i in range(3):
                    for j in range(3):
                        global_matrix[i][j] /= 2.5

            quaternion_matrix = trimesh.transformations.quaternion_matrix(quaternion)

            geometry.apply_transform(transform_matrix @ global_matrix)

            geometry.apply_transform(quaternion_matrix)

            if calculate_affordable_platforms == False:
                # for vertice_id, vertice in enumerate(geometry.vertices):
                #     geometry.vertices[vertice_id][2] -= 0.1

                new_glb = trimesh.Trimesh(
                    vertices=geometry.vertices, faces=geometry.faces
                )
                if new_glb.is_volume == False:

                    try:
                        envelope_mesh = create_room_envelope_mesh(
                            new_glb, wall_thickness=0.15
                        )
                        if envelope_mesh.is_watertight:
                            geometries.append(envelope_mesh)
                        else:
                            components = new_glb.split(only_watertight=False)
                            for i in range(len(components)):
                                if components[i].is_volume == False:
                                    component = components[i]
                                    components[i] = MeshProcessor.repair_mesh(
                                        components[i]
                                    )
                                    if len(components[i].vertices) < 4:
                                        continue
                                    if components[i].is_volume == False:
                                        components[i].vertices += (
                                            np.random.rand(
                                                *components[i].vertices.shape
                                            )
                                            * 1e-3
                                        )
                                        components[i] = components[i].convex_hull
                                        if components[i].volume > 1:
                                            continue
                                geometries.append(components[i])
                    except Exception as e:

                        components = new_glb.split(only_watertight=False)
                        for i in range(len(components)):
                            if components[i].is_volume == False:
                                component = components[i]
                                components[i] = MeshProcessor.repair_mesh(components[i])
                                if len(components[i].vertices) < 4:
                                    continue
                                if components[i].is_volume == False:
                                    components[i].vertices += (
                                        np.random.rand(*components[i].vertices.shape)
                                        * 1e-3
                                    )
                                    components[i] = components[i].convex_hull
                                    if components[i].volume > 1:
                                        continue
                            geometries.append(components[i])
            else:
                geometries.append(geometry)

        idx += 1

        if len(geometries) > 1 and "tvstand" in mesh["template_name"]:
            combined_vertices = []
            combined_faces = []
            vertex_offset = 0

            for geometry in geometries:
                combined_vertices.extend(geometry.vertices)
                combined_faces.extend(geometry.faces + vertex_offset)
                vertex_offset += len(geometry.vertices)

            combined_vertices = np.array(combined_vertices)
            combined_faces = np.array(combined_faces)
            combined_mesh = trimesh.Trimesh(
                vertices=combined_vertices, faces=combined_faces
            )

            # Compute the convex hull of the combined mesh
            convex_hull = combined_mesh.convex_hull

            geometries = [
                trimesh.Trimesh(vertices=convex_hull.vertices, faces=convex_hull.faces)
            ]

        rpy = trimesh.transformations.euler_from_quaternion(quaternion, axes="sxyz")

        if calculate_affordable_platforms == False:
            for geometry in geometries:
                new_objects = SceneObject(
                    geometries=[geometry],
                    centroid_translation=centroid_translation,
                    quaternion=quaternion,
                    rpy=rpy,
                    name=mesh["template_name"],
                )
                new_objects.repair_object()
                if new_objects.mesh is None:
                    glog.warning(
                        f"Object {mesh['template_name']} has no mesh after repair, skipping."
                    )
                    continue
                new_objects.cal_heading()
                new_objects.cal_convex_hull_2d()
                objects.append(new_objects)
        else:
            new_object = SceneObject(
                geometries=geometries,
                centroid_translation=centroid_translation,
                quaternion=quaternion,
                rpy=rpy,
                name=mesh["template_name"],
            )
            new_object.repair_object()
            if new_object.mesh is None:
                glog.warning(
                    f"Object {mesh['template_name']} has no mesh after repair, skipping."
                )
                continue
            new_object.cal_heading()
            new_object.cal_convex_hull_2d()
            new_object.calculate_affordable_platforms()
            objects.append(new_object)

            # pass
            bottom = new_object.get_bounding_box()[0][2]
            if bottom < 0.1:
                new_object.cal_visible_direction()

    return objects


def main():
    path_prefix = "D:/workplace/scene_graph/task_generation/task_generation/"
    glb = trimesh.load(
        os.path.join(
            path_prefix,
            "scene_datasets/replica_cad_dataset/objects/frl_apartment_stool_02.glb",
        )
    )
    geometries = list(glb.geometry.values())
    for i in range(len(geometries)):
        transform_matrix = np.array(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )
        geometries[i].apply_transform(transform_matrix)
    quaternion = (0.7071067811865476, 0, 0, 0.7071067811865476)
    rpy = trimesh.transformations.euler_from_quaternion(quaternion, axes="sxyz")
    new_object = SceneObject(
        geometries=geometries,
        centroid_translation=(0, 0, 0),
        quaternion=quaternion,
        rpy=rpy,
        name="tvstand",
    )
    new_object.repair_object()
    potential_derivatives = new_object.calculate_affordable_platforms()


if __name__ == "__main__":
    main()
