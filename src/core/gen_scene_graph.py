# %%
"""
todo:
refactorize this.

implement check every object's


"""

# %%
import json
import numpy as np
from sympy import im
import glog

# from visualize_tree import visualize_tree
from scipy.spatial.transform import Rotation as R
import argparse
import copy
from src.preprocessing import scene_parser
from src.geometry.convex_hull_processor import (
    ConvexHullProcessor_2d,
)
from src.geometry.basic_geometries import Basic2DGeometry
from src.geometry.object_mesh_processor import MeshProcessor
from src.geometry.rectangle_query_processor import RectangleQuery2D
from src.geometry.ground_coverage_analyzer import GroundCoverageAnalyzer
from src.utils.string_convertor import StringConvertor
import sapien
import transforms3d
from src.utils.image_renderer import image_render_processor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from ..image_renderer import image_render_processor

INF = 1e6
eps = 1e-6


# %%
# Function to load a JSON file from disk
def load_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


EIGHT_DIRECTIONS = [
    "rear",
    "rear-left",
    "left",
    "front-left",
    "front",
    "front-right",
    "right",
    "rear-right",
]
OPPOSITE_DIRECTIONS = {
    "front": "rear",
    "front-right": "rear-left",
    "right": "left",
    "rear-right": "front-left",
    "rear": "front",
    "rear-left": "front-right",
    "left": "right",
    "front-left": "rear-right",
}
FOUR_DIAGONAL_DIRECTIONS = ["rear-left", "front-left", "front-right", "rear-right"]


class TreePlatform:
    def __init__(
        self,
        name="",
        children=[],
        convex_hull_2d=None,
        heading=None,
        avl_height=3.1,
        bottom=0,
        bel_object=None,
        visible_directions={
            "front": False,
            "rear": False,
            "right": False,
            "left": False,
        },
    ):
        self.name = name
        self.children = children
        self.convex_hull_2d = convex_hull_2d
        self.heading = heading
        self.avl_height = avl_height
        self.bottom = bottom
        self.bel_object = bel_object
        self.placement_dict = {}
        self.bbox = convex_hull_2d.get_headed_bbox_instance()
        self.rect_list = []
        self.visible_directions = visible_directions
        self.free_space = None
        self.standing_point_list = []

        if self.free_space is None and self.name != "GROUND":
            cos_theta, sin_theta = heading[0], heading[1]
            points = self.convex_hull_2d.get_headed_bbox_instance_with_heading(heading)

            freespace_seg_list = [
                [points[0], points[1]],
                [points[1], points[2]],
                [points[2], points[3]],
                [points[3], points[0]],
            ]

            # for i, freespace_seg in enumerate(freespace_seg_list):
            #     if np.linalg.norm(freespace_seg[0] - freespace_seg[1]) < 0.6:
            #         freespace_seg_list[i] = [
            #             (freespace_seg[0] + freespace_seg[1]) / 2 + (freespace_seg[0] - freespace_seg[1]) / 2 * 0.6 ,
            #             (freespace_seg[0] + freespace_seg[1]) / 2 - (freespace_seg[0] - freespace_seg[1]) / 2 * 0.6,
            #         ]
            #         continue
            def extend_point(p, direction):
                if direction == "right":
                    normal = np.array([cos_theta, sin_theta])
                elif direction == "left":
                    normal = np.array([-cos_theta, -sin_theta])
                elif direction == "front":
                    normal = np.array([-sin_theta, cos_theta])
                elif direction == "rear":
                    normal = np.array([sin_theta, -cos_theta])
                elif direction == "rear-left":
                    normal = np.array([sin_theta, -cos_theta]) + np.array(
                        [-cos_theta, -sin_theta]
                    )
                elif direction == "front-left":
                    normal = np.array([-sin_theta, cos_theta]) + np.array(
                        [-cos_theta, -sin_theta]
                    )
                elif direction == "front-right":
                    normal = np.array([-sin_theta, cos_theta]) + np.array(
                        [cos_theta, sin_theta]
                    )
                elif direction == "rear-right":
                    normal = np.array([sin_theta, -cos_theta]) + np.array(
                        [cos_theta, sin_theta]
                    )
                else:
                    return np.array(p)
                return np.array(p) + 0.5 * np.array(normal)

            self.free_space = [
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(freespace_seg_list[3][1], "rear"),
                        freespace_seg_list[3][1],
                        freespace_seg_list[3][0],
                        extend_point(freespace_seg_list[3][0], "rear"),
                    ],
                    "Critical_space": [
                        extend_point(freespace_seg_list[3][1], "rear"),
                        freespace_seg_list[3][1],
                        freespace_seg_list[3][0],
                        extend_point(freespace_seg_list[3][0], "rear"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "rear-left"),
                        extend_point(points[0], "left"),
                        points[0],
                        extend_point(points[0], "rear"),
                    ],
                    "Critical_space": [
                        extend_point(points[0], "rear-left"),
                        extend_point(points[0], "left"),
                        points[0],
                        extend_point(points[0], "rear"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(freespace_seg_list[0][0], "left"),
                        extend_point(freespace_seg_list[0][1], "left"),
                        freespace_seg_list[0][1],
                        freespace_seg_list[0][0],
                    ],
                    "Critical_space": [
                        extend_point(freespace_seg_list[0][0], "left"),
                        extend_point(freespace_seg_list[0][1], "left"),
                        freespace_seg_list[0][1],
                        freespace_seg_list[0][0],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[1], "left"),
                        extend_point(points[1], "front-left"),
                        extend_point(points[1], "front"),
                        points[1],
                    ],
                    "Critical_space": [
                        extend_point(points[1], "left"),
                        extend_point(points[1], "front-left"),
                        extend_point(points[1], "front"),
                        points[1],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        freespace_seg_list[1][0],
                        extend_point(freespace_seg_list[1][0], "front"),
                        extend_point(freespace_seg_list[1][1], "front"),
                        freespace_seg_list[1][1],
                    ],
                    "Critical_space": [
                        freespace_seg_list[1][0],
                        extend_point(freespace_seg_list[1][0], "front"),
                        extend_point(freespace_seg_list[1][1], "front"),
                        freespace_seg_list[1][1],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[2],
                        extend_point(points[2], "front"),
                        extend_point(points[2], "front-right"),
                        extend_point(points[2], "right"),
                    ],
                    "Critical_space": [
                        points[2],
                        extend_point(points[2], "front"),
                        extend_point(points[2], "front-right"),
                        extend_point(points[2], "right"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        freespace_seg_list[2][1],
                        freespace_seg_list[2][0],
                        extend_point(freespace_seg_list[2][0], "right"),
                        extend_point(freespace_seg_list[2][1], "right"),
                    ],
                    "Critical_space": [
                        freespace_seg_list[2][1],
                        freespace_seg_list[2][0],
                        extend_point(freespace_seg_list[2][0], "right"),
                        extend_point(freespace_seg_list[2][1], "right"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[3], "rear"),
                        points[3],
                        extend_point(points[3], "right"),
                        extend_point(points[3], "rear-right"),
                    ],
                    "Critical_space": [
                        extend_point(points[3], "rear"),
                        points[3],
                        extend_point(points[3], "right"),
                        extend_point(points[3], "rear-right"),
                    ],
                },
            ]

    def get_bbox_line(self, dir):
        if dir % 2 != 0 or dir < 0 or dir > 7:
            return None

        bbox = self.convex_hull_2d.get_headed_bbox_instance()
        return [bbox[dir // 2], bbox[(dir // 2 + 3) % 4]]

    #        glog.info(f"platform name: {self.name}, visible_directions: {self.visible_directions}")

    def get_name_for_interaction(self):
        name_for_interaction = self.name
        if "Object/" in name_for_interaction:
            name_for_interaction = name_for_interaction.split("Object/")[-1]
        if "RoboTHOR_" in name_for_interaction:
            name_for_interaction = name_for_interaction.split("RoboTHOR_")[-1]
        if "frl_apartment_" in name_for_interaction:
            name_for_interaction = name_for_interaction.split("frl_apartment_")[-1]

        return f"object_{StringConvertor.get_noslash_name_wo_id(name_for_interaction)}_platform_{StringConvertor.get_id(self.name)}"

    def get_categories_and_objects(self):
        """
        get the number of categories of children and the number of children.
        """
        major_category = set()
        minor_category = set()
        for child in self.children:
            major_category.add(StringConvertor.get_category(child.name))
            minor_category.add(StringConvertor.get_name_wo_id(child.name))

        major_category_list = list(major_category)
        minor_category_list = list(minor_category)
        children_name_list = [
            child.get_name_for_interaction() for child in self.children
        ]

        return major_category_list, minor_category_list, children_name_list

    def single_freespace_available(self, object_node, single_type="all"):
        object_node_convex = ConvexHullProcessor_2d(
            vertices=object_node.object.convex_hull_2d.vertices, heading=self.heading
        )
        if single_type == "empty":
            if len(self.children) > 0:
                return False, np.array([-1e5, -1e5])
            platform_bbox = self.convex_hull_2d.get_headed_bbox_instance()
            platform_13_bbox = np.array(
                [
                    (platform_bbox[i] - platform_bbox[0]) / 3 + platform_bbox[0]
                    for i in range(4)
                ]
            )
            if object_node_convex.can_fit_in(platform_13_bbox):
                return True, object_node_convex.get_fit_in_translation(platform_13_bbox)
            return False, np.array([-1e6, -1e6])

        if len(self.children) == 0:
            return False, np.array([-1e5, -1e5])

        if single_type == "all":
            freespace_list = self.get_freespace_list()
        else:
            freespace_id_list = []
            if len(single_type) == 1:
                child_name = single_type[0]
                child_id = self.get_child_id(child_name)
                if child_id == -1:
                    return False, np.array([-1e6, -1e6])
                freespace_id_list = [(child_id, i) for i in range(8)]
            elif len(single_type) == 2:
                if isinstance(single_type[-1], str):
                    freespace_id_list_a = []
                    freespace_id_list_b = []
                    child_name_a = single_type[0]
                    child_name_b = single_type[1]
                    child_id_a = self.get_child_id(child_name_a)
                    child_id_b = self.get_child_id(child_name_b)
                    for i in range(8):
                        freespace_id_list_a.append((child_id_a, i))
                    for i in range(8):
                        freespace_id_list_b.append((child_id_b, i))

                    for i in range(8):
                        if object_node_convex.can_fit_in(
                            self.children[child_id_a].free_space[i]["Critical_space"]
                        ) and object_node_convex.can_fit_in(
                            self.children[child_id_b].free_space[(i + 4) % 8][
                                "Critical_space"
                            ]
                        ):
                            return True, object_node_convex.get_fit_in_translation(
                                self.children[child_id_a].free_space[i][
                                    "Critical_space"
                                ]
                            )
                    return False, np.array([-1e6, -1e6])
                else:
                    child_name_a = single_type[0]
                    child_id_a = self.get_child_id(child_name_a)
                    freespace_id_list = [(child_id_a, single_type[1])]

            freespace_list = self.get_freespace_list(
                freespace_id_list=freespace_id_list
            )
        for freespace in freespace_list:
            if object_node_convex.can_fit_in(freespace):
                return True, object_node_convex.get_fit_in_translation(freespace)
        return False, np.array([-1e6, -1e6])

    def rename_with_name_map(self, name_map):
        """

        rename the platform and all its children with the name map.

        Args:
            name_map (dict): A dictionary mapping old names to new names.

        Returns:
            None. The function modifies all the features of the platform according to the name_map.

        """
        self.name = StringConvertor.rename_with_map(self.name, name_map)
        for child in self.children:
            child.rename_with_name_map(name_map)
        self.bel_object = StringConvertor.rename_with_map(self.bel_object, name_map)
        return self

    @staticmethod
    def sort_platforms(platforms):
        """

        sort the platforms according to the bottom height.

        Args:
            platforms (list): A list of TreePlatform objects.

        Returns:
            sorted_platforms (list): A list of TreePlatform objects sorted by their bottom height.
        """
        sorted_platforms = sorted(platforms, key=lambda x: x.bottom)
        return sorted_platforms

    def bfs_children(self):
        queue = []
        for child in self.children:
            queue.append(child)
        while len(queue) > 0:
            cur_node = queue.pop(0)
            for child in cur_node.children:
                queue.append(child)

        self.children = queue
        return queue

    def freespace_is_visible(self, dir):
        dir = EIGHT_DIRECTIONS[dir]
        if dir not in self.visible_directions:
            return False
        return self.visible_directions[dir]

    def get_freespace_list(self, freespace_id_list=None):

        freespace_list = []
        if freespace_id_list is None:
            for child in self.children:
                for dir in range(8):
                    freespace_list.append(child.free_space[dir]["Critical_space"])
        else:
            for object_id, freespace_id in freespace_id_list:
                if object_id >= len(self.children):
                    continue
                child = self.children[object_id]
                if freespace_id >= 8:
                    continue
                freespace_list.append(child.free_space[freespace_id]["Critical_space"])

        return freespace_list

    def get_child_id(self, child_name):
        for i, child in enumerate(self.children):
            if child.name == child_name:
                return i
        return -1

    def get_dir_point(self, item_node=None, dir="center"):
        object_node_convex = ConvexHullProcessor_2d(
            vertices=item_node.object.convex_hull_2d.vertices, heading=self.heading
        )

        platform_bbox = self.convex_hull_2d.get_headed_bbox_instance()
        object_node_bbox = object_node_convex.get_headed_bbox_instance()
        if dir == "center":
            return np.mean(platform_bbox, axis=0)

        direction = EIGHT_DIRECTIONS.index(dir)
        platform_point = (
            platform_bbox[direction // 2]
            if direction % 2 == 1
            else (
                platform_bbox[direction // 2] + platform_bbox[(3 + direction // 2) % 4]
            )
            / 2
        )
        if direction in [7, 0, 1]:
            platform_point += (object_node_bbox[1] - object_node_bbox[0]) * 0.5
        if direction in [1, 2, 3]:
            platform_point += (object_node_bbox[2] - object_node_bbox[1]) * 0.5
        if direction in [3, 4, 5]:
            platform_point += (object_node_bbox[3] - object_node_bbox[2]) * 0.5
        if direction in [5, 6, 7]:
            platform_point += (object_node_bbox[0] - object_node_bbox[3]) * 0.5

        return platform_point

        pass

    def take_picture_with_marked_rect(
        self,
        scene,
        view="human_full",  # top
        camera_xy=None,
        need_mark_rectangle_list=[],
        same_color=False,
        width=1920,
        height=1080,
        focus_ratio=0.6,
        save_path=None,
    ):

        highest_child = (
            max([child.top for child in self.children])
            if self.children != []
            else self.bottom + 0.15
        )

        z_range = [highest_child, self.bottom + self.avl_height + 0.3]
        platform_2d_bbox = self.convex_hull_2d.get_headed_bbox_instance()

        platform_3d_bbox = [
            np.append(point, self.bottom) for point in platform_2d_bbox
        ] + [
            np.append(point, self.bottom + self.avl_height)
            for point in platform_2d_bbox
        ]

        might_mark_freespace_list = []

        for rect in need_mark_rectangle_list:
            rect_3d = [np.append(point, self.bottom) for point in rect]
            might_mark_freespace_list.append(rect_3d)

        optimal_pose, optimal_fovy = (
            image_render_processor.auto_get_optimal_camera_pose_for_object(
                view=view,
                camera_xy=camera_xy,
                z_range=z_range,
                object_bbox=platform_3d_bbox,
                platform_rect=platform_3d_bbox[:4],
                width=width,
                height=height,
                focus_ratio=focus_ratio,
            )
        )

        img = image_render_processor.auto_render_image_refactored(
            scene,
            pose=optimal_pose,
            fovy=optimal_fovy,
            width=width,
            height=height,
            might_mark_object_cuboid_list=[],
            might_mark_freespace_list=might_mark_freespace_list,
            rectangle_grey=same_color,
            save_path=save_path,
        )

        pass
        return img

    def find_available_places(
        self,
        obstacle_list,
        freespace_list,
        target,
        min_step=0.02,
    ):
        table_vertices = self.convex_hull_2d.get_headed_bbox_instance()
        rotation_angle = np.arctan2(self.heading[1], self.heading[0])

        transformed_table = np.array(
            [
                Basic2DGeometry.rotate_point_counterclockwise(
                    table_vertice, rotation_angle
                )
                for table_vertice in table_vertices
            ]
        )

        transformed_obstacles = np.array(
            [
                np.array(
                    [
                        Basic2DGeometry.rotate_point_counterclockwise(
                            obstacle_vertice, rotation_angle
                        )
                        for obstacle_vertice in obstacle
                    ]
                )
                for obstacle in obstacle_list
            ]
        )
        transformed_freespaces = np.array(
            [
                np.array(
                    [
                        Basic2DGeometry.rotate_point_counterclockwise(
                            freespace_vertice, rotation_angle
                        )
                        for freespace_vertice in freespace
                    ]
                )
                for freespace in freespace_list
            ]
        )

        #        glog.info(f"transformed_obstacles: {transformed_obstacles}")

        transformed_target = np.array(
            [
                Basic2DGeometry.rotate_point_counterclockwise(
                    target_vertice, rotation_angle
                )
                for target_vertice in target
            ]
        )

        x_min, x_max, y_min, y_max = (
            np.min(transformed_table[:, 0]),
            np.max(transformed_table[:, 0]),
            np.min(transformed_table[:, 1]),
            np.max(transformed_table[:, 1]),
        )
        target_x_min, target_x_max, target_y_min, target_y_max = (
            np.min(transformed_target[:, 0]),
            np.max(transformed_target[:, 0]),
            np.min(transformed_target[:, 1]),
            np.max(transformed_target[:, 1]),
        )
        target_x_len, target_y_len = (
            target_x_max - target_x_min,
            target_y_max - target_y_min,
        )

        if len(obstacle_list) == 0:
            result = ""
            if target_x_len < x_max - x_min and target_y_len < y_max - y_min:
                result = "wholetable"
            if (
                target_x_len < (x_max - x_min) / 3 * 2
                and target_y_len < (y_max - y_min) / 3 * 2
            ):
                result += ",center"
            return result, [], [], [], [], {}, {}, {}

        prefix_points, prefix_values = [], []
        sensible_xs, sensible_ys = [], []
        obstacle_query = RectangleQuery2D(dimension=2)

        for obs in transformed_obstacles:
            obs_x_min, obs_x_max, obs_y_min, obs_y_max = (
                np.min(obs[:, 0]),
                np.max(obs[:, 0]),
                np.min(obs[:, 1]),
                np.max(obs[:, 1]),
            )
            obstacle_query.add_rectangle(obs)

            sensible_xs.append(obs_x_min)
            sensible_xs.append(obs_x_max)
            sensible_ys.append(obs_y_min)
            sensible_ys.append(obs_y_max)

        sensible_xs.extend([x_min, x_max])
        sensible_ys.extend([y_min, y_max])

        sensible_xs = [x + min_step * 2 for x in sensible_xs] + [
            x - min_step * 2 for x in sensible_xs
        ]
        sensible_ys = [y + min_step * 2 for y in sensible_ys] + [
            y - min_step * 2 for y in sensible_ys
        ]

        freespace_query = RectangleQuery2D(dimension=2)
        for freespace in transformed_freespaces:
            freespace_query.add_rectangle(freespace)

        available_positions = []

        for op in range(4):
            op_offset = transformed_target[0] - transformed_target[op]
            for x in sensible_xs:
                for y in sensible_ys:

                    x1, x2, y1, y2 = x, x + target_x_len, y, y + target_y_len
                    x1, x2, y1, y2 = (
                        x1 + op_offset[0],
                        x2 + op_offset[0],
                        y1 + op_offset[1],
                        y2 + op_offset[1],
                    )
                    if x1 < x_min or x2 > x_max or y1 < y_min or y2 > y_max:
                        continue

                    tmp_rect = np.array(
                        [
                            np.array([x1, y1]),
                            np.array([x1, y2]),
                            np.array([x2, y2]),
                            np.array([x2, y1]),
                        ]
                    )

                    results, ids = obstacle_query.query_rectangle(tmp_rect)
                    if len(results) == 0:
                        available_positions.append(tmp_rect)

        put_on_platform = len(available_positions) > 0
        put_around_object_set = set()
        put_on_object_dir_set = set()
        put_between_2_object_set = set()
        put_around_object_dict = {}
        put_on_object_dir_dict = {}
        put_between_2_object_dict = {}

        for avl_rect in available_positions:
            at_freespace_id, at_freespace = freespace_query.query_rectangle(avl_rect)
            at_freespace_object_id = [id // 8 for id in at_freespace_id]
            at_freespace_dir_id = [[] for i in range(8)]
            avl_rect_center = np.mean(avl_rect, axis=0)
            original_avl_rect_center = Basic2DGeometry.rotate_point_counterclockwise(
                avl_rect_center, -rotation_angle
            )
            for id in at_freespace_id:
                at_freespace_dir_id[id % 8].append(id // 8)

            counts = {}
            for object_id in at_freespace_object_id:
                if object_id not in counts:
                    counts[object_id] = 0
                counts[object_id] += 1

            for freespace_id in at_freespace_id:
                if counts[freespace_id // 8] >= 1:
                    put_on_object_dir_dict[(freespace_id // 8, freespace_id % 8)] = (
                        original_avl_rect_center
                    )
                    put_on_object_dir_set.add((freespace_id // 8, freespace_id % 8))

            for dir in range(0, 4):
                for item_ida in at_freespace_dir_id[dir]:
                    for item_idb in at_freespace_dir_id[dir + 4]:
                        put_between_2_object_set.add((item_ida, item_idb))
                        put_between_2_object_dict[(item_ida, item_idb)] = (
                            original_avl_rect_center
                        )

            for object_id in at_freespace_object_id:
                put_around_object_dict[object_id] = original_avl_rect_center
                put_around_object_set.add(object_id)

        put_around_object_list = list(put_around_object_set)
        put_on_object_dir_list = list(put_on_object_dir_set)
        put_between_2_object_list = list(put_between_2_object_set)

        return (
            "non_empty",
            put_on_platform,
            put_around_object_list,
            put_on_object_dir_list,
            put_between_2_object_list,
            put_around_object_dict,
            put_on_object_dir_dict,
            put_between_2_object_dict,
        )

    def get_fitable_tasks_any_number(self, item, all_standing_directions=[0, 2, 4, 6]):
        """
             def find_available_places(
            self, obstacle_list, freespace_2d_list, target, min_step=0.02, solve_mode="all_intensive"
        ):
        """

        target_convex = item.object.convex_hull_2d
        target_bbox = ConvexHullProcessor_2d.get_headed_bbox(
            vertices=target_convex.vertices, heading=self.heading
        )
        obstacle_list = []
        freespace_list = []
        child_name_list = [
            child.name
            for child in self.children
            if child.name != item.name and child.parent.name == self.bel_object
        ]
        for child in self.children:
            if not (child.name != item.name and child.parent.name == self.bel_object):
                continue
            child_bbox = ConvexHullProcessor_2d.get_headed_bbox(
                vertices=child.object.convex_hull_2d.vertices, heading=self.heading
            )
            obstacle_list.append(child_bbox)
            for dir in range(8):
                freespace_list.append(child.free_space[dir]["Critical_space"])

        (
            result,
            put_on_platform,
            put_around_object_list,
            put_on_object_dir_list,
            put_between_2_object_list,
            put_around_object_dict,
            put_on_object_dir_dict,
            put_between_2_object_dict,
        ) = self.find_available_places(
            obstacle_list=obstacle_list,
            freespace_list=freespace_list,
            target=target_bbox,
            min_step=0.01,
        )

        if len(all_standing_directions) == 0:
            return ("non_empty", False, [], [], [], {}, {}, {})

        for i, put_around_object_task in enumerate(put_around_object_list):
            put_around_object_dict[child_name_list[put_around_object_task]] = (
                put_around_object_dict.pop(put_around_object_task)
            )
            put_around_object_list[i] = child_name_list[put_around_object_task]
        for i, put_on_object_task in enumerate(put_on_object_dir_list):
            for standing_directions in all_standing_directions:
                new_key = (
                    child_name_list[put_on_object_task[0]],
                    EIGHT_DIRECTIONS[
                        (put_on_object_task[1] - standing_directions + 8) % 8
                    ],
                )

                put_on_object_dir_dict[new_key] = put_on_object_dir_dict.pop(
                    put_on_object_task
                )

                put_on_object_dir_list[i] = new_key

        for i, put_between_2_object_task in enumerate(put_between_2_object_list):
            new_key = (
                child_name_list[put_between_2_object_task[0]],
                child_name_list[put_between_2_object_task[1]],
            )
            put_between_2_object_dict[new_key] = put_between_2_object_dict.pop(
                put_between_2_object_task
            )
            put_between_2_object_list[i] = new_key
            pass

        put_on_object_dir_list = list(set(put_on_object_dir_list))
        put_between_2_object_list = list(set(put_between_2_object_list))

        return (
            result,
            put_on_platform,
            put_around_object_list,
            put_on_object_dir_list,
            put_between_2_object_list,
            put_around_object_dict,
            put_on_object_dir_dict,
            put_between_2_object_dict,
        )


class TreeNode:
    robot_fit_threshold = 0.35
    minimal_freespace_width = 0.02

    def __init__(
        self,
        name,
        entity_config=None,
        removed=False,
        heading=None,
        free_space=None,
        parent=None,
        children=None,
        convex_hull_2d=None,
        bottom=None,
        top=None,
        bbox=None,
        mesh=None,
        centroid_translation=None,
    ):
        self.name = name
        self.entity_config = entity_config
        self.object = scene_parser.SceneElement(
            name=name,
            heading=heading,
            convex_hull_2d=convex_hull_2d,
            bbox=bbox,
            instancetype="object",
        )
        self.heading = heading
        self.bottom = bottom
        self.top = top
        self.free_space = free_space
        self.free_space_height = [bottom, INF]
        self.critical_free_space = free_space
        self.removed = removed
        self.mesh = mesh
        self.centroid_translation = centroid_translation
        self.mesh.apply_offset(centroid_translation)

        if self.free_space is None and self.name != "GROUND":
            cos_theta, sin_theta = heading[0], heading[1]
            points = self.object.convex_hull_2d.get_headed_bbox_instance_with_heading(
                heading
            )

            def extend_point(p, direction):
                if direction == "right":
                    normal = np.array([cos_theta, sin_theta])
                elif direction == "left":
                    normal = np.array([-cos_theta, -sin_theta])
                elif direction == "front":
                    normal = np.array([-sin_theta, cos_theta])
                elif direction == "rear":
                    normal = np.array([sin_theta, -cos_theta])
                elif direction == "rear-left":
                    normal = np.array([sin_theta, -cos_theta]) + np.array(
                        [-cos_theta, -sin_theta]
                    )
                elif direction == "front-left":
                    normal = np.array([-sin_theta, cos_theta]) + np.array(
                        [-cos_theta, -sin_theta]
                    )
                elif direction == "front-right":
                    normal = np.array([-sin_theta, cos_theta]) + np.array(
                        [cos_theta, sin_theta]
                    )
                elif direction == "rear-right":
                    normal = np.array([sin_theta, -cos_theta]) + np.array(
                        [cos_theta, sin_theta]
                    )
                else:
                    return np.array(p)
                return np.array(p) + 10 * np.array(normal)

            # point older: rear_left, front_left, front_right, rear_right
            self.free_space = [
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "rear"),
                        points[0],
                        points[3],
                        extend_point(points[3], "rear"),
                    ],
                    "Critical_space": [
                        extend_point(points[0], "rear"),
                        points[0],
                        points[3],
                        extend_point(points[3], "rear"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "rear-left"),
                        extend_point(points[0], "left"),
                        points[0],
                        extend_point(points[0], "rear"),
                    ],
                    "Critical_space": [
                        extend_point(points[0], "rear-left"),
                        extend_point(points[0], "left"),
                        points[0],
                        extend_point(points[0], "rear"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "left"),
                        extend_point(points[1], "left"),
                        points[1],
                        points[0],
                    ],
                    "Critical_space": [
                        extend_point(points[0], "left"),
                        extend_point(points[1], "left"),
                        points[1],
                        points[0],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[1], "left"),
                        extend_point(points[1], "front-left"),
                        extend_point(points[1], "front"),
                        points[1],
                    ],
                    "Critical_space": [
                        extend_point(points[1], "left"),
                        extend_point(points[1], "front-left"),
                        extend_point(points[1], "front"),
                        points[1],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[1],
                        extend_point(points[1], "front"),
                        extend_point(points[2], "front"),
                        points[2],
                    ],
                    "Critical_space": [
                        points[1],
                        extend_point(points[1], "front"),
                        extend_point(points[2], "front"),
                        points[2],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[2],
                        extend_point(points[2], "front"),
                        extend_point(points[2], "front-right"),
                        extend_point(points[2], "right"),
                    ],
                    "Critical_space": [
                        points[2],
                        extend_point(points[2], "front"),
                        extend_point(points[2], "front-right"),
                        extend_point(points[2], "right"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[3],
                        points[2],
                        extend_point(points[2], "right"),
                        extend_point(points[3], "right"),
                    ],
                    "Critical_space": [
                        points[3],
                        points[2],
                        extend_point(points[2], "right"),
                        extend_point(points[3], "right"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[3], "rear"),
                        points[3],
                        extend_point(points[3], "right"),
                        extend_point(points[3], "rear-right"),
                    ],
                    "Critical_space": [
                        extend_point(points[3], "rear"),
                        points[3],
                        extend_point(points[3], "right"),
                        extend_point(points[3], "rear-right"),
                    ],
                },
            ]

        self.depth = 0
        self.parent = parent
        self.children = children if children is not None else []

        self.on_platform = None
        self.own_platform = []
        self.bel_ground_object = None
        self.bel_ground_platform = None
        self.all_children = children if children is not None else []
        self.children_per_platform = []
        self.offspring_per_platform = []

    @property
    def is_ambiguous(self):
        if self.bel_ground_platform is None:
            self.get_bel_ground_platform()
        if self.bel_ground_object is None:
            self.get_bel_ground_object()
        if self.depth <= 1:
            return False
        sibling = self.bel_ground_platform.children
        cnt = 0
        for child in sibling:
            if child.name.split("_")[0] == self.name.split("_")[0]:
                cnt += 1
        return cnt > 1

    def rename_with_name_map(self, name_map):
        self.name = StringConvertor.rename_with_map(self.name, name_map)
        self.object.name = StringConvertor.rename_with_map(self.object.name, name_map)
        if self.bel_ground_object is not None:
            self.bel_ground_object.name = StringConvertor.rename_with_map(
                self.bel_ground_object.name, name_map
            )
        if self.bel_ground_platform is not None:
            self.bel_ground_platform.name = StringConvertor.rename_with_map(
                self.bel_ground_platform.name, name_map
            )
        for child in self.children:
            child.name = StringConvertor.rename_with_map(child.name, name_map)
        for child in self.all_children:
            child.name = StringConvertor.rename_with_map(child.name, name_map)
        for platform_child in self.children_per_platform:
            for child in platform_child:
                child.name = StringConvertor.rename_with_map(child.name, name_map)
        for platform_child in self.offspring_per_platform:
            for child in platform_child:
                child.name = StringConvertor.rename_with_map(child.name, name_map)
        for platform in self.own_platform:
            platform.name = StringConvertor.rename_with_map(platform.name, name_map)

        return self

    def get_bel_ground_object(self):
        #   if self.bel_ground_object is not None or self.depth <= 1:
        #        return self.bel_ground_object

        tmp_node = self.parent
        tmp_platform = self.on_platform
        while tmp_node.depth > 1:
            tmp_platform = tmp_node.on_platform
            tmp_node = tmp_node.parent

        self.bel_ground_platform = tmp_platform
        self.bel_ground_object = tmp_node
        return self.bel_ground_object

    def get_name_for_interaction(self):
        name_for_interaction = self.name

        return f"{StringConvertor.get_noslash_name_wo_id(name_for_interaction)}"

    def get_bel_ground_platform(self):
        #       if self.bel_ground_platform is not None or self.depth <= 1:
        #         return self.bel_ground_platform

        tmp_node = self.parent
        tmp_platform = self.on_platform
        while tmp_node.depth > 1:
            tmp_platform = tmp_node.on_platform
            tmp_node = tmp_node.parent

        self.bel_ground_platform = tmp_platform
        self.bel_ground_object = tmp_node
        return self.bel_ground_platform

    def get_children_per_platform(self):
        if len(self.children_per_platform) == 0:
            self.update_children_belong_platform()
        return self.children_per_platform

    def get_object_id_on_platform(self):
        source_item_id = [
            id
            for id, _ in enumerate(self.get_bel_ground_platform().children)
            if _.name == self.name
        ]
        if len(source_item_id) == 0:
            glog.warning("get_object_id_on_platform: object not found")
            return -1
        return source_item_id[0] + 1

    def get_object_id(self):
        if self.bel_ground_object is None:
            self.get_bel_ground_object()
        return self.bel_ground_object.object_id

    def is_freespace_big_enough(self, freespace_id, min_width=0.03):
        # return xth freespace with enough width
        # freespace_dir: 0-7
        if self.free_space[freespace_id]["Critical_space"] == []:
            return False
        width = np.linalg.norm(
            self.free_space[freespace_id]["Critical_space"][0]
            - self.free_space[freespace_id]["Critical_space"][1]
        )
        height = np.linalg.norm(
            self.free_space[freespace_id]["Critical_space"][1]
            - self.free_space[freespace_id]["Critical_space"][2]
        )
        if width < min_width or height < min_width:
            return False
        return True

    def get_on_picture_freespace_id(self, dir, min_width=0.03):
        # return xth freespace with enough width
        # picture_id: 1-8
        res = 0
        for i in range(0, dir + 1):
            if self.is_freespace_big_enough(i, min_width):
                res += 1

        glog.warning("freespace dir might be illegal")
        return -1 if not self.is_freespace_big_enough(dir) else res

    def get_freespace_id_on_picture(self, freespace_dir, min_width=0.03):
        # return xth freespace with enough width
        # freespace_dir: 1-8
        for dir in range(8):
            if self.is_freespace_big_enough(dir, min_width):
                freespace_dir -= 1
                if freespace_dir == 0:
                    return dir

        glog.warning("freespace dir might be illegal")
        return -1

    def update_children_belong_platform(self) -> None:
        self.children_per_platform = []
        self.offspring_per_platform = []
        for platform_id, platform in enumerate(self.own_platform):
            child_per_platform = []
            offspring_per_platform = []
            for child in self.children:
                if child.on_platform.name == platform.name:
                    child_per_platform.append(child)
                    child_queue = []
                    child_queue.append(child)
                    while len(child_queue) > 0:
                        cur_child = child_queue.pop(0)
                        offspring_per_platform.append(cur_child)
                        child_queue.extend(cur_child.children)
            self.children_per_platform.append(child_per_platform)
            self.offspring_per_platform.append(offspring_per_platform)
            self.own_platform[platform_id].children = offspring_per_platform
        return None

    def is_multilayer_object(self):

        real_layers = 0
        for platform in self.own_platform:
            real_layers += (
                len(
                    [
                        dir
                        for dir in range(0, 8, 2)
                        if self.freespace_is_standable(dir)
                        and platform.freespace_is_visible(dir)
                    ]
                )
                > 0
            )

        return self.depth == 1 and real_layers > 1

    def get_categories_and_objects_for_mlo(self):
        major_category = set()
        minor_category = set()
        for platform in self.own_platform:
            major_category_list, minor_category_list, _ = (
                platform.get_categories_and_objects()
            )
            major_category.update(major_category_list)
            minor_category.update(minor_category_list)

        major_category = list(major_category)
        minor_category = list(minor_category)
        return major_category, minor_category

    def renew_heading(self, heading):
        if "wall_cabinet_02_19" in self.name:
            import ipdb

            ipdb.set_trace()
        self.heading = heading
        self.object.convex_hull_2d.heading = heading
        if self.name != "GROUND":
            cos_theta, sin_theta = float(heading[0]), float(heading[1])
            points = self.object.convex_hull_2d.get_headed_bbox_instance()

            def extend_point(p, direction):
                if direction == "right":
                    normal = np.array([cos_theta, sin_theta])
                elif direction == "left":
                    normal = np.array([-cos_theta, -sin_theta])
                elif direction == "front":
                    normal = np.array([-sin_theta, cos_theta])
                elif direction == "rear":
                    normal = np.array([sin_theta, -cos_theta])
                elif direction == "rear-left":
                    normal = np.array([sin_theta, -cos_theta]) + np.array(
                        [-cos_theta, -sin_theta]
                    )
                elif direction == "front-left":
                    normal = np.array([-sin_theta, cos_theta]) + np.array(
                        [-cos_theta, -sin_theta]
                    )
                elif direction == "front-right":
                    normal = np.array([-sin_theta, cos_theta]) + np.array(
                        [cos_theta, sin_theta]
                    )
                elif direction == "rear-right":
                    normal = np.array([sin_theta, -cos_theta]) + np.array(
                        [cos_theta, sin_theta]
                    )
                else:
                    return np.array(p)

                return np.array(p) + 10 * np.array(normal)

            # point older: rear_left, front_left, front_right, rear_right
            self.free_space = [
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "rear"),
                        points[0],
                        points[3],
                        extend_point(points[3], "rear"),
                    ],
                    "Critical_space": [
                        extend_point(points[0], "rear"),
                        points[0],
                        points[3],
                        extend_point(points[3], "rear"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "rear-left"),
                        extend_point(points[0], "left"),
                        points[0],
                        extend_point(points[0], "rear"),
                    ],
                    "Critical_space": [
                        extend_point(points[0], "rear-left"),
                        extend_point(points[0], "left"),
                        points[0],
                        extend_point(points[0], "rear"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "left"),
                        extend_point(points[1], "left"),
                        points[1],
                        points[0],
                    ],
                    "Critical_space": [
                        extend_point(points[0], "left"),
                        extend_point(points[1], "left"),
                        points[1],
                        points[0],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[1], "left"),
                        extend_point(points[1], "front-left"),
                        extend_point(points[1], "front"),
                        points[1],
                    ],
                    "Critical_space": [
                        extend_point(points[1], "left"),
                        extend_point(points[1], "front-left"),
                        extend_point(points[1], "front"),
                        points[1],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[1],
                        extend_point(points[1], "front"),
                        extend_point(points[2], "front"),
                        points[2],
                    ],
                    "Critical_space": [
                        points[1],
                        extend_point(points[1], "front"),
                        extend_point(points[2], "front"),
                        points[2],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[2],
                        extend_point(points[2], "front"),
                        extend_point(points[2], "front-right"),
                        extend_point(points[2], "right"),
                    ],
                    "Critical_space": [
                        points[2],
                        extend_point(points[2], "front"),
                        extend_point(points[2], "front-right"),
                        extend_point(points[2], "right"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[3],
                        points[2],
                        extend_point(points[2], "right"),
                        extend_point(points[3], "right"),
                    ],
                    "Critical_space": [
                        points[3],
                        points[2],
                        extend_point(points[2], "right"),
                        extend_point(points[3], "right"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[3], "rear"),
                        points[3],
                        extend_point(points[3], "right"),
                        extend_point(points[3], "rear-right"),
                    ],
                    "Critical_space": [
                        extend_point(points[3], "rear"),
                        points[3],
                        extend_point(points[3], "right"),
                        extend_point(points[3], "rear-right"),
                    ],
                },
            ]

    def reset_free_space(self):
        cos_theta, sin_theta = self.heading[0], self.heading[1]
        points = self.object.convex_hull_2d.get_headed_bbox_instance()

        def extend_point(p, direction):
            if direction == "right":
                normal = np.array([cos_theta, sin_theta])
            elif direction == "left":
                normal = np.array([-cos_theta, -sin_theta])
            elif direction == "front":
                normal = np.array([-sin_theta, cos_theta])
            elif direction == "rear":
                normal = np.array([sin_theta, -cos_theta])
            elif direction == "rear-left":
                normal = np.array([sin_theta, -cos_theta]) + np.array(
                    [-cos_theta, -sin_theta]
                )
            elif direction == "front-left":
                normal = np.array([-sin_theta, cos_theta]) + np.array(
                    [-cos_theta, -sin_theta]
                )
            elif direction == "front-right":
                normal = np.array([-sin_theta, cos_theta]) + np.array(
                    [cos_theta, sin_theta]
                )
            elif direction == "rear-right":
                normal = np.array([sin_theta, -cos_theta]) + np.array(
                    [cos_theta, sin_theta]
                )
            else:
                return np.array(p)
            return np.array(p) + 10 * np.array(normal)

        # point older: rear_left, front_left, front_right, rear_right
        self.free_space = [
            {
                "Objects": [],
                "Available_space": [
                    extend_point(points[0], "rear"),
                    points[0],
                    points[3],
                    extend_point(points[3], "rear"),
                ],
                "Critical_space": [
                    extend_point(points[0], "rear"),
                    points[0],
                    points[3],
                    extend_point(points[3], "rear"),
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    extend_point(points[0], "rear-left"),
                    extend_point(points[0], "left"),
                    points[0],
                    extend_point(points[0], "rear"),
                ],
                "Critical_space": [
                    extend_point(points[0], "rear-left"),
                    extend_point(points[0], "left"),
                    points[0],
                    extend_point(points[0], "rear"),
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    extend_point(points[0], "left"),
                    extend_point(points[1], "left"),
                    points[1],
                    points[0],
                ],
                "Critical_space": [
                    extend_point(points[0], "left"),
                    extend_point(points[1], "left"),
                    points[1],
                    points[0],
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    extend_point(points[1], "left"),
                    extend_point(points[1], "front-left"),
                    extend_point(points[1], "front"),
                    points[1],
                ],
                "Critical_space": [
                    extend_point(points[1], "left"),
                    extend_point(points[1], "front-left"),
                    extend_point(points[1], "front"),
                    points[1],
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    points[1],
                    extend_point(points[1], "front"),
                    extend_point(points[2], "front"),
                    points[2],
                ],
                "Critical_space": [
                    points[1],
                    extend_point(points[1], "front"),
                    extend_point(points[2], "front"),
                    points[2],
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    points[2],
                    extend_point(points[2], "front"),
                    extend_point(points[2], "front-right"),
                    extend_point(points[2], "right"),
                ],
                "Critical_space": [
                    points[2],
                    extend_point(points[2], "front"),
                    extend_point(points[2], "front-right"),
                    extend_point(points[2], "right"),
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    points[3],
                    points[2],
                    extend_point(points[2], "right"),
                    extend_point(points[3], "right"),
                ],
                "Critical_space": [
                    points[3],
                    points[2],
                    extend_point(points[2], "right"),
                    extend_point(points[3], "right"),
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    extend_point(points[3], "rear"),
                    points[3],
                    extend_point(points[3], "right"),
                    extend_point(points[3], "rear-right"),
                ],
                "Critical_space": [
                    extend_point(points[3], "rear"),
                    points[3],
                    extend_point(points[3], "right"),
                    extend_point(points[3], "rear-right"),
                ],
            },
        ]

    def update_free_space(self, other_node_name, direction):
        assert (
            len(self.free_space[direction]["Available_space"]) == 4
            and direction >= 0
            and direction < 8
        )
        # print(self.name, direction, other_node.name, self.free_space[EIGHT_DIRECTIONS.index(direction)]["Available_space"])
        self.free_space[direction]["Objects"].append(other_node_name)

    def get_critical_space_len(self, direction):
        return [
            np.linalg.norm(
                self.free_space[direction]["Critical_space"][0]
                - self.free_space[direction]["Critical_space"][1]
            ),
            np.linalg.norm(
                self.free_space[direction]["Critical_space"][1]
                - self.free_space[direction]["Critical_space"][2]
            ),
        ]

    def get_multiple_standing_point(
        self, platform_rect, standing_direction, max_visclear_width=0.8
    ):
        bbox_line = self.get_bbox_line(standing_direction)
        perpendicular_line = (
            self.get_bbox_line((standing_direction + 2) % 8)[1]
            - self.get_bbox_line((standing_direction + 2) % 8)[0]
        )
        perpendicular_line_normalized = (
            perpendicular_line
            / np.linalg.norm(perpendicular_line)
            * self.robot_fit_threshold
        )
        length = int(
            np.ceil(np.linalg.norm(bbox_line[1] - bbox_line[0]) / max_visclear_width)
        )
        return [
            bbox_line[1]
            + (i * 2 + 1) / (length * 2) * (bbox_line[0] - bbox_line[1])
            + perpendicular_line_normalized
            for i in range(length)
        ]

    def get_standing_point(self, direction):
        if direction == 0:
            return (
                self.free_space[0]["Critical_space"][1]
                + self.free_space[0]["Critical_space"][2]
            ) / 2 + (
                self.free_space[0]["Available_space"][0]
                - self.free_space[0]["Available_space"][1]
            ) * TreeNode.robot_fit_threshold / np.linalg.norm(
                self.free_space[0]["Available_space"][0]
                - self.free_space[0]["Available_space"][1]
            )
        elif direction == 2:
            return (
                self.free_space[2]["Critical_space"][2]
                + self.free_space[2]["Critical_space"][3]
            ) / 2 + (
                self.free_space[2]["Available_space"][1]
                - self.free_space[2]["Available_space"][2]
            ) * TreeNode.robot_fit_threshold / np.linalg.norm(
                self.free_space[2]["Available_space"][1]
                - self.free_space[2]["Available_space"][2]
            )
        elif direction == 4:
            return (
                self.free_space[4]["Critical_space"][3]
                + self.free_space[4]["Critical_space"][0]
            ) / 2 + (
                self.free_space[4]["Available_space"][2]
                - self.free_space[4]["Available_space"][3]
            ) * TreeNode.robot_fit_threshold / np.linalg.norm(
                self.free_space[4]["Available_space"][2]
                - self.free_space[4]["Available_space"][3]
            )
        elif direction == 6:
            return (
                self.free_space[6]["Critical_space"][0]
                + self.free_space[6]["Critical_space"][1]
            ) / 2 + (
                self.free_space[6]["Available_space"][3]
                - self.free_space[6]["Available_space"][0]
            ) * TreeNode.robot_fit_threshold / np.linalg.norm(
                self.free_space[6]["Available_space"][3]
                - self.free_space[6]["Available_space"][0]
            )
        else:
            return None

    def get_center(self):
        return (
            self.free_space[0]["Critical_space"][1]
            + self.free_space[4]["Critical_space"][3]
        ) / 2

    def get_bbox(self):
        return [
            self.free_space[0]["Critical_space"][1],
            self.free_space[2]["Critical_space"][2],
            self.free_space[4]["Critical_space"][3],
            self.free_space[6]["Critical_space"][0],
        ]

    def get_bbox_line(self, dir):
        if dir % 2 != 0 or dir < 0 or dir > 7:
            return None

        bbox = self.get_bbox()
        return [bbox[dir // 2], bbox[(dir // 2 + 3) % 4]]

    def get_bbox_line_length(self, dir):
        if dir % 2 != 0 or dir < 0 or dir > 7:
            return None

        bbox = self.get_bbox()
        return np.linalg.norm(bbox[dir // 2] - bbox[(dir // 2 + 3) % 4])

    def get_critical_space_center(self, direction):
        assert (
            len(self.free_space[direction]["Critical_space"]) == 4
            and direction >= 0
            and direction < 8
        )
        return np.mean(self.free_space[direction]["Critical_space"], axis=0)

    def get_critical_space_left(self, direction):
        assert (
            len(self.free_space[direction]["Critical_space"]) == 4
            and direction >= 0
            and direction < 8
        )
        return (
            np.mean(self.free_space[direction]["Critical_space"][:2], axis=0)
            + np.mean(self.free_space[direction]["Critical_space"], axis=0)
        ) / 2

    def get_critical_space_right(self, direction):
        assert (
            len(self.free_space[direction]["Critical_space"]) == 4
            and direction >= 0
            and direction < 8
        )
        return (
            np.mean(self.free_space[direction]["Critical_space"][2:], axis=0)
            + np.mean(self.free_space[direction]["Critical_space"], axis=0)
        ) / 2

    def get_num_of_critical_space(self):
        cnt = 0
        for dir in range(8):
            if self.is_freespace_big_enough(dir):
                cnt += 1
        return cnt

    def freespace_is_visible(self, standing_direction, platform_id):
        assert (
            standing_direction >= 0
            and standing_direction < 8
            and standing_direction % 2 == 0
        )
        dir = EIGHT_DIRECTIONS[standing_direction]
        plat = self.own_platform[platform_id]
        if dir not in plat.visible_directions:
            return False
        return plat.visible_directions[dir]

    def freespace_is_standable(self, direction):
        if self.free_space is None:
            return False
        assert (
            len(self.free_space[direction]["Critical_space"]) == 4
            and direction >= 0
            and direction < 8
        )

        if direction == 0 or direction == 4:
            return (
                np.linalg.norm(
                    self.free_space[direction]["Critical_space"][0]
                    - self.free_space[direction]["Critical_space"][1]
                )
                > TreeNode.robot_fit_threshold
            )
        elif direction == 2 or direction == 6:
            return (
                np.linalg.norm(
                    self.free_space[direction]["Critical_space"][1]
                    - self.free_space[direction]["Critical_space"][2]
                )
                > TreeNode.robot_fit_threshold
            )
        else:
            return 0

    def get_critical_space_bbox(self):
        return [
            self.free_space[1]["Critical_space"][0],
            self.free_space[3]["Critical_space"][1],
            self.free_space[5]["Critical_space"][2],
            self.free_space[7]["Critical_space"][3],
        ]

    def sweep_platform(self):
        max_size = -INF
        for platform in self.own_platform:
            max_size = max(
                max_size,
                (platform.bbox[1][0] - platform.bbox[0][0])
                * (platform.bbox[1][1] - platform.bbox[0][1]),
            )
        self.own_platform = [
            platform
            for platform in self.own_platform
            if (platform.bbox[1][0] - platform.bbox[0][0])
            * (platform.bbox[1][1] - platform.bbox[0][1])
            > max_size * 0.25
        ]

    def at_which_part(self):
        platform_bbox = self.on_platform.convex_hull_2d.get_headed_bbox_instance()
        vertices = self.object.convex_hull_2d.vertices
        item_bbox = self.object.convex_hull_2d.get_headed_bbox_instance()
        vertices = np.append(vertices, item_bbox, axis=0)
        direction_mappings = {
            (0, 0): "rear-left",
            (0, 1): "rear",
            (0, 2): "rear-right",
            (1, 0): "left",
            (1, 1): "center",
            (1, 2): "right",
            (2, 0): "front-left",
            (2, 1): "front",
            (2, 2): "front-right",
        }
        result_direction_list = []
        for i in range(3):
            for j in range(3):
                rect = [
                    platform_bbox[0]
                    + i * (platform_bbox[1] - platform_bbox[0]) / 3
                    + j * (platform_bbox[3] - platform_bbox[0]) / 3,
                    platform_bbox[0]
                    + (i + 1) * (platform_bbox[1] - platform_bbox[0]) / 3
                    + j * (platform_bbox[3] - platform_bbox[0]) / 3,
                    platform_bbox[0]
                    + (i + 1) * (platform_bbox[1] - platform_bbox[0]) / 3
                    + (j + 1) * (platform_bbox[3] - platform_bbox[0]) / 3,
                    platform_bbox[0]
                    + i * (platform_bbox[1] - platform_bbox[0]) / 3
                    + (j + 1) * (platform_bbox[3] - platform_bbox[0]) / 3,
                ]
                for vertex in vertices:
                    if Basic2DGeometry.is_inside_rectangle(vertex, rect):
                        result_direction_list.append(direction_mappings[(i, j)])
                        break

        return result_direction_list

    def get_fitable_pose(self, avl_height, place_bbox):
        if self.top - self.bottom > avl_height:
            return None
        return self.object.convex_hull_2d.get_fit_in_translation(place_bbox)

    def get_fitable_pose_for_list(self, avl_height, rectangle_list, heading=None):
        if self.top - self.bottom > avl_height:
            return None
        if heading is not None:
            self.object.convex_hull_2d.heading = heading
        target_rect = self.object.convex_hull_2d.get_headed_bbox_instance()
        placement = Basic2DGeometry.find_optimal_placement_with_rotation(
            self.object.convex_hull_2d, rectangle_list
        )
        pass

    def rotate_free_space(self, front):
        if front == 0:
            return
        elif front == 4:
            self.heading = [-self.heading[0], -self.heading[1]]
            self.free_space = self.free_space[4:] + self.free_space[:4]
            for platform in self.own_platform:
                platform.heading = self.heading
                platform.visible_directions = (
                    platform.visible_directions[2:] + platform.visible_directions[:2]
                )

            for i in range(8):
                self.free_space[i]["Available_space"] = (
                    self.free_space[i]["Available_space"][2:]
                    + self.free_space[i]["Available_space"][:2]
                )
                self.free_space[i]["Critical_space"] = (
                    self.free_space[i]["Critical_space"][2:]
                    + self.free_space[i]["Critical_space"][:2]
                )
        elif front == 2:
            self.heading = [self.heading[1], -self.heading[0]]
            self.free_space = self.free_space[2:] + self.free_space[:2]

            for platform in self.own_platform:
                platform.heading = self.heading
                platform.visible_directions = (
                    platform.visible_directions[1:] + platform.visible_directions[:1]
                )

            for i in range(8):
                self.free_space[i]["Available_space"] = (
                    self.free_space[i]["Available_space"][1:]
                    + self.free_space[i]["Available_space"][:1]
                )
                self.free_space[i]["Critical_space"] = (
                    self.free_space[i]["Critical_space"][1:]
                    + self.free_space[i]["Critical_space"][:1]
                )
        elif front == 6:
            self.heading = [-self.heading[1], self.heading[0]]
            self.free_space = self.free_space[6:] + self.free_space[:6]

            for platform in self.own_platform:
                platform.heading = self.heading
                platform.visible_directions = (
                    platform.visible_directions[3:] + platform.visible_directions[:3]
                )
            for i in range(8):
                self.free_space[i]["Available_space"] = (
                    self.free_space[i]["Available_space"][3:]
                    + self.free_space[i]["Available_space"][:3]
                )
                self.free_space[i]["Critical_space"] = (
                    self.free_space[i]["Critical_space"][3:]
                    + self.free_space[i]["Critical_space"][:3]
                )

    def rotate_free_space_for_all_children(self, front):
        self.rotate_free_space(front)
        for child in self.children:
            child.rotate_free_space_for_all_children(front)

    def set_free_space_to_platform(self, platform_hull):
        platform_hull.heading = self.heading
        platform_bbox = platform_hull.get_headed_bbox_instance()

        platform_left_side = [platform_bbox[0], platform_bbox[1]]
        platform_right_side = [platform_bbox[2], platform_bbox[3]]
        platform_front_side = [platform_bbox[1], platform_bbox[2]]
        platform_rear_side = [platform_bbox[3], platform_bbox[0]]
        for left_part in range(1, 4):

            space_front_side = [
                self.free_space[left_part]["Available_space"][1],
                self.free_space[left_part]["Available_space"][2],
            ]
            space_rear_side = [
                self.free_space[left_part]["Available_space"][0],
                self.free_space[left_part]["Available_space"][3],
            ]

            intersect_left_front = Basic2DGeometry.intersection_of_line(
                platform_left_side, space_front_side
            )
            intersect_left_rear = Basic2DGeometry.intersection_of_line(
                platform_left_side, space_rear_side
            )
            #    if self.name == 'frl_apartment_choppingboard_02_81':
            #        print(intersect_left_front, platform_left_side, space_front_side)
            if intersect_left_front is not None and intersect_left_rear is not None:
                if Basic2DGeometry.is_on_segment(
                    intersect_left_front, space_front_side
                ) and Basic2DGeometry.is_on_segment(
                    intersect_left_rear, space_rear_side
                ):
                    (
                        self.free_space[left_part]["Available_space"][0],
                        self.free_space[left_part]["Available_space"][1],
                    ) = (intersect_left_rear, intersect_left_front)
                elif Basic2DGeometry.is_on_segment(
                    space_front_side[1], [space_front_side[0], intersect_left_front]
                ):
                    (
                        self.free_space[left_part]["Available_space"][0],
                        self.free_space[left_part]["Available_space"][1],
                    ) = (
                        self.free_space[left_part]["Available_space"][3],
                        self.free_space[left_part]["Available_space"][2],
                    )

        for right_part in range(5, 8):

            space_front_side = [
                self.free_space[right_part]["Available_space"][1],
                self.free_space[right_part]["Available_space"][2],
            ]
            space_rear_side = [
                self.free_space[right_part]["Available_space"][0],
                self.free_space[right_part]["Available_space"][3],
            ]
            intersect_right_front = Basic2DGeometry.intersection_of_line(
                platform_right_side, space_front_side
            )
            intersect_right_rear = Basic2DGeometry.intersection_of_line(
                platform_right_side, space_rear_side
            )
            if intersect_left_front is not None and intersect_left_rear is not None:
                if Basic2DGeometry.is_on_segment(
                    intersect_right_front, space_front_side
                ) and Basic2DGeometry.is_on_segment(
                    intersect_right_rear, space_rear_side
                ):
                    (
                        self.free_space[right_part]["Available_space"][3],
                        self.free_space[right_part]["Available_space"][2],
                    ) = (intersect_right_rear, intersect_right_front)
                elif Basic2DGeometry.is_on_segment(
                    space_front_side[0], [space_front_side[1], intersect_right_front]
                ):
                    (
                        self.free_space[right_part]["Available_space"][2],
                        self.free_space[right_part]["Available_space"][3],
                    ) = (
                        self.free_space[right_part]["Available_space"][1],
                        self.free_space[right_part]["Available_space"][0],
                    )

        for front_part in range(3, 6):

            space_left_side = [
                self.free_space[front_part]["Available_space"][0],
                self.free_space[front_part]["Available_space"][1],
            ]
            space_right_side = [
                self.free_space[front_part]["Available_space"][3],
                self.free_space[front_part]["Available_space"][2],
            ]
            intersect_front_left = Basic2DGeometry.intersection_of_line(
                platform_front_side, space_left_side
            )
            intersect_front_right = Basic2DGeometry.intersection_of_line(
                platform_front_side, space_right_side
            )
            if intersect_front_left is not None and intersect_front_right is not None:
                if Basic2DGeometry.is_on_segment(
                    intersect_front_left, space_left_side
                ) and Basic2DGeometry.is_on_segment(
                    intersect_front_right, space_right_side
                ):
                    (
                        self.free_space[front_part]["Available_space"][1],
                        self.free_space[front_part]["Available_space"][2],
                    ) = (intersect_front_left, intersect_front_right)
                elif Basic2DGeometry.is_on_segment(
                    space_left_side[0], [space_left_side[1], intersect_front_left]
                ):
                    (
                        self.free_space[front_part]["Available_space"][1],
                        self.free_space[front_part]["Available_space"][2],
                    ) = (
                        self.free_space[front_part]["Available_space"][0],
                        self.free_space[front_part]["Available_space"][3],
                    )

        for rear_part in [7, 0, 1]:

            space_left_side = [
                self.free_space[rear_part]["Available_space"][0],
                self.free_space[rear_part]["Available_space"][1],
            ]
            space_right_side = [
                self.free_space[rear_part]["Available_space"][3],
                self.free_space[rear_part]["Available_space"][2],
            ]
            intersect_rear_left = Basic2DGeometry.intersection_of_line(
                platform_rear_side, space_left_side
            )
            intersect_rear_right = Basic2DGeometry.intersection_of_line(
                platform_rear_side, space_right_side
            )
            if intersect_rear_left is not None and intersect_rear_right is not None:
                if Basic2DGeometry.is_on_segment(
                    intersect_rear_left, space_left_side
                ) and Basic2DGeometry.is_on_segment(
                    intersect_rear_right, space_right_side
                ):
                    (
                        self.free_space[rear_part]["Available_space"][0],
                        self.free_space[rear_part]["Available_space"][3],
                    ) = (intersect_rear_left, intersect_rear_right)
                elif Basic2DGeometry.is_on_segment(
                    space_left_side[1], [space_left_side[0], intersect_rear_left]
                ):
                    (
                        self.free_space[rear_part]["Available_space"][0],
                        self.free_space[rear_part]["Available_space"][3],
                    ) = (
                        self.free_space[rear_part]["Available_space"][1],
                        self.free_space[rear_part]["Available_space"][2],
                    )

        pass

    def sync_critical_free_space(self):
        for dir in range(8):
            self.free_space[dir]["Critical_space"] = copy.deepcopy(
                self.free_space[dir]["Available_space"]
            )

    def clean_free_space(self):
        if self.free_space == None:
            return

        def rotate_point_clockwise(point, theta):
            rotation_matrix = np.array(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )
            return np.dot(rotation_matrix, point)

        def has_area(points, heading):
            left_rear, left_front, right_front, right_rear = points
            cos_theta, sin_theta = heading[0], heading[1]

            # calculate the angle between the x-axis and the heading
            theta = np.arctan2(sin_theta, cos_theta)

            left_rear_rotated = rotate_point_clockwise(left_rear, theta)
            left_front_rotated = rotate_point_clockwise(left_front, theta)
            right_rear_rotated = rotate_point_clockwise(right_rear, theta)
            right_front_rotated = rotate_point_clockwise(right_front, theta)

            if (
                right_rear_rotated[0] < left_rear_rotated[0]
                or right_front_rotated[0] < left_front_rotated[0]
            ):
                return False
            if (
                right_rear_rotated[1] > left_rear_rotated[1]
                or right_front_rotated[1] < left_front_rotated[1]
            ):
                return False

            return True

        for direction in range(len(self.free_space)):
            if self.free_space[direction]["Available_space"] != "not available":
                if not has_area(
                    self.free_space[direction]["Available_space"], self.heading
                ):
                    self.free_space[direction]["Objects"] = []
                    self.free_space[direction]["Available_space"] = "not available"
            if self.free_space[direction]["Critical_space"] != "not available":
                if not has_area(
                    self.free_space[direction]["Critical_space"], self.heading
                ):
                    self.free_space[direction]["Objects"] = []
                    self.free_space[direction]["Critical_space"] = "not available"

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def get_child(self, child_name):
        for child in self.children:
            if child.name == child_name:
                return child

    def update_all_child(self):
        self.all_children = []

        que = [self]
        while que:
            current_node = que.pop(0)
            self.all_children.append(current_node)
            que.extend(current_node.children)

        self.all_children = self.all_children[1:]  # Exclude the root node itself

    def get_available_place_for_object(self, object_node):
        self.update_children_belong_platform()
        result = []
        for i, platform_children in enumerate(self.own_platform):
            platform_result = []
            obstacle_obj_list = []
            for child in self.children_per_platform[i]:
                if child.name != object_node.name:
                    obstacle_obj_list.append(child)
            platform_result = (
                platform_children.get_all_available_freespace_combinations(
                    obstacle_obj_list, object_node.object, min_step=0.01
                )
            )
            result.append(platform_result)

        return result

    def judge_critical_direction(self, node_name, mild=True):
        dir_list = []
        for dir in range(8):
            for id, dir_node in enumerate(self.free_space[dir]["Objects"]):
                if dir_node.name == node_name:
                    if mild:
                        dir_list.append(dir)
                        break
                    tmp_freespace = self.free_space[dir]["Available_space"]
                    all_siblings = self.free_space[dir]["Objects"]
                    all_siblings[id], all_siblings[-1] = (
                        all_siblings[-1],
                        all_siblings[id],
                    )

                    if dir in [1, 2, 3]:
                        for id, object_node in enumerate(all_siblings):
                            near_side = [
                                tmp_freespace[3],
                                tmp_freespace[2],
                            ]
                            far_side = [
                                tmp_freespace[0],
                                tmp_freespace[1],
                            ]
                            new_near_side, new_far_side = (
                                object_node.object.convex_hull_2d.cut_free_space_with_convex(
                                    near_side, far_side
                                )
                            )
                            if id == len(self.free_space[dir]["Objects"]) - 1:
                                if np.allclose(new_far_side, far_side) == False:
                                    dir_list.append(dir)

                            tmp_freespace = [
                                new_far_side[0],
                                new_far_side[1],
                                new_near_side[1],
                                new_near_side[0],
                            ]
                    if dir in [3, 4, 5]:
                        for id, object_node in enumerate(all_siblings):
                            near_side = [
                                tmp_freespace[0],
                                tmp_freespace[3],
                            ]
                            far_side = [
                                tmp_freespace[1],
                                tmp_freespace[2],
                            ]
                            new_near_side, new_far_side = (
                                object_node.object.convex_hull_2d.cut_free_space_with_convex(
                                    near_side, far_side
                                )
                            )
                            if id == len(self.free_space[dir]["Objects"]) - 1:
                                if np.allclose(new_far_side, far_side) == False:
                                    dir_list.append(dir)

                            tmp_freespace = [
                                new_near_side[0],
                                new_far_side[0],
                                new_far_side[1],
                                new_near_side[1],
                            ]
                    if dir in [5, 6, 7]:
                        for id, object_node in enumerate(all_siblings):
                            near_side = [
                                tmp_freespace[0],
                                tmp_freespace[1],
                            ]
                            far_side = [
                                tmp_freespace[3],
                                tmp_freespace[2],
                            ]
                            new_near_side, new_far_side = (
                                object_node.object.convex_hull_2d.cut_free_space_with_convex(
                                    near_side, far_side
                                )
                            )
                            if id == len(self.free_space[dir]["Objects"]) - 1:
                                if np.allclose(new_far_side, far_side) == False:
                                    dir_list.append(dir)

                            tmp_freespace = [
                                new_near_side[0],
                                new_near_side[1],
                                new_far_side[1],
                                new_far_side[0],
                            ]
                    if dir in [7, 0, 1]:
                        for id, object_node in enumerate(all_siblings):
                            near_side = [
                                tmp_freespace[1],
                                tmp_freespace[2],
                            ]
                            far_side = [
                                tmp_freespace[0],
                                tmp_freespace[3],
                            ]
                            new_near_side, new_far_side = (
                                object_node.object.convex_hull_2d.cut_free_space_with_convex(
                                    near_side, far_side
                                )
                            )
                            if id == len(self.free_space[dir]["Objects"]) - 1:
                                if np.allclose(new_far_side, far_side) == False:
                                    dir_list.append(dir)

                            tmp_freespace = [
                                new_far_side[0],
                                new_near_side[0],
                                new_near_side[1],
                                new_far_side[1],
                            ]

                    break
        dir_list = list(set(dir_list))
        return dir_list

    def remove_child(self, child_node):
        for i in range(len(self.children)):
            if self.children[i].name == child_node.name:
                self.children[i].parent = None
                self.children.pop(i)
                break

    def display_unique_information(self, ambiguous=False):
        if self.depth == 0:
            return ""
        else:
            str_buffer = ""
            str_buffer += f"object: {self.name}\n"
            str_buffer += f"belonged_to_platform: {self.on_platform.name if self.on_platform is not None and self.depth > 1 else None}\n"
            if self.is_ambiguous:
                str_buffer += f"Its surroundings\n"
                for dir in range(8):
                    if (
                        self.free_space[dir]["Objects"] is not None
                        and len(self.free_space[dir]["Objects"]) > 0
                    ):
                        str_buffer += f'{EIGHT_DIRECTIONS[dir]}, {[objs.name for objs in self.free_space[dir]["Objects"]]}\n'
            if self.depth > 1:
                str_buffer += f"Its direct parent:\n"
            str_buffer += self.parent.display_unique_information(ambiguous) + "\n"

        return str_buffer

    def auto_take_non_ground_object_picture(
        self,
        scene,
        view="human_full",  # 'human_focus', 'human_full', 'top_focus', 'top_full'
        mark_object=False,  # if True, mark all the object on the same platform with cuboid.
        only_mark_itself=False,  # if True, only mark itself
        mark_freespace=False,
        diagonal_mode="old",  # 'old', 'new_largest_rect', 'new_all', 'new_combined_freespace'
        need_afford_rect=None,  # If not none, only mark the freespaces with size larger than it.
        standing_direction=0,
        width=640,
        height=480,
        focus_ratio=0.8,
        fovy_range=[np.deg2rad(5), np.deg2rad(60)],
        save_path=None,
    ):
        if self.depth == 1:
            glog.error("This function is only for non-ground object")
            return None
        self.update_children_belong_platform()
        bel_ground_node = self.get_bel_ground_object()
        bel_ground_node.update_children_belong_platform()
        bel_ground_platform = self.get_bel_ground_platform()
        if bel_ground_node is None:
            glog.error("This non-ground object has no ground node")
            return None

        bel_ground_platform_id = int(bel_ground_platform.name.split("_")[-1])

        # standing_direction = 0

        self_center = self.get_center()
        border = bel_ground_platform.get_bbox_line(standing_direction)

        perp_intersection = Basic2DGeometry.get_perpendicular_intersection(
            self_center, border
        )

        camera_xy = (
            perp_intersection
            + (perp_intersection - self_center)
            / np.linalg.norm(perp_intersection - self_center)
            * 0.35
            if "top" not in view
            else self.get_center()
        )
        # import ipdb
        # ipdb.set_trace()
        if "top" in view:
            z_range = [
                bel_ground_platform.bottom + 0.1,
                min(3.08, bel_ground_platform.bottom + bel_ground_platform.avl_height),
            ]
        else:
            z_range = [
                self.top + 0.1,
                min(
                    self.top + 0.3,
                    bel_ground_platform.bottom + bel_ground_platform.avl_height,
                ),
            ]
        object_2d_bbox = self.object.convex_hull_2d.get_headed_bbox_instance()
        object_2d_bbox_extended = [
            Basic2DGeometry.get_perpendicular_intersection(point, border)
            for point in object_2d_bbox
        ]

        object_2d_bbox[standing_direction // 2] = object_2d_bbox_extended[
            standing_direction // 2
        ]
        object_2d_bbox[(standing_direction // 2 + 3) % 4] = object_2d_bbox_extended[
            (standing_direction // 2 + 3) % 4
        ]

        object_3d_bbox = [np.append(point, self.bottom) for point in object_2d_bbox] + [
            np.append(point, self.top) for point in object_2d_bbox
        ]
        platform_2d_rect = bel_ground_platform.bbox
        platform_3d_rect = [
            np.append(point, bel_ground_platform.bottom) for point in platform_2d_rect
        ]
        might_mark_cuboid_list = []
        might_mark_freespace_rect_list = []
        might_mark_freespace_id_list = []

        if mark_object:
            bel_ground_platform_idx = int(bel_ground_platform.name.split("_")[-1])
            glog.info(f"bel_ground_platform_idx: {bel_ground_platform_idx}")
            glog.info(
                f"bel_ground_node_per_platform: {(bel_ground_node.children_per_platform)}"
            )
            for child in bel_ground_platform.children:
                child_2d_bbox = child.object.convex_hull_2d.get_headed_bbox_instance()
                child_3d_bbox = [
                    np.append(point, child.bottom) for point in child_2d_bbox
                ] + [np.append(point, child.top) for point in child_2d_bbox]
                might_mark_cuboid_list.append(child_3d_bbox)

        elif only_mark_itself:
            child_2d_bbox = self.object.convex_hull_2d.get_headed_bbox_instance()
            child_3d_bbox = [
                np.append(point, self.bottom) for point in child_2d_bbox
            ] + [np.append(point, self.top) for point in child_2d_bbox]
            might_mark_cuboid_list.append(child_3d_bbox)

        if mark_freespace:
            if diagonal_mode == "old":
                for dir in range(0, 8, 1):
                    if not self.is_freespace_big_enough(dir):
                        continue
                    might_mark_freespace_rect_list.append(
                        self.free_space[dir]["Critical_space"]
                    )
                    might_mark_freespace_id_list.append(dir)

        for i, rect in enumerate(might_mark_freespace_rect_list):
            might_mark_freespace_rect_list[i] = [
                np.append(point, self.bottom) for point in rect
            ]

        # if 'human' in view:
        #     import ipdb
        #     ipdb.set_trace()

        optimal_pose, optimal_fovy = (
            image_render_processor.auto_get_optimal_camera_pose_for_object(
                view=view,
                camera_xy=camera_xy,
                z_range=z_range,
                object_bbox=object_3d_bbox,
                platform_rect=platform_3d_rect,
                width=width,
                height=height,
                fovy_range=fovy_range,
                focus_ratio=focus_ratio,
            )
        )
        rectangle_grey = not mark_freespace

        import os

        current_path = os.path.dirname(os.path.abspath(__file__))
        glog.info(
            [
                child.name
                for child in self.bel_ground_platform.children
                if child.name != self.name
            ]
        )
        img = image_render_processor.auto_render_image_refactored(
            scene,
            name=self.name,
            transparent_item_list=[
                child.name
                for child in self.bel_ground_platform.children
                if child.name != self.name
            ],
            pose=optimal_pose,
            fovy=optimal_fovy,
            width=width,
            height=height,
            might_mark_object_cuboid_list=might_mark_cuboid_list,
            might_mark_freespace_list=might_mark_freespace_rect_list,
            rectangle_grey=rectangle_grey,
            save_path=save_path,
        )

        return img

    def auto_take_ground_object_picture(
        self,
        scene,
        view="human_full",  # 'human_full', 'overall_top',
        mark_object=False,
        mark_freespace=False,
        need_afford_rect=None,  # If not none, only mark the freespaces with size larger than it.
        platform_id=0,
        camera_xy=None,
        standing_direction=0,
        width=1920,
        height=1080,
        focus_ratio=0.5,
        save_path=None,
    ):
        if self.depth != 1:
            glog.warning("This function is only for ground object")
            return None

        self.update_children_belong_platform()

        platform = self.own_platform[platform_id]

        if len(platform.standing_point_list) == 0:
            glog.warning(
                f"This ground object {self.name} has no standing_points in direction {standing_direction}"
            )
            return None

        camera_xy = camera_xy if "top" not in view else self.get_center()
        camera_xy_list = []

        if not any(len(point) > 0 for point in platform.standing_point_list):
            glog.warning(
                f"This ground object {self.name} has no standing_points in direction {standing_direction}"
            )
            return None
        standing_direction //= 2
        if camera_xy is None:
            standing_direction = (
                standing_direction
                if standing_direction < len(platform.standing_point_list)
                else 0
            )
            while len(platform.standing_point_list[standing_direction]) == 0:
                standing_direction = (standing_direction + 1) % len(
                    platform.standing_point_list
                )
            glog.info(
                f"Using standing direction {standing_direction} for platform {platform_id} with name {platform.name}"
            )
            camera_xy = (
                platform.standing_point_list[standing_direction][0]
                + platform.standing_point_list[standing_direction][-1]
            ) / 2
            camera_xy_list = platform.standing_point_list[standing_direction]

        platform_children = platform.children
        platform_children_top_max = (
            max([child.top for child in platform_children])
            if len(platform_children) > 0
            else platform.bottom + 0.35
        )
        z_range = [platform_children_top_max + 0.1, platform_children_top_max + 0.3]

        platform_cuboid_bbox = platform.convex_hull_2d.get_headed_bbox_instance()
        platform_cuboid_bbox_bottom = [
            np.append(point, platform.bottom) for point in platform_cuboid_bbox
        ]
        platform_cuboid_bbox_top = [
            np.append(point, platform.bottom + platform.avl_height)
            for point in platform_cuboid_bbox
        ]
        platform_cuboid_bbox = platform_cuboid_bbox_bottom + platform_cuboid_bbox_top

        platform_rect = platform_cuboid_bbox[:4]

        might_mark_cuboid_list = []
        might_mark_freespace_rect_list = []
        might_mark_freespace_id_list = []

        if mark_object:
            for child in self.offspring_per_platform[platform_id]:

                child_bbox_2d = child.object.convex_hull_2d.get_headed_bbox_instance()
                child_bbox_3d = [
                    np.append(point, child.bottom) for point in child_bbox_2d
                ] + [np.append(point, child.top) for point in child_bbox_2d]
                might_mark_cuboid_list.append(child_bbox_3d)

        if mark_freespace:

            if need_afford_rect is None:
                for i, j in [
                    (0, 1),
                    (0, 0),
                    (1, 0),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                    (1, 2),
                    (0, 2),
                    (1, 1),
                ]:
                    rect_part_3d = [
                        platform_cuboid_bbox[0]
                        + i * (platform_cuboid_bbox[1] - platform_cuboid_bbox[0]) / 3
                        + j * (platform_cuboid_bbox[3] - platform_cuboid_bbox[0]) / 3,
                        platform_cuboid_bbox[0]
                        + (i + 1)
                        * (platform_cuboid_bbox[1] - platform_cuboid_bbox[0])
                        / 3
                        + j * (platform_cuboid_bbox[3] - platform_cuboid_bbox[0]) / 3,
                        platform_cuboid_bbox[0]
                        + (i + 1)
                        * (platform_cuboid_bbox[1] - platform_cuboid_bbox[0])
                        / 3
                        + (j + 1)
                        * (platform_cuboid_bbox[3] - platform_cuboid_bbox[0])
                        / 3,
                        platform_cuboid_bbox[0]
                        + i * (platform_cuboid_bbox[1] - platform_cuboid_bbox[0]) / 3
                        + (j + 1)
                        * (platform_cuboid_bbox[3] - platform_cuboid_bbox[0])
                        / 3,
                    ]
                    might_mark_freespace_rect_list.append(rect_part_3d)
                    might_mark_freespace_id_list.append(i * 3 + j)

        optimal_pose, optimal_fovy = (
            image_render_processor.auto_get_optimal_camera_pose_for_object(
                view=view,
                camera_xy=camera_xy,
                z_range=z_range,
                object_bbox=platform_cuboid_bbox,
                platform_rect=platform_rect,
                width=width,
                height=height,
                focus_ratio=focus_ratio,
            )
        )
        img = None
        img = image_render_processor.auto_render_image_refactored(
            scene,
            pose=optimal_pose,
            fovy=optimal_fovy,
            width=width,
            height=height,
            might_mark_object_cuboid_list=might_mark_cuboid_list,
            might_mark_freespace_list=might_mark_freespace_rect_list,
            rectangle_grey=True,
            save_path=save_path,
        )

        pose_2d_list = camera_xy_list

        pose_list = [optimal_pose for i in range(len(pose_2d_list))]
        for i, pose_2d in enumerate(pose_2d_list):
            pose_list[i] = sapien.Pose(
                p=np.append(pose_2d, pose_list[i].p[2]), q=optimal_pose.q
            )

        img_list = [
            image_render_processor.auto_render_image_refactored(
                scene,
                pose=pose,
                fovy=optimal_fovy,
                width=width,
                height=height,
                might_mark_object_cuboid_list=might_mark_cuboid_list,
                might_mark_freespace_list=might_mark_freespace_rect_list,
                rectangle_grey=True,
                save_path=save_path.replace(
                    ".png", f"_{i+1}_out_of_{len(pose_list)}.png"
                ),
            )
            for i, pose in enumerate(pose_list)
        ]

        color_list = image_render_processor.get_high_contrast_colors()

        if len(might_mark_cuboid_list) > 0 and len(might_mark_freespace_rect_list) == 0:
            need_mark_rect_list = [rect[:4] for rect in might_mark_cuboid_list]
            platform_bbox = platform_cuboid_bbox[:4]
            camera_pose_list = pose_2d_list

            fig, ax = plt.subplots()

            # Plot platform bbox
            ax.plot(
                [platform_bbox[0][0], platform_bbox[1][0]],
                [platform_bbox[0][1], platform_bbox[1][1]],
                "k-",
                label="Platform",
            )
            ax.plot(
                [platform_bbox[1][0], platform_bbox[2][0]],
                [platform_bbox[1][1], platform_bbox[2][1]],
                "k-",
            )
            ax.plot(
                [platform_bbox[2][0], platform_bbox[3][0]],
                [platform_bbox[2][1], platform_bbox[3][1]],
                "k-",
            )
            ax.plot(
                [platform_bbox[3][0], platform_bbox[0][0]],
                [platform_bbox[3][1], platform_bbox[0][1]],
                "k-",
            )

            # Plot need_mark_rect_list
            for i, rect in enumerate(need_mark_rect_list):
                color = color_list[i % len(color_list)]
                color = [c / 255.0 for c in color]  # Normalize color values
                rect = [point[:2] for point in rect]
                ax.add_patch(
                    patches.Polygon(
                        rect,
                        closed=True,
                        facecolor=color,
                        alpha=1,
                        label=f"Item {i+1}",
                    )
                )

            # Plot camera poses
            camera_x = [pose[0] for pose in camera_pose_list]
            camera_y = [pose[1] for pose in camera_pose_list]
            ax.plot(camera_x, camera_y, "ro", label="CameraPoses")
            plt.subplots_adjust(left=0.01, right=0.5, top=0.91, bottom=0.01)

            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            ax.set_title("A top down view")
            ax.axis("equal")
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
                ncol=int(len(might_mark_cuboid_list) / 15 + 1),
            )
            ax.set_xticks([])  # Remove x axis ticks
            ax.set_yticks([])  # Remove y axis ticks

            plt.savefig(save_path.replace(".png", "_2d_visualization.png"))
            plt.close(fig)
            pass

        return img, img_list


class Tree:

    def __init__(self):
        self.nodes = {}  # Dictionary to access nodes by name
        self.edges = {}
        self.platforms = {}
        self.corresponding_scene = None
        self.mesh_checker = GroundCoverageAnalyzer()

    def get_node_categories_in_scene(self):
        major_categories = set()
        minor_categories = set()
        for node_name, node in self.nodes.items():
            if node.depth > 1:
                major_category = StringConvertor.get_category(node_name)
                minor_category = StringConvertor.get_name_wo_id(node_name)
                major_categories.add(major_category)
                minor_categories.add(minor_category)

        major_categories = list(major_categories)
        minor_categories = list(minor_categories)
        return major_categories, minor_categories

    def rename_all_features(self, new_name_map):
        new_nodes_dict = {}
        new_platform_dict = {}
        for node_name, node in self.nodes.items():

            new_node = node.rename_with_name_map(new_name_map)
            assert new_node is not None
            new_node_name = StringConvertor.rename_with_map(node_name, new_name_map)
            new_nodes_dict[new_node_name] = new_node

        self.nodes = new_nodes_dict

        for platform_name, platform in self.platforms.items():
            new_platform_name = StringConvertor.rename_with_map(
                platform_name, new_name_map
            )
            new_platform = platform.rename_with_name_map(new_name_map)
            new_platform_dict[new_platform_name] = new_platform

        self.platforms = new_platform_dict
        self.edges = {
            (
                StringConvertor.rename_with_map(edge[0], new_name_map),
                StringConvertor.rename_with_map(edge[1], new_name_map),
            ): platform
            for edge, platform in self.edges.items()
        }

        for entity in self.corresponding_scene.entities:
            new_entity_name = StringConvertor.rename_with_map(entity.name, new_name_map)
            entity.set_name(new_entity_name)

        return

    def load_corresponding_scene(self, scene):
        self.corresponding_scene = scene
        # Dictionary to access edges by name tuple(name1, name2, platform), edge[(name1, name2)]=platform means name1 is on a platform named platform, and platform is in name2

        # (self, name, heading=None, free_space=None, parent=None, children=None,convex_hull_2d=None, top = None, bbox=None):

    def get_non_ground_object_list(self):
        return [node for node in self.nodes.values() if node.depth > 1]

    def get_ground_object_list(self):
        return [node for node in self.nodes.values() if node.depth == 1]

    def get_ground_object_mesh_list(self):
        ground_object_list = self.get_ground_object_list()
        mesh_list = []
        for ground_object in ground_object_list:
            if ground_object.mesh is not None and ground_object.mesh.mesh is not None:
                mesh_list.append(ground_object.mesh.mesh)
        return mesh_list

    def cal_standable_area_for_platforms(self):

        ground_object_mesh_list = self.get_ground_object_mesh_list()
        for ground_mesh in ground_object_mesh_list:
            self.mesh_checker.add_ground_mesh(ground_mesh)

        self.mesh_checker.precompute_free_regions()

        for platform_name, platform in self.platforms.items():
            if self.nodes[platform.bel_object].depth != 1:
                continue

            standing_points_list = []
            for dir in range(0, 8, 2):
                standing_points, rectangles = self.mesh_checker.analyze_coverage(
                    platform.free_space[dir]["Available_space"],
                )

                standing_points_list.append(standing_points)

            self.platforms[platform_name].standing_point_list = standing_points_list
            for dir in range(0, 8, 2):
                print(platform.free_space[dir]["Available_space"])
            print(
                platform.convex_hull_2d.get_headed_bbox_instance(), standing_points_list
            )
            # ipdb.set_trace()

        pass

    def get_sensible_platform_list(self):

        platform_list = []
        ground_object_list = self.get_ground_object_list()
        for ground_object in ground_object_list:
            for platform in ground_object.own_platform:
                if (len(platform.children) > 0 or platform.avl_height > 0.1) and any(
                    [
                        len(platform.standing_point_list[dir // 2])
                        and platform.freespace_is_visible(dir)
                        for dir in range(0, 8, 2)
                    ]
                ):
                    platform_list.append(platform)
        return platform_list

    def auto_take_platform_picture(
        self,
        platform_name,
        view,
        mark_object=False,
        mark_freespace=False,
        standing_direction=0,
        camera_xy=None,
        width=1920,
        height=1080,
        focus_ratio=0.8,
        save_path="image.png",
    ):
        """

        take platform picture for a certain platform.

        This function is dependent on " auto_take_ground_object_picture".

        Args:
            platform_name: the name of the platform, which is a string.
            view: the view of the camera, which is a string. 'human_full', 'human_focus', 'top_full', 'top_focus'
            mark_object: if True, mark all the object on the same platform with cuboid.
            mark_freespace: if True, mark all the free space on the platform with cuboid.
            standing_direction: the direction of the camera, which is a int. 0, 1, 2, 3, 4, 5, 6, 7
            width: the width of the image, which is a int.
            height: the height of the image, which is a int.
            focus_ratio: the ratio of the image, which is a float.
            save_path: the path to save the image, which is a string.

        Returns:
            img: the image of the platform, which is a numpy array.
            img_list: the image list of the platform, which is a list of numpy array.


        """
        platform_bel_object_name = StringConvertor.get_name_wo_id(platform_name)
        platform_id_object = int(StringConvertor.get_id(platform_name))
        platform_bel_object = self.nodes[platform_bel_object_name]
        img, img_list = platform_bel_object.auto_take_ground_object_picture(
            scene=self.corresponding_scene,
            view=view,
            mark_object=mark_object,
            mark_freespace=mark_freespace,
            platform_id=platform_id_object,
            standing_direction=standing_direction,
            camera_xy=camera_xy,
            width=width,
            height=height,
            focus_ratio=focus_ratio,
            save_path=save_path,
        )
        return img, img_list

    def update_platform_children(self):
        self.platforms = {}
        for node_name, node in self.nodes.items():
            if node.depth == 1:
                self.nodes[node_name].update_children_belong_platform()
                for platform_id, platform in enumerate(node.own_platform):
                    self.platforms[platform.name] = node.own_platform[platform_id]

    def from_scene_platform_list(self, object_platform_list, contacts):
        for item in object_platform_list:
            if item.belong == None:
                item_node = TreeNode(
                    name=item.name,
                    heading=item.heading,
                    free_space=None,
                    bbox=item.bbox,
                    convex_hull_2d=item.convex_hull_2d,
                    bottom=item.height,
                    top=item.top,
                    mesh=item.mesh,
                    centroid_translation=item.centroid_translation,
                )
                self.set_node(item_node)
            else:
                # self.nodes[item.belong].own_platform.append(item)
                self.nodes[item.belong].own_platform.append(
                    TreePlatform(
                        name=item.name,
                        children=[],
                        heading=item.heading,
                        convex_hull_2d=item.convex_hull_2d,
                        bottom=item.height,
                        avl_height=item.avl_height,
                        bel_object=item.belong,
                        visible_directions=item.visible_directions,
                    )
                )

        # sort platform_list by height and rename them.
        platform_name_mapping = {}
        for node_name in self.nodes:
            platform_list = self.nodes[node_name].own_platform
            platform_list = TreePlatform.sort_platforms(platform_list)

            self.nodes[node_name].own_platform = platform_list
            for i, platform in enumerate(platform_list):
                new_name = node_name + "_" + str(i)
                platform_name_mapping[platform.name] = new_name
                self.platforms[new_name] = platform
                self.nodes[node_name].own_platform[i].name = new_name

        import ipdb

        ipdb.set_trace()
        #   glog.info('contacts: %s', contacts)

        for contact in contacts:

            if contact[0].belong is None:
                print("contact[0] should be a platform but belong is None")
                continue
            if contact[1].belong is not None:
                continue
                print("contact[1] should be a object but belong is not None")
            if (
                self.nodes[self.nodes[contact[0].belong].name].parent
                == self.nodes[contact[1].name]
            ):
                continue
            self.edges[(self.nodes[contact[0].belong].name, contact[1].name)] = contact[
                0
            ].name
            self.nodes[self.nodes[contact[0].belong].name].add_child(
                self.nodes[contact[1].name]
            )
            self.nodes[contact[1].name].parent = self.nodes[
                self.nodes[contact[0].belong].name
            ]

            contact_platform_name = platform_name_mapping[contact[0].name]
            self.nodes[contact[1].name].on_platform = self.platforms[
                contact_platform_name
            ]
            self.nodes[contact[1].name].free_space_height = [
                contact[0].height,
                contact[0].height + contact[0].avl_height,
            ]

            if contact_platform_name in self.platforms:

                self.platforms[contact_platform_name].children.append(
                    self.nodes[contact[1].name]
                )
            else:
                glog.warning(
                    f"contact[0].name not in platforms: {contact_platform_name}, {contact[1].name}"
                )

        on_platform_dict = {}
        ambiguous_items = set()
        for node in self.nodes.values():
            if node.on_platform is not None:
                node_category = StringConvertor.get_category(node.name)
                if node_category not in on_platform_dict:
                    on_platform_dict[node_category] = [node.on_platform]
                else:
                    for platform_have_this_item in on_platform_dict[node_category]:
                        if platform_have_this_item.name == node.on_platform.name:
                            ambiguous_items.add((node_category, node.on_platform.name))
                    on_platform_dict[node_category].append(node.on_platform)

        #   print("ambiguous items:", ambiguous_items)

        pass

        for node in self.nodes.values():
            node.update_children_belong_platform()

    def dfs_for_freespace(self, node):

        for child in node.children:
            child.depth = node.depth + 1
            self.nodes[child.name] = child
            if node.name == "GROUND":
                self.dfs_for_freespace(child)
            else:
                child.renew_heading(node.heading)

                self.dfs_for_freespace(child)

        if node.depth > 1:
            for i in range(len(node.children)):
                child = node.children[i]
                child.reset_free_space()
                child.set_free_space_to_platform(child.on_platform.convex_hull_2d)
                child.sync_critical_free_space()

        for i in range(len(node.children)):
            for j in range(len(node.children)):
                if i == j:
                    continue
                child1, child2 = node.children[i], node.children[j]
                child1_obj, child2_obj = child1.object, child2.object

                # Must belong to the same platform.
                if child1.on_platform.name != child2.on_platform.name:
                    continue

                surface_directions = []
                for k in range(8):
                    rect = child1.free_space[k]["Available_space"]
                    #    if 'picture_03' in child1.name:
                    #        print(child1.name, child2.name, rect, child2_obj.convex_hull_2d.get_headed_bbox_instance())
                    #        print(child2_obj.convex_hull_2d.is_intersected_with_rectangle(rect))
                    if child2_obj.convex_hull_2d.is_intersected_with_rectangle(rect):
                        surface_directions.append(k)

                for k in range(0, 8, 2):
                    if (
                        k not in surface_directions
                        and (k + 1) % 8 in surface_directions
                        and (k + 7) % 8 in surface_directions
                    ):
                        surface_directions.append(k)

                # if child1.name == 'frl_apartment_wall_cabinet_01_5':
                #    print(child2.name, surface_directions)

                for direction in surface_directions:
                    #          print(child1.name, child2.name, direction)
                    child1.update_free_space(child2, direction)

        for i in range(len(node.children)):
            child = node.children[i]
            for left_part in range(1, 4):

                for object_node in child.free_space[left_part]["Objects"]:
                    near_side = [
                        child.free_space[left_part]["Critical_space"][3],
                        child.free_space[left_part]["Critical_space"][2],
                    ]
                    far_side = [
                        child.free_space[left_part]["Critical_space"][0],
                        child.free_space[left_part]["Critical_space"][1],
                    ]

                    # if child.name == 'frl_apartment_wall_cabinet_01_5':
                    #    print(object_node.name, EIGHT_DIRECTIONS[left_part])
                    #    print('near',near_side,'far', far_side)
                    #    print('convexhull', object_node.object.convex_hull_2d.get_vertices_on_convex_hull())
                    new_near_side, new_far_side = (
                        object_node.object.convex_hull_2d.cut_free_space_with_convex(
                            near_side, far_side
                        )
                    )
                    child.free_space[left_part]["Critical_space"] = [
                        new_far_side[0],
                        new_far_side[1],
                        new_near_side[1],
                        new_near_side[0],
                    ]
                    # if child.name == 'frl_apartment_wall_cabinet_01_5':
                    #    print('new near far', new_near_side, new_far_side)

            for right_part in range(5, 8):

                for object_node in child.free_space[right_part]["Objects"]:
                    near_side = [
                        child.free_space[right_part]["Critical_space"][0],
                        child.free_space[right_part]["Critical_space"][1],
                    ]
                    far_side = [
                        child.free_space[right_part]["Critical_space"][3],
                        child.free_space[right_part]["Critical_space"][2],
                    ]

                    # if child.name == 'frl_apartment_wall_cabinet_01_5':
                    #    print(object_node.name, EIGHT_DIRECTIONS[right_part])
                    #    print(near_side, far_side)
                    #    print(object_node.object.convex_hull_2d.get_vertices_on_convex_hull())
                    new_near_side, new_far_side = (
                        object_node.object.convex_hull_2d.cut_free_space_with_convex(
                            near_side, far_side
                        )
                    )
                    child.free_space[right_part]["Critical_space"] = [
                        new_near_side[0],
                        new_near_side[1],
                        new_far_side[1],
                        new_far_side[0],
                    ]
                    # if child.name == 'frl_apartment_wall_cabinet_01_5':
                    #    print('new near far', new_near_side, new_far_side)
            for front_part in range(3, 6):

                for object_node in child.free_space[front_part]["Objects"]:
                    near_side = [
                        child.free_space[front_part]["Critical_space"][0],
                        child.free_space[front_part]["Critical_space"][3],
                    ]
                    far_side = [
                        child.free_space[front_part]["Critical_space"][1],
                        child.free_space[front_part]["Critical_space"][2],
                    ]

                    new_near_side, new_far_side = (
                        object_node.object.convex_hull_2d.cut_free_space_with_convex(
                            near_side, far_side
                        )
                    )
                    child.free_space[front_part]["Critical_space"] = [
                        new_near_side[0],
                        new_far_side[0],
                        new_far_side[1],
                        new_near_side[1],
                    ]

            for rear_part in [7, 0, 1]:

                for object_node in child.free_space[rear_part]["Objects"]:
                    near_side = [
                        child.free_space[rear_part]["Critical_space"][1],
                        child.free_space[rear_part]["Critical_space"][2],
                    ]
                    far_side = [
                        child.free_space[rear_part]["Critical_space"][0],
                        child.free_space[rear_part]["Critical_space"][3],
                    ]

                    new_near_side, new_far_side = (
                        object_node.object.convex_hull_2d.cut_free_space_with_convex(
                            near_side, far_side
                        )
                    )

                    child.free_space[rear_part]["Critical_space"] = [
                        new_far_side[0],
                        new_near_side[0],
                        new_near_side[1],
                        new_far_side[1],
                    ]

        return None

    def calculate_free_space(self):
        for node in self.nodes.values():
            # print(node.name, node.parent.name if node.parent is not None else None, node.on_platform)
            if node.depth == 0:
                if "GROUND" in node.name:
                    self.dfs_for_freespace(node)
        return None

    def clean_zero_area_free_space(self):
        for node in self.nodes.values():
            node.clean_free_space()

    def cut_free_space_with_stage(self, stage):

        min_intersection_height = INF
        max_stage_height = -INF
        for stage_obj in stage:

            geometry = stage_obj.mesh
            max_stage_height = max(max_stage_height, geometry.mesh.bounds[1][2])

            for node_name, node in self.nodes.items():

                if node.free_space is None:
                    continue
                bottom, top = node.bottom, node.top
                if bottom > top:
                    continue

                node_bbox = node.object.convex_hull_2d.get_headed_bbox_instance()

                node_bbox_8_directions = [
                    node_bbox[0],
                    (node_bbox[0] + node_bbox[1]) * 0.5,
                    node_bbox[1],
                    (node_bbox[1] + node_bbox[2]) * 0.5,
                    node_bbox[2],
                    (node_bbox[2] + node_bbox[3]) * 0.5,
                    node_bbox[3],
                    (node_bbox[3] + node_bbox[0]) * 0.5,
                ]

                for left_part in range(1, 4):
                    rect = node.free_space[left_part]["Critical_space"]
                    if isinstance(rect, str):
                        continue

                    cuboid_vertices = [
                        [rect[0][0], rect[0][1], bottom],
                        [rect[1][0], rect[1][1], bottom],
                        [rect[2][0], rect[2][1], bottom],
                        [rect[3][0], rect[3][1], bottom],
                        [rect[0][0], rect[0][1], top],
                        [rect[1][0], rect[1][1], top],
                        [rect[2][0], rect[2][1], top],
                        [rect[3][0], rect[3][1], top],
                    ]
                    # cuboid_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]

                    if (
                        np.min(np.array(rect)) < -1e10
                        or np.max(np.array(rect)) > 1e10
                        or bottom < -1e10
                        or top > 1e10
                    ):
                        print("overflow", node_name)

                    cuboid = MeshProcessor.create_cuboid_from_vertices(cuboid_vertices)
                    # print(bottom, top, rect,    cuboid.mesh.volume)
                    intersect = geometry.intersection_with_cuboid(cuboid)

                    if (
                        intersect is not None
                        and intersect.mesh.bounds[1][2] - intersect.mesh.bounds[0][2]
                        > 1e-1
                    ):
                        #     if node_name == 'frl_apartment_table_02_49':
                        #         print(node_name, stage_obj.name, 'left',EIGHT_DIRECTIONS[left_part], intersect.mesh.bounds)
                        intersect_2d_convex = intersect.cal_convex_hull_2d()
                        near_side = [rect[3], rect[2]]
                        far_side = [rect[0], rect[1]]
                        new_near_side, new_far_side = (
                            intersect_2d_convex.cut_free_space_with_point_cloud(
                                near_side,
                                far_side,
                                node_bbox_8_directions[left_part],
                            )
                        )
                        self.nodes[node_name].free_space[left_part][
                            "Critical_space"
                        ] = [
                            new_far_side[0],
                            new_far_side[1],
                            new_near_side[1],
                            new_near_side[0],
                        ]
                        self.nodes[node_name].free_space[left_part][
                            "Available_space"
                        ] = [
                            new_far_side[0],
                            new_far_side[1],
                            new_near_side[1],
                            new_near_side[0],
                        ]
                        #      if node_name == 'frl_apartment_table_02_49':
                        #           print('near',near_side,'far', far_side)
                        #          print('newnear', new_near_side,'newfar', new_far_side)
                        min_intersection_height = min(
                            min_intersection_height, intersect.mesh.bounds[0][2]
                        )
                for right_part in range(5, 8):
                    rect = node.free_space[right_part]["Critical_space"]
                    if isinstance(rect, str):
                        continue

                    cuboid_vertices = [
                        [rect[0][0], rect[0][1], bottom],
                        [rect[1][0], rect[1][1], bottom],
                        [rect[2][0], rect[2][1], bottom],
                        [rect[3][0], rect[3][1], bottom],
                        [rect[0][0], rect[0][1], top],
                        [rect[1][0], rect[1][1], top],
                        [rect[2][0], rect[2][1], top],
                        [rect[3][0], rect[3][1], top],
                    ]
                    cuboid = MeshProcessor.create_cuboid_from_vertices(cuboid_vertices)
                    intersect = geometry.intersection_with_cuboid(cuboid)
                    #  if node_name == 'frl_apartment_table_04_14':
                    #         print(node_name, stage_obj.name, EIGHT_DIRECTIONS[right_part], intersect.mesh.bounds if intersect is not None else None)

                    if (
                        intersect is not None
                        and intersect.mesh.bounds[1][2] - intersect.mesh.bounds[0][2]
                        > 1e-1
                    ):
                        intersect_2d_convex = intersect.cal_convex_hull_2d()
                        near_side = [rect[0], rect[1]]
                        far_side = [rect[3], rect[2]]
                        new_near_side, new_far_side = (
                            intersect_2d_convex.cut_free_space_with_point_cloud(
                                near_side,
                                far_side,
                                node_bbox_8_directions[right_part],
                            )
                        )
                        self.nodes[node_name].free_space[right_part][
                            "Critical_space"
                        ] = [
                            new_near_side[0],
                            new_near_side[1],
                            new_far_side[1],
                            new_far_side[0],
                        ]
                        self.nodes[node_name].free_space[right_part][
                            "Available_space"
                        ] = [
                            new_near_side[0],
                            new_near_side[1],
                            new_far_side[1],
                            new_far_side[0],
                        ]
                        min_intersection_height = min(
                            min_intersection_height, intersect.mesh.bounds[0][2]
                        )
                for front_part in range(3, 6):
                    rect = node.free_space[front_part]["Critical_space"]
                    if isinstance(rect, str):
                        continue

                    cuboid_vertices = [
                        [rect[0][0], rect[0][1], bottom],
                        [rect[1][0], rect[1][1], bottom],
                        [rect[2][0], rect[2][1], bottom],
                        [rect[3][0], rect[3][1], bottom],
                        [rect[0][0], rect[0][1], top],
                        [rect[1][0], rect[1][1], top],
                        [rect[2][0], rect[2][1], top],
                        [rect[3][0], rect[3][1], top],
                    ]
                    cuboid = MeshProcessor.create_cuboid_from_vertices(cuboid_vertices)
                    intersect = geometry.intersection_with_cuboid(cuboid)
                    if (
                        intersect is not None
                        and intersect.mesh.bounds[1][2] - intersect.mesh.bounds[0][2]
                        > 1e-1
                    ):
                        intersect_2d_convex = intersect.cal_convex_hull_2d()
                        near_side = [rect[0], rect[3]]
                        far_side = [rect[1], rect[2]]
                        new_near_side, new_far_side = (
                            intersect_2d_convex.cut_free_space_with_point_cloud(
                                near_side,
                                far_side,
                                node_bbox_8_directions[front_part],
                            )
                        )
                        self.nodes[node_name].free_space[front_part][
                            "Critical_space"
                        ] = [
                            new_near_side[0],
                            new_far_side[0],
                            new_far_side[1],
                            new_near_side[1],
                        ]
                        self.nodes[node_name].free_space[front_part][
                            "Available_space"
                        ] = [
                            new_near_side[0],
                            new_far_side[0],
                            new_far_side[1],
                            new_near_side[1],
                        ]
                        min_intersection_height = min(
                            min_intersection_height, intersect.mesh.bounds[0][2]
                        )
                for rear_part in [7, 0, 1]:
                    rect = node.free_space[rear_part]["Critical_space"]
                    if isinstance(rect, str):
                        continue

                    cuboid_vertices = [
                        [rect[0][0], rect[0][1], bottom],
                        [rect[1][0], rect[1][1], bottom],
                        [rect[2][0], rect[2][1], bottom],
                        [rect[3][0], rect[3][1], bottom],
                        [rect[0][0], rect[0][1], top],
                        [rect[1][0], rect[1][1], top],
                        [rect[2][0], rect[2][1], top],
                        [rect[3][0], rect[3][1], top],
                    ]
                    cuboid = MeshProcessor.create_cuboid_from_vertices(cuboid_vertices)
                    intersect = geometry.intersection_with_cuboid(cuboid)
                    if (
                        intersect is not None
                        and intersect.mesh.bounds[1][2] - intersect.mesh.bounds[0][2]
                        > 1e-1
                    ):
                        intersect_2d_convex = intersect.cal_convex_hull_2d()
                        near_side = [rect[1], rect[2]]
                        far_side = [rect[0], rect[3]]
                        new_near_side, new_far_side = (
                            intersect_2d_convex.cut_free_space_with_point_cloud(
                                near_side,
                                far_side,
                                node_bbox_8_directions[rear_part],
                            )
                        )
                        self.nodes[node_name].free_space[rear_part][
                            "Critical_space"
                        ] = [
                            new_far_side[0],
                            new_near_side[0],
                            new_near_side[1],
                            new_far_side[1],
                        ]
                        self.nodes[node_name].free_space[rear_part][
                            "Available_space"
                        ] = [
                            new_far_side[0],
                            new_near_side[0],
                            new_near_side[1],
                            new_far_side[1],
                        ]
                        min_intersection_height = min(
                            min_intersection_height, intersect.mesh.bounds[0][2]
                        )

        for node_name, node in self.nodes.items():
            self.nodes[node_name].free_space_height[1] = min(
                self.nodes[node_name].free_space_height[1], max_stage_height
            )

    def fix_heading_for_all_ground_objects(self):
        for ground_object_name, ground_object in self.nodes.items():
            if ground_object.depth == 1:
                standable_heading = 0
                for i in range(0, 8, 2):
                    if ground_object.freespace_is_standable(i):
                        standable_heading = i
                        break
                self.nodes[ground_object_name].rotate_free_space_for_all_children(
                    standable_heading
                )

    def set_node(self, node):
        self.nodes[node.name] = node

    def get_node(self, name):
        return self.nodes.get(name, None)

    def remove_node_from_scene(self, name):
        node = self.get_node(name)
        # import ipdb
        # ipdb.set_trace()
        if node is not None:
            for entity in self.corresponding_scene.entities:
                if node.name in entity.get_name():
                    entity.remove_from_scene()

    def add_node_to_scene(self, name):
        node = self.get_node(name)
        # import ipdb

        # ipdb.set_trace()
        if node is not None:
            obj = node.entity_config
            object_file_path = obj["visual_path"]
            collision_path = obj.get("collision_path", None)
            position = [
                obj["centroid_translation"]["x"],
                obj["centroid_translation"]["y"],
                obj["centroid_translation"]["z"],
            ]

            quaternion = [
                obj["quaternion"]["w"],
                obj["quaternion"]["x"],
                obj["quaternion"]["y"],
                obj["quaternion"]["z"],
            ]

            if "cushion_03" in object_file_path:
                position[2] += 0.1

            rpy = transforms3d.euler.quat2euler(quaternion, axes="sxyz")
            quaternion = transforms3d.euler.euler2quat(
                rpy[0] + np.deg2rad(90), rpy[1], rpy[2], axes="sxyz"
            )

            builder = self.corresponding_scene.create_actor_builder()
            builder.add_visual_from_file(filename=object_file_path)
            if collision_path is not None:
                builder.add_multiple_convex_collisions_from_file(
                    filename=collision_path
                )
            else:
                builder.add_convex_collision_from_file(filename=object_file_path)
            # import ipdb

            # ipdb.set_trace()
            mesh = builder.build_static(name=name)
            mesh.set_pose(sapien.Pose(p=position, q=quaternion))

    def remove_node(self, name):
        node = self.get_node(name)

        if node is not None:
            parent_name = node.parent.name if node.parent is not None else None

            self.nodes[name].removed = True
            if node.parent is not None:
                self.nodes[parent_name].remove_child(node)

            for dir in range(8):
                self.nodes[name].free_space[dir]["Objects"] = []
            platform_name = (
                node.on_platform.name if node.on_platform is not None else None
            )
            for child in node.all_children:
                # child.parent = node.parent
                # child.on_platform = node.on_platform
                child.free_space_height = node.free_space_height.copy()
                self.nodes[parent_name].add_child(child)

                self.platforms[platform_name] = node.on_platform

            for platform in node.own_platform:
                platform.children = []

                if platform.name in self.platforms:
                    self.platforms[platform.name] = platform

            self.dfs_for_freespace(self.nodes[parent_name])
            node.parent = None

            self.remove_node_from_scene(name)
            node.update_all_child()
            import ipdb

            ipdb.set_trace()

            for child in node.all_children:
                self.remove_node_from_scene(child.name)

            self.update_platform_children()

    def add_node(self, node_name, parent_name, platform_name, translation):

        if parent_name not in self.nodes.keys():
            print("Parent node not found", parent_name, platform_name)
            return
        node = self.nodes[node_name]
        parent = self.nodes[parent_name]
        platform = self.platforms[platform_name]

        if platform is None:
            print("Platform not found", parent_name, platform_name)
            return
        node.update_all_child()
        node_add_list = [node] + node.all_children
        relative_translation = np.array(translation) - np.array(
            [
                node.entity_config["centroid_translation"]["x"],
                node.entity_config["centroid_translation"]["y"],
            ]
        )
        relative_bottom_translation = platform.bottom - node.bottom

        node.parent = parent
        node.on_platform = platform
        node.free_space_height = [
            platform.bottom,
            platform.bottom + platform.avl_height,
        ]
        import ipdb

        ipdb.set_trace()
        for n in node_add_list:
            n.removed = False

            n.heading = n.object.heading = n.object.convex_hull_2d.heading = (
                parent.heading
            )

            n.object.convex_hull_2d.vertices += np.array(relative_translation)
            new_bottom = platform.bottom
            n.entity_config["centroid_translation"]["x"] += relative_translation[0]
            n.entity_config["centroid_translation"]["y"] += relative_translation[1]
            n.entity_config["centroid_translation"]["z"] += relative_bottom_translation
            n.top += relative_bottom_translation
            n.bottom += relative_bottom_translation

            self.nodes[n.name] = n
        # node.removed = False
        # node.parent = parent
        # node.on_platform = platform
        # node.free_space_height = [
        #     platform.bottom,
        #     platform.bottom + platform.avl_height,
        # ]

        # node.heading = node.object.heading = node.object.convex_hull_2d.heading = (
        #     parent.heading
        # )
        # node.object.convex_hull_2d.vertices += np.array(translation) - np.array(
        #     node.get_center()[:2]
        # )

        # new_bottom = platform.bottom
        # node.entity_config["centroid_translation"]["x"] = translation[0]
        # node.entity_config["centroid_translation"]["y"] = translation[1]
        # node.entity_config["centroid_translation"]["z"] += new_bottom - node.bottom

        # node.top = node.top - node.bottom + new_bottom
        # node.bottom = new_bottom

        # self.nodes[node_name] = node
        parent.add_child(self.nodes[node_name])
        platform.children.append(self.nodes[node_name])
        self.dfs_for_freespace(self.nodes[parent_name])
        # import ipdb
        # ipdb.set_trace()
        node.update_all_child()

        self.add_node_to_scene(node.name)
        for child in node.all_children:
            self.add_node_to_scene(child.name)
        self.nodes[node_name].display_unique_information()
        pass


def print_tree(node, depth=0):
    print("  " * depth + f"Object Name: {node.name}")

    all_free = True
    for direction in node.free_space.keys():
        if node.free_space[direction]["Empty"] == False:
            print(
                "  " * depth
                + f'*Occupied Direction: {direction}, Objects: {node.free_space[direction]["Objects"]}'
            )
            all_free = False
    if all_free:
        print("  " * depth + f"*All Directions are Free")

    for child in node.children:
        print_tree(child, depth + 1)


# Function to generate the tree structure starting from ground objects
# key function
def gen_multi_layer_graph_with_free_space(json_data):

    all_objects = [obj for obj in json_data["object_instances"]]
    import ipdb

    ipdb.set_trace()
    scene_platform_list = []
    stage = [
        {
            "name": "stage",
            "visual_path": json_data["background_file_path"],
            "centroid_translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "bbox": "deprecated",
        }
    ]

    ground = scene_parser.SceneObject(name="GROUND")
    ground.set_ground()
    scene_platform_list.append(
        scene_parser.SceneElement(
            name="GROUND_0",
            heading=(1, 0),
            height=0,
            avl_height=3.08,
            bbox=[[-INF, -INF], [INF, INF]],
            instancetype="platform",
            convex_hull_2d=ConvexHullProcessor_2d(
                vertices=[[-INF, -INF], [INF, -INF], [INF, INF], [-INF, INF]],
                heading=(1, 0),
            ),
            belong="GROUND",
        )
    )
    object_list = scene_parser.create_object_list(
        all_objects, calculate_affordable_platforms=True
    )
    object_list.append(ground)

    stage_list = scene_parser.create_object_list(
        stage, calculate_affordable_platforms=False
    )
    for object in object_list:
        # the height of object is the height of the object's bottom.
        # Note: for object, get_bounding_box has already calculated the bbox_min and bbox_max in the world coordinate system, added the centroid_translation.
        bbox_min, bbox_max = object.get_bounding_box()
        scene_platform_list.append(
            scene_parser.SceneElement(
                name=object.name,
                heading=object.heading,
                height=bbox_min[2],
                top=bbox_max[2],
                bbox=[bbox_min[:2], bbox_max[:2]],
                bounding_points=object.bounding_points,
                centroid_translation=object.centroid_translation,
                convex_hull_2d=object.convex_hull_2d,
                mesh=object.mesh,
                instancetype="object",
            )
        )
        # the height of platform is the height of the platform's top.
        geometry = object.mesh
        for platform in geometry.affordable_platforms:
            scene_platform_list.append(
                scene_parser.SceneElement(
                    name=object.name + platform.name,
                    heading=object.heading,
                    bbox=platform.bbox
                    + np.array(
                        [
                            object.centroid_translation[:2],
                            object.centroid_translation[:2],
                        ]
                    ),
                    height=platform.get_height()[1] + object.centroid_translation[2],
                    avl_height=platform.available_height,
                    convex_hull_2d=platform.get_convex_hull_2d()
                    + object.centroid_translation[:2],
                    instancetype="platform",
                    visible_directions=platform.visible_directions,
                    belong=object.name,
                )
            )
            # print(platform.get_convex_hull_2d() + object.centroid_translation[:2])

    sorted_scene_platform_list, contacts_id = (
        scene_parser.SceneElement.calculate_contact_conditions(scene_platform_list)
    )

    contacts = [
        (sorted_scene_platform_list[i[0]], sorted_scene_platform_list[i[1]])
        for i in contacts_id
    ]

    scene_graph_tree = Tree()
    scene_graph_tree.from_scene_platform_list(sorted_scene_platform_list, contacts)

    # import ipdb
    # ipdb.set_trace()

    scene_graph_tree.calculate_free_space()

    # scene_graph_tree.clean_zero_area_free_space()
    scene_graph_tree.cal_standable_area_for_platforms()
    scene_graph_tree.cut_free_space_with_stage(stage_list)

    scene_graph_tree.update_platform_children()

    for obj_instance in all_objects:
        if "name" not in obj_instance and "template_name" in obj_instance:
            obj_instance["name"] = obj_instance["template_name"]
        if scene_graph_tree.get_node(obj_instance["name"]) is None:
            print(obj_instance["name"], "not in the tree")
        else:
            scene_graph_tree.nodes[obj_instance["name"]].entity_config = obj_instance

    # scene_graph_tree.clean_zero_area_free_space()
    if "cabinet_3_body" in scene_graph_tree.nodes.keys():
        print(scene_graph_tree.nodes["cabinet_3_body"].heading)
        pass
    return scene_graph_tree


def main():
    # Load JSON data and generate the tree structure
    # input_file_path = './replica_apt_0.json'
    # input_file_path = './toy_scene_2.json'
    parser = argparse.ArgumentParser(
        description="Generate scene graph with free space information."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="entities_apt_0.json",
        required=False,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./result/replica_apt_0.txt",
        required=False,
        help="Path to the output text file.",
    )
    parser.add_argument(
        "--output_scene_graph_tree",
        type=str,
        default="./result/scene_graph_tree_0.txt",
        required=False,
        help="Path to the output text file.",
    )
    args = parser.parse_args()

    input_file_path = args.input
    output_file_path = args.output
    output_scene_graph_path = args.output_scene_graph_tree
    input_json = load_json_file(input_file_path)

    with open(output_file_path, "w") as output_file:
        output_file.write("----------------------------------\n")
        output_file.write("Establishing the tree structure:\n")
        result_tree = gen_multi_layer_graph_with_free_space(input_json)

        output_file.write("\n----------------------------------\n")
        output_file.write("Result tree:\n")
        for root_node in result_tree.nodes.values():
            if (
                root_node.parent is None
            ):  # Only print root nodes (those without a parent)
                output_file.write("---\n")
                output_file.write(f"Object Name: {root_node.name}\n")
                all_free = True
                if root_node.free_space is None:
                    output_file.write(f"*No Free Space Information\n")
                    continue
                for direction in range(len(root_node.free_space)):
                    if len(root_node.free_space[direction]["Objects"]):
                        output_file.write(
                            f'*Occupied Direction : {direction}, Objects: {root_node.free_space[direction]["Objects"]}\n'
                        )
                        all_free = False
                if all_free:
                    output_file.write(f"*All Directions are Free\n")
                output_file.write("---\n")

        output_file.write("\n----------------------------------\n")
        output_file.write("Node details in result tree:\n")
        for obj in input_json["object_instances"]:
            node = result_tree.get_node(obj["name"])
            if node is None:
                continue
            output_file.write("---\n")
            output_file.write(
                f"Name: {node.name}, Parent: {node.parent.name if node.parent else 'None'}, Children: {[child.name for child in node.children]}\n"
            )
            output_file.write(
                f"convex_bbox: {node.object.convex_hull_2d.get_headed_bbox_instance()}\n"
            )
            output_file.write(
                f"convex: {node.object.convex_hull_2d.get_vertices_on_convex_hull()}\n"
            )
            output_file.write(f"Heading: {node.heading}\n")
            output_file.write(f"Top: {node.top}\n")
            output_file.write(f"Belong to platform: {node.on_platform.name}\n")
            for direction in range(len(root_node.free_space)):
                free_space_info = node.free_space[direction]["Available_space"]
                critical_space_info = node.free_space[direction]["Critical_space"]

                output_file.write("-\n")
                output_file.write(
                    f"Direction: {EIGHT_DIRECTIONS[direction]},  Objects: {[object.name for object in node.free_space[direction]['Objects']]}\n"
                )
                output_file.write(f"Available Space: \n[{free_space_info}]\n")
                if isinstance(critical_space_info, str):
                    output_file.write(f"Critical Available Space: \nNot Available\n")
                else:
                    output_file.write(
                        f"Critical Available Space: \n[{critical_space_info}]\n"
                    )
                output_file.write(
                    f"Free Space Height: \n[{float(node.free_space_height[0])},{float(node.free_space_height[1])}]\n"
                )
            output_file.write("-\n")
            output_file.write(f"Own Platform: \n")
            for platform in node.own_platform:
                output_file.write(f"{platform.name}\n")
                output_file.write(
                    f"bbox: \n[[{platform.bbox[0][0]},{platform.bbox[0][1]}],[{platform.bbox[1][0]},{platform.bbox[1][1]}]] \n"
                )
                output_file.write(f"height: \n{platform.height}\n")
                output_file.write(f"available height: \n{platform.avl_height}\n")
            output_file.write("\n")


if __name__ == "__main__":
    main()


"""


"""
# %%
