import numpy as np
from queue import Queue
from enum import Enum
from src.geometry.convex_hull_processor import ConvexHullProcessor_2d
from src.geometry.basic_geometries import Basic2DGeometry
from src.utils.string_convertor import StringConvertor

import itertools
import jsonschema
import glog
import time
import os

NINE_DIRECTIONS = [
    "rear",
    "rear-left",
    "left",
    "front-left",
    "front",
    "front-right",
    "right",
    "rear-right",
    "center",
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
fovy_range_default = [np.deg2rad(5), np.deg2rad(100)]


class ItemType(Enum):
    GROUND = 0
    GROUND_OBJECT = 1
    NON_GROUND_OBJECT = 2
    MOVEABLE_OBJECT = 3
    DEFAULT = -1


class TaskType(Enum):

    NOT_FOR_TASK = -1
    ONLY_FOR_WANDER = -2
    # move a from A to B's empty platform
    MOVE_TO_EMPTY_PLATFORM = 0
    MOVE_TO_EMPTY_PLATFORM_9_GRID = 1
    # move a from A to b's {dir} freespace
    MOVE_AROUND_OBJECT = 2
    MOVE_TO_OBJECT_FREESPACE_9_GRID = 3

    # move a fromt A to b and c's middle (i.e. any shared freespace of B and C)
    MOVE_TO_MIDDLE_OF_OBJECTS = 4


class TaskStatusCode(Enum):
    SUCCESS = 0
    FAILURE = 1


class Task:

    def __init__(
        self,
        item,
        destination,
        type: TaskType = None,
        feature=None,
        intermediate_state_list=None,
        goal_translation=None,
        single_freespace_translation=None,
        need_merge_freespace=False,
    ):
        # print(type)
        # assert isinstance(type, TaskType)
        # assert dir in NINE_DIRECTIONS
        self.item = item
        self.destination = destination
        self.type = type
        self.feature = feature
        """
        MOVE_TO_EMPTY_PLATFORM:
        feature = {}
        MOVE_TO_EMPTY_PLATFORM_9_GRID:
        feature = {dir: str}
        MOVE_AROUND_OBJECT:
        feature = {object: node}
        MOVE_TO_OBJECT_FREESPACE_9_GRID:
        feature = {object: node, dir: str}
        MOVE_TO_MIDDLE_OF_OBJECTS:
        feature = {object1: node, object2: node}
        """

        self.goal_translation = goal_translation
        self._goal_primitive_expression = None
        self.single_freespace_translation = single_freespace_translation
        self.need_merge_freespace = need_merge_freespace
        self.intermediate_state_list = intermediate_state_list
        self.intermedaite_state_repr_list = None

        if self.intermediate_state_list is None:
            self.generate_default_intermediate_state()

    @property
    def goal_primitive_expression(self):
        if self._goal_primitive_expression is None:
            from src.core.task_primitive import TaskPrimitive

            self._goal_primitive_expression = TaskPrimitive.from_task(self)
        return self._goal_primitive_expression

    def generate_default_intermediate_state(self):
        item_platform_name = (
            self.item.get_bel_ground_platform().get_name_for_interaction()
        )
        item_name = self.item.get_name_for_interaction()
        destination_name = self.destination.get_name_for_interaction()
        self.intermediate_state_list = [
            {"holding": "any", "at_platform": item_platform_name},
            {"holding": self.item.name, "at_platform": "any"},
            {"holding": self.item.name, "at_platform": destination_name},
        ]

        self.intermediate_state_repr_list = [
            f"Go to {item_platform_name} where {item_name} is located.",
            f"Pick up {item_name} from the {item_platform_name}",
            f"Go to {destination_name} to place {item_name}.",
            f"Place {item_name} on the {destination_name} and make sure it is placed according the task requirement.",
        ]

    def check_pattern_jsonschema(data, schema):
        try:
            jsonschema.validate(data, schema)
        except jsonschema.exceptions.ValidationError as e:
            return False
        return True

    def validate_task(self):

        if self.type == TaskType.MOVE_TO_EMPTY_PLATFORM:
            if not self.check_pattern_jsonschema(self.feature, {}):
                return False
        elif self.type == TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID:
            if not self.check_pattern_jsonschema(self.feature, {"dir": str}):
                return False
        elif self.type == TaskType.MOVE_AROUND_OBJECT:
            if not self.check_pattern_jsonschema(self.feature, {"object": str}):
                return False
        elif self.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID:
            if not self.check_pattern_jsonschema(
                self.feature, {"object": str, "dir": str}
            ):
                return False
        elif self.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
            if not self.check_pattern_jsonschema(
                self.feature, {"object1": str, "object2": str}
            ):
                return False

    def debug_check_bboxes(self):
        print(
            self.destination.bbox[3] - self.destination.bbox[0],
            self.destination.bbox[1] - self.destination.bbox[0],
        )
        heading = (
            self.destination.bbox[3] - self.destination.bbox[0]
        ) / np.linalg.norm(self.destination.bbox[3] - self.destination.bbox[0])
        headed_bbox = ConvexHullProcessor_2d.get_headed_bbox(
            self.item.object.convex_hull_2d.vertices, heading
        )
        print(headed_bbox[3] - headed_bbox[0], headed_bbox[1] - headed_bbox[0])

    def is_ambiguous(self, scene_graph_tree):
        if (
            self.type == TaskType.MOVE_TO_EMPTY_PLATFORM
            or self.type == TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID
        ):
            return scene_graph_tree.nodes[self.item.name].is_ambiguous
        elif (
            self.type == TaskType.MOVE_AROUND_OBJECT
            or self.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID
        ):
            return (
                scene_graph_tree.nodes[self.item.name].is_ambiguous
                or scene_graph_tree.nodes[self.feature[0]].is_ambiguous
            )
        elif self.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
            return (
                scene_graph_tree.nodes[self.item.name].is_ambiguous
                or scene_graph_tree.nodes[self.feature[0]].is_ambiguous
                or scene_graph_tree.nodes[self.feature[1]].is_ambiguous
            )
        pass

    def task_debug(self, scene_graph_tree, id=-1):
        img_save_path = f'./image4debug/goal/Task{id}_{self.item.get_name_for_interaction()}_{self.destination.get_name_for_interaction()}_{[feature.split("/") for feature in self.feature ]if self.feature is not None else None}.png'
        scene_graph_tree.update_platform_children()

        scene_graph_tree.remove_node(self.item.name)

        scene_graph_tree.add_node(
            self.item.name,
            self.destination.bel_object,
            self.destination.name,
            self.goal_translation,
        )

        item_node = scene_graph_tree.nodes[self.item.name]
        destination_bel_object = scene_graph_tree.nodes[self.destination.bel_object]
        standing_direction = [
            dir
            for dir in range(0, 8, 2)
            if self.destination.freespace_is_visible(dir)
            and destination_bel_object.freespace_is_standable(dir)
        ]
        if len(standing_direction) == 0:
            glog.info(
                f"Item {self.item.name} cannot be placed on {self.destination.name}"
            )
            return
        img = item_node.auto_take_non_ground_object_picture(
            scene=scene_graph_tree.corresponding_scene,
            view="human_focus",
            mark_object=False,
            only_mark_itself=False,
            mark_freespace=False,
            diagonal_mode="old",
            need_afford_rect=None,
            standing_direction=standing_direction[0],
            width=1366,
            height=768,
            focus_ratio=0.5,
            fovy_range=[np.deg2rad(40), np.deg2rad(60)],
            save_path=img_save_path,
        )

        goal_translation = self.goal_translation

    def check_task_success(self, scene_graph_tree):
        """
        wait to be implemented

        """
        if self.type == TaskType.MOVE_TO_EMPTY_PLATFORM:
            return self.check_empty_platform(scene_graph_tree)
        elif self.type == TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID:
            return self.check_empty_platform_9_grid(scene_graph_tree)
        elif self.type == TaskType.MOVE_AROUND_OBJECT:
            return self.check_around_object(scene_graph_tree)
        elif self.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID:
            return self.check_object_freespace_9_grid(scene_graph_tree)
        elif self.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
            return self.check_middle_of_objects(scene_graph_tree)
        pass

    def __repr__(self):

        if self.type == TaskType.MOVE_TO_EMPTY_PLATFORM:
            return f"Move {self.item.name} to {self.destination.name}.\n\n"
        elif self.type == TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID:
            return f"Move {self.item.name} to {self.destination.name}'s {self.feature[0]} part\n\n"
        elif self.type == TaskType.MOVE_AROUND_OBJECT:
            return f"Move {self.item.name} around {self.feature[0]} \n\n"
        elif self.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID:
            return f"Move {self.item.name} to {self.feature[0]}'s {self.feature[1]} freespace\n\n"
        elif self.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
            return f"Move {self.item.name} between {self.feature[0]} and {self.feature[1]} \n\n"

    def __repr_rough__(self):

        rough_item_name = StringConvertor.get_noslash_name_wo_id(self.item.name)
        rough_destination_name = self.destination.get_name_for_interaction()
        if self.feature and len(self.feature) > 0:
            rough_feature0_name = StringConvertor.get_noslash_name_wo_id(
                self.feature[0]
            )
        if self.feature and len(self.feature) > 1:
            rough_feature1_name = StringConvertor.get_noslash_name_wo_id(
                self.feature[1]
            )
        if self.type == TaskType.MOVE_TO_EMPTY_PLATFORM:
            return f"Move {rough_item_name} to {rough_destination_name}."
        elif self.type == TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID:
            return f"Move {rough_item_name} to {rough_destination_name}'s {rough_feature0_name} part"
        elif self.type == TaskType.MOVE_AROUND_OBJECT:
            return f"Move {rough_item_name} around {rough_feature0_name}"
        elif self.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID:
            return f"Move {rough_item_name} to {rough_feature0_name}'s {rough_feature1_name} freespace"
        elif self.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
            return f"Move {rough_item_name} between {rough_feature0_name} and {rough_feature1_name}"

    def initial_state_information(self):
        item_platform_name = (
            self.item.get_bel_ground_platform().get_name_for_interaction()
        )
        destination_name = self.destination.get_name_for_interaction()
        rough_item_name = StringConvertor.get_noslash_name_wo_id(self.item.name)
        feature_0_name = (
            StringConvertor.get_noslash_name_wo_id(self.feature[0])
            if self.feature and len(self.feature) > 0
            else ""
        )
        feature_1_name = (
            StringConvertor.get_noslash_name_wo_id(self.feature[1])
            if self.feature and len(self.feature) > 1
            else ""
        )

        if self.type == TaskType.MOVE_TO_EMPTY_PLATFORM:
            return f"\nInitially, {rough_item_name} is on {item_platform_name}\n"
        elif self.type == TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID:
            return f"\nInitially, {rough_item_name} is on {item_platform_name}, and {destination_name} is empty.\n"
        elif self.type == TaskType.MOVE_AROUND_OBJECT:
            return f"\nInitially, {rough_item_name} is on {item_platform_name}, and {feature_0_name} is on {destination_name}.\n"
        elif self.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID:
            return f"\nInitially, {rough_item_name} is on {item_platform_name}, and {feature_0_name} is on {destination_name}.\n"
        elif self.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
            return f"\nInitially, {rough_item_name} is on {item_platform_name}, and {feature_0_name} is on {destination_name}, together with {feature_1_name}.\n"

    @staticmethod
    def generate_pic_description(
        pic_type="object",
        child_name_list=[],
        child_node=None,
        standing_direction=0,
    ):

        if pic_type == "occupied":
            child_info_list = [
                f"{i+1}: {child_name_list[i]}" for i in range(len(child_name_list))
            ]
            prompt = f"The picture(s) show the objects on the platform and their indices. They are:{child_info_list} respectively."
            pass
        elif pic_type == "empty":

            direction_list = NINE_DIRECTIONS[:-1]
            direction_list = (
                direction_list[-standing_direction:]
                + direction_list[:-standing_direction]
                + NINE_DIRECTIONS[-1:]
            )
            platform_region_info_list = [
                f"Region No. {i+1}: {direction_list[i]}" for i in range(9)
            ]
            prompt = f"The picture(s) show the platform, divided into 3x3 grid with each grid region has a number. They are:{platform_region_info_list} respectively."
            pass
        elif pic_type == "object":
            glog.info(f"standing_direction:{standing_direction}")
            probable_directions = [dir for dir in range(0, 8)]
            probable_directions = (
                probable_directions[-standing_direction:]
                + probable_directions[:-standing_direction]
            )
            dir_list = []
            for dir in probable_directions:
                if child_node.is_freespace_big_enough(dir):
                    i = len(dir_list) + 1
                    dir_list.append(f"freespace No. {i}: {NINE_DIRECTIONS[(dir ) % 8]}")
            prompt = f"The picture shows a focused view of object {child_node.get_name_for_interaction()} and its freespace. The freespace are: {dir_list} respectively."
            pass

        return prompt

    def generate_goal_information(
        self,
        scene_graph_tree,
        width=683,
        height=384,
        id=-1,
        model="ai2thor-claude-3-5-haiku",
        current_path=os.path.dirname(os.path.abspath(__file__)),
        detailed_pic_explanation=True,
    ):

        goal_action_list = []
        goal_explanation_list = []
        goal_picture_list = []
        source_platform = self.item.get_bel_ground_platform()
        source_platform_name = (
            self.item.get_bel_ground_platform().get_name_for_interaction()
        )
        source_object = scene_graph_tree.nodes[self.item.get_bel_ground_object().name]
        source_item_name = StringConvertor.get_noslash_name_wo_id(self.item.name)
        source_item_id = [
            id
            for id, _ in enumerate(self.item.get_bel_ground_platform().children)
            if _.name == self.item.name
        ][0] + 1

        destination_platform_name = self.destination.get_name_for_interaction()
        destination_object = scene_graph_tree.nodes[self.destination.bel_object]

        available_directions = [
            dir
            for dir in range(0, 8, 2)
            if source_platform.freespace_is_visible(dir)
            and source_object.freespace_is_standable(dir)
        ]
        standing_direction = (
            available_directions[0] if len(available_directions) > 0 else 0
        )

        goto_source_platform_action = f"goto_{source_platform_name}"
        goto_source_platform_explanation = (
            f"Go to {source_platform_name} where {source_item_name} is located."
        )

        goal_action_list.append(goto_source_platform_action)
        goal_explanation_list.append(goto_source_platform_explanation)
        goal_picture_list.append([])

        platform_img, platform_img_list = scene_graph_tree.auto_take_platform_picture(
            platform_name=source_platform.name,
            view="human_full",
            mark_object=True,
            mark_freespace=len(source_platform.children) == 0,
            standing_direction=standing_direction,
            width=width,
            height=height,
            focus_ratio=0.6,
            save_path=f"{current_path}/image4reflection/{model}/Task{id}_before_pickup_object.png",
        )
        n_platform_img_list = len(platform_img_list)
        image_name_list = [
            f"{current_path}/image4reflection/{model}/Task{id}_before_pickup_object_{(i+1)}_out_of_{n_platform_img_list}.png"
            for i in range(len(platform_img_list))
        ]

        pickup_source_item_action = (
            f"pick_up_object_{source_item_id}_of_current_platform"
        )
        pickup_source_item_explanation = f"Now you see single or multiple images of the current platform, they are {image_name_list}, showing you the items on the platform and their index in one or more view angles. We can recognize the number of {source_item_name} is {source_item_id},  and pick up {source_item_name} from the current platform."
        scene_graph_tree.remove_node(self.item.name)

        goal_action_list.append(pickup_source_item_action)
        goal_explanation_list.append(pickup_source_item_explanation)
        goal_picture_list.append(image_name_list)

        platform_img, platform_img_list = scene_graph_tree.auto_take_platform_picture(
            platform_name=source_platform.name,
            view="human_full",
            mark_object=True,
            mark_freespace=len(source_platform.children) == 0,
            standing_direction=standing_direction,
            width=width,
            height=height,
            focus_ratio=0.6,
            save_path=f"{current_path}/image4reflection/{model}/Task{id}_before_goto_destination.png",
        )
        n_platform_img_list = len(platform_img_list)
        image_name_list = [
            f"{current_path}/image4reflection/{model}/Task{id}_before_goto_destination_{(i+1)}_out_of_{n_platform_img_list}.png"
            for i in range(len(platform_img_list))
        ]
        goto_destination_platform_action = f"goto_{destination_platform_name}"
        goto_destination_platform_explanation = f"Now you have picked up object, you can see the scene {image_name_list}. Go to {destination_platform_name} to place {source_item_name}."

        goal_action_list.append(goto_destination_platform_action)
        goal_explanation_list.append(goto_destination_platform_explanation)
        goal_picture_list.append(image_name_list)

        available_directions = [
            dir
            for dir in range(0, 8, 2)
            if self.destination.freespace_is_visible(dir)
            and destination_object.freespace_is_standable(dir)
        ]
        first_standing_direction = available_directions[
            0
        ]  # if len(available_directions) > 1 else available_directions[0]
        first_standing_direction_id = 0  # if len(available_directions) > 1 else 0
        if len(available_directions) > 1 and destination_object.get_bbox_line_length(
            available_directions[0]
        ) < destination_object.get_bbox_line_length(available_directions[1]):
            first_standing_direction = available_directions[1]
            first_standing_direction_id = 1
            platform_img, platform_img_list = (
                scene_graph_tree.auto_take_platform_picture(
                    platform_name=self.destination.name,
                    view="human_full",
                    mark_object=True,
                    mark_freespace=len(self.destination.children) == 0,
                    standing_direction=available_directions[0],
                    width=width,
                    height=height,
                    focus_ratio=0.6,
                    save_path=f"{current_path}/image4reflection/{model}/Task{id}_before_rotate_view.png",
                )
            )
            n_platform_img_list = len(platform_img_list)
            image_name_list = [
                f"{current_path}/image4reflection/{model}/Task{id}_before_rotate_view_{(i+1)}_out_of_{n_platform_img_list}.png"
                for i in range(len(platform_img_list))
            ]
            goal_picture_list.append(image_name_list)
            goal_action_list.append(f"rotate_observation_view_of_current_platform")
            goal_explanation_list.append(
                f"Now you see single or multiple images of the current platform, they are {image_name_list}, showing you the items on the platform and their index in one or more view angles, and an image you've required in previous step, showing an object and its freespace with their indices."
                f"rotate your observation view angle to avoid a bad and could-be-good view angle. If the view angle is getting worse, just repeat this action for at most 4 times until you find a good view angle."
            )

        place_destination_platform_action = ""
        place_destination_platform_explanation = ""
        freespace_list = []

        platform_img, platform_img_list = scene_graph_tree.auto_take_platform_picture(
            platform_name=self.destination.name,
            view="human_full",
            mark_object=True,
            mark_freespace=len(self.destination.children) == 0,
            standing_direction=first_standing_direction,
            width=width,
            height=height,
            focus_ratio=0.6,
            save_path=f"{current_path}/image4reflection/{model}/Task{id}_before_place_object.png",
        )
        n_platform_img_list = len(platform_img_list)
        image_name_list = [
            f"{current_path}/image4reflection/{model}/Task{id}_before_place_object_{(i+1)}_out_of_{n_platform_img_list}.png"
            for i in range(len(platform_img_list))
        ]
        goal_picture_list.append(image_name_list)
        child_name_list = [
            child.get_name_for_interaction() for child in self.destination.children
        ]

        if self.type == TaskType.MOVE_TO_EMPTY_PLATFORM:
            place_destination_platform_action = "place_at_anywhere"
            place_destination_platform_explanation = f"Now you see single or multiple images of the current platform, showing you the platform, divided into 3x3 grid with each grid has a number, in one or more view angles. The task only asks you to put the object on the destination platform, so just select all the regions of the platform."

        elif self.type == TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID:

            direction_id = (
                (NINE_DIRECTIONS.index(self.feature[0]) + first_standing_direction) % 8
                if self.feature[0] != "center"
                else 8
            )

            place_destination_platform_action = f"place_at_freespace_[{direction_id}]"

            cardinal_extend_direction_list = [direction_id, 8]

            diagonal_extend_direction_list = [
                direction_id,
                (direction_id + 1) % 8,
                (direction_id + 7) % 8,
                8,
            ]
            cardinal_extend_more_direction_list = [
                direction_id,
                (direction_id + 1) % 8,
                (direction_id + 7) % 8,
                (direction_id + 2) % 8,
                (direction_id + 6) % 8,
                8,
            ]

            direction_id += 1

            cardinal_extend_direction_list = [
                dir + 1 for dir in cardinal_extend_direction_list
            ]

            diggonal_extend_direction_list = [
                dir + 1 for dir in diagonal_extend_direction_list
            ]
            cardinal_extend_more_direction_list = [
                dir + 1 for dir in cardinal_extend_more_direction_list
            ]

            empty_platform_prompt = Task.generate_pic_description(
                self,
                pic_type="empty",
                child_name_list=[],
                child_node=None,
                standing_direction=first_standing_direction,
            )
            if direction_id in [1, 3, 5, 7]:
                place_destination_platform_action = place_destination_platform_action
                place_destination_platform_explanation = f"Now you see single or multiple images of the current platform, showing you the platform, divided into 3x3 grid with each grid has a number, in one or more view angles.{empty_platform_prompt} the task asks you to put at {self.feature[0]}, {direction_id} corresponds to that direction according to the picture; in case the free space cannot afford this item, try place_at_freespace_{cardinal_extend_direction_list}  or place_at_freespace_{cardinal_extend_more_direction_list} as  the region {direction_id} is adjacent to them according to the picture in 8-directional space."
            elif direction_id in [2, 4, 6, 8]:
                place_destination_platform_action = place_destination_platform_action
                place_destination_platform_explanation = f"Now you see single or multiple images of the current platform, showing you the platform, divided into 3x3 grid with each grid has a number, in one or more view angles.{empty_platform_prompt} the task asks you to put at {self.feature[0]}, {direction_id} corresponds to that direction according to the picture; in case the free space cannot afford this item, try  place_at_freespace_{diggonal_extend_direction_list} as  the region {direction_id} is adjacent to them according to the picture in 8-directional space."
            else:
                place_destination_platform_action = place_destination_platform_action
                place_destination_platform_explanation = f"Now you see single or multiple images of the current platform, showing you the platform, divided into 3x3 grid with each grid has a number, in one or more view angles.{empty_platform_prompt} the task asks you to put at {self.feature[0]}, {direction_id} corresponds to that direction according to the picture; in case the free space cannot afford this item, try  place_at_anywhere."

        elif self.type == TaskType.MOVE_AROUND_OBJECT:
            if not self.feature or len(self.feature) == 0:
                glog.error("MOVE_AROUND_OBJECT task without feature is invalid.")
                return [], []

            dest_item_node = scene_graph_tree.nodes[self.feature[0]]
            dest_item_name = StringConvertor.get_noslash_name_wo_id(self.feature[0])
            freespace_num = dest_item_node.get_num_of_critical_space()
            dest_item_id = [
                id
                for id, _ in enumerate(self.destination.children)
                if _.name == self.feature[0]
            ][0] + 1

            freespace_list = [(dest_item_id, i) for i in range(1, freespace_num + 1)]

            occupied_platform_prompt = Task.generate_pic_description(
                self,
                pic_type="occupied",
                child_name_list=child_name_list,
                child_node=None,
                standing_direction=first_standing_direction,
            )
            goal_action_list.append(f"show_freespace_of_object_{dest_item_id}")
            goal_explanation_list.append(
                f"Now you see single or multiple images of the current platform, they are {image_name_list},  showing you the items on the platform and their index in one or more view angles. {occupied_platform_prompt}"
                f"Because we want to place item near {dest_item_name} , we check the freespace of {dest_item_name} (or the number you think correct) to see if you've recognized the correct item and where is available for placing the object. Repeat if you find your recognition is wrong or you need to check the freespace of other objects."
            )

            # next step's picture
            object_to_show = scene_graph_tree.nodes[self.feature[0]]
            img = object_to_show.auto_take_non_ground_object_picture(
                scene=scene_graph_tree.corresponding_scene,
                view="human_focus",
                mark_object=False,
                only_mark_itself=False,
                mark_freespace=True,
                diagonal_mode="old",
                need_afford_rect=None,
                standing_direction=standing_direction,
                width=width,
                height=height,
                focus_ratio=0.5,
                fovy_range=[np.deg2rad(40), np.deg2rad(60)],
                save_path=f"{current_path}/image4reflection/{model}/Task{id}_showfreespace.png",
            )
            image_name_list = [
                f"{current_path}/image4reflection/{model}/Task{id}_showfreespace.png"
            ]
            image_name_list += goal_picture_list[-1]
            goal_picture_list.append(image_name_list)

            place_destination_platform_action = f"place_at_freespace_{freespace_list}"
            place_destination_platform_explanation = f"Now you see single or multiple images of the current platform, showing you the items on the platform and their index in one or more view angles, they are {image_name_list[:-1]}; and an image you've required in previous step, showing an object and its freespace with their indices, it is {image_name_list[-1]}."
            place_destination_platform_explanation += f"recognize the number of {dest_item_name} is {dest_item_id} on that platform, and then place {source_item_name} around {dest_item_name} at any of the available freespace. If you can't recognize at once, use 'show_freespace_of_object_x' to check xth object and its freespace to see if x is the number you're looking for;  In case the free space cannot afford this item, try combining freespace of other objects, "
            place_destination_platform_explanation += f"or use 'rotate_observation_view_of_current_platform' to change your view angle."

        elif self.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID:
            if not self.feature or len(self.feature) < 2:
                glog.error(
                    "MOVE_TO_OBJECT_FREESPACE_9_GRID task without feature is invalid."
                )
                return [], []

            dest_item_node = scene_graph_tree.nodes[self.feature[0]]
            dest_item_name = StringConvertor.get_noslash_name_wo_id(self.feature[0])
            dest_item_id = [
                id
                for id, _ in enumerate(self.destination.children)
                if _.name == self.feature[0]
            ][0] + 1
            direction_id = (
                NINE_DIRECTIONS.index(self.feature[1]) + first_standing_direction
            ) % 8
            direction_id = dest_item_node.get_on_picture_freespace_id(direction_id)

            for i in range(4):
                if direction_id != -1:
                    break
                platform_img, platform_img_list = (
                    scene_graph_tree.auto_take_platform_picture(
                        platform_name=self.destination.name,
                        view="human_full",
                        mark_object=True,
                        mark_freespace=len(self.destination.children) == 0,
                        standing_direction=available_directions[
                            first_standing_direction_id
                        ],
                        width=width,
                        height=height,
                        focus_ratio=0.6,
                        save_path=f"{current_path}/image4reflection/{model}/Task{id}_rotate_view.png",
                    )
                )
                n_platform_img_list = len(platform_img_list)
                image_name_list = [
                    f"{current_path}/image4reflection/{model}/Task{id}_rotate_view_{available_directions[first_standing_direction_id]}_{(i+1)}_out_of_{n_platform_img_list}.png"
                    for i in range(len(platform_img_list))
                ]

                first_standing_direction_id = (first_standing_direction_id + 1) % len(
                    available_directions
                )
                first_standing_direction = available_directions[
                    first_standing_direction_id
                ]
                direction_id = (
                    NINE_DIRECTIONS.index(self.feature[1]) + first_standing_direction
                ) % 8
                direction_id = dest_item_node.get_on_picture_freespace_id(direction_id)
                goal_picture_list.append(image_name_list)
                goal_action_list.append(f"rotate_observation_view_of_current_platform")
                goal_explanation_list.append(
                    f"Now you see single or multiple images of the current platform, they are {image_name_list},  showing you the items on the platform and their index in one or more view angles."
                    f"rotate your observation view angle to avoid a bad and could-be-good view angle. If the view angle is getting worse, just repeat this action for at most 4 times until you find a good view angle."
                )
            if direction_id == -1:
                direction_id = 2

            freespace_list = [(dest_item_id, direction_id)]

            occupied_platform_prompt = Task.generate_pic_description(
                self,
                pic_type="occupied",
                child_name_list=child_name_list,
                child_node=None,
                standing_direction=first_standing_direction,
            )
            goal_action_list.append(f"show_freespace_of_object_{dest_item_id}")
            goal_explanation_list.append(
                f"Now you see single or multiple images of the current platform, showing you the items on the platform and their index in one or more view angles. they are {image_name_list}. {occupied_platform_prompt}"
                f"Because we want to place item near {dest_item_name} , we check the freespace of {dest_item_name} (or the number you think correct) to see if you've recognized the correct item and where is available for placing the object. Repeat if you find your recognition is wrong or you need to check the freespace of other objects."
            )

            object_to_show = scene_graph_tree.nodes[self.feature[0]]
            img = object_to_show.auto_take_non_ground_object_picture(
                scene=scene_graph_tree.corresponding_scene,
                view="human_focus",
                mark_object=False,
                only_mark_itself=False,
                mark_freespace=True,
                diagonal_mode="old",
                need_afford_rect=None,
                standing_direction=first_standing_direction,
                width=width,
                height=height,
                focus_ratio=0.5,
                fovy_range=[np.deg2rad(40), np.deg2rad(60)],
                save_path=f"{current_path}/image4reflection/{model}/Task{id}_show_freespace.png",
            )
            image_name_list = [
                f"{current_path}/image4reflection/{model}/Task{id}_show_freespace.png"
            ]
            image_name_list += goal_picture_list[-1]
            goal_picture_list.append(image_name_list)

            object_prompt = Task.generate_pic_description(
                self,
                pic_type="object",
                child_name_list=[],
                child_node=dest_item_node,
                standing_direction=first_standing_direction,
            )
            place_destination_platform_action = f"place_at_freespace_{freespace_list}"
            place_destination_platform_explanation = f"Now you see single or multiple images of the current platform, they are {image_name_list}, showing you the items on the platform and their index in one or more view angles, and an image you've required in previous step, showing an object and its freespace with their indices. {object_prompt}"
            place_destination_platform_explanation += f"recognize the number of {dest_item_name} is {dest_item_id} on that platform and the number of freespace {self.feature[1]} is {direction_id}, and then place {source_item_name} at the freespace {direction_id} of {dest_item_name}. If you can't recognize at once, use 'show_freespace_of_object_x' to check xth object and its freespace to see if x is the number you're looking for;  In case the free space cannot afford this item, try combining freespace of other objects,  or use 'rotate_observation_view_of_current_platform' to change your view angle."
            """
            dir_id = (
                self.platform_list[self.at_place]
                .children[child_id]
                .get_on_picture_freespace_id(freespace_pair_list[i][1])
            )
            """
        elif self.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
            dest_item_node_a = scene_graph_tree.nodes[self.feature[0]]
            dest_item_node_b = scene_graph_tree.nodes[self.feature[1]]

            dest_item_a_name = StringConvertor.get_noslash_name_wo_id(self.feature[0])
            dest_item_b_name = StringConvertor.get_noslash_name_wo_id(self.feature[1])

            select_item_id_a = [
                id
                for id, _ in enumerate(self.destination.children)
                if _.name == self.feature[0]
            ][0] + 1
            select_item_id_b = [
                id
                for id, _ in enumerate(self.destination.children)
                if _.name == self.feature[1]
            ][0] + 1

            ba_dir = []
            ab_dir = []
            while dest_item_node_a.depth > 2:
                dest_item_node_a = dest_item_node_a.parent
            while dest_item_node_b.depth > 2:
                dest_item_node_b = dest_item_node_b.parent

            for dir in range(8):
                if dest_item_node_b in dest_item_node_a.free_space[dir]["Objects"]:
                    ba_dir.append(dir)
                if dest_item_node_a in dest_item_node_b.free_space[dir]["Objects"]:
                    ab_dir.append(dir)

            ba_dir = [
                dest_item_node_a.get_on_picture_freespace_id(dir) for dir in ba_dir
            ]
            ab_dir = [
                dest_item_node_b.get_on_picture_freespace_id(dir) for dir in ab_dir
            ]

            freespace_list = [
                (select_item_id_a, dir) for dir in ba_dir if dir != -1
            ] + [(select_item_id_b, dir) for dir in ab_dir if dir != -1]

            occupied_platform_prompt = Task.generate_pic_description(
                self,
                pic_type="occupied",
                child_name_list=child_name_list,
                child_node=None,
                standing_direction=first_standing_direction,
            )
            goal_action_list.append(f"show_freespace_of_object_{select_item_id_a}")
            goal_explanation_list.append(
                f"Now you see single or multiple images of the current platform, showing you the items on the platform and their index in one or more view angles, they are {image_name_list}. {occupied_platform_prompt}"
                f"Because we want to place item near {dest_item_a_name} , we check the freespace of {dest_item_a_name} (or the number you think correct) to see if you've recognized the correct item and where is available for placing the object. Repeat if you find your recognition is wrong or you need to check the freespace of other objects."
            )

            object_to_show = scene_graph_tree.nodes[self.feature[0]]
            img = object_to_show.auto_take_non_ground_object_picture(
                scene=scene_graph_tree.corresponding_scene,
                view="human_focus",
                mark_object=False,
                only_mark_itself=False,
                mark_freespace=True,
                diagonal_mode="old",
                need_afford_rect=None,
                standing_direction=first_standing_direction,
                width=width,
                height=height,
                focus_ratio=0.5,
                fovy_range=[np.deg2rad(40), np.deg2rad(60)],
                save_path=f"{current_path}/image4reflection/{model}/Task{id}_showfreespace_a.png",
            )
            image_name_list = [
                f"{current_path}/image4reflection/{model}/Task{id}_showfreespace_a.png"
            ]
            image_name_list += goal_picture_list[-1]
            goal_picture_list.append(image_name_list)

            object_prompt = Task.generate_pic_description(
                self,
                pic_type="object",
                child_name_list=[],
                child_node=dest_item_node_a,
                standing_direction=first_standing_direction,
            )
            goal_action_list.append(f"show_freespace_of_object_{select_item_id_b}")
            goal_explanation_list.append(
                f"Now you see single or multiple images of the current platform, showing you the items on the platform and their index in one or more view angles, they are {image_name_list[:-1]}; and an image you've required in previous step, showing an object and its freespace with their indices, it is {image_name_list [-1]}. {object_prompt}"
                f"Because we also want to place item near {dest_item_b_name} , we should further check the freespace of {dest_item_b_name} (or the number you think correct) to see if you've recognized the correct item and where is available for placing the object. Repeat if you find your recognition is wrong or you need to check the freespace of other objects."
            )

            object_to_show = scene_graph_tree.nodes[self.feature[1]]
            img = object_to_show.auto_take_non_ground_object_picture(
                scene=scene_graph_tree.corresponding_scene,
                view="human_focus",
                mark_object=False,
                only_mark_itself=False,
                mark_freespace=True,
                diagonal_mode="old",
                need_afford_rect=None,
                standing_direction=first_standing_direction,
                width=width,
                height=height,
                focus_ratio=0.5,
                fovy_range=[np.deg2rad(40), np.deg2rad(60)],
                save_path=f"{current_path}/image4reflection/{model}/Task{id}_showfreespace_b.png",
            )
            image_name_list = [
                f"{current_path}/image4reflection/{model}/Task{id}_showfreespace_b.png"
            ]
            image_name_list += goal_picture_list[-1]
            goal_picture_list.append(image_name_list)

            object_prompt = Task.generate_pic_description(
                self,
                pic_type="object",
                child_name_list=[],
                child_node=dest_item_node_b,
                standing_direction=first_standing_direction,
            )
            place_destination_platform_action = f"place_at_freespace_{freespace_list}"
            place_destination_platform_explanation = f"Now you see single or multiple images of the current platform, showing you the items on the platform and their index in one or more view angles, and an image you've required in previous step, showing an object and its freespace with their indices. the pictures you can see are {image_name_list[-1]}. {object_prompt} Note that you can only see one image at a time, so please memorize the last image you've seen."
            place_destination_platform_explanation += f"we have recognized the number of {dest_item_a_name} is {select_item_id_a} and the number of {dest_item_b_name} is {select_item_id_b} on that platform, and the indices of the freespace between them are {freespace_list}, so we place {source_item_name} at the freespace {freespace_list}. If you can't recognize at once, use 'show_freespace_of_object_x' to check xth object and its freespace to see if x is the number you're looking for;  In case the free space cannot afford this item, try combining freespace of other objects,  or use 'rotate_observation_view_of_current_platform' to change your view angle."

        goal_action_list.append(place_destination_platform_action)
        goal_explanation_list.append(place_destination_platform_explanation)

        return goal_action_list, goal_explanation_list, goal_picture_list

        pass


class TaskGeneration:
    MIN_PLATFORM_LENGTH = 0.03
    direction_mapping = {
        (0, 0): "rear-left",
        (0, 1): "left",
        (0, 2): "front-left",
        (1, 0): "rear",
        (1, 1): "center",
        (1, 2): "front",
        (2, 0): "rear-right",
        (2, 1): "right",
        (2, 2): "front-right",
    }
    diagonal_direction_mapping = {
        (0, 0): "rear-left",
        (0, 2): "front-left",
        (2, 0): "rear-right",
        (2, 2): "front-right",
    }
    horizontal_direction_mapping = {
        (0, 1): "left",
        (1, 2): "front",
    }
    vertical_direction_mapping = {
        (0, 1): "rear",
        (2, 1): "front",
    }

    def __init__(self, scene_graph_tree=None, items=[], places=[]):
        self.scene_graph_tree = scene_graph_tree
        self.tasks = []
        self.place_for_index = {}
        scene_graph_tree.update_platform_children()
        for node in scene_graph_tree.get_ground_object_list():
            for platform in node.own_platform:
                self.place_for_index[platform.name] = platform

    def parse_from_file(self, tree_file=None, node_details_file=None):
        if tree_file:
            self.scene_graph_tree.parse_tree_from_file(tree_file)
        if node_details_file:
            self.scene_graph_tree.parse_node_details_from_file(node_details_file)

    def generate_task_from_scene_graph_tree(self, root_node=None):

        item_node_list = self.scene_graph_tree.get_non_ground_object_list()
        place_node_list = self.scene_graph_tree.get_ground_object_list()
        object_task_info_dict = {}
        glog.info(
            len(item_node_list)
            * np.sum([len(place_node.own_platform) for place_node in place_node_list])
        )
        ts = time.perf_counter()
        task_set = set()
        for item_node in item_node_list:
            glog.info(f"Generating tasks for {item_node.name}")

            for place_node in place_node_list:
                platform_list = [
                    platform
                    for platform in place_node.own_platform
                    if any(
                        [platform.freespace_is_visible(dir) for dir in range(0, 8, 2)]
                    )
                    and any(
                        [
                            len(platform.standing_point_list[dir // 2]) > 0
                            for dir in range(0, 8, 2)
                        ]
                    )
                    and (platform.avl_height > 0.05 or len(platform.children) > 0)
                ]
                # glog.info([platform.name for platform in platform_list])
                for place_platform in platform_list:

                    if (item_node.top - item_node.bottom) > place_platform.avl_height:
                        continue
                    available_directions = [
                        dir
                        for dir in range(0, 8, 2)
                        if len(place_platform.standing_point_list[dir // 2]) > 0
                        and place_platform.freespace_is_visible(dir)
                    ]
                    if len(available_directions) == 0 and place_platform.name not in [
                        "objects/RoboTHOR_hemnes_day_bed_0",
                        "objects/RoboTHOR_hemnes_day_bed_1",
                        "objects/RoboTHOR_hemnes_day_bed_2",
                    ]:
                        continue
                    available_directions = (
                        available_directions[:1] if len(available_directions) else [0]
                    )

                    (
                        result,
                        put_on_platform,
                        put_around_object_list,
                        put_on_object_dir_list,
                        put_between_2_object_list,
                        put_around_object_dict,
                        put_on_object_dir_dict,
                        put_between_2_object_dict,
                    ) = place_platform.get_fitable_tasks_any_number(
                        item_node, all_standing_directions=available_directions
                    )

                    if result == "non_empty":
                        if put_on_platform:
                            if (
                                put_around_object_list == []
                                and put_on_object_dir_list == []
                                and put_between_2_object_list == []
                            ):
                                continue
                            new_empty_platform_task = Task(
                                item=item_node,
                                destination=place_platform,
                                type=TaskType.MOVE_TO_EMPTY_PLATFORM,
                                feature=None,
                                goal_translation=list(put_on_object_dir_dict.values())[
                                    0
                                ],
                                need_merge_freespace=not place_platform.single_freespace_available(
                                    item_node, single_type="all"
                                ),
                            )
                            self.tasks.append(new_empty_platform_task)

                            for put_around_object_info in put_around_object_list:

                                new_object_around_task = Task(
                                    item=item_node,
                                    destination=place_platform,
                                    type=TaskType.MOVE_AROUND_OBJECT,
                                    feature=[put_around_object_info],
                                    goal_translation=put_around_object_dict[
                                        put_around_object_info
                                    ],
                                    need_merge_freespace=not place_platform.single_freespace_available(
                                        item_node, single_type=put_around_object_info
                                    ),
                                )
                                self.tasks.append(new_object_around_task)
                                pass
                            for put_on_object_dir_info in put_on_object_dir_list:
                                new_object_freespace_task = Task(
                                    item=item_node,
                                    destination=place_platform,
                                    type=TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID,
                                    feature=[
                                        put_on_object_dir_info[0],
                                        put_on_object_dir_info[1],
                                    ],
                                    goal_translation=put_on_object_dir_dict[
                                        put_on_object_dir_info
                                    ],
                                    need_merge_freespace=not place_platform.single_freespace_available(
                                        item_node,
                                        single_type=[
                                            put_on_object_dir_info[0],
                                            (
                                                NINE_DIRECTIONS.index(
                                                    put_on_object_dir_info[1]
                                                )
                                                + available_directions[0]
                                            )
                                            % 8,
                                        ],
                                    ),
                                )
                                self.tasks.append(new_object_freespace_task)
                                pass
                            for put_between_2_object_info in put_between_2_object_list:
                                new_object_between_task = Task(
                                    item=item_node,
                                    destination=place_platform,
                                    type=TaskType.MOVE_TO_MIDDLE_OF_OBJECTS,
                                    feature=[
                                        put_between_2_object_info[0],
                                        put_between_2_object_info[1],
                                    ],
                                    goal_translation=put_between_2_object_dict[
                                        put_between_2_object_info
                                    ],
                                    need_merge_freespace=not place_platform.single_freespace_available(
                                        item_node, single_type=put_between_2_object_info
                                    ),
                                )
                                self.tasks.append(new_object_between_task)

                            pass
                        pass
                    else:
                        place_platform_convex_hull_2d = place_platform.convex_hull_2d
                        place_platform_bounding_box = (
                            place_platform_convex_hull_2d.get_headed_bbox_instance()
                        )

                        if "wholetable" in result:
                            new_empty_platform_task = Task(
                                item=item_node,
                                destination=place_platform,
                                type=TaskType.MOVE_TO_EMPTY_PLATFORM,
                                feature=None,
                                goal_translation=place_platform.get_dir_point(
                                    dir="center", item_node=item_node
                                ),
                                need_merge_freespace=not place_platform.single_freespace_available(
                                    item_node, single_type="empty"
                                ),
                            )
                            self.tasks.append(new_empty_platform_task)
                        if "center" in result:
                            for i in range(9):
                                new_empty_platform_task = Task(
                                    item=item_node,
                                    destination=place_platform,
                                    type=TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID,
                                    feature=[NINE_DIRECTIONS[i]],
                                    goal_translation=place_platform.get_dir_point(
                                        dir=(
                                            NINE_DIRECTIONS[
                                                (i + available_directions[0]) % 8
                                            ]
                                            if i < 8
                                            else "center"
                                        ),
                                        item_node=item_node,
                                    ),
                                    need_merge_freespace=not place_platform.single_freespace_available(
                                        item_node, single_type="empty"
                                    ),
                                )
                                self.tasks.append(new_empty_platform_task)
                        # elif "diagonal" in result:
                        #     for i in range(1, 8, 2):
                        #         new_empty_platform_task = Task(
                        #             item=item_node,
                        #             destination=place_platform,
                        #             type=TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID,
                        #             feature=[NINE_DIRECTIONS[i]],
                        #         )
                        #         self.tasks.append(new_empty_platform_task)
                        # elif "horizontal" in result:
                        #     for i in range(2, 8, 4):
                        #         new_empty_platform_task = Task(
                        #             item=item_node,
                        #             destination=place_platform,
                        #             type=TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID,
                        #             feature=[NINE_DIRECTIONS[i]],
                        #         )
                        #         self.tasks.append(new_empty_platform_task)
                        # elif "vertical" in result:
                        #     for i in range(0, 7, 4):
                        #         new_empty_platform_task = Task(
                        #             item=item_node,
                        #             destination=place_platform,
                        #             type=TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID,
                        #             feature=[NINE_DIRECTIONS[i]],
                        #         )
                        #         self.tasks.append(new_empty_platform_task)
                        pass

                    pass
        glog.info(
            f"Task generation finished, {len(self.tasks)} tasks generated, lasts {time.perf_counter() - ts} seconds"
        )
        return self.tasks


def main():
    pass


# scene_graph_tree = SceneGraphTree()
# scene_graph_tree.parse_tree_from_file()
# scene_graph_tree.print_tree(scene_graph_tree.root)
# scene_graph_tree.parse_node_details_from_file()

# atomic_motions = TaskGeneration(scene_graph_tree)
# atomic_motions.generate_items_and_places()
# $atomic_motions.generate_tasks()

if __name__ == "__main__":
    main()
    pass
