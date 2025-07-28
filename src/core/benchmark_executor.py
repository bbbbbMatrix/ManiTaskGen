from operator import is_
import numpy as np
from enum import Enum
from src.geometry.convex_hull_processor import ConvexHullProcessor_2d
from src.geometry.placement_helper import PlacementHelper

from . import process_based_task_generation
from src.vlm_interaction.interact_prompt_helper import (
    StatePromptManager,
    HintPromptManager,
    ReflectionPromptManager,
)
from src.core.process_based_task_generation import TaskType
from src.utils.config_manager import (
    get_benchmark_executor_config,
    get_task_interaction_config,
)

import copy
import glog
import json
from abc import ABC, abstractmethod
import re

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


class InteractStates(Enum):
    DEFAULT = -1
    NAVIGATION = 0
    IDLE = 1
    HOLDING_OCCUPIED_PLATFORM = 2
    HOLDING_EMPTY_PLATFORM = 3
    CALL_END = -2

    @classmethod
    def from_string(cls, state_str):
        state_str = state_str.upper()
        try:
            return cls[state_str]
        except KeyError:
            raise ValueError(f"Invalid InteractState string: {state_str}")

    @classmethod
    def to_string(cls, state):
        if isinstance(state, cls):
            return state.name.lower()
        else:
            raise ValueError(f"Invalid InteractState: {state}")


class ActionType(Enum):
    INVALID = -1
    CALL_END = 0
    NOT_AVAILABLE = 1
    PLACE_IDLE = 2
    PLACE_INVALID = 3
    GOTO_PLATFORM = 100
    PICKUP_OBJECT = 200
    PLACE_ANYWHERE = 300
    PLACE_EMPTY = 301
    PLACE_OCCUPIED = 302

    ROTATE = 400
    SHOW_OBJECT = 500


class InteractSystemState(ABC):
    state_prompt_helper = StatePromptManager(
        prompt_templates="vlm_interactor/prompts/task_interact_prompts.json"
    )

    def __init__(self, state_type):
        self.state = state_type

    @abstractmethod
    def get_action_space(self):
        pass

    @staticmethod
    def validate_action(action_str) -> ActionType:

        if not isinstance(action_str, str) or action_str is None:
            return (ActionType.INVALID,)
        if action_str.startswith("`") and action_str.endswith("`"):
            action_str = action_str[1:-1]
        if action_str.startswith('"') and action_str.endswith('"'):
            action_str = action_str[1:-1]
        if action_str.startswith("'") and action_str.endswith("'"):
            action_str = action_str[1:-1]

        if action_str.endswith("_of_current_platform") or action_str.endswith(
            "_on_current_platform"
        ):
            action_str = action_str[: -len("_of_current_platform")]
        if action_str.upper() == "CALL_END":
            return ActionType.CALL_END, 0
        elif action_str == "rotate_observation_view":
            return ActionType.ROTATE, 0
        elif action_str == "place_at_anywhere":
            return ActionType.PLACE_ANYWHERE, 0

        elif action_str.startswith("goto_"):
            try:
                platform_name = (action_str[len("goto_") :]).strip()
                return ActionType.GOTO_PLATFORM, platform_name
            except:
                return ActionType.INVALID, 0
        elif action_str.startswith("pick_up_object_"):
            try:
                object_num = int(action_str[len("pick_up_object_") :].strip())
                if object_num > 0:
                    return ActionType.PICKUP_OBJECT, object_num
            except:
                return ActionType.INVALID, 0
        elif action_str.startswith("show_freespace_of_object_"):
            try:
                object_num = int(action_str[len("show_freespace_of_object_") :].strip())
                if object_num > 0:
                    return ActionType.SHOW_OBJECT, object_num
            except:
                return ActionType.INVALID, 0
        elif action_str.startswith("place_at_freespace_["):
            try:
                action_str = action_str[len("place_at_freespace_") :]
                action_list = eval(action_str)
                if isinstance(action_list, list):
                    all_numbers = all(
                        isinstance(action, int) and 1 <= action <= 9
                        for action in action_list
                    )
                    all_tuples = all(
                        isinstance(action, tuple) for action in action_list
                    )
                    # import ipdb
                    # ipdb.set_trace()
                    if all_numbers:
                        return ActionType.PLACE_EMPTY, action_list
                    elif all_tuples:
                        return ActionType.PLACE_OCCUPIED, action_list
                    else:
                        return ActionType.INVALID, 0
                else:
                    return ActionType.INVALID, 0
            except:
                return ActionType.INVALID, 0

        return ActionType.INVALID, 0
        pass

    @abstractmethod
    def validate_action_details(action_str):
        pass

    @abstractmethod
    def generate_prompt(context: dict):
        pass


class NavigationState(InteractSystemState):
    action_space_type = [ActionType.CALL_END, ActionType.GOTO_PLATFORM]

    def __init__(self):
        super().__init__(InteractStates.NAVIGATION)

    @staticmethod
    def validate_action_details(action_str, platform_name_list):

        action_type, action_param = NavigationState.validate_action(action_str)
        glog.info(f"Action type: {action_type}, Action param: {action_param}")
        if action_type not in NavigationState.action_space_type:
            return (
                ActionType.NOT_AVAILABLE
                if action_type != ActionType.INVALID
                else ActionType.INVALID
            ), 0

        if action_type == ActionType.CALL_END:
            return action_type, 0
        elif action_type == ActionType.GOTO_PLATFORM:
            if action_param in platform_name_list:
                return action_type, action_param

        return ActionType.INVALID, 0

    @staticmethod
    def generate_prompt(
        scene_description,
        platform_list,
        task_description,
        steps_used,
        total_steps,
        location_action_list,
    ):
        prompt = InteractSystemState.state_prompt_helper.generate_state_prompts(
            state=InteractStates.NAVIGATION.name.lower(),
            context={
                "scene_description": "",
                "platform_list": platform_list,
                "task_description": task_description,
                "steps_used": steps_used,
                "total_steps": total_steps,
                "location_action_list": location_action_list,
            },
        )
        return prompt


class IdleState(InteractSystemState):
    action_space_type = [
        ActionType.ROTATE,
        ActionType.CALL_END,
        ActionType.GOTO_PLATFORM,
        ActionType.PICKUP_OBJECT,
        ActionType.SHOW_OBJECT,
    ]

    def __init__(self):
        super().__init__(InteractStates.IDLE)

    @staticmethod
    def validate_action_details(action_str, platform_name_list, object_num):
        action_type, action_param = IdleState.validate_action(action_str)
        if action_type not in IdleState.action_space_type:
            return (
                ActionType.NOT_AVAILABLE
                if action_type != ActionType.INVALID
                else ActionType.INVALID
            ), 0
        if action_type == ActionType.CALL_END or action_type == ActionType.ROTATE:
            return action_type, 0
        elif action_type == ActionType.GOTO_PLATFORM:
            if action_param in platform_name_list:
                return action_type, action_param
        elif (
            action_type == ActionType.PICKUP_OBJECT
            or action_type == ActionType.SHOW_OBJECT
        ):
            if action_param > 0 and action_param <= object_num:
                return action_type, action_param

        return ActionType.INVALID, 0

    @staticmethod
    def generate_prompt(
        scene_description,
        platform_list,
        task_description,
        steps_used,
        total_steps,
        platform_name,
        holding_object,
        image_type,
        image_info_list,
        location_action_list,
        object_action_list,
        show_freespace_of_object_action_list,
        n_image,
        image_name_list,
    ):
        prompt = InteractSystemState.state_prompt_helper.generate_state_prompts(
            state=InteractStates.IDLE.name.lower(),
            context={
                "scene_description": scene_description,
                "platform_list": platform_list,
                "task_description": task_description,
                "steps_used": steps_used,
                "total_steps": total_steps,
                "platform_name": platform_name,
                "holding_object": holding_object,
                "image_type": image_type,
                "image_info_list": image_info_list,
                "location_action_list": location_action_list,
                "object_action_list": object_action_list,
                "show_freespace_of_object_action_list": show_freespace_of_object_action_list,
                "n_image": n_image,
                "image_name_list": image_name_list,
            },
        )
        return prompt


class HoldingEmptyPlatformState(InteractSystemState):
    action_space_type = [
        ActionType.ROTATE,
        ActionType.CALL_END,
        ActionType.GOTO_PLATFORM,
        ActionType.PLACE_EMPTY,
        ActionType.PLACE_ANYWHERE,
        ActionType.SHOW_OBJECT,
    ]

    def __init__(self):
        super().__init__(InteractStates.HOLDING_EMPTY_PLATFORM)

    @staticmethod
    def validate_action_details(action_str, platform_name_list, object_num):
        action_type, action_param = HoldingEmptyPlatformState.validate_action(
            action_str
        )
        if action_type not in HoldingEmptyPlatformState.action_space_type:
            return (
                ActionType.NOT_AVAILABLE
                if action_type != ActionType.INVALID
                else ActionType.INVALID
            ), 0
        if (
            action_type == ActionType.CALL_END
            or action_type == ActionType.ROTATE
            or action_type == ActionType.PLACE_ANYWHERE
        ):
            return action_type, 0
        elif action_type == ActionType.PLACE_EMPTY:
            if isinstance(action_param, list):
                return action_type, action_param

        elif action_type == ActionType.GOTO_PLATFORM:
            if action_param in platform_name_list:
                return action_type, action_param
        elif action_type == ActionType.SHOW_OBJECT:
            if action_param > 0 and action_param <= object_num:
                return action_type, action_param
        return ActionType.INVALID, 0

    @staticmethod
    def generate_prompt(
        scene_description,
        platform_list,
        task_description,
        steps_used,
        total_steps,
        platform_name,
        holding_object,
        image_type,
        image_info_list,
        location_action_list,
        show_freespace_of_object_action_list,
        n_image,
        image_name_list,
    ):
        prompt = InteractSystemState.state_prompt_helper.generate_state_prompts(
            state=InteractStates.HOLDING_EMPTY_PLATFORM.name.lower(),
            context={
                "scene_description": scene_description,
                "platform_list": platform_list,
                "task_description": task_description,
                "steps_used": steps_used,
                "total_steps": total_steps,
                "platform_name": platform_name,
                "holding_object": holding_object,
                "image_type": image_type,
                "image_info_list": image_info_list,
                "location_action_list": location_action_list,
                "show_freespace_of_object_action_list": show_freespace_of_object_action_list,
                "object_action_list": [],
                "n_image": n_image,
                "image_name_list": image_name_list,
            },
        )
        return prompt


class HoldingOccupiedPlatformState(InteractSystemState):
    action_space_type = [
        ActionType.ROTATE,
        ActionType.CALL_END,
        ActionType.GOTO_PLATFORM,
        ActionType.PLACE_OCCUPIED,
        ActionType.PLACE_ANYWHERE,
        ActionType.SHOW_OBJECT,
    ]

    def __init__(self):
        super().__init__(InteractStates.HOLDING_OCCUPIED_PLATFORM)

    @staticmethod
    def validate_action_details(
        action_str, object_num, platform_name_list, freespace_pair_list
    ):
        action_type, action_param = HoldingOccupiedPlatformState.validate_action(
            action_str
        )
        if action_type not in HoldingOccupiedPlatformState.action_space_type:
            return ActionType.INVALID, 0
        elif (
            action_type == ActionType.CALL_END
            or action_type == ActionType.ROTATE
            or action_type == ActionType.PLACE_ANYWHERE
        ):
            return action_type, 0
        elif action_type == ActionType.GOTO_PLATFORM:
            if action_param in platform_name_list:
                return action_type, action_param
            else:
                return ActionType.INVALID, 0

        elif action_type == ActionType.PLACE_OCCUPIED:
            if isinstance(action_param, list):
                if all(item in freespace_pair_list for item in action_param):
                    return action_type, action_param
                else:
                    return ActionType.INVALID, 0
            else:
                return ActionType.INVALID, 0
        elif action_type == ActionType.SHOW_OBJECT:
            if action_param > 0 and action_param <= object_num:
                return action_type, action_param
            else:
                return ActionType.INVALID, 0
        else:
            return ActionType.INVALID, 0
        return ActionType.INVALID, 0

    @staticmethod
    def generate_prompt(
        scene_description,
        platform_list,
        task_description,
        steps_used,
        total_steps,
        platform_name,
        holding_object,
        image_type,
        image_info_list,
        available_freespace_pair_list,
        n_image,
        image_name_list,
        location_action_list,
        show_freespace_of_object_action_list,
        n_object,
    ):
        prompt = InteractSystemState.state_prompt_helper.generate_state_prompts(
            state=InteractStates.HOLDING_OCCUPIED_PLATFORM.name.lower(),
            context={
                "scene_description": scene_description,
                "platform_list": platform_list,
                "task_description": task_description,
                "steps_used": steps_used,
                "total_steps": total_steps,
                "platform_name": platform_name,
                "holding_object": holding_object,
                "image_type": image_type,
                "image_info_list": image_info_list,
                "available_freespace_pair_list": available_freespace_pair_list,
                "location_action_list": location_action_list,
                "show_freespace_of_object_action_list": show_freespace_of_object_action_list,
                "n_image": n_image,
                "image_name_list": image_name_list,
                "n_object": n_object,
            },
        )
        return prompt


class MistakeLogGenerator:

    def __init__(
        self,
        scene_graph=None,
        task=None,
        task_id=0,
        action_history_list=None,
        state_history_list=None,
        image_path=None,
    ):

        self.scene_graph = scene_graph
        self.task = task
        self.task_id = task_id
        self.action_history_list = action_history_list
        self.mistake_logic = None
        self.state_history_list = state_history_list
        self.reflection_prompt_helper = ReflectionPromptManager(
            prompt_templates="vlm_interactor/prompts/reflection_prompts.json"
        )
        self.picture_width = 683
        self.picture_height = 384
        self.image_path = image_path

    @staticmethod
    def _rotated(self, action_list):
        for action in action_list:
            if action.startswith("rotate_"):
                return True
        return False
        pass

    @staticmethod
    def _place_repeatedly(self, action_list):
        place_cnt = 0
        for action in action_list:
            if action.startswith("place_at_freespace_"):
                place_cnt += 1
        return place_cnt > 2
        pass

    @staticmethod
    def _too_much_invalid_action(action_list, number=3):
        invalid_cnt = 0
        for action in action_list:
            if "invalid" in action:
                invalid_cnt += 1
        return invalid_cnt >= number
        pass

    @staticmethod
    def __attempted_place_before_looking(action_list):
        show_freespace = False
        for action in action_list:
            if action.startswith("show_freespace_of_object_"):
                show_freespace = True
            if action.startswith("place_") and not show_freespace:
                return True
        return False

    def generate_mistake_log(self):
        def judge_dict_equal_for_partial_score(dict1, dict2):

            if len(dict1) != len(dict2):
                return False
            for key, value in dict1.items():
                if key not in dict2 or dict2[key] != value and dict2[key] != "any":
                    return False

            return True

        intermediate_state_list = self.task.intermediate_state_list
        state_history_list = self.state_history_list
        n_score = len(intermediate_state_list)

        correct_action_list = [[] for i in range(4)]
        score = 0
        for i, status in enumerate(self.status_history_list):
            if score >= n_score + 1:
                break
            correct_action_list[score].append(self.action_history_list[i])
            if judge_dict_equal_for_partial_score(
                status, self.task.intermediate_state_list[self.partial_score]
            ):
                score += 1

        current_path = self.image_path
        destination = self.task.destination
        standing_direction = 0

        for step in range(score + 1, n_score + 1 + 1):

            text_prompt = ""
            save_path = f"{current_path}/image4reflection/{self.model}/Task{self.task_id}_step_{step}.png"
            image_path_prompt_list = []
            if step == 1:
                correct_platform_name = (
                    self.task.item.on_platform.get_name_for_interaction()
                )
                self.reflection_prompt_helper.generate_reflection_prompts(
                    prompt_type="goto_platform",
                    platform_name=correct_platform_name,
                )
                pass
            elif step == 2:
                platform = self.task.item.on_platform
                platform_object_name = platform.name[: platform.name.rfind("_")]
                platform_object = self.scene_graph.nodes[platform_object_name]
                standing_direction = platform.get_first_standing_direction()

                platform_img, platform_img_list = (
                    self.scene_graph.auto_take_platform_picture(
                        platform_name=platform.name,
                        view="human_full",
                        mark_object=True,
                        mark_freespace=len(platform.children) == 0,
                        standing_direction=standing_direction,
                        width=self.picture_width,
                        height=self.picture_height,
                        focus_ratio=0.6,
                        save_path=save_path,
                    )
                )

                correct_object_name = self.task.item.get_name_for_interaction()
                correct_object_idx = self.task.item.get_object_id_on_platform()
                self.scene_graph.remove_node(self.task.item.name)
                self.reflection_prompt_helper.generate_reflection_prompts(
                    prompt_type="pick_up_object",
                    object_name=correct_object_name,
                    object_idx=correct_object_idx,
                )
                image_path_prompt_list.extend(platform_img_list)
                pass
            elif step == 3:
                platform = self.task.item.on_platform
                standing_direction = platform.get_first_standing_direction()
                platform_img, platform_img_list = (
                    self.scene_graph.auto_take_platform_picture(
                        platform_name=platform.name,
                        view="human_full",
                        mark_object=True,
                        mark_freespace=len(platform.children) == 0,
                        standing_direction=self.standing_direction,
                        width=self.picture_width,
                        height=self.picture_height,
                        focus_ratio=0.6,
                        save_path=save_path,
                    )
                )
                correct_platform_name = self.task.destination.get_name_for_interaction()
                self.reflection_prompt_helper.generate_reflection_prompts(
                    prompt_type="goto_platform",
                    platform_name=correct_platform_name,
                )
                image_path_prompt_list.extend(platform_img_list)
                pass
            elif step == 4:
                platform = self.task.destination
                standing_direction = platform.get_first_standing_direction()
                platform_img, platform_img_list = (
                    self.scene_graph.auto_take_platform_picture(
                        platform_name=platform.name,
                        view="human_full",
                        mark_object=True,
                        mark_freespace=len(platform.children) == 0,
                        standing_direction=self.standing_direction,
                        width=self.picture_width,
                        height=self.picture_height,
                        focus_ratio=0.6,
                        save_path=save_path,
                    )
                )
                image_path_prompt_list.extend(platform_img_list)

                if self.task.type == TaskType.MOVE_TO_EMPTY_PLATFORM:
                    self.reflection_prompt_helper.generate_reflection_prompts(
                        prompt_type="place_object",
                        place_type="on_platform",
                        direction=self.task.feature[0],
                        direction_idx=NINE_DIRECTIONS.index(self.task.feature[0]),
                        object_name=self.task.item.get_name_for_interaction(),
                    )
                    pass
                elif self.task.type == TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID:
                    platform_name = platform.get_name_for_interaction()
                    direction_id = (
                        (NINE_DIRECTIONS.index(self.feature[0]) + standing_direction)
                        % 8
                        + 1
                        if self.feature[0] != "center"
                        else 8
                    )

                    place_destination_platform_action = (
                        f"place_at_freespace_[{direction_id}]"
                    )

                    center_extend_direction_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                    cardinal_extend_direction_list = [direction_id, 9]
                    diagonal_extend_direction_list = [
                        direction_id,
                        (direction_id + 1) % 8,
                        (direction_id + 7) % 8,
                        9,
                    ]

                    cardinal_extend_more_direction_list = [
                        direction_id,
                        (direction_id + 1) % 8,
                        (direction_id + 7) % 8,
                        (direction_id + 2) % 8,
                        (direction_id + 6) % 8,
                        9,
                    ]
                    diagonal_extend_direction_list = [
                        8 if dir == 0 else dir for dir in diagonal_extend_direction_list
                    ]
                    cardinal_extend_more_direction_list = [
                        8 if dir == 0 else dir
                        for dir in cardinal_extend_more_direction_list
                    ]

                    if self.feature[0] == "center" or "-" in self.feature[0]:
                        extend_direction_list = (
                            center_extend_direction_list
                            if self.feature[0] == "center"
                            else diagonal_extend_direction_list
                        )
                        self.reflection_prompt_helper.generate_reflection_prompts(
                            prompt_type="place_object",
                            place_type="on_platform_dir_diagonal",
                            object_name=self.task.item.get_name_for_interaction(),
                            platform_name=platform_name,
                            direction=self.feature[0],
                            direction_idx=direction_id,
                            extend_direction_list=extend_direction_list,
                            place_destination_platform_action=place_destination_platform_action,
                        )
                    else:
                        self.reflection_prompt_helper.generate_reflection_prompts(
                            prompt_type="place_object",
                            place_type="on_platform_dir_cardinal",
                            object_name=self.task.item.get_name_for_interaction(),
                            platform_name=platform_name,
                            direction=self.feature[0],
                            direction_idx=direction_id,
                            extend_direction_cardinal_list=cardinal_extend_direction_list,
                            more_extending_direction_list=cardinal_extend_more_direction_list,
                            place_destination_platform_action=place_destination_platform_action,
                        )

                elif self.task.type == TaskType.MOVE_AROUND_OBJECT:
                    dest_item_node = self.scene_graph.nodes[self.feature[0]]
                    dest_item_name = dest_item_node.get_name_for_interaction()
                    freespace_num = dest_item_node.get_num_of_critical_space()
                    dest_item_id = [
                        id
                        for id, _ in enumerate(
                            dest_item_node.get_bel_ground_platform().children
                        )
                        if _.name == self.feature[0]
                    ][0] + 1
                    adjacent_object_node_list = [
                        child
                        for child in self.task.destination.children
                        if child.depth <= 2
                        and child.get_name_for_interaction() != correct_object_name
                    ]
                    adjacent_object_name_list = [
                        child.get_name_for_interaction()
                        for child in adjacent_object_node_list
                    ]
                    adjacent_object_idx_list = [
                        child.get_object_id_on_platform()
                        for child in adjacent_object_node_list
                    ]
                    freespace_list = [
                        (dest_item_id, i) for i in range(1, freespace_num + 1)
                    ]
                    self.reflection_prompt_helper.generate_reflection_prompts(
                        prompt_type="place_object",
                        place_type="around_object",
                        freespace_list=freespace_list,
                        object_name=dest_item_name,
                        object_idx=dest_item_id,
                        adjacent_object_name_list=adjacent_object_name_list,
                        adjacent_object_idx_list=adjacent_object_idx_list,
                    )
                    self.reflection_prompt_helper.generate_reflection_prompts(
                        prompt_type="place_object",
                        place_type="around_object",
                    )
                    pass
                elif self.task.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID:
                    # dest_item_node = scene_graph.nodes[self.feature[0]]
                    # dest_item_name = dest_item_node.get_name_for_interaction()
                    # dest_item_id = [id for id, _ in enumerate(dest_item_node.get_bel_ground_platform().children) if _.name == self.feature[0]][0] + 1
                    # direction_id = ( NINE_DIRECTIONS.index(self.feature[1]) + first_standing_direction) % 8
                    # direction_id = dest_item_node.get_on_picture_freespace_id(direction_id)

                    # for i in range(4):
                    #     if (direction_id != -1):
                    #         break
                    #     first_standing_direction_id = (first_standing_direction_id + 1) % len(available_directions)
                    #     first_standing_direction = available_directions[first_standing_direction_id]
                    #     direction_id = ( NINE_DIRECTIONS.index(self.feature[1]) + first_standing_direction) % 8
                    #     direction_id = dest_item_node.get_on_picture_freespace_id(direction_id)

                    #     goal_action_list.append(f'rotate_observation_view_of_current_platform')
                    #     goal_explanation_list.append(f"rotate your observation view angle to avoid a bad and could-be-good view angle. If the view angle is getting worse, just repeat this action for at most 4 times until you find a good view angle.")
                    # if direction_id == -1:
                    #     direction_id = 2

                    # self.reflection_prompt_helper.generate_reflection_prompts(
                    #     prompt_type="place_object",
                    #     place_type='on_direction_of_object',
                    #     freespace_list=freespace_list,
                    #     object_name=dest_item_name,
                    #     object_idx=dest_item_id,
                    #     adjacent_object_name_list=adjacent_object_name_list,
                    #     adjacent_object_idx_list=adjacent_object_idx_list,
                    #     ad

                    # )
                    pass
                elif self.task.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
                    object_name_a = self.task.feature[0]
                    object_name_b = self.task.feature[1]
                    object_node_a = [
                        child
                        for child in self.task.destination.children
                        if child.get_name_for_interaction() == object_name_a
                    ][0]
                    object_node_b = [
                        child
                        for child in self.task.destination.children
                        if child.get_name_for_interaction() == object_name_b
                    ][0]

                    while object_node_a.depth > 2:
                        object_node_a = object_node_a.parent
                    while object_node_b.depth > 2:
                        object_node_b = object_node_b.parent

                    object_idx_a = object_node_a.get_object_id_on_platform()
                    object_idx_b = object_node_b.get_object_id_on_platform()
                    adjacent_object_node_list = [
                        child
                        for child in self.task.destination.children
                        if child.depth <= 2
                        and child.get_name_for_interaction() != object_name_a
                        and child.get_name_for_interaction() != object_name_b
                    ]
                    adjacent_object_name_list = [
                        child.get_name_for_interaction()
                        for child in adjacent_object_node_list
                    ]
                    adjacent_object_idx_list = [
                        child.get_object_id_on_platform()
                        for child in adjacent_object_node_list
                    ]

                    self.reflection_prompt_helper.generate_reflection_prompts(
                        prompt_type="place_object",
                        place_type="middle_of_2_objects",
                        object_name_a=object_name_a.get_name_for_interaction(),
                        object_name_b=object_name_b.get_name_for_interaction(),
                        object_idx_a=object_idx_a,
                        object_idx_b=object_idx_b,
                        adjacent_object_name_list=adjacent_object_name_list,
                        adjacent_object_idx_list=adjacent_object_idx_list,
                    )
                    pass
                else:
                    pass
                # self.reflection_prompt_helper.generate_reflection_prompts(
                #     prompt_type="show_freespace_of_object",
                # )
                if (
                    self.task.type == TaskType.MOVE_AROUND_OBJECT
                    or self.task.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID
                ):
                    correct_object_name4interaction = self.task.feature[0]
                    correct_object_node = [
                        child
                        for child in self.task.destination.children
                        if child.get_name_for_interaction()
                        == correct_object_name4interaction
                    ][0]

                    correct_object_name = correct_object_node.get_name_for_interaction()
                    correct_object_idx = correct_object_node.get_object_id_on_platform()

                    self.reflection_prompt_helper.generate_reflection_prompts(
                        prompt_type="on_direction_of_object",
                        object_name=correct_object_name,
                        object_idx=correct_object_idx,
                        freespace_list=freespace_list,
                    )
                elif self.task.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
                    correct_object_name4interaction = self.task.feature[0]
                    correct_object_node = [
                        child
                        for child in self.task.destination.children
                        if child.get_name_for_interaction()
                        == correct_object_name4interaction
                    ][0]

                    correct_object_name = correct_object_node.get_name_for_interaction()
                    correct_object_idx = correct_object_node.get_object_id_on_platform()
                    self.reflection_prompt_helper.generate_reflection_prompts(
                        prompt_type="pick_up_object",
                        object_name=correct_object_name,
                        object_idx=correct_object_idx,
                    )

                    correct_object_name4interaction = self.task.feature[1]
                    correct_object_node = [
                        child
                        for child in self.task.destination.children
                        if child.get_name_for_interaction()
                        == correct_object_name4interaction
                    ][0]

                    correct_object_name = correct_object_node.get_name_for_interaction()
                    correct_object_idx = correct_object_node.get_object_id_on_platform()
                    self.reflection_prompt_helper.generate_reflection_prompts(
                        prompt_type="pick_up_object",
                        object_name=correct_object_name,
                        object_idx=correct_object_idx,
                    )
                pass
            pass

            if len(image_path_prompt_list) > 0:
                for image_path_prompt in image_path_prompt_list:
                    self.vlm_interactor.add_content(
                        content=image_path_prompt,
                        content_type="image",
                        role="user",
                    )
            if text_prompt != "":
                self.vlm_interactor.add_content(
                    content=text_prompt,
                    content_type="text",
                    role="user",
                )

        if not MistakeLogGenerator._rotated(self.action_history_list[3]):
            self.reflection_prompt_helper.generate_reflection_prompts(
                prompt_type="no_rotate_hint",
                context={
                    "task_id": self.task.task_id,
                    "task_type": self.task.type,
                },
            )

        if MistakeLogGenerator._place_repeatedly(
            correct_action_list[1] + correct_action_list[2]
        ):
            if score > 2:
                self.reflection_prompt_helper.generate_reflection_prompts(
                    prompt_type="attempted_place_repeatedly_at_beginning_hint",
                    context={},
                )
            else:
                self.reflection_prompt_helper.generate_reflection_prompts(
                    prompt_type="attempted_place_repeatedly_at_end_with_wrong_item_hint",
                    action_list=correct_action_list[1] + correct_action_list[2],
                )
        if MistakeLogGenerator._too_much_invalid_action(self.action_history_list):
            self.reflection_prompt_helper.generate_reflection_prompts(
                prompt_type="too_much_invalid_action_hint",
            )

        if MistakeLogGenerator.__attempted_place_before_looking(
            self.action_history_list
        ):
            if (
                self.task.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS
                or self.task.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID
            ):
                self.reflection_prompt_helper.generate_reflection_prompts(
                    prompt_type="attempted_place_before_look_hint",
                    action_list=self.action_history_list,
                )

        pass

    def calc_intermediate_score(self):
        def judge_dict_equal_for_partial_score(dict1, dict2):

            if len(dict1) != len(dict2):
                return False
            for key, value in dict1.items():
                if key not in dict2 or dict2[key] != value and dict2[key] != "any":
                    return False

            return True

        partial_score = 0

        for i, status in enumerate(self.status_history_list):
            if partial_score >= len(self.task.intermediate_state_list):
                break
            if judge_dict_equal_for_partial_score(
                status, self.task.intermediate_state_list[partial_score]
            ):
                partial_score += 1
        self.partial_score = partial_score
        return partial_score

        pass


class TaskInteractHelper:

    def __init__(
        self,
        scene_graph=None,
        scene=None,
        platform_list=None,
        task=None,
        task_id=None,
        intermediate_task_id=None,
        intermediate_task=None,
        vlm_interactor=None,
        img_path=None,
        model_name=None,
        debug_mode=False,
        generate_mistake_note=False,
        use_mistake_note=0,
        reflection_file_path=None,
    ):
        self.executor_config = get_benchmark_executor_config()
        self.interaction_config = get_task_interaction_config()
        self.model = model_name
        self.scene_graph = scene_graph
        tmp_scene = self.scene_graph.corresponding_scene
        self.scene_graph.corresponding_scene = None
        self.initial_scene_graph = copy.deepcopy(scene_graph)
        self.scene_graph.corresponding_scene = tmp_scene
        self.scene = scene
        self.platform_list = scene_graph.get_sensible_platform_list()
        self.platform_name_list = [
            platform.get_name_for_interaction() for platform in self.platform_list
        ]
        # for i, platform in enumerate(self.platform_list):
        #     glog.info(f"Platform {i}: {platform.name}")
        #     glog.info(f"children: {platform.children}")
        self.task = task
        self.task_id = task_id
        self.intermediate_task_id = intermediate_task_id
        self.intermediate_task = intermediate_task

        self.is_dual_task_mode = intermediate_task is not None

        if self.is_dual_task_mode:
            self.task_description = f"{self.intermediate_task}, then {self.task}"
            self.task_initial_information = f"For the 1st atomic task, {intermediate_task.initial_state_information()}, for the 2nd atomic task, {task.initial_state_information()}"
        else:
            self.task_description = task.__repr__()
            self.task_initial_information = task.initial_state_information()

        self.state = InteractStates.DEFAULT
        self.vlm_interactor = vlm_interactor
        self.picture_width = self.executor_config.picture_width
        self.picture_height = self.executor_config.picture_height
        self.reflection_file_path = reflection_file_path

        self.status = (None,)

        self.at_place = -1
        self.standing_direction = 0
        self.object_in_hand = None
        self.image_path = img_path
        self.standing_direction_while_placing = {}
        self.state_prompt_helper = StatePromptManager(
            prompt_templates=self.executor_config.prompt_templates_path
        )
        self.hint_prompt_helper = HintPromptManager(
            prompt_templates=self.executor_config.prompt_templates_path
        )
        self.reflection_prompt_helper = ReflectionPromptManager(
            prompt_templates=self.executor_config.reflection_prompts_path
        )
        self.action_history_list = []
        self.status_history_list = []
        self.partial_score = 0
        self.debug_mode = debug_mode
        self.generate_mistake_note = generate_mistake_note
        self.use_mistake_note = use_mistake_note

    def calc_partial_score(self):
        """Calculate partial score for the main task"""

        def judge_dict_equal_for_partial_score(dict1, dict2):
            if len(dict1) != len(dict2):
                return False
            for key, value in dict1.items():
                if key not in dict2 or dict2[key] != value and dict2[key] != "any":
                    return False
            return True

        partial_score = 0
        for i, status in enumerate(self.status_history_list):
            if partial_score >= len(self.task.intermediate_state_list):
                break
            if judge_dict_equal_for_partial_score(
                status, self.task.intermediate_state_list[partial_score]
            ):
                partial_score += 1
        self.partial_score = partial_score
        return partial_score

    def calc_intermediate_task_partial_score(self):
        """Calculate partial score for the intermediate task (dual task mode only)"""
        if not self.is_dual_task_mode:
            return 0

        def judge_dict_equal_for_partial_score(dict1, dict2):
            if len(dict1) != len(dict2):
                return False
            for key, value in dict1.items():
                if key not in dict2 or dict2[key] != value and dict2[key] != "any":
                    return False
            return True

        partial_score = 0
        for i, status in enumerate(self.status_history_list):
            if partial_score >= len(self.intermediate_task.intermediate_state_list):
                break
            if judge_dict_equal_for_partial_score(
                status, self.intermediate_task.intermediate_state_list[partial_score]
            ):
                partial_score += 1
        return partial_score

    def if_intermediate_task_finished(self):
        """Check if the intermediate task is completed (dual task mode only)"""
        if not self.is_dual_task_mode:
            return True  # In single task mode, intermediate task is always considered complete

        self.intermediate_partial_score = max(
            self.intermediate_partial_score, self.calc_intermediate_task_partial_score()
        )

        if self.object_in_hand is not None:
            glog.debug("Object in hand - intermediate task not finished")
            return False

        previous_nodes = self.initial_scene_graph.nodes
        current_nodes = self.scene_graph.nodes

        for node_name, node in current_nodes.items():
            if node.depth <= 1:
                continue
            if node_name not in previous_nodes:
                glog.debug(f"{node_name} not in previous nodes")
                return False

        node = self.scene_graph.nodes[self.intermediate_task.item.name]
        node_previous = previous_nodes.get(self.intermediate_task.item.name)
        parent_previous = previous_nodes.get(
            self.intermediate_task.destination.bel_object
        )
        parent_current = current_nodes.get(
            self.intermediate_task.destination.bel_object
        )

        if (
            node is None
            or node_previous is None
            or parent_previous is None
            or parent_current is None
        ):
            glog.debug("Task item or destination not in the scene graph tree")
            return False

        if parent_previous.on_platform.name != parent_current.on_platform.name:
            glog.debug("Destination Item is changed, which is not allowed")
            return False

        if node.on_platform.name != self.intermediate_task.destination.name:
            glog.debug(
                f"Item is not on the correct platform, expected: {self.intermediate_task.destination.name}, actual: {node.on_platform.name}"
            )
            return False

        # Check task completion based on intermediate task type
        return self._check_task_completion(self.intermediate_task, node)

    def if_task_finished(self):
        """Check if the main task is completed"""
        self.partial_score = self.calc_partial_score()

        if self.object_in_hand is not None:
            glog.debug("Object in hand - task not finished")
            return False

        previous_nodes = self.initial_scene_graph.nodes
        current_nodes = self.scene_graph.nodes

        for node_name, node in current_nodes.items():
            if node.depth <= 1:
                continue
            if node_name not in previous_nodes:
                glog.debug(f"{node_name} not in previous nodes")
                return False

        node = self.scene_graph.nodes[self.task.item.name]
        node_previous = previous_nodes.get(self.task.item.name)
        parent_previous = previous_nodes.get(self.task.destination.bel_object)
        parent_current = current_nodes.get(self.task.destination.bel_object)

        if (
            node is None
            or node_previous is None
            or parent_previous is None
            or parent_current is None
        ):
            glog.debug("Task item or destination not in the scene graph tree")
            return False

        if parent_previous.on_platform.name != parent_current.on_platform.name:
            glog.debug("Destination Item is changed, which is not allowed")
            return False

        if node.on_platform.name != self.task.destination.name:
            glog.debug("Item is not on the correct platform")
            return False

        # Check task completion based on task type
        return self._check_task_completion(self.task, node)

    def _check_task_completion(self, task, node):
        """Check task completion status for specific task type"""
        if task.type == TaskType.MOVE_TO_EMPTY_PLATFORM:
            return True
        elif task.type == TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID:
            return self._check_9_grid_placement(task, node)
        elif task.type == TaskType.MOVE_AROUND_OBJECT:
            return True
        elif task.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID:
            return self._check_object_freespace_9_grid(task, node)
        elif task.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
            return self._check_middle_of_objects(task, node)
        return False

    def _check_9_grid_placement(self, task, node):
        """Check if 9-grid placement is correct"""
        if task.item.name not in self.standing_direction_while_placing:
            self.standing_direction_while_placing[node.name] = 0

        required_direction = (NINE_DIRECTIONS.index(task.feature[0])) % 8
        if task.feature[0] == "center":
            required_direction = 8
        part = node.at_which_part()

        correct_direction = (
            (required_direction + self.standing_direction_while_placing[node.name]) % 8
            if required_direction != 8
            else 8
        )

        if NINE_DIRECTIONS[correct_direction] in part:
            glog.info("Target object placed in correct direction.")
            return True
        else:
            glog.info(
                f"Target object placed in wrong direction, expected: {NINE_DIRECTIONS[correct_direction]}, actual: {part}"
            )
            return False

    def _check_object_freespace_9_grid(self, task, node):
        """Check if object freespace 9-grid placement is correct"""
        if task.item.name not in self.standing_direction_while_placing:
            self.standing_direction_while_placing[node.name] = 0
        destination_node = self.scene_graph.nodes[task.feature[0]]
        required_direction = (NINE_DIRECTIONS.index(task.feature[1])) % 8
        in_dir = destination_node.judge_critical_direction(node.name)

        correct_direction = (
            required_direction + self.standing_direction_while_placing[node.name]
        ) % 8

        if correct_direction in in_dir:
            glog.info("Target object placed in correct direction.")
            return True
        else:
            glog.info(
                f"Target object placed in wrong direction, expected: {NINE_DIRECTIONS[correct_direction]}, actual: {NINE_DIRECTIONS[in_dir[0]] if len(in_dir) > 0 else None}"
            )
            return False

    def _check_middle_of_objects(self, task, node):
        """Check if object is correctly placed between two objects"""
        if task.item.name not in self.standing_direction_while_placing:
            self.standing_direction_while_placing[node.name] = 0
        destination_nodea = self.scene_graph.nodes[task.feature[0]]
        destination_nodeb = self.scene_graph.nodes[task.feature[1]]

        for dira in range(8):
            dira_freespace = destination_nodea.free_space[dira]["Available_space"]
            dira_freespace = ConvexHullProcessor_2d(vertices=dira_freespace)
            is_intersect = False
            oppositing_directions = [(dira + 4) % 8]

            for dirb in oppositing_directions:
                dirb_freespace = destination_nodeb.free_space[dirb]["Available_space"]
                dirb_freespace = ConvexHullProcessor_2d(vertices=dirb_freespace)
                if (
                    dira_freespace.intersect_with_another_convex(dirb_freespace)
                    is not None
                ):
                    is_intersect = True
                    break

            dir_objects = [
                obj.name for obj in destination_nodea.free_space[dira]["Objects"]
            ]
            if is_intersect:
                glog.debug(f"{dira} and {dirb} intersect")
                if node.name in dir_objects:
                    glog.info("Target object placed between two items.")
                    return True
        glog.info("Target object not placed between two items.")
        return False

    # def calc_intermediate_score(self):
    #     def judge_dict_equal_for_partial_score(dict1, dict2):

    #         if len(dict1) != len(dict2):
    #             return False
    #         for key, value in dict1.items():
    #             if key not in dict2 or dict2[key] != value and dict2[key] != "any":
    #                 return False

    #         return True

    #     partial_score = 0

    #     for i, status in enumerate(self.status_history_list):
    #         if partial_score >= len(self.task.intermediate_state_list):
    #             break
    #         if judge_dict_equal_for_partial_score(
    #             status, self.task.intermediate_state_list[partial_score]
    #         ):
    #             partial_score += 1
    #     self.partial_score = partial_score
    #     return partial_score

    #     pass

    # def see_the_result(self):

    #     if self.task.type == TaskType.MOVE_TO_EMPTY_PLATFORM:
    #         pass

    # def if_task_finished(self):
    #     """
    #     The logic of checking if the task is finished:
    #     1. All the object have correct parent.
    #     The tree may not be the same if an object is placed in a wrong place then placed back.

    #     2. The object in hand is placed in correct position.
    #     Put on platform:
    #     No direction requirement
    #     Put on platform 9-grid:
    #     the last standing direction + requirement should = the actual standing possition
    #     Put around an Item:
    #     No direcion requirement
    #     Put around an Item 9-grid:
    #     the last standing direction + requirement should = the actual standing possition
    #     Put between 2 item:
    #     The item should be in both items' free space.

    #     """

    #     self.partial_score = self.calc_intermediate_score()

    #     if self.object_in_hand is not None:
    #         print("Object in hand")
    #         return False

    #     previous_nodes = self.initial_scene_graph.nodes
    #     current_nodes = self.scene_graph.nodes

    #     for node_name, node in current_nodes.items():
    #         if node.depth <= 1:
    #             continue
    #         if node_name not in previous_nodes:
    #             print(node_name, "not in previous nodes")
    #             return False

    #     # import ipdb

    #     # ipdb.set_trace()

    #     node = self.scene_graph.nodes[self.task.item.name]
    #     node_previous = previous_nodes.get(self.task.item.name)
    #     parent_previous = previous_nodes.get(self.task.destination.bel_object)

    #     parent_current = current_nodes.get(self.task.destination.bel_object)
    #     if (
    #         node is None
    #         or node_previous is None
    #         or parent_previous is None
    #         or parent_current is None
    #     ):
    #         print("Task item or destination not in the scene graph tree")
    #         return False

    #     if parent_previous.on_platform.name != parent_current.on_platform.name:
    #         print("Destination Item is changed, which is not allowed")
    #         return False

    #     if node.on_platform.name != self.task.destination.name:
    #         print("Item is not on the correct platform")
    #         return False

    #     if self.task.type == process_based_task_generation.TaskType.MOVE_TO_EMPTY_PLATFORM:
    #         return True
    #     elif (
    #         self.task.type
    #         == process_based_task_generation.TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID
    #     ):
    #         if self.task.item.name not in self.standing_direction_while_placing:
    #             self.standing_direction_while_placing[node.name] = 0

    #         required_direction = (NINE_DIRECTIONS.index(self.task.feature[0])) % 8
    #         if self.task.feature[0] == "center":
    #             required_direction = 8
    #         part = node.at_which_part()

    #         correct_direction = (
    #             (required_direction + self.standing_direction_while_placing[node.name])
    #             % 8
    #             if required_direction != 8
    #             else 8
    #         )

    #         if NINE_DIRECTIONS[correct_direction] in part:
    #             print("Target object placed in correct direction.")
    #             return True
    #         else:
    #             print(
    #                 "Target object placed in wrong direction, expected:",
    #                 NINE_DIRECTIONS[correct_direction],
    #                 "actual:",
    #                 part,
    #             )
    #             return False

    #     elif self.task.type == process_based_task_generation.TaskType.MOVE_AROUND_OBJECT:
    #         return True
    #     elif (
    #         self.task.type
    #         == process_based_task_generation.TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID
    #     ):
    #         if self.task.item.name not in self.standing_direction_while_placing:
    #             self.standing_direction_while_placing[node.name] = 0
    #         destination_node = self.scene_graph.nodes[self.task.feature[0]]
    #         required_direction = (NINE_DIRECTIONS.index(self.task.feature[1])) % 8
    #         in_dir = destination_node.judge_critical_direction(node.name)

    #         correct_direction = (
    #             required_direction + self.standing_direction_while_placing[node.name]
    #         ) % 8

    #         if correct_direction in in_dir:
    #             print("Target object placed in correct direction.")
    #             return True
    #         else:
    #             print(
    #                 "Target object placed in wrong direction, expected:",
    #                 NINE_DIRECTIONS[correct_direction],
    #                 "actual:",
    #                 NINE_DIRECTIONS[in_dir[0]] if len(in_dir) > 0 else None,
    #             )
    #             return False

    #     elif (
    #         self.task.type == process_based_task_generation.TaskType.MOVE_TO_MIDDLE_OF_OBJECTS
    #     ):
    #         if self.task.item.name not in self.standing_direction_while_placing:
    #             self.standing_direction_while_placing[node.name] = 0
    #         destination_nodea = self.scene_graph.nodes[self.task.feature[0]]
    #         destination_nodeb = self.scene_graph.nodes[self.task.feature[1]]
    #         for dira in range(8):
    #             dira_freespace = destination_nodea.free_space[dira]["Available_space"]
    #             dira_freespace = ConvexHullProcessor_2d(vertices=dira_freespace)
    #             is_intersect = False
    #             oppositing_directions = [(dira + 4) % 8]

    #             for dirb in oppositing_directions:
    #                 dirb_freespace = destination_nodeb.free_space[dirb][
    #                     "Available_space"
    #                 ]
    #                 dirb_freespace = ConvexHullProcessor_2d(vertices=dirb_freespace)
    #                 if (
    #                     dira_freespace.intersect_with_another_convex(dirb_freespace)
    #                     is not None
    #                 ):
    #                     is_intersect = True
    #                     break
    #             dir_objects = [
    #                 obj.name for obj in destination_nodea.free_space[dira]["Objects"]
    #             ]
    #             print(dir_objects)
    #             if is_intersect:
    #                 print(f"{dira} and {dirb} intersect")
    #                 dir_objects = [
    #                     obj.name
    #                     for obj in destination_nodea.free_space[dira]["Objects"]
    #                 ]
    #                 print(dir_objects)
    #                 if node.name in dir_objects:
    #                     print("Target object placed between two items.")
    #                     return True
    #         print("Target object not placed between two items.")
    #         return False

    #     pass

    def __goto_platform(self, platform_name):
        """

        Goto platform with id = platform_id.

        Args:
            platform_id (_type_): _description_
        """
        platform_id = self.platform_name_list.index(platform_name)
        self.action_history_list.append(f"goto_{self.platform_name_list[platform_id]}")
        self.at_place = platform_id
        self.standing_direction = 0
        if self.state == InteractStates.NAVIGATION:
            self.state = InteractStates.IDLE
        elif (
            self.state == InteractStates.HOLDING_EMPTY_PLATFORM
            or self.state == InteractStates.HOLDING_OCCUPIED_PLATFORM
        ):
            self.state = (
                InteractStates.HOLDING_EMPTY_PLATFORM
                if len(self.platform_list[self.at_place].children) == 0
                else InteractStates.HOLDING_OCCUPIED_PLATFORM
            )
        pass

    def __handle_ambiguous_item(self, task, task_id=0, task_order=1):
        """Handle ambiguous objects (adapted for dual tasks)"""
        if not task.is_ambiguous(self.scene_graph):
            return

        ambiguous_hint = f"For the No. {task_order} atomic task: {task.__repr__()}"
        ambiguous_hint += self.hint_prompt_helper.generate_hint_prompts(
            hint_type="identical_object_involved"
        )
        self.vlm_interactor.add_content(
            content=ambiguous_hint, content_type="text", role="user"
        )

        # Original ambiguous handling logic...
        if task.item.is_ambiguous:
            object_to_show = task.item
            save_path = f"{self.image_path}/image4interact/{self.model}/Task_{task_id}_AmbiguousObject.png"
            img = object_to_show.auto_take_non_ground_object_picture(
                scene=self.scene_graph.corresponding_scene,
                view="human_focus",
                mark_object=False,
                only_mark_itself=False,
                mark_freespace=False,
                diagonal_mode="old",
                need_afford_rect=None,
                standing_direction=self.standing_direction,
                width=self.picture_width,
                height=self.picture_height,
                focus_ratio=0.5,
                fovy_range=[np.deg2rad(40), np.deg2rad(60)],
                save_path=save_path,
            )

            identical_object_to_move_prompt = self.state_prompt_helper.generate_certain_type_prompts(
                prompt_type="image_description",
                context={
                    "image_type": "identical_object_to_move",
                    "image_name": save_path,
                    "n_image": 1,
                    "image_name_list": [save_path],
                    "image_info_list": [
                        {
                            "image_name": save_path,
                            "source_object_name": object_to_show.get_name_for_interaction(),
                            "source_platform_name": object_to_show.get_bel_ground_platform().get_name_for_interaction(),
                        }
                    ],
                },
            )

            self.vlm_interactor.add_content(
                content=save_path, content_type="image", role="user"
            )

            self.vlm_interactor.add_content(
                content=identical_object_to_move_prompt,
                content_type="text",
                role="user",
            )

        if (
            self.task.type == TaskType.MOVE_AROUND_OBJECT
            or self.task.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID
        ):
            destination_object_name = self.task.feature[0]
            destination_object = self.scene_graph.nodes[destination_object_name]
            if destination_object.is_ambiguous:
                save_path = f"{self.image_path}/image4interact/{self.model}/Task_{self.task_id}_AmbiguousDestinationObject.png"
                object_to_show = destination_object
                img = object_to_show.auto_take_non_ground_object_picture(
                    scene=self.scene_graph.corresponding_scene,
                    view="human_focus",
                    mark_object=False,
                    only_mark_itself=False,
                    mark_freespace=False,
                    diagonal_mode="old",
                    need_afford_rect=None,
                    standing_direction=self.standing_direction,
                    width=self.picture_width,
                    height=self.picture_height,
                    focus_ratio=0.5,
                    fovy_range=[np.deg2rad(40), np.deg2rad(60)],
                    save_path=save_path,
                )

                identical_object_destination_prompt = self.state_prompt_helper.generate_certain_type_prompts(
                    prompt_type="image_description",
                    context={
                        "image_type": "identical_object_destination",
                        "image_name": save_path,
                        "n_image": 1,
                        "image_name_list": [save_path],
                        "image_info_list": [
                            {
                                "image_name": save_path,
                                "destination_object_name": object_to_show.get_name_for_interaction(),
                                "destination_platform_name": object_to_show.get_bel_ground_platform().get_name_for_interaction(),
                            }
                        ],
                    },
                )
                self.vlm_interactor.add_content(
                    content=save_path, content_type="image", role="user"
                )

                self.vlm_interactor.add_content(
                    content=identical_object_destination_prompt,
                    content_type="text",
                    role="user",
                )
        elif self.task.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
            destination_object_a_name, destination_object_b_name = (
                self.task.feature[0],
                self.task.feature[1],
            )
            destination_object_a, destination_object_b = (
                self.scene_graph.nodes[destination_object_a_name],
                self.scene_graph.nodes[destination_object_b_name],
            )
            if destination_object_a.is_ambiguous:
                save_path = f"{self.image_path}/image4interact/{self.model}/Task_{self.task_id}_AmbiguousDestinationObjectA.png"
                object_to_show = destination_object_a
                img = object_to_show.auto_take_non_ground_object_picture(
                    scene=self.scene_graph.corresponding_scene,
                    view="human_focus",
                    mark_object=False,
                    only_mark_itself=False,
                    mark_freespace=False,
                    diagonal_mode="old",
                    need_afford_rect=None,
                    standing_direction=self.standing_direction,
                    width=self.picture_width,
                    height=self.picture_height,
                    focus_ratio=0.5,
                    fovy_range=[np.deg2rad(40), np.deg2rad(60)],
                    save_path=save_path,
                )

                identical_object_destination_prompt = self.state_prompt_helper.generate_certain_type_prompts(
                    prompt_type="image_description",
                    context={
                        "image_type": "identical_object_destination",
                        "image_name": save_path,
                        "n_image": 1,
                        "image_name_list": [save_path],
                        "image_info_list": [
                            {
                                "image_name": save_path,
                                "destination_object_name": object_to_show.get_name_for_interaction(),
                                "destination_platform_name": object_to_show.get_bel_ground_platform().get_name_for_interaction(),
                            }
                        ],
                    },
                )
                self.vlm_interactor.add_content(
                    content=save_path, content_type="image", role="user"
                )

                self.vlm_interactor.add_content(
                    content=identical_object_destination_prompt,
                    content_type="text",
                    role="user",
                )
            if destination_object_b.is_ambiguous:
                save_path = f"{self.image_path}/image4interact/{self.model}/Task_{self.task_id}_AmbiguousDestinationObjectB.png"
                object_to_show = destination_object_b
                img = object_to_show.auto_take_non_ground_object_picture(
                    scene=self.scene_graph.corresponding_scene,
                    view="human_focus",
                    mark_object=False,
                    only_mark_itself=False,
                    mark_freespace=False,
                    diagonal_mode="old",
                    need_afford_rect=None,
                    standing_direction=self.standing_direction,
                    width=self.picture_width,
                    height=self.picture_height,
                    focus_ratio=0.5,
                    fovy_range=[np.deg2rad(40), np.deg2rad(60)],
                    save_path=save_path,
                )

                identical_object_destination_prompt = self.state_prompt_helper.generate_certain_type_prompts(
                    prompt_type="image_description",
                    context={
                        "image_type": "identical_object_destination",
                        "image_name": save_path,
                        "n_image": 1,
                        "image_name_list": [save_path],
                        "image_info_list": [
                            {
                                "image_name": save_path,
                                "destination_object_name": object_to_show.get_name_for_interaction(),
                                "destination_platform_name": object_to_show.get_bel_ground_platform().get_name_for_interaction(),
                            }
                        ],
                    },
                )
                self.vlm_interactor.add_content(
                    content=save_path, content_type="image", role="user"
                )

                self.vlm_interactor.add_content(
                    content=identical_object_destination_prompt,
                    content_type="text",
                    role="user",
                )

        pass

    def __auto_rotate_to_standable(self):
        if self.platform_bel_object is None:
            return
        prev_standing_direction = self.standing_direction
        for i in range(4):
            if len(self.platform.standing_point_list[self.standing_direction // 2]):
                break
            self.standing_direction = (self.standing_direction + 2) % 8

        glog.info(f"Standing direction is {self.standing_direction}")

    def __pickup_object(self, object_id):
        """
        Pick up object with id = object_id.
        Args:
            object_id (_type_): _description_
        """
        platform = self.platform_list[self.at_place]
        platform_bel_object = self.scene_graph.nodes[platform.bel_object]

        prohibit = []
        if (
            self.task.type == process_based_task_generation.TaskType.MOVE_AROUND_OBJECT
            or self.task.type
            == process_based_task_generation.TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID
        ):
            prohibit.append(self.task.feature[0])
        elif (
            self.task.type
            == process_based_task_generation.TaskType.MOVE_TO_MIDDLE_OF_OBJECTS
        ):
            prohibit.append(self.task.feature[0])
            prohibit.append(self.task.feature[1])

        # Intermediate task prohibition list (dual task mode only)
        if self.is_dual_task_mode:
            if (
                self.intermediate_task.type == TaskType.MOVE_AROUND_OBJECT
                or self.intermediate_task.type
                == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID
            ):
                prohibit.append(self.intermediate_task.feature[0])
            elif self.intermediate_task.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
                prohibit.append(self.intermediate_task.feature[0])
                prohibit.append(self.intermediate_task.feature[1])

        if platform.children[object_id - 1].name in prohibit:
            attempt_moving_destination_prompt = (
                self.hint_prompt_helper.generate_hint_prompts(
                    hint_type="attempt_moving_destination_hint",
                )
            )
            self.vlm_interactor.add_content(
                content=attempt_moving_destination_prompt,
                role="user",
                content_type="text",
            )
            self.state = InteractStates.IDLE
            return 0

        self.object_in_hand = platform.children[object_id - 1]
        self.action_history_list.append(f"pickup_object_{object_id}")

        self.scene_graph.remove_node(self.object_in_hand.name)
        for i, platform in enumerate(self.platform_list):
            self.platform_list[i] = self.scene_graph.platforms[platform.name]
        self.scene_graph.update_platform_children()

        self.state = (
            InteractStates.HOLDING_EMPTY_PLATFORM
            if len(self.platform.children) == 0
            else InteractStates.HOLDING_OCCUPIED_PLATFORM
        )

        return 0

    def __place_empty_platform(self, freespace_list):
        """
        place the object on empty platform

        Args:
        The number

        """
        platform = self.platform_list[self.at_place]
        platform_bbox = platform.convex_hull_2d.get_headed_bbox_instance()

        DIRECTION_MAPPING = [
            (0, 1),
            (0, 0),
            (1, 0),
            (2, 0),
            (2, 1),
            (2, 2),
            (1, 2),
            (0, 2),
            (1, 1),
        ]

        rectangle_list = []
        for c in freespace_list:
            i, j = DIRECTION_MAPPING[int(c) - 1]
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
            rectangle_list.append(rect)

        target_object_convex = self.object_in_hand.object.convex_hull_2d
        target_object_convex.heading = platform.heading
        target_rectangle = target_object_convex.get_headed_bbox_instance()

        translation_status, translation = PlacementHelper.place_rectangle(
            moving_rect=target_rectangle,
            fixed_rect_list=rectangle_list,
        )

        if translation_status == -1:
            placement_too_small_hint = self.hint_prompt_helper.generate_hint_prompts(
                hint_type="placement_failure_area_too_small",
            )
            self.vlm_interactor.add_content(
                content=placement_too_small_hint,
                role="user",
                content_type="text",
            )
            glog.info(f"{placement_too_small_hint}")
            self.state = (
                InteractStates.HOLDING_EMPTY_PLATFORM
                if len(self.platform.children) == 0
                else InteractStates.HOLDING_OCCUPIED_PLATFORM
            )

        elif translation_status == -2:

            placement_no_intersection_hint = (
                self.hint_prompt_helper.generate_hint_prompts(
                    hint_type="placement_failure_no_intersection",
                )
            )
            self.vlm_interactor.add_content(
                content=placement_no_intersection_hint,
                role="user",
                content_type="text",
            )
            glog.info(f"{placement_no_intersection_hint}")
            self.standing_direction_while_placing[self.object_in_hand.name] = (
                self.standing_direction
            )
            self.scene_graph.add_node(
                self.object_in_hand.name,
                platform.bel_object,
                platform.name,
                translation,
            )
            self.state = InteractStates.IDLE
            for i, platform in enumerate(self.platform_list):
                self.platform_list[i] = self.scene_graph.platforms[platform.name]
            self.object_in_hand = None

        elif translation_status == 0:
            placement_made_hint = self.hint_prompt_helper.generate_hint_prompts(
                hint_type="placement_made"
            )

            self.vlm_interactor.add_content(
                content=placement_made_hint,
                role="user",
                content_type="text",
            )

            glog.info(f"{placement_made_hint}")

            self.standing_direction_while_placing[self.object_in_hand.name] = (
                self.standing_direction
            )
            self.scene_graph.add_node(
                self.object_in_hand.name,
                platform.bel_object,
                platform.name,
                translation,
            )
            self.state = InteractStates.IDLE
            for i, platform in enumerate(self.platform_list):
                self.platform_list[i] = self.scene_graph.platforms[platform.name]
            self.object_in_hand = None

        place_type = "success" if translation_status != -1 else "failure"

        self.action_history_list.append(
            f"place_empty_platform_{freespace_list}_({place_type})"
        )

        pass

    def __place_occupied_platform(self, freespace_pair_list):

        platform = self.platform_list[self.at_place]
        rectangle_list = []
        target_object_convex = self.object_in_hand.object.convex_hull_2d
        target_object_convex.heading = platform.heading
        target_rectangle = target_object_convex.get_headed_bbox_instance()

        for i in range(len(freespace_pair_list)):

            # ipdb.set_trace()
            child_id = freespace_pair_list[i][0] - 1
            if child_id >= len(self.platform_list[self.at_place].children):
                glog.info("Error: Invalid child id detected.")
                return -1
            dir_id = (
                self.platform_list[self.at_place]
                .children[child_id]
                .get_freespace_id_on_picture(freespace_pair_list[i][1])
            )

            if dir_id == -1:
                glog.info("Error: Invalid freespace id detected.")
                return -1
            rect = (
                self.platform_list[self.at_place]
                .children[child_id]
                .free_space[dir_id]["Critical_space"]
            )
            rectangle_list.append(rect)

        translation_status, translation = PlacementHelper.place_rectangle(
            moving_rect=target_rectangle,
            fixed_rect_list=rectangle_list,
        )

        if translation_status == -1:
            placement_too_small_hint = self.hint_prompt_helper.generate_hint_prompts(
                hint_type="placement_failure_area_too_small",
            )
            self.vlm_interactor.add_content(
                content=placement_too_small_hint,
                role="user",
                content_type="text",
            )
            glog.info(f"{placement_too_small_hint}")
            self.state = (
                InteractStates.HOLDING_EMPTY_PLATFORM
                if len(self.platform.children) == 0
                else InteractStates.HOLDING_OCCUPIED_PLATFORM
            )

        elif translation_status == -2:

            placement_no_intersection_hint = (
                self.hint_prompt_helper.generate_hint_prompts(
                    hint_type="placement_failure_no_intersection",
                )
            )
            self.vlm_interactor.add_content(
                content=placement_no_intersection_hint,
                role="user",
                content_type="text",
            )
            glog.info(f"{placement_no_intersection_hint}")
            self.standing_direction_while_placing[self.object_in_hand.name] = (
                self.standing_direction
            )
            self.scene_graph.add_node(
                self.object_in_hand.name,
                platform.bel_object,
                platform.name,
                translation,
            )
            self.state = InteractStates.IDLE
            self.object_in_hand = None
            for i, platform in enumerate(self.platform_list):
                self.platform_list[i] = self.scene_graph.platforms[platform.name]

        elif translation_status == 0:
            placement_made_hint = self.hint_prompt_helper.generate_hint_prompts(
                hint_type="placement_made"
            )

            self.vlm_interactor.add_content(
                content=placement_made_hint,
                role="user",
                content_type="text",
            )

            glog.info(f"{placement_made_hint}")
            self.standing_direction_while_placing[self.object_in_hand.name] = (
                self.standing_direction
            )
            self.scene_graph.add_node(
                self.object_in_hand.name,
                platform.bel_object,
                platform.name,
                translation,
            )
            self.state = InteractStates.IDLE
            self.object_in_hand = None
            for i, platform in enumerate(self.platform_list):
                self.platform_list[i] = self.scene_graph.platforms[platform.name]

        place_type = "success" if translation_status != -1 else "failure"

        self.action_history_list.append(
            f"place_occupied_platform_{freespace_pair_list}_({place_type})"
        )

        pass

    def __place_anywhere(self):

        platform = self.platform
        platform_rect = self.platform.convex_hull_2d.get_headed_bbox_instance()
        fixed_rect_list = [
            child.object.convex_hull_2d.get_headed_bbox_instance()
            for child in self.platform.children
        ]
        target_object_convex = self.object_in_hand.object.convex_hull_2d
        target_object_convex.heading = self.platform.heading
        target_rectangle = target_object_convex.get_headed_bbox_instance()

        translation_status, translation = PlacementHelper.place_rectangle_anywhere(
            moving_rect=target_rectangle,
            fixed_rect_list=fixed_rect_list,
            bound_rect=platform_rect,
        )

        if translation_status == -1:
            placement_too_small_hint = self.hint_prompt_helper.generate_hint_prompts(
                hint_type="placement_failure_area_too_small",
            )
            self.vlm_interactor.add_content(
                content=placement_too_small_hint,
                role="user",
                content_type="text",
            )

            self.state = (
                InteractStates.HOLDING_EMPTY_PLATFORM
                if len(self.platform.children) == 0
                else InteractStates.HOLDING_OCCUPIED_PLATFORM
            )
            return 0

        self.action_history_list.append("place_anywhere")

        if translation_status == 0:
            placement_made_hint = self.hint_prompt_helper.generate_hint_prompts(
                hint_type="placement_made"
            )

            self.vlm_interactor.add_content(
                content=placement_made_hint,
                role="user",
                content_type="text",
            )

            glog.info(f"{placement_made_hint}")
            self.standing_direction_while_placing[self.object_in_hand.name] = (
                self.standing_direction
            )
            self.scene_graph.add_node(
                self.object_in_hand.name,
                platform.bel_object,
                platform.name,
                translation,
            )
            self.state = InteractStates.IDLE
            self.object_in_hand = None
            for i, platform in enumerate(self.platform_list):
                self.platform_list[i] = self.scene_graph.platforms[platform.name]
            return 0
        pass

    def __rotate(self):
        """
        rotate the standing direction

        Send rotation_failed_hint if unable to rotate to another direction.
        """
        previous_standing_direction = self.standing_direction
        for i in range(4):
            self.standing_direction = (self.standing_direction + 2) % 8
            if len(self.platform.standing_point_list[self.standing_direction // 2]):
                break

        if self.standing_direction == previous_standing_direction:

            rotation_failed_hint = self.hint_prompt_helper.generate_hint_prompts(
                hint_type="rotation_failed_hint",
            )
            self.vlm_interactor.add_content(
                content=rotation_failed_hint,
                role="user",
                content_type="text",
            )

        self.action_history_list.append("rotate_observation_view_of_current_platform")

        pass

    def __show_object(self, object_id, save_path):
        """
        Show the object with id = object_id.

        Args:
            object_id (_type_): _description_
        """
        platform = self.platform_list[self.at_place]
        # import ipdb
        # ipdb.set_trace()
        if object_id > 0 and object_id <= len(platform.children):
            object_to_show = platform.children[object_id - 1]
            img = object_to_show.auto_take_non_ground_object_picture(
                scene=self.scene_graph.corresponding_scene,
                view="human_focus",
                mark_object=False,
                only_mark_itself=False,
                mark_freespace=True,
                diagonal_mode="old",
                need_afford_rect=None,
                standing_direction=self.standing_direction,
                width=self.picture_width,
                height=self.picture_height,
                focus_ratio=0.5,
                fovy_range=[np.deg2rad(40), np.deg2rad(60)],
                save_path=save_path,
            )
            if img is not None:
                show_freespace_of_object_freespace_prompt = self.state_prompt_helper.generate_certain_type_prompts(
                    prompt_type="image_description",
                    context={
                        "image_type": "show_freespace_of_object",
                        "image_name": save_path,
                        "n_image": 1,
                        "image_name_list": [save_path],
                        "image_info_list": [
                            {
                                "image_name": save_path,
                                "object_idx": object_id,
                                "freespace_num": object_to_show.get_num_of_critical_space(),
                            }
                        ],
                    },
                )

                self.vlm_interactor.add_content(
                    content=show_freespace_of_object_freespace_prompt,
                    role="user",
                    content_type="text",
                )

                self.vlm_interactor.add_content(
                    content=save_path,
                    role="user",
                    content_type="image",
                )
                self.action_history_list.append(f"show_freespace_of_object_{object_id}")
            else:
                return -1
        else:
            glog.info("Invalid object id")
            return -1

        return 0
        pass

    def apply_action(self, state=None):
        """Unified action application method"""
        if state is not None:
            self.state = state

        scene_graph = self.scene_graph
        scene = self.scene_graph.corresponding_scene
        task = self.task
        platform_list = self.platform_list
        platform_name_list = self.platform_name_list = [
            platform.get_name_for_interaction() for platform in platform_list
        ]
        id = self.task_id
        current_path = self.image_path
        width, height = self.picture_width, self.picture_height

        while True:
            platform = self.platform = platform_list[self.at_place]
            platform_bel_object = self.platform_bel_object = (
                scene_graph.nodes[platform.bel_object] if self.at_place != -1 else None
            )

            # Check task completion status - key modification point
            if self.is_dual_task_mode:
                intermediate_status = self.if_intermediate_task_finished()
                if intermediate_status:
                    self.intermediate_partial_score = 4

            # Record status history
            self.status_history_list.append(
                {
                    "holding": (
                        self.object_in_hand.name
                        if self.object_in_hand is not None
                        else "nothing"
                    ),
                    "at_platform": (
                        platform.get_name_for_interaction()
                        if platform is not None
                        else "none"
                    ),
                }
            )

            self.__auto_rotate_to_standable()

            # Check maximum interaction count
            if (
                self.vlm_interactor.interaction_count
                > self.vlm_interactor.MAX_INTERACTION_COUNT
            ):
                glog.info("Maximum interaction count reached.")
                break
            if self.state == InteractStates.NAVIGATION:

                navigation_state_prompt = NavigationState.generate_prompt(
                    scene_description="",
                    platform_list=platform_name_list,
                    task_description=task.__repr_rough__()
                    + task.initial_state_information(),
                    steps_used=self.vlm_interactor.interaction_count,
                    total_steps=self.vlm_interactor.MAX_INTERACTION_COUNT,
                    location_action_list=[
                        f"goto_{platform_list[i].get_name_for_interaction()}"
                        for i in range(len(platform_list))
                    ],
                )

                navigation_state_prompt_system = navigation_state_prompt[
                    : navigation_state_prompt.find("Current task:")
                ]
                navigation_state_prompt_user = navigation_state_prompt[
                    navigation_state_prompt.find("Current task:") :
                ]
                self.vlm_interactor.add_content(
                    content=navigation_state_prompt_system,
                    role="system",
                    content_type="text",
                )
                if self.is_dual_task_mode:
                    self.__handle_ambiguous_item(
                        task=self.intermediate_task,
                        task_id=self.intermediate_task_id,
                        task_order=1,
                    )
                self.__handle_ambiguous_item(
                    task=self.task,
                    task_id=self.task_id,
                    task_order=2 if self.is_dual_task_mode else 1,
                )

                if self.use_mistake_note:
                    self._add_mistake_notes()
                # mistake_note_a = []
                # mistake_note_b = []

                # def read_reflections_from_file(
                #     filename="claude-3-5-haiku-mistake_note-05131200-replica.txt",
                # ):
                #     """Read reflection data line by line, parse each line as a list"""
                #     reflections = []
                #     try:
                #         with open(filename, "r", encoding="utf-8") as f:
                #             for line in f:
                #                 if line.strip():  # Skip empty lines
                #                     reflections.append(json.loads(line))
                #         return reflections
                #     except Exception as e:
                #         print(f"Error reading file: {e}")
                #         return []

                # if self.use_mistake_note:

                #     reflections = read_reflections_from_file()
                #     # import ipdb
                #     # ipdb.set_trace()
                #     note_num = 0
                #     for reflection_list in reflections:

                #         for reflection in reflection_list:
                #             if isinstance(reflection["content"], list):
                #                 for content in reflection["content"]:
                #                     self.vlm_interactor.add_content(
                #                         content=content,
                #                         role="user",
                #                         content_type=reflection["content_type"],
                #                     )
                #             elif isinstance(reflection["content"], str):
                #                 self.vlm_interactor.add_content(
                #                     content=reflection["content"],
                #                     role="user",
                #                     content_type=reflection["content_type"],
                #                 )
                #         note_num += 1
                #         if note_num >= self.use_mistake_note:
                #             break

                self.vlm_interactor.add_content(
                    content=navigation_state_prompt_user,
                    role="user",
                    content_type="text",
                )
                status_code, choice = self.vlm_interactor.send_content_n_request()
                action_type, action_param = NavigationState.validate_action_details(
                    choice, platform_name_list
                )

                # import ipdb
                # ipdb.set_trace()
            elif self.state == InteractStates.IDLE:

                platform_img, platform_img_list = (
                    self.scene_graph.auto_take_platform_picture(
                        platform_name=platform.name,
                        view="human_full",
                        mark_object=True,
                        mark_freespace=len(platform.children) == 0,
                        standing_direction=self.standing_direction,
                        width=width,
                        height=height,
                        focus_ratio=0.6,
                        save_path=f"{current_path}/image4interact/{self.model}/Task{id}_Idle_{self.vlm_interactor.interaction_count}.png",
                    )
                )
                n_platform_img_list = len(platform_img_list)
                image_info_list = [
                    {
                        "image_name": f"{current_path}/image4interact/{self.model}/Task{id}_Idle_{self.vlm_interactor.interaction_count}_{(i+1)}_out_of_{ n_platform_img_list}.png",
                        "image_type": "show_platform",
                        "platform_name": platform.get_name_for_interaction(),
                    }
                    for i in range(len(platform_img_list))
                ]
                idle_state_prompt = IdleState.generate_prompt(
                    scene_description="",
                    platform_list=platform_name_list,
                    task_description=task.__repr_rough__(),
                    steps_used=self.vlm_interactor.interaction_count,
                    total_steps=self.vlm_interactor.MAX_INTERACTION_COUNT,
                    platform_name=platform.get_name_for_interaction(),
                    holding_object=(
                        self.object_in_hand.get_name_for_interaction()
                        if self.object_in_hand is not None
                        else "nothing" if self.object_in_hand is not None else "nothing"
                    ),
                    image_type="show_platform",
                    image_info_list=image_info_list,
                    n_image=n_platform_img_list,
                    image_name_list=[
                        image_info["image_name"] for image_info in image_info_list
                    ],
                    location_action_list=[
                        f"goto_{platform_list[i].get_name_for_interaction()}"
                        for i in range(len(platform_list))
                    ],
                    object_action_list=(
                        [
                            f"pick_up_object_{i+1}_of_current_platform"
                            for i in range(len(platform.children))
                        ]
                        if len(platform.children) > 0
                        else []
                    ),
                    show_freespace_of_object_action_list=(
                        [
                            f"show_freespace_of_object_{i+1}_of_current_platform"
                            for i in range(len(platform.children))
                        ]
                        if len(platform.children) > 0
                        else []
                    ),
                )
                self.vlm_interactor.add_content(
                    content=idle_state_prompt,
                    role="user",
                    content_type="text",
                )

                for i in range(len(platform_img_list)):
                    self.vlm_interactor.add_content(
                        content=image_info_list[i]["image_name"],
                        role="user",
                        content_type="image",
                    )
                status_code, choice = self.vlm_interactor.send_content_n_request()
                self.vlm_interactor.clear_history_pictures()
                action_type, action_param = IdleState.validate_action_details(
                    action_str=choice,
                    platform_name_list=platform_name_list,
                    object_num=len(platform.children),
                )
                # import ipdb
                # ipdb.set_trace()

            elif self.state == InteractStates.HOLDING_EMPTY_PLATFORM:
                platform_img, platform_img_list = (
                    self.scene_graph.auto_take_platform_picture(
                        platform_name=platform.name,
                        view="human_full",
                        mark_object=True,
                        mark_freespace=len(platform.children) == 0,
                        standing_direction=self.standing_direction,
                        width=width,
                        height=height,
                        focus_ratio=0.6,
                        save_path=f"{current_path}/image4interact/{self.model}/Task{id}_HoldingAtEmptyPlatform_{self.vlm_interactor.interaction_count}.png",
                    )
                )
                n_platform_img_list = len(platform_img_list)
                image_info_list = [
                    {
                        "image_name": f"{current_path}/image4interact/{self.model}/Task{id}_HoldingAtEmptyPlatform_{self.vlm_interactor.interaction_count}_{(i+1)}_out_of_{ n_platform_img_list}.png",
                        "image_type": "show_platform",
                        "platform_name": platform.get_name_for_interaction(),
                    }
                    for i in range(len(platform_img_list))
                ]
                holding_state_prompt = HoldingEmptyPlatformState.generate_prompt(
                    scene_description="",
                    platform_list=platform_name_list,
                    task_description=task.__repr_rough__(),
                    steps_used=self.vlm_interactor.interaction_count,
                    total_steps=self.vlm_interactor.MAX_INTERACTION_COUNT,
                    platform_name=platform.get_name_for_interaction(),
                    holding_object=(
                        self.object_in_hand.get_name_for_interaction()
                        if self.object_in_hand is not None
                        else "nothing"
                    ),
                    image_type="show_platform",
                    image_info_list=image_info_list,
                    n_image=n_platform_img_list,
                    image_name_list=[
                        image_info["image_name"] for image_info in image_info_list
                    ],
                    location_action_list=[
                        f"goto_{platform_list[i].get_name_for_interaction()}"
                        for i in range(len(platform_list))
                    ],
                    show_freespace_of_object_action_list=(
                        [
                            f"show_freespace_of_object_{i+1}_of_current_platform"
                            for i in range(len(platform.children))
                        ]
                        if len(platform.children) > 0
                        else []
                    ),
                )

                self.vlm_interactor.add_content(
                    content=holding_state_prompt,
                    role="user",
                    content_type="text",
                )

                for i in range(len(platform_img_list)):
                    self.vlm_interactor.add_content(
                        content=image_info_list[i]["image_name"],
                        role="user",
                        content_type="image",
                    )
                status_code, choice = self.vlm_interactor.send_content_n_request()
                self.vlm_interactor.clear_history_pictures()
                action_type, action_param = (
                    HoldingEmptyPlatformState.validate_action_details(
                        action_str=choice,
                        platform_name_list=platform_name_list,
                        object_num=len(platform.children),
                    )
                )
                # import ipdb
                # ipdb.set_trace()
            elif self.state == InteractStates.HOLDING_OCCUPIED_PLATFORM:
                platform_img, platform_img_list = (
                    self.scene_graph.auto_take_platform_picture(
                        platform_name=platform.name,
                        view="human_full",
                        mark_object=True,
                        mark_freespace=False,
                        standing_direction=self.standing_direction,
                        width=width,
                        height=height,
                        focus_ratio=0.6,
                        save_path=f"{current_path}/image4interact/{self.model}/Task{id}_HoldingOccupiedPlatformState_{self.vlm_interactor.interaction_count}.png",
                    )
                )

                n_platform_img_list = len(platform_img_list)

                image_info_list = [
                    {
                        "image_name": f"{current_path}/image4interact/{self.model}/Task{id}_HoldingOccupiedPlatformState_{self.vlm_interactor.interaction_count}_{(i+1)}_out_of_{ n_platform_img_list}.png",
                        "image_type": "show_platform",
                        "platform_name": platform.get_name_for_interaction(),
                    }
                    for i in range(len(platform_img_list))
                ]

                available_freespace_pair_list = []
                for i, child in enumerate(platform.children):
                    cnt = child.get_num_of_critical_space()
                    for j in range(cnt):
                        available_freespace_pair_list.append((i + 1, j + 1))

                placement_empty_state_prompt = (
                    HoldingOccupiedPlatformState.generate_prompt(
                        scene_description="",
                        platform_list=platform_name_list,
                        task_description=task.__repr_rough__(),
                        steps_used=self.vlm_interactor.interaction_count,
                        total_steps=self.vlm_interactor.MAX_INTERACTION_COUNT,
                        platform_name=platform.get_name_for_interaction(),
                        holding_object=(
                            self.object_in_hand.get_name_for_interaction()
                            if self.object_in_hand is not None
                            else "nothing"
                        ),
                        image_type="show_platform",
                        image_info_list=image_info_list,
                        n_image=n_platform_img_list,
                        image_name_list=[
                            image_info["image_name"] for image_info in image_info_list
                        ],
                        available_freespace_pair_list=available_freespace_pair_list,
                        n_object=len(platform.children),
                        location_action_list=[
                            f"goto_{platform_list[i].get_name_for_interaction()}"
                            for i in range(len(platform_list))
                        ],
                        show_freespace_of_object_action_list=(
                            [
                                f"show_freespace_of_object_{i+1}_of_current_platform"
                                for i in range(len(platform.children))
                            ]
                            if len(platform.children) > 0
                            else []
                        ),
                    )
                )

                self.vlm_interactor.add_content(
                    content=placement_empty_state_prompt,
                    role="user",
                    content_type="text",
                )

                for i in range(len(platform_img_list)):
                    self.vlm_interactor.add_content(
                        content=image_info_list[i]["image_name"],
                        role="user",
                        content_type="image",
                    )
                status_code, choice = self.vlm_interactor.send_content_n_request()
                self.vlm_interactor.clear_history_pictures()
                action_type, action_param = (
                    HoldingOccupiedPlatformState.validate_action_details(
                        action_str=choice,
                        platform_name_list=platform_name_list,
                        object_num=len(platform.children),
                        freespace_pair_list=available_freespace_pair_list,
                    )
                )

            else:
                glog.info("Invalid state")
                break

            # import ipdb
            # ipdb.set_trace()

            if action_type == ActionType.CALL_END:
                break
            elif action_type == ActionType.NOT_AVAILABLE:
                not_available_action_hint = (
                    self.hint_prompt_helper.generate_hint_prompts(
                        hint_type="not_available_action_hint",
                    )
                )
                self.vlm_interactor.add_content(
                    content=not_available_action_hint, role="user", content_type="text"
                )
            elif action_type == ActionType.PLACE_IDLE:
                placement_failure_idle_hint = (
                    self.hint_prompt_helper.generate_hint_prompts(
                        hint_type="placement_failure_idle",
                    )
                )
                self.vlm_interactor.add_content(
                    content=placement_failure_idle_hint,
                    role="user",
                    content_type="text",
                )

            elif action_type == ActionType.GOTO_PLATFORM:
                self.__goto_platform(action_param)
            elif action_type == ActionType.PICKUP_OBJECT:
                self.__pickup_object(action_param)
            elif action_type == ActionType.PLACE_EMPTY:
                self.__place_empty_platform(action_param)
            elif action_type == ActionType.PLACE_OCCUPIED:
                self.__place_occupied_platform(action_param)
            elif action_type == ActionType.PLACE_ANYWHERE:
                self.__place_anywhere()
            elif action_type == ActionType.ROTATE:
                self.__rotate()
            elif action_type == ActionType.SHOW_OBJECT:
                self.__show_object(
                    action_param,
                    save_path=f"{self.executor_config.image_save_base_path}/{self.model}/Task{id}_ShowObject_object_{self.vlm_interactor.interaction_count}.png",
                    # f"{current_path}/image4interact/{self.model}/Task{id}_ShowObject_object_{self.vlm_interactor.interaction_count}.png",
                )

            else:
                self.action_history_list.append(f"**invalid_action**")
                invalid_action_hint = self.hint_prompt_helper.generate_hint_prompts(
                    hint_type="invalid_action_hint",
                )
                self.vlm_interactor.add_content(
                    content=invalid_action_hint, role="user", content_type="text"
                )
                glog.info("Invalid action type")

        self.status = self.if_task_finished()

        if not self.status and self.generate_mistake_note:
            self.scene_graph = self.initial_scene_graph
            self.scene_graph.corresponding_scene = self.scene

            def judge_dict_equal_for_partial_score(dict1, dict2):

                if len(dict1) != len(dict2):
                    return False
                for key, value in dict1.items():
                    if key not in dict2 or dict2[key] != value and dict2[key] != "any":
                        return False

                return True

            intermediate_state_list = self.task.intermediate_state_list
            state_history_list = self.status_history_list
            n_score = len(intermediate_state_list)

            correct_action_list = [[] for i in range(4)]
            self.status_history_list = self.status_history_list[1:]
            score = 0
            for i, status in enumerate(self.status_history_list):
                if score >= n_score or i >= len(self.action_history_list):
                    break
                correct_action_list[score].append(self.action_history_list[i])
                if judge_dict_equal_for_partial_score(
                    status, self.task.intermediate_state_list[score]
                ):
                    score += 1

            goal_action_list, goal_explanation_list, goal_picture_list = (
                task.generate_goal_information(
                    scene_graph=self.scene_graph,
                    current_path=self.image_path,
                    id=self.task_id,
                )
            )

            def judge_dict_equal_for_partial_score(dict1, dict2):

                if len(dict1) != len(dict2):
                    return False
                for key, value in dict1.items():
                    if key not in dict2 or dict2[key] != value and dict2[key] != "any":
                        return False

                return True

            intermediate_state_list = self.task.intermediate_state_list
            n_score = len(intermediate_state_list)

            correct_action_list = [[] for i in range(4)]
            score = 0
            for i, status in enumerate(self.status_history_list):
                if score >= n_score or i >= len(self.action_history_list):
                    break
                correct_action_list[score].append(self.action_history_list[i])
                if judge_dict_equal_for_partial_score(
                    status, self.task.intermediate_state_list[score]
                ):
                    score += 1

            correct_action_list[score] = []

            action_list = []
            for i, actions in enumerate(correct_action_list):
                if len(actions) > 0:
                    action_list.extend(actions)

            self.task.generate_default_intermediate_state()

            mistake_note_prompt = f"We also provide a task you had failed in another scene, including the steps you've made and a correct way to finish the task. Each action step includes the images you will see when making that decision. You can learn from it to better finish the coming task. The task is: {self.task.__repr_rough__()}, and your action history list is: {self.action_history_list}"

            partial_score_prompt = (
                self.reflection_prompt_helper.generate_reflection_prompt(
                    prompt_type="partial_score",
                    action_list=action_list,
                    correct_state_list=self.task.intermediate_state_repr_list[:score],
                    failed_state_list=self.task.intermediate_state_repr_list[score:],
                )
            )

            one_possible_correct_answer_prompt = (
                self.reflection_prompt_helper.generate_reflection_prompt(
                    prompt_type="one_possible_correct_answer",
                    solution_list="",
                    failed_state_list=self.task.intermediate_state_repr_list[score:],
                )
            )

            reflection_conversation_list = []

            reflection_conversation_list.append(
                {
                    "role": "system",
                    "content": mistake_note_prompt,
                    "content_type": "text",
                }
            )

            reflection_conversation_list.append(
                {
                    "role": "system",
                    "content": partial_score_prompt,
                    "content_type": "text",
                }
            )

            reflection_conversation_list.append(
                {
                    "role": "system",
                    "content": one_possible_correct_answer_prompt,
                    "content_type": "text",
                }
            )

            n = len(goal_action_list)
            for i in range(score, n):

                for goal_picture in goal_picture_list[i]:
                    reflection_conversation_list.append(
                        {
                            "role": "system",
                            "content": goal_picture,
                            "content_type": "image",
                        }
                    )

                reflection_conversation_list.append(
                    {
                        "role": "system",
                        "content": goal_explanation_list[i],
                        "content_type": "text",
                    }
                )
                reflection_conversation_list.append(
                    {
                        "role": "system",
                        "content": f"choose action: {goal_action_list[i]}",
                        "content_type": "text",
                    }
                )
                pass

            def write_reflection_to_file(
                reflection_conversation_list,
                filename=self.executor_config.reflection_txt_save_path,
            ):

                try:
                    with open(filename, "a", encoding="utf-8") as f:
                        # Use json.dumps to convert the entire list to a single-line JSON string
                        json_str = json.dumps(
                            reflection_conversation_list, ensure_ascii=False
                        )
                        f.write(
                            json_str + "\n"
                        )  # Add newline to separate different records
                except Exception as e:
                    print(f"Error writing to file: {e}")

            write_reflection_to_file(reflection_conversation_list)
            # import ipdb
            # ipdb.set_trace()
            pass

        self.partial_score += self.status
        print(
            self.status,
            f"{self.partial_score}/{len(self.task.intermediate_state_list) + 1}",
        )
        return self.status

    def _add_mistake_notes(self):
        """Add mistake notes"""

        def read_reflections_from_file(
            filename=self.executor_config.reflection_txt_load_path,
        ):
            reflections = []
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            reflections.append(json.loads(line))
                return reflections
            except Exception as e:
                glog.error(f"Error reading file: {e}")
                return []

        if self.use_mistake_note:
            reflections = read_reflections_from_file()
            note_num = 0
            for reflection_list in reflections:
                for reflection in reflection_list:
                    if isinstance(reflection["content"], list):
                        for content in reflection["content"]:
                            self.vlm_interactor.add_content(
                                content=content,
                                role="user",
                                content_type=reflection["content_type"],
                            )
                    elif isinstance(reflection["content"], str):
                        self.vlm_interactor.add_content(
                            content=reflection["content"],
                            role="user",
                            content_type=reflection["content_type"],
                        )
                note_num += 1
                if note_num >= self.use_mistake_note:
                    break


def main():
    pass
