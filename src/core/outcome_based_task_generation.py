"""
outcome_based_task_generation/outcome_based_task_generation.py

This module provides classes and functions for generating outcome-based tasks given ManiTaskOT-200.

Not finished, mainly for the tasks involving 1 type of object/platform/category.
"""

import re
import glog
import random
import numpy as np
from src.utils.string_convertor import StringConvertor
from src.core.gen_scene_graph import Tree

from src.vlm_interaction.interact_prompt_helper import FeasibilityPromptFormatter
from src.utils.config_manager import (
    get_config_manager,
    get_config,
    get_outcome_based_task_generation_config,
)
from src.vlm_interaction.vlm_interactor import VLMInteractor
from src.core.task_feasibility_evaluator import TaskFeasibilityEvaluator

import os
import pickle


class OBTSurfaceObject:
    def __init__(self, object_index: int, surface_object_name: str = None):
        """
        Initializes the OutcomeBasedTaskSurfaceObject with a specific object index.

        Args:
            object_index (int): The index of the object.
        """
        self.object_index = object_index
        self.surface_object_name = surface_object_name

        self.major_category = StringConvertor.get_category(surface_object_name)
        self.minor_category = StringConvertor.get_name_wo_id(surface_object_name)
        self.name = surface_object_name


class OBTPlatform:
    def __init__(
        self,
        platform,
        platform_index: int,
        platform_name: str = None,
        children_list: list = [],
    ):
        """
        Initializes the OutcomeBasedTaskPlatform with a specific platform index.

        Args:
            platform_index (int): The index of the platform.
        """
        self.platform_index = platform_index
        self.platform_name = platform_name
        major_category_list = []
        minor_category_list = []
        single_object_list = []

        for child in children_list:
            if child.major_category not in major_category_list:
                major_category_list.append(child.major_category)
            if child.minor_category not in minor_category_list:
                minor_category_list.append(child.minor_category)
            if child.surface_object_name not in single_object_list:
                single_object_list.append(child.surface_object_name)

        # unify the major category and minor category
        self.major_category_list = major_category_list
        self.minor_category_list = minor_category_list
        self.single_object_list = single_object_list

    def is_available(self, major_num=0, minor_num=0, single_num=0):
        """
        Checks if the platform is available based on the number of categories and objects.

        Args:
            major_num (int): The number of major categories.
            minor_num (int): The number of minor categories.
            single_num (int): The number of single objects.

        Returns:
            bool: True if the platform is available, False otherwise.
        """
        return (
            len(self.major_category_list) >= major_num
            and len(self.minor_category_list) >= minor_num
            and len(self.single_object_list) >= single_num
        )

    def random_pick(self, major_num=0, minor_num=0, single_num=0):
        """
        Randomly picks categories and objects from the platform.

        Args:
            major_num (int): The number of major categories to pick.
            minor_num (int): The number of minor categories to pick.
            single_num (int): The number of single objects to pick.

        Returns:
            tuple: A tuple containing the picked major categories, minor categories, and single objects.
        """
        major_category = random.sample(self.major_category_list, major_num)
        minor_category = random.sample(self.minor_category_list, minor_num)
        single_object = random.sample(self.single_object_list, single_num)

        return major_category, minor_category, single_object

    def add_major_category(self, category):
        self.major_category_list.append(category)

    def add_minor_category(self, category):
        self.minor_category_list.append(category)

    def add_single_object(self, obj):
        self.single_object_list.append(obj)


class OBTMultilayerObject:
    def __init__(
        self,
        object_index: int,
        multilayer_object_name: str = None,
        children_list: list = [],
        layer_num: int = 0,
    ):
        """
        Initializes the OutcomeBasedTaskMultilayerObject with a specific object index.

        Args:
            object_index (int): The index of the object.
        """
        self.object_index = object_index
        self.major_category_list = []
        self.minor_category_list = []
        self.layer_num = layer_num
        self.multilayer_object_name = multilayer_object_name

        for child in children_list:
            if child.major_category not in self.major_category_list:
                self.major_category_list.append(child.major_category)
            if child.minor_category not in self.minor_category_list:
                self.minor_category_list.append(child.minor_category)

    def is_available(self, major_num=0, minor_num=0):
        """
        Checks if the multilayer object is available based on the number of categories and objects.

        Args:
            major_num (int): The number of major categories.
            minor_num (int): The number of minor categories.

        Returns:
            bool: True if the multilayer object is available, False otherwise.
        """
        return (
            len(self.major_category_list) >= major_num
            and len(self.minor_category_list) >= minor_num
        )

    def random_pick(self, requirement: dict):
        top_layer_name = None
        specific_layer_name = None
        category_name_list = []
        small_category_name_list = []

        if requirement is None:
            return {
                "top_layer_name": None,
                "specific_layer_name": None,
                "category_name_list": None,
                "small_category_name_list": None,
            }

        if "requires_top_layer" in requirement and requirement["requires_top_layer"]:
            # Randomly pick from the top layer
            top_layer_name = self.multilayer_object_name + "_top_layer"
            pass
        if (
            "requires_specific_layer" in requirement
            and requirement["requires_specific_layer"]
        ):
            # Randomly pick from the specific layer
            specific_layer_name = (
                self.multilayer_object_name
                + f"_{random.randint(0,self.layer_num - 1)}th_layer"
            )
            pass
        if (
            "category_objects_count" in requirement
            and requirement["category_objects_count"] > 0
        ):
            category_name_list = random.sample(
                self.major_category_list, requirement["category_objects_count"]
            )
        if (
            "small_category_objects_count" in requirement
            and requirement["small_category_objects_count"] > 0
        ):
            small_category_name_list = random.sample(
                self.minor_category_list, requirement["small_category_objects_count"]
            )

        return {
            "top_layer_name": top_layer_name,
            "specific_layer_name": specific_layer_name,
            "category_name_list": category_name_list,
            "small_category_name_list": small_category_name_list,
        }


def compute_vote_score(verdicts):
    """Score = number of 'Feasible' verdicts (0..len(vlm_list))."""
    return sum(1 for v in verdicts if v == "Feasible")


class VLMVoter:

    def __init__(self):
        # You have to set up your API keys and the environment variables first.

        self.global_config = get_config()
        self.config = get_outcome_based_task_generation_config()
        self.vlm_list = self.config.vlm_list
        self.evaluator = TaskFeasibilityEvaluator()
        self.vlm_interactor = VLMInteractor(mode="online")

    def vote_task(self, task, scene_graph, task_id, image4vote_path):
        """Vote a task with all configured VLMs. Returns a result dict with score."""
        task_image_dir = f"{image4vote_path}/task_{task_id}"
        os.makedirs(task_image_dir, exist_ok=True)
        verdicts = []
        for vlm in self.config.vlm_list:
            self.vlm_interactor.change_model_name(vlm)
            self.vlm_interactor.model_name = vlm
            verdict = self.evaluator.evaluate_task_feasibility(
                self.vlm_interactor,
                task,
                scene_graph,
                width=512,
                height=512,
                save_path=task_image_dir,
            )
            verdicts.append({"model": vlm, "verdict": verdict})
        score = compute_vote_score([v["verdict"] for v in verdicts])
        keep_min = getattr(self.config, "keep_min_score", 2)
        return {
            "task_id": task_id,
            "task_description": task.task_description,
            "pattern": task.task_pattern.task_pattern if hasattr(task.task_pattern, "task_pattern") else str(task.task_pattern),
            "score": score,
            "verdicts": verdicts,
            "feasible": score >= keep_min,
            "image_dir": task_image_dir,
            "platforms": [p.platform_name for p in getattr(task, "platform_list", [])],
            "objects": [m.multilayer_object_name for m in getattr(task, "multi_layer_object_list", [])],
        }

    def is_task_feasible(self, task, scene_graph, task_id="legacy", image4vote_path=None):
        """Backward-compatible bool wrapper over vote_task."""
        image4vote_path = image4vote_path or self.global_config.image4vote_path
        return self.vote_task(task, scene_graph, task.task_id if hasattr(task, "task_id") and task.task_id else task_id, image4vote_path)["feasible"]


class OutcomeBasedTaskPattern:
    def __init__(self, task_pattern: str):
        """
        Initializes the OutcomeBasedTaskPattern with a specific task pattern.

        Args:
            task_pattern (str): The pattern for the task.
        """
        self.task_pattern = task_pattern
        self.task_base_type = None

        self.require_platform_list = (
            OutcomeBasedTaskPattern.parse_platform_requirements(self.task_pattern)
        )
        self.require_multilayer_object_info = (
            OutcomeBasedTaskPattern.parse_multilayer_object_requirements(
                self.task_pattern
            )
        )
        self.require_room_objects = (
            OutcomeBasedTaskPattern.parse_room_object_requirements(self.task_pattern)
        )

        self.pattern_info_dict = {}

        if self.require_multilayer_object_info is not None:
            self.task_base_type = "multi-layer-object"
        elif (
            self.require_platform_list
        ):  # If not multi-layer but has platform requirements
            self.task_base_type = "platforms"
        elif (
            self.require_room_objects["large_category_objects_count"] > 0
            or self.require_room_objects["small_category_objects_count"] > 0
        ):
            self.task_base_type = "room-objects"  # Or some other appropriate type

    def __repr__(self):
        return f"OutcomeBasedTaskPattern(task_pattern={self.task_pattern}, require_platform_list = {self.require_platform_list},require_multilayer_object_info = {self.require_multilayer_object_info},task_base_type={self.task_base_type})"

    def generate_task_description(
        self, platform_list, multilayer_object_list, room_object_list
    ):
        """
        Generates a task using the specified platforms.

        Args:
            platforms (list): The list of platforms to use for the task.
        """
        # Placeholder for actual task generation logic
        pass
        platform_name_list = []
        platform_object_list = []
        multilayer_feature = None
        multilayer_object_name_list = [
            multilayer_object.multilayer_object_name
            for multilayer_object in multilayer_object_list
        ]

        relevant_platform_list = []
        relevant_multilayer_list = []

        random_room_object_list = room_object_list.copy()
        random_multilayer_object_list = multilayer_object_list.copy()
        random_platform_list = platform_list.copy()
        random.shuffle(random_multilayer_object_list)
        random.shuffle(random_platform_list)
        random.shuffle(random_room_object_list)
        if self.require_multilayer_object_info is not None:
            for multilayer_object in random_multilayer_object_list:
                if not multilayer_object.is_available(
                    major_num=self.require_multilayer_object_info.get(
                        "category_objects_count", 0
                    ),
                    minor_num=self.require_multilayer_object_info.get(
                        "small_category_objects_count", 0
                    ),
                ):
                    continue
                multilayer_feature = multilayer_object.random_pick(
                    self.require_multilayer_object_info
                )
                relevant_multilayer_list.append(multilayer_object)
                break

            if multilayer_feature is None:
                glog.warning(
                    f"{self.task_pattern} cannot be fulfilled by the given scene."
                )
                return None, None, None

        id = 0
        for i, platform in enumerate(random_platform_list):
            if id >= len(self.require_platform_list):
                break
            requirement = self.require_platform_list[id]

            if (
                platform.is_available(requirement[0], requirement[1], requirement[2])
                is False
            ):
                continue

            major, minor, single = platform.random_pick(
                requirement[0], requirement[1], requirement[2]
            )
            platform_name_list.append(platform.platform_name)
            platform_object_list.append([major, minor, single])
            relevant_platform_list.append(platform)
            id += 1

        if id < len(self.require_platform_list) or (
            multilayer_feature is None
            and self.require_multilayer_object_info is not None
        ):
            glog.warning(f"{self.task_pattern} cannot be fulfilled by the given scene.")
            return None, None, None

        room_major_category_list = [
            room_major_category.major_category
            for room_major_category in room_object_list
        ]
        room_minor_category_list = [
            room_minor_category.minor_category
            for room_minor_category in room_object_list
        ]
        random.shuffle(room_major_category_list)
        random.shuffle(room_minor_category_list)
        if self.require_room_objects is not None:

            room_major_category_list = room_major_category_list[
                : self.require_room_objects.get("large_category_objects_count", 0)
            ]
            room_minor_category_list = room_minor_category_list[
                : self.require_room_objects.get("small_category_objects_count", 0)
            ]

            pass

        #  import ipdb; ipdb.set_trace()
        def replace_placeholder(match):
            placeholder = match.group(0)

            if placeholder.startswith("[PLATFORM"):

                platform_match = re.match(r"\[PLATFORM(\d+)\]", placeholder)
                if platform_match:
                    index = int(platform_match.group(1))
                    if index < len(platform_name_list):
                        return platform_name_list[index]
                    else:
                        return f"PLATFORM_NAME_{index}_NOT_FOUND"

            elif placeholder.startswith("[MULTILAYER-OBJECT"):
                multilayer_match = re.match(r"\[MULTILAYER-OBJECT(\d+)\]", placeholder)
                if multilayer_match:
                    index = int(multilayer_match.group(1))
                    if index < len(multilayer_object_name_list):
                        return multilayer_object_name_list[index]
                    else:
                        return f"MULTILAYER_OBJECT_{index}_NOT_FOUND"

            elif placeholder.startswith("[SUB-OBJECT-CATEGORY"):
                category_match = re.match(
                    r"\[SUB-OBJECTS?-CATEGORY(\d+)\]", placeholder
                )
                if category_match:
                    index = int(category_match.group(1))
                    if index < len(room_major_category_list):
                        return f"{room_major_category_list[index]}(s)"
                    else:
                        return f"ROOM_MAJOR_CATEGORY_{index}_NOT_FOUND"

            elif placeholder.startswith("[SUB-OBJECT"):
                object_match = re.match(r"\[SUB-OBJECTS?(\d+)\]", placeholder)
                if object_match:
                    index = int(object_match.group(1))
                    if index < len(room_minor_category_list):
                        return room_minor_category_list[index]
                    else:
                        return f"ROOM_MINOR_CATEGORY_{index}_NOT_FOUND"

            elif placeholder.startswith("[SUB-PLATFORM-CATEGORY-OBJECT"):
                category_match = re.match(
                    r"\[SUB-PLATFORM-CATEGORY-OBJECTS?(\d+)(\d+)\]", placeholder
                )
                if category_match:
                    platform_index = int(category_match.group(1))
                    object_index = int(category_match.group(2))
                    if platform_index < len(
                        platform_object_list
                    ) and object_index < len(platform_object_list[platform_index][0]):
                        return f"{platform_object_list[platform_index][0][object_index]}(s)"
                    else:
                        return f"PLATFORM_{platform_index}_CATEGORY_{object_index}_NOT_FOUND"

            elif placeholder.startswith("[SUB-PLATFORM-OBJECT"):
                object_match = re.match(
                    r"\[SUB-PLATFORM-OBJECTS?(\d+)(\d+)\]", placeholder
                )
                if object_match:
                    platform_index = int(object_match.group(1))
                    object_index = int(object_match.group(2))
                    if platform_index < len(
                        platform_object_list
                    ) and object_index < len(platform_object_list[platform_index][1]):
                        return f"{platform_object_list[platform_index][1][object_index]}(s)"
                    else:
                        return (
                            f"PLATFORM_{platform_index}_OBJECT_{object_index}_NOT_FOUND"
                        )

            elif placeholder.startswith("[SUB-PLATFORM-SINGLE-OBJECT"):
                single_match = re.match(
                    r"\[SUB-PLATFORM-SINGLE-OBJECTS?(\d+)(\d+)\]", placeholder
                )
                if single_match:
                    platform_index = int(single_match.group(1))
                    object_index = int(single_match.group(2))
                    if platform_index < len(
                        platform_object_list
                    ) and object_index < len(platform_object_list[platform_index][2]):
                        return platform_object_list[platform_index][2][object_index]
                    else:
                        return f"PLATFORM_{platform_index}_SINGLE_OBJECT_{object_index}_NOT_FOUND"

            elif placeholder.startswith("[TOP-LAYER]"):
                if multilayer_feature["top_layer_name"]:
                    return re.sub(
                        r"_platform_\d+$",
                        "'s top layer",
                        multilayer_feature["top_layer_name"],
                    )
                return multilayer_feature["top_layer_name"]

            elif placeholder.startswith("[SPECIFIC-LAYER]"):
                return multilayer_feature["specific_layer_name"]

            elif placeholder.startswith("[SUB-LAYER-CATEGORY-OBJECT"):
                layer_category_match = re.match(
                    r"\[SUB-LAYER-CATEGORY-OBJECTS?(\d+)\]", placeholder
                )
                if layer_category_match:
                    object_index = int(layer_category_match.group(1))
                    if multilayer_feature["category_name_list"] and object_index < len(
                        multilayer_feature["category_name_list"]
                    ):
                        return f"{multilayer_feature['category_name_list'][object_index]}(s)"
                    else:
                        return f"LAYER_CATEGORY_{object_index}_NOT_FOUND"

            elif placeholder.startswith("[SUB-LAYER-OBJECT"):
                layer_object_match = re.match(
                    r"\[SUB-LAYER-OBJECTS?(\d+)\]", placeholder
                )
                if layer_object_match:
                    object_index = int(layer_object_match.group(1))
                    if multilayer_feature[
                        "small_category_name_list"
                    ] and object_index < len(
                        multilayer_feature["small_category_name_list"]
                    ):
                        return f"{multilayer_feature['small_category_name_list'][object_index]}(s)"
                    else:
                        return f"LAYER_OBJECT_{object_index}_NOT_FOUND"

            else:
                return placeholder

        task_description = re.sub(r"\[.*?\]", replace_placeholder, self.task_pattern)

        print(f"Generated Task Description: {task_description}")
        return task_description, relevant_platform_list, relevant_multilayer_list

    def generate_task_images(
        self,
        scene_graph,
        platform_name_list,
        multilayer_object_list,
        room_object_list,
        width=1366,
        height=768,
        current_path=None,
    ):

        for id, platform in enumerate(platform_name_list):
            platform = None
            for (
                platform_in_tree_name,
                platform_in_tree,
            ) in scene_graph.platforms.items():
                if (
                    platform_in_tree.get_name_for_interaction()
                    == self.platform_name_list[id]
                ):
                    platform = platform_in_tree
                    break

            if platform is not None:
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

        for id, multilayer_object in enumerate(multilayer_object_list):
            multilayer_object = None
            for (
                multilayer_object_in_tree_name,
                multilayer_object_in_tree,
            ) in scene_graph.multilayer_objects.items():
                if (
                    multilayer_object_in_tree.get_name_for_interaction()
                    == self.multilayer_object_name_list[id]
                ):
                    multilayer_object = multilayer_object_in_tree
                    break

            if multilayer_object is not None:
                img = multilayer_object.auto_take_non_ground_object_picture(
                    scene=scene_graph.corresponding_scene,
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
                )

        pass

    @staticmethod
    def parse_platform_requirements(pattern_string: str) -> list:
        """ """
        platform_details = {}
        max_seen_platform_index = -1
        explicitly_declared_platforms = set()

        def ensure_platform_details(p_idx):
            nonlocal max_seen_platform_index
            max_seen_platform_index = max(max_seen_platform_index, p_idx)
            if p_idx not in platform_details:
                platform_details[p_idx] = {
                    "counts": [0, 0, 0],
                    "seen_category_indices": set(),
                    "seen_small_category_indices": set(),
                    "seen_single_object_indices": set(),
                }

        platform_direct_mentions = re.findall(r"\[PLATFORM(\d+)\]", pattern_string)
        for p_idx_str in platform_direct_mentions:
            p_idx = int(p_idx_str)
            explicitly_declared_platforms.add(p_idx)
            ensure_platform_details(p_idx)

        # Regex based on pattern.txt (e.g., SUB-PLATFORM-CATEGORY-OBJECT00 - no 'S')
        category_matches = re.findall(
            r"\[SUB-PLATFORM-CATEGORY-OBJECTS?(\d+)(\d+)\]", pattern_string
        )
        for p_idx_str, obj_idx_str in category_matches:
            p_idx = int(p_idx_str)
            obj_idx = int(obj_idx_str)
            ensure_platform_details(p_idx)
            if p_idx not in explicitly_declared_platforms:
                glog.warning(
                    f"Platform {p_idx} inferred from [SUB-PLATFORM-CATEGORY-OBJECTS{p_idx_str}{obj_idx_str}] "
                    f'but not explicitly declared with [PLATFORM{p_idx}]. Pattern: "{pattern_string}"'
                )
            if obj_idx not in platform_details[p_idx]["seen_category_indices"]:
                platform_details[p_idx]["seen_category_indices"].add(obj_idx)
                platform_details[p_idx]["counts"][0] = max(
                    platform_details[p_idx]["counts"][0], obj_idx + 1
                )

        small_category_matches = re.findall(
            r"\[SUB-PLATFORM-OBJECTS?(\d+)(\d+)\]", pattern_string
        )
        for p_idx_str, obj_idx_str in small_category_matches:
            p_idx = int(p_idx_str)
            obj_idx = int(obj_idx_str)
            ensure_platform_details(p_idx)
            if p_idx not in explicitly_declared_platforms:
                glog.warning(
                    f"Platform {p_idx} inferred from [SUB-PLATFORM-OBJECTS{p_idx_str}{obj_idx_str}] "
                    f'but not explicitly declared with [PLATFORM{p_idx}]. Pattern: "{pattern_string}"'
                )
            if obj_idx not in platform_details[p_idx]["seen_small_category_indices"]:
                platform_details[p_idx]["seen_small_category_indices"].add(obj_idx)
                platform_details[p_idx]["counts"][1] = max(
                    platform_details[p_idx]["counts"][1], obj_idx + 1
                )

        single_object_matches = re.findall(
            r"\[SUB-PLATFORM-SINGLE-OBJECTS?(\d+)(\d+)\]", pattern_string
        )
        for p_idx_str, obj_idx_str in single_object_matches:
            p_idx = int(p_idx_str)
            obj_idx = int(obj_idx_str)
            ensure_platform_details(p_idx)
            if p_idx not in explicitly_declared_platforms:
                glog.warning(
                    f"Platform {p_idx} inferred from [SUB-PLATFORM-SINGLE-OBJECTS{p_idx_str}{obj_idx_str}] "
                    f'but not explicitly declared with [PLATFORM{p_idx}]. Pattern: "{pattern_string}"'
                )
            if obj_idx not in platform_details[p_idx]["seen_single_object_indices"]:
                platform_details[p_idx]["seen_single_object_indices"].add(obj_idx)
                platform_details[p_idx]["counts"][2] = max(
                    platform_details[p_idx]["counts"][2], obj_idx + 1
                )

        # import ipdb
        # ipdb.set_trace()

        if max_seen_platform_index == -1:
            if not (
                platform_direct_mentions
                or category_matches
                or small_category_matches
                or single_object_matches
            ):
                return []
            return []

        num_platforms = max_seen_platform_index + 1
        result_list = [(0, 0, 0)] * num_platforms
        for p_idx, details in platform_details.items():
            if 0 <= p_idx < num_platforms:
                result_list[p_idx] = tuple(details["counts"])
            else:
                glog.error(
                    f"Platform index {p_idx} from platform_details is out of bounds "
                    f'for result_list of size {num_platforms}. Pattern: "{pattern_string}"'
                )
        return result_list

    @staticmethod
    def parse_multilayer_object_requirements(pattern_string: str) -> dict:
        """
        Parse the string to extract requirements for multilayer objects.
        """
        if "[MULTILAYER-OBJECT0]" not in pattern_string:
            return None

        requires_top_layer = bool(
            re.search(r"\[TOP-LAYER\]|\[TOP-SHELF\]", pattern_string)
        )
        requires_specific_layer = bool(
            re.search(r"\[SPECIFIC-LAYER\]|\[SPECIFIC-SHELF\]", pattern_string)
        )

        # Regex based on pattern.txt (e.g. SUB-LAYER-CATEGORY-OBJECT0 - no 'S')
        category_object_indices = set(
            re.findall(r"\[SUB-LAYER-CATEGORY-OBJECTS?(\d+)\]", pattern_string)
        )
        category_objects_count = len(category_object_indices)

        small_object_indices = set(
            re.findall(r"\[SUB-LAYER-OBJECTS?(\d+)\]", pattern_string)
        )
        small_category_objects_count = len(small_object_indices)

        return {
            "requires_top_layer": requires_top_layer,
            "requires_specific_layer": requires_specific_layer,
            "category_objects_count": category_objects_count,
            "small_category_objects_count": small_category_objects_count,
        }

    @staticmethod
    def parse_room_object_requirements(pattern_string: str) -> dict:
        """
        Parse the string to extract requirements for room objects.
        """
        # Regex based on pattern.txt (e.g. SUB-OBJECT-CATEGORY0 - no 'S')
        large_category_indices = set(
            re.findall(r"\[SUB-OBJECT-CATEGORY(\d+)\]", pattern_string)
        )
        small_category_indices = set(
            re.findall(r"\[SUB-OBJECTS?(\d+)\]", pattern_string)
        )

        return {
            "large_category_objects_count": len(large_category_indices),
            "small_category_objects_count": len(small_category_indices),
        }


class OutcomeBasedTask:
    def __init__(
        self,
        task_description: str = None,
        task_pattern: OutcomeBasedTaskPattern = None,
        multi_layer_object_list: list = None,
        platform_list: list = None,
        room_object_list: list = None,
        task_id: str = None,
    ):
        self.task_description = task_description
        self.task_pattern = task_pattern
        self.task_id = task_id
        self.multi_layer_object_list = (
            multi_layer_object_list if multi_layer_object_list is not None else []
        )
        self.platform_list = platform_list if platform_list is not None else []
        self.room_object_list = room_object_list if room_object_list is not None else []
        self.config = get_outcome_based_task_generation_config()
        self.task_num_per_pattern = self.config.task_num_per_pattern

    def __str__(self):
        return f"Task: {self.task_description}, Pattern: {self.task_pattern}, Multi-layer Objects: {self.multi_layer_object_list}, Platforms: {self.platform_list}, Room Objects: {self.room_object_list}"



def build_histogram(results, n_vlm=3):
    hist = {i: 0 for i in range(n_vlm + 1)}
    for r in results:
        s = r["score"]
        if s in hist:
            hist[s] += 1
    return hist


class OutcomeVotingRunner:
    """Orchestrates candidate generation -> voting -> aggregation -> outputs."""

    def __init__(self, generator, vlm_voter, scene_graph, image4vote_path, out_dir, kept_txt_path=None):
        self.generator = generator
        self.vlm_voter = vlm_voter
        self.scene_graph = scene_graph
        self.image4vote_path = image4vote_path
        self.out_dir = out_dir
        self.kept_txt_path = kept_txt_path

    def run(self):
        import os
        os.makedirs(self.out_dir, exist_ok=True)
        tasks = self.generator.generate_task_with_all_patterns()
        results = []
        for task in tasks:
            tid = task.task_id or f"t{len(results):03d}"
            results.append(self.vlm_voter.vote_task(task, self.scene_graph, tid, self.image4vote_path))
        keep_min = getattr(self.vlm_voter.config, "keep_min_score", 2)
        vlm_list = self.vlm_voter.vlm_list
        n_vlm = len(self.vlm_voter.vlm_list)
        self.write_results_json(results, os.path.join(self.out_dir, "vote_results.json"), keep_min, vlm_list, n_vlm)
        kept_txt_target = self.kept_txt_path or os.path.join(self.out_dir, "outcome_based_task.txt")
        kept_txt_dir = os.path.dirname(kept_txt_target)
        if kept_txt_dir:
            os.makedirs(kept_txt_dir, exist_ok=True)
        self.write_kept_txt(results, kept_txt_target)
        self.write_review_gallery(results, os.path.join(self.out_dir, "review_gallery.html"))
        glog.info(f"Voting done: {build_histogram(results, n_vlm=n_vlm)} kept={sum(1 for r in results if r['feasible'])}/{len(results)}")
        return results

    def write_results_json(self, results, path, keep_min_score, vlm_list, n_vlm=3):
        import json
        kept = [r for r in results if r["feasible"]]
        payload = {
            "keep_min_score": keep_min_score,
            "vlm_list": list(vlm_list),
            "histogram": build_histogram(results, n_vlm=n_vlm),
            "kept_tasks": kept,
            "tasks": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def write_kept_txt(self, results, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                if r["feasible"]:
                    f.write(f"{r['task_description']}\n")

    def write_review_gallery(self, results, path):
        import os, html as html_mod
        rows = []
        for r in sorted(results, key=lambda x: (-x["score"], x["task_id"])):
            imgs = ""
            img_dir = r.get("image_dir")
            if img_dir and os.path.isdir(img_dir):
                for fn in sorted(os.listdir(img_dir)):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                        rel = os.path.relpath(os.path.join(img_dir, fn), os.path.dirname(path))
                        imgs += f'<img src="{html_mod.escape(rel)}" style="max-width:320px;margin:4px;border:1px solid #ccc;">'
            verdicts = ", ".join(f'{v["model"]}={v["verdict"]}' for v in r.get("verdicts", []))
            kept = "KEPT" if r["feasible"] else ""
            rows.append(
                f'<div class="card score-{r["score"]}">'
                f'<h3>Score {r["score"]} <small>{kept}</small> &middot; {html_mod.escape(r["task_id"])}</h3>'
                f'<p>{html_mod.escape(r["task_description"])}</p>'
                f'<p class="v">pattern: {html_mod.escape(str(r["pattern"]))} | {html_mod.escape(verdicts)}</p>'
                f'<div>{imgs}</div></div>'
            )
        doc = (
            "<!doctype html><html><head><meta charset='utf-8'><title>Outcome Vote Review</title>"
            "<style>body{font-family:sans-serif;margin:16px} .card{border:1px solid #ddd;border-radius:6px;}"
            "padding:10px;margin:10px 0} .score-0{background:#fdd} .score-1{background:#fec}"
            " .score-2{background:#efd} .score-3{background:#dfd} .v{color:#555;font-size:small}</style></head>"
            "<body><h1>Outcome-based VLM voting review</h1>"
            + "".join(rows) + "</body></html>"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)



def generate_candidate_tasks(task_pattern_list, task_num_per_pattern, platform_list,
                             multilayer_object_list, room_object_list):
    """Generate up to task_num_per_pattern unique candidate tasks per pattern.

    Deduplicates identical task descriptions within this run. Assigns each
    surviving candidate a stable task_id. Patterns the scene cannot fulfil
    (generate_task_description returns None) are skipped.
    """
    candidates = []
    seen_descriptions = set()
    for p_idx, pattern in enumerate(task_pattern_list):
        produced = 0
        attempts = 0
        # cap attempts to avoid infinite loops when few unique tasks exist
        max_attempts = task_num_per_pattern * 6
        while produced < task_num_per_pattern and attempts < max_attempts:
            attempts += 1
            desc, rel_platforms, rel_multilayer = pattern.generate_task_description(
                platform_list=platform_list,
                multilayer_object_list=multilayer_object_list,
                room_object_list=room_object_list,
            )
            if desc is None:
                # The real generate_task_description returns None deterministically
                # when the scene cannot fulfil this pattern (no matching platform /
                # multilayer). Retrying cannot help, so stop this pattern.
                break
            if desc in seen_descriptions:
                continue
            seen_descriptions.add(desc)
            task_id = f"p{p_idx:03d}_t{produced:03d}"
            candidates.append(
                OutcomeBasedTask(
                    task_description=desc,
                    task_pattern=pattern,
                    multi_layer_object_list=rel_multilayer,
                    platform_list=rel_platforms,
                    room_object_list=room_object_list,
                    task_id=task_id,
                )
            )
            produced += 1
    return candidates


class OutcomeBasedTaskGenerator:
    def __init__(self, task_pattern_file: str = None, scene_graph=None):
        """
        Initializes the OutcomeBasedTaskGenerator with a file containing task patterns.

        Args:
            task_pattern_file (str): The file containing task patterns.
        """
        self.global_config = get_config()
        self.config = get_outcome_based_task_generation_config()
        self.task_pattern_file = (
            self.global_config.manitaskot_pattern_file
            if task_pattern_file is None
            else task_pattern_file
        )
        self.task_pattern_list = []
        self.task_num_per_pattern = self.config.task_num_per_pattern
        if task_pattern_file and os.path.exists(task_pattern_file):
            with open(task_pattern_file, "r", encoding="utf-8") as file:
                for line in file:
                    task_pattern = line.strip()
                    if task_pattern:  # Ignore empty lines
                        self.task_pattern_list.append(
                            OutcomeBasedTaskPattern(task_pattern)
                        )

        # self.task_pattern_list = [
        #     "Locate all [SUB-OBJECTS0] in the room, group them by size.",
        #     "On the top layer of [PLATFORM0], gather all loose [SUB-PLATFORM-OBJECTS00].",
        #     "Optimize shelving layout of [ALL_LAYERS] on [MULTILAYER-OBJECT0].",
        #     "Group objects on the [SPECIFIC-LAYER] of [MULTILAYER-OBJECT0] by color.",
        #     "Reorganize the contents of [MULTILAYER-OBJECT0] such that all items related to [SUB-LAYER-OBJECTS00] are grouped together on the top shelf, while [SUB-LAYER-OBJECTS01] and [SUB-LAYER-OBJECTS02] are arranged on the lower shelves.",
        #     "Reconfigure the setup of [PLATFORM0], [PLATFORM1], and [PLATFORM2] to optimize for a multi-step workflow.",
        #     "Reconfigure the items on [PLATFORM0] to establish a central focal point for [SUB-PLATFORM-SINGLE-OBJECT00].",
        # ]
        self.task_pattern_list = [
            OutcomeBasedTaskPattern(pattern) for pattern in self.task_pattern_list
        ]

        self.scene_graph = scene_graph
        self.platform_list = []
        self.multilayer_object_list = []

        self.platform_list, self.multilayer_object_list, self.room_object_list = (
            self.parse_scene_graph()
        )
        self.task_list = []

        if task_pattern_file:
            self.load_task_patterns(task_pattern_file)

    def load_task_patterns(self, task_pattern_file: str):
        """
        Loads task patterns from a specified file.

        Args:
            task_pattern_file (str): The file containing task patterns.
        """
        try:
            with open(task_pattern_file, "r", encoding="utf-8") as file:
                for line in file:
                    # Assuming each line contains a task pattern
                    task_pattern = line.strip()
                    if task_pattern:  # Ignore empty lines
                        self.task_pattern_list.append(
                            OutcomeBasedTaskPattern(task_pattern)
                        )
        except FileNotFoundError:
            print(f"Error: The file '{task_pattern_file}' was not found.")
        except Exception as e:
            print(f"An error occurred while loading task patterns: {e}")

    def generate_prompt_for_task(self, task: OutcomeBasedTaskPattern):
        """
        Generates a prompt for a given task.

        Args:
            task (OutcomeBasedTaskPattern): The task for which to generate a prompt.
        """
        # Placeholder for actual prompt generation logic

        conversation_list = []

        conversation_list.append(
            {
                "role": "system",
                "content": "Please evaluate the feasibility of the following platform-based task: Task: [TASK_DESCRIPTION]\n",
            }
        )

        return f"Prompt for task: {task.task_pattern}"

    def generate_all_possible_tasks(self):
        """
        Generates all possible tasks based on the loaded task patterns.
        """
        all_tasks = []

        if not self.platform_list or not self.multilayer_object_list:
            glog.error(
                "Platform list or multilayer object list is empty. Cannot generate tasks."
            )
            return all_tasks

    def generate_sample_task(self):
        """ """

        # Placeholder for actual task generation logic
        from itertools import combinations

        multilayer_object_list = self.multilayer_object_list
        platform_list = self.platform_list

        well_occupied_platform_list = []
        for id, platform in enumerate(self.platform_list):
            if len(platform.single_object_list) > 1:
                well_occupied_platform_list.append(platform)
        if not well_occupied_platform_list:
            glog.error("No well-occupied platforms found. Cannot generate tasks.")
            return None
        all_task_list = []
        task_pattern = self.task_pattern_list
        for task_pattern in self.task_pattern_list:
            required_platform_list = task_pattern.require_platform_list
            required_multilayer_object_list = (
                task_pattern.require_multilayer_object_info
            )
            if len(required_platform_list) != 1:
                continue
            junior_platform_combinations = combinations(
                platform_list, len(required_platform_list) - 1
            )

            for senior_platform in well_occupied_platform_list:
                # import ipdb
                # ipdb.set_trace()
                if senior_platform.is_available(
                    task_pattern.require_platform_list[0][0],
                    task_pattern.require_platform_list[0][1],
                    task_pattern.require_platform_list[0][2],
                ):

                    flag = True

                    if flag:
                        if (
                            required_multilayer_object_list is not None
                            and len(required_multilayer_object_list) > 0
                        ):
                            for multilayer_object in multilayer_object_list:
                                if (
                                    not task_pattern.require_multilayer_object_info
                                    or multilayer_object.is_available(
                                        task_pattern.require_multilayer_object_info[
                                            "category_objects_count"
                                        ],
                                        task_pattern.require_multilayer_object_info[
                                            "small_category_objects_count"
                                        ],
                                    )
                                ):
                                    # Generate a task using the selected platforms

                                    # Generate a task using the selected platforms

                                    task_description = (
                                        task_pattern.generate_task_description(
                                            platform_list=[senior_platform],
                                            multilayer_object_list=[multilayer_object],
                                            room_object_list=[],
                                        )
                                    )

                                    task = {
                                        "PLATFORM0": senior_platform,
                                        "MULTILAYER-OBJECT0": multilayer_object,
                                        "task_description": task_description,
                                        "task_pattern": task_pattern,
                                    }
                                    all_task_list.append(task)
                                    # Print or log the generated task
                        else:
                            # Generate a task using the selected platforms
                            task_description = task_pattern.generate_task_description(
                                platform_list=[senior_platform],
                                multilayer_object_list=[],
                                room_object_list=[],
                            )

                            task = {
                                "PLATFORM0": senior_platform,
                                "task_description": task_description,
                                "task_pattern": task_pattern,
                            }
                            all_task_list.append(task)
                            # Print or log the generated task

            pass

        pass
        self.all_task_list = all_task_list
        glog.info(f"Generated {len(all_task_list)} tasks.")
        return self.all_task_list

    def generate_task_with_all_patterns(self, task_num=None, desired_pattern_list=None):
        task_num = self.task_num_per_pattern if task_num is None else task_num
        patterns = (
            [self.task_pattern_list[i] for i in desired_pattern_list]
            if desired_pattern_list is not None
            else self.task_pattern_list
        )
        self.task_list = generate_candidate_tasks(
            patterns,
            task_num_per_pattern=task_num,
            platform_list=self.platform_list,
            multilayer_object_list=self.multilayer_object_list,
            room_object_list=self.room_object_list,
        )
        glog.info(f"Generated {len(self.task_list)} candidate outcome tasks.")
        return self.task_list

    def parse_scene_graph(self):

        platform_list = []
        multilayer_object_list = []
        room_object_list = []

        for platform_name, platform in self.scene_graph.platforms.items():

            child_name_list = platform.get_child_name_list()

            child_list = []

            for child_name in child_name_list:
                child_list.append(
                    OBTSurfaceObject(
                        object_index=len(child_list), surface_object_name=child_name
                    )
                )
            room_object_list.extend(child_list)

            platform_list.append(
                OBTPlatform(
                    platform=platform,
                    platform_index=len(platform_list),
                    platform_name=platform.get_name_for_interaction(),
                    children_list=child_list,
                )
            )

        for node_name, node in self.scene_graph.nodes.items():
            if node.is_multilayer_object():
                major_category_list, minor_category_list = (
                    node.get_categories_and_objects_for_mlo()
                )
                child_name_list = node.get_child_name_list()
                child_list = []
                for child_name in child_name_list:
                    child_list.append(
                        OBTSurfaceObject(
                            object_index=len(child_list), surface_object_name=child_name
                        )
                    )

                multilayer_object_list.append(
                    OBTMultilayerObject(
                        multilayer_object_name=node.name,
                        children_list=child_list,
                        object_index=len(multilayer_object_list),
                        layer_num=len(node.own_platform),
                    )
                )

        self.platform_list = platform_list
        self.multilayer_object_list = multilayer_object_list
        self.room_object_list = room_object_list

        return (platform_list, multilayer_object_list, room_object_list)
        pass

    pass


import json


def main():
    print("--- Outcome-based Task Pattern Parsing Tests ---")

    scene_graph = None

    scene_graph_pkl_load_path = "./scene_graph.pkl"

    if scene_graph_pkl_load_path is not None and os.path.exists(
        scene_graph_pkl_load_path
    ):
        glog.info(f"Loading scene graph from {scene_graph_pkl_load_path}")
        with open(scene_graph_pkl_load_path, "rb") as f:
            scene_graph = pickle.load(f)

    rename_dict = json.load(open("rename_dict_replica.json", "r"))
    scene_graph.rename_all_features(rename_dict)

    outcome_based_task_generator = OutcomeBasedTaskGenerator(scene_graph=scene_graph)

    patterns_to_test = [
        "Arrange [SUB-PLATFORM-CATEGORY-OBJECTS00] on [PLATFORM0] by height and type.",
        "Reorganize the left side of the top of [PLATFORM0].",
        "Align all [SUB-PLATFORM-OBJECTS00] on [PLATFORM0] symmetrically.",
        "On [PLATFORM0], create a symmetrical arrangement of [SUB-PLATFORM-OBJECTS00] and [SUB-PLATFORM-OBJECTS01].",
        "Organize all the items on [PLATFORM0] by separating them into categories: place all [SUB-PLATFORM-OBJECT00] in a single stack, align any [SUB-PLATFORM-OBJECT01] in a row, and move [SUB-PLATFORM-OBJECT02] to the left side.",
        "Use items from [PLATFORM0] and [PLATFORM1] to create an artistic installation on [PLATFORM2].",
        "Move frequently used items from [PLATFORM0] and [PLATFORM1] to the top of [PLATFORM2].",
        "Arrange [PLATFORM0], [PLATFORM1], and [PLATFORM2] for collaborative work.",
        "Take all [SUB-PLATFORM-OBJECTS00] from [PLATFORM0] and arrange them on [PLATFORM1].",
        "Transform [PLATFORM0] into a presentation space by placing colorful [SUB-PLATFORM-CATEGORY-OBJECT00].",
        "On [PLATFORM0], organize the [SUB-PLATFORM-CATEGORY-OBJECTS00] and [SUB-PLATFORM-CATEGORY-OBJECTS01] by size.",
        "Find all [SUB-LAYER-OBJECTS0] and [SUB-LAYER-OBJECTS1] scattered across [MULTILAYER-OBJECT0].",
        "Locate all [SUB-OBJECTS0] in the room, group them by size.",
        "Clear [PLATFORM0] by folding [SUB-PLATFORM-OBJECTS00] and placing them on [PLATFORM1].",
        "Leave room in the corner of the [PLATFORM0] for the [SUB-PLATFORM-OBJECTS10] from [PLATFORM1].",
        "Transform [PLATFORM0] into a visually appealing display by organizing [SUB-PLATFORM-CATEGORY-OBJECTS02].",
        "Gather all the [SUB-PLATFORM-OBJECTS00] from [PLATFORM0], then organize them on the top shelf of [MULTILAYER-OBJECT0].",
        "Transform the [TOP-LAYER] of [MULTILAYER-OBJECT0] by aligning all [SUB-LAYER-CATEGORY-OBJECTS00].",
        "Rearrange and optimize the [SPECIFIC-LAYER] of [MULTILAYER-OBJECT0].",
        "Gather all [SUB-PLATFORM-OBJECTS00], [SUB-PLATFORM-OBJECTS10], [SUB-PLATFORM-OBJECTS20] scattered across [PLATFORM0], [PLATFORM1], and [PLATFORM2], and arrange them on [PLATFORM3].",
        "Sort items from [PLATFORM0] and [PLATFORM1] into categories and place them on [PLATFORM2] and [PLATFORM3].",
        "Gather all [SUB-OBJECT-CATEGORY0] from the [ROOM], clean them, and place them on [PLATFORM0].",
        "Clear the top of [PLATFORM0], then reposition the [SUB-PLATFORM-SINGLE-OBJECTS00], [SUB-PLATFORM-SINGLE-OBJECTS01], and [SUB-PLATFORM-SINGLE-OBJECTS02].",
        "Conduct a thorough inspection of [PLATFORM0], placing vivid [SUB-PLATFORM-SINGLE-OBJECTS00] at eye level.",
        "Optimize shelving layout of [ALL_LAYERS] on [MULTILAYER-OBJECT0].",
        "Group objects on the [SPECIFIC-LAYER] of [MULTILAYER-OBJECT0] by color.",
        "Reorganize the contents of [MULTILAYER-OBJECT0] such that all items related to [SUB-LAYER-OBJECTS00] are grouped together on the top shelf, while [SUB-LAYER-OBJECTS01] and [SUB-LAYER-OBJECTS02] are arranged on the lower shelves.",
        "Reconfigure the setup of [PLATFORM0], [PLATFORM1], and [PLATFORM2] to optimize for a multi-step workflow.",
        "Reconfigure the items on [PLATFORM0] to establish a central focal point for [SUB-PLATFORM-SINGLE-OBJECT00].",
    ]

    for i, pattern in enumerate(patterns_to_test):
        print(f"\n--- Test Pattern {i+1} ---")
        print(f'Pattern: "{pattern}"')

        task_instance = OutcomeBasedTaskPattern(pattern)

        print(f"  Platform Requirements: {task_instance.require_platform_list}")
        print(
            f"  Multilayer Object Info: {task_instance.require_multilayer_object_info}"
        )
        print(f"  Room Object Requirements: {task_instance.require_room_objects}")
        print(f"  Determined Task Base Type: {task_instance.task_base_type}")

    outcome_based_task_generator.generate_task_with_all_patterns()

    vlm_voter = VLMVoter()
    for task in outcome_based_task_generator.task_list:
        vlm_voter.is_task_feasible(task, scene_graph)

    # Example from pattern.txt for SUB-OBJECT-CATEGORY0
    pattern_locatel = "Locate all [SUB-OBJECT0] in the room, group them by size, and place them in descending order along the edge of [PLATFORM0]."
    print(f"\n--- Test Pattern (from file) ---")
    print(f'Pattern: "{pattern_locatel}"')
    task_locatel = OutcomeBasedTaskPattern(pattern_locatel)
    print(f"  Platform Requirements: {task_locatel.require_platform_list}")
    print(f"  Multilayer Object Info: {task_locatel.require_multilayer_object_info}")
    print(f"  Room Object Requirements: {task_locatel.require_room_objects}")
    print(f"  Determined Task Base Type: {task_locatel.task_base_type}")


if __name__ == "__main__":
    main()
