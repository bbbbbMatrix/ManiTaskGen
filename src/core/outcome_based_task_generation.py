"""
outcome_based_task_generation/outcome_based_task_generation.py

This module provides classes and functions for generating outcome-based tasks given ManiTaskOT-200.

Not finished, mainly for the tasks involving 1 type of object/platform/category.
"""

import re
import glog
import random
import numpy as np


class OBTPlatform:
    def __init__(
        self,
        platform_index: int,
        platform_name: str = None,
        major_category_list: list = [],
        minor_category_list: list = [],
        single_object_list: list = [],
    ):
        """
        Initializes the OutcomeBasedTaskPlatform with a specific platform index.

        Args:
            platform_index (int): The index of the platform.
        """
        self.platform_index = platform_index
        self.platform_name = platform_name
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
        major_category_list: list = [],
        minor_category_list: list = [],
        layer_num: int = 0,
    ):
        """
        Initializes the OutcomeBasedTaskMultilayerObject with a specific object index.

        Args:
            object_index (int): The index of the object.
        """
        self.object_index = object_index
        self.major_category_list = major_category_list
        self.minor_category_list = minor_category_list
        self.layer_num = layer_num
        self.multilayer_object_name = multilayer_object_name

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
                + f"_{random.randint(self.layer_num - 1)}_layer"
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


class OBTRoomMajorCategory:
    def __init__(self, object_index: int, major_category: str = None):
        """
        Initializes the OutcomeBasedTaskRoomMajorCategory with a specific object index.

        Args:
            object_index (int): The index of the object.
        """
        self.object_index = object_index
        self.major_category = major_category


class OBTRoomMinorCategory:
    def __init__(self, object_index: int, minor_category: str = None):
        """
        Initializes the OutcomeBasedTaskRoomMinorCategory with a specific object index.

        Args:
            object_index (int): The index of the object.
        """
        self.object_index = object_index
        self.minor_category = minor_category


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
        platform_name_list = [platform.platform_name for platform in platform_list]
        platform_object_list = []
        multilayer_object_name_list = [
            multilayer_object.multilayer_object_name
            for multilayer_object in multilayer_object_list
        ]
        multilayer_feature = (
            multilayer_object_list[0].random_pick(self.require_multilayer_object_info)
            if len(multilayer_object_list) > 0
            else None
        )
        room_major_category_list = [
            room_major_category.major_category
            for room_major_category in room_object_list
        ]
        room_minor_category_list = [
            room_minor_category.minor_category
            for room_minor_category in room_object_list
        ]

        for id, platform in enumerate(platform_list):
            requirement = self.require_platform_list[id]
            major, minor, single = platform.random_pick(
                requirement[0], requirement[1], requirement[2]
            )
            platform_object_list.append([major, minor, single])

        def replace_placeholder(match):
            placeholder = match.group(0)
            if placeholder.startswith("[PLATFORM"):
                index = int(placeholder[9:-1])
                if index < len(platform_name_list):
                    return platform_name_list[index]
                else:
                    return f"PLATFORM_NAME_{index}_NOT_FOUND"
            elif placeholder.startswith("[MULTILAYER-OBJECT"):
                index = 0
                if index < len(multilayer_object_name_list):
                    return multilayer_object_name_list[index]
                else:
                    return "MULTILAYER_OBJECT_NOT_FOUND"
            elif placeholder.startswith("[SUB-OBJECT-CATEGORY"):
                index = int(placeholder[20:-1])
                if index < len(room_major_category_list):
                    return room_major_category_list[index]
                else:
                    return f"ROOM_MAJOR_CATEGORY_{index}_NOT_FOUND"
            elif placeholder.startswith("[SUB-OBJECT"):
                index = int(placeholder[11:-1])
                if index < len(room_minor_category_list):
                    return room_minor_category_list[index]
                else:
                    return f"ROOM_MINOR_CATEGORY_{index}_NOT_FOUND"
            elif placeholder.startswith("[SUB-PLATFORM-CATEGORY-OBJECT"):
                print(placeholder)
                platform_index = int(placeholder[29:30])
                object_index = int(placeholder[30:31])
                if platform_index < len(platform_list) and object_index < len(
                    platform_list[platform_index].major_category_list
                ):
                    return platform_list[platform_index].major_category_list[
                        object_index
                    ]
                else:
                    return (
                        f"PLATFORM_{platform_index}_CATEGORY_{object_index}_NOT_FOUND"
                    )
            elif placeholder.startswith("[SUB-PLATFORM-OBJECT"):
                platform_index = int(placeholder[20:21])
                object_index = int(placeholder[21:-1])
                if platform_index < len(platform_list) and object_index < len(
                    platform_list[platform_index].minor_category_list
                ):
                    return platform_list[platform_index].minor_category_list[
                        object_index
                    ]
                else:
                    return f"PLATFORM_{platform_index}_OBJECT_{object_index}_NOT_FOUND"
            elif placeholder.startswith("[SUB-PLATFORM-SINGLE-OBJECT"):
                platform_index = int(placeholder[27:28])
                object_index = int(placeholder[28:29])
                if platform_index < len(platform_list) and object_index < len(
                    platform_list[platform_index].single_object_list
                ):
                    return platform_list[platform_index].single_object_list[
                        object_index
                    ]
                else:
                    return f"PLATFORM_{platform_index}_SINGLE_OBJECT_{object_index}_NOT_FOUND"
            elif placeholder.startswith("[MULTILAYER-OBJECT"):
                index = int(placeholder[18:-1])
                if index < len(multilayer_object_name_list):
                    return multilayer_object_name_list[index]
                else:
                    return f"MULTILAYER_OBJECT_{index}_NOT_FOUND"
            elif placeholder.startswith("[TOP-LAYER]"):
                return multilayer_feature["top_layer_name"]
            elif placeholder.startswith("[SPECIFIC-LAYER]"):
                return multilayer_feature["specific_layer_name"]
            elif placeholder.startswith("[SUB-LAYER-CATEGORY-OBJECT"):
                object_index = int(placeholder[26:-1])
                return multilayer_feature["category_name_list"][object_index]
            elif placeholder.startswith("[SUB-LAYER-OBJECT"):
                object_index = int(placeholder[17:-1])
                return multilayer_feature["small_category_name_list"][object_index]
            else:
                return placeholder  # In case there's a new placeholder

        task_description = re.sub(r"\[.*?\]", replace_placeholder, self.task_pattern)
        print(f"Generated Task Description: {task_description}")
        return task_description

    def generate_task_images(
        self,
        scene_graph_tree,
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
            ) in scene_graph_tree.platforms.items():
                if (
                    platform_in_tree.get_name_for_interaction()
                    == self.platform_name_list[id]
                ):
                    platform = platform_in_tree
                    break

            if platform is not None:
                platform_img, platform_img_list = (
                    self.scene_graph_tree.auto_take_platform_picture(
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
            ) in scene_graph_tree.multilayer_objects.items():
                if (
                    multilayer_object_in_tree.get_name_for_interaction()
                    == self.multilayer_object_name_list[id]
                ):
                    multilayer_object = multilayer_object_in_tree
                    break

            if multilayer_object is not None:
                img = multilayer_object.auto_take_non_ground_object_picture(
                    scene=scene_graph_tree.corresponding_scene,
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
    def generate_task_feasibility_assessment_conversation(
        task_description,
        task_type,  # "platform" or "multilayer" or "combined"
        platform_images=None,
        object_images=None,
        multilayer_images=None,
    ):
        """
        Generate a conversation list for VLM task feasibility assessment.

        Args:
            task_description (str): Description of the task to be evaluated
            task_type (str): Type of task - "platform", "multilayer", or "combined"
            platform_images (dict): Dict of platform images with metadata {platform_name: {"image": image_path, "metadata": {...}}}
            object_images (dict): Dict of object images with metadata
            multilayer_images (dict): Dict of multilayer object images with metadata

        Returns:
            list: A list of conversation dictionaries for the VLM
        """
        conversation_list = []

        # Initial system prompt based on task type
        if task_type == "platform":
            system_prompt = f"""Please evaluate the feasibility of the following platform-based task:

        Task: {task_description}

        I'll provide images of relevant platforms and objects. Based on these images, please assess whether this task can be completed successfully.

        Assessment criteria:
        1. Are all required objects present in the scene?
        2. Is there sufficient free space on the target platform for placement?
        3. Are the spatial relationships between objects as required by the task achievable?
        4. Can the objects be physically manipulated as needed (considering size, shape, and stability)?
        5. Would completing the task create any unstable or physically impossible arrangements?

        Please provide a detailed assessment and conclude whether the task is:
        - Feasible (can be completed as described)
        - Partially feasible (can be completed with modifications)
        - Not feasible (cannot be completed)

        Include specific reasons for your conclusion based on the visual evidence."""

        elif task_type == "multilayer":
            system_prompt = f"""Please evaluate the feasibility of the following multi-layer object task:

        Task: {task_description}

        I'll provide images of the multi-layer object structure and its components. Based on these images, please assess whether this task can be completed successfully.

        Assessment criteria:
        1. Does the multi-layer structure have the necessary layers for task completion?
        2. Is there sufficient free space on the target layer(s) for object placement?
        3. Can objects be physically placed as required by the task (considering size, shape, and stability)?
        4. Would the final arrangement maintain structural stability of the multi-layer object?
        5. Are there any accessibility issues that would prevent task completion?

        Please provide a detailed assessment and conclude whether the task is:
        - Feasible (can be completed as described)
        - Partially feasible (can be completed with modifications)
        - Not feasible (cannot be completed)

        Include specific reasons for your conclusion based on the visual evidence."""

        else:  # combined
            system_prompt = f"""Please evaluate the feasibility of the following task involving both platforms and multi-layer objects:

        Task: {task_description}

        I'll provide images of relevant platforms, multi-layer objects, and individual objects in the scene. Based on these images, please assess whether this task can be completed successfully.

        Assessment criteria:
        1. Are all required objects and structures present in the scene?
        2. Is there sufficient free space on the target platform/layer for placement?
        3. Are the spatial relationships required by the task physically achievable?
        4. Can objects be moved between platforms and multi-layer structures as needed?
        5. Would the final arrangement be stable and physically realistic?

        Please provide a detailed assessment and conclude whether the task is:
        - Feasible (can be completed as described)
        - Partially feasible (can be completed with modifications)
        - Not feasible (cannot be completed)

        Include specific reasons for your conclusion based on the visual evidence."""

        # Add initial system prompt to conversation
        conversation_list.append(
            {"role": "system", "content": system_prompt, "content_type": "text"}
        )

        # Add platform images if available
        if platform_images:
            for platform_name, platform_data in platform_images.items():
                # Add platform image
                conversation_list.append(
                    {
                        "role": "system",
                        "content": platform_data["image"],
                        "content_type": "image",
                    }
                )

                # Add platform image explanation
                platform_description = f"""This image shows {platform_name}, a platform in the current scene.

    Key information about this platform:
    - Type: {platform_data['metadata'].get('type', 'Unknown')}
    - Location: {platform_data['metadata'].get('location', 'Unknown')}
    - Current objects: {', '.join(platform_data['metadata'].get('objects', ['None']))}
    - Available free spaces: {', '.join(platform_data['metadata'].get('free_spaces', ['None']))}

    The platform {platform_data['metadata'].get('space_status', 'has sufficient')} space for additional object placement."""

                conversation_list.append(
                    {
                        "role": "system",
                        "content": platform_description,
                        "content_type": "text",
                    }
                )

        # Add multilayer object images if available
        if multilayer_images:
            for layer_name, layer_data in multilayer_images.items():
                # Add layer image
                conversation_list.append(
                    {
                        "role": "system",
                        "content": layer_data["image"],
                        "content_type": "image",
                    }
                )

                # Add layer image explanation
                layer_description = f"""This image shows {layer_name} of the {layer_data['metadata'].get('parent_name', 'multilayer object')}.

    Key information about this layer:
    - Position in structure: {layer_data['metadata'].get('position', 'Unknown')}
    - Current objects: {', '.join(layer_data['metadata'].get('objects', ['None']))}
    - Available free spaces: {', '.join(layer_data['metadata'].get('free_spaces', ['None']))}

    This layer {layer_data['metadata'].get('space_status', 'has sufficient')} space for additional object placement."""

                conversation_list.append(
                    {
                        "role": "system",
                        "content": layer_description,
                        "content_type": "text",
                    }
                )

        # Add individual object images if available
        if object_images:
            for object_name, object_data in object_images.items():
                # Add object image
                conversation_list.append(
                    {
                        "role": "system",
                        "content": object_data["image"],
                        "content_type": "image",
                    }
                )

                # Add object image explanation
                object_description = f"""This image shows {object_name} currently located on {object_data['metadata'].get('location', 'Unknown')}.

    Key information about this object:
    - Type: {object_data['metadata'].get('type', 'Unknown')}
    - Size: {object_data['metadata'].get('size', 'Unknown')}
    - Position: {object_data['metadata'].get('position', 'Unknown')}
    - Free spaces around it: {', '.join(object_data['metadata'].get('free_spaces', ['None']))}

    The object {object_data['metadata'].get('mobility_status', 'can')} be moved as required by the task."""

                conversation_list.append(
                    {
                        "role": "system",
                        "content": object_description,
                        "content_type": "text",
                    }
                )

        # Final analysis request
        conversation_list.append(
            {
                "role": "system",
                "content": "Based on all the images and information provided, please analyze the feasibility of the task. Focus on whether the objects can be arranged as required, whether there's sufficient space, and whether the result would be stable and physically realistic.",
                "content_type": "text",
            }
        )

        return conversation_list

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
            r"\[SUB-PLATFORM-CATEGORY-OBJECT(\d+)(\d+)\]", pattern_string
        )
        for p_idx_str, obj_idx_str in category_matches:
            p_idx = int(p_idx_str)
            obj_idx = int(obj_idx_str)
            ensure_platform_details(p_idx)
            if p_idx not in explicitly_declared_platforms:
                glog.warning(
                    f"Platform {p_idx} inferred from [SUB-PLATFORM-CATEGORY-OBJECT{p_idx_str}{obj_idx_str}] "
                    f'but not explicitly declared with [PLATFORM{p_idx}]. Pattern: "{pattern_string}"'
                )
            if obj_idx not in platform_details[p_idx]["seen_category_indices"]:
                platform_details[p_idx]["seen_category_indices"].add(obj_idx)
                platform_details[p_idx]["counts"][0] = max(
                    platform_details[p_idx]["counts"][0], obj_idx + 1
                )

        small_category_matches = re.findall(
            r"\[SUB-PLATFORM-OBJECT(\d+)(\d+)\]", pattern_string
        )
        for p_idx_str, obj_idx_str in small_category_matches:
            p_idx = int(p_idx_str)
            obj_idx = int(obj_idx_str)
            ensure_platform_details(p_idx)
            if p_idx not in explicitly_declared_platforms:
                glog.warning(
                    f"Platform {p_idx} inferred from [SUB-PLATFORM-OBJECT{p_idx_str}{obj_idx_str}] "
                    f'but not explicitly declared with [PLATFORM{p_idx}]. Pattern: "{pattern_string}"'
                )
            if obj_idx not in platform_details[p_idx]["seen_small_category_indices"]:
                platform_details[p_idx]["seen_small_category_indices"].add(obj_idx)
                platform_details[p_idx]["counts"][1] = max(
                    platform_details[p_idx]["counts"][1], obj_idx + 1
                )

        single_object_matches = re.findall(
            r"\[SUB-PLATFORM-SINGLE-OBJECT(\d+)(\d+)\]", pattern_string
        )
        for p_idx_str, obj_idx_str in single_object_matches:
            p_idx = int(p_idx_str)
            obj_idx = int(obj_idx_str)
            ensure_platform_details(p_idx)
            if p_idx not in explicitly_declared_platforms:
                glog.warning(
                    f"Platform {p_idx} inferred from [SUB-PLATFORM-SINGLE-OBJECT{p_idx_str}{obj_idx_str}] "
                    f'but not explicitly declared with [PLATFORM{p_idx}]. Pattern: "{pattern_string}"'
                )
            if obj_idx not in platform_details[p_idx]["seen_single_object_indices"]:
                platform_details[p_idx]["seen_single_object_indices"].add(obj_idx)
                platform_details[p_idx]["counts"][2] = max(
                    platform_details[p_idx]["counts"][2], obj_idx + 1
                )

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

        requires_top_layer = bool(re.search(r"\[TOP-LAYER\]", pattern_string))
        requires_specific_layer = bool(re.search(r"\[SPECIFIC-LAYER\]", pattern_string))

        # Regex based on pattern.txt (e.g. SUB-LAYER-CATEGORY-OBJECT0 - no 'S')
        category_object_indices = set(
            re.findall(r"\[SUB-LAYER-CATEGORY-OBJECT(\d+)\]", pattern_string)
        )
        category_objects_count = len(category_object_indices)

        small_object_indices = set(
            re.findall(r"\[SUB-LAYER-OBJECT(\d+)\]", pattern_string)
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
        small_category_indices = set(re.findall(r"\[SUB-OBJECT(\d+)\]", pattern_string))

        return {
            "large_category_objects_count": len(large_category_indices),
            "small_category_objects_count": len(small_category_indices),
        }


class OutcomeBasedTaskGenerator:
    def __init__(self, task_pattern_file: str = None, scene_graph_tree=None):
        """
        Initializes the OutcomeBasedTaskGenerator with a file containing task patterns.

        Args:
            task_pattern_file (str): The file containing task patterns.
        """
        self.task_pattern_file = task_pattern_file
        self.task_pattern_list = []

        self.scene_graph_tree = scene_graph_tree
        self.platform_list = []
        self.multilayer_object_list = []
        self.room_major_category_list = []
        self.room_minor_category_list = []

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

    def parse_scene_graph_tree(self):

        platform_list = []
        multilayer_object_list = []
        room_major_category_list = []
        room_minor_category_list = []

        major_category_in_tree_list, minor_category_in_tree_list = (
            self.scene_graph_tree.get_node_categories_in_scene()
        )

        for id, major_category in enumerate(major_category_in_tree_list):
            room_major_category_list.append(
                OBTRoomMajorCategory(object_index=id, major_category=major_category)
            )

        for id, minor_category in enumerate(minor_category_in_tree_list):
            room_minor_category_list.append(
                OBTRoomMinorCategory(object_index=id, minor_category=minor_category)
            )

        for platform_name, platform in self.scene_graph_tree.platforms.items():

            major_category_list, minor_category_list, single_object_list = (
                platform.get_categories_and_objects()
            )
            platform_list.append(
                OBTPlatform(
                    platform_index=len(platform_list),
                    platform_name=platform.get_name_for_interaction(),
                    major_category_list=major_category_list,
                    minor_category_list=minor_category_list,
                    single_object_list=single_object_list,
                )
            )

        for node_name, node in self.scene_graph_tree.nodes.items():
            if node.is_multilayer_object():
                major_category_list, minor_category_list = (
                    node.get_categories_and_objects_for_mlo()
                )
                multilayer_object_list.append(
                    OBTMultilayerObject(
                        multilayer_object_name=node.name,
                        object_index=len(multilayer_object_list),
                        major_category_list=major_category_list,
                        minor_category_list=minor_category_list,
                        layer_num=len(node.own_platform),
                    )
                )

        self.platform_list = platform_list
        self.multilayer_object_list = multilayer_object_list
        self.room_major_category_list = room_major_category_list
        self.room_minor_category_list = room_minor_category_list

        return (
            platform_list,
            multilayer_object_list,
            room_major_category_list,
            room_minor_category_list,
        )
        pass

    pass


def main():
    print("--- Outcome-based Task Pattern Parsing Tests ---")

    patterns_to_test = [
        "Arrange [SUB-PLATFORM-OBJECT00] on [PLATFORM0].",
        "Transform the [TOP-LAYER] of [MULTILAYER-OBJECT0] with [SUB-LAYER-CATEGORY-OBJECT0].",
        "Gather all [SUB-OBJECT-CATEGORY0] and [SUB-OBJECT0] from the room.",
        "On [PLATFORM0], use [SUB-OBJECT1] to sort [SUB-PLATFORM-OBJECT00].",
        "Clean [MULTILAYER-OBJECT0] and find [SUB-OBJECT0].",
        "A task for [PLATFORM0], [MULTILAYER-OBJECT0], and [SUB-OBJECT-CATEGORY1].",
        "Just a simple sentence with no special patterns.",
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
