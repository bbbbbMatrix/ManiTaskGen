from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import glog


@dataclass
class PromptTemplate:
    template_type: str
    content: str


@dataclass
class ImagePromptTemplate:
    base_template: PromptTemplate
    variants: Dict[str, PromptTemplate]


class PromptFormatter(ABC):
    """
    Abstract base class for formatting prompts.
    """

    @abstractmethod
    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format the prompt using the provided template and keyword arguments.
        """
        pass

    def _replace_value(self, prompt_string: str, **kwargs) -> str:
        """
        Replaces the key in the prompt string with the value.
        """
        for key, value in kwargs.items():
            if (
                isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, list)
            ):
                prompt_string = prompt_string.replace(f"{{{key}}}", str(value))
            else:
                glog.error(f"Unsupported type for key {key}: {type(value)}")
                import ipdb

                ipdb.set_trace()
        return prompt_string


class BasicPromptFormatter(PromptFormatter):
    """
    Basic implementation of the PromptFormatter.
    """

    def __init__(self, template: PromptTemplate = None):
        self.template = template

    def format_prompt(self, template: str, **kwargs) -> str:
        prompt = template
        prompt = self._replace_value(prompt, **kwargs)
        return prompt


class ImagePromptFormatter(PromptFormatter):
    """
    Implementation of the PromptFormatter for image prompts.
    Difference is that it handles different images, they have different image info, but they share the same template.
    """

    def __init__(self, template: ImagePromptTemplate = None):
        self.template = template

    def new_format_prompt(
        self, variant_type: str, image_info_list: List[Dict], shared_vars: Dict = None
    ) -> str:
        """
        Format a group of image descriptions

        Args:
            variant_type: Variant type (e.g., 'identical_object_to_move')
            image_info_list: List of image information
            shared_vars: Variables shared by all images
        """
        if variant_type not in self.template.variants:
            raise ValueError(f"Unknown variant type: {variant_type}")

        variant = self.template.variants[variant_type]
        formatted_descriptions = []

        # Validate shared variables
        if shared_vars and variant.shared_variables:
            missing_vars = variant.shared_variables - set(shared_vars.keys())
            if missing_vars:
                raise ValueError(f"Missing shared variables: {missing_vars}")

        # Process each image
        for img_info in image_info_list:
            # Validate required variables
            missing_vars = variant.required_variables - set(img_info.keys())
            if missing_vars:
                raise ValueError(
                    f"Image information missing required variables: {missing_vars}"
                )

            # Merge variables
            variables = {**(shared_vars or {}), **img_info}

            # Format variant content
            variant_content = self._replace_value(variant.content, **variables)

            # Format base template
            final_content = self._replace_value(
                self.template.base_template.content,
                image_description=variant_content,
                **variables,
            )

            formatted_descriptions.append(final_content)

        return "\n\n".join(formatted_descriptions)

    def format_prompt(
        self, template: str, image_info_list: List[Dict], **kwargs
    ) -> str:
        formatted_prompt_list = []
        for image_info in image_info_list:
            prompt = template
            prompt = self._replace_value(prompt, **image_info)
            prompt = self._replace_value(prompt, **kwargs)
            formatted_prompt_list.append(prompt)
        return "\n".join(formatted_prompt_list)


class StatePromptManager:
    regular_prompt_types = [
        "system_initialization",
        "task_instructions",
        "understanding_directions",
        "task_success_criteria",
        "current_state",
        "task_remainder",
        "possible_action_space_description",
        "possible_image_type_description",
        "important_suggestion",
    ]

    def __init__(
        self,
        prompt_templates: str = "./vlm_interactor/prompts/task_interact_prompts.json",
        formatter: PromptFormatter = {
            "basic": BasicPromptFormatter(),
            "image": ImagePromptFormatter(),
        },
    ):
        try:
            with open(prompt_templates, "r", encoding="utf-8") as f:
                self.prompt_templates = json.load(f)  # Use load instead of loads
            self.state_config = self.prompt_templates["states"]
        except FileNotFoundError:
            glog.error(f"Cannot find prompt template file: {prompt_templates}")
            raise
        except json.JSONDecodeError as e:
            glog.error(f"JSON format error: {e}")
            raise
        except KeyError as e:
            glog.error(f"Missing required key in prompt templates: {e}")
            raise

        self.formatter = formatter

        try:
            self.state_config = self.prompt_templates["states"]
        except KeyError as e:
            glog.error(f"Missing key in prompt templates: {e}")

    def __check_variables(self, required_variable_list: list, context: Dict):

        missing_variable = []

        for required_variable in required_variable_list:
            if required_variable not in context:
                missing_variable.append(required_variable)

        if len(missing_variable) == 0:
            return 0, "ok"
        else:
            return 1, f"{list(context.keys())} missing variables {missing_variable}"

    def generate_certain_type_prompts(self, prompt_type: str, context: Dict) -> str:
        """
        Generate prompts based on the prompt type and context.
        """
        try:
            if prompt_type in StatePromptManager.regular_prompt_types:
                return self.__handle_regular_prompt(prompt_type, context)
            elif "action_space" in prompt_type:
                action_space_type = prompt_type[
                    prompt_type.find("action_space") + len("action_space") + 1 :
                ]
                return self.__handle_action_space(action_space_type, context)
            elif prompt_type == "image_description":
                # Image name will be a list, but the names will share the same template.
                image_type = context["image_type"]

                try:
                    assert (
                        "image_info_list" in context
                    ), "Image info list is missing in context"
                    assert isinstance(
                        context["image_info_list"], list
                    ), "Image info list should be a list"
                    assert (
                        len(context["image_info_list"]) > 0
                    ), "Image info list is empty"
                    return self.__handle_image_description(image_type, context)
                except AssertionError as e:
                    glog.error(f"Assertion Error: {e}")
            elif prompt_type == "idle":
                return self._handle_idle(context)
            else:
                glog.error(f"Unknown prompt type: {prompt_type}")
                return ""
        except KeyError as e:
            glog.error(f"Missing key in prompt templates: {e}")
            import ipdb

            ipdb.set_trace()
        return ""

    def generate_state_prompts(self, state: str, context: Dict) -> str:
        """
        Generate prompts based on the current state and context.
        """
        prompt_list = []
        if state in self.state_config:
            try:
                state_config = self.state_config[state]["prompts"]
                for prompt_type in state_config:
                    prompt_list.append(
                        self.generate_certain_type_prompts(prompt_type, context)
                    )

                return "\n".join(prompt_list)
            except KeyError as e:
                glog.error(f"Missing key in state config: {e}")
            except TypeError as e:
                glog.error(f"Type error in state config: {e}")
        else:
            glog.error(f"Unknown state: {state}")

    def __handle_regular_prompt(self, regular_prompt_type: str, context: Dict) -> str:
        """
        Handle task instructions prompts.
        """
        try:
            status, log = self.__check_variables(
                self.prompt_templates[regular_prompt_type]["required_variables"],
                context,
            )
            if status:
                glog.warning(f"handle {regular_prompt_type} log:{log}")
            template = self.prompt_templates[regular_prompt_type]["content"]

            return self.formatter["basic"].format_prompt(template, **context)
        except KeyError as e:
            glog.error(f"Missing key in {regular_prompt_type} template: {e}")

    def __handle_action_space(self, state: str, context: Dict) -> str:
        """
        Handle action space prompts.
        """

        required_vars = (
            self.prompt_templates["action_space"]["variants"][state][
                "required_variables"
            ]
            + self.prompt_templates["action_space"]["template"]["required_variables"]
        )
        if not all(var in context for var in required_vars):
            glog.error(
                f"context {context} Missing required variables for action space: {required_vars}"
            )
            return ""

        try:
            status, log = self.__check_variables(
                self.prompt_templates["action_space"]["variants"][state][
                    "required_variables"
                ],
                context,
            )
            if status:
                glog.warning(f"handle environment_description log:{log}")
            variant_content = self.prompt_templates["action_space"]["variants"][state][
                "content"
            ]
            template_content = self.prompt_templates["action_space"]["template"][
                "content"
            ]
            filled_variant = self.formatter["basic"].format_prompt(
                variant_content, **context
            )
            context["action_space_content"] = filled_variant

            final_prompt = self.formatter["basic"].format_prompt(
                template_content, filled_variant=filled_variant, **context
            )
            return final_prompt
        except KeyError as e:
            glog.error(f"Missing key in action space template: {e}")

    def __handle_image_description(self, image_type: str, context: Dict) -> str:
        """
        Handle image description prompts.
        """
        try:

            content_template = self.prompt_templates["image_description"]["template"][
                "content"
            ]
            image_type_template = self.prompt_templates["image_description"][
                "variants"
            ][image_type]["content"]
            image_list = context["image_info_list"]

            image_description_prompt = self.formatter["image"].format_prompt(
                image_type_template, image_list=image_list, **context
            )
            return self.formatter["basic"].format_prompt(
                content_template, image_description=image_description_prompt, **context
            )

        except KeyError as e:
            glog.error(f"Missing key in image description template: {e}")
            import ipdb

            ipdb.set_trace()

        return ""

    def __handle_image_name(self, context: Dict) -> str:
        try:
            content_template = self.prompt_templates["image_name"]["content"]

            image_description_prompt = self.formatter["basic"].format_prompt(
                self.prompt_templates["image_name"]["variants"]["image_name"][
                    "content"
                ],
                **context,
            )
            return image_description_prompt
        except KeyError as e:
            glog.error(f"Missing key in image name template: {e}")
        return ""

    def _handle_idle(self, context: Dict) -> str:
        """
        Handle idle state prompts.
        """
        template = self.prompt_templates["idle"]
        return self.formatter["basic"].format_prompt(template, **context)


class HintPromptManager:
    """
    Class to manage hint prompts.
    """

    def __init__(self, prompt_templates):
        prompt_templates_path = prompt_templates
        try:
            with open(prompt_templates_path, "r", encoding="utf-8") as f:
                self.prompt_templates = json.load(f)  # Use load instead of loads
            self.state_config = self.prompt_templates["states"]
        except FileNotFoundError:
            glog.error(f"Cannot find prompt template file: {prompt_templates_path}")
            raise
        except json.JSONDecodeError as e:
            glog.error(f"JSON format error: {e}")
            raise
        except KeyError as e:
            glog.error(f"Missing required key in prompt templates: {e}")
            raise

        self.formatter = {"basic": BasicPromptFormatter()}
        try:
            self.hint_config = self.prompt_templates["hints"]
        except KeyError as e:
            glog.error(f"Missing key in prompt templates: {e}")

    def generate_hint_prompts(self, **kwargs) -> str:
        """
        Generate a hint prompt using the provided keyword arguments.
        """
        try:
            hint_type = kwargs["hint_type"]
            hint_template = self.hint_config[hint_type]["content"]
            return self.formatter["basic"].format_prompt(hint_template, **kwargs)
        except KeyError as e:
            glog.error(f"Missing key in hint config: {e}")

        return self.hint_prompt.format(**kwargs)


class ReflectionPromptManager:
    """
    Class to manage reflection prompts.
    """

    regular_prompt_types = [
        "partial_score",
        "one_possible_correct_answer",
        "goto_platform",
        "pick_up_object",
    ]

    def __init__(self, prompt_templates):
        prompt_templates_path = prompt_templates
        try:
            with open(prompt_templates_path, "r", encoding="utf-8") as f:
                self.prompt_templates = json.load(f)  # Use load instead of loads
        except FileNotFoundError:
            glog.error(f"Cannot find prompt template file: {prompt_templates_path}")
            raise
        except json.JSONDecodeError as e:
            glog.error(f"JSON format error: {e}")
            raise
        except KeyError as e:
            glog.error(f"Missing required key in prompt templates: {e}")
            raise

        self.formatter = {"basic": BasicPromptFormatter()}

    def __generate_partial_score_prompt(self, **kwargs) -> str:
        """
        Generate a partial score prompt using the provided keyword arguments.
        """
        try:
            partial_score_template = self.prompt_templates["partial_score"]["content"]
            required_vars = self.prompt_templates["partial_score"]["required_variables"]
            if not all(var in kwargs for var in required_vars):
                glog.warning(
                    f"Partial score prompt is missing required variables: find {kwargs}, expect {required_vars}"
                )

            return self.formatter["basic"].format_prompt(
                partial_score_template, **kwargs
            )
        except KeyError as e:
            glog.error(f"Missing key in partial score config: {e}")
            return ""

    def __generate_one_possible_correct_action_prompt(self, **kwargs) -> str:
        """
        Generate a prompt for one possible correct action using the provided keyword arguments.
        """
        try:
            one_possible_correct_action_template = self.prompt_templates[
                "one_possible_correct_action"
            ]["content"]
            required_vars = self.prompt_templates["one_possible_correct_action"][
                "required_variables"
            ]
            if not all(var in kwargs for var in required_vars):
                glog.warning(
                    f"One possible correct action prompt is missing required variables: find {kwargs}, expect {required_vars}"
                )

            return self.formatter["basic"].format_prompt(
                one_possible_correct_action_template, **kwargs
            )
        except KeyError as e:
            glog.error(f"Missing key in one possible correct action config: {e}")
            return ""

    def __generate_action_reason_prompt(self, **kwargs) -> str:
        try:
            task_type = kwargs.get("task_type", None)
            if task_type not in "action_reason":
                glog.error(f"task_type {task_type} is not in action_reason")
                return ""
            action_reason_template = self.prompt_templates["action_reason"][task_type]
            try:
                required_vars = self.prompt_templates["action_reason"][task_type][
                    "required_variables"
                ]
                if not all(var in kwargs for var in required_vars):
                    glog.warning(
                        f"Action reason prompt is missing required variables: find {kwargs}, expect {required_vars}"
                    )
                return self.formatter["basic"].format_prompt(
                    action_reason_template, **kwargs
                )
            except KeyError as e:
                glog.error(f"Missing key in action reason {task_type} sub config: {e}")
                return ""
        except:
            glog.error(f"Missing key in action reason main config: {e}")
            return ""

    def __generate_regular_text_prompt(self, **kwargs) -> str:
        """
        Generate a regular text prompt using the provided keyword arguments.
        """
        try:
            prompt_type = kwargs.get("prompt_type", None)
            if prompt_type not in ReflectionPromptManager.regular_prompt_types:
                glog.error(f"prompt_type {prompt_type} is missing")
                return ""
            regular_text_template = self.prompt_templates[prompt_type]["content"]
            required_vars = self.prompt_templates[prompt_type]["required_variables"]
            if not all(var in kwargs for var in required_vars):
                glog.warning(
                    f"Regular text prompt is missing required variables: find {kwargs}, expect {required_vars}"
                )

            return self.formatter["basic"].format_prompt(
                regular_text_template, **kwargs
            )
        except KeyError as e:
            glog.error(f"Missing key in regular text config: {e}")
            return ""

    def generate_reflection_prompt(self, **kwargs) -> str:
        """
        Generate a reflection prompt using the provided keyword arguments.
        """

        prompt_type = kwargs.get("prompt_type", None)
        if (
            prompt_type.endswith("hint")
            or prompt_type in ReflectionPromptManager.regular_prompt_types
        ):
            return self.__generate_regular_text_prompt(**kwargs)
