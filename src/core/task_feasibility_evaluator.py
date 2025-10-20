from src.vlm_interaction.interact_prompt_helper import FeasibilityPromptFormatter
from src.vlm_interaction.vlm_interactor import InteractStatusCode
from src.utils.config_manager import get_config_manager


class TaskFeasibilityEvaluator:
    """
    Main class for evaluating task feasibility using VLM.
    """

    def __init__(self, prompt_formatter: FeasibilityPromptFormatter = None):
        self.formatter = prompt_formatter or FeasibilityPromptFormatter()

    def evaluate_task_feasibility(
        self,
        vlminteractor,
        task,
        scene_graph,
        width: int = 512,
        height: int = 512,
        save_path: str = None,
    ):
        """
        Evaluate task feasibility by sending images and prompts to VLM.
        """
        if save_path is None:
            save_path = "./image4vote/"

        # Reset conversation
        vlminteractor.conversation = []

        # Get task information
        task_description = task.task_description
        # Add system prompt
        platform_name_list = [p.platform_name for p in task.platform_list]
        multilayer_object_name_list = [
            m.multilayer_object_name for m in task.multi_layer_object_list
        ]
        system_prompt = self.formatter.get_system_prompt(
            task_description=task_description,
            platform_name_list=platform_name_list,
            multilayer_object_name_list=multilayer_object_name_list,
        )
        vlminteractor.add_content(
            content=system_prompt, role="system", content_type="text"
        )

        for platform in task.platform_list:
            self._add_platform_images(
                vlminteractor,
                task,
                platform.platform_name,
                scene_graph,
                save_path,
                width,
                height,
            )

        for ml_object in task.multi_layer_object_list:
            self._add_multilayer_images(
                vlminteractor,
                task,
                ml_object.multilayer_object_name,
                scene_graph,
                save_path,
                width,
                height,
            )

        status_code, choice = vlminteractor.send_content_n_request()

        if status_code == InteractStatusCode.SUCCESS:
            print("Request successful:", choice)
            return choice
        else:
            print("Request failed with status code:", status_code)
            return "Request failed"

    def _add_platform_images(
        self, vlminteractor, task, platform_name, scene_graph, save_path, width, height
    ):
        """
        Add platform images and descriptions to the conversation.
        """
        platform = [
            p
            for p in scene_graph.platforms.values()
            if p.get_name_for_interaction() == platform_name
        ][0]

        # Take platform picturesta
        platform_img, platform_img_list = scene_graph.auto_take_platform_picture(
            platform_name=platform.name,
            view="human_full",
            save_path=f"{save_path}{platform.name}.png",
        )

        # Add images to conversation
        n_platform_img_list = len(platform_img_list)
        if n_platform_img_list > 1:
            for i, platform_img in enumerate(platform_img_list):
                image_path = (
                    f"{save_path}{platform.name}_{i+1}_out_of_{n_platform_img_list}.png"
                )
                vlminteractor.add_content(
                    content=image_path, role="user", content_type="image"
                )
        else:
            vlminteractor.add_content(
                content=f"{save_path}{platform.name}.png",
                role="user",
                content_type="image",
            )

        # Add platform description
        child_name_list = [
            f"No. {i+1}: {child.name}" for i, child in enumerate(platform.children)
        ]
        explanation = self.formatter.get_platform_introduction(
            platform_name=platform.name,
            child_name_list=child_name_list,
            image_count=n_platform_img_list,
        )
        vlminteractor.add_content(content=explanation, role="user", content_type="text")

    def _add_multilayer_images(
        self,
        vlminteractor,
        task,
        multilayer_object_name,
        scene_graph,
        save_path,
        width,
        height,
    ):
        """
        Add multilayer object images and descriptions to the conversation.
        """
        # Add multilayer introduction
        multilayer_intro = self.formatter.get_multilayer_introduction()
        vlminteractor.add_content(
            content=multilayer_intro, role="user", content_type="text"
        )

        multilayer_object = scene_graph.nodes[multilayer_object_name]

        # Process each layer
        for layer_id, platform in enumerate(multilayer_object.own_platform):
            # Take platform picture
            platform_img, platform_img_list = scene_graph.auto_take_platform_picture(
                platform_name=platform.name,
                save_path=f"{save_path}{platform.name}.png",
                view="human_full",
            )

            # Add images to conversation
            n_platform_img_list = len(platform_img_list)
            if n_platform_img_list > 1:
                for i, platform_img in enumerate(platform_img_list):
                    image_path = f"{save_path}{platform.name}_{i+1}_out_of_{n_platform_img_list}.png"
                    vlminteractor.add_content(
                        content=image_path, role="user", content_type="image"
                    )
            else:
                vlminteractor.add_content(
                    content=f"{save_path}{platform.name}.png",
                    role="user",
                    content_type="image",
                )

            # Add layer description
            child_name_list = [
                f"No. {i+1}: {child.get_name_for_interaction()}"
                for i, child in enumerate(platform.children)
            ]
            is_top_layer = layer_id == len(multilayer_object.own_platform) - 1

            explanation = self.formatter.get_multilayer_layer_description(
                platform_name=platform.name,
                child_name_list=child_name_list,
                is_top_layer=is_top_layer,
            )
            vlminteractor.add_content(
                content=explanation, role="user", content_type="text"
            )
