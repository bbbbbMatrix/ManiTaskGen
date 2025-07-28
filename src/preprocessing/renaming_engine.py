import vlm_interactor
import json
import os
import glog
import sapien
import time

from src.vlm_interaction import vlm_interactor


class RenamingEngine:
    def __init__(self, model="GPT4o"):
        self.interactor = vlm_interactor.vlm_interactor.VLMInteractor(
            mode="online", model=model
        )
        self.interactor.initcount()
        self.interactor.chkcount()
        self.prompts = json.load(open("./vlm_interactor/prompts/renaming_engine.json"))

    def classify(self, img_path_folder, msg=None):

        image_files = [
            f
            for f in os.listdir(img_path_folder)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        new_name_dict = {}
        img_path_list = [os.path.join(img_path_folder, f) for f in image_files]

        for i in range(0, len(img_path_list), 1):
            self.interactor.add_content(
                content=self.prompts["task_config"]["system_prompt"],
                role="system",
                content_type="text",
            )
            self.interactor.add_content(
                content=self.prompts["task_config"]["user_prompt"],
                role="user",
                content_type="text",
            )
            self.interactor.add_content(
                content=f"previously you have classified these items: {list(new_name_dict.values())}, please DO NOT repeat them",
                role="user",
                content_type="text",
            )
            # glog.info(f'previously you have classified these items{new_name_dict.values()}, please do not repeat them')

            for j in range(i, min(i + 1, len(img_path_list))):
                self.interactor.add_content(
                    content=img_path_list[j], role="user", content_type="image"
                )
                self.interactor.add_content(
                    content=self.prompts["task_config"]["per_pic_prompt"],
                    role="user",
                    content_type="text",
                )
                status_code, answer1 = self.interactor.send_content_n_request()
                if (
                    status_code
                    == vlm_interactor.vlm_interactor.InteractStatusCode.SUCCESS
                ):

                    img_file = image_files[j]
                    print(answer1, img_file[: img_file.rfind(".")])
                    new_name_dict[img_file[: img_file.rfind(".")]] = answer1
                    self.interactor.add_content(
                        content=answer1, role="assistant", content_type="text"
                    )
                else:
                    glog.info("VLM Interactor send failed")
                    new_name_dict[img_file[: img_file.rfind(".")]] = (
                        "VLM Interactor send failed"
                    )
                pass

            self.interactor.clear_history()

        return new_name_dict

    def rename_objects_with_engine(self, scene_graph, image_folder_path, device="cuda"):
        """
        Rename objects in the scene graph using the renaming engine.
        """
        glog.info("Renaming objects with the renaming engine...")
        for node_name, node in scene_graph.nodes.items():
            if node.depth > 1:
                node.auto_take_non_ground_object_picture(
                    scene=scene_graph.corresponding_scene,
                    width=1280,
                    height=720,
                    save_path=os.path.join(image_folder_path, f"{node_name}.jpg"),
                )
        """
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
    )
        
        """

        rename_dict = self.classify(
            img_path_folder=image_folder_path,
        )

        glog.info("Renaming completed.")
        return rename_dict


def main():
    # Example usage
    classifier = ItemClassifier()
    img_path_folder = "./image4classify"
    msg = "Classify these items"
    ts = time.perf_counter()
    new_names = classifier.classify(img_path_folder, msg)

    print(new_names)
    glog.info(f"Time taken: {time.perf_counter() - ts} seconds")


if __name__ == "__main__":
    main()
    # Example usage
