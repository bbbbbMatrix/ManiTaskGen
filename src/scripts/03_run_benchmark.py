# %%
import os
import sys
import sapien
import argparse


scene = sapien.Scene()
script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append("d:/workplace/scene_graph/task_generation/")
from src.preprocessing import (
    visualize_scene_sapien,
    RawSceneParserFactory,
    renaming_engine,
)
from src.geometry.convex_hull_processor import ConvexHullProcessor_2d
from src.utils.image_renderer import image_render_processor
from src.utils import visualization_tools
from src.utils.config_manager import ConfigManager
from src.core import gen_scene_graph, process_based_task_generation, benchmark_executor
from src.core.outcome_based_task_generation import (
    OutcomeBasedTaskGenerator,
    VLMVoter,
    OutcomeBasedTaskPattern,
)
from src.core.task_feasibility_evaluator import TaskFeasibilityEvaluator
from src.vlm_interaction import vlm_interactor


import pickle
import random
import time
from enum import Enum
import colorama
from colorama import Fore, Style
import glog

import copy
import json

# %%


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Task Generation System")

    # Basic configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file, None for default config.",
    )

    # Important global configuration parameters - these will override settings in config file
    parser.add_argument(
        "--input_json_path",
        type=str,
        default=None,
        help="Path to the input JSON scene file",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        default=None,
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--entity_json_path",
        type=str,
        default=None,
        help="Path to the entity JSON file",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for results"
    )

    # Task related configuration
    parser.add_argument(
        "--task_num", type=int, default=None, help="Number of tasks to generate"
    )

    # Model and mode configuration
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["online", "offline", "manual"],
        help="Mode: online, offline, or manual",
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="Model name for VLM interaction"
    )
    parser.add_argument(
        "--rename_dict_path",
        type=str,
        default=None,
        help="Path to load atomic task pickle file",
    )
    # File path configuration
    parser.add_argument(
        "--scene_graph_pkl_save_path",
        type=str,
        default=None,
        help="Path to save scene graph pickle file",
    )
    parser.add_argument(
        "--atomic_task_pkl_load_path",
        type=str,
        default=None,
        help="Path to load atomic task pickle file",
    )
    return parser.parse_args()


def update_config_from_args(config_manager, args):
    """Update configuration using command line arguments, command line args have higher priority"""
    config_dict = {}

    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:
            config_dict[key] = value

    config_manager._update_config_from_dict(config_dict)


def main(args):

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    current_path = os.path.dirname(os.path.abspath(__file__))

    # 0. Initialize the configuration manager
    # We only need to load the config once.

    config_path = args.config

    config_manager = ConfigManager(
        config_file_path=config_path, run_dir=args.output_dir
    )
    update_config_from_args(config_manager, args)

    # export the final config in this run, named by the timestamp

    if not os.path.exists(config_manager.config_file_export_dir):
        os.makedirs(config_manager.config_file_export_dir)
    config_manager.save_to_yaml(
        os.path.join(
            config_manager.config_file_export_dir,
            f"used_config_{int(time.time())}.yaml",
        )
    )

    # import ipdb
    # ipdb.set_trace()

    main_config = config_manager.config

    input_json_path = main_config.input_json_path
    output_json_path = main_config.output_json_path
    entity_json_path = main_config.entity_json_path
    sapien_scene_manager = visualize_scene_sapien.SapienSceneManager()
    scene = sapien_scene_manager.create_scene()

    sapien_scene_manager.load_objects_from_json(scene, json_file_path=output_json_path)

    if main_config.scene_graph_pkl_load_path is not None and os.path.exists(
        main_config.scene_graph_pkl_load_path
    ):
        glog.info(f"Loading scene graph from {main_config.scene_graph_pkl_load_path}")
        with open(main_config.scene_graph_pkl_load_path, "rb") as f:
            scene_graph = pickle.load(f)
    else:

        ts = time.perf_counter()
        json_tree_path = gen_scene_graph.load_json_file(entity_json_path)
        scene_graph = gen_scene_graph.gen_multi_layer_graph_with_free_space(
            json_tree_path
        )
        glog.info(f"scene graph tree generation time:  {time.perf_counter() - ts}")

    if main_config.process_based_task_pkl_load_path is not None and os.path.exists(
        main_config.process_based_task_pkl_load_path
    ):
        glog.info(
            f"Loading atomic tasks from {main_config.process_based_task_pkl_load_path}"
        )
        with open(main_config.process_based_task_pkl_load_path, "rb") as f:
            chained_task = pickle.load(f)
    else:
        glog.warning("No  pkl file found.")

    initial_atomic_task = copy.deepcopy(chained_task)
    initial_scene_graph = copy.deepcopy(scene_graph)

    task_sample = chained_task.tasks
    task_sample_ids = [chained_task.tasks.index(task) for task in task_sample]

    rename_dict = {}
    if main_config.use_renaming_engine:
        if main_config.rename_dict_path is not None and os.path.exists(
            main_config.rename_dict_path
        ):
            glog.info(
                f"Using provided renaming dictionary {main_config.rename_dict_path} to rename the objects."
            )
            rename_dict = json.load(open(main_config.rename_dict_path, "r"))
        else:
            glog.warning("No rename dict found, using empty dict.")
            rename_dict = {}

    scene_graph.rename_all_features(rename_dict)

    # 5 test tasks
    initial_atomic_task = copy.deepcopy(chained_task)
    initial_scene_graph = copy.deepcopy(scene_graph)
    result = []
    histories = []
    task_list = random.sample(
        range(len(task_sample_ids)), min(main_config.task_num, len(task_sample_ids))
    )
    total_score = 0
    total_sr = 0
    sapien_scene_manager = visualize_scene_sapien.SapienSceneManager()

    # import ipdb;
    # ipdb.set_trace()

    for i in task_list:

        task = task_sample[i]

        another_scene = sapien.Scene()
        another_scene.set_timestep(1 / 100)
        another_scene.add_ground(altitude=0)
        sapien_scene_manager.load_objects_from_json(
            another_scene, json_file_path=output_json_path
        )
        another_scene.set_ambient_light([0.5, 0.5, 0.5])
        another_scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        for j in range(1000):

            another_scene.step()
            another_scene.update_render()

        # description of task has moved into apply function.
        manual_vlm_interactor = vlm_interactor.VLMInteractor(
            mode=main_config.mode, model=main_config.model_name
        )
        scene_graph.corresponding_scene = another_scene
        scene_graph.rename_all_features(rename_dict)
        scene_graph.corresponding_scene = scene
        scene_graph.rename_all_features(rename_dict)

        glog.info(task)

        # return TaskStatusCode.SUCCESS or TaskStatusCode.FAILURE
        intermediate_task, intermediate_task_id = None, None

        task = benchmark_executor.BenchmarkExecutor(
            task=task,
            task_id=i,
            intermediate_task=intermediate_task,
            intermediate_task_id=intermediate_task_id,
            scene_graph=scene_graph,
            scene=another_scene,
            vlm_interactor=manual_vlm_interactor,
            model_name=main_config.model_name,
            generate_mistake_note=main_config.generate_mistake_note,
            use_mistake_note=main_config.use_mistake_note,
        )

        task.apply_action(state=benchmark_executor.InteractStates.NAVIGATION)
        result.append([task.status, task.partial_score])
        histories.append(task.action_history_list)
        scene = sapien.Scene()
        scene.set_timestep(1 / 100)
        scene.add_ground(altitude=0)

        sapien_scene_manager.load_objects_from_json(
            scene, json_file_path=output_json_path
        )
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        for j in range(1000):
            scene.step()
            scene.update_render()

        scene_graph = copy.deepcopy(initial_scene_graph)
        atomic_task = copy.deepcopy(initial_atomic_task)

        # import ipdb; ipdb.set_trace()
        # start_task_msg_buffer = ""
        total_score += sum(task.out_of_order_partial_scores)
        total_sr += int(task.status == True)
        with open(main_config.result_file_path, "a") as f:
            f.write(f"Task {i}: {task.status}, Score: {task.partial_score}\n")
            f.write(
                f"Out-of-order Subtask Scores: {task.out_of_order_partial_scores}\n"
            )
            f.write(f"Task Info: {task.task.__repr_rough__()}\n")
            f.write(f"History: {task.action_history_list}\n")

    with open(main_config.result_file_path, "a") as f:
        f.write(f"Total Score: {total_score  / len(task_list)}\n")
        f.write(f"Total Success Rate: {total_sr / len(task_list)}\n")


# %%

if __name__ == "__main__":

    """
    Args:




    """

    args = parse_arguments()
    main(args)


# %%
