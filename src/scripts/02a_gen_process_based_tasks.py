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

    parser.add_argument(
        "--atomic_task_pkl_load_path",
        type=str,
        default=None,
        help="Path to load atomic task pickle file",
    )
    parser.add_argument(
        "--rename_dict",
        type=str,
        default=None,
        help="Path to load atomic task pickle file",
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--process_tasks_pkl_path",
        type=str,
        default=None,
        help="Path to save process-based task pickle file",
    )
    parser.add_argument(
        "--process_tasks_txt_path",
        type=str,
        default=None,
        help="Path to save process-based task text file",
    )

    return parser.parse_args()


def update_config_from_args(config_manager, args):
    """Update configuration using command line arguments, command line args have higher priority"""
    config_dict = {}

    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:
            config_dict[key] = value


def main(args):

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    current_path = os.path.dirname(os.path.abspath(__file__))

    # 0. Initialize the configuration manager
    # We only need to load the config once.

    config_path = args.config

    config_manager = ConfigManager(config_file_path=config_path)
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

    t = time.perf_counter()

    if main_config.scene_graph_pkl_load_path is not None and os.path.exists(
        main_config.scene_graph_pkl_load_path
    ):
        glog.info(f"Loading scene graph from {main_config.scene_graph_pkl_load_path}")
        with open(main_config.scene_graph_pkl_load_path, "rb") as f:
            scene_graph = pickle.load(f)
    else:
        glog.error(f"{main_config.scene_graph_pkl_load_path} does not exist.")
        raise ValueError("Please provide a valid scene graph pickle file path.")

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
            glog.info("Using renaming engine to rename the objects.")
            rename_engine = renaming_engine.RenamingEngine()
            scene_graph.corresponding_scene = scene
            rename_dict = rename_engine.rename_objects_with_engine(
                scene_graph, main_config.image4rename_path
            )
    scene_graph.rename_all_features(rename_dict)
    #     scene_graph.rename_all_features(rename_dict)

    original_scene_graph = copy.deepcopy(scene_graph)
    # random.seed(123)
    chained_task = process_based_task_generation.TaskGeneration(scene_graph)

    chained_task.generate_task(max_task_num=2, task_length=3)

    if main_config.process_based_task_pkl_save_path is not None:
        with open(main_config.process_based_task_pkl_save_path, "wb") as f:
            pickle.dump(chained_task, f)

        with open(main_config.process_based_task_txt_save_path, "w") as f:
            for i, task in enumerate(chained_task.tasks):
                print(f"Task {i}: {task.__repr_rough__()}", file=f)
                print(f"Goal State: ", file=f)

                for j in range(len(task.subtask_list)):
                    print(
                        f"  Subtask {j}: {task.subtask_list[j].__repr_rough__()}",
                        file=f,
                    )
                    print(
                        f"    Goal Info: {task.subtask_list[j].goal_primitive_expression.to_pddl_goal()}",
                        file=f,
                    )

    scene_graph = original_scene_graph

    glog.info(f"Time cost for task generation: {time.perf_counter() - t:.2f} seconds")
    # import ipdb
    # ipdb.set_trace()
    # for i, task in enumerate(atomic_task.tasks):
    #     print(f"Task {i}: {task.__repr_rough__()}", file=open(os.path.join(main_config.output_dir, 'tasks.txt'), 'a'))

    task_sample = chained_task.tasks
    task_sample_ids = [chained_task.tasks.index(task) for task in task_sample]


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    # Example usage
