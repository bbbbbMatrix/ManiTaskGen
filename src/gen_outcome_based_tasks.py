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
        "--entity_json_path",
        type=str,
        default=None,
        help="Path to the entity JSON file, overrides config if provided.",
    )

    parser.add_argument(
        "--scene_graph_pkl_load_path",
        type=str,
        default=None,
        help="Path to load pre-generated scene graph pickle file, None to generate from JSON.",
    )

    parser.add_argument(
        "--rename_dict_path",
        type=str,
        default=None,
        help="Path to the renaming dictionary JSON file, None to skip renaming.",
    )

    # The default value is set the same as in config_manager.py.

    parser.add_argument(
        "--vlm_list",
        type=str,
        nargs="+",
        default=[
            "openai/gpt-4.1",
            "anthropic/claude-3.5-haiku",
            "google/gemini-2.5-flash-lite-preview-06-17",
        ],
        help="List of VLM models to use.",
    )

    parser.add_argument(
        "--manitaskot_pattern_file",
        type=str,
        default="./src/utils/manitask-ot200/manitask_ot200.txt",
        help="Path to the file containing ManiTaskOT patterns, one per line.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./out/outcome_based_task.txt",
        help="Directory to save outputs like images and logs.",
    )

    return parser.parse_args()


def update_config_from_args(config_manager, args):
    """Update configuration using command line arguments, command line args have higher priority"""
    config = config_manager.config

    # Update global configuration - only update when parameter is not None
    if args.input_json_path is not None:
        config.input_json_path = args.input_json_path


def main(args):

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    current_path = os.path.dirname(os.path.abspath(__file__))

    # 0. Initialize the configuration manager
    # We only need to load the config once.

    # export the final config in this run, named by the timestamp

    # 0.5 Initialize the Scene, add shaders and lights.
    sapien_scene_manager = visualize_scene_sapien.SapienSceneManager()
    scene = sapien_scene_manager.create_scene()

    scene_graph_pkl_load_path = args.scene_graph_pkl_load_path
    entity_json_path = args.entity_json_path
    rename_dict_path = args.rename_dict_path

    # 2 Generate the scene graph
    scene_graph = None
    if scene_graph_pkl_load_path is not None and os.path.exists(
        scene_graph_pkl_load_path
    ):
        glog.info(f"Loading scene graph from {scene_graph_pkl_load_path}")
        with open(scene_graph_pkl_load_path, "rb") as f:
            scene_graph = pickle.load(f)
    else:

        ts = time.perf_counter()
        json_tree_path = gen_scene_graph.load_json_file(entity_json_path)
        scene_graph = gen_scene_graph.gen_multi_layer_graph_with_free_space(
            json_tree_path
        )
        glog.info(f"scene graph tree generation time:  {time.perf_counter() - ts}")

    rename_dict_path = args.rename_dict_path
    rename_dict = {}
    if rename_dict_path is not None and os.path.exists(rename_dict_path):
        with open(rename_dict_path, "r") as f:
            rename_dict = json.load(f)
    else:
        rename_dict = {}

    scene_graph.rename_all_features(rename_dict)
    scene_graph.corresponding_scene = scene
    outcome_based_task_generator = OutcomeBasedTaskGenerator(scene_graph=scene_graph)
    vlm_voter = VLMVoter()

    feasible_task_list = []
    for task in outcome_based_task_generator.task_list:
        vlm_voter.is_task_feasible(task, scene_graph)
        if task.is_feasible:
            feasible_task_list.append(task)

    with open(args.output_dir, "w") as f:
        for task in feasible_task_list:
            f.write(str(task) + "\n")


# %%

if __name__ == "__main__":

    """
    Args:




    """

    args = parse_arguments()
    main(args)


# %%
