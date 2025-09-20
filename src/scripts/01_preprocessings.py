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

    parser.add_argument(
        "--desired_objects",
        type=str,
        nargs="*",
        default=None,
        help="List of desired objects to include",
    )
    parser.add_argument(
        "--not_desired_objects",
        type=str,
        nargs="*",
        default=None,
        help="List of objects to exclude",
    )
    parser.add_argument(
        "--stage_config_path",
        type=str,
        default=None,
        help="Path to the stage configuration",
    )

    parser.add_argument(
        "--dataset_root_path", type=str, default=None, help="Root path for the dataset"
    )
    parser.add_argument(
        "--object_config_path",
        type=str,
        default=None,
        help="Path to the object configuration",
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
        "--scene_graph_pkl_save_path",
        type=str,
        default=None,
        help="Path to save scene graph pickle file",
    )

    # Renaming and interaction related
    parser.add_argument(
        "--use_renaming_engine",
        action="store_true",
        default=None,
        help="Whether to use renaming engine",
    )
    parser.add_argument(
        "--image4rename_path",
        type=str,
        default=None,
        help="Path to the image for renaming",
    )
    parser.add_argument(
        "--rename_dict_path",
        type=str,
        default=None,
        help="Path to the renaming dictionary",
    )

    return parser.parse_args()


def main(args):

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    current_path = os.path.dirname(os.path.abspath(__file__))

    # 0. Initialize the configuration manager
    # We only need to load the config once.

    config_path = args.config

    config_manager = ConfigManager(config_file_path=config_path)
    # update_config_from_args(config_manager, args)

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

    # 0.5 Initialize the Scene, add shaders and lights.
    sapien_scene_manager = visualize_scene_sapien.SapienSceneManager()
    scene = sapien_scene_manager.create_scene()

    # 1 Parse the input JSON file
    t = time.perf_counter()
    glog.info(f"Parsing input JSON file: {input_json_path}")
    maniskill_parser = RawSceneParserFactory.create_parser("replica")
    maniskill_parser.parse_scene(input_json_path, output_json_path)

    entity_json_path = main_config.entity_json_path
    glog.info(f"Parsed output JSON file: {output_json_path}")
    glog.info(f"Time cost for parsing: {time.perf_counter() - t:.2f} seconds")
    # 1.5 Maybe we need to adjust the scene with gravity.
    if main_config.adjust_with_gravity:
        # TODO: refine to a single function adjust_w_gravity

        sapien_entity_exporter = visualize_scene_sapien.EntityExporter()

        sapien_scene_manager.load_objects_from_json(
            scene, json_file_path=output_json_path
        )

        for i in range(10000):
            scene.step()
            scene.update_render()

        sapien_entity_exporter.reget_entities_from_sapien(
            scene, output_json_path, entity_json_path
        )

    else:
        entity_json_path = output_json_path

    # 2 Generate the scene graph
    scene_graph = None
    # if main_config.scene_graph_pkl_load_path is not None and os.path.exists(main_config.scene_graph_pkl_load_path):
    #     glog.info(f"Loading scene graph from {main_config.scene_graph_pkl_load_path}")
    #     with open(main_config.scene_graph_pkl_load_path, 'rb') as f:
    #         scene_graph = pickle.load(f)
    # else:

    ts = time.perf_counter()
    json_tree_path = gen_scene_graph.load_json_file(entity_json_path)
    scene_graph = gen_scene_graph.gen_multi_layer_graph_with_free_space(json_tree_path)
    glog.info(f"scene graph tree generation time:  {time.perf_counter() - ts}")

    if main_config.scene_graph_pkl_save_path is not None:
        with open(main_config.scene_graph_pkl_save_path, "wb") as f:
            pickle.dump(scene_graph, f)

    rename_dict = {}
    if main_config.use_renaming_engine:
        # if main_config.rename_dict_path is not None and os.path.exists(main_config.rename_dict_path):
        #     glog.info(f"Using provided renaming dictionary {main_config.rename_dict_path} to rename the objects.")
        #     rename_dict = json.load(open(main_config.rename_dict_path, 'r'))
        # else:
        glog.info("Using renaming engine to rename the objects.")
        rename_engine = renaming_engine.RenamingEngine()
        scene_graph.corresponding_scene = scene
        rename_dict = rename_engine.rename_objects_with_engine(
            scene_graph, main_config.image4rename_path
        )
        scene_graph.rename_all_features(rename_dict)

    # entity_json_path = './parsed_scene_iTHOR_1.json'


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
