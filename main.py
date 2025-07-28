# %%
import os
import sys
import sapien
import argparse


scene = sapien.Scene()
script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append("d:/workplace/scene_graph/task_generation/")
from src.preprocessing import visualize_scene_sapien, RawSceneParserFactory, renaming_engine
from src.geometry.convex_hull_processor import ConvexHullProcessor_2d
from src.utils.image_renderer import image_render_processor
from src.utils import visualization_tools
from src.utils.config_manager import ConfigManager
from src.core import gen_scene_graph, process_based_task_generation, benchmark_executor, task_primitive
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

#%%


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Task Generation System')
    
    # Basic configuration
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to the configuration file, None for default config.')
    
    # Important global configuration parameters - these will override settings in config file
    parser.add_argument('--input_json_path', type=str, default=None,
                       help='Path to the input JSON scene file')
    parser.add_argument('--output_json_path', type=str, default=None,
                       help='Path to the output JSON file')
    parser.add_argument('--entity_json_path', type=str, default=None,
                       help='Path to the entity JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    
    # Task related configuration
    parser.add_argument('--task_num', type=int, default=None,
                       help='Number of tasks to generate')
    parser.add_argument('--use_lv3_task', action='store_true', default=None,
                       help='Whether to use level 3 tasks (dual tasks)')
    
    # Model and mode configuration
    parser.add_argument('--mode', type=str, default=None, choices=['online', 'offline', 'manual'],
                       help='Mode: online, offline, or manual')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name for VLM interaction')
    
    # File path configuration
    parser.add_argument('--scene_graph_pkl_load_path', type=str, default=None,
                       help='Path to load scene graph pickle file')
    parser.add_argument('--scene_graph_pkl_save_path', type=str, default=None,
                       help='Path to save scene graph pickle file')
    parser.add_argument('--atomic_task_pkl_load_path', type=str, default=None,
                       help='Path to load atomic task pickle file')
    parser.add_argument('--atomic_task_pkl_save_path', type=str, default=None,
                       help='Path to save atomic task pickle file')
    parser.add_argument('--result_file_path', type=str, default=None,
                       help='Path to save the result file')
    
    # Renaming and interaction related
    parser.add_argument('--use_renaming_engine', action='store_true', default=None,
                       help='Whether to use renaming engine')
    parser.add_argument('--image4rename_path', type=str, default=None,
                       help='Path to the image for renaming')
    parser.add_argument('--rename_dict_path', type=str, default=None,
                       help='Path to the renaming dictionary')
    
    # Mistake notes and reflection related
    parser.add_argument('--generate_mistake_note', action='store_true', default=None,
                       help='Whether to generate mistake notes')
    parser.add_argument('--use_mistake_note', type=int, default=None,
                       help='Number of mistake notes to use (0 to disable)')
    parser.add_argument('--reflection_txt_load_path', type=str, default=None,
                       help='Path to the reflection text file')
    parser.add_argument('--reflection_txt_save_path', type=str, default=None,
                       help='Path to save reflection text')
    
    # Physics and scene related
    parser.add_argument('--adjust_with_gravity', action='store_true', default=None,
                       help='Whether to adjust scene with gravity')
    parser.add_argument('--bbox_only', action='store_true', default=None,
                       help='Whether to use bounding box only')
    
    # Debug and logging
    parser.add_argument('--log_level', type=str, default=None, 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--cache_enabled', action='store_true', default=None,
                       help='Whether to enable caching')
    
    return parser.parse_args()


def update_config_from_args(config_manager, args):
    """Update configuration using command line arguments, command line args have higher priority"""
    config = config_manager.config
    
    # Update global configuration - only update when parameter is not None
    if args.input_json_path is not None:
        config.input_json_path = args.input_json_path
    if args.output_json_path is not None:
        config.output_json_path = args.output_json_path
    if args.entity_json_path is not None:
        config.entity_json_path = args.entity_json_path
    if args.output_dir is not None:
        config.output_dir = args.output_dir
        
    if args.task_num is not None:
        config.task_num = args.task_num
    if args.use_lv3_task is not None:
        config.use_lv3_task = args.use_lv3_task
        
    if args.mode is not None:
        config.mode = args.mode
    if args.model_name is not None:
        config.model_name = args.model_name
        
    if args.scene_graph_pkl_load_path is not None:
        config.scene_graph_pkl_load_path = args.scene_graph_pkl_load_path
    if args.scene_graph_pkl_save_path is not None:
        config.scene_graph_pkl_save_path = args.scene_graph_pkl_save_path
    if args.atomic_task_pkl_load_path is not None:
        config.atomic_task_pkl_load_path = args.atomic_task_pkl_load_path
    if args.atomic_task_pkl_save_path is not None:
        config.atomic_task_pkl_save_path = args.atomic_task_pkl_save_path
    if args.result_file_path is not None:
        config.result_file_path = args.result_file_path
        
    if args.use_renaming_engine is not None:
        config.use_renaming_engine = args.use_renaming_engine
    if args.image4rename_path is not None:
        config.image4rename_path = args.image4rename_path
    if args.rename_dict_path is not None:
        config.rename_dict_path = args.rename_dict_path
        
    if args.generate_mistake_note is not None:
        config.generate_mistake_note = args.generate_mistake_note
    if args.use_mistake_note is not None:
        config.use_mistake_note = args.use_mistake_note
    if args.reflection_txt_load_path is not None:
        config.reflection_txt_load_path = args.reflection_txt_load_path
    if args.reflection_txt_save_path is not None:
        config.reflection_txt_save_path = args.reflection_txt_save_path
        
    if args.adjust_with_gravity is not None:
        config.adjust_with_gravity = args.adjust_with_gravity
    if args.bbox_only is not None:
        config.bbox_only = args.bbox_only
        
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.random_seed is not None:
        config.random_seed = args.random_seed
    if args.cache_enabled is not None:
        config.cache_enabled = args.cache_enabled





def main(args):

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    current_path = os.path.dirname(os.path.abspath(__file__))

    # 0. Initialize the configuration manager
    # We only need to load the config once.
    config_path = args.config
    config_manager = ConfigManager(config_path)
    update_config_from_args(config_manager, args)
    
    
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
    maniskill_parser = RawSceneParserFactory.create_parser('replica')
    maniskill_parser.parse_scene(
        input_json_path, 
        output_json_path
        )
    
    entity_json_path = main_config.entity_json_path
    glog.info(f"Parsed output JSON file: {output_json_path}")
    glog.info(f"Time cost for parsing: {time.perf_counter() - t:.2f} seconds")
    #1.5 Maybe we need to adjust the scene with gravity.
    if main_config.adjust_with_gravity:
        
        
        sapien_entity_exporter = visualize_scene_sapien.EntityExporter()
        
        sapien_scene_manager.load_objects_from_json(scene, json_file_path=output_json_path)

        for i in range(10000):
            scene.step()
            scene.update_render()
            
        sapien_entity_exporter.reget_entities_from_sapien(scene, output_json_path, entity_json_path)
        
    else:
        entity_json_path = output_json_path
    
    #entity_json_path = './replica_apt_0_parsed.json'

    glog.info(f"input_json_path: {input_json_path}")
    glog.info(f"output_json_path: {output_json_path}")
    # parse_replica.parse_replica(input_json_path, output_json_path)
    glog.info(f"Time cost for parsing: {time.perf_counter() - t:.2f} seconds")
    
    # 2 Generate the scene graph
    scene_graph = None 
    if main_config.scene_graph_pkl_load_path is not None and os.path.exists(main_config.scene_graph_pkl_load_path):
        glog.info(f"Loading scene graph from {main_config.scene_graph_pkl_load_path}")
        with open(main_config.scene_graph_pkl_load_path, 'rb') as f:
            scene_graph = pickle.load(f)
    else:
        
        ts = time.perf_counter()
        json_tree_path = gen_scene_graph.load_json_file(entity_json_path)
        scene_graph = gen_scene_graph.gen_multi_layer_graph_with_free_space(
            json_tree_path
        )
        glog.info(f"scene graph tree generation time:  {time.perf_counter() - ts}")
    
    if main_config.scene_graph_pkl_save_path is not None:
        with open(main_config.scene_graph_pkl_save_path, 'wb') as f:
            pickle.dump(scene_graph, f)
    
    # 2.5 Visualize the scene graph
    glog.info(f"Time cost for generating scene graph: {time.perf_counter() - t:.2f} seconds")
    visualization_tools.quick_visualize_scene(scene_graph, os.path.join(main_config.output_dir, 'scene_graph.dot'))
    
    
    #3 Rename the objects with the renaming engine
    rename_dict = {}
    if main_config.use_renaming_engine:
        if main_config.image4rename_path is None:
            rename_dict = json.load(open(main_config.rename_dict_path, 'r'))
        else:
            glog.info("Using renaming engine to rename the objects.")
            rename_engine = renaming_engine.RenamingEngine()
            scene_graph.corresponding_scene = scene
            rename_dict = rename_engine.rename_objects_with_engine(
                scene_graph, 
                main_config.image4rename_path
            )
        scene_graph.rename_all_features(rename_dict)
        
    glog.info(f"Time cost for object renaming: {time.perf_counter() - t:.2f} seconds")
    #4 Generate the atomic tasks 
    atomic_task = None
    if main_config.atomic_task_pkl_load_path is not None and os.path.exists(main_config.atomic_task_pkl_load_path):
        glog.info(f"Loading atomic tasks from {main_config.atomic_task_pkl_load_path}")
        with open(main_config.atomic_task_pkl_load_path, 'rb') as f:
            atomic_task = pickle.load(f)
    else:
        atomic_task = process_based_task_generation.TaskGeneration(scene_graph)
        atomic_task.generate_task_from_scene_graph()
        
    if main_config.atomic_task_pkl_save_path is not None:
        with open(main_config.atomic_task_pkl_save_path, 'wb') as f:
            pickle.dump(atomic_task, f)
    
    glog.info(f"Time cost for task generation: {time.perf_counter() - t:.2f} seconds")
    
    for i, task in enumerate(atomic_task.tasks):
        print(f"Task {i}: {task.__repr_rough__()}", file=open(os.path.join(main_config.output_dir, 'tasks.txt'), 'a'))
 
    
    
    task_sample = atomic_task.tasks
    task_sample_ids = [atomic_task.tasks.index(task) for task in task_sample]
    
    
    
    #5 test tasks (level 1, 2, 3)
    initial_atomic_task = copy.deepcopy(atomic_task)
    initial_scene_graph = copy.deepcopy(scene_graph)
    result = []
    histories = []
    task_list = random.sample(range(len(task_sample_ids)), main_config.task_num)
    total_score = 0
    total_sr = 0
    for i in task_list:
        
        task = task_sample[i]
        
        
        another_scene = sapien.Scene()
        another_scene.set_timestep(1 / 100)
        another_scene.add_ground(altitude=0)
        sapien_scene_manager.load_objects_from_json(another_scene, json_file_path=output_json_path)
        another_scene.set_ambient_light([0.5, 0.5, 0.5])
        another_scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        for j in range(1000):
            
            another_scene.step()
            another_scene.update_render()
        
        
        # description of task has moved into apply function.
        manual_vlm_interactor = vlm_interactor.VLMInteractor(mode=main_config.mode)
        scene_graph.corresponding_scene = another_scene
        scene_graph.rename_all_features(rename_dict)
        scene_graph.corresponding_scene = scene
        scene_graph.rename_all_features(rename_dict)
        
        
        glog.info(task.__repr_rough__())
        # return TaskStatusCode.SUCCESS or TaskStatusCode.FAILURE
        intermediate_task, intermediate_task_id = None, None
        
        if main_config.use_lv3_task:
            intermediate_task_id = random.randint(0, len(task_sample) - 1)
            intermediate_task = task_sample[intermediate_task_id]


            glog.info(f"Using intermediate task: {intermediate_task.__repr_rough__()}")
        
        task = benchmark_executor.TaskInteractHelper(
            task=task,
            task_id=i,
            intermediate_task=intermediate_task,
            intermediate_task_id=intermediate_task_id,
            scene_graph=scene_graph,
            scene=another_scene,
            vlm_interactor=manual_vlm_interactor,
            img_path=f"{current_path}",
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

        sapien_scene_manager.load_objects_from_json(scene, json_file_path=output_json_path)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        for j in range(1000):
            scene.step()
            scene.update_render()

        scene_graph = copy.deepcopy(initial_scene_graph)
        atomic_task = copy.deepcopy(initial_atomic_task)

        # import ipdb; ipdb.set_trace()
        # start_task_msg_buffer = ""
        total_score += task.partial_score
        total_sr += int(task.status == True)
        with open(main_config.result_file_path, "a") as f:
            f.write(f"Task {i}: {task.status}, Task Type: level {task_sample[i].is_ambiguous(scene_graph) + 1}, type {task_sample[i].type}, Score: {task.partial_score}\n")
            f.write(f"Task Info: {task.task.__repr_rough__()}\n")
            f.write(f"History: {task.action_history_list}\n")

    

    # with open(result_file_path, "a") as f:
    #     f.write(f"Total Score: {total_score  / np.sum(task_type_cnt)}\n")
    #     f.write(f"Total Success Rate: {total_sr / np.sum(task_type_cnt)}\n")
    #     f.write(f"Task Type Count: {task_type_cnt}\n")




#%%

if __name__ == "__main__":
    
    '''
    Args:
        
        
    
    
    '''
    
    args = parse_arguments()
    main(args)
    
    
# %%
