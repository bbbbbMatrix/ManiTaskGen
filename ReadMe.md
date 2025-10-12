# **ManiTaskGen: A Comprehensive Task Generator for Benchmarking and Improving Vision-Language Agents on Embodied Decision-Making**





[![arXiv](https://img.shields.io/badge/arXiv-2505.20726-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![Website](https://img.shields.io/badge/Project%20Page-Visit-brightgreen?style=for-the-badge&logo=globe)](https://manitaskgen.github.io/)



This is the official repository for the ManiTaskGen project. It Includes instructions on dataset downloading, environment setting up and commands for ManiTaskGen benchmarks and agent finetuning pipelines.




## Code Organization 

```src/
├── core/                               # Core Data Structures
│   ├── gen_scene_graph.py              # Building scene graph from processed objects
│   ├── task_primitive.py               # Express the goal of tasks into primitives
│   ├── process_based_task_generator.py # Generating process-based manipulation tasks   
│   ├── outcome_based_task_generator.py # Generating outcome-based manipulation tasks(Yet unavailable to use for benchmarking)
│   └── benchmark_executor.py           # Task execution and interaction management 
├── preprocessing/                      # Scene Preprocessing
│   ├── affordable_platform.py          # Maintain affordable platforms.
│   ├── base_parser.py                  # Base class for scene supports
│   ├── maniskill_parser.py             # Pre-parse ManiSkill-style scenes
│   ├── sunrgbd_parser.py               # Pre-parse SUNRGBD-style scenes
│   ├── visualize_scene_sapien.py       # Build sapien scene with pre-parsed data
│   ├── scene_parser.py                 # Further parse the scenes for gen_scene_graph
│   └── renaming_engine.py              # Object renaming and standardization
├── vlm_interaction/                    # VLM Interaction
│   ├── vlm_interactor.py               # Prompt management and VLM interface communication
│   └── interact_prompt_helper.py       # Helper for generating prompts
├── geometry/                           # Custom Geometry Modules
│   ├── basic_geometries.py             # Basic geometric operations and utilities
│   ├── convex_hull_processor.py        # Convex hull computation and processing
│   ├── concave_hull_processor.py       # Concave polygon decomposition and processing
│   ├── ground_coverage_analyzer.py     # Examine ground coverages to determine where for agent to 'stand'
│   ├── rectangle_query_processor.py    # Rectangular region queries and spatial analysis
│   ├── object_mesh_processor.py        # Processing object meshes 
│   ├── polygon_processor.py            # General polygon operations and transformations
│   └── placement_helper.py             # Object placement validation and assistance
├── config/                             # Config files
│   └── default_config.yaml             # Config files for the whole project in yaml
└── utils/                              # Utilities
    ├── image_renderer/                 # Image renderer
    │   ├── coordinate_convertor.py     # Convert coordinates between world, camera & image systems.
    │   └── image_render_processor.py   # Render images in Sapien
    ├── visualization_tools.py          # Visualization tools for debugging and analysis
    ├── config_manager.py               # Configuration management module
    ├── string_convertor.py             # Stem the object names
    ├── manitask-ot200/                 # Path of our dataset.
    ├── prompts/                        # Prompt templates, including several prompts.
    └── VLMEvalKit/                     # VLMEvalKit, hardcoded with OPENROUTER api
```







## Installation

For installation, refer to  [INSTALLATION.md](./docs/INSTALLATION.md) 

We also provide the configuration file exported by the conda environment in ``config/env.yml``.



## QuickStart

## Environment Setup

Please follow the instructions in [INSTALLATION.md](./docs/INSTALLATION.md) to set up the environment.

### Usage Examples 


### Maniskill-style Scenes (AI2THOR & ReplicaCAD)

First, set the configuration file in 'scripts/config.sh'. 

Move or link the dataset under the "data/dataset" directory. there are two empty folders named "replica_dataset" and "ai2thor" for ReplicaCAD and AI2THOR datasets respectively, please substitute your own dataset. 

Then, modify the config file from the example global config file. The two config you probably have to modify is ``input_json_path`` and ``openrouter/api_key``.  After that, run the following codes:



```bash
CONFIG_FILE=path/to/config.yaml bash scripts/run_01_preprocessing.sh     # gen scene graph & rename objects
CONFIG_FILE=path/to/config.yaml bash scripts/run_02a_gen_process_tasks.sh# gen process based tasks
CONFIG_FILE=path/to/config.yaml bash scripts/run_02b_gen_outcome_tasks.sh# gen outcome based tasks
CONFIG_FILE=path/to/config.yaml bash scripts/run_03_run_benchmark.sh    # run benchmark
CONFIG_FILE=path/to/config.yaml bash scripts/run_99_item_modification_only.sh   # run benchmark
```

It's recommended to set ``path/to/config.yaml`` to ``latest_config/used_config.yaml`` after you modified the config file for the first time, so that the latest config will be used by default.


The following table shows the features and IOs for each script:

| Scripts                                       | Feature                                                      | Input                                                        | Output(default path)                                         |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ``scripts/run_01_preprocessing.sh``           | (a) Parse original dataset, generate scene graph and dump them<br />(b) Rename objects | data/datasets/ai2thor                                        | runs/cache/entity_scene.json<br />runs/cache/scene_graph.pkl<br />runs/visualizations/scene_graph.dot,<br />runs/visualizations/scene_graph.txt<br />data/cache/rename_dict.json<br />data/images/image4rename/xxx.png |
| ``scripts/run_02a_gen_process_based_task.sh`` | generate process based tasks                                 | data/cache/scene_graph.pkl<br />                             | data/cache/process_based_task.pkl<br />data/output/process_based_task.txt |
| ``scripts/run_02b_gen_outcome_based_task.sh`` | generate outcome based tasks                                 | data/cache/scene_graph.pkl<br />data/cache/entity_scene.json | <br />data/output/outcome_based_task.txt                     |
| ``scripts/run_03_run_benchmark.sh``           | run benchmark                                                | data/cache/process_based_task.pkl<br />data/cache/reflection_notes.txt(optional) | data/images/image4interact/xxx.png<br />data/output/results.txt |
| ``scripts/run_99_item_modification_only.sh``                         | run item_modification interaction                        | /                                                            | ``visualizations/final_scene_graph.json``                                                           |
| ``scripts/config.sh``                         | A auxiliary script for setting paths                         | /                                                            | /                                                           |

Note that the part 02a&02b requires ``scene_graph.pkl`` and part 03 requires ``process_based_task.pkl``, which means you have to run former scripts before the latter scripts in order to avoid errors.





```shell



#### Manual Input Testing (Human Baseline)

To run ManiTaskGen on a ReplicaCAD dataset scene and simulate Benchmarking on Embodied decision-making with single-step (level 1 & 2) tasks using manual input decisions, please change the dataset path in `AppConfig`, `RawSceneConfig` and `SapienConfig` classes in `src/utils/config_manager.py` accordingly after installation, then run the following code:

​```shell
python main.py --config config/default_config.yml --input_json_path /path/to/input/json/scene/file --output_json_path /path/to/output --mode manual --model_name human --adjust_with_gravity True 
```

To enable item renaming, first enter OpenRouter API key and model address, then set `use_renaming_engine=True` in the command line arguments. This will use a VLM to rename objects based on their descriptions.

```shell
python main.py --config config/default_config.yml --input_json_path /path/to/input/json/scene/file --output_json_path /path/to/output --mode manual --model_name human --adjust_with_gravity True --use_renaming_engine True
```

As intermediate results, after the code execution, the `./output/` directory will contain the following files:

- `scene_graph.pkl`: The scene graph of the parsed scene. If this file exists in subsequent runs, it can be loaded directly to skip the scene graph generation step.
- `atomic_task.pkl`: The atomic tasks generated from the scene. Similarly, if this file exists, it can be loaded directly to skip the atomic task generation step.
- `scene_graph.dot`: The scene graph in DOT format for visualization purposes.
- `tasks.txt`: The generated subtasks in text format for reference.
- `image4rename/`: A directory containing images used for object renaming.
- `rename_dict.json`: A JSON file containing the renaming dictionary generated by the renaming engine.


### SunRGBD-style Scenes (SUNRGBD)

Though unable to benchmark, you can run the following command to parse the sunrgbd scene, and get a json file for building scene graph and generating tasks:

```shell
python src/preprocessing/sunrgbd_parser.py --scene_path=path/to/SUNRGBD/dataset --output_path=/path/to/output/folder
```

```

### Configuration Priority

1. **Command line arguments** (highest priority)
2. **Configuration file** (medium priority)  
3. **Default values** (lowest priority)

Command line arguments will override any settings in the configuration file, allowing for flexible experimentation without modifying configuration files.

The following tables summarizes the core global configuration parameters and their default values. For configurations on specific modules, please refer to the "Implementation Details" of that module and the `config/default_config.yml` file.


### Core Configuration

| Parameter               | Type   | Default  | Description                                                  |
| ----------------------- | ------ | -------- | ------------------------------------------------------------ |
| `adjust_with_gravity`   | `bool` | `true`   | Enable gravity simulation. When the original scene has **collision path for objects** and **exists floating or irrational placements**, this can be set to `True` to adjust object poses with gravity. |
| `use_renaming_engine`   | `bool` | `false`  | Enable object renaming. When the original scene has ambiguous names, like the `ReplicaCAD`, this can be set to `True` to use a VLM to rename objects. |
| `bbox_only`             | `bool` | `false`  | Use bbox-only mode. Every objects will be treated as cuboids, mainly for RGBD scenes. Benchmarking is disabled in this mode. |
| `mode`                  | `str`  | `manual` | Execution mode. `"online"` for API-based VLM, `"manual"` for human tests. |
| `model_name`            | `str`  | `human`  | Model name for VLM interaction. Affects the path for saving images during interaction. |
| `task_num`              | `int`  | `5`      | The number of tasks given to VLM in total.                   |
| `use_lv3_task`          | `bool` | `false`  | Whether to use level 3 tasks (dual tasks with intermediate steps). |
| `generate_mistake_note` | `bool` | `false`  | Whether to generate mistake notes for reflection.            |
| `use_mistake_note`      | `int`  | `0`      | How many trial notes in the reflection file are to use.      |
| `cache_enabled`         | `bool` | `true`   | Whether to enable caching for performance optimization.      |
| `random_seed`           | `int`  | `null`   | Random seed for reproducibility. If null, uses system time.  |
| `log_level`             | `str`  | `INFO`   | Logging level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

### File Paths

| Parameter                  | Type  | Default                         | Description                                   |
| -------------------------- | ----- | ------------------------------- | --------------------------------------------- |
| `input_json_path`          | `str` | `apt_0.scene_instance.json`     | Input scene file path.                        |
| `output_json_path`         | `str` | `./replica_apt_0_parsed.json`   | Parsed output file path.                      |
| `entity_json_path`         | `str` | `./replica_apt_0_entities.json` | Entity file path after gravity adjustment.    |
| `output_dir`               | `str` | `./output/`                     | Output directory for all generated files.     |
| `image4rename_path`        | `str` | `./image4rename/`               | Path for images used in VLM renaming process. |
| `rename_dict_path`         | `str` | `./rename_dict.json`            | Path to the renaming dictionary file.         |
| `result_file_path`         | `str` | `./result.txt`                  | Path to save benchmark results.               |
| `reflection_txt_load_path` | `str` | `./load_reflection.txt`         | Path for loading reflection notes.            |
| `reflection_txt_save_path` | `str` | `./save_reflection.txt`         | Path for saving reflection notes.             |

### Pickle Files

| Parameter                   | Type       | Default             | Description                                         |
| --------------------------- | ---------- | ------------------- | --------------------------------------------------- |
| `scene_graph_pkl_load_path` | `str/null` | `./scene_graph.pkl` | Scene graph load path. If exists, skip generation.  |
| `scene_graph_pkl_save_path` | `str/null` | `./scene_graph.pkl` | Scene graph save path for future use.               |
| `atomic_task_pkl_load_path` | `str/null` | `./atomic_task.pkl` | Atomic tasks load path. If exists, skip generation. |
| `atomic_task_pkl_save_path` | `str/null` | `./atomic_task.pkl` | Atomic tasks save path for future use.              |


## Adding Custom Datasets

Aside from AI2THOR and ReplicaCAD, other maniskill-style scenes can also be parsed with ``src/preprocessing/maniskill_parser.py``. 

If you want to run the benchmark on other scene datasets with different formats, refer to ``src/preprocessing/base_parser.py``, ``src/preprocessing/maniskill_parser.py`` and ``src/preprocessing/sunrgbd_parser.py`` to add new parsers.













