# **ManiTaskGen: A Comprehensive Task Generator for Benchmarking and Improving Vision-Language Agents on Embodied Decision-Making**





[![arXiv](https://img.shields.io/badge/arXiv-2505.20726-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit-brightgreen?style=for-the-badge&logo=globe)](https://manitaskgen.github.io/)



This is the official repository for the ManiTaskGen project. It Includes instructions for downloading and running the ManiTaskGen benchmark.





## Overview 

This codebase contains a universal task generation framework for ManiSkill-style scenes, facilitating both benchmarking and improvement of embodied decision-making agents. 





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

For installation, refer to  [INSTALLATION.md](../docs/INSTALLATION.md) 

The configuration file exported by the conda environment used by the author can be found in ``config/env.yml``.



## QuickStart


### Usage Examples 

#### Manual Input Testing (Human Baseline)

To run ManiTaskGen on a ReplicaCAD dataset scene and simulate Benchmarking on Embodied decision-making with single-step (level 1 & 2) tasks using manual input decisions, please change the dataset path in `AppConfig`, `RawSceneConfig` and `SapienConfig` classes in `src/utils/config_manager.py` accordingly after installation, then run the following code:

```shell
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

#### Online testing

For online testing, please first enter your OpenRouter API key and model address in `default_config.yml`. Then run the following code:

```shell
python main.py --config config/default_config.yml --input_json_path /path/to/input/json/scene/file --output_json_path /path/to/output --mode online --model_name {model_name} --adjust_with_gravity True
```

#### For multi-step tasks

To test level 3 tasks, you need to set `use_lv3_task=True` （in `default_config.yml` or command line arguments）. This will enable the generation of multi-step tasks. The following command can be used:

```shell
python main.py --config config/default_config.yml --input_json_path /path/to/input/json/scene/file --output_json_path /path/to/output --mode manual --model_name human --use_lv3_task
```

#### Inference-time Fine-tuning with Reflection Notes

For reflection-based methods to achieve inference-time fine-tuning of VLM agents, you need to first store reflection notes in one run, then load the first few lines of the reflection notes generated in the first run during the second run:

```shell
# First run: Generate and save reflection notes
python main.py --config config/default_config.yml --input_json_path /path/to/input/json/scene/file --output_json_path /path/to/output --mode online --model_name {model_name} --adjust_with_gravity --generate_mistake_note --reflection_txt_save_path /path/to/reflection/notes --use_mistake_note 0

# Second run: Load and use reflection notes
python main.py --config config/default_config.yml --input_json_path /path/to/input/json/scene/file --output_json_path /path/to/output --mode online --model_name {model_name} --adjust_with_gravity --reflection_txt_load_path /path/to/reflection/notes --use_mistake_note 5
```

### Configuration Priority

1. **Command line arguments** (highest priority)
2. **Configuration file** (medium priority)  
3. **Default values** (lowest priority)

Command line arguments will override any settings in the configuration file, allowing for flexible experimentation without modifying configuration files.

The following tables summarizes the core global configuration parameters and their default values. For configurations on specific modules, please refer to the "Implementation Details" of that module and the `config/default_config.yml` file.


### Core Configuration

| Parameter               | Type   | Default    | Description                                                  |
| ----------------------- | ------ | ---------- | ------------------------------------------------------------ |
| `adjust_with_gravity`   | `bool` | `true`     | Enable gravity simulation. When the original scene has **collision path for objects** and **exists floating or irrational placements**, this can be set to `True` to adjust object poses with gravity. |
| `use_renaming_engine`   | `bool` | `false`    | Enable object renaming. When the original scene has ambiguous names, like the `ReplicaCAD`, this can be set to `True` to use a VLM to rename objects. |
| `bbox_only`             | `bool` | `false`    | Use bbox-only mode. Every objects will be treated as cuboids, mainly for RGBD scenes. Benchmarking is disabled in this mode. |
| `mode`                  | `str`  | `manual`   | Execution mode. `"online"` for API-based VLM, `"manual"` for human tests. |
| `model_name`            | `str`  | `human`    | Model name for VLM interaction. Affects the path for saving images during interaction. |
| `task_num`              | `int`  | `5`        | The number of tasks given to VLM in total.                   |
| `use_lv3_task`          | `bool` | `false`    | Whether to use level 3 tasks (dual tasks with intermediate steps). |
| `generate_mistake_note` | `bool` | `false`    | Whether to generate mistake notes for reflection.            |
| `use_mistake_note`      | `int`  | `0`        | How many trial notes in the reflection file are to use.      |
| `cache_enabled`         | `bool` | `true`     | Whether to enable caching for performance optimization.      |
| `random_seed`           | `int`  | `null`     | Random seed for reproducibility. If null, uses system time. |
| `log_level`             | `str`  | `INFO`     | Logging level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

### File Paths

| Parameter                    | Type  | Default                                | Description                                                  |
| ---------------------------- | ----- | -------------------------------------- | ------------------------------------------------------------ |
| `input_json_path`            | `str` | `apt_0.scene_instance.json`           | Input scene file path.                                       |
| `output_json_path`           | `str` | `./replica_apt_0_parsed.json`         | Parsed output file path.                                     |
| `entity_json_path`           | `str` | `./replica_apt_0_entities.json`       | Entity file path after gravity adjustment.                   |
| `output_dir`                 | `str` | `./output/`                           | Output directory for all generated files.                    |
| `image4rename_path`          | `str` | `./image4rename/`                     | Path for images used in VLM renaming process.               |
| `rename_dict_path`           | `str` | `./rename_dict.json`                  | Path to the renaming dictionary file.                       |
| `result_file_path`           | `str` | `./result.txt`                        | Path to save benchmark results.                              |
| `reflection_txt_load_path`   | `str` | `./load_reflection.txt`               | Path for loading reflection notes.                           |
| `reflection_txt_save_path`   | `str` | `./save_reflection.txt`               | Path for saving reflection notes.                            |

### Pickle Files

| Parameter                     | Type       | Default                | Description                                    |
| ----------------------------- | ---------- | ---------------------- | ---------------------------------------------- |
| `scene_graph_pkl_load_path`   | `str/null` | `./scene_graph.pkl`   | Scene graph load path. If exists, skip generation. |
| `scene_graph_pkl_save_path`   | `str/null` | `./scene_graph.pkl`   | Scene graph save path for future use.         |
| `atomic_task_pkl_load_path`   | `str/null` | `./atomic_task.pkl`   | Atomic tasks load path. If exists, skip generation. |
| `atomic_task_pkl_save_path`   | `str/null` | `./atomic_task.pkl`   | Atomic tasks save path for future use.        |


## Adding Custom Datasets

Aside from AI2THOR and ReplicaCAD, other maniskill-style scenes can also be parsed with ``src/preprocessing/maniskill_parser.py``. 

If you want to run the benchmark on other scenes, refer to ``src/preprocessing/base_parser.py``, ``src/preprocessing/maniskill_parser.py`` and ``src/preprocessing/sunrgbd_parser.py`` to add new parsers for other data formats.


##  Implementation Details

【To Be Updated】


## VLM API

We use OpenRouter API for VLM interaction. To benchmark VLM agents, you need to set up your OpenRouter API key and model address in the configuration file or command line arguments.

### Requirements:

* Openrouter API key (begin with `sk-or-v1`)

### Encode your OpenRouter API_key

Modify  ``OpenRouterConfig `` class in `src/utils/config_manager.py` (or your yml config file) with your API key.

For more details on using the OpenRouter API, refer to the OpenRouter  [official documentation](https://openrouter.ai/docs/quickstart).














