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
CONFIG_FILE=path/to/config.yaml bash scripts/run_01_preprocessings.sh 
CONFIG_FILE=path/to/config.yaml bash scripts/run_99_item_modification_only.sh   # run item modification interaction
```

For first time usage, you can just run:

```bash
CONFIG_FILE=config/default_config.yml bash scripts/run_01_preprocessings.sh 
CONFIG_FILE=config/default_config.yml bash scripts/run_99_item_modification_only.sh   # run item modification interaction
```

It's recommended to set ``path/to/config.yaml`` to ``latest_config/used_config.yaml`` after you modified the config file for the first time, so that the latest config will be used by default.


The following table shows the features and IOs for each script:

| Scripts                                       | Feature                                                      | Input                                                        | Output(default path)                                         |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ``scripts/run_01_preprocessing.sh``                         | (a) Parse original dataset, generate scene graph and dump them<br>(b) Rename objects interaction                        | data/datasets/ai2thor                                                           | runs/cache/entity_scene.json
runs/cache/scene_graph.pkl
runs/visualizations/scene_graph.dot,
runs/visualizations/scene_graph.txt
data/cache/rename_dict.json
data/images/image4rename/xxx.png                                                     |
| ``scripts/run_99_item_modification_only.sh``                         | run item_modification interaction                        | /                                                            | visualizations/final_scene_graph.json                                                           |








