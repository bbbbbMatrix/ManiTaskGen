# **ManiTaskGen: A Comprehensive Task Generator for Benchmarking and Improving Vision-Language Agents on Embodied Decision-Making**





[![arXiv](https://img.shields.io/badge/arXiv-2505.20726-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![Website](https://img.shields.io/badge/Project%20Page-Visit-brightgreen?style=for-the-badge&logo=globe)](https://manitaskgen.github.io/)



This is the official repository for the **ManiTaskGen** project. It Includes instructions on dataset downloading, environment setting up and commands for **ManiTaskGen** benchmarks and agent finetuning pipelines.



## Core Features & Value Propositions



**ManiTaskGen** is introduced as a novel system that addresses this limitation by automatically generating a **comprehensive, diverse, and logically near-exhaustive** set of mobile manipulation tasks for any given scene. This system provides a crucial resource for both the rigorous evaluation and iterative improvement of Vision-Language Agents (VLAs) on embodied decision-making.



### Key Features:



- **Comprehensive Task Generation:** Automatically explores the full spectrum of feasible tasks within an arbitrary scene, surpassing the scale and diversity of manually annotated datasets.
- **Dual Task Modalities:** ManiTaskGen generates tasks covering two critical paradigms of embodied intelligence:
  - **Process-based Tasks:** Specific, step-by-step instructions focusing on the required action sequence (e.g., "move the object from X to Y").
  - **Outcome-based Tasks:** Abstract instructions focusing on achieving a desired final state or configuration (e.g., "clean the table").
- **Universal Scene Applicability:** Demonstrates validity across both **simulated environments** (e.g., ReplicaCAD, AI2THOR) and **real-world scene datasets** (e.g., SUN-RGBD).
- **Automatic Benchmarking:** The system leverages the generated task sets to automatically construct large-scale benchmarks for systematic and in-depth evaluation of existing VLM Agents.
- **Resource for Improvement:** The rich dataset of generated tasks provides a valuable foundation for improving general VLM policies through methods like inference-time reinforcement learning (RL) and model refinement.





## Code Organization 

```
.
├── scripts/                          # END-TO-END WORKFLOW ENTRY POINTS
│   ├── 01_preprocessing.sh           # STAGE 1: Execute scene standardization and object renaming.
│   ├── 02a_gen_process_based_task.sh # STAGE 2A: Generate sequential (Process-based) tasks.
│   ├── 02b_gen_outcome_based_tasks.sh# STAGE 2B: Generate final-state (Outcome-based) tasks.
│   └── 03_run_benchmark.sh           # STAGE 3: Run the VLM Agent benchmark executor. (NOTE: Only supports Process-based Tasks.)
├── config/                           # SYSTEM & RUNTIME CONFIGURATION
│   ├── default_config.yml            # Default system configuration (paths, VLM models, parameters).
│   └── env.yml                       # Environment variables and sensitive API keys.
├── data/                             # VLM INTERACTION ASSETS
│   └── templates/                    # Core VLM prompts used for various stages.
│       ├── manitask_ot200.txt        # Outcome-based task templates (MANITASKOT-200).
│       ├── renaming_engine.json      # Prompts for object renaming and standardization.
│       ├── benchmark_prompts.json    # Prompts for the VLM Agent during benchmark execution.
│       └── ...                       # Other specialized prompts (e.g., reflection, voting).
├── src/                              # CORE PYTHON LOGIC
│   ├── core/                         # MAIN TASK ENGINE: Definition, Generation, and Execution.
│   │   ├── gen_scene_graph.py        # CAUSE: Processed objects are unstructured. EFFECT: Builds the structured Scene Graph for task logic.
│   │   ├── task_primitive.py         # Defines the low-level goal representations (primitives) used by task generators.
│   │   ├── process_based_task_generator.py # Generates tasks based on prescriptive action sequences.
│   │   ├── outcome_based_task_generator.py # Generates tasks based on final outcome states.
│   │   ├── task_feasibility_evaluator.py # validate the feasibility of generated tasks using VLM voting.
│   │   └── benchmark_executor.py     # Manages task execution and interaction (The final testing module).
│   ├── geometry/                     # PHYSICAL CONSTRAINTS & FEASIBILITY CHECKS
│   │   ├── placement_helper.py       # Object placement validation and assistance (crucial for generating realistic tasks).
│   │   ├── ground_coverage_analyzer.py # Examines ground coverages for agent interaction analysis.
│   │   └── ...                       # Other files for geometric processing (hulls, meshes, queries).
│   ├── preprocessing/                # SCENE STANDARDIZATION PIPELINE
│   │   ├── renaming_engine.py        # Object renaming and standardization using VLM consensus.
│   │   ├── sunrgbd_parser.py         # Specialized parser for SUNRGBD-style (depth/image) scenes.
│   │   └── ...                       # Other scene parsers (base, maniskill) and visualization helpers.
│   ├── vlm_interaction/              # VLM API ABSTRACTION LAYER
│   │   └── ...                       # Modules to communicate uniformly with various VLM backends.
│   └── utils/                        # SHARED UTILITIES
│       ├── config_manager.py         # Handles loading and managing all system configurations.
│       ├── string_convertor.py       # Utilities for cleaning and normalizing object names.
│       └── image_renderer/           # Tools for converting coordinates and rendering scenes for VLM input.
└── docs/                             # DOCUMENTATION & GUIDES
```



## Installation

For installation, refer to **[INSTALLATION.md](./INSTALLATION.md)**




## QuickStart






Please follow the instructions in **[INSTALLATION.md](./INSTALLATION.md)** to set up the environment.

### Real-world Scenes (SUNRGBD)

Though the real-world scene benchmark is not yet supported in the current version, you can still run the preprocessing and task generation steps on SUNRGBD dataset with the following command:

```
python src/preprocessing/sunrgbd_parser.py --scene_path=path/to/SUNRGBD/dataset --output_path=/path/to/output/folder
```


### Maniskill-style Scenes (AI2THOR & ReplicaCAD)





After finishing environment setup, modify the config file from the example global config file. The two config you probably have to modify is `stage1_pre_processing:input_json_path` and `common:openrouter:api_key`. After that, run the following scripts:

```
CONFIG_FILE=path/to/config.yaml bash scripts/run_01_preprocessings.sh 
CONFIG_FILE=path/to/config.yaml bash scripts/run_02a_gen_process_tasks.sh 
CONFIG_FILE=path/to/config.yaml bash scripts/run_02b_gen_outcome_tasks.sh 
CONFIG_FILE=path/to/config.yaml bash scripts/run_03_run_benchmark.sh 
```



For first time usage, you can modify the configs in ``config/default_config.yml``, and just run:

```
CONFIG_FILE=config/default_config.yml bash scripts/run_01_preprocessings.sh 
CONFIG_FILE=config/default_config.yml bash scripts/run_02a_gen_process_tasks.sh 
CONFIG_FILE=config/default_config.yml bash scripts/run_02b_gen_outcome_tasks.sh 
CONFIG_FILE=config/default_config.yml bash scripts/run_03_run_benchmark.sh 
```



Under default configs, once run:

* `latest_config/used_config.yaml` records your latest config. 
* ``run/configs_used`` folder records all your config, named by ``used_config_{ts}.yaml``. 
* The `runs/` folder stores all intermediate artifacts and final outputs generated during runtime. Re-running the same step (`.sh` script) of the pipeline may **overwrite** the existing contents within this folder.



It's recommended to set `path/to/config.yaml` to `latest_config/used_config.yaml` after you modified the config file for the first time, so that the latest config will be used by default.











## Modular Execution Guide 



The ManiTaskGen pipeline is structured into a sequential, four-step execution process. You must execute these modules in order, as the output of one step serves as the essential input for the next.



### Overview





| Scripts                           | Feature                                                      | Input                                                        | Output(default path)                                         |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ``run_01_preprocessings.sh``          | Scene preprocessing                                          | `data/datasets/{dataset}`                                    | `runs/cache/scene_entities.json`<br />`runs/cache/scene_parsed.json`<br />`runs/cache/rename_dict.json`<br />`runs/images/image4rename/xxx.png`<br />`runs/cache/scene_graph.pkl` |
| ``run_02a_gen_process_based_task.sh`` | Generate process-based tasks                                | `runs/cache/scene_graph.pkl`<br />`runs/cache/scene_entities.json` | `runs/cache/process_based_task.pkl`<br />`runs/output/process_based_task.txt` |
| ``run_02b_gen_outcome_based_task.sh`` | Generate outcome-based tasks                                 | `runs/cache/scene_graph.pkl`<br />`runs/cache/scene_entities.json` | `runs/output/outcome_based_task.txt`<br />`runs/images/image4vote/xxx.png` |
| ``run_03_run_benchmark.sh ``          | Run benchmark execution                                      | `runs/cache/process_based_task.pkl`                         | `runs/output/result.txt`<br />`runs/images/image4interact/xxx.png`|
| ``config.sh ``                        | An auxiliary script for setting paths                         | /                                                            | /                                                            |





Below are the detailed instructions for running each script in sequence. 




### Step 1: Scene Preprocessing (``run_01_preprocessings.sh``)

This initial step is crucial for transforming raw scene data into a structured format suitable for robust task generation, primarily focusing on resolving object ambiguity.

- **Goal:** Convert raw scene information (e.g., object poses, bounding boxes) into a standardized format and resolve object naming ambiguities using a VLM. The parsed scene data will be stored in ``runs/cache/scene_entities.json`` and ``runs/cache/scene_parsed.json`` for subsequent task generation.
- **Functionality Overview**:  
  * **Generating Scene Graph**: Output a structured file containing the **Receptacle-Aware 3D Scene Graph**, which includes crucial information about object-receptacle relationships. The graph will be stored in serialized (`runs/cache/scene_graph.pkl`) formats.
  * **VLM-Enhanced Renaming (Optional but Recommended):**  To address ambiguities arising from casual object naming in some datasets (e.g., identical names distinguished only by numerical suffixes ), a user-configured Vision-Language Model (VLM) can be leveraged. **This step is highly recommended** for process-based tasks as it ensures the renaming of objects into a more descriptive `(category_name)_(specific_name)` format, which is essential for accurate task difficulty classification (Level 1 vs. Level 2). The renaming results will be saved in ``runs/cache/rename_dict.json`` (if renaming is disabled, this file will be an empty dict), and images used for VLM querying will be stored in ``runs/images/image4rename/``.
  * **Gravity Adjustment (Optional but Recommended):** To address object dislocation in the dataset, we'll load and save the object information once in Sapien before start processing. This requires setting correct ``collision_path``. ``runs/cache/scene_parsed.json`` will contain the original object poses, and ``runs/cache/scene_entities.json`` will contain the adjusted object poses.

* **Dependencies:** 

  * dataset
  * access to a configured VLM (e.g. via API key) for the object renaming, enabled when ``use_renaming_engine=True``
  * object collision paths for the Gravity adjustment step, enabled when ``adjust_with_gravity=True``

* **Key Arguments:**

  * For full argument examples, please refer to ``stage1_pre_processing`` column under ``configs/staged_config.yaml``. 

  * | **Key Argument**          | **Description**                                              | **YAML Path**                                 | **Default/Usage**                            |
    | ------------------------- | ------------------------------------------------------------ | --------------------------------------------- | -------------------------------------------- |
    | `use_renaming_engine`     | whether enable VLM-Enhanced Renaming.                        | `stage1_pre_processing:use_renaming_engine`   | `false`                                      |
    | `rename_engine:model`     | model used for renaming.                                     | `stage1_pre_processing:rename_engine:model`   | `openai/gpt-4.1-mini`                        |
    | `input_json_path`         | path to the scene json.                                      | `stage1_pre_processing:input_json_path`       | `./data/.../apt_0.scene_instance.json`       |
    | ``output_json_path``      | path to the parsed scene json file.                          | `stage1_pre_processing:output_json_path`      | `${run_dir}/cache/scene_parsed.json` |
    | ``object_config_path``    | path to the object configs.                                  | `stage1_pre_processing:object_config_path`    |                                              |
    | ``collision_path_prefix`` | path to collisions.  needed to be valid when ``adjust_with_gravity=True`` | `stage1_pre_processing:collision_path_prefix` |                                              |
    | ``rename_dict_path``      | path to the renaming results.                                | `stage1_pre_processing:rename_dict_path`      | `${run_dir}/cache/rename_dict.json`          |

  



### Step 2a: Generate Process-based Tasks (``run_02a_gen_process_based_task.sh``)

This module systematically generates specific, step-by-step mobile manipulation tasks (Level 1, 2, and 3). 

These tasks are directly executable by an embodied agent. 

* **Goal**: Generate a large, diverse set of **Process-based Tasks** that specify the required action sequence, .
* **Functionality Overview**: 
  * **Task Construction**: Task instances are generated via the systematic sampling of objects and receptacles within the scene. The output consists of a task serialization ``runs/cache/process_based_task.pkl`` file for programmatic loading, alongside a ``runs/output/process_based_task.txt`` file containing the task's natural language and PDDL definitions. The generated tasks cover three complexity levels (as defined in the paper):
    * **Level 1 (Single Step - Unique):** Simple Pick-and-Place involving a uniquely identifiable target object.
    * **Level 2 (Single Step - Ambiguous):** Simple Pick-and-Place where the target object requires additional descriptive attributes for disambiguation.
    * **Level 3 (Multi Steps):** Sequential execution of any number of Level 1 or Level 2 tasks, connected by **THEN** operators. 
    Refer to the key arguments for how to control the level of task by setting the maximum task length and number of tasks generated.
* **Dependencies:** Requires the preprocessed scene file from Step 1, i.e. the ``runs/cache/scene_graph.pkl`` and ``runs/cache/scene_entities.json``. See Overview table for details.
* **Key Arguments:**
  
  * For full argument examples, please refer to ``stage2a_process_task_generation`` column under ``configs/staged_config.yaml``. 
  
  * | **Key Argument**                      | **Description**                                             | **YAML Path**                                                | **Default/Usage**                                            |
    | ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | `process_based_task:max_task_length`  | Maximum length of the generated tasks.                       | `stage2a_process_task_generation:process_based_task:max_task_length` | `5` (Level 1-2 tasks require 1, Level 3 tasks require min 2) |
    | `process_based_task:max_task_num`     | Number of tasks to generate (if the scene allows).           | `stage2a_process_task_generation:process_based_task:max_task_num` | `10`                                                         |
    | `process_based_task:use_level1_tasks` | Whether to exclusively generate Level 1 tasks. Setting this to `true` enforces `max_task_length` to 1 and uses only non-ambiguous objects. | `stage2a_process_task_generation:process_based_task:use_level1_tasks` | `false`                                                      |
    | `process_based_task_txt_save_path`    | Path to save the generated process-based task description file (`.txt`). | `stage2a_process_task_generation:process_based_task_txt_save_path` | `${run_dir}/output/process_based_task.txt`                   |




### Step 2b: Generate Process-based Tasks (``run_02b_gen_outcome_based_task.sh``)

This module generates abstract tasks that describe a desired final state of the environment (Level 4). We left benchmarking these tasks into future work.

- **Goal:** Generate **Outcome-based Tasks** that focus on the desired goal state rather than the execution process.
- **Functionality Overview:**
  * **Template Instantiation:** Tasks are generated by instantiating a set of carefully curated, human-designed abstract goal templates (e.g., "Sort all objects...", "Group items...").
  * **VLM Feasibility Voting (Critical):** To ensure task realism, multiple VLMs (configurable) are queried to vote on the **feasibility** of the instantiated abstract tasks within the specific scene. Only tasks with high consensus are kept. The images used for voting are stored in ``runs/images/image4vote/``, and the final tasks are saved in ``runs/output/outcome_based_task.txt``.
- **Dependencies:** Requires the preprocessed scene file from Step 1, i.e. the ``runs/cache/scene_graph.pkl`` and ``runs/cache/scene_entities.json``. See Overview table for details.
- **Key Arguments (Please Supplement):**
  
  - For full argument examples, please refer to ``stage2b_outcome_task_generation`` column under ``configs/staged_config.yaml``. 
  
  - | **Key Argument**                          | **Description**                                             | **YAML Path**                                                | **Default/Usage**                          |
    | ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------ |
    | `outcome_based_task:task_num_per_pattern` | The number of tasks generated using each distinct template/pattern. | `stage2b_outcome_task_generation:outcome_based_task:task_num_per_pattern` | `5`                                        |
    | `outcome_based_task:vlm_list` | The vlms for VLM feasibility voting. | `stage2b_outcome_task_generation:outcome_based_task:vlm_list` | `["openai/gpt-4.1", "anthropic/claude-3.5-haiku", "google/gemini-2.5-flash-lite-preview-06-17"]` |
    | `manitaskot_pattern_file`                 | Path to the template file (`MANITASKOT-200`) used for generating outcome-based tasks. | `stage2b_outcome_task_generation:manitaskot_pattern_file`    | `data/templates/manitask_ot200.txt`        |
    | `image4vote_path`                         | Path where scene images will be stored before being sent to the VLM ensemble for feasibility voting. | `stage2b_outcome_task_generation:image4vote_path`            | `${run_dir}/images/image4vote`             |
    | `outcome_based_task_txt_save_path`        | Path to save the generated outcome-based task description file. | `stage2b_outcome_task_generation:outcome_based_task_txt_save_path` | `${run_dir}/output/outcome_based_task.txt` |




### Step 3: Run Benchmark Execution (`run_03_run_benchmark.sh`)



This module acts as the benchmark executor, connecting a target VLM Agent to the embodied simulation environment to evaluate its performance. This executor currently **only supports Process-based Tasks (Level 1-3)** for automated, end-to-end evaluation. We left the benchmarking of Outcome-based Tasks into future work.



- **Goal:** Execute the generated tasks on a specified VLM Agent and collect performance metrics.
- **Functionality Overview:**
  * **VLM Agent Integration:** Connects the chosen VLM Agent to the sapien simulator via an abstract, discrete action space interface. The images for interaction are stored in ``runs/images/image4interact/``, and the results of the benchmarking are saved in ``runs/output/result.txt``.
  * **Performance Evaluation:** Runs the agent through all tasks in the input file and logs the performance. Key metrics include **Success Rate (SR)** and **Intermediate Points (IP)**.
- **Dependencies:** Requires the `.pkl` file from Step 1 and Step 2A and a working environment/simulator setup. See Overview table for details.
- **Key Arguments :**
  
  - For full argument examples, please refer to ``stage3_benchmark`` column under ``configs/staged_config.yaml``. 
  
  - | **Key Argument**                       | **Description**                                             | **YAML Path**                                           | **Default/Usage**                  |
    | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------- | ---------------------------------- |
    | **`benchmark_model_name`**             | **Target Agent.** The name of the VLM model to be benchmarked. Refer to the  https://openrouter.ai/ for valid values. | `stage3_benchmark:benchmark_model_name`                 | `openai/gpt-4.1-mini`              |
    | `mode`                                 | Execution mode. Set to `online` to automatically test the target VLM model, or `manual` for human-simulated interaction. | `stage3_benchmark:mode`                                 | `manual`                           |
    | `vlm_interactor:MAX_INTERACTION_COUNT` | Maximum interaction steps allowed per task. Exceeding this limit forces a `CALL_END` and task evaluation. | `stage3_benchmark:vlm_interactor:MAX_INTERACTION_COUNT` | `20`                               |
    | `generate_mistake_note`                | Whether to generate mistake notes used for self-reflection (part of the VLM improvement method). | `stage3_benchmark:generate_mistake_note`                | `true`                             |
    | `result_file_path`                     | Output path for the final benchmark results (Success Rate, Intermediate Points, etc.). | `stage3_benchmark:result_file_path`                     | `${run_dir}/output/result.txt`     |
    | `image4interaction_path`               | Path to store images generated during the benchmarking interaction process. | `stage3_benchmark:image4interaction_path`               | `${run_dir}/images/image4interact` |



## Additional Utilities 



Configs that not listed above should not require frequent changes. For their usage, see **[FULL_CONFIG_REFERENCE.md](./FULL_CONFIG_REFERENCE.md)**



## Advanced Resources 



To facilitate development, research, and contribution to the **ManiTaskGen** framework, we provide detailed documentation on the system's architecture and core algorithms.

- **Technical Details:** Understand **how** **ManiTaskGen** implements the very process elaborated in the paper and guarantees task feasibility. Read **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)**. 
- **API Reference:** Find comprehensive documentation on reusable code components, utility libraries (e.g., `basic_geometries`), and the internal data structures. Read **[API_REFERENCE.md](API_REFERENCE.md)**



## Contributing

We warmly welcome and appreciate contributions of all forms—from bug reports and feature suggestions to documentation improvements and code development. ManiTaskGen aims to be a universal framework, and community input is essential for its growth.



### How to Contribute





#### Reporting Bugs and Issues



If you encounter a bug, an error, or unexpected behavior while running the pipeline or testing an agent:

- Please open an **Issue** on the project repository.
- Use the designated bug report template if available.
- Include the following key information: the version of ManiTaskGen you are using, the configuration file (`staged_config.yaml`) used, the specific script (`.sh`) that failed, and the full error traceback.



#### Suggesting Features and Improvements



If you have ideas for enhancing task generation diversity, improving the benchmarking capabilities, or supporting new VLMs:

- Open an **Issue** and label it as `Feature Request`.
- Clearly describe the proposed feature and explain its potential value to the community or its relevance to embodied decision-making research.



#### Submitting Code (Pull Requests)



We welcome code contributions, especially in the following areas:


- **Executor Improvements:** Extending the benchmark executor (`03_run_benchmark.sh`) to support new agent interfaces or automated evaluation for Outcome-based Tasks (Level 4).
- **Core Utility Enhancement:** Improving or optimizing core algorithms in libraries like `basic_geometries`.

**Code Submission Guidelines:**

* **Branching:** Base your work off the main branch (e.g., `main` or `dev` branch, if specified).

* **Coding Style:** All Python code submissions **must** be formatted using **Black** for consistent code style across the project.

```bash
black .
```

* **Testing:** Ensure your changes do not break existing functionality. Run relevant tests before submitting your Pull Request.

* **Documentation:** All new functions, classes, and complex code blocks must include comprehensive **Docstrings** and be referenced in the appropriate documentation files (`TECHNICAL_DETAILS.md` and `API_REFERENCE.md`).