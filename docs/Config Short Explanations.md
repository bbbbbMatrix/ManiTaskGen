# Configuration Parameters Documentation

This document describes all available configuration parameters in the task generation system.


## Running 


run_

## Table of Contents

- [Configuration Parameters Documentation](#configuration-parameters-documentation)
  - [Running](#running)
  - [Table of Contents](#table-of-contents)
  - [Raw Scene Configuration (RawSceneConfig)](#raw-scene-configuration-rawsceneconfig)
  - [SAPIEN Configuration (SapienConfig)](#sapien-configuration-sapienconfig)
    - [Path Configuration](#path-configuration)
    - [Rendering Configuration](#rendering-configuration)
    - [Lighting Configuration](#lighting-configuration)
    - [Material and Adjustment Configuration](#material-and-adjustment-configuration)
    - [Shading Configuration](#shading-configuration)
    - [File Configuration](#file-configuration)
  - [Scene Type Configuration (SceneType)](#scene-type-configuration-scenetype)
  - [Basic Geometry Configuration (BasicGeometryConfig)](#basic-geometry-configuration-basicgeometryconfig)
  - [Image Renderer Configuration (ImageRendererConfig)](#image-renderer-configuration-imagerendererconfig)
    - [Basic Rendering Parameters](#basic-rendering-parameters)
    - [Camera Configuration](#camera-configuration)
    - [Image Parameters](#image-parameters)
    - [Optimization Parameters](#optimization-parameters)
    - [Color Configuration](#color-configuration)
  - [Concave Processor Configuration (ConcaveProcessorConfig)](#concave-processor-configuration-concaveprocessorconfig)
  - [Ground Coverage Configuration (GroundCoverageConfig)](#ground-coverage-configuration-groundcoverageconfig)
  - [Task Primitive Configuration (TaskPrimitiveConfig)](#task-primitive-configuration-taskprimitiveconfig)
  - [Atomic Task Configuration (AtomicTaskConfig)](#atomic-task-configuration-atomictaskconfig)
  - [Scene Configuration (SceneConfig)](#scene-configuration-sceneconfig)
  - [OpenRouter Configuration (OpenRouterConfig)](#openrouter-configuration-openrouterconfig)
  - [Mesh Processor Configuration (MeshProcessorConfig)](#mesh-processor-configuration-meshprocessorconfig)
  - [VLM Interactor Configuration (VlmInteractorConfig)](#vlm-interactor-configuration-vlminteractorconfig)
  - [Scene Element Configuration (SceneElementConfig)](#scene-element-configuration-sceneelementconfig)
  - [Rectangle Query Configuration (RectangleQueryConfig)](#rectangle-query-configuration-rectanglequeryconfig)
  - [Benchmark Executor Configuration (BenchmarkExecutorConfig)](#benchmark-executor-configuration-benchmarkexecutorconfig)
    - [Basic Configuration](#basic-configuration)
    - [File Path Configuration](#file-path-configuration)
    - [Camera and Rotation Configuration](#camera-and-rotation-configuration)
    - [Task Execution Configuration](#task-execution-configuration)
    - [Feature Switches](#feature-switches)
  - [Task Interaction Configuration (TaskInteractionConfig)](#task-interaction-configuration-taskinteractionconfig)
  - [Configuration Manager](#configuration-manager)
    - [Key Methods](#key-methods)
    - [Utility Functions](#utility-functions)
  - [Global Configuration](#global-configuration)
    - [Core Feature Switches](#core-feature-switches)
    - [File Path Configuration](#file-path-configuration-1)
    - [Runtime Mode Configuration](#runtime-mode-configuration)
    - [Task Configuration](#task-configuration)
  - [Usage Examples](#usage-examples)
  - [Configuration File Format](#configuration-file-format)
  - [SAPIEN Configuration (SapienConfig)](#sapien-configuration-sapienconfig-1)
    - [YAML Format](#yaml-format)
    - [JSON Format](#json-format)

## Raw Scene Configuration (RawSceneConfig)

```

class RawSceneConfig:
    dataset_root_path: str = (
        "/mnt/windows_e/workplace/task_generation/replica_dataset"  # Path to the Replica dataset root directory
    )
    object_config_path: str = (
        "/mnt/windows_e/workplace/task_generation/replica_dataset/configs"  # Path to the object configuration
    )

    desired_objects: List[str] = field(default_factory=lambda: None)
    not_desired_objects: List[str] = field(
        default_factory=lambda: [
            "frl_apartment_handbag",
            "frl_apartment_cushion_01",
            "frl_apartment_monitor",
            "frl_apartment_cloth_01",
            "frl_apartment_cloth_02",
            "frl_apartment_cloth_03",
            "frl_apartment_cloth",
            "frl_apartment_umbrella",
            "frl_apartment_tv_screen",
            "frl_apartment_indoor_plant_01",
            "frl_apartment_monitor_stand",
            "frl_apartment_setupbox",
            "frl_apartment_beanbag",
            "frl_apartment_bike_01",
            "frl_apartment_bike_02",
            "frl_apartment_indoor_plant_02",
            "frl_apartment_picture_01",
            "frl_apartment_towel",
            "frl_apartment_rug_01",
            "frl_apartment_rug_02",
            "frl_apartment_rug_03",
            "frl_apartment_mat",
            "frl_apartment_tv_object",
        ]
    )
    desired_articulations: List[str] = field(default_factory=lambda: None)
    not_desired_articulations: List[str] = field(default_factory=lambda: None)

```

Controls the loading and processing of raw scene data.

| Parameter                   | Type        | Default Value                                                | Description                                                  |
| --------------------------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `dataset_root_path`         | `str`       | `/mnt/windows_e/workplace/task_generation/replica_dataset`   | Path to the Replica dataset root directory                   |
| `object_config_path`        | `str`       | `/mnt/windows_e/workplace/task_generation/replica_dataset/configs` | Path to the object configuration                             |
| `desired_objects`           | `List[str]` | `None`                                                       | List of desired objects to load. If it's set, only the  objects in the list will be loaded to the scene. |
| `not_desired_objects`       | `List[str]` | [See code]                                                   | List of objects to exclude, includes various furniture and decorative items |
| `desired_articulations`     | `List[str]` | `None`                                                       | List of desired articulated objects                          |
| `not_desired_articulations` | `List[str]` | `None`                                                       | List of articulated objects to exclude                       |

## SAPIEN Configuration (SapienConfig)

Configuration parameters for the SAPIEN physics simulation engine.

### Path Configuration
| Parameter               | Type  | Default Value                                                | Description                       |
| ----------------------- | ----- | ------------------------------------------------------------ | --------------------------------- |
| `dataset_root_path`     | `str` | `/mnt/windows_e/workplace/task_generation/replica_dataset`   | SAPIEN dataset root directory     |
| `visual_path_prefix`    | `str` | `/mnt/windows_e/workplace/task_generation/replica_dataset/objects` | Path prefix for visual objects    |
| `collision_path_prefix` | `str` | `/mnt/windows_e/workplace/task_generation/replica_dataset/objects/convex` | Path prefix for collision objects |
| `urdf_path_prefix`      | `str` | `/mnt/windows_e/workplace/task_generation/replica_dataset/urdf` | Path prefix for URDF files        |

### Rendering Configuration
| Parameter              | Type    | Default Value | Description          |
| ---------------------- | ------- | ------------- | -------------------- |
| `camera_shader`        | `str`   | `"default"`   | Camera shader type   |
| `ray_tracing_denoiser` | `str`   | `"none"`      | Ray tracing denoiser |
| `time_step`            | `float` | `0.01`        | Simulation time step |
| `ground_altitude`      | `float` | `0.0`         | Ground altitude      |

### Lighting Configuration
| Parameter           | Type          | Default Value     | Description                     |
| ------------------- | ------------- | ----------------- | ------------------------------- |
| `ambient_light`     | `List[float]` | `[0.5, 0.5, 0.5]` | Ambient light color             |
| `directional_light` | `List[Dict]`  | [See code]        | Directional light configuration |
| `point_lights`      | `List[Dict]`  | [See code]        | Point light configuration list  |

### Material and Adjustment Configuration
| Parameter                  | Type             | Default Value | Description                        |
| -------------------------- | ---------------- | ------------- | ---------------------------------- |
| `default_material`         | `Dict[str, Any]` | [See code]    | Default material properties        |
| `cushion_z_offset`         | `float`          | `0.1`         | Z offset for cushions              |
| `urdf_z_adjustment_offset` | `float`          | `0.005`       | Z adjustment offset for URDF files |
| `rotation_offset_x`        | `int`            | `90`          | Rotation offset in degrees         |
| `correction_rotation_x`    | `int`            | `-90`         | Correction rotation in degrees     |

### Shading Configuration
| Parameter                  | Type        | Default Value | Description                     |
| -------------------------- | ----------- | ------------- | ------------------------------- |
| `excluded_shading_objects` | `List[str]` | `["book_03"]` | Objects to exclude from shading |
| `default_shading_mode`     | `int`       | `1`           | Default shading mode            |

### File Configuration
| Parameter             | Type  | Default Value                 | Description              |
| --------------------- | ----- | ----------------------------- | ------------------------ |
| `default_json_file`   | `str` | `"replica_apt_0_parsed.json"` | Default JSON file name   |
| `default_output_file` | `str` | `"entities_apt_0.json"`       | Default output file name |

## Scene Type Configuration (SceneType)

Defines scene type related behavior parameters.

| Parameter                   | Type   | Default Value | Description                                               |
| --------------------------- | ------ | ------------- | --------------------------------------------------------- |
| `NEED_COLLISION_ADJUSTMENT` | `bool` | `True`        | Whether to use collision detection for gravity adjustment |
| `RGBD_SCENE`                | `bool` | `False`       | Whether to use RGBD scene (treats all objects as cuboids) |

## Basic Geometry Configuration (BasicGeometryConfig)

Configuration parameters for basic geometry calculations.

| Parameter | Type    | Default Value | Description                                   |
| --------- | ------- | ------------- | --------------------------------------------- |
| `EPS`     | `float` | `1e-5`        | Precision threshold for geometry calculations |

## Image Renderer Configuration (ImageRendererConfig)

Configuration parameters related to image rendering.

### Basic Rendering Parameters
| Parameter             | Type          | Default Value                           | Description                      |
| --------------------- | ------------- | --------------------------------------- | -------------------------------- |
| `EPS`                 | `float`       | `1e-6`                                  | Rendering precision threshold    |
| `default_fovy`        | `float`       | `np.deg2rad(60.0)`                      | Default field of view in radians |
| `default_fovy_range`  | `List[float]` | `[np.deg2rad(10.0), np.deg2rad(100.0)]` | Field of view range              |
| `default_focus_ratio` | `float`       | `0.5`                                   | Default focus ratio              |
| `default_near`        | `float`       | `0.1`                                   | Near clipping plane distance     |
| `default_far`         | `float`       | `100.0`                                 | Far clipping plane distance      |

### Camera Configuration
| Parameter           | Type          | Default Value             | Description                   |
| ------------------- | ------------- | ------------------------- | ----------------------------- |
| `default_camera_xy` | `List[float]` | `[0.0, 0.0]`              | Default camera XY position    |
| `z_range`           | `List[float]` | `[0.2, 2.5]`              | Camera Z-axis range           |
| `roll_range`        | `List[float]` | `[-np.pi/900, np.pi/900]` | Roll angle range (very small) |

### Image Parameters
| Parameter          | Type    | Default Value | Description             |
| ------------------ | ------- | ------------- | ----------------------- |
| `width`            | `int`   | `1920`        | Image width             |
| `height`           | `int`   | `1080`        | Image height            |
| `font_size`        | `int`   | `48`          | Font size               |
| `number_font_size` | `int`   | `40`          | Number font size        |
| `trans_visiblity`  | `float` | `0.2`         | Transparency visibility |

### Optimization Parameters
| Parameter                        | Type    | Default Value | Description                           |
| -------------------------------- | ------- | ------------- | ------------------------------------- |
| `default_scipy_minimize_ftol`    | `float` | `1e-3`        | SciPy optimization function tolerance |
| `default_scipy_minimize_maxiter` | `int`   | `250`         | SciPy optimization maximum iterations |

### Color Configuration
| Parameter                  | Type                         | Default Value | Description                                           |
| -------------------------- | ---------------------------- | ------------- | ----------------------------------------------------- |
| `high_contrast_color_list` | `List[Tuple[int, int, int]]` | [See code]    | High contrast color list with 27 different RGB colors |

## Concave Processor Configuration (ConcaveProcessorConfig)

Configuration parameters for processing concave geometries.

| Parameter             | Type    | Default Value | Description                                                 |
| --------------------- | ------- | ------------- | ----------------------------------------------------------- |
| `eps`                 | `float` | `1e-4`        | Precision threshold                                         |
| `min_polygon_area`    | `float` | `0.1`         | Minimum polygon area threshold                              |
| `target_aspect_ratio` | `float` | `1.8`         | Target aspect ratio                                         |
| `max_target_strips`   | `int`   | `4`           | Maximum target strips                                       |
| `merge_tolerance`     | `float` | `0.01`        | Merging tolerance                                           |
| `concave_threshold`   | `float` | `0.2`         | Concavity threshold for determining if a polygon is concave |
| `concave_min_area`    | `float` | `1e-5`        | Minimum concave polygon area threshold                      |

## Ground Coverage Configuration (GroundCoverageConfig)

Configuration parameters for ground coverage analysis.

| Parameter       | Type          | Default Value            | Description                                |
| --------------- | ------------- | ------------------------ | ------------------------------------------ |
| `eps`           | `float`       | `1e-3`                   | Precision threshold                        |
| `resolution`    | `float`       | `0.01`                   | Grid resolution in meters                  |
| `min_rect_size` | `float`       | `0.4`                    | Minimum rectangle size                     |
| `global_bounds` | `List[float]` | `[-5.0, 5.0, -5.0, 5.0]` | Global bounds (min_x, max_x, min_y, max_y) |
| `z_range`       | `List[float]` | `[0.2, 1]`               | Z-axis range                               |

## Task Primitive Configuration (TaskPrimitiveConfig)

Configuration parameters for task primitives.

| Parameter            | Type        | Default Value                                          | Description                                 |
| -------------------- | ----------- | ------------------------------------------------------ | ------------------------------------------- |
| `default_action`     | `str`       | `"move"`                                               | Default action type                         |
| `support_directions` | `List[str]` | [See code]                                             | List of supported directions (9 directions) |
| `support_relations`  | `List[str]` | `["at", "on", "in", "around", "between", "freespace"]` | List of supported spatial relations         |

## Atomic Task Configuration (AtomicTaskConfig)

Configuration parameters for atomic tasks.

| Parameter         | Type  | Default Value | Description                      |
| ----------------- | ----- | ------------- | -------------------------------- |
| `max_task_length` | `int` | `5`           | Maximum length of an atomic task |
| `max_task_num`    | `int` | `1000`        | Maximum number of atomic tasks   |

## Scene Configuration (SceneConfig)

Scene-related configuration parameters.

| Parameter                 | Type          | Default Value            | Description                           |
| ------------------------- | ------------- | ------------------------ | ------------------------------------- |
| `global_bounds`           | `List[float]` | `[-5.0, 5.0, -5.0, 5.0]` | Global bounds                         |
| `safety_margin`           | `float`       | `0.1`                    | Safety margin                         |
| `collision_check_enabled` | `bool`        | `True`                   | Whether to enable collision detection |

## OpenRouter Configuration (OpenRouterConfig)

Configuration parameters for OpenRouter API.

| Parameter | Type  | Default Value                                    | Description        |
| --------- | ----- | ------------------------------------------------ | ------------------ |
| `api_key` | `str` | `"Bearer sk-or-v1-YOUR_OPENROUTER_API_KEY_HERE"` | OpenRouter API key |
| `model`   | `str` | `"google/gemini-2.5-flash-lite-preview-06-17"`   | Model name to use  |

## Mesh Processor Configuration (MeshProcessorConfig)

Configuration parameters for mesh processing.

| Parameter             | Type    | Default Value | Description                            |
| --------------------- | ------- | ------------- | -------------------------------------- |
| `min_size`            | `float` | `0.0025`      | Minimum size                           |
| `relative_size_ratio` | `float` | `0.25`        | Relative size ratio                    |
| `EPS`                 | `float` | `1e-6`        | Precision threshold                    |
| `coverage_threshold`  | `float` | `0.6`         | Coverage threshold                     |
| `height_threshold`    | `float` | `0.01`        | Minimum height threshold for platforms |

## VLM Interactor Configuration (VlmInteractorConfig)

Configuration parameters for Vision-Language Model interactor.

| Parameter               | Type  | Default Value | Description               |
| ----------------------- | ----- | ------------- | ------------------------- |
| `MAX_INTERACTION_COUNT` | `int` | `20`          | Maximum interaction count |

## Scene Element Configuration (SceneElementConfig)

Configuration parameters for scene element processing.

| Parameter                 | Type    | Default Value | Description                      |
| ------------------------- | ------- | ------------- | -------------------------------- |
| `contact_eps`             | `float` | `5e-2`        | Contact precision threshold      |
| `bbox_eps`                | `float` | `4e-1`        | Bounding box precision threshold |
| `ground_level_correction` | `float` | `1e-4`        | Ground level correction          |

## Rectangle Query Configuration (RectangleQueryConfig)

Configuration parameters for rectangle queries.

| Parameter | Type    | Default Value | Description         |
| --------- | ------- | ------------- | ------------------- |
| `EPS`     | `float` | `1e-3`        | Precision threshold |

## Benchmark Executor Configuration (BenchmarkExecutorConfig)

Configuration parameters for benchmark executor.

### Basic Configuration
| Parameter                     | Type  | Default Value | Description                          |
| ----------------------------- | ----- | ------------- | ------------------------------------ |
| `max_interaction_count`       | `int` | `20`          | Maximum interaction count            |
| `picture_width`               | `int` | `1366`        | Picture width                        |
| `picture_height`              | `int` | `768`         | Picture height                       |
| `intermediate_task_max_score` | `int` | `4`           | Maximum score for intermediate tasks |

### File Path Configuration
| Parameter                  | Type            | Default Value                                    | Description                  |
| -------------------------- | --------------- | ------------------------------------------------ | ---------------------------- |
| `prompt_templates_path`    | `str`           | `"src/utils/prompts/benchmark_prompts.json"` | Prompt templates path        |
| `reflection_prompts_path`  | `str`           | `"src/utils/prompts/reflection_prompts.json"`    | Reflection prompts path      |
| `image_save_base_path`     | `str`           | `"image4interact/"`                              | Base path for saving images  |
| `reflection_txt_load_path` | `Optional[str]` | `"./load_reflection.txt"`                        | Reflection text loading path |
| `reflection_txt_save_path` | `Optional[str]` | `"./save_reflection.txt"`                        | Reflection text saving path  |

### Camera and Rotation Configuration
| Parameter                    | Type    | Default Value | Description                              |
| ---------------------------- | ------- | ------------- | ---------------------------------------- |
| `default_standing_direction` | `int`   | `0`           | Default standing direction               |
| `rotation_step`              | `int`   | `2`           | Rotation step                            |
| `max_rotation_attempts`      | `int`   | `4`           | Maximum rotation attempts                |
| `default_fovy_deg_min`       | `float` | `40.0`        | Default minimum field of view in degrees |
| `default_fovy_deg_max`       | `float` | `60.0`        | Default maximum field of view in degrees |
| `focus_ratio`                | `float` | `0.6`         | Focus ratio                              |

### Task Execution Configuration
| Parameter                  | Type    | Default Value | Description                         |
| -------------------------- | ------- | ------------- | ----------------------------------- |
| `task_timeout_seconds`     | `float` | `300.0`       | Task timeout in seconds             |
| `enable_detailed_logging`  | `bool`  | `False`       | Whether to enable detailed logging  |
| `save_interaction_history` | `bool`  | `True`        | Whether to save interaction history |
| `validate_actions`         | `bool`  | `True`        | Whether to validate actions         |
| `allow_invalid_actions`    | `bool`  | `False`       | Whether to allow invalid actions    |

### Feature Switches
| Parameter                         | Type   | Default Value | Description                                       |
| --------------------------------- | ------ | ------------- | ------------------------------------------------- |
| `auto_rotate_enabled`             | `bool` | `True`        | Whether to enable auto rotation                   |
| `visibility_check_enabled`        | `bool` | `True`        | Whether to enable visibility check                |
| `object_naming_for_interaction`   | `bool` | `True`        | Whether to enable object naming for interaction   |
| `platform_naming_for_interaction` | `bool` | `True`        | Whether to enable platform naming for interaction |

## Task Interaction Configuration (TaskInteractionConfig)

Task interaction specific configuration parameters.

| Parameter                        | Type   | Default Value | Description                               |
| -------------------------------- | ------ | ------------- | ----------------------------------------- |
| `enable_hint_prompts`            | `bool` | `True`        | Whether to enable hint prompts            |
| `enable_ambiguous_item_handling` | `bool` | `True`        | Whether to enable ambiguous item handling |
| `enable_reflection_prompts`      | `bool` | `True`        | Whether to enable reflection prompts      |

## Configuration Manager

The `ConfigManager` class provides methods to load, save, and manage configurations.

### Key Methods

| Method                          | Description                                           |
| ------------------------------- | ----------------------------------------------------- |
| `load_config(config_file_path)` | Load configuration from file (supports YAML and JSON) |
| `load_from_yaml(config_path)`   | Load configuration from YAML file                     |
| `load_from_json(config_path)`   | Load configuration from JSON file                     |
| `update_from_args(args)`        | Update configuration from command line arguments      |
| `save_to_yaml(config_path)`     | Save current configuration to YAML file               |
| `save_to_json(config_path)`     | Save current configuration to JSON file               |
| `print_config()`                | Print current configuration                           |

### Utility Functions

| Function                          | Return Type               | Description                                              |
| --------------------------------- | ------------------------- | -------------------------------------------------------- |
| `get_config()`                    | `AppConfig`               | Get global configuration                                 |
| `get_config_manager()`            | `ConfigManager`           | Get the global config manager instance                   |
| `get_ground_coverage_config()`    | `GroundCoverageConfig`    | Get ground coverage configuration                        |
| `get_image_renderer_config()`     | `ImageRendererConfig`     | Get image renderer configuration                         |
| `get_concave_processor_config()`  | `ConcaveProcessorConfig`  | Get concave processor configuration                      |
| `get_atomic_task_config()`        | `AtomicTaskConfig`        | Get atomic task configuration                            |
| `get_scene_config()`              | `SceneConfig`             | Get scene configuration                                  |
| `get_scene_type_config()`         | `SceneType`               | Get scene type configuration                             |
| `get_basic_geometry_config()`     | `BasicGeometryConfig`     | Get basic geometry configuration                         |
| `get_openrouter_config()`         | `OpenRouterConfig`        | Get OpenRouter configuration                             |
| `get_mesh_processor_config()`     | `MeshProcessorConfig`     | Get mesh processor configuration                         |
| `get_vlm_interactor_config()`     | `VlmInteractorConfig`     | Get VLM interactor configuration                         |
| `get_scene_element_config()`      | `SceneElementConfig`      | Get scene element configuration                          |
| `get_rectangle_query_config()`    | `RectangleQueryConfig`    | Get rectangle query configuration                        |
| `get_raw_scene_config()`          | `RawSceneConfig`          | Get raw scene configuration                              |
| `get_sapien_config()`             | `SapienConfig`            | Get SAPIEN configuration                                 |
| `get_benchmark_executor_config()` | `BenchmarkExecutorConfig` | Get benchmark executor configuration                     |
| `get_task_interaction_config()`   | `TaskInteractionConfig`   | Get task interaction configuration                       |
| `init_config(config_file_path)`   | `None`                    | Initialize global configuration manager with config file |

## Global Configuration

Global application configuration parameters.

### Core Feature Switches
| Parameter             | Type   | Default Value | Description                                        |
| --------------------- | ------ | ------------- | -------------------------------------------------- |
| `adjust_with_gravity` | `bool` | `True`        | Whether to adjust gravity (may affect object pose) |
| `use_renaming_engine` | `bool` | `False`       | Whether to use renaming engine                     |
| `bbox_only`           | `bool` | `False`       | Whether to use bounding box only                   |
| `cache_enabled`       | `bool` | `True`        | Whether to enable caching                          |

### File Path Configuration
| Parameter                   | Type            | Default Value                     | Description                          |
| --------------------------- | --------------- | --------------------------------- | ------------------------------------ |
| `input_json_path`           | `Optional[str]` | [See code]                        | Scene file path                      |
| `output_json_path`          | `Optional[str]` | `"./replica_apt_0_parsed.json"`   | Output file path                     |
| `entity_json_path`          | `Optional[str]` | `"./replica_apt_0_entities.json"` | Entity file path                     |
| `output_dir`                | `str`           | `"./output/"`                     | Output directory for results         |
| `scene_graph_pkl_load_path` | `Optional[str]` | `"./scene_graph.pkl"`             | Scene graph pickle file loading path |
| `scene_graph_pkl_save_path` | `Optional[str]` | `"./scene_graph.pkl"`             | Scene graph pickle file saving path  |
| `atomic_task_pkl_load_path` | `Optional[str]` | `None`                            | Atomic task pickle file loading path |
| `atomic_task_pkl_save_path` | `Optional[str]` | `None`                            | Atomic task pickle file saving path  |
| `image4rename_path`         | `Optional[str]` | `"./image4rename/"`               | Image for renaming path              |
| `image4interaction_path`    | `Optional[str]` | `"./image4interaction/"`          | Image for interaction path           |
| `rename_dict_path`          | `Optional[str]` | `"./rename_dict.json"`            | Renaming dictionary path             |
| `reflection_txt_load_path`  | `Optional[str]` | `"./load_reflection.txt"`         | Reflection text loading path         |
| `reflection_txt_save_path`  | `Optional[str]` | `"./save_reflection.txt"`         | Reflection text saving path          |
| `result_file_path`          | `Optional[str]` | `"./result.txt"`                  | Result file saving path              |

### Runtime Mode Configuration
| Parameter     | Type            | Default Value | Description                          |
| ------------- | --------------- | ------------- | ------------------------------------ |
| `mode`        | `str`           | `"manual"`    | Runtime mode ("online" or "offline") |
| `model_name`  | `str`           | `"human"`     | Model name                           |
| `log_level`   | `str`           | `"INFO"`      | Log level                            |
| `random_seed` | `Optional[int]` | `None`        | Random seed                          |

### Task Configuration
| Parameter               | Type   | Default Value | Description                       |
| ----------------------- | ------ | ------------- | --------------------------------- |
| `task_num`              | `int`  | `5`           | Number of tasks to generate       |
| `generate_mistake_note` | `bool` | `False`       | Whether to generate mistake notes |
| `use_mistake_note`      | `int`  | `0`           | Whether to use mistake notes      |
| `use_lv3_task`          | `bool` | `False`       | Whether to use level 3 tasks      |

## Usage Examples

```python
from config_manager import get_config, ConfigManager, init_config

# Initialize configuration with file
init_config("config.yaml")

# Get global configuration
config = get_config()

# Access specific configurations
image_config = config.image_renderer
sapien_config = config.sapien

# Modify configuration
config.task_num = 10
config.image_renderer.width = 1280
config.image_renderer.height = 720

# Load configuration from file
config_manager = ConfigManager("config.yaml")

# Save configuration
config_manager.save_to_yaml("output_config.yaml")
config_manager.save_to_json("output_config.json")

# Print current configuration
config_manager.print_config()
```

## Configuration File Format

Supports both YAML and JSON format configuration files. Example:

## SAPIEN Configuration (SapienConfig)

| Parameter           | Type         | Description                                                  |
| ------------------- | ------------ | ------------------------------------------------------------ |
| `dataset_root_path` | `str`        | SAPIEN dataset root directory                                |
| `point_lights`      | `List[Dict]` | <details><summary>Point light configurations</summary>Array of 10 point light objects with position and color properties</details> |

<details>
<summary>Point Lights Default Configuration</summary>

```python
[
    {"position": [1.989, -5.822, 1], "color": [0.5, 0.5, 0.5]},
    {"position": [2.15, -5.9, 1.0], "color": [0.4, 0.4, 0.4]},
    # ... additional 8 point lights
]
```
</details>





### YAML Format
```yaml
image_renderer:
  width: 1280
  height: 720
  default_fovy: 1.047
  
task_num: 10
mode: "offline"
log_level: "DEBUG"

ground_coverage:
  resolution: 0.02
  min_rect_size: 0.5

sapien:
  time_step: 0.02
  camera_shader: "rt"
```

### JSON Format
```json
{
  "image_renderer": {
    "width": 1280,
    "height": 720,
    "default_fovy": 1.047
  },
  "task_num": 10,
  "mode": "offline",
  "log_level": "DEBUG",
  "ground_coverage": {
    "resolution": 0.02,
    "min_rect_size": 0.5
  }
}
```