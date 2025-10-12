from code import interact
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
import numpy as np
import json

# from core import outcome_based_task_generation
import glog
from src.utils.run_context import PathResolver


@dataclass
class RawSceneConfig:

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


@dataclass
class SapienConfig:

    camera_shader: str = "default"
    ray_tracing_denoiser: str = "none"
    time_step: float = 1 / 100.0  # Simulation time step
    ground_altitude: float = 0.0
    ambient_light: List[float] = field(
        default_factory=lambda: [0.5, 0.5, 0.5]
    )  # Ambient light color
    directional_light: List[Dict[str, List[float]]] = field(
        default_factory=lambda: [{"direction": [0, 1, -1], "color": [0.5, 0.5, 0.5]}]
    )
    point_lights: List[Dict[str, List[float]]] = field(
        default_factory=lambda: [
            {"position": [1.989, -5.822, 1], "color": [0.5, 0.5, 0.5]},
            {"position": [2 + 0.15, -5.75 - 0.15, 1.0], "color": [0.4, 0.4, 0.4]},
        ]
    )

    # camera_position: Dict[str, float] = field(
    #     default_factory=lambda: {"x": -5, "y": 0, "z": 6}
    # )
    # camera_rotation: Dict[str, float] = field(
    #     default_factory=lambda: {"r": 0, "p": -np.arctan2(2, 4), "y": 0}
    # )
    # camera_parameters: Dict[str, float] = field(
    #     default_factory=lambda: {"near": 0.05, "far": 100, "fovy": 1}
    # )

    default_material: Dict[str, Any] = field(
        default_factory=lambda: {
            "base_color": [1, 1, 1],
            "metallic": 0.0,
            "roughness": 0.5,
            "specular": 0.5,
            "transmission": 9.0,
        }
    )

    cushion_z_offset: float = 0.1  # Z offset for cushions
    urdf_z_adjustment_offset: float = 0.005  # Z adjustment offset for URDF files

    rotation_offset_x: int = 90  # Rotation offset in degrees
    correction_rotation_x: int = -90  # Correction rotation in degrees

    excluded_shading_objects: List[str] = field(
        default_factory=lambda: ["book_03"]
    )  # Objects to exclude from shading
    default_shading_mode: int = 1  # Default shading mode

    default_json_file: str = "replica_apt_0_parsed.json"  # Default JSON file name
    default_output_file: str = "entities_apt_0.json"  # Default output file name


@dataclass
class SceneType:
    """Scene Type configuration"""

    NEED_COLLISION_ADJUSTMENT: bool = True  # use collision to first apply gravity.
    RGBD_SCENE: bool = False  # use RGBD scene, which will treat everything as cuboids.


@dataclass
class BasicGeometryConfig:
    """Basic Geometry configuration"""

    EPS: float = 1e-5


@dataclass
class ImageRendererConfig:
    """Image Renderer configuration"""

    EPS = 1e-6
    default_fovy: float = np.deg2rad(60.0)  # Default field of view in radians
    default_fovy_range: List[float] = field(
        default_factory=lambda: [np.deg2rad(10.0), np.deg2rad(100.0)]
    )
    default_focus_ratio: float = 0.5
    default_near: float = 0.1
    default_far: float = 100.0
    default_camera_xy: List[float] = field(default_factory=lambda: [0.0, 0.0])
    z_range: List[float] = field(
        default_factory=lambda: [0.2, 2.5]
    )  # Default Z range for camera
    font_size: int = 48
    width: int = 1920
    height: int = 1080
    roll_range: List[float] = field(
        default_factory=lambda: [-np.pi / 900, np.pi / 900]
    )  # Very small range for roll

    number_font_size: int = 40
    default_scipy_minimize_ftol: float = 1e-3
    default_scipy_minimize_maxiter: int = 250
    trans_visiblity: float = 0.2

    high_contrast_color_list = [
        # Deep red series
        (180, 30, 45),  # Deep red
        (220, 50, 50),  # Bright red
        (150, 30, 70),  # Wine red
        # Blue series
        (30, 70, 150),  # Deep blue
        (0, 90, 170),  # Royal blue
        (70, 130, 180),  # Steel blue
        # Green series
        (30, 120, 50),  # Deep green
        (0, 100, 80),  # Forest green
        (60, 140, 60),  # Grass green
        # Yellow/Orange series
        (200, 120, 0),  # Deep orange
        (180, 140, 20),  # Golden yellow
        (215, 150, 0),  # Amber
        # Purple series
        (110, 40, 120),  # Deep purple
        (140, 70, 140),  # Violet
        (90, 50, 140),  # Blue-purple
        # Cyan series
        (0, 110, 120),  # Deep cyan
        (40, 140, 140),  # Blue-green
        (0, 130, 130),  # Teal
        # Neutral tones
        (80, 80, 80),  # Dark gray
        (120, 100, 80),  # Brown
        (70, 90, 100),  # Blue-gray
        # Other high contrast colors
        (180, 80, 80),  # Indian red
        (70, 100, 50),  # Olive green
        (90, 60, 140),  # Deep blue-purple
        (170, 70, 0),  # Ochre
        (80, 60, 30),  # Dark brown
        (150, 80, 100),  # Berry color
    ]


@dataclass
class ConcaveProcessorConfig:
    eps = 1e-4
    min_polygon_area = 0.1  # Minimum polygon area threshold
    target_aspect_ratio = 1.8  # Target aspect ratio
    max_target_strips = 4  # Maximum target strips
    merge_tolerance = 0.01  # Tolerance for merging polygons
    concave_threshold = (
        0.2  # Concavity threshold for determining if a polygon is concave
    )
    concave_min_area = 1e-5  # Minimum concave polygon area threshold


@dataclass
class GroundCoverageConfig:
    """Ground coverage configuration"""

    eps = 1e-3
    resolution: float = 0.01  # Grid resolution (meters)
    min_rect_size: float = 0.4  # Minimum rectangle size
    standing_point_distance: float = 0.8  # Distance from standing point to object
    global_bounds: List[float] = field(
        default_factory=lambda: [-5.0, 5.0, -5.0, 5.0]
    )  # Global bounds (min_x, max_x, min_y, max_y)
    z_range: List[float] = field(default_factory=lambda: [0.1, 1])  # Z-axis range


@dataclass
class TaskPrimitiveConfig:
    """Task primitive configuration"""

    default_action: str = "move"
    support_directions: List[str] = field(
        default_factory=lambda: [
            "front",
            "rear",
            "left",
            "right",
            "center",
            "front-left",
            "front-right",
            "rear-left",
            "rear-right",
        ]
    )
    support_relations: List[str] = field(
        default_factory=lambda: ["at", "on", "in", "around", "between", "freespace"]
    )


@dataclass
class ProcessbasedTaskConfig:
    """Atomic task configuration"""

    max_task_length: int = 5  # Maximum length of an atomic task
    max_task_num: int = 1000  # Maximum number of atomic tasks


@dataclass
class SceneConfig:
    """Scene configuration"""

    global_bounds: List[float] = field(default_factory=lambda: [-5.0, 5.0, -5.0, 5.0])
    safety_margin: float = 0.1
    collision_check_enabled: bool = True


@dataclass
class OpenRouterConfig:
    """OpenRouter configuration"""

    api_key: str = (
        "Bearer sk-or-v1-YourAPIKeyHere"  # Replace with your OpenRouter API key
    )
    model: str = (
        "google/gemini-2.5-flash-lite-preview-06-17"  # See https://openrouter.ai for available models. some are listed below:
    )
    """
     # 'openai/gpt-4.1'
        # 'openai/gpt-4.1-mini'
        # 'qwen/qwen2.5-vl-72b-instruct'
        # 'meta-llama/llama-3.3-70b-instruct'
        #'openai/gpt-4o-2024-11-20'
        #'google/gemini-2.5-flash-preview'
        #'openai/gpt-4o-2024-11-20'
        #'google/gemini-2.5-pro-preview-03-25'
        # "anthropic/claude-3.7-sonnet"
        # anthropic/claude-3.5-haiku
        # "meta-llama/llama-3.2-90b-vision-instruct" #"openai/gpt-4.1-nano"  # "openai/chatgpt-4o-latest" #"meta-llama/llama-3.2-90b-vision-instruct"

    
    
    """


@dataclass
class MeshProcessorConfig:
    """MeshProcessor configuration"""

    min_size: float = 0.0025
    relative_size_ratio: float = 0.25
    EPS: float = 1e-6
    coverage_threshold: float = 0.6
    height_threshold: float = 0.01  # Minimum height threshold for platforms


@dataclass
class VlmInteractorConfig:
    """VLM Interactor configuration"""

    MAX_INTERACTION_COUNT: int = 20


@dataclass
class SceneElementConfig:
    """Scene Element configuration"""

    contact_eps: float = 5e-2
    bbox_eps: float = 4e-1
    ground_level_correction: float = 1e-4


@dataclass
class RectangleQueryConfig:
    """Rectangle Query configuration"""

    EPS: float = 1e-3
    placement_coverage_threshold: float = 0.95


@dataclass
class BenchmarkExecutorConfig:
    """Benchmark Executor configuration"""

    max_interaction_count: int = 20

    picture_width: int = 1366
    picture_height: int = 768

    # prompt_templates_path: str = "data/templates/benchmark_prompts.json"
    # reflection_prompts_path: str = "data/templates/reflection_prompts.json"
    # image_save_base_path: str = "image4interact/"

    default_fovy_deg_min: float = 40.0
    default_fovy_deg_max: float = 60.0
    focus_ratio: float = 0.6

    # reflection_txt_load_path: Optional[str] = (
    #     "./out/load_reflection.txt"  # Path to the reflection text file
    # )
    # reflection_txt_save_path: Optional[str] = "./out/save_reflection.txt"  # Path to save

    enable_detailed_logging: bool = False
    save_interaction_history: bool = True


@dataclass
class TaskInteractionConfig:
    """Task Interaction specific configuration"""

    enable_hint_prompts: bool = True
    enable_ambiguous_item_handling: bool = True
    enable_reflection_prompts: bool = True


@dataclass
class OutcomeBaseGenerationConfig:
    """Configuration for outcome-based task generation"""

    vlm_list = [
        "openai/gpt-4.1",
        "anthropic/claude-3.5-haiku",
        "google/gemini-2.5-flash-lite-preview-06-17",
    ]
    # manitaskot_pattern_file: str = "data/templates/manitask_ot200.txt"
    # output_dir: str = "./out/outcome_based_task.txt"


# @dataclass
# class InteractPromptHelperConfig:
#     prompt_template_path: str = "data/templates/benchmark_prompts.json"


@dataclass
class RenameEngineConfig:
    model: str = "openai/gpt-4.1-mini"


@dataclass
class AppConfig:
    """Main application configuration"""

    ground_coverage: GroundCoverageConfig = field(default_factory=GroundCoverageConfig)
    task_primitive: TaskPrimitiveConfig = field(default_factory=TaskPrimitiveConfig)
    process_based_task: ProcessbasedTaskConfig = field(
        default_factory=ProcessbasedTaskConfig
    )
    scene: SceneConfig = field(default_factory=SceneConfig)
    scene_type: SceneType = field(default_factory=SceneType)
    basic_geometry: BasicGeometryConfig = field(default_factory=BasicGeometryConfig)
    image_renderer: ImageRendererConfig = field(default_factory=ImageRendererConfig)
    concave_processor: ConcaveProcessorConfig = field(
        default_factory=ConcaveProcessorConfig
    )
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    meshprocessor: MeshProcessorConfig = field(default_factory=MeshProcessorConfig)
    vlm_interactor: VlmInteractorConfig = field(default_factory=VlmInteractorConfig)
    scene_element: SceneElementConfig = field(default_factory=SceneElementConfig)
    rectangle_query: RectangleQueryConfig = field(default_factory=RectangleQueryConfig)
    raw_scene: RawSceneConfig = field(default_factory=RawSceneConfig)
    sapien: SapienConfig = field(default_factory=SapienConfig)
    benchmark_executor: BenchmarkExecutorConfig = field(
        default_factory=BenchmarkExecutorConfig
    )
    task_interaction: TaskInteractionConfig = field(
        default_factory=TaskInteractionConfig
    )
    outcome_based_task_generation: OutcomeBaseGenerationConfig = field(
        default_factory=OutcomeBaseGenerationConfig
    )
    rename_engine: RenameEngineConfig = field(default_factory=RenameEngineConfig)

    # Global configuration

    adjust_with_gravity: bool = (
        True  # Whether to adjust gravity (may affect object pose)
    )
    use_renaming_engine: bool = True  # Whether to use renaming engine
    bbox_only: bool = False  # Whether to use bounding box only
    input_json_path: Optional[str] = (
        "./data/datasets/replica_dataset/configs/scenes/apt_0.scene_instance.json"  # Scene file path
    )
    output_json_path: Optional[str] = "./replica_apt_0_parsed.json"  # Output file path
    entity_json_path: Optional[str] = (
        "./replica_apt_0_entities.json"  # Entity file path
    )
    output_dir: str = "./output/"  # Output directory for results
    mode: str = "online"  # Mode: "online" or "manual"
    model_name: str = "openai/gpt-4.1"  # Model name
    log_level: str = "INFO"

    cache_enabled: bool = True
    random_seed: Optional[int] = None

    task_num: int = 5  # Number of tasks to generate

    dataset_root_path: str = (
        "data/datasets/replica_dataset"  # Path to the SAPIEN dataset root directory
    )
    stage_path_prefix: str = (
        "data/datasets/replica_dataset"  # Path prefix for stage files
    )
    visual_path_prefix: str = (
        "data/datasets/replica_dataset/objects"  # Path prefix for visual objects
    )
    collision_path_prefix: str = (
        "data/datasets/replica_dataset/objects/convex"  # Path prefix for collision objects, don't need if the dataset has no collision objects
    )
    urdf_path_prefix: str = (
        "data/datasets/replica_dataset/urdf"  # Path prefix for URDF files, don't need if the dataset has no URDF files
    )
    object_config_path: str = (
        "data/datasets/replica_dataset/configs/objects"  # Path to the object configuration
        # "/mnt/windows_e/workplace/task_generation/replica_dataset/configs/objects"  # Path to the object configuration
    )

    benchmark_prompt_template_path: str = "data/templates/benchmark_prompts.json"
    reflection_prompt_template_path: str = "data/templates/reflection_prompts.json"
    rename_engine_prompt_template_path: str = "data/templates/renaming_engine.json"
    interact_prompt_template_path: str = "data/templates/interact_prompts.json"

    scene_graph_pkl_load_path: Optional[str] = (
        "${run_dir}/cache/scene_graph.pkl"  # Path to the input JSON scene file
    )
    scene_graph_pkl_save_path: Optional[str] = (
        "${run_dir}/cache/scene_graph.pkl"  # Path to the scene graph pickle file
    )
    process_based_task_pkl_load_path: Optional[str] = (
        "${run_dir}/cache/process_based_task.pkl"
    )
    process_based_task_pkl_save_path: Optional[str] = (
        "${run_dir}/cache/process_based_task.pkl"
    )
    process_based_task_txt_save_path: Optional[str] = (
        "${run_dir}/output/process_based_task.txt"
    )
    outcome_based_task_txt_save_path: Optional[str] = (
        "${run_dir}/output/outcome_based_task.txt"
    )
    image4rename_path: Optional[str] = (
        "${run_dir}/images/image4rename"  # Path to the image for renaming
    )
    image4interaction_path: Optional[str] = (
        "${run_dir}/images/image4interact"  # Path to the image for interaction
    )
    image4vote_path: Optional[str] = (
        "${run_dir}/images/image4vote"  # Path to the image for voting
    )
    reflection_txt_load_path: Optional[str] = (
        "${run_dir}/reflection/load_reflection.txt"  # Path to the reflection text file
    )
    reflection_txt_save_path: Optional[str] = (
        "${run_dir}/reflection/save_reflection.txt"  # Path to save
    )

    benchmark_results_save_path: Optional[str] = (
        "./benchmark_results.json"  # Path to save benchmark results
    )

    rename_dict_path: Optional[str] = (
        "${run_dir}/cache/rename_dict.json"  # Path to the renaming dictionary
    )

    image_path: Optional[str] = "${run_dir}/data/images"  # Path to save images

    benchmark_model_name: str = "openai/gpt-4.1-mini"
    generate_mistake_note: bool = True  # Whether to generate mistake notes
    use_mistake_note: int = 1  # Whether to use mistake notes (

    result_file_path: Optional[str] = (
        "${run_dir}/output/result.txt"  # Path to save the result file
    )
    manitaskot_pattern_file: str = "data/templates/manitask_ot200.txt"


"""
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_denoiser('oidn')
parser.add_argument("--adjust_with_gravity", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument("--use_renaming_engine", action="store_true", default=True,help="Use renaming engine for item classification")
    parser.add_argument("--bbox_only", action="store_true", default=False, help="Use bounding box only for item representation")
    parser.add_argument('--input_file', type=str, help="Path to the scene file")
    parser.add_argument("--mode", type=str, default="online")
    parser.add_argument("--model_name", type=str, default="human")
    parser.add_argument("--generate_reflection", type=bool, default=False)
    parser.add_argument("--use_mistake_note", type=int, default=0)
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
"""


class ConfigManager:
    """Configuration Manager"""

    _instance = None

    def __new__(
        cls, config_file_path: Optional[str] = None, run_dir: Optional[str] = None
    ):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of ConfigManager"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(
        self, config_file_path: Optional[str] = None, run_dir: Optional[str] = None
    ):
        if (
            self._initialized
            and self.config_file_path == config_file_path
            and self.run_dir == run_dir
        ):
            return
        # import ipdb; ipdb.set_trace()
        if run_dir:
            self.run_dir = run_dir
        else:
            self.run_dir = os.path.abspath(os.getcwd())

        self.path_resolver = PathResolver(self.run_dir)

        self.config = AppConfig()
        self.config_file_path = config_file_path

        self.config_file_export_dir = "run/configs_used/"
        self._initialized = True
        # import ipdb; ipdb.set_trace()
        #  glog.info("config_file_path", config_file_path)
        if config_file_path:

            self.load_config(config_file_path)

        self.save_used_config()

        self._resolve_all_paths()

    def _resolve_all_paths(self):

        import dataclasses

        for field in dataclasses.fields(self.config):
            value = getattr(self.config, field.name)

            if isinstance(value, str) and value.startswith("${run_dir}"):
                resolved_path = value.replace("${run_dir}", self.run_dir)
                setattr(self.config, field.name, resolved_path)

    def save_used_config(self):
        config_path = f"{self.run_dir}/latest_config/used_config.yaml"
        self.save_to_yaml(config_path)
        glog.info(f"Saved used configuration to {config_path}")

    def load_config(self, config_file_path: str) -> None:
        """Load configuration from file (supports both YAML and JSON)"""
        if config_file_path.endswith(".yaml") or config_file_path.endswith(".yml"):
            self.load_from_yaml(config_file_path)
        elif config_file_path.endswith(".json"):
            self.load_from_json(config_file_path)
        else:
            print(f"Unsupported configuration file format: {config_file_path}")

    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            print(
                f"Configuration file {config_path} does not exist, using default configuration"
            )
            return
        glog.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            self._update_config_from_dict(config_dict)
            self.config_file_path = config_path
            print(f"Loaded configuration file: {config_path}")
        except Exception as e:
            print(f"Failed to load configuration file: {e}")

    def load_from_json(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        if not config_path.exists():
            print(
                f"Configuration file {config_path} does not exist, using default configuration"
            )
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            self._update_config_from_dict(config_dict)
            self.config_file_path = config_path
            print(f"Loaded configuration file: {config_path}")
        except Exception as e:
            print(f"Failed to load configuration file: {e}")

    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""

        # Update ground coverage configuration
        if "ground_coverage" in config_dict:
            gc_config = config_dict["ground_coverage"]
            for key, value in gc_config.items():
                if hasattr(self.config.ground_coverage, key):
                    setattr(self.config.ground_coverage, key, value)

        # Update task primitive configuration
        if "task_primitive" in config_dict:
            tp_config = config_dict["task_primitive"]
            for key, value in tp_config.items():
                if hasattr(self.config.task_primitive, key):
                    setattr(self.config.task_primitive, key, value)

        # Update process-based task configuration
        if "process_based_task" in config_dict:
            at_config = config_dict["process_based_task"]
            for key, value in at_config.items():
                if hasattr(self.config.process_based_task, key):
                    setattr(self.config.process_based_task, key, value)

        # Update scene configuration
        if "scene" in config_dict:
            scene_config = config_dict["scene"]
            for key, value in scene_config.items():
                if hasattr(self.config.scene, key):
                    setattr(self.config.scene, key, value)

        # Update scene type configuration
        if "scene_type" in config_dict:
            st_config = config_dict["scene_type"]
            for key, value in st_config.items():
                if hasattr(self.config.scene_type, key):
                    setattr(self.config.scene_type, key, value)

        # Update basic geometry configuration
        if "basic_geometry" in config_dict:
            bg_config = config_dict["basic_geometry"]
            for key, value in bg_config.items():
                if hasattr(self.config.basic_geometry, key):
                    setattr(self.config.basic_geometry, key, value)

        # Update image renderer configuration
        if "image_renderer" in config_dict:
            ir_config = config_dict["image_renderer"]
            for key, value in ir_config.items():
                if hasattr(self.config.image_renderer, key):
                    setattr(self.config.image_renderer, key, value)

        # Update concave processor configuration
        if "concave_processor" in config_dict:
            cp_config = config_dict["concave_processor"]
            for key, value in cp_config.items():
                if hasattr(self.config.concave_processor, key):
                    setattr(self.config.concave_processor, key, value)

        # Update OpenRouter configuration
        if "openrouter" in config_dict:
            or_config = config_dict["openrouter"]
            for key, value in or_config.items():
                if hasattr(self.config.openrouter, key):
                    setattr(self.config.openrouter, key, value)

        # Update mesh processor configuration
        if "meshprocessor" in config_dict:
            mp_config = config_dict["meshprocessor"]
            for key, value in mp_config.items():
                if hasattr(self.config.meshprocessor, key):
                    setattr(self.config.meshprocessor, key, value)

        # Update VLM interactor configuration
        if "vlm_interactor" in config_dict:
            vlm_config = config_dict["vlm_interactor"]
            for key, value in vlm_config.items():
                if hasattr(self.config.vlm_interactor, key):
                    setattr(self.config.vlm_interactor, key, value)

        # Update scene element configuration
        if "scene_element" in config_dict:
            se_config = config_dict["scene_element"]
            for key, value in se_config.items():
                if hasattr(self.config.scene_element, key):
                    setattr(self.config.scene_element, key, value)

        # Update rectangle query configuration
        if "rectangle_query" in config_dict:
            rq_config = config_dict["rectangle_query"]
            for key, value in rq_config.items():
                if hasattr(self.config.rectangle_query, key):
                    setattr(self.config.rectangle_query, key, value)

        # Update raw scene configuration
        if "raw_scene" in config_dict:
            rs_config = config_dict["raw_scene"]
            for key, value in rs_config.items():
                if hasattr(self.config.raw_scene, key):
                    setattr(self.config.raw_scene, key, value)

        # Update sapien configuration
        if "sapien" in config_dict:
            sap_config = config_dict["sapien"]
            for key, value in sap_config.items():
                if hasattr(self.config.sapien, key):
                    setattr(self.config.sapien, key, value)

        # Update benchmark executor configuration
        if "benchmark_executor" in config_dict:
            be_config = config_dict["benchmark_executor"]
            for key, value in be_config.items():
                if hasattr(self.config.benchmark_executor, key):
                    setattr(self.config.benchmark_executor, key, value)

        # Update task interaction configuration
        if "task_interaction" in config_dict:
            ti_config = config_dict["task_interaction"]
            for key, value in ti_config.items():
                if hasattr(self.config.task_interaction, key):
                    setattr(self.config.task_interaction, key, value)

        # Update outcome-based task generation configuration
        if "outcome_based_task_generation" in config_dict:
            obtg_config = config_dict["outcome_based_task_generation"]
            for key, value in obtg_config.items():
                if hasattr(self.config.outcome_based_task_generation, key):
                    setattr(self.config.outcome_based_task_generation, key, value)

        # Update rename engine configuration
        if "rename_engine" in config_dict:
            re_config = config_dict["rename_engine"]
            for key, value in re_config.items():
                if hasattr(self.config.rename_engine, key):
                    setattr(self.config.rename_engine, key, value)

        # Update global configuration
        for key in [
            "input_json_path",
            "output_json_path",
            "entity_json_path",
            "adjust_with_gravity",
            "use_renaming_engine",
            "bbox_only",
            "mode",
            "model_name",
            "log_level",
            "output_dir",
            "cache_enabled",
            "random_seed",
            "generate_mistake_note",
            "use_mistake_note",
            "result_file_path",
            "reflection_txt_load_path",
            "reflection_txt_save_path",
            "task_num",
            "scene_graph_pkl_load_path",
            "scene_graph_pkl_save_path",
            "process_based_task_pkl_load_path",
            "process_based_task_pkl_save_path",
            "image4rename_path",
            "image4interaction_path",
            "image4vote_path",
            "benchmark_prompt_template_path",
            "reflection_prompt_template_path",
            "rename_engine_prompt_template_path",
            "interact_prompt_template_path",
            "dataset_root_path",
            "stage_path_prefix",
            "visual_path_prefix",
            "collision_path_prefix",
            "urdf_path_prefix",
            "object_config_path",
            "rename_dict_path",
            "benchmark_model_name",
        ]:
            if key in config_dict:
                setattr(self.config, key, config_dict[key])

        self._resolve_all_paths()

    def update_from_args(self, args: argparse.Namespace) -> None:
        """Update configuration from command line arguments"""
        # Update ground coverage configuration
        if hasattr(args, "resolution") and args.resolution is not None:
            self.config.ground_coverage.resolution = args.resolution
        if hasattr(args, "min_rect_size") and args.min_rect_size is not None:
            self.config.ground_coverage.min_rect_size = args.min_rect_size

        # Update global configuration
        if hasattr(args, "log_level") and args.log_level is not None:
            self.config.log_level = args.log_level
        if hasattr(args, "output_dir") and args.output_dir is not None:
            self.config.output_dir = args.output_dir
        if hasattr(args, "debug") and args.debug is not None:
            self.config.process_based_task.debug_mode = args.debug
        if hasattr(args, "random_seed") and args.random_seed is not None:
            self.config.random_seed = args.random_seed

        # Update additional global configuration from args
        if (
            hasattr(args, "adjust_with_gravity")
            and args.adjust_with_gravity is not None
        ):
            self.config.adjust_with_gravity = args.adjust_with_gravity
        if (
            hasattr(args, "use_renaming_engine")
            and args.use_renaming_engine is not None
        ):
            self.config.use_renaming_engine = args.use_renaming_engine
        if hasattr(args, "bbox_only") and args.bbox_only is not None:
            self.config.bbox_only = args.bbox_only
        if hasattr(args, "input_file") and args.input_file is not None:
            self.config.input_file = args.input_file
        if hasattr(args, "mode") and args.mode is not None:
            self.config.mode = args.mode
        if hasattr(args, "model_name") and args.model_name is not None:
            self.config.model_name = args.model_name

    # def save_to_yaml(self, config_path: str) -> None:
    #     """Save current configuration to YAML file"""
    #     config_path = Path(config_path)
    #     try:
    #         with open(config_path, "w", encoding="utf-8") as f:
    #             yaml.dump(
    #                 self.config.__dict__,
    #                 f,
    #                 default_flow_style=False,
    #                 allow_unicode=True,
    #             )
    #         print(f"Configuration saved to: {config_path}")
    #     except Exception as e:
    #         print(f"Failed to save configuration: {e}")
    def save_to_yaml(self, config_path: str) -> None:
        """Save current configuration to YAML file"""
        config_path = Path(config_path)
        try:
            # Convert the dataclass to a dictionary
            config_dict = self._config_to_dict(self.config)
            # Ensure the directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                )
            print(f"Configuration saved to: {config_path}")
        except Exception as e:
            print(f"Failed to save configuration: {e}")

    def _convert_value_to_serializable(self, value: Any) -> Any:
        """Recursively converts values to be YAML-serializable, handling numpy types."""
        # Check if the value is a Numpy integer type
        if isinstance(value, np.integer):
            return int(value)
        # Check if the value is a Numpy floating type
        if isinstance(value, np.floating):
            return float(value)
        # Check if the value is a Numpy array
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, list):
            return [self._convert_value_to_serializable(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._convert_value_to_serializable(item) for item in value)
        if isinstance(value, dict):
            return {k: self._convert_value_to_serializable(v) for k, v in value.items()}

        # Check if the value is another dataclass instance, recursively convert
        if hasattr(value, "__dict__") and not isinstance(value, type):
            return self._config_to_dict(value)

        return value

    def _config_to_dict(self, config_obj: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary recursively."""
        if not hasattr(config_obj, "__dict__"):
            return config_obj

        result = {}
        for key, value in config_obj.__dict__.items():
            result[key] = self._convert_value_to_serializable(value)
        return result

    def print_config(self) -> None:
        """Print current configuration"""
        print("=" * 50)
        print("Current Configuration:")
        print(f"Ground Coverage Analyzer:")
        print(f"  Resolution: {self.config.ground_coverage.resolution}")
        print(f"  Min rectangle size: {self.config.ground_coverage.min_rect_size}")
        print(f"  Z range: {self.config.ground_coverage.z_range}")
        print(f"Task Primitive:")
        print(f"  Default action: {self.config.task_primitive.default_action}")
        print(
            f"  Support directions: {len(self.config.task_primitive.support_directions)} directions"
        )
        print(f"Atomic Task:")
        print(
            f"  Max intermediate states: {self.config.process_based_task.max_intermediate_states}"
        )
        print(f"  Debug mode: {self.config.process_based_task.debug_mode}")
        print(f"Scene Type:")
        print(
            f"  Need collision adjustment: {self.config.scene_type.NEED_COLLISION_ADJUSTMENT}"
        )
        print(f"  RGBD scene: {self.config.scene_type.RGBD_SCENE}")
        print(f"Basic Geometry:")
        print(f"  EPS: {self.config.basic_geometry.EPS}")
        print(f"Image Renderer:")
        print(f"  Default FOV: {self.config.image_renderer.default_fovy}")
        print(
            f"  Resolution: {self.config.image_renderer.width}x{self.config.image_renderer.height}"
        )
        print(f"  Z range: {self.config.image_renderer.z_range}")
        print(f"Concave Processor:")
        print(f"  Min polygon area: {self.config.concave_processor.min_polygon_area}")
        print(
            f"  Target aspect ratio: {self.config.concave_processor.target_aspect_ratio}"
        )
        print(f"  Max target strips: {self.config.concave_processor.max_target_strips}")
        print(f"OpenRouter:")
        print(f"  Model: {self.config.openrouter.model}")
        print(f"Global:")
        print(f"  Adjust with gravity: {self.config.adjust_with_gravity}")
        print(f"  Use renaming engine: {self.config.use_renaming_engine}")
        print(f"  BBox only: {self.config.bbox_only}")
        print(f"  Mode: {self.config.mode}")
        print(f"  Model name: {self.config.model_name}")
        print(f"  Log level: {self.config.log_level}")
        print(f"  Output directory: {self.config.output_dir}")
        print(f"  Random seed: {self.config.random_seed}")
        print("=" * 50)

    def save_to_json(self, config_path: str) -> None:
        """Save current configuration to JSON file"""
        config_path = Path(config_path)
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config.__dict__, f, indent=4, ensure_ascii=False)
            print(f"Configuration saved to: {config_path}")
        except Exception as e:
            print(f"Failed to save configuration: {e}")

    def temp_set_model(self, model_name: str):

        model_config_dict = {"openrouter": {"model": model_name}}
        self._update_config_from_dict(model_config_dict)
        glog.info(f"Temporarily set model to {model_name}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get global configuration"""
    return config_manager.config


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance"""
    return config_manager


def get_ground_coverage_config() -> GroundCoverageConfig:
    """Get ground coverage configuration"""
    return config_manager.config.ground_coverage


def get_image_renderer_config() -> ImageRendererConfig:
    """Get image renderer configuration"""
    return config_manager.config.image_renderer


def get_concave_processor_config() -> ConcaveProcessorConfig:
    """Get concave processor configuration"""
    return config_manager.config.concave_processor


def get_process_based_task_config() -> ProcessbasedTaskConfig:
    """Get atomic task configuration"""
    return config_manager.config.process_based_task


def get_scene_config() -> SceneConfig:
    """Get scene configuration"""
    return config_manager.config.scene


def get_scene_type_config() -> SceneType:
    """Get scene type configuration"""
    return config_manager.config.scene_type


def get_basic_geometry_config() -> BasicGeometryConfig:
    """Get basic geometry configuration"""
    return config_manager.config.basic_geometry


def get_openrouter_config() -> OpenRouterConfig:
    """Get OpenRouter configuration"""
    return config_manager.config.openrouter


def get_mesh_processor_config() -> MeshProcessorConfig:
    """Get MeshProcessor configuration"""
    return config_manager.config.meshprocessor


def get_vlm_interactor_config() -> VlmInteractorConfig:
    """Get VLM Interactor configuration"""
    return config_manager.config.vlm_interactor


def get_scene_element_config() -> SceneElementConfig:
    """Get Scene Element configuration"""
    return config_manager.config.scene_element


def get_rectangle_query_config() -> RectangleQueryConfig:
    """Get Rectangle Query configuration"""
    return config_manager.config.rectangle_query


def get_raw_scene_config() -> RawSceneConfig:
    """Get raw scene configuration"""
    return config_manager.config.raw_scene


def get_sapien_config() -> SapienConfig:
    """Get SAPIEN configuration"""
    return config_manager.config.sapien


def get_benchmark_executor_config() -> BenchmarkExecutorConfig:
    """Get Benchmark Executor configuration"""
    return config_manager.config.benchmark_executor


def get_task_interaction_config() -> TaskInteractionConfig:
    """Get Task Interaction configuration"""
    return config_manager.config.task_interaction


def get_outcome_based_task_generation_config() -> OutcomeBaseGenerationConfig:
    """Get Outcome Base Generation configuration"""
    return OutcomeBaseGenerationConfig()


def get_rename_engine_config() -> RenameEngineConfig:
    """Get Rename Engine configuration"""
    return config_manager.config.rename_engine


def init_config(config_file_path: Optional[str] = None) -> None:
    """Initialize global configuration manager with config file"""
    global config_manager
    config_manager = ConfigManager(config_file_path)
