# Gen Scene Graph Module Documentation

## Overview

This module's main function is to construct a **hierarchical scene graph with platform information** from scene configuration files (JSON format). It organizes 3D scene object relationships, spatial layouts, and available space information into a tree structure, providing foundational data structures for robot task planning and spatial reasoning.

## Core Features

### Main Characteristics
- **Hierarchical Scene Representation**: Organizes objects in the scene into tree structures based on spatial containment relationships
- **Platform Detection and Management**: Automatically identifies available platform areas on object surfaces
- **Free Space (Receptacle) Calculation**: Calculates available space in 8 directions around each object
- **Contact Relationship Analysis**: Detects support and contact relationships between objects
- **Visual Rendering Support**: Provides scene visualization and image generation functionality

### Data Structure Design

#### 1. TreeNode (Tree Node)
- **Function**: Represents a single object in the scene, maintains its position information in the scene graph, receptacle information, and automatically focuses when obtaining its images based on this information.
- **Key Attributes**:
  - `name`: Object name
  - `entity_config`: Object configuration, used to read the original object config and load it into SAPIEN
  - `parent/children`: Parent-child node relationships
  - `convex_hull_2d`: 2D convex hull representation. Object defined in src/geometry/convex_hull_processor
  - `free_space`: 8-direction free space information. Each direction's receptacle is maintained as a list of four vertices forming a rectangle, with vertex order: rear-left, front-left, front-right, rear-right
  - `on_platform`: Platform this object belongs to
  - `own_platform`: List of platforms contained by this object

#### 2. TreePlatform (Platform)
- **Function**: Represents available platform areas on object surfaces. Maintains its position information in the scene graph and automatically focuses when obtaining its images based on this information.
- **Key Attributes**:
  - `bel_object`: Object that owns this platform
  - `convex_hull_2d`: Platform's 2D geometric shape
  - `avl_height`: Platform's allowed height for placing objects
  - `visible_directions`: Accessible directions. Whether the agent can see most of the platform from this direction (determined by TreePlatform.freespace_is_visible, with specific calculations in ())
  - `standing_point_list`: Agent-fittable positions in four directions. If not obstructed by other objects, these points are uniformly distributed at regular intervals on a line at an appropriate distance from the platform. Specific calculations in ()
  - `children`: List of objects on the platform

#### 3. Tree (Scene Graph)
- **Function**: Manages the hierarchical structure of the entire scene. Its attributes consist of TreeNode and TreePlatform objects.
- **Key Methods**:
  - `from_scene_platform_list()`: Construct scene graph from platform list
  - `calculate_free_space()`: Calculate free space
  - `cal_standable_area_for_platforms()`: Calculate standable areas for platforms

[calculate_free_space, cal_standable_area_for_platforms illustrated with diagrams]

## Scene Graph Construction Process

### 1. Scene Loading

Separately load object GLB files and stage information. For objects, after obtaining position information, first parse their meshes and calculate their affordable platforms. The algorithm logic is detailed in (). For stages, we note that meshes forming scene walls often lack proper volume, making it inconvenient to calculate positions where agents can fit in the scene. Therefore, we uniformly use convex hulls for stage meshes.

### 2. Calculate Object Contact Relationships
Calculate contact relationships between all objects and platforms based on their positional relationships. The logic is detailed in scene_parser.SceneElement.calculate_contact_conditions(scene_platform_list).

### 3. Build Tree Structure
Sequentially call three key methods of the Tree class: first construct the scene graph from object and platform contact relationships, then calculate receptacles defined by each surface object as anchor object, and calculate standable areas for platforms.

## Configuration Parameters

### Key Settings
```python
# Accessible through get_concave_processor_config()
target_aspect_ratio: float = 2.0        # Target width/height ratio
min_polygon_area: float = 0.01          # Minimum area for valid polygons
merge_tolerance: float = 0.1            # Tolerance for rectangle merging
max_target_strips: int = 8              # Maximum decomposition strips
concave_threshold: float = 0.1          # Threshold for concavity detection
eps: float = 1e-6                       # Precision tolerance

```

## Usage Example
```python
json_tree_path = gen_scene_graph.load_json_file(accurate_entity_path)
scene_graph_tree = gen_scene_graph.gen_multi_layer_graph_with_free_space(
        json_tree_path
)

```