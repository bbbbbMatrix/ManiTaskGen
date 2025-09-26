# Gen Scene Graph 模块文档

## 概述

这个模块的主要功能是从场景配置文件（JSON格式）构建一个**带有平台（platform）信息的层次化场景图（Scene Graph）**。它将3D场景中的物体关系、空间布局和可用空间信息组织成树形结构，为机器人任务规划和空间推理提供基础数据结构。

## 核心功能

### 主要特性
- **层次化场景表示**：将场景中的物体按照空间包含关系组织成树形结构
- **平台检测与管理**：自动识别物体表面的可用平台区域
- **自由空间(receptacle)计算**：计算每个物体周围8个方向的可用空间
- **接触关系分析**：检测物体间的支撑和接触关系
- **视觉渲染支持**：提供场景可视化和图像生成功能

### 数据结构设计

#### 1. TreeNode（树节点）
- **功能**：表示场景中的单个物体，维护它在scene graph中的位置信息，receptacle信息，并在需要获取其图片时根据这些信息自动对焦。
- **关键属性**：
  - `name`: 物体名称
  - `entity_config`: 物体config，用于读取原始的物体config并加载到sapien中。
  - `parent/children`: 父子节点关系
  - `convex_hull_2d`: 2D凸包表示.对象定义在src/geometry/convex_hull_processor中.
  - `free_space`: 8方向自由空间信息. 每个方向的receptacle用矩形的四个顶点组成的list维护，list中的顶点顺序是rear-left, front-left, front-right, rear-right.
  - `on_platform`: 所属平台
  - `own_platform`: 物体包含的平台列表。

#### 2. TreePlatform（平台）
- **功能**：表示物体表面的可用平台区域。维护他在scene graph中的位置信息，并在需要获取其图片时根据这些信息自动对焦。
- **关键属性**：
  - `bel_object`: 平台所属的物体。
  - `convex_hull_2d`: 平台的2D几何形状
  - `avl_height`: 平台的允许放物品的高度。
  - `visible_directions`: 可访问方向。Agent是否可以在这个方向上看到平台的大部分，(由TreePlatform.freespace_is_visible判断，具体的计算在（）中)，
  - `standing_point_list`: 四个方向上，Agent可以fit的位置。如果没有其他物体遮挡，这些点会有一定间隔地均匀分布在离平台适当距离的一条线上。具体的计算在（）中。
  - `children`: 平台上的物体列表

#### 3. Tree（场景图）
- **功能**：管理整个场景的层次结构。其属性由TreeNode, TreePlatform对象构成。
- **关键方法**：
  - `from_scene_platform_list()`: 从平台列表构建场景图
  - `calculate_free_space()`: 计算自由空间
  - `cal_standable_area_for_platforms()`: 计算平台可站立区域。

【calculate_free_space, cal_standable_area_for_platforms用图介绍】

##  场景图构建流程

### 1.读取场景

分开读取物体(object)的glb和场景(stage)的信息。对于物体，在获取位置信息之后先解析其mesh，计算出它的affordable platforms，其算法逻辑见（）。对于场景，我们注意到构成场景墙壁的mesh常有不成体积的情况，给计算Agent可以fit场景的位置带来不便，因此对于场景mesh统一取用它们的凸包。

### 2.计算物体接触关系
根据物体和平台的位置关系计算所有物体和平台的接触关系。其逻辑见（scene_parser.SceneElement.calculate_contact_conditions(scene_platform_list)

### 3.构建树结构
先后调用Tree类的三个关键方法，先从物体和平台的接触关系构建场景图，然后计算以每个表面物体为anchor object所定义的receptacle和计算平台的可站立区域。


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

## 使用示例

json_tree_path = gen_scene_graph.load_json_file(accurate_entity_path)
scene_graph_tree = gen_scene_graph.gen_multi_layer_graph_with_free_space(
        json_tree_path
)

在主程序中的调用位置： main.py L246




### 8方向系统
[配图]

