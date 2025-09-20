# Concave Processor Module

## Introduction

The Concave Processor module provides functionality for decomposing concave polygons derived from 3D OBJ files into multiple rectangular regions. The primary goal is to break down complex concave shapes (such as L-shaped or U-shaped platforms) into simpler convex polygons that approximate rectangles with a specified aspect ratio.

## Main Features

### Core Functionality
- **Concave Polygon Detection**: Automatically identifies whether a polygon is concave and requires decomposition
- **Rectangular Decomposition**: Breaks down concave polygons into rectangular strips with target aspect ratios
- **Adaptive Merging**: Intelligently merges adjacent rectangles to reduce the number of resulting polygons
- **Original Vertex Preservation**: Maintains geometric fidelity by preserving vertices close to the original polygon faces

### Decomposition Strategies
- **Rectangular with Merge** (Recommended): Combines rectangular decomposition with intelligent merging
- **Basic Rectangular**: Standard rectangular strip decomposition
- **Grid-based**: Decomposes using a regular grid pattern
- **Mixed Strategy**: Automatically chooses between vertical, horizontal, or grid decomposition

## Architecture Design

### Main Classes
- `ConcaveProcessor`: Primary class for polygon processing and decomposition
  - Handles 3D vertices and face data from OBJ files
  - Configurable parameters for decomposition quality and performance

### Key Methods
- `decompose()`: Main decomposition method with selectable strategies
<!-- 
## Usage Examples

### Basic Usage
```python
from src.geometry.concave_processor import ConcaveProcessor
import numpy as np

# Define vertices from OBJ file (L-shaped platform)
vertices = np.array([
    [0, 0, 0],    # Bottom-left corner
    [3, 0, 0],    # Bottom-right of horizontal part
    [3, 1, 0],    # Top-right of horizontal part
    [1, 1, 0],    # Inner corner (concave point)
    [1, 3, 0],    # Top-right of vertical part
    [0, 3, 0],    # Top-left corner
])

# Define faces (triangulation)
faces = [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]]

# Create processor and decompose
processor = ConcaveProcessor(vertices, faces)
results = processor.decompose(strategy="rectangular_with_merge")

print(f"Decomposed into {len(results)} rectangular regions")
```

### Static Method Usage
```python
# Quick decomposition using static method
results = ConcaveProcessor.decompose_concave_polygon(vertices, faces)
```

### Custom Configuration
```python
# The processor uses configuration from config_manager
# Key parameters include:
# - target_aspect_ratio: Desired width/height ratio for rectangles
# - min_polygon_area: Minimum area threshold for filtering small polygons
# - merge_tolerance: Tolerance for merging adjacent rectangles
# - max_target_strips: Maximum number of strips for decomposition
``` -->

## Algorithm Overview

我们希望把桌面分割成大小适当，且长宽比接近于target_aspect_ratio的矩形。
为了达到这一点， 
会经过以下步骤：

### 1. Concavity Detection
- Compares actual polygon area with convex hull area. The reason for using this method is to ignore small holes (e.g. water sink) in the whole.
我们希望把桌面分割成大小适当，且长宽比接近于target_aspect_ratio的矩形。
为了达到这一点， 
会经过以下步骤：

(0)- Compares actual polygon area with convex hull area. The reason for using this method is to ignore small holes (e.g. water sink) in the whole.
（1）取整个平面的bounding box, 进行(strip generation）。根据 Polygon aspect ratio, 选择横着、纵着或者混合切分
（2）切分之后，尽可能的合并拼在一起接近矩形的图形 通过 ``(_merge_adjacent_rectangles)``
（3）作Vertex Preservation, 对于每个分割出的矩形区域只保留来自mesh的vertices. 
### 2. Decomposition Strategy Selection

Based on polygon aspect ratio:
- **Wide polygons**: Vertical strip decomposition (horizontal cutting)
- **Tall polygons**: Horizontal strip decomposition (vertical cutting)
- **Square-like polygons**: Mixed grid-based decomposition

### 3. Strip Generation
- Calculates optimal strip width/height based on target aspect ratio
- Creates rectangular strips that intersect with the original polygon
- Generates valid Shapely polygons from intersections

### 4. Intelligent Merging
- Builds adjacency graph of decomposed rectangles
- Identifies merge groups using breadth-first search
- Merges adjacent rectangles that maintain rectangular properties

### 5. Vertex Preservation
- Filters decomposed polygons to preserve original vertices
- Maintains geometric accuracy by keeping vertices close to original faces
- Uses convex hull computation for final polygon generation

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

ConcaveProcessor.decompose_concave_polygon(vertices_3d, faces)

其中vertices_3d, faces为类似trimesh里faces, vertices 

在主程序中的调用位置： src.geometry.object_mesh_processor L550

separate_face_division = ConcaveProcessor.decompose_concave_polygon(
                vertices_3d=separate_face.vertices,
                faces=separate_face.faces,
            )