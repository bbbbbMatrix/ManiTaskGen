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

The goal is to decompose desktop surfaces into appropriately sized rectangles with aspect ratios close to the target_aspect_ratio.
This is achieved through the following pipeline:

### 1. Concavity Detection
- Compares actual polygon area with convex hull area. This method is used to ignore small holes (e.g., water sinks) in the surface.

### 2. Strip Generation
- Extract the bounding box of the entire plane and perform strip cutting
- Based on polygon aspect ratio, select horizontal, vertical, or mixed cutting strategies

### 3. Intelligent Merging  
- After initial cutting, merge adjacent shapes that together approximate rectangular forms
- Implemented through `_merge_adjacent_rectangles()` method

### 4. Vertex Preservation
- For each divided rectangular region, preserve only vertices that originate from the original mesh
- Maintains geometric fidelity with the source data
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

## Usage Examples

ConcaveProcessor.decompose_concave_polygon(vertices_3d, faces)

where vertices_3d and faces are attributes similar to faces and vertices in trimesh

Called in the main program at: src.geometry.object_mesh_processor L550

separate_face_division = ConcaveProcessor.decompose_concave_polygon(
                vertices_3d=separate_face.vertices,
                faces=separate_face.faces,
            )