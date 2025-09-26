# Basic Geometries Module

This document introduces the basic_geometry module, which provides fundamental geometric computation functionality in the codebase.

## Introduction

This module provides basic functionality for 2D and 3D geometric calculations, including operations on geometric objects such as points, lines, rectangles, and triangles. All geometric computations in the program are handled here.

It is primarily used for geometric collision detection, spatial analysis, and object placement tasks.

## Main Features

- 2D Geometric Operations
  - Point Operations
    - Point rotation
    - Distance from point to line segment/line
    - Perpendicular foot from point to line
    - Whether point is inside rectangle
    - Whether point is inside polygon
    - Whether point is between parallel lines
    - Whether point is on line segment
  - Line and Line Segment Operations
    - Line-line intersection
    - Whether line segments intersect
    - Finding rectangle with line segment as diagonal
  - Rectangle Operations
    - Parallel rectangle intersection
    - Parallel rectangle intersection area
    - Whether rectangle can fit inside another rectangle
    - Rectangle rotation
    - Calculate cosine of vector angle
- 3D Geometric Operations: Point-to-plane distance, ray-triangle intersection, quaternion rotation, etc.
  - Point Operations
    - Distance from point to line segment/line/plane/triangle
  - Line Operations
    - Ray-triangle intersection
  - Vector Operations
    - Calculate quaternion for vector angle
    - Vector to quaternion/RPY conversion

## Architecture Design
- `Basic2DGeometry`: Collection of static methods for 2D geometric operations
- `Basic3DGeometry`: Collection of static methods for 3D geometric operations
- Configuration Management: Precision parameters obtained through `get_basic_geometry_config()`