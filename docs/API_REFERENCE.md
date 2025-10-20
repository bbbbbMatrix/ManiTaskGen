# API Reference

This document provides comprehensive documentation for reusable code components, utility libraries, and internal data structures used in ManiTaskGen.

## Overview

* Geometry Processing Libraries
* VLM usage

## Geometry Processing Libraries

### Basic Geometries (`basic_geometries`)

``src/geometry/basic_geometries.py`` provides fundamental geometric computation functionality in the codebase and is applied throughout various modules.

For detailed implementation, see [basic_geometry.md](basic_geometry.md)

### Concave Processor (`concave_processor`)

The Concave Processor is responsible for handling and processing concave shapes when encountered concave affordable platforms (e.g. L-shaped kitchen counters) in 3D scenes.

For detailed implementation, see [concave_processor.md](concave_processor.md)

### Convex Hull Processor (`convex_hull_processor`)

In the codebase, the bounding volumes of platforms and objects are simplified into 2D convex hulls in the XOY plane and corresponding cylindrical shapes with these hulls as bases. The Convex Hull Processor maintains the necessary computations for this process.


For detailed implementation, see [convex_hull_processor.md](convex_hull_processor.md)

### Object Mesh Processor (`object_mesh_processor`)

The Object Mesh Processor is responsible for handling and processing 3D object meshes when building scene graphs and generating tasks. 

For detailed implementation, see [object_mesh_processor.md](object_mesh_processor.md)

## VLM Usage

### VLM Agent Interface (`vlm_agent_interface`)

The VLM Agent Interface provides the mechanism for interacting with Vision Language Models (VLMs) in the codebase. Currently, the repository calls VLMs by hardcoding the OpenRouter API interface within the VLMEvalKit code.

For detailed implementation, see [calling_vlm.md](calling_vlm.md)