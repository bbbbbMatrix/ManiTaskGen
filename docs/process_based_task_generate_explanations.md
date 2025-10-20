# Process-Based Task Generation Module Documentation

This document explains all the possible patterns in the ManiTaskOT-200, and how they are used for outcome-based task generation.

## Overview

The main function of this module is **process-based robotic manipulation task generation**. It can automatically generate various types of object manipulation tasks from scene graphs, including object movement, placement, and rearrangement. This module provides a structured task generation framework for robot task planning and multimodal AI agents, supporting the generation of task chains of arbitrary length.

## Core Functions

### Main Features
- **Multi-type task generation**: Supports 5 different types of manipulation tasks
- **Task chain construction**: Supports generating continuous multi-step task sequences
- **Intelligent spatial reasoning**: Automatically generates reasonable tasks based on spatial relationships between objects
- **Image generation and description**: Automatically generates visualization images of task execution processes

### Supported Task Types

#### 1. Basic Platform Placement Tasks
- **MOVE_TO_EMPTY_PLATFORM**: Move objects to empty platforms
- **MOVE_TO_EMPTY_PLATFORM_9_GRID**: Place objects in specific areas of platforms (nine-grid layout)

#### 2. Relative Position Tasks
- **MOVE_AROUND_OBJECT**: Place objects around certain objects
- **MOVE_TO_OBJECT_FREESPACE_9_GRID**: Place objects in specific directions relative to designated objects

#### 3. Complex Spatial Relationship Tasks
- **MOVE_TO_MIDDLE_OF_OBJECTS**: Place objects between two objects

## Data Structure Design

### Core Class Structure

#### 1. Task (Subtask)
- **Function**: Represents a single subtask.
   
- **Key Attributes**:
  - `item`: The item node to be moved.
  - `destination`: The target platform where the object needs to be moved.
  - `type`: One of the five basic task types.
  - `feature`: Task feature parameters. The parameter formats for different tasks are as follows:
  - `MOVE_TO_EMPTY_PLATFORM`: `{}`
  - `MOVE_TO_EMPTY_PLATFORM_9_GRID`: `{dir: str}`
  - `MOVE_AROUND_OBJECT`: `{object: node}`
  - `MOVE_TO_OBJECT_FREESPACE_9_GRID`: `{object: node, dir: str}`
  - `MOVE_TO_MIDDLE_OF_OBJECTS`: `{object1: node, object2: node}`
  - `goal_translation`: A possible translation where the item can be moved to complete the task.

#### 2. TaskChain (Task Chain)
- **Function**: Represents a single subtask.

- **Key Attributes**:
  - `subtask_list`: A list composed of subtasks.

#### 3. TaskGeneration (Task Generator)

- **Function**: Generates process-based task chains.

- **Key Attributes**:
  - `scene_graph`: The scene graph for which tasks are to be generated.
  - `tasks`: The list of generated tasks

## Task Generation Process

The core idea of task generation is to repeatedly randomly select one task from the currently available tasks and simulate its completion state in the scene graph until a complete task is obtained.

When selecting tasks, the program first selects items and platforms and calculates all fitable positions for items on platforms. If available, it selects from corresponding task types based on whether the platform is empty, and if the platform is not empty, it randomly selects target anchor objects and directions. All "random selections" generate a random permutation of all objects to be selected, then enumerate them.

## Key Parameter Configuration

### Task Generation Parameters
- `task_length`: Task chain length (default: 5)
- `max_task_num`: Maximum number of tasks (default: 10)
- `chain_num`: Number of subtasks in a single task chain


