# Process-Based Task Generation 模块文档

This document explains all the possible patterns in the ManiTaskOT-200, and how they are used for outcome-based task generation.

## 概述

这个模块的主要功能是**基于过程的机器人操作任务生成**。它可以从场景图（Scene Graph）中自动生成多种类型的物体操作任务，包括物体移动、放置、重新排列等。该模块为机器人任务规划和多模态AI代理提供了结构化的任务生成框架，支持生成任意长度的任务链条。

## 核心功能

### 主要特性
- **多类型任务生成**：支持5种不同类型的操作任务
- **任务链构建**：支持生成连续的多步骤任务序列
- **智能空间推理**：基于物体间的空间关系自动生成合理的任务
- **图像生成与描述**：自动生成任务执行过程的可视化图像

### 支持的任务类型

#### 1. 基础平台放置任务
- **MOVE_TO_EMPTY_PLATFORM**: 将物体移动到空平台上
- **MOVE_TO_EMPTY_PLATFORM_9_GRID**: 将物体放置到平台的特定区域（九宫格）

#### 2. 相对位置任务
- **MOVE_AROUND_OBJECT**: 将物体放置在某个物体周围
- **MOVE_TO_OBJECT_FREESPACE_9_GRID**: 将物体放置到指定物体的特定方向

#### 3. 复杂空间关系任务
- **MOVE_TO_MIDDLE_OF_OBJECTS**: 将物体放置在两个物体之间

## 数据结构设计

### 核心类结构

#### 1. Task（子任务）
- **功能**：表示单个子任务。
   
- **关键属性**：
  - `item`: 待移动的物品节点。
  - `destination`: 物体需移动到的目标平台。
  - `type`: 基本五个任务类型中的一个。
  - `feature`: 任务特征参数。不同任务所对应的参数格式如下：
    - `MOVE_TO_EMPTY_PLATFORM`: `{}`
    - `MOVE_TO_EMPTY_PLATFORM_9_GRID`: `{dir: str}`
    - `MOVE_AROUND_OBJECT`: `{object: node}`
    - `MOVE_TO_OBJECT_FREESPACE_9_GRID`: `{object: node, dir: str}`
    - `MOVE_TO_MIDDLE_OF_OBJECTS`: `{object1: node, object2: node}`
  - `goal_translation`: 为完成任务，物品可以移动到的一个possible translation.

#### 2. TaskChain（任务链）
- **功能**：表示单个子任务。

- **关键属性**：
  - `subtask_list`: 由子任务组成的list. 

#### 3. TaskGeneration（任务生成器）

- **功能**：生成process-based任务链。

- **关键属性**：
  - `scene_graph`: 待生成任务的场景图。
  - `tasks`: 生成的任务列表




## 任务生成流程

任务生成的核心思路是 repeatedly 从当前可以进行的任务中随机抽取一个并在场景图中模拟其完成的状态，直到得到一个完整的任务。

而在抽取任务时，程序会先抽取物品和平台并计算所有物品在平台上的fitable位置。如果有，则根据平台是否为空从对应的任务类型中选择，且若平台非空，再随机抽取目标anchor object和方向。所有的“随机抽取”都会生成一个所有待抽取对象的随机排列，然后枚举。



## 关键参数配置


### 任务生成参数
- `task_length`: 任务链长度（默认值：1）
- `max_task_num`: 最大任务数量（默认值：10）
- `chain_num`: 单个任务链中的子任务数量

## 使用示例
