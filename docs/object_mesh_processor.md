# Basic Geometries Module

本文档介绍object_mesh_processor部分, 代码中的基础几何计算模块

## Introduction

这个模块基于scipy.spatial.ConvexHull维护带朝向(heading)的二维凸包类ConvexHullProcessor_2d，来作为表面物体或平台的轮廓。在代码中，将表面物体和平台都视作二维平面内的凸包，为了准确并方便地判断它们的位置关系。

## Main Features.
- 凸包/凸包间操作
  - 获取凸包的点
  - 凸包面积
  - 求凸包的bounding box
  - 求凸包交/面积
- 凸包-几何操作
  - 判断矩形与凸包是否有交
  - 判断矩形是否能fit in 凸包，在可以fit in时求translation.
  - 求凸包到直线的最近/最远点
  - 以凸包为障碍物更新一块receptacle.[配图]

