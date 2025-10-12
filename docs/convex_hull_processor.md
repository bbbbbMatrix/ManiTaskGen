# Convex Hull Processor Module

This document introduces the convex_hull_processor component, the basic geometric computation module in the codebase.

## Introduction

This module maintains a 2D convex hull class `ConvexHullProcessor_2d` with heading information based on `scipy.spatial.ConvexHull`, serving as the contour representation for surface objects or platforms. In the code, both surface objects and platforms are treated as convex hulls in 2D planes to accurately and conveniently determine their spatial relationships.

## Main Features

- **Convex Hull to Convex Hull Operations**
  - Get convex hull vertices
  - Calculate convex hull area
  - Compute convex hull bounding box
  - Calculate convex hull intersection and intersection area

- **Convex Hull to Geometry Operations**
  - Check if rectangle intersects with convex hull
  - Determine if rectangle can fit in convex hull, and compute translation when fitting is possible
  - Find closest/farthest points from convex hull to line
  - Update a receptacle area using convex hull as obstacle