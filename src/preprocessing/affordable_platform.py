"""
scene_graph_tree_new/affordable_platform.py

This module defines the AffordablePlatform class, which represents a platform in a scene graph.
It includes methods for maintaining an affordable platform, such as calculating the platform's height, area, and intersection with rectangles.

"""

import trimesh
from src.geometry.polygon_processor import PolygonProcessor
from src.geometry.convex_hull_processor import ConvexHullProcessor_2d
from shapely.geometry import Polygon, box
import numpy as np


class AffordablePlatform:
    def __init__(self, vertices=[], faces=[], available_height=1e9, name=""):
        # vertices and faces
        self.vertices = vertices
        self.faces = faces
        self.bbox = np.min(vertices, axis=0)[:2], np.max(vertices, axis=0)[:2]
        # available height of the platform's free space
        self.available_height = 1e9
        # derivative of the platform (not sure if it is useful)
        self.derivative = []
        # name of the platform. object_name + platform_index
        self.name = name
        # is top platform or not
        self.is_top_platform = False
        pass

    # get the top and bottom height of the platform
    def get_height(self):
        return self.vertices[:, 2].max(), self.vertices[:, 2].min()

    def get_convex_hull_2d(self):
        return ConvexHullProcessor_2d(self.vertices, (1, 0))

    # get the platform as shapely.geometry.Polygon
    def get_polygon(self):
        return PolygonProcessor.vertices_and_face_to_polygon(self.vertices, self.faces)

    def get_area(self):
        polygons = self.get_polygon()
        return sum([polygon.area for polygon in polygons])

    def intersect_rectangle_area(self, rectangle):
        # Get the platform polygon
        platform_polygon = self.get_polygon()

        # Create the rectangle polygon
        minx, miny = rectangle[0]
        maxx, maxy = rectangle[1]
        rectangle_polygon = box(minx, miny, maxx, maxy)

        # Check for intersection and return the intersection area
        if platform_polygon.intersects(rectangle_polygon):
            intersection = platform_polygon.intersection(rectangle_polygon)
            return intersection.area
        else:
            return 0
