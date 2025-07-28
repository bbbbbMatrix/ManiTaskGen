from rtree import index
import numpy as np
from src.utils.config_manager import get_rectangle_query_config


class RectangleQuery2D:

    @classmethod
    def get_config(cls):
        return get_rectangle_query_config()

    def __init__(self, dimension=2):

        self.property = index.Property()
        self.property.dimension = dimension
        self.property.overwrite = True

        self.index = index.Index(properties=self.property)
        self.data = {}
        self.current_id = 0
        self.rectangle_list = []

    def add_rectangle(self, rectangle):
        import glog

        #        glog.info(f"Adding rectangle: {rectangle}")
        x1 = np.min(rectangle[:, 0])
        y1 = np.min(rectangle[:, 1])
        x2 = np.max(rectangle[:, 0])
        y2 = np.max(rectangle[:, 1])

        self.index.insert(self.current_id, (x1, y1, x2, y2))
        self.data[self.current_id] = rectangle
        self.rectangle_list.append(rectangle)
        self.current_id += 1

    def query_rectangle(self, rectangle, tol=None):

        if tol is None:
            tol = self.get_config().EPS / 2
        x1 = np.min(rectangle[:, 0])
        y1 = np.min(rectangle[:, 1])
        x2 = np.max(rectangle[:, 0])
        y2 = np.max(rectangle[:, 1])
        x1 += tol
        y1 += tol
        x2 -= tol
        y2 -= tol

        results = list(self.index.intersection((x1, y1, x2, y2)))
        return results, [self.data[i] for i in results]
