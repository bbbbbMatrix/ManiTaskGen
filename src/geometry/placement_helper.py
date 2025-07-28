"""
custom_geometry_helper_new/placement_helper.py

After the agent decide to place the object, this module provides functionality to find suitable placements for rectangles in a 2D space.

It includes methods to find placements that are not intersecting with bounding boxes of other objects(treated as fixed rectangles), and to ensure that the moving rectangle is contained within a bounding rectangle.

For place_rectangle_anywhere, it finds any placement of the moving rectangle that is contained by the bound rectangle, and not intersecting with the fixed rectangles.

For place_rectangle, it first attempts to place the rectangle in a pose that intersects with all the given rectangles, and then tries to place the rectangle in the union of given rectangles.
"""

from .basic_geometries import Basic2DGeometry
from .convex_hull_processor import ConvexHullProcessor_2d
from .polygon_processor import PolygonProcessor
from .rectangle_query_processor import RectangleQuery2D
import numpy as np
import glog

from src.utils.config_manager import get_rectangle_query_config


class PlacementHelper:
    placement_coverage_threshold = 0.95
    eps = 1e-6

    def __init__(self):

        pass

    @classmethod
    def get_config(cls):
        """Get configuration for placement helper"""
        return get_rectangle_query_config()

    @staticmethod
    def place_rectangle_anywhere(fixed_rect_list, moving_rect, bound_rect):
        """
        Find any placement of the moving rectangle that is contained by the bound rectangle, and not intersecting with the fixed rectangles

        Note that if there are valid placements, there exists one pose that one of the 4 vertices of the moving rectangle has x, y coordinates that are identical to some vertices of the fixed rectangles.
        We can speed up the search by only checking these coordinates.

        Args:
            fixed_rect_list: The list of fixed rectangles
            moving_rect: The moving rectangle
            bound_rect: The bounding rectangle

        Returns:
            The placement of the moving rectangle that is contained by the bound rectangle, and not intersecting with the fixed rectangles
        """

        normalized_moving_rect, angle = Basic2DGeometry.normalize_rectangle(moving_rect)
        normalized_bound_rect, angle = Basic2DGeometry.normalize_rectangle(bound_rect)
        normalized_fixed_rect_list = [
            Basic2DGeometry.normalize_rectangle(fixed_rect)[0]
            for fixed_rect in fixed_rect_list
        ]

        bound_x_min = np.min(normalized_bound_rect[:, 0])
        bound_x_max = np.max(normalized_bound_rect[:, 0])
        bound_y_min = np.min(normalized_bound_rect[:, 1])
        bound_y_max = np.max(normalized_bound_rect[:, 1])

        normalized_moving_rect_center = np.mean(normalized_moving_rect, axis=0)
        #        glog.info(f"normalized_moving_rect_center: {normalized_moving_rect_center}")
        normalized_moving_rect_xmin = np.min(normalized_moving_rect[:, 0])
        normalized_moving_rect_xmax = np.max(normalized_moving_rect[:, 0])
        normalized_moving_rect_ymin = np.min(normalized_moving_rect[:, 1])
        normalized_moving_rect_ymax = np.max(normalized_moving_rect[:, 1])

        x_coords = [
            bound_x_min
            + (normalized_moving_rect_xmax - normalized_moving_rect_xmin) / 2,
            bound_x_max
            - (normalized_moving_rect_xmax - normalized_moving_rect_xmin) / 2,
        ]
        y_coords = [
            bound_y_min
            + (normalized_moving_rect_ymax - normalized_moving_rect_ymin) / 2,
            bound_y_max
            - (normalized_moving_rect_ymax - normalized_moving_rect_ymin) / 2,
        ]

        for fixed_rect in normalized_fixed_rect_list:
            x_coords.extend(
                [
                    fixed_rect[i][0]
                    - (normalized_moving_rect_xmax - normalized_moving_rect_xmin) / 2
                    for i in range(4)
                ]
            )
            x_coords.extend(
                [
                    fixed_rect[i][0]
                    + (normalized_moving_rect_xmax - normalized_moving_rect_xmin) / 2
                    for i in range(4)
                ]
            )
            y_coords.extend(
                [
                    fixed_rect[i][1]
                    - (normalized_moving_rect_ymax - normalized_moving_rect_ymin) / 2
                    for i in range(4)
                ]
            )
            y_coords.extend(
                [
                    fixed_rect[i][1]
                    + (normalized_moving_rect_ymax - normalized_moving_rect_ymin) / 2
                    for i in range(4)
                ]
            )

        x_coords = sorted(list(set(x_coords)))
        y_coords = sorted(list(set(y_coords)))

        x_coords_new = []
        for x in x_coords:
            if not x_coords_new:
                x_coords_new.append(x)
            else:
                if np.abs(x - x_coords_new[-1]) > PlacementHelper.get_config().eps:
                    x_coords_new.append(x)

        y_coords_new = []
        for y in y_coords:
            if not y_coords_new:
                y_coords_new.append(y)
            else:
                if np.abs(y - y_coords_new[-1]) > PlacementHelper.get_config().eps:
                    y_coords_new.append(y)

        rectangle_query_processor = RectangleQuery2D()
        for fixed_rect in normalized_fixed_rect_list:
            rectangle_query_processor.add_rectangle(fixed_rect)

        import random

        random.shuffle(x_coords)
        random.shuffle(y_coords)
        for x in x_coords:
            for y in y_coords:
                moved_rectangle = np.array(
                    [
                        normalized_moving_rect[0]
                        + np.array([x, y])
                        - normalized_moving_rect_center,
                        normalized_moving_rect[1]
                        + np.array([x, y])
                        - normalized_moving_rect_center,
                        normalized_moving_rect[2]
                        + np.array([x, y])
                        - normalized_moving_rect_center,
                        normalized_moving_rect[3]
                        + np.array([x, y])
                        - normalized_moving_rect_center,
                    ]
                )
                is_moved_rectangle_in_bound = (
                    np.min(moved_rectangle[:, 0]) >= bound_x_min
                    and np.max(moved_rectangle[:, 0]) <= bound_x_max
                    and np.min(moved_rectangle[:, 1]) >= bound_y_min
                    and np.max(moved_rectangle[:, 1]) <= bound_y_max
                )
                if not is_moved_rectangle_in_bound:
                    continue

                results, _ = rectangle_query_processor.query_rectangle(moved_rectangle)
                if len(results) == 0:
                    moved_rectangle = Basic2DGeometry.unnormalize_rectangle(
                        moved_rectangle, angle
                    )
                    return 0, np.mean(moved_rectangle, axis=0)

        return -1, "No placement found. The free space is too small"

    @staticmethod
    def find_parallel_rectangle_placement_bounds(fixed_rect, moving_rect):
        """
        Find the bounds of the moving rectangle that are parallel to the fixed rectangle


        Args:
            fixed_rect: The fixed rectangle
            moving_rect: The moving rectangle
        Returns:
            The bounds of the moving rectangle that are parallel to the fixed rectangle.
            In other words, the area where the center of moving rectangle can be placed such that it is parallel to the fixed rectangle.
        """
        normalized_fixed_rect, normalized_angle = Basic2DGeometry.normalize_rectangle(
            fixed_rect
        )
        normalized_moving_rect, _ = Basic2DGeometry.normalize_rectangle(moving_rect)

        normalized_moving_rect_center = np.mean(normalized_moving_rect, axis=0)

        if np.abs(normalized_angle - _) > PlacementHelper.get_config().eps:
            glog.warning("The angle of the fixed and moving rectangle are not the same")
        normalized_placement_bounds = np.array(
            [
                normalized_fixed_rect[0]
                + normalized_moving_rect[0]
                - normalized_moving_rect_center,
                normalized_fixed_rect[1]
                + normalized_moving_rect[1]
                - normalized_moving_rect_center,
                normalized_fixed_rect[2]
                + normalized_moving_rect[2]
                - normalized_moving_rect_center,
                normalized_fixed_rect[3]
                + normalized_moving_rect[3]
                - normalized_moving_rect_center,
            ]
        )

        placement_bounds = Basic2DGeometry.unnormalize_rectangle(
            normalized_placement_bounds, normalized_angle
        )

        return placement_bounds

    @staticmethod
    def find_placement_bounds_intersecting_all_rectangles(fixed_rect_list, moving_rect):
        """
        Find the bounds of the moving rectangle that are intersecting with all the fixed rectangles

        Args:
            fixed_rect_list: The list of fixed rectangles
            moving_rect: The moving rectangle
        Returns:
            The bounds of the moving rectangle that are intersecting with all the fixed rectangles.
            In other words, the area where the center of moving rectangle can be placed such that it is intersecting with all the fixed rectangles.
        """
        pass

        result_rect = None
        for fixed_rect in fixed_rect_list:
            placement_bounds = PlacementHelper.find_parallel_rectangle_placement_bounds(
                fixed_rect, moving_rect
            )
            # import ipdb
            # ipdb.set_trace()
            if result_rect is None:
                result_rect = placement_bounds
            else:
                result_rect = Basic2DGeometry.intersection_of_parallel_rectangle(
                    result_rect, placement_bounds
                )
                if result_rect is None:
                    return None

        return result_rect

    @staticmethod
    def find_placement_bounds_contained_by_rectangle_union(
        fixed_rect_list, moving_rect, all_intersected=False
    ):
        """

        Args:
            fixed_rect_list (_type_): _description_
            moving_rect (_type_): _description_
        """
        normalized_moving_rect, _ = Basic2DGeometry.normalize_rectangle(moving_rect)
        normalized_fixed_rect_list = [
            Basic2DGeometry.normalize_rectangle(fixed_rect)[0]
            for fixed_rect in fixed_rect_list
        ]
        normalized_moving_rect_center = np.mean(normalized_moving_rect, axis=0)
        normalized_moving_rect_xmin = np.min(normalized_moving_rect[:, 0])
        normalized_moving_rect_xmax = np.max(normalized_moving_rect[:, 0])
        normalized_moving_rect_ymin = np.min(normalized_moving_rect[:, 1])
        normalized_moving_rect_ymax = np.max(normalized_moving_rect[:, 1])

        x_coords = []
        y_coords = []
        for fixed_rect in normalized_fixed_rect_list:
            x_coords.extend(
                [
                    fixed_rect[i][0]
                    - (normalized_moving_rect_xmax - normalized_moving_rect_xmin) / 2
                    for i in range(4)
                ]
            )
            x_coords.extend(
                [
                    fixed_rect[i][0]
                    + (normalized_moving_rect_xmax - normalized_moving_rect_xmin) / 2
                    for i in range(4)
                ]
            )
            y_coords.extend(
                [
                    fixed_rect[i][1]
                    - (normalized_moving_rect_ymax - normalized_moving_rect_ymin) / 2
                    for i in range(4)
                ]
            )
            y_coords.extend(
                [
                    fixed_rect[i][1]
                    + (normalized_moving_rect_ymax - normalized_moving_rect_ymin) / 2
                    for i in range(4)
                ]
            )

        x_coords = sorted(list(set(x_coords)))
        y_coords = sorted(list(set(y_coords)))

        x_coords_new = []
        for x in x_coords:
            if not x_coords_new:
                x_coords_new.append(x)
            else:
                if np.abs(x - x_coords_new[-1]) > PlacementHelper.get_config().eps:
                    x_coords_new.append(x)

        y_coords_new = []
        for y in y_coords:
            if not y_coords_new:
                y_coords_new.append(y)
            else:
                if np.abs(y - y_coords_new[-1]) > PlacementHelper.get_config().eps:
                    y_coords_new.append(y)

        glog.info(f"x_coords_new: {x_coords_new}")
        glog.info(f"y_coords_new: {y_coords_new}")
        glog.info(f"normalized_moving_rect: {normalized_moving_rect}")

        intersect_01_matrix = np.zeros((len(x_coords_new), len(y_coords_new)))

        for idx, x in enumerate(x_coords_new):
            for idy, y in enumerate(y_coords_new):
                moved_rectangle = np.array(
                    [
                        normalized_moving_rect[0]
                        + np.array([x, y])
                        - normalized_moving_rect_center,
                        normalized_moving_rect[1]
                        + np.array([x, y])
                        - normalized_moving_rect_center,
                        normalized_moving_rect[2]
                        + np.array([x, y])
                        - normalized_moving_rect_center,
                        normalized_moving_rect[3]
                        + np.array([x, y])
                        - normalized_moving_rect_center,
                    ]
                )

                intersection_list = [
                    Basic2DGeometry.intersection_of_parallel_rectangle(
                        moved_rectangle, normalized_fixed_rect_list[i]
                    )
                    for i in range(len(normalized_fixed_rect_list))
                ]

                intersection_area = PolygonProcessor.union_numpy_polygon_list_area(
                    intersection_list
                )

                if (
                    intersection_area
                    > (normalized_moving_rect_xmax - normalized_moving_rect_xmin)
                    * (normalized_moving_rect_ymax - normalized_moving_rect_ymin)
                    * PlacementHelper.get_config().placement_coverage_threshold
                ):

                    intersect_01_matrix[idx, idy] = 1
                else:
                    intersect_01_matrix[idx, idy] = 0

                pass

            pass

        placement_rectangle_list = []
        glog.info(f"intersect_01_matrix: {intersect_01_matrix}")
        for idx, x in enumerate(x_coords_new):
            for idy, y in enumerate(y_coords_new):
                if (
                    idx + 1 < len(x_coords_new)
                    and idy + 1 < len(y_coords_new)
                    and (
                        intersect_01_matrix[idx, idy]
                        + intersect_01_matrix[idx + 1, idy]
                        + intersect_01_matrix[idx, idy + 1]
                        + intersect_01_matrix[idx + 1, idy + 1]
                        == 4
                    )
                ):
                    placement_rectangle_list.append(
                        [
                            [x_coords_new[idx], y_coords_new[idy]],
                            [x_coords_new[idx], y_coords_new[idy + 1]],
                            [x_coords_new[idx + 1], y_coords_new[idy + 1]],
                            [x_coords_new[idx + 1], y_coords_new[idy]],
                        ]
                    )

        glog.info(f"placement_rectangle_list: {placement_rectangle_list}")
        placement_rectangle_list = [
            Basic2DGeometry.unnormalize_rectangle(placement_rectangle, _)
            for placement_rectangle in placement_rectangle_list
        ]
        glog.warning(f"placement_rectangle_list: {placement_rectangle_list}")

        return placement_rectangle_list

    @staticmethod
    def place_rectangle(fixed_rect_list, moving_rect, need_all_intersect=True):
        """
        Place the moving_rect in the area where it intersects with all the rectangles in fixed_rect_list.


        Here are the steps of the algorithm:
        1. Normalize the moving rectangle and fixed rectangles.
        2. First check if the moving rectangle can be placed in the area that is contained by the union of all fixed rectangles. if not, return -1.
        3. Then for each fixed rectangle, find the "intersecting bounds" of the moving rectangle that are intersecting with the fixed rectangle.
        4. If there are intersections for all the "intersecting bounds", then return any pose inside the intersecting bounds, meaning that we find a placement that is contained by the bound rectangle, and not intersecting with the fixed rectangles.
        Otherwise, return any pose and -2.

        We want to ensure that the moving rectangle can be placed in a pose that intersects with all the fixed rectangles whenever necessary, because we don't know which rectangle in the fixed_rect_list is the one that the agent is really trying to place while others are just to ensure the placement is valid.


        Args:
            fixed_rect_list: The list of fixed rectangles
            moving_rect: The moving rectangle
            need_all_intersect: If True, the moving rectangle must intersect with all the fixed rectangles. If False, the moving rectangle can intersect with any of the fixed rectangles.
        Returns:
            status: 0 if the placement is found, -1 if no placement is found, -2 if the placement is found but not intersecting with all the fixed rectangles.

        """
        rough_placement_rectangle_list = (
            PlacementHelper.find_placement_bounds_contained_by_rectangle_union(
                fixed_rect_list=fixed_rect_list,
                moving_rect=moving_rect,
                all_intersected=False,
            )
        )
        if len(rough_placement_rectangle_list) == 0:
            glog.warning("No placement found")
            return -1, "No placement found. The free space is too small"

        placement_bounds = (
            PlacementHelper.find_placement_bounds_intersecting_all_rectangles(
                fixed_rect_list, moving_rect
            )
        )
        # import ipdb

        # ipdb.set_trace()

        placement_rectangle_list = []
        if need_all_intersect:
            for rough_placement_rectangle in rough_placement_rectangle_list:
                placement_convex = Basic2DGeometry.intersection_of_parallel_rectangle(
                    rough_placement_rectangle, placement_bounds
                )
                if placement_convex is not None:
                    placement_rectangle_list.append(placement_convex)
        else:
            placement_rectangle_list = rough_placement_rectangle_list

        status = 0 if len(placement_rectangle_list) > 0 else -2

        if len(placement_rectangle_list):
            selected_rect_idx = np.random.choice(len(placement_rectangle_list))
            selected_rect = placement_rectangle_list[selected_rect_idx]
        else:
            selected_rect_idx = np.random.choice(len(rough_placement_rectangle_list))
            selected_rect = rough_placement_rectangle_list[selected_rect_idx]

        return status, np.mean(selected_rect, axis=0)

        pass

    pass


def main():
    rectangle_list = [
        [np.array([-3, 0]), np.array([-3, 3]), np.array([0, 3]), np.array([0, 0])],
        [np.array([0, 0]), np.array([0, 1]), np.array([2, 1]), np.array([2, 0])],
        [np.array([0, 1]), np.array([0, 2]), np.array([2, 2]), np.array([2, 1])],
        [np.array([0, 2]), np.array([0, 3]), np.array([1, 3]), np.array([1, 2])],
    ]

    moving_rectangle = [
        np.array([0, 0]),
        np.array([0, 2.1]),
        np.array([2.5, 2.1]),
        np.array([2.5, 0]),
    ]

    rectangle_list = [
        Basic2DGeometry.rotate_points_counterclockwise(rectangle, 2.51)
        for rectangle in rectangle_list
    ]
    moving_rectangle = Basic2DGeometry.rotate_points_counterclockwise(
        moving_rectangle, 2.51
    )

    placement_rectangle_list = (
        PlacementHelper.find_placement_bounds_contained_by_rectangle_union(
            rectangle_list, moving_rectangle
        )
    )

    # print(rectangle_list)
    # print(moving_rectangle)
    # print(placement_rectangle_list)

    pass


if __name__ == "__main__":

    main()
