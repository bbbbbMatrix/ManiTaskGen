import numpy as np
import trimesh
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from typing import List, Tuple, Optional
import glog
from src.utils.config_manager import get_ground_coverage_config


class GroundCoverageAnalyzer:
    # Use the global config instance
    def __init__(self):  # 1 cm = 0.01 meters
        """
        Initialize ground coverage analyzer

        Args:
            resolution: Rasterization resolution in meters (default 1 cm)
        """

        self.resolution = 0.01
        self.ground_meshes = []  # Store list of ground meshes
        self.ground_polygons = []  # Store list of ground polygons
        self.precomputed_freespace_data = (
            None  # Store precomputed freespace region data
        )

    @classmethod
    def get_config(cls):
        """
        Get configuration for ground coverage analyzer

        Returns:
            GroundCoverageConfig instance
        """
        return get_ground_coverage_config()

    def add_ground_mesh(self, mesh: trimesh.Trimesh):
        """
        Add ground object mesh to analyzer

        Args:
            mesh: 3D mesh object
        """
        self.ground_meshes.append(mesh)

        polygon = self.mesh_to_2d_polygon(mesh)
        if polygon and not polygon.is_empty:
            self.ground_polygons.append(polygon)

    def get_ground_polygons(self) -> List[Polygon]:
        """
        Get 2D polygon representation of all ground objects

        Returns:
            List of ground polygons
        """
        return self.ground_polygons

    def precompute_free_regions(
        self,
        global_bounds: Tuple[float, float, float, float] = None,
        min_rect_size=None,
    ) -> None:
        """
        Precompute all 0.5×0.5 freespace regions in the entire space

        Args:
            global_bounds: Global bounds (min_x, max_x, min_y, max_y), auto-calculated if None
            min_rect_size: Minimum rectangle size
        """
        # Get obstacle polygons
        obstacle_polygons = self.get_ground_polygons()
        min_rect_size = (
            self.get_config().min_rect_size if min_rect_size is None else min_rect_size
        )
        # Calculate global bounds
        if global_bounds is None:
            if obstacle_polygons:
                all_bounds = [
                    poly.bounds for poly in obstacle_polygons if not poly.is_empty
                ]
                if all_bounds:
                    min_x = min(bounds[0] for bounds in all_bounds)
                    max_x = max(bounds[2] for bounds in all_bounds)
                    min_y = min(bounds[1] for bounds in all_bounds)
                    max_y = max(bounds[3] for bounds in all_bounds)

                    # Expand bounds
                    margin = max(1.0, min_rect_size * 2)
                    global_bounds = (
                        min_x - margin,
                        max_x + margin,
                        min_y - margin,
                        max_y + margin,
                    )
                else:
                    # Default bounds
                    global_bounds = self.get_config().global_bounds
            else:
                global_bounds = self.get_config().global_bounds

        self.global_bounds = global_bounds
        min_x, max_x, min_y, max_y = global_bounds

        # Create global grid
        x_range = np.arange(min_x, max_x, self.get_config().resolution)
        y_range = np.arange(min_y, max_y, self.get_config().resolution)
        grid_x, grid_y = np.meshgrid(x_range, y_range)

        # Create obstacle mask
        obstacle_mask = np.zeros(grid_x.shape, dtype=int)

        if obstacle_polygons:
            valid_polygons = [
                p for p in obstacle_polygons if not p.is_empty and p.is_valid
            ]
            for poly in valid_polygons:
                poly_bounds = poly.bounds
                j_min = int((poly_bounds[0] - min_x) / self.get_config().resolution)
                j_max = int((poly_bounds[2] - min_x) / self.get_config().resolution)
                i_min = int((poly_bounds[1] - min_y) / self.get_config().resolution)
                i_max = int((poly_bounds[3] - min_y) / self.get_config().resolution)
                # import ipdb
                # ipdb.set_trace()
                for i in range(i_min, i_max):
                    for j in range(j_min, j_max):
                        point = Point(grid_x[i, j], grid_y[i, j])
                        if poly.contains(point) or poly.touches(point):
                            obstacle_mask[i, j] = 1

        # Calculate window size
        window_size = int(np.ceil(min_rect_size / self.get_config().resolution))

        # Create freespace mask (mark whether each position can place a min_rect_size×min_rect_size rectangle)
        free_region_mask = np.zeros(grid_x.shape, dtype=int)

        obstacle_mask_prefix = obstacle_mask.copy()
        obstacle_mask_prefix = np.cumsum(obstacle_mask_prefix, axis=0)
        obstacle_mask_prefix = np.cumsum(obstacle_mask_prefix, axis=1)

        # Sliding window check for each position
        for i in range(window_size // 2, grid_x.shape[0] - window_size // 2):
            for j in range(window_size // 2, grid_x.shape[1] - window_size // 2):
                # Calculate top-left and bottom-right indices of current window
                start_i = int(i - window_size // 2)
                end_i = int(i + window_size // 2)
                start_j = int(j - window_size // 2)
                end_j = int(j + window_size // 2)

                # Get obstacle count in current window
                obstacle_count = (
                    obstacle_mask_prefix[end_i, end_j]
                    - obstacle_mask_prefix[start_i, end_j]
                    - obstacle_mask_prefix[end_i, start_j]
                    + obstacle_mask_prefix[start_i, start_j]
                )

                # If no obstacles in window, mark as available
                if obstacle_count == 0:
                    free_region_mask[i, j] = 1

        # Create 2D prefix sum
        prefix_sum = np.cumsum(free_region_mask, axis=0)
        prefix_sum = np.cumsum(prefix_sum, axis=1)

        # Store precomputed data
        """
            free_region_mask: 2D mask marking whether each position can place a 0.5×0.5 rectangle
            prefix_sum: 2D prefix sum of free_region_mask
            grid_x, grid_y: Grid coordinate ranges
            window_size: Window size for checking 0.5×0.5 freespace regions
            min_rect_size: Minimum rectangle size (unit: meters)
        """
        self.precomputed_freespace_data = {
            "prefix_sum": prefix_sum,
            "obstacle_mask": obstacle_mask,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "window_size": window_size,
            "min_rect_size": min_rect_size,
        }

    def mesh_to_2d_polygon(self, mesh: trimesh.Trimesh, z_range=None) -> Polygon:
        """
        Project 3D mesh to ground plane (z=0) and convert to 2D polygon
        Preserve original shape without forcing convex hull conversion

        Args:
            mesh: 3D mesh object
            z_threshold: Z coordinate threshold (meters), only consider vertices below this threshold

        Returns:
            2D polygon (may be non-convex)
        """
        # Get vertices close to ground (vertices with small z coordinates)
        vertices = mesh.vertices

        vertices_2d = vertices[:, :2]

        z_range = self.get_config().z_range if z_range is None else z_range

        # Filter vertices by z-threshold
        valid_vertices_mask = vertices[:, 2] < z_range[1]

        vertices_2d = vertices[valid_vertices_mask][:, :2]

        # Project faces to 2D and filter out degenerate triangles
        polygons = []
        if hasattr(mesh, "faces") and len(mesh.faces) > 0 and len(vertices_2d) > 0:
            for face in mesh.faces:
                # Get the 2D coordinates of each vertex in the face
                triangle_points = vertices[:, :2][face]

                # Skip degenerate triangles (those that form a line or point)
                # Calculate triangle area using cross product
                v1 = triangle_points[1] - triangle_points[0]
                v2 = triangle_points[2] - triangle_points[0]
                area = abs(np.cross(v1, v2)) / 2.0

                if area > 1e-8:  # Non-degenerate triangle
                    polygons.append(Polygon(triangle_points))

            if polygons:
                # Merge all triangles into a single polygon
                glog.info(f"Found {len(polygons)} valid faces in the mesh.")
                merged_polygon = unary_union(polygons)
                print(
                    f"Merged polygon has {len(merged_polygon.geoms) if isinstance(merged_polygon, MultiPolygon) else 1} components."
                )
                return merged_polygon

        # Fall back to convex hull if no valid faces found or mesh has no faces
        if not polygons:
            # Try to create a simple polygon from the 2D vertices
            if len(vertices_2d) >= 3:
                return Polygon(vertices_2d).convex_hull

        return Polygon()  # Return an empty polygon if no valid vertices found

    def get_rotated_rectangle_bounds(
        self, rect_vertices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of rotated rectangle

        Args:
            rect_vertices: Four vertices of rectangle (unit: meters) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            min_coords, max_coords: Minimum and maximum coordinates of bounding box (unit: meters)
        """
        min_coords = np.min(rect_vertices, axis=0)
        max_coords = np.max(rect_vertices, axis=0)
        return min_coords, max_coords

    def _query_rectangle_free_regions(
        self, rect_vertices: np.ndarray
    ) -> Tuple[bool, List[np.ndarray]]:
        """
        Fast query whether rectangle contains 0.5×0.5 freespace regions (using precomputed data)

        Args:
            rect_vertices: Rectangle vertices

        Returns:
            (has_free_region, sample_points): Whether there are freespace regions and list of sample points
        """
        if self.precomputed_freespace_data is None:
            raise ValueError(
                "Please call precompute_free_regions() first for precomputation"
            )

        prefix_sum = self.precomputed_freespace_data["prefix_sum"]
        obstacle_mask = self.precomputed_freespace_data["obstacle_mask"]
        grid_x = self.precomputed_freespace_data["grid_x"]
        grid_y = self.precomputed_freespace_data["grid_y"]
        window_size = self.precomputed_freespace_data["window_size"]
        min_rect_size = self.precomputed_freespace_data["min_rect_size"]

        # Get rectangle bounds
        min_x, max_x = np.min(rect_vertices[:, 0]), np.max(rect_vertices[:, 0])
        min_y, max_y = np.min(rect_vertices[:, 1]), np.max(rect_vertices[:, 1])

        # Convert rectangle coordinates to grid indices
        i_min = np.maximum(0, np.searchsorted(grid_y[:, 0], min_y, side="left") - 1)
        i_max = np.minimum(
            grid_y.shape[0] - 1, np.searchsorted(grid_y[:, 0], max_y, side="right")
        )
        j_min = np.maximum(0, np.searchsorted(grid_x[0, :], min_x, side="left") - 1)
        j_max = np.minimum(
            grid_x.shape[1] - 1, np.searchsorted(grid_x[0, :], max_x, side="right")
        )

        # If rectangle is outside grid range, return no free regions
        if i_min >= i_max or j_min >= j_max:
            return False, [], []

        # Create polygon for rectangular region
        rect_polygon = Polygon(rect_vertices)

        # Use prefix sum to quickly check if there are free regions in rectangle area
        region_sum = (
            prefix_sum[i_max, j_max]
            - prefix_sum[i_min, j_max]
            - prefix_sum[i_max, j_min]
            + prefix_sum[i_min, j_min]
        )

        sample_points = []
        rectangles = []
        # import ipdb
        # ipdb.set_trace()
        if region_sum > 0:
            # Find all free points in rectangle region
            free_points_i, free_points_j = np.where(
                obstacle_mask[i_min:i_max, j_min:j_max] == 0
            )

            free_points_i += i_min
            free_points_j += j_min

            if len(free_points_i) > 0:
                # Calculate coordinates of each free point
                coords = np.column_stack(
                    (
                        grid_x[free_points_i, free_points_j],
                        grid_y[free_points_i, free_points_j],
                    )
                )

                # Filter points inside rotated rectangle
                in_rect = np.array(
                    [rect_polygon.contains(Point(x, y)) for x, y in coords]
                )
                valid_coords = coords[in_rect]

                if len(valid_coords) > 0:

                    dists = np.linalg.norm(
                        valid_coords
                        - np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5]),
                        axis=1,
                    )
                    closest_idx = np.argmin(dists)
                    closest_point = valid_coords[closest_idx]

                    # Add to results
                    sample_points = [closest_point]

                    # Build corresponding min_rect_size x min_rect_size rectangle
                    half_size = min_rect_size / 2
                    rect = np.array(
                        [
                            [
                                closest_point[0] - half_size,
                                closest_point[1] - half_size,
                            ],
                            [
                                closest_point[0] + half_size,
                                closest_point[1] - half_size,
                            ],
                            [
                                closest_point[0] + half_size,
                                closest_point[1] + half_size,
                            ],
                            [
                                closest_point[0] - half_size,
                                closest_point[1] + half_size,
                            ],
                        ]
                    )
                    rectangles = [rect]

        # import ipdb
        # ipdb.set_trace()
        return len(sample_points) > 0, sample_points, rectangles

    def analyze_coverage(
        self,
        rect_vertices: np.ndarray,
        min_rect_size=None,  # 0.5 meters = 50 cm
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Analyze uncovered regions in L×0.5 meter rectangle

        Args:
            rect_vertices: Four vertices of L×0.5 meter rectangle (unit: meters) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            min_rect_size: Minimum rectangle size threshold (unit: meters, default 0.5 meters = 50 cm)
            z_threshold: Z coordinate threshold (unit: meters) for mesh projection

        Returns:
            List of uncovered large rectangle regions, each element is (rectangle vertices, geometric center), units in meters

        * When passing in, rect_vertices(0, 1) should be the length of the rectangle.

        """

        min_rect_size = (
            self.get_config().min_rect_size if min_rect_size is None else min_rect_size
        )

        if (
            np.abs(np.linalg.norm(rect_vertices[1] - rect_vertices[0]) - min_rect_size)
            < self.get_config().eps
        ):
            rect_vertices = (
                rect_vertices[1:] + rect_vertices[:1]
            )  # Ensure rect_vertices[0, 1] is the length of rectangle

        # Convert all meshes to 2D polygons
        obstacle_polygons = self.get_ground_polygons()

        orientation = rect_vertices[1] - rect_vertices[0]
        orientation_length = np.linalg.norm(orientation)
        orientation = (
            orientation / orientation_length
            if orientation_length > 0
            else np.array([1, 0])
        )

        max_pieces = int(np.ceil(orientation_length / min_rect_size))

        # Calculate length of each small rectangle
        piece_length = orientation_length / max_pieces

        rotated_rectangle_bounds = self.get_rotated_rectangle_bounds(rect_vertices)

        sample_point_list = []
        rectangle_list = []

        for i in range(max_pieces):
            # Calculate four vertices of current small rectangle
            piece_start = rect_vertices[0] + i * piece_length * orientation
            piece_end = rect_vertices[0] + (i + 1) * piece_length * orientation

            # Calculate four vertices of small rectangle
            piece_vertices = np.array(
                [
                    piece_start,
                    piece_end,
                    piece_end
                    + np.array([-orientation[1], orientation[0]]) * min_rect_size,
                    piece_start
                    + np.array([-orientation[1], orientation[0]]) * min_rect_size,
                ]
            )

            # Query whether there are freespace regions in this small rectangle
            has_free_region, sample_points, rectangles = (
                self._query_rectangle_free_regions(piece_vertices)
            )

            sample_point_list.extend(sample_points)
            rectangle_list.extend(rectangles)

        return sample_point_list, rectangle_list


# Usage example
def example_usage():
    """Usage example"""

    # Create analyzer (1 cm resolution)
    analyzer = GroundCoverageAnalyzer(resolution=0.01)

    # Add some example meshes (simulate ground objects, unit: meters)
    # Example mesh 1: a cube (30cm × 30cm × 10cm)
    mesh1 = trimesh.creation.box(extents=[0.3, 0.3, 0.1])
    mesh1.apply_translation([0.8, 0.9, 0.05])
    analyzer.add_ground_mesh(mesh1)

    # Example mesh 2: a cylinder (radius 15cm, height 10cm)
    mesh2 = trimesh.creation.cylinder(radius=0.15, height=0.1)
    mesh2.apply_translation([1.5, 1.2, 0.05])
    analyzer.add_ground_mesh(mesh2)

    # Define L×0.5 meter rectangle (example: 2-meter length, 0.5-meter width rectangle with rotation)
    L = 2.0  # 2 meters
    width = 0.5  # 0.5 meters = 50 cm
    angle = np.pi / 6  # 30 degree rotation

    # Rectangle center (unit: meters)
    center = np.array([1.0, 1.0])

    # Calculate four vertices of rotated rectangle
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    half_L, half_w = L / 2, width / 2

    local_vertices = np.array(
        [
            [-half_L, -half_w],  # bottom-left
            [half_L, -half_w],  # bottom-right
            [half_L, half_w],  # top-right
            [-half_L, half_w],  # top-left
        ]
    )

    # Apply rotation and translation
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rect_vertices = np.dot(local_vertices, rotation_matrix.T) + center

    # Perform analysis (find regions larger than 0.5m×0.5m)
    large_rectangles = analyzer.analyze_coverage(rect_vertices, min_rect_size=0.5)

    # Output results
    print(
        f"Found {len(large_rectangles)} uncovered rectangle regions larger than 0.5m×0.5m:"
    )
    for i, (vertices, center) in enumerate(large_rectangles):
        print(f"Rectangle {i+1}:")
        print(f"  Vertices (meters): {vertices}")
        print(f"  Geometric center (meters): {center}")
        print()


if __name__ == "__main__":
    example_usage()
