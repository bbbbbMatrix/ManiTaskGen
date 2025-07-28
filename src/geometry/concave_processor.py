"""
custom_geometry_helper_new/concave_processor.py

This module provides functionality for processing concave polygons, including detection and decomposition.

This is mainly used when detected concave platforms(e.g. L-shaped, U-shaped counters). It will divide the concave polygon into multiple convex polygons, which can be used for further geometric operations or task generation.

The criteria for decomposition include:

- Ensuring that the resulting polygons are convex
- Maintaining a minimum area for each decomposed polygon
- Preserving the overall shape and features of the original concave polygon

"""

import trimesh
import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Tuple, Optional
from collections import deque
import glog
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from src.geometry.basic_geometries import Basic3DGeometry
from src.utils.config_manager import get_concave_processor_config


class ConcaveProcessor:
    """
    Class for processing concave platforms, providing detection and decomposition functionality.
    Focuses on rectangular segmentation, generating convex polygons as close to rectangles as possible,
    and merging adjacent rectangles.

    This class handles the decomposition of concave polygons into simpler convex shapes through
    various strategies including rectangular decomposition, grid-based decomposition, and
    adaptive approaches with merging capabilities.
    """

    @classmethod
    def get_config(cls):
        """Get the current concave processor configuration"""
        return get_concave_processor_config()

    def __init__(self, vertices: np.ndarray = None, faces: List[List[int]] = None):
        """
        Initialize the concave polygon processor.

        Args:
            vertices: 3D vertex array containing polygon vertices
            faces: List of face indices defining triangular faces

        Attributes:
            min_polygon_area: Minimum polygon area threshold for filtering
            target_aspect_ratio: Target aspect ratio for rectangular decomposition
            max_target_strips: Maximum number of strips for decomposition
            merge_tolerance: Tolerance for merging adjacent rectangles

        """

        self.vertices = vertices if vertices is not None else np.array([])
        self.faces = faces if faces is not None else []

    @staticmethod
    def decompose_concave_polygon(
        vertices_3d: np.ndarray, faces: List[List[int]]
    ) -> List[Tuple[np.ndarray, List[List[int]]]]:
        """
        Static method for decomposing concave polygons.

        This is a convenience method that creates a ConcaveProcessor instance
        and performs decomposition using the rectangular with merge strategy.

        Args:
            vertices_3d: 3D vertices of the polygon
            faces: Face indices defining the polygon structure

        Returns:
            List of decomposed polygons as (vertices, faces) tuples
        """
        processor = ConcaveProcessor(vertices_3d, faces)
        return processor.decompose(strategy="rectangular_with_merge")

    def decompose(
        self, strategy: str = "rectangular_with_merge"
    ) -> List[Tuple[np.ndarray, List[List[int]]]]:
        """
        Decompose concave polygon into multiple simple polygons.

        This method applies the specified decomposition strategy to break down
        a concave polygon into simpler convex shapes. The rectangular_with_merge
        strategy provides the best balance of simplicity and efficiency.

        Args:
            strategy: Decomposition strategy. Options:
                - 'rectangular_with_merge': Rectangular decomposition with adjacent merging
                - 'rectangular': Basic rectangular decomposition
                - 'grid': Grid-based decomposition
                - 'adaptive': Adaptive decomposition (placeholder)

        Returns:
            List of decomposed polygons as (vertices, faces) tuples
        """
        if len(self.vertices) < 3:
            return []

        # Check if decomposition is needed
        if not self._is_concave():
            glog.info("Polygon is already convex, no decomposition needed")
            return [(self.vertices, self.faces)]

        # glog.info(f"Starting decomposition using '{strategy}' strategy")

        if strategy == "rectangular_with_merge":
            # Perform rectangular decomposition, then merge adjacent rectangles
            rect_results = self._rectangular_decomposition()
            merged_results = self._merge_adjacent_rectangles(rect_results)
            rectanglize_results = self._perserve_original_vertices(merged_results)

            # import ipdb
            # ipdb.set_trace()
            return rectanglize_results
        elif strategy == "rectangular":
            return self._rectangular_decomposition()
        elif strategy == "grid":
            return self._grid_decomposition()
        elif strategy == "adaptive":
            return self._adaptive_decomposition()
        else:
            return self._rectangular_decomposition()

    def _perserve_original_vertices(
        self, polygons: List[Tuple[np.ndarray, List[List[int]]]]
    ) -> List[Tuple[np.ndarray, List[List[int]]]]:
        """
        Preserve original vertices in the decomposed polygons.

        This method filters the vertices of decomposed polygons to keep only those
        that are close to the original polygon's faces, maintaining geometric fidelity
        while creating convex hull representations.

        Args:
            polygons: List of decomposed polygons as (vertices, faces) tuples

        Returns:
            List of polygons with preserved original vertices
        """
        if not polygons:
            return []

        # glog.info(f"Starting to preserve original vertices, current {len(polygons)} polygons")

        preserved_polygons = []

        for vertices, faces in polygons:
            # Preserve original vertices
            vertices_3d = np.array(vertices)
            # Extract all original vertex coordinates (3D)
            original_vertices = self.vertices.copy()

            z_value = vertices_3d[0, 2]  # Get current Z value

            # Record preserved vertices
            preserved_verts = []
            eps = self.get_config().eps  # Get EPS value from config
            for vertex in vertices_3d:
                # Check if current vertex is close enough to any original face or on an edge
                is_close_to_face = False

                # Iterate through all original face edges
                for face in self.faces:
                    # Each face has 3 edges

                    # Calculate point to triangle distance
                    distance = Basic3DGeometry.point_to_triangle_distance(
                        point=vertex,
                        triangle=(
                            original_vertices[face[0]],
                            original_vertices[face[1]],
                            original_vertices[face[2]],
                        ),
                    )

                    if distance <= eps:  # Distance threshold, adjustable
                        is_close_to_face = True
                        break

                if is_close_to_face:
                    preserved_verts.append(vertex)

            # Convert to numpy array

            # Convert to numpy array
            preserved_verts = (
                np.array(preserved_verts) if preserved_verts else np.array([])
            )

            # Skip polygon if fewer than 3 vertices are preserved
            if len(preserved_verts) < 3:
                glog.warning(f"Fewer than 3 vertices preserved, skipping this polygon")
                continue

            # Remove duplicate points
            unique_verts = []
            for vert in preserved_verts:
                is_duplicate = False
                for existing_vert in unique_verts:
                    if np.allclose(vert, existing_vert, rtol=1e-8, atol=1e-8):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_verts.append(vert)

            unique_verts = np.array(unique_verts)

            # Skip if fewer than 3 points after deduplication
            if len(unique_verts) < 3:
                glog.warning(
                    f"Fewer than 3 vertices after deduplication, skipping this polygon"
                )
                continue

            # Calculate 2D convex hull for preserved points (using only x,y coordinates)
            try:
                # Extract 2D coordinates
                verts_2d = unique_verts[:, :2]

                # Calculate 2D convex hull
                hull = ConvexHull(verts_2d)

                # Get convex hull vertex indices
                hull_vertex_indices = hull.vertices

                # Build 3D convex hull vertices (maintain original Z value)
                hull_vertices_3d = unique_verts[hull_vertex_indices]

                # Generate faces (fan triangulation)
                hull_faces = []
                n_hull_verts = len(hull_vertices_3d)
                if n_hull_verts >= 3:
                    for i in range(1, n_hull_verts - 1):
                        hull_faces.append([0, i, i + 1])

                # Add to result list
                preserved_polygons.append((hull_vertices_3d, hull_faces))

            except Exception as e:
                glog.warning(f"Error calculating convex hull: {e}")
                continue

        # glog.info(f"Original vertex preservation complete, total {len(preserved_polygons)} polygons")
        return preserved_polygons

    def _merge_adjacent_rectangles(
        self, polygons: List[Tuple[np.ndarray, List[List[int]]]]
    ) -> List[Tuple[np.ndarray, List[List[int]]]]:
        """
        Merge adjacent rectangles to reduce the number of polygons.

        This method identifies adjacent rectangular polygons and merges them into
        larger rectangles when possible, reducing complexity while maintaining
        geometric accuracy.

        Args:
            polygons: List of polygons as (vertices, faces) tuples

        Returns:
            List of merged polygons
        """
        if len(polygons) <= 1:
            return polygons

        glog.info(
            f"Starting to merge adjacent rectangles, current {len(polygons)} polygons"
        )

        # Create Shapely polygon list
        shapely_polygons = []
        original_indices = []

        for i, (vertices, faces) in enumerate(polygons):
            vertices_2d = vertices[:, :2]
            try:
                poly = Polygon(vertices_2d)
                if poly.is_valid and not poly.is_empty:
                    shapely_polygons.append(poly)
                    original_indices.append(i)
            except Exception as e:
                glog.warning(f"Unable to create polygon {i}: {e}")

        if not shapely_polygons:
            return polygons

        # Build adjacency graph
        adjacency = self._build_adjacency_graph(shapely_polygons)

        # Find groups of rectangles that can be merged
        merge_groups = self._find_merge_groups(shapely_polygons, adjacency)

        # Perform merging
        merged_results = []
        used_indices = set()

        for group in merge_groups:
            if len(group) > 1:
                # Merge this group of polygons
                merged_poly = self._merge_polygon_group(
                    [shapely_polygons[i] for i in group]
                )
                if merged_poly and not merged_poly.is_empty:
                    # Convert back to 3D format
                    z_value = polygons[original_indices[group[0]]][0][
                        0, 2
                    ]  # Use Z value from first polygon
                    vertices_3d, faces_3d = self._polygon_to_3d(merged_poly, z_value)
                    merged_results.append((vertices_3d, faces_3d))
                    used_indices.update(group)
                    glog.info(f"Merged {len(group)} rectangles")

        # Add unmerged polygons
        for i, poly_idx in enumerate(original_indices):
            if i not in used_indices:
                merged_results.append(polygons[poly_idx])

        glog.info(
            f"Merging complete: {len(polygons)} -> {len(merged_results)} polygons"
        )
        return merged_results

    def _build_adjacency_graph(self, polygons: List[Polygon]) -> List[List[int]]:
        """
        Build adjacency graph for polygons.

        Creates a graph where each polygon is a node and edges represent
        adjacency relationships between polygons that share boundaries.

        Args:
            polygons: List of Shapely polygons

        Returns:
            Adjacency list representation of the graph
        """
        n = len(polygons)
        adjacency = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if self._are_adjacent_rectangles(polygons[i], polygons[j]):
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        return adjacency

    def _are_adjacent_rectangles(self, poly1: Polygon, poly2: Polygon) -> bool:
        """
        Determine if two rectangles are adjacent (share a boundary).

        Two polygons are considered adjacent if they share a boundary segment
        of sufficient length, indicating they can potentially be merged.

        Args:
            poly1: First polygon
            poly2: Second polygon

        Returns:
            True if polygons are adjacent, False otherwise
        """
        try:
            # Check if there's a shared boundary
            intersection = poly1.boundary.intersection(poly2.boundary)

            if intersection.is_empty:
                return False

            merge_tolerance = self.get_config().merge_tolerance
            # If intersection is a line segment with sufficient length, consider them adjacent
            if hasattr(intersection, "length"):
                # Single line segment
                return intersection.length > merge_tolerance
            elif hasattr(intersection, "geoms"):
                # Multiple geometries
                total_length = sum(
                    geom.length
                    for geom in intersection.geoms
                    if hasattr(geom, "length")
                )
                return total_length > merge_tolerance

            return False

        except Exception as e:
            glog.warning(f"Error checking adjacency: {e}")
            return False

    def _find_merge_groups(
        self, polygons: List[Polygon], adjacency: List[List[int]]
    ) -> List[List[int]]:
        """
        Find groups of rectangles that can be merged.

        Uses breadth-first search to find connected components of adjacent
        rectangles that can be merged into larger rectangles.

        Args:
            polygons: List of polygons to analyze
            adjacency: Adjacency graph of the polygons

        Returns:
            List of groups, where each group contains polygon indices that can be merged
        """
        n = len(polygons)
        visited = [False] * n
        merge_groups = []

        for i in range(n):
            if visited[i]:
                continue

            # BFS to find connected rectangle groups
            group = []
            queue = deque([i])
            visited[i] = True

            while queue:
                current = queue.popleft()
                group.append(current)

                for neighbor in adjacency[current]:
                    if not visited[neighbor] and self._can_merge_rectangles(
                        polygons, group + [neighbor]
                    ):
                        visited[neighbor] = True
                        queue.append(neighbor)

            merge_groups.append(group)

        return merge_groups

    def _can_merge_rectangles(
        self, polygons: List[Polygon], indices: List[int]
    ) -> bool:
        """
        Determine if a group of rectangles can be merged into a larger rectangle.

        Tests whether the union of the specified polygons results in a shape
        that is approximately rectangular, indicating successful merging is possible.

        Args:
            polygons: List of all polygons
            indices: Indices of polygons to test for merging

        Returns:
            True if the rectangles can be merged, False otherwise
        """
        if len(indices) <= 1:
            return True

        try:
            # Try to merge these polygons
            polys_to_merge = [polygons[i] for i in indices]
            merged = unary_union(polys_to_merge)

            if merged.is_empty or not merged.is_valid:
                return False

            # Check if merged result is approximately rectangular
            return self._is_approximately_rectangle(merged)

        except Exception as e:
            glog.warning(f"Error checking merge possibility: {e}")
            return False

    def _is_approximately_rectangle(
        self, polygon: Polygon, tolerance: float = 0.2
    ) -> bool:
        """
        Determine if a polygon is approximately rectangular.

        Compares the polygon's area with its bounding box area to determine
        if it's close enough to a rectangle for merging purposes.

        Args:
            polygon: Polygon to test
            tolerance: Tolerance for rectangle similarity (0.0 to 1.0)

        Returns:
            True if polygon is approximately rectangular, False otherwise
        """
        try:
            if polygon.is_empty:
                return False

            # Get bounding box
            minx, miny, maxx, maxy = polygon.bounds
            bbox = box(minx, miny, maxx, maxy)

            # Calculate similarity between polygon and its bounding box
            intersection_area = polygon.intersection(bbox).area
            union_area = polygon.union(bbox).area

            if union_area == 0:
                return False

            similarity = intersection_area / union_area

            # If similarity is high enough, consider it approximately rectangular
            return similarity > (1 - tolerance)

        except Exception as e:
            glog.warning(f"Error checking rectangle similarity: {e}")
            return False

    def _merge_polygon_group(self, polygons: List[Polygon]) -> Optional[Polygon]:
        """
        Merge a group of polygons into a single polygon.

        Combines multiple polygons using unary union and approximates the result
        with a bounding box if it's not sufficiently rectangular.

        Args:
            polygons: List of polygons to merge

        Returns:
            Merged polygon or None if merging fails
        """
        try:
            if len(polygons) == 1:
                return polygons[0]

            # Use unary_union to merge
            merged = unary_union(polygons)

            if merged.is_valid and not merged.is_empty:
                # If result is not rectangular, try to approximate with bounding box
                if not self._is_approximately_rectangle(merged, tolerance=0.2):
                    minx, miny, maxx, maxy = merged.bounds
                    merged = box(minx, miny, maxx, maxy)

                return merged

            return None

        except Exception as e:
            glog.warning(f"Error merging polygon group: {e}")
            return None

    def _rectangular_decomposition(self) -> List[Tuple[np.ndarray, List[List[int]]]]:
        """
        Rectangular decomposition strategy.

        Decomposes the polygon using rectangular strips based on the aspect ratio
        of the polygon's bounding box. Chooses between vertical, horizontal, or
        mixed decomposition strategies.

        Returns:
            List of decomposed polygons as (vertices, faces) tuples
        """
        vertices_2d = self.vertices[:, :2]

        # Calculate bounding box
        min_x, max_x = np.min(vertices_2d[:, 0]), np.max(vertices_2d[:, 0])
        min_y, max_y = np.min(vertices_2d[:, 1]), np.max(vertices_2d[:, 1])
        width = max_x - min_x
        height = max_y - min_y

        config = self.get_config()
        # Decide decomposition direction
        if width > height * config.target_aspect_ratio:
            # Horizontal decomposition (vertical cutting lines)
            return self._vertical_strip_decomposition()
        elif height > width * config.target_aspect_ratio:
            # Vertical decomposition (horizontal cutting lines)
            return self._horizontal_strip_decomposition()
        else:
            # Mixed decomposition
            return self._mixed_decomposition()

    def _vertical_strip_decomposition(self) -> List[Tuple[np.ndarray, List[List[int]]]]:
        """
        Vertical strip decomposition (decompose using vertical cutting lines).

        Divides the polygon into vertical strips of approximately equal width,
        with width determined by target aspect ratio and maximum strip count.

        Returns:
            List of decomposed polygons from vertical strips
        """
        vertices_2d = self.vertices[:, :2]
        min_x, max_x = np.min(vertices_2d[:, 0]), np.max(vertices_2d[:, 0])
        min_y, max_y = np.min(vertices_2d[:, 1]), np.max(vertices_2d[:, 1])

        width = max_x - min_x
        height = max_y - min_y

        config = self.get_config()
        # Calculate ideal strip count - reduce number of divisions
        ideal_strip_width = max(
            height * config.target_aspect_ratio * 1.5, width / config.max_target_strips
        )  # Increase strip width
        num_strips = max(1, int(np.ceil(width / ideal_strip_width)))

        glog.info(
            f"Vertical decomposition: {num_strips} strips, ideal width: {ideal_strip_width:.3f}"
        )

        # Create original polygon
        original_polygon = self._create_polygon_from_vertices(vertices_2d)
        if original_polygon is None or original_polygon.is_empty:
            return [(self.vertices, self.faces)]

        results = []
        strip_width = width / num_strips

        for i in range(num_strips):
            x_start = min_x + i * strip_width
            x_end = min_x + (i + 1) * strip_width

            # Create strip rectangle
            strip_rect = Polygon(
                [(x_start, min_y), (x_end, min_y), (x_end, max_y), (x_start, max_y)]
            )

            # Calculate intersection
            intersection = original_polygon.intersection(strip_rect)

            if not intersection.is_empty:
                strip_polygons = self._extract_valid_polygons(intersection)
                for poly in strip_polygons:
                    vertices_3d, faces = self._polygon_to_3d(poly, self.vertices[0, 2])
                    if len(vertices_3d) >= 3:
                        results.append((vertices_3d, faces))

        return self._post_process_results(results)

    def _horizontal_strip_decomposition(
        self,
    ) -> List[Tuple[np.ndarray, List[List[int]]]]:
        """
        Horizontal strip decomposition (decompose using horizontal cutting lines).

        Divides the polygon into horizontal strips of approximately equal height,
        with height determined by target aspect ratio and maximum strip count.

        Returns:
            List of decomposed polygons from horizontal strips
        """
        vertices_2d = self.vertices[:, :2]
        min_x, max_x = np.min(vertices_2d[:, 0]), np.max(vertices_2d[:, 0])
        min_y, max_y = np.min(vertices_2d[:, 1]), np.max(vertices_2d[:, 1])

        width = max_x - min_x
        height = max_y - min_y

        config = self.get_config()
        # Calculate ideal strip count - reduce number of divisions
        ideal_strip_height = max(
            width / config.target_aspect_ratio * 1.5, height / config.max_target_strips
        )  # Increase strip height
        num_strips = max(1, int(np.ceil(height / ideal_strip_height)))

        # glog.info(f"Horizontal decomposition: {num_strips} strips, ideal height: {ideal_strip_height:.3f}")

        # Create original polygon
        original_polygon = self._create_polygon_from_vertices(vertices_2d)
        if original_polygon is None or original_polygon.is_empty:
            return [(self.vertices, self.faces)]

        results = []
        strip_height = height / num_strips

        for i in range(num_strips):
            y_start = min_y + i * strip_height
            y_end = min_y + (i + 1) * strip_height

            # Create strip rectangle
            strip_rect = Polygon(
                [(min_x, y_start), (max_x, y_start), (max_x, y_end), (min_x, y_end)]
            )

            # Calculate intersection
            intersection = original_polygon.intersection(strip_rect)

            if not intersection.is_empty:
                strip_polygons = self._extract_valid_polygons(intersection)
                for poly in strip_polygons:
                    vertices_3d, faces = self._polygon_to_3d(poly, self.vertices[0, 2])
                    if len(vertices_3d) >= 3:
                        results.append((vertices_3d, faces))

        return self._post_process_results(results)

    def _mixed_decomposition(self) -> List[Tuple[np.ndarray, List[List[int]]]]:
        """
        Mixed decomposition strategy - reduced grid density.

        Uses a grid-based approach with larger cells when the polygon doesn't
        have a clear dominant direction. Aims for fewer, larger cells.

        Returns:
            List of decomposed polygons from grid cells
        """
        vertices_2d = self.vertices[:, :2]
        min_x, max_x = np.min(vertices_2d[:, 0]), np.max(vertices_2d[:, 0])
        min_y, max_y = np.min(vertices_2d[:, 1]), np.max(vertices_2d[:, 1])

        width = max_x - min_x
        height = max_y - min_y

        # Calculate grid size - use larger cells
        total_area = width * height
        target_cell_area = total_area / 3  # Target 3 cells instead of 6
        cell_size = np.sqrt(target_cell_area)

        nx = max(1, int(np.ceil(width / cell_size)))
        ny = max(1, int(np.ceil(height / cell_size)))

        # Create original polygon
        original_polygon = self._create_polygon_from_vertices(vertices_2d)
        if original_polygon is None or original_polygon.is_empty:
            return [(self.vertices, self.faces)]

        results = []
        cell_width = width / nx
        cell_height = height / ny

        for i in range(nx):
            for j in range(ny):
                x_start = min_x + i * cell_width
                x_end = min_x + (i + 1) * cell_width
                y_start = min_y + j * cell_height
                y_end = min_y + (j + 1) * cell_height

                # Create cell rectangle
                cell_rect = Polygon(
                    [
                        (x_start, y_start),
                        (x_end, y_start),
                        (x_end, y_end),
                        (x_start, y_end),
                    ]
                )

                # Calculate intersection
                intersection = original_polygon.intersection(cell_rect)

                if not intersection.is_empty:
                    cell_polygons = self._extract_valid_polygons(intersection)
                    for poly in cell_polygons:
                        vertices_3d, faces = self._polygon_to_3d(
                            poly, self.vertices[0, 2]
                        )
                        if len(vertices_3d) >= 3:
                            results.append((vertices_3d, faces))

        return self._post_process_results(results)

    # Keep original other methods
    def _is_concave(self) -> bool:
        """
        Determine if polygon is concave.

        Uses two methods to detect concavity:
        1. Compares actual area with convex hull area
        2. Checks for reflex angles in the polygon

        Args:
            threshold: Threshold for area ratio comparison (0.0 to 1.0)

        Returns:
            True if polygon is concave, False otherwise
        """
        try:

            vertices_2d = self.vertices[:, :2]

            if len(vertices_2d) < 4:
                return False

            # Method 1: Compare convex hull area
            hull = ConvexHull(vertices_2d)
            hull_area = hull.volume
            actual_area = sum(Polygon(vertices_2d[face]).area for face in self.faces)
            print(f"Actual area: {actual_area}, Hull area: {hull_area}")
            config = self.get_config()

            if hull_area > config.concave_min_area:
                area_ratio = actual_area / hull_area
                if area_ratio < (1 - config.concave_threshold):
                    return True
                else:
                    return False
            else:
                return False

        except Exception as e:
            glog.warning(f"Concave detection failed: {e}")
            return False

    def _has_reflex_angles(self, vertices_2d: np.ndarray) -> bool:
        """
        Check for reflex angles in the polygon.

        Examines each vertex to determine if it forms a reflex angle (>180°),
        which indicates the polygon is concave.

        Args:
            vertices_2d: 2D vertices of the polygon

        Returns:
            True if polygon has reflex angles, False otherwise
        """
        if len(vertices_2d) < 4:
            return False

        # Sort vertices by angle
        center = np.mean(vertices_2d, axis=0)
        angles = np.arctan2(
            vertices_2d[:, 1] - center[1], vertices_2d[:, 0] - center[0]
        )
        sorted_indices = np.argsort(angles)
        sorted_vertices = vertices_2d[sorted_indices]

        n = len(sorted_vertices)
        for i in range(n):
            p1 = sorted_vertices[(i - 1) % n]
            p2 = sorted_vertices[i]
            p3 = sorted_vertices[(i + 1) % n]

            v1 = p1 - p2
            v2 = p3 - p2

            # Calculate cross product to determine turning direction
            cross = np.cross(v1, v2)
            if cross < -1e-6:  # Right turn indicates concave point
                return True

        return False

    def _create_polygon_from_vertices(
        self, vertices_2d: np.ndarray
    ) -> Optional[Polygon]:
        """
        Create Shapely polygon from vertices.

        Sorts vertices by angle around centroid and creates a valid Shapely polygon.
        Attempts to repair invalid polygons using buffering.

        Args:
            vertices_2d: 2D vertices to create polygon from

        Returns:
            Valid Shapely polygon or None if creation fails
        """
        try:
            if len(vertices_2d) < 3:
                return None

            # Sort vertices by angle around centroid
            center = np.mean(vertices_2d, axis=0)
            angles = np.arctan2(
                vertices_2d[:, 1] - center[1], vertices_2d[:, 0] - center[0]
            )
            sorted_indices = np.argsort(angles)
            sorted_vertices = vertices_2d[sorted_indices]

            # Create polygon
            polygon = Polygon(sorted_vertices)

            # Check polygon validity
            if not polygon.is_valid:
                # Try to repair
                polygon = polygon.buffer(0)

            return polygon if polygon.is_valid else None

        except Exception as e:
            glog.warning(f"Polygon creation failed: {e}")
            return None

    def _extract_valid_polygons(self, geom) -> List[Polygon]:
        """
        Extract valid polygons from geometric object.

        Handles both single polygons and collections of polygons (MultiPolygon,
        GeometryCollection), filtering out polygons smaller than the minimum area.

        Args:
            geom: Shapely geometry object to extract polygons from

        Returns:
            List of valid polygons meeting minimum area requirement
        """
        polygons = []

        if geom.is_empty:
            return polygons

        config = self.get_config()
        if hasattr(geom, "geoms"):
            # MultiPolygon or GeometryCollection
            for sub_geom in geom.geoms:
                if hasattr(sub_geom, "exterior"):  # Polygon
                    if sub_geom.area > config.min_polygon_area:
                        polygons.append(sub_geom)
        elif hasattr(geom, "exterior"):  # Single Polygon
            if geom.area > config.min_polygon_area:
                polygons.append(geom)

        return polygons

    def _polygon_to_3d(
        self, polygon: Polygon, z_value: float
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Convert 2D polygon to 3D vertices and faces.

        Takes a 2D Shapely polygon and converts it to 3D representation with
        fan triangulation for face generation.

        Args:
            polygon: 2D Shapely polygon
            z_value: Z coordinate value to assign to all vertices

        Returns:
            Tuple of (3D vertices array, face indices list)
        """
        # Get exterior boundary coordinates
        coords = list(polygon.exterior.coords[:-1])  # Remove duplicate last point

        # Convert to 3D vertices
        vertices_3d = np.array([[x, y, z_value] for x, y in coords])

        # Create faces (fan triangulation)
        faces = []
        n = len(vertices_3d)

        if n >= 3:
            for i in range(1, n - 1):
                faces.append([0, i, i + 1])

        return vertices_3d, faces

    def _post_process_results(
        self, results: List[Tuple[np.ndarray, List[List[int]]]]
    ) -> List[Tuple[np.ndarray, List[List[int]]]]:
        """
        Post-process decomposition results.

        Filters out polygons that are too small and logs the processing summary.
        Returns original polygon if no valid results remain.

        Args:
            results: List of decomposed polygons

        Returns:
            Filtered list of valid polygons
        """
        if not results:
            return [(self.vertices, self.faces)]

        # Filter out polygons that are too small
        filtered_results = []
        config = self.get_config()
        for vertices, faces in results:
            if len(vertices) >= 3:
                area = self._polygon_area_2d(vertices[:, :2])
                if area > config.min_polygon_area:
                    filtered_results.append((vertices, faces))

        glog.info(
            f"Post-processing complete: {len(results)} -> {len(filtered_results)} polygons"
        )

        return filtered_results if filtered_results else [(self.vertices, self.faces)]

    def _polygon_area_2d(self, points: np.ndarray) -> float:
        """
        Calculate 2D polygon area using the shoelace formula.

        Computes the area of a polygon defined by 2D points using the
        shoelace (surveyor's) formula.

        Args:
            points: 2D points defining the polygon

        Returns:
            Area of the polygon
        """
        if len(points) < 3:
            return 0.0

        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * abs(
            sum(
                x[i] * y[(i + 1) % len(x)] - x[(i + 1) % len(x)] * y[i]
                for i in range(len(x))
            )
        )


# Usage example
def example_usage():
    """
    Usage example demonstrating concave polygon decomposition.

    Creates a complex concave polygon (L-shaped) and demonstrates
    different decomposition strategies, comparing the original
    rectangular decomposition with the new merging strategy.
    """

    # Create a complex concave polygon (L-shaped)
    vertices = np.array(
        [
            [0, 0, 0],  # 0
            [3, 0, 0],  # 1
            [3, 1, 0],  # 2
            [1, 1, 0],  # 3 (concave point)
            [1, 3, 0],  # 4
            [0, 3, 0],  # 5
        ]
    )

    # Simple faces
    faces = [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]]

    # Decompose concave polygon - using new merging strategy
    processor = ConcaveProcessor(vertices, faces)

    print("=== Comparing Different Strategies ===")

    # Original rectangular decomposition
    results_original = processor.decompose(strategy="rectangular")
    print(f"Original rectangular decomposition: {len(results_original)} polygons")

    # New merging strategy
    results_merged = processor.decompose(strategy="rectangular_with_merge")
    print(f"Merged results: {len(results_merged)} polygons")

    print(f"Reduced by {len(results_original) - len(results_merged)} polygons")


if __name__ == "__main__":
    example_usage()
