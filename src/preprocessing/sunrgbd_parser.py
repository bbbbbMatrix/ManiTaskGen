"""
SUNRGBD Dataset Parser
Handles parsing and processing of SUNRGBD dataset including RGB-D images, 
camera parameters, and 3D bounding box annotations.
"""

import cv2
import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.transform import Rotation
import logging


from .base_parser import BaseRawSceneParser, RawSceneParserFactory

logger = logging.getLogger(__name__)


class SUNRGBDParser(BaseRawSceneParser):
    """Parser for SUNRGBD dataset"""

    def __init__(self, config=None):
        """
        Initialize SUNRGBD parser

        Args:
            config: Configuration object containing dataset paths and settings
        """
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "dataset_root_path": "/mnt/windows_e/workplace/SUNRGBD",
            "output_coordinate_system": "standard",  # "standard" or "sunrgbd"
            "height_offset": 1.3,  # Height adjustment offset
            "overlap_threshold": 0.2,  # Height difference threshold for overlap detection
        }

    def parse_scene(self, scene_path: str, output_path: str) -> Dict[str, Any]:
        """
        Parse a SUNRGBD scene and save to output file

        Args:
            scene_path: Path to the scene directory containing RGB, depth, and annotation files
            output_path: Output JSON file path

        Returns:
            Parsed scene data dictionary
        """
        scene_path = Path(scene_path)

        # Load scene components
        rgb_image = self._load_rgb_image(scene_path)
        depth_image = self._load_depth_image(scene_path)
        intrinsics = self._load_camera_intrinsics(scene_path)
        extrinsics = self._load_camera_extrinsics(scene_path)
        scene_info = self._load_scene_info(scene_path)
        objects = self._load_3d_annotations(scene_path)

        # Process objects
        processed_objects = []
        for obj_data in objects:
            if obj_data is None or "polygon" not in obj_data:
                continue

            processed_obj = self._process_object(obj_data)
            if processed_obj is not None:
                processed_objects.append(processed_obj)

        # Detect overlaps
        overlaps = self._detect_object_overlaps(processed_objects)

        # Create scene data structure
        scene_data = {
            "scene_info": {
                "rgb_image_shape": rgb_image.shape if rgb_image is not None else None,
                "depth_image_shape": (
                    depth_image.shape if depth_image is not None else None
                ),
                "camera_intrinsics": (
                    intrinsics.tolist() if intrinsics is not None else None
                ),
                "camera_extrinsics": (
                    extrinsics.tolist() if extrinsics is not None else None
                ),
                "scene_metadata": scene_info,
                "object_overlaps": overlaps,
            },
            "object_instances": self._convert_to_template_format(processed_objects),
            "articulate_instances": [],  # SUNRGBD doesn't have articulated objects
        }

        # Save to output file
        with open(output_path, "w") as f:
            json.dump(scene_data, f, indent=2)

        logger.info(f"Parsed SUNRGBD scene with {len(processed_objects)} objects")
        return scene_data

    def _load_rgb_image(self, scene_path: Path) -> Optional[np.ndarray]:
        """Load RGB image"""
        rgb_files = list(scene_path.glob("fullres/*.jpg"))
        if not rgb_files:
            logger.warning(f"No RGB image found in {scene_path}")
            return None

        rgb_path = rgb_files[0]
        return cv2.imread(str(rgb_path))

    def _load_depth_image(self, scene_path: Path) -> Optional[np.ndarray]:
        """Load depth image"""
        depth_files = list(scene_path.glob("depth/*.png"))
        if not depth_files:
            logger.warning(f"No depth image found in {scene_path}")
            return None

        depth_path = depth_files[0]
        return cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    def _load_camera_intrinsics(self, scene_path: Path) -> Optional[np.ndarray]:
        """Load camera intrinsic parameters"""
        intrinsics_path = scene_path / "intrinsics.txt"
        if not intrinsics_path.exists():
            logger.warning(f"Intrinsics file not found: {intrinsics_path}")
            return None

        return np.loadtxt(intrinsics_path)

    def _load_camera_extrinsics(self, scene_path: Path) -> Optional[np.ndarray]:
        """Load camera extrinsic parameters"""
        extrinsics_dir = scene_path / "extrinsics"
        if not extrinsics_dir.exists():
            logger.warning(f"Extrinsics directory not found: {extrinsics_dir}")
            return None

        extrinsics_files = list(extrinsics_dir.glob("*.txt"))
        if not extrinsics_files:
            logger.warning(f"No extrinsics file found in {extrinsics_dir}")
            return None

        return np.loadtxt(extrinsics_files[0])

    def _load_scene_info(self, scene_path: Path) -> Optional[str]:
        """Load scene information"""
        scene_file = scene_path / "scene.txt"
        if not scene_file.exists():
            logger.warning(f"Scene file not found: {scene_file}")
            return None

        with open(scene_file, "r") as f:
            return f.read().strip()

    def _load_3d_annotations(self, scene_path: Path) -> List[Dict[str, Any]]:
        """Load 3D bounding box annotations"""
        annotation_file = scene_path / "annotation3Dfinal" / "index.json"
        if not annotation_file.exists():
            logger.warning(f"Annotation file not found: {annotation_file}")
            return []

        with open(annotation_file, "r") as f:
            data = json.load(f)

        return data.get("objects", [])

    def _process_object(self, obj_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single object annotation"""
        try:
            bbox_data = obj_data["polygon"][0]

            # Fix Y coordinate order if needed
            if bbox_data["Ymin"] > bbox_data["Ymax"]:
                bbox_data["Ymin"], bbox_data["Ymax"] = (
                    bbox_data["Ymax"],
                    bbox_data["Ymin"],
                )

            # Convert to standard format
            result = self._convert_bbox_to_standard(bbox_data)
            result["name"] = obj_data["name"]
            result["original_data"] = obj_data

            return result

        except Exception as e:
            logger.error(
                f"Error processing object {obj_data.get('name', 'unknown')}: {e}"
            )
            return None

    def _convert_bbox_to_standard(self, bbox_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SUNRGBD bounding box format to standard representation"""
        # Extract coordinates
        x_coords = np.array(bbox_data["X"])
        z_coords = np.array(bbox_data["Z"])
        y_min = bbox_data["Ymin"]
        y_max = bbox_data["Ymax"]

        # Calculate 8 corners of the bounding box
        bottom_corners = np.zeros((4, 3))
        for i in range(4):
            bottom_corners[i] = [x_coords[i], y_min, z_coords[i]]

        top_corners = bottom_corners.copy()
        top_corners[:, 1] = y_max

        corners = np.vstack((bottom_corners, top_corners))

        # Calculate centroid
        centroid = np.mean(corners, axis=0)

        # Calculate orientation
        edge1 = bottom_corners[1] - bottom_corners[0]
        edge2 = bottom_corners[3] - bottom_corners[0]

        # Normalize edges
        edge1 = edge1 / np.linalg.norm(edge1)
        edge2 = edge2 / np.linalg.norm(edge2)

        # Create rotation matrix
        up_direction = np.array([0, 1, 0])
        rotation_matrix = np.column_stack((edge1, up_direction, edge2))

        # Convert to quaternion
        try:
            r = Rotation.from_matrix(rotation_matrix)
            quaternion = r.as_quat()  # [x, y, z, w] format
        except:
            quaternion = np.array([0, 0, 0, 1])  # Identity quaternion

        # Calculate dimensions
        width = np.linalg.norm(bottom_corners[1] - bottom_corners[0])
        height = y_max - y_min
        depth = np.linalg.norm(bottom_corners[3] - bottom_corners[0])

        dimensions = np.array([width, height, depth])

        return {
            "centroid": centroid,
            "quaternion": quaternion,
            "dimensions": dimensions,
            "corners": corners,
        }

    def _detect_object_overlaps(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect overlapping objects"""
        overlaps = []
        threshold = self.config.get("overlap_threshold", 0.2)

        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects[i + 1 :], i + 1):
                # Check height difference
                height_diff = obj_a["corners"][0][1] - obj_b["corners"][4][1]

                if abs(height_diff) < threshold:
                    # Check 2D overlap
                    a_bottom = [point[0::2] for point in obj_a["corners"][:4]]
                    b_bottom = [point[0::2] for point in obj_b["corners"][:4]]

                    if self._check_rectangles_intersect(a_bottom, b_bottom):
                        overlaps.append(
                            {
                                "object_a": obj_a["name"],
                                "object_b": obj_b["name"],
                                "height_difference": float(height_diff),
                                "overlap_type": "spatial",
                            }
                        )

        return overlaps

    def _check_rectangles_intersect(
        self, rect1: List[List[float]], rect2: List[List[float]]
    ) -> bool:
        """Check if two rectangles intersect"""

        def point_in_rectangle(point, rect):
            p1, p2, p3, p4 = [np.array(p) for p in rect]
            point = np.array(point)

            cross1 = np.cross(p2 - p1, point - p1)
            cross2 = np.cross(p3 - p2, point - p2)
            cross3 = np.cross(p4 - p3, point - p3)
            cross4 = np.cross(p1 - p4, point - p4)

            return (
                cross1 >= -1e-6
                and cross2 >= -1e-6
                and cross3 >= -1e-6
                and cross4 >= -1e-6
            ) or (
                cross1 <= 1e-6 and cross2 <= 1e-6 and cross3 <= 1e-6 and cross4 <= 1e-6
            )

        # Check if any vertex of one rectangle is inside the other
        for point in rect1:
            if point_in_rectangle(point, rect2):
                return True

        for point in rect2:
            if point_in_rectangle(point, rect1):
                return True

        return False

    def _convert_to_template_format(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert objects to template format"""
        template_objects = []

        for obj in objects:
            centroid = obj["centroid"]
            quaternion = obj["quaternion"]
            dimensions = obj["dimensions"]

            # Coordinate system conversion
            if self.config.get("output_coordinate_system") == "standard":
                x = centroid[0]
                y = centroid[2]
                z = -centroid[1] + self.config.get("height_offset", 1.3)

                # Convert quaternion
                quat_x = quaternion[0]
                quat_y = quaternion[2]
                quat_z = -quaternion[1]
                quat_w = quaternion[3]
            else:
                # Keep original coordinates
                x, y, z = centroid
                quat_x, quat_y, quat_z, quat_w = quaternion

            template_obj = {
                "name": f"{obj['name']}_{len(template_objects)}",
                "template_name": f"objects/{obj['name']}",
                "motion_type": "DYNAMIC",
                "visual_path": f"/path/to/assets/objects/{obj['name']}_modified.glb",
                "collision_path": f"/path/to/assets/objects/{obj['name']}_modified.glb",
                "centroid_translation": {"x": float(x), "y": float(y), "z": float(z)},
                "quaternion": {
                    "w": float(quat_w),
                    "x": float(quat_x),
                    "y": float(quat_y),
                    "z": float(quat_z),
                },
                "bbox": {
                    "x_length": float(dimensions[0]),
                    "y_length": float(dimensions[1]),
                    "z_length": float(dimensions[2]),
                },
            }

            template_objects.append(template_obj)

        return template_objects


RawSceneParserFactory.register_parser("sunrgbd", SUNRGBDParser)


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Parse SUNRGBD scene data")
    parser.add_argument(
        "--scene_path", type=str, required=True, help="Path to SUNRGBD scene directory"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output JSON file path"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Configuration file path"
    )

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

    # Parse scene
    parser_instance = SUNRGBDParser(config)
    result = parser_instance.parse_scene(args.scene_path, args.output_path)

    print(f"Successfully parsed scene with {len(result['object_instances'])} objects")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
