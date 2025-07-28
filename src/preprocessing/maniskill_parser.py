"""
Replica Dataset Parser
Handles parsing and conversion of Replica dataset scene files
"""

import json
import numpy as np
import transforms3d
import os
import sys
from pathlib import Path
import os.path as osp
import trimesh
from scipy.spatial.transform import Rotation as R
import glog
from typing import Dict, Any, Optional

from .base_parser import BaseRawSceneParser, RawSceneParserFactory


class ReplicaSceneParser(BaseRawSceneParser):
    """Replica dataset parser for converting scene files"""

    def __init__(self, config=None):
        """Initialize Replica parser"""
        from ..utils.config_manager import get_raw_scene_config

        self.config = config or get_raw_scene_config()

    @classmethod
    def get_config(cls):

        from ..utils.config_manager import get_raw_scene_config

        return get_raw_scene_config()

    def get_glb_path(self, template_name: str) -> str:

        obj_config_path = (
            Path(self.config.dataset_root_path)
            / f"configs/objects/{template_name}.object_config.json"
        )

        with open(obj_config_path, "r") as f:
            obj_config = json.load(f)

        relative_glb_path = obj_config["render_asset"]
        glb_file_path = os.path.normpath(obj_config_path.parent / relative_glb_path)
        return glb_file_path

        obj_config_path = (
            Path(self.config.dataset_root_path)
            / f"configs/objects/{template_name}.object_config.json"
        )

        with open(obj_config_path, "r") as f:
            obj_config = json.load(f)

        if obj_config.get("collision_asset"):
            relative_collision_path = obj_config["collision_asset"]
            collision_file_path = os.path.normpath(
                obj_config_path.parent / relative_collision_path
            )
            return collision_file_path
        else:
            assert (
                obj_config.get("use_bounding_box_for_collision")
                and obj_config["use_bounding_box_for_collision"]
            )
            return None

    def get_urdf_path(self, template_name: str) -> str:

        base_name = osp.basename(template_name)
        urdf_path = (
            Path(self.config.dataset_root_path)
            / "urdf"
            / f"{base_name}/{base_name}.urdf"
        )

        relative_urdf_path = f"../../urdf/{base_name}/{base_name}.urdf"
        urdf_path = os.path.normpath(urdf_path.parent / relative_urdf_path)
        return urdf_path

    def calculate_bbox(self, glb_path: str) -> Dict[str, float]:

        mesh = trimesh.load(glb_path, force="scene")
        bbox_min, bbox_max = mesh.bounds
        bbox_size = bbox_max - bbox_min

        return {
            "x_length": bbox_size[0],
            "y_length": bbox_size[2],
            "z_length": bbox_size[1],
        }

    def _process_object_instance(
        self, obj: Dict[str, Any], obj_idx: int
    ) -> Optional[Dict[str, Any]]:

        name = obj["template_name"].split("/")[-1]

        desired_objects = self.config.desired_objects or []
        not_desired_objects = self.config.not_desired_objects or []

        if len(desired_objects) and name not in desired_objects:
            return None

        if len(not_desired_objects) and name in not_desired_objects:
            return None

        glb_path = self.get_glb_path(name)
        collision_path = self.get_collision_path(name)
        glb = trimesh.load(glb_path)
        geometries = list(glb.geometry.values())

        node_name = next(key for key in glb.graph.nodes if key != "world")

        transformed_vertices = trimesh.transform_points(
            geometries[0].vertices, glb.graph[node_name][0]
        )
        bbox_min, bbox_max = np.min(transformed_vertices, axis=0), np.max(
            transformed_vertices, axis=0
        )

        bbox = bbox_max - bbox_min
        bbox = {"x_length": bbox[0], "y_length": bbox[2], "z_length": bbox[1]}

        motion_type = obj["motion_type"]
        if len(desired_objects) and name not in desired_objects:
            motion_type = "KEEP_FIXED"
        if len(not_desired_objects) and name in not_desired_objects:
            motion_type = "KEEP_FIXED"

        q_offset = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )

        translation = obj["translation"]
        rotation = obj["rotation"]

        corrected_rotation = transforms3d.quaternions.qmult(q_offset, rotation)
        rpy = transforms3d.euler.quat2euler(corrected_rotation, axes="sxyz")
        corrected_rotation = transforms3d.euler.euler2quat(
            rpy[0] - np.deg2rad(90), rpy[1], rpy[2], axes="sxyz"
        )

        return {
            "name": f"{name}_{obj_idx}",
            "motion_type": motion_type,
            "visual_path": glb_path,
            "collision_path": collision_path,
            "centroid_translation": {
                "x": translation[0],
                "y": -translation[2],
                "z": translation[1],
            },
            "quaternion": {
                "w": corrected_rotation[0],
                "x": corrected_rotation[1],
                "y": corrected_rotation[2],
                "z": corrected_rotation[3],
            },
            "bbox": bbox,
        }

    def _process_articulated_instance(
        self, articulated_meta: Dict[str, Any], articulate_idx: int
    ) -> Optional[Dict[str, Any]]:

        template_name = articulated_meta["template_name"]

        desired_articulations = self.config.desired_articulations or []
        not_desired_articulations = self.config.not_desired_articulations or []

        if len(desired_articulations) and template_name not in desired_articulations:
            return None
        if (
            len(not_desired_articulations)
            and template_name in not_desired_articulations
        ):
            return None

        pos = articulated_meta["translation"]
        rotation = articulated_meta["rotation"]
        fixed_base = articulated_meta.get("fixed_base", False)
        uniform_scale = articulated_meta.get("uniform_scale", 1.0)

        urdf_path = self.get_urdf_path(template_name)

        q_offset = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )
        corrected_rotation = transforms3d.quaternions.qmult(q_offset, rotation)
        rpy = transforms3d.euler.quat2euler(corrected_rotation, axes="sxyz")
        corrected_rotation = transforms3d.euler.euler2quat(
            rpy[0] - np.deg2rad(90), rpy[1], rpy[2], axes="sxyz"
        )

        return {
            "name": f"{template_name}_{articulate_idx}",
            "translation": {"x": pos[0], "y": -pos[2], "z": pos[1]},
            "urdf_path": urdf_path,
            "rotation": {
                "w": corrected_rotation[0],
                "x": corrected_rotation[1],
                "y": corrected_rotation[2],
                "z": corrected_rotation[3],
            },
            "fixed_base": fixed_base,
            "uniform_scale": uniform_scale,
        }

    def parse_scene(
        self, input_json_path: str, output_json_path: str
    ) -> Dict[str, Any]:

        with open(input_json_path, "r") as f:
            data = json.load(f)

        background_template_name = data["stage_instance"]["template_name"].split("/")[
            -1
        ]
        bg_path = osp.join(
            self.config.dataset_root_path, f"stages/{background_template_name}.glb"
        )

        output_data = {
            "background_file_path": bg_path,
            "object_instances": [],
            "articulate_instances": [],
        }

        obj_idx = 0
        for obj in data.get("object_instances", []):

            processed_obj = self._process_object_instance(obj, obj_idx)
            name = obj["template_name"].split("/")[-1]
            if processed_obj:
                output_data["object_instances"].append(processed_obj)
                obj_idx += 1

        articulate_idx = 0
        for articulated_meta in data.get("articulated_object_instances", []):
            processed_articulate = self._process_articulated_instance(
                articulated_meta, articulate_idx
            )
            if processed_articulate:
                output_data["articulate_instances"].append(processed_articulate)
                articulate_idx += 1

        if os.path.exists(output_json_path):
            os.remove(output_json_path)

        with open(output_json_path, "w") as f:
            json.dump(output_data, f, indent=4)

        glog.info(
            f"Replica scene parsed: {len(output_data['object_instances'])} objects, {len(output_data['articulate_instances'])} articulated objects"
        )
        glog.info(f"Output saved to: {output_json_path}")

        return output_data


RawSceneParserFactory.register_parser("replica", ReplicaSceneParser)


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Parse Replica dataset scene")
    parser.add_argument(
        "--input_json_path",
        type=str,
        required=True,
        help="Input Replica scene JSON file path",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        required=True,
        help="Output parsed scene JSON file path",
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
    replica_parser = ReplicaSceneParser(config)
    result = replica_parser.parse_scene(args.input_json_path, args.output_json_path)

    print(f"Successfully parsed Replica scene")
    print(f"Objects: {len(result['object_instances'])}")
    print(f"Articulated objects: {len(result['articulate_instances'])}")


if __name__ == "__main__":
    main()
