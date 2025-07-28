import sapien as sapien
from sapien.utils import Viewer
import numpy as np
import json
import transforms3d
import os
from pathlib import Path
import os.path as osp
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Set, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.config_manager import get_sapien_config

from pathlib import Path
import numpy as np
import glog


class URDFProcessor:
    """Utilities for URDF processing"""

    @classmethod
    def get_config(cls):

        return get_sapien_config()

    @staticmethod
    def urdf_to_dict(urdf_path: str) -> Dict[str, Any]:
        """Convert URDF file to dictionary structure"""
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        def parse_element(element):
            parsed = {}
            for child in element:
                if len(child) > 0:
                    name = f"_{child.attrib['name']}" if "name" in child.attrib else ""
                    parsed[f"{child.tag}{name}"] = parse_element(child)
                else:
                    parsed[child.tag] = child.attrib
            return parsed

        urdf_dict = {root.tag: parse_element(root)}
        return urdf_dict


class SapienSceneManager:
    """Manager for Sapien scene creation and manipulation"""

    def __init__(self):
        self.fixed_objects: Set[str] = set()
        self.config = get_sapien_config()

    @classmethod
    def get_config(cls):

        return get_sapien_config()

    def create_scene(self) -> sapien.Scene:
        """Create and configure a new Sapien scene"""
        scene = sapien.Scene()
        scene.set_timestep(self.config.time_step)
        scene.add_ground(altitude=self.config.ground_altitude)

        directional_light = self.config.directional_light
        point_lights = self.config.point_lights

        camera_shader = self.config.camera_shader
        ray_tracing_denoiser = self.config.ray_tracing_denoiser

        for dir_light in directional_light:
            scene.add_directional_light(dir_light["direction"], dir_light["color"])
        for point_light in point_lights:
            scene.add_point_light(point_light["position"], point_light["color"])
        sapien.render.set_camera_shader_dir(camera_shader)
        sapien.render.set_ray_tracing_denoiser(ray_tracing_denoiser)

        # scene.set_timestep(time_step)

        return scene

    def setup_lighting(self, scene: sapien.Scene):
        """Setup lighting for the scene"""
        # Ambient light
        scene.set_ambient_light(self.config.ambient_light)

        # Directional light
        scene.add_directional_light(
            self.config.directional_light["direction"],
            self.config.directional_light["color"],
        )

        # Point lights
        for light in self.config.point_lights:
            scene.add_point_light(light["position"], light["color"])

    def setup_camera(self, viewer):
        """Setup camera for the viewer"""
        viewer.set_camera_xyz(**self.config.camera_position)
        viewer.set_camera_rpy(**self.config.camera_rotation)
        viewer.window.set_camera_parameters(**self.config.camera_parameters)

    def get_glb_path(self, template_name: str) -> str:
        """Get GLB file path for a template"""
        obj_config_path = (
            Path(self.config.dataset_root_path)
            / "configs/objects"
            / f"{osp.basename(template_name)}.object_config.json"
        )

        with open(obj_config_path, "r") as f:
            obj_config = json.load(f)

        relative_glb_path = obj_config["render_asset"]
        glb_file_path = os.path.normpath(obj_config_path.parent / relative_glb_path)
        return glb_file_path

    def _create_object_material(
        self, use_default: bool = False
    ) -> sapien.render.RenderMaterial:
        """Create material for objects"""
        material = sapien.render.RenderMaterial()

        if use_default:
            material.set_base_color(self.config.default_material["base_color"])
            material.set_metallic(self.config.default_material["metallic"])
            material.set_roughness(self.config.default_material["roughness"])
            material.set_specular(self.config.default_material["specular"])
            material.set_transmission(self.config.default_material["transmission"])
        else:
            material.transmission = 1e9
            material.base_color = [1.0, 1.0, 1.0, 1.0]

        return material

    def _apply_rotation_correction(self, quaternion: List[float]) -> List[float]:
        """Apply rotation correction to quaternion"""
        rpy = transforms3d.euler.quat2euler(quaternion, axes="sxyz")
        corrected_quaternion = transforms3d.euler.euler2quat(
            rpy[0] + np.deg2rad(self.config.rotation_offset_x),
            rpy[1],
            rpy[2],
            axes="sxyz",
        )
        return corrected_quaternion

    def add_object_to_scene(self, scene: sapien.Scene, obj: Dict[str, Any]):
        """Add a single object to the scene"""
        object_file_path = obj["visual_path"]
        collision_path = obj.get("collision_path", None)
        position = [
            obj["centroid_translation"]["x"],
            obj["centroid_translation"]["y"],
            obj["centroid_translation"]["z"],
        ]

        quaternion = [
            obj["quaternion"]["w"],
            obj["quaternion"]["x"],
            obj["quaternion"]["y"],
            obj["quaternion"]["z"],
        ]

        # Special case adjustments
        if "cushion_03" in object_file_path:
            position[2] += self.config.cushion_z_offset

        # Rotation correction
        quaternion = self._apply_rotation_correction(quaternion)

        # Create actor
        builder = scene.create_actor_builder()
        material = self._create_object_material(use_default=False)
        builder.add_visual_from_file(filename=object_file_path)

        # Add collision
        if collision_path is not None:
            builder.add_multiple_convex_collisions_from_file(filename=collision_path)
        else:
            builder.add_convex_collision_from_file(filename=object_file_path)

        # Build actor
        if obj.get("motion_type") in ["STATIC", "KEEP_FIXED"]:
            mesh = builder.build_static(name=obj["name"])
        else:
            mesh = builder.build(name=obj["name"])

        mesh.set_pose(sapien.Pose(p=position, q=quaternion))

    def add_articulation_to_scene(
        self, scene: sapien.Scene, articulated_meta: Dict[str, Any]
    ):
        """Add an articulated object to the scene"""
        template_name = articulated_meta["name"]
        pos = articulated_meta["translation"]
        pos = [pos["x"], pos["y"], pos["z"]]

        rot = articulated_meta["rotation"]
        rot = [rot["w"], rot["x"], rot["y"], rot["z"]]

        # Rotation correction
        quaternion = self._apply_rotation_correction(rot)

        # URDF loading
        urdf_path = articulated_meta["urdf_path"]
        urdf_loader = scene.create_urdf_loader()
        urdf_loader.name = f"{template_name}"
        urdf_loader.fix_root_link = articulated_meta["fixed_base"]
        urdf_loader.disable_self_collisions = True

        if "uniform_scale" in articulated_meta:
            urdf_loader.scale = urdf_loader.uniform_scale = articulated_meta[
                "uniform_scale"
            ]

        # Position adjustment based on URDF
        base_name = template_name[: template_name.rfind("_")]
        urdf_adjustment = {
            "fridge": 0.022461603782394812,
            "kitchen_counter": 0.02441653738006261,
            "kitchenCupboard_01": 0,
            "chestOfDrawers_01": -0.003584124570854732,
            "cabinet": 0.019934424604485024,
        }

        if base_name in urdf_adjustment:
            pos[2] -= urdf_adjustment[base_name] - self.config.urdf_z_adjustment_offset

        # Build articulation
        builder = urdf_loader.parse(urdf_path)[0][0]
        pose = sapien.Pose(pos, quaternion)
        builder.initial_pose = pose
        articulation = builder.build()

    def load_objects_from_json(self, scene: sapien.Scene, json_file_path: str):
        """Load objects from JSON file into the scene"""
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Load background
        q = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(self.config.rotation_offset_x)
        )
        bg_pose = sapien.Pose(q=q)
        bg_path = data["background_file_path"]

        builder = scene.create_actor_builder()
        builder.add_visual_from_file(bg_path)
        builder.add_nonconvex_collision_from_file(bg_path)

        bg = builder.build_static(name="scene_background")
        bg.set_pose(bg_pose)

        # Load objects
        for obj in data["object_instances"]:
            if "name" in obj and obj["name"] == "GROUND":
                continue
            if "motion_type" in obj and obj["motion_type"] == "KEEP_FIXED":
                self.fixed_objects.add(obj["name"])

            self.add_object_to_scene(scene, obj)

        # Load articulated objects
        for articulated_meta in data.get("articulate_instances", []):
            self.add_articulation_to_scene(scene, articulated_meta)

    def apply_entity_shading(self, scene: sapien.Scene):
        """Apply special shading to entities"""
        for entity in scene.entities:
            if any(
                excluded in entity.get_name()
                for excluded in self.config.excluded_shading_objects
            ):
                continue
            for component in entity.get_components():
                if isinstance(component, sapien.pysapien.render.RenderBodyComponent):
                    component.shading_mode = self.config.default_shading_mode

    def remove_objects(self, scene: sapien.Scene, object_names: List[str]):
        """Remove objects from scene by names"""
        for entity in scene.entities:
            if entity.get_name() in object_names:
                entity.remove_from_scene()

    def add_single_object(self, scene: sapien.Scene, obj: Dict[str, Any]):
        """Add a single object with custom material (for debugging)"""
        object_file_path = obj["visual_path"]
        collision_path = obj.get("collision_path", None)
        position = [
            obj["centroid_translation"]["x"],
            obj["centroid_translation"]["y"],
            obj["centroid_translation"]["z"],
        ]

        quaternion = [
            obj["quaternion"]["w"],
            obj["quaternion"]["x"],
            obj["quaternion"]["y"],
            obj["quaternion"]["z"],
        ]

        if "cushion_03" in object_file_path:
            position[2] += self.config.cushion_z_offset

        quaternion = self._apply_rotation_correction(quaternion)

        builder = scene.create_actor_builder()
        material = self._create_object_material(use_default=True)

        builder.add_visual_from_file(filename=object_file_path, material=material)
        if collision_path is not None:
            builder.add_multiple_convex_collisions_from_file(filename=collision_path)
        else:
            builder.add_convex_collision_from_file(filename=object_file_path)

        if obj["motion_type"] in ["STATIC", "KEEP_FIXED"]:
            mesh = builder.build_static(name=obj["name"])
        else:
            mesh = builder.build(name=obj["name"])
        mesh.set_pose(sapien.Pose(p=position, q=quaternion))


class EntityExporter:
    """Export scene entities to JSON format"""

    def __init__(self):
        self.urdf_processor = URDFProcessor()
        self.config = get_sapien_config()

    @classmethod
    def get_config(cls):

        return get_sapien_config()

    @staticmethod
    def convert_to_float(obj):
        """Convert numpy types to Python float for JSON serialization"""
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: EntityExporter.convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [EntityExporter.convert_to_float(i) for i in obj]
        return obj

    def reget_entities_from_sapien(
        self,
        scene,
        json_file_path: str,
        path: str = "entities.json",
        fixed_objects: Set[str] = None,
    ):
        """Export entities from Sapien scene to JSON format"""
        if fixed_objects is None:
            fixed_objects = set()

        # Load original data
        with open(json_file_path, "r") as f:
            data = json.load(f)

        res = {
            "background_file_path": data["background_file_path"],
            "object_instances": [],
        }

        articulations = data.get("articulate_instances", [])
        articulation_idx = -1

        # Process entities
        for entity in scene.entities:
            entity_instance = {}
            print(f"Processing entity: {entity.get_name()}")
            if len(entity.get_name()) == 0:

                continue
            if entity.get_name() == "root":
                articulation_idx += 1
                continue

            if articulation_idx == -1:
                # Regular object
                entity_instance["name"] = entity.get_name()
                filename = entity.get_name()

                if filename in fixed_objects or "ground" in filename:
                    continue

                filename = filename[: filename.rfind("_")]
                entity_instance["visual_path"] = (
                    f"{self.config.visual_path_prefix}{filename}.glb"
                )

            elif articulation_idx < len(articulations):
                if "body" not in entity.get_name():
                    glog.info(
                        f"Skipping entity {entity.get_name()} as it is not body part of an articulated object"
                    )
                    continue
                # Articulated object
                entity_instance["name"] = (
                    f'{articulations[articulation_idx]["name"]}_{entity.get_name()}'
                )
                urdf_path = articulations[articulation_idx]["urdf_path"]
                urdf_dict = self.urdf_processor.urdf_to_dict(urdf_path)

                if f"link_{entity.get_name()}" not in urdf_dict["robot"]:
                    print(f"Warning: {entity.get_name()} not found in URDF")
                    continue

                filename = urdf_dict["robot"][f"link_{entity.get_name()}"]["visual"][
                    "geometry"
                ]["mesh"]["filename"]
                name = articulations[articulation_idx]["name"]
                name = name[: name.rfind("_")]
                entity_instance["visual_path"] = (
                    f"{self.config.urdf_path_prefix}/{name}/{filename}"
                )
            else:
                print("Error: articulation index out of range")
                continue

            # Extract pose information
            quaternion = [self.convert_to_float(val) for val in entity.get_pose().q]
            rpy = transforms3d.euler.quat2euler(quaternion, axes="sxyz")
            quaternion = transforms3d.euler.euler2quat(
                rpy[0] + np.deg2rad(self.config.correction_rotation_x),
                rpy[1],
                rpy[2],
                axes="sxyz",
            )

            q = {
                "w": self.convert_to_float(quaternion[0]),
                "x": self.convert_to_float(quaternion[1]),
                "y": self.convert_to_float(quaternion[2]),
                "z": self.convert_to_float(quaternion[3]),
            }

            p = {
                "x": self.convert_to_float(entity.get_pose().p[0]),
                "y": self.convert_to_float(entity.get_pose().p[1]),
                "z": self.convert_to_float(entity.get_pose().p[2]),
            }

            entity_instance["centroid_translation"] = p
            entity_instance["quaternion"] = q
            entity_instance["bbox"] = "deprecated"
            res["object_instances"].append(entity_instance)

        # Merge original object properties
        for obj in data.get("object_instances", []):
            for obj_idx in range(len(res["object_instances"])):
                if res["object_instances"][obj_idx]["name"] == obj["name"]:
                    res["object_instances"][obj_idx]["motion_type"] = obj["motion_type"]
                    res["object_instances"][obj_idx]["collision_path"] = obj.get(
                        "collision_path", None
                    )
                    res["object_instances"][obj_idx]["visual_path"] = obj["visual_path"]
                    break

        # Save result
        with open(path, "w") as f:
            json.dump(res, f, indent=4)
        # Initialize res


def main():
    parser = argparse.ArgumentParser(description="Export Sapien scene entities to JSON")
    parser.add_argument(
        "--json_file_path", type=str, required=True, help="Path to the JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="entities.json",
        help="Output path for exported entities",
    )
    args = parser.parse_args()

    exporter = EntityExporter()
    scene = sapien.Scene()

    sapien_scene_manager = SapienSceneManager()
    sapien_scene_manager.load_objects_from_json(scene, args.json_file_path)

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-5, y=0, z=6)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    scene.add_point_light([1.989, -5.822, 1], [0.5, 0.5, 0.5])
    """
    scene.add_point_light([1.2931, -5.7490, 1.0273],[1, 1, 1])
    scene.add_point_light([2.2649, -6.4652, 1.0273],[1, 1, 1])
    scene.add_point_light([2.6857, -5.8942, 1.0273],[1, 1, 1])
    scene.add_point_light([1.7139, -5.1780, 1.0273],[1, 1, 1])
    """
    scene.add_point_light([2 + 0.15, -5.75 - 0.15, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([1.2 + 0.15, -5.75 - 0.15, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([1.6 + 0.15, -6.5 - 0.15, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([1.6 + 0.15, -5 - 0.15, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([2 + 0.15, -6.5 - 0.15, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([2 + 0.15, -5.0 - 0.15, 1.0], [0.4, 0.4, 0.4])

    scene.add_point_light([3 - 0.1, -7.35, 2.0], [0.4, 0.4, 0.4])
    scene.add_point_light([3 + 0.1, -7.85, 2.0], [0.4, 0.4, 0.4])
    #  scene.add_point_light([1.2,-6.5,1.5],[2,2,2])
    scene.add_point_light([1.2 + 0.15, -5 - 0.15, 1.0], [0.4, 0.4, 0.4])

    print(scene.entities, dir(scene), dir(scene.entities[0]))
    for entity in scene.entities:
        print(entity.get_name(), entity.get_pose(), entity.get_components())

    # start simulation
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()

    exporter.reget_entities_from_sapien(scene, args.json_file_path, args.output_path)


if __name__ == "__main__":
    main()
