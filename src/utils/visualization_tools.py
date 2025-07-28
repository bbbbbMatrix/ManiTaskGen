# utils/visualization_tools.py
"""
Visualization tools for scene graph trees and tasks
"""

import os
import json
import graphviz
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import glog


class SceneGraphVisualizer:
    """Scene graph tree visualization tool"""

    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer

        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_tree_to_dot(
        self,
        tree_root,
        output_file: str = "scene_graph",
        format: str = "png",
        include_properties: bool = True,
        max_depth: Optional[int] = None,
    ) -> str:
        """
        Export scene graph tree to DOT format and render

        Args:
            tree_root: Root node of the tree (should have 'children' attribute)
            output_file: Output file name (without extension)
            format: Output format ('png', 'pdf', 'svg', etc.)
            include_properties: Whether to include node properties in visualization
            max_depth: Maximum depth to visualize (None for all)

        Returns:
            Path to generated file
        """
        dot = graphviz.Digraph(comment="Scene Graph Tree")
        dot.attr(rankdir="TB")  # Top to bottom layout
        dot.attr("node", shape="box", style="rounded,filled", fontname="Arial")
        dot.attr("edge", fontname="Arial")

        # Color scheme for different node types/depths
        colors = [
            "lightblue",
            "lightgreen",
            "lightcoral",
            "lightyellow",
            "lightpink",
            "lightgray",
            "lightcyan",
            "lavender",
        ]

        def add_node_recursive(node, depth: int = 0, parent_id: str = None):
            """Recursively add nodes to the graph"""
            if max_depth is not None and depth > max_depth:
                return

            # Generate unique node ID
            node_id = f"node_{id(node)}"

            # Prepare node label
            node_name = getattr(node, "name", str(node))
            label_parts = [node_name]

            if include_properties:
                # Add node properties to label
                properties = []

                # Check for common attributes
                if hasattr(node, "depth"):
                    properties.append(f"Depth: {node.depth}")
                if hasattr(node, "entity_config") and node.entity_config:
                    if "motion_type" in node.entity_config:
                        properties.append(
                            f"Motion: {node.entity_config['motion_type']}"
                        )
                if hasattr(node, "bbox") and node.bbox:
                    bbox_str = f"BBox: {node.bbox.get('x_length', 0):.2f}×{node.bbox.get('y_length', 0):.2f}×{node.bbox.get('z_length', 0):.2f}"
                    properties.append(bbox_str)
                if hasattr(node, "own_platform") and node.own_platform:
                    properties.append(f"Platforms: {len(node.own_platform)}")

                if properties:
                    label_parts.extend(properties)

            label = "\\n".join(label_parts)

            # Choose color based on depth or node type
            color = colors[depth % len(colors)]

            # Special coloring for specific node types
            if hasattr(node, "removed") and node.removed:
                color = "lightgray"
            elif hasattr(node, "depth"):
                if node.depth == 0:
                    color = "gold"  # Root node
                elif node.depth == 1:
                    color = "lightblue"  # Ground objects
                else:
                    color = "lightgreen"  # Higher level objects

            # Add node to graph
            dot.node(node_id, label, fillcolor=color)

            # Add edge from parent
            if parent_id:
                dot.edge(parent_id, node_id)

            # Process children
            if hasattr(node, "children"):
                for child in node.children:
                    add_node_recursive(child, depth + 1, node_id)
            elif hasattr(node, "all_children"):
                for child in node.all_children:
                    add_node_recursive(child, depth + 1, node_id)

        # Start recursive traversal
        add_node_recursive(tree_root)

        # Save and render
        output_path = self.output_dir / output_file
        dot.render(str(output_path), format=format, cleanup=True)

        # Also save the DOT source
        dot_file = output_path.with_suffix(".dot")
        with open(dot_file, "w") as f:
            f.write(dot.source)

        glog.info(f"Scene graph visualization saved to: {output_path}.{format}")
        return str(output_path.with_suffix(f".{format}"))

    def export_full_tree(
        self, scene_graph_tree, output_file: str = "full_scene_graph"
    ) -> str:
        """
        Export the complete scene graph tree

        Args:
            scene_graph_tree: Scene graph tree object with nodes dictionary
            output_file: Output file name

        Returns:
            Path to generated file
        """
        if not hasattr(scene_graph_tree, "nodes"):
            raise ValueError("Scene graph tree must have 'nodes' attribute")

        dot = graphviz.Digraph(comment="Complete Scene Graph")
        dot.attr(rankdir="TB")
        dot.attr("node", shape="box", style="rounded,filled", fontname="Arial")
        dot.attr("edge", fontname="Arial")

        # Add all nodes
        for node_name, node in scene_graph_tree.nodes.items():
            # Prepare node label
            label_parts = [node_name]

            # Add properties
            properties = []
            if hasattr(node, "depth"):
                properties.append(f"D:{node.depth}")
            if hasattr(node, "removed") and node.removed:
                properties.append("REMOVED")
            if hasattr(node, "own_platform") and node.own_platform:
                properties.append(f"P:{len(node.own_platform)}")

            if properties:
                label_parts.append(" | ".join(properties))

            label = "\\n".join(label_parts)

            # Color based on depth and status
            if hasattr(node, "removed") and node.removed:
                color = "lightgray"
            elif hasattr(node, "depth"):
                if node.depth == 0:
                    color = "gold"
                elif node.depth == 1:
                    color = "lightblue"
                else:
                    color = "lightgreen"
            else:
                color = "white"

            dot.node(node_name, label, fillcolor=color)

        # Add edges based on parent-child relationships
        for node_name, node in scene_graph_tree.nodes.items():
            if hasattr(node, "parent") and node.parent:
                dot.edge(node.parent.name, node_name)

        # Render
        output_path = self.output_dir / output_file
        dot.render(str(output_path), format="png", cleanup=True)

        # Save DOT source
        dot_file = output_path.with_suffix(".dot")
        with open(dot_file, "w") as f:
            f.write(dot.source)

        glog.info(f"Complete scene graph saved to: {output_path}.png")
        return str(output_path.with_suffix(".png"))


class TaskVisualizer:
    """Task visualization and analysis tool"""

    def __init__(self, output_dir: str = "visualizations/tasks"):
        """
        Initialize the task visualizer

        Args:
            output_dir: Directory to save task analysis files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_task_info(
        self, task, task_id: int, scene_graph_tree=None, output_file: str = None
    ) -> str:
        """
        Export detailed task information

        Args:
            task: Task object
            task_id: Task identifier
            scene_graph_tree: Scene graph tree for context
            output_file: Output file name (auto-generated if None)

        Returns:
            Path to generated file
        """
        if output_file is None:
            output_file = f"task_{task_id}_info.txt"

        output_path = self.output_dir / output_file

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"=== TASK {task_id} INFORMATION ===\n\n")

            # Basic task information
            f.write("1. TASK DESCRIPTION:\n")
            if hasattr(task, "__repr_rough__"):
                f.write(f"   {task.__repr_rough__()}\n\n")
            elif hasattr(task, "__str__"):
                f.write(f"   {str(task)}\n\n")
            else:
                f.write(f"   {repr(task)}\n\n")

            # Task type and properties
            f.write("2. TASK PROPERTIES:\n")
            if hasattr(task, "type"):
                f.write(f"   Type: {task.type}\n")
            if hasattr(task, "item") and task.item:
                f.write(f"   Item: {getattr(task.item, 'name', str(task.item))}\n")
            if hasattr(task, "destination") and task.destination:
                f.write(
                    f"   Destination: {getattr(task.destination, 'name', str(task.destination))}\n"
                )
            if hasattr(task, "reference_objects") and task.reference_objects:
                ref_names = [
                    getattr(obj, "name", str(obj)) for obj in task.reference_objects
                ]
                f.write(f"   Reference Objects: {', '.join(ref_names)}\n")

            # Ambiguity check
            if scene_graph_tree and hasattr(task, "is_ambiguous"):
                is_ambiguous = task.is_ambiguous(scene_graph_tree)
                f.write(f"   Is Ambiguous: {is_ambiguous}\n")
                task_level = 2 if is_ambiguous else 1
                f.write(f"   Task Level: Level {task_level}\n")

            f.write("\n")

            # Primitive information
            f.write("3. TASK PRIMITIVES:\n")
            if hasattr(task, "primitive"):
                primitive = task.primitive
                if isinstance(primitive, dict):
                    for key, value in primitive.items():
                        f.write(f"   {key}: {value}\n")
                else:
                    f.write(f"   {primitive}\n")
            else:
                f.write("   No primitive information available\n")

            f.write("\n")

            # Additional task details
            f.write("4. DETAILED ANALYSIS:\n")

            # Object positions and properties
            if hasattr(task, "item") and task.item:
                f.write(f"   Item Details:\n")
                item = task.item
                if hasattr(item, "entity_config") and item.entity_config:
                    config = item.entity_config
                    if "centroid_translation" in config:
                        pos = config["centroid_translation"]
                        f.write(
                            f"     Position: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})\n"
                        )
                    if "bbox" in config and config["bbox"] != "deprecated":
                        bbox = config["bbox"]
                        if isinstance(bbox, dict):
                            f.write(
                                f"     BBox: {bbox.get('x_length', 0):.2f}×{bbox.get('y_length', 0):.2f}×{bbox.get('z_length', 0):.2f}\n"
                            )
                    if "motion_type" in config:
                        f.write(f"     Motion Type: {config['motion_type']}\n")

            if hasattr(task, "destination") and task.destination:
                f.write(f"   Destination Details:\n")
                dest = task.destination
                if hasattr(dest, "entity_config") and dest.entity_config:
                    config = dest.entity_config
                    if "centroid_translation" in config:
                        pos = config["centroid_translation"]
                        f.write(
                            f"     Position: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})\n"
                        )

            # Free space analysis
            if hasattr(task, "item") and task.item and hasattr(task.item, "free_space"):
                f.write("   Free Space Analysis:\n")
                for direction, space_info in enumerate(task.item.free_space):
                    if space_info and "Objects" in space_info:
                        objects = space_info["Objects"]
                        if objects:
                            obj_names = [
                                getattr(obj, "name", str(obj)) for obj in objects
                            ]
                            f.write(
                                f"     Direction {direction}: {', '.join(obj_names)}\n"
                            )

        glog.info(f"Task {task_id} information saved to: {output_path}")
        return str(output_path)

    def export_task_batch_summary(
        self,
        tasks: List,
        scene_graph_tree=None,
        output_file: str = "task_batch_summary.json",
    ) -> str:
        """
        Export summary of multiple tasks

        Args:
            tasks: List of task objects
            scene_graph_tree: Scene graph tree for context analysis
            output_file: Output file name

        Returns:
            Path to generated file
        """
        output_path = self.output_dir / output_file

        summary = {
            "total_tasks": len(tasks),
            "task_types": {},
            "ambiguity_stats": {"level_1": 0, "level_2": 0},
            "tasks": [],
        }

        for i, task in enumerate(tasks):
            task_info = {
                "id": i,
                "description": getattr(task, "__repr_rough__", lambda: str(task))(),
                "type": str(getattr(task, "type", "Unknown")),
                "item": getattr(getattr(task, "item", None), "name", "Unknown"),
                "destination": getattr(
                    getattr(task, "destination", None), "name", "Unknown"
                ),
            }

            # Type statistics
            task_type = task_info["type"]
            summary["task_types"][task_type] = (
                summary["task_types"].get(task_type, 0) + 1
            )

            # Ambiguity analysis
            if scene_graph_tree and hasattr(task, "is_ambiguous"):
                is_ambiguous = task.is_ambiguous(scene_graph_tree)
                task_info["is_ambiguous"] = is_ambiguous
                task_info["level"] = 2 if is_ambiguous else 1

                if is_ambiguous:
                    summary["ambiguity_stats"]["level_2"] += 1
                else:
                    summary["ambiguity_stats"]["level_1"] += 1

            # Primitive information
            if hasattr(task, "primitive"):
                task_info["primitive"] = task.primitive

            summary["tasks"].append(task_info)

        # Save summary
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        glog.info(f"Task batch summary saved to: {output_path}")
        return str(output_path)


class VisualizationManager:
    """Main visualization manager that combines all visualization tools"""

    def __init__(self, base_output_dir: str = "visualizations"):
        """
        Initialize the visualization manager

        Args:
            base_output_dir: Base directory for all visualizations
        """
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.scene_visualizer = SceneGraphVisualizer(
            str(self.base_dir / "scene_graphs")
        )
        self.task_visualizer = TaskVisualizer(str(self.base_dir / "tasks"))

    def visualize_scene_and_tasks(
        self, scene_graph_tree, tasks: List, session_name: str = "session"
    ) -> Dict[str, str]:
        """
        Complete visualization of scene graph and tasks

        Args:
            scene_graph_tree: Scene graph tree object
            tasks: List of tasks
            session_name: Name for this visualization session

        Returns:
            Dictionary with paths to generated files
        """
        results = {}

        # Create session directory
        session_dir = self.base_dir / session_name
        session_dir.mkdir(exist_ok=True)

        # Visualize scene graph
        try:
            scene_file = self.scene_visualizer.export_full_tree(
                scene_graph_tree, f"{session_name}_scene_graph"
            )
            results["scene_graph"] = scene_file
        except Exception as e:
            glog.error(f"Failed to visualize scene graph: {e}")

        # Export task summary
        try:
            task_summary = self.task_visualizer.export_task_batch_summary(
                tasks, scene_graph_tree, f"{session_name}_task_summary.json"
            )
            results["task_summary"] = task_summary
        except Exception as e:
            glog.error(f"Failed to export task summary: {e}")

        # Export individual task details (for first 10 tasks as example)
        task_files = []
        for i, task in enumerate(tasks[:10]):
            try:
                task_file = self.task_visualizer.export_task_info(
                    task, i, scene_graph_tree, f"{session_name}_task_{i}.txt"
                )
                task_files.append(task_file)
            except Exception as e:
                glog.error(f"Failed to export task {i}: {e}")

        results["sample_tasks"] = task_files

        glog.info(f"Visualization completed for session: {session_name}")
        return results


# Convenience functions for easy usage
def quick_visualize_scene(scene_graph_tree, output_name: str = "quick_scene"):
    """Quick scene graph visualization"""
    visualizer = SceneGraphVisualizer()
    return visualizer.export_full_tree(scene_graph_tree, output_name)


def quick_visualize_task(task, task_id: int, scene_graph_tree=None):
    """Quick task visualization"""
    visualizer = TaskVisualizer()
    return visualizer.export_task_info(task, task_id, scene_graph_tree)


def quick_visualize_all(
    scene_graph_tree, tasks: List, session_name: str = "quick_session"
):
    """Quick complete visualization"""
    manager = VisualizationManager()
    return manager.visualize_scene_and_tasks(scene_graph_tree, tasks, session_name)


if __name__ == "__main__":
    # Example usage
    print("Visualization tools loaded successfully!")
    print("Usage examples:")
    print("1. quick_visualize_scene(scene_graph_tree)")
    print("2. quick_visualize_task(task, task_id, scene_graph_tree)")
    print("3. quick_visualize_all(scene_graph_tree, tasks)")
