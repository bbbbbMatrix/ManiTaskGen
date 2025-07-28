from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum
from src.core.process_based_task_generation import Task, TaskType


class SpatialRelation(Enum):
    """Spatial relationships for task primitives"""

    ON = "on"  # Basic placement
    AT = "at"  # At a specific direction/part
    AROUND = "around"  # Around an object
    BETWEEN = "between"  # Between two objects
    FREESPACE = "freespace"  # In freespace of an object


class Direction(Enum):
    """Spatial directions"""

    FRONT = "front"
    REAR = "rear"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    FRONT_LEFT = "front-left"
    FRONT_RIGHT = "front-right"
    REAR_LEFT = "rear-left"
    REAR_RIGHT = "rear-right"


@dataclass
class TaskPrimitive:
    """Structured task primitive representation"""

    action: str  # "move"
    relation: SpatialRelation  # spatial relationship
    target_platform: str  # target platform
    object: str  # object to operate on
    target_object0: Optional[str] = None  # primary target object (if applicable)
    target_object1: Optional[str] = None  # secondary target object (if applicable)
    direction: Optional[Direction] = None  # direction (if applicable)
    part: Optional[str] = None  # part (if applicable)

    @staticmethod
    def from_task(task: Task) -> "TaskPrimitive":
        """Convert from atomic task to task primitive"""
        target_platform = (
            task.destination.name if task.destination else "default_platform"
        )
        object_name = task.item.name
        target_object0, target_object1 = None, None
        direction = None
        part = None
        relation = SpatialRelation.ON  # Default relation

        # Determine relation and extract features based on task type
        if task.type == TaskType.MOVE_TO_OBJECT_FREESPACE_9_GRID:
            # Format 4: move (item) to (platform), (target_item)'s (direction) freespace
            relation = SpatialRelation.FREESPACE
            if hasattr(task, "feature") and len(task.feature) > 0:
                target_object0 = (
                    task.feature[0].name
                    if hasattr(task.feature[0], "name")
                    else str(task.feature[0])
                )
                if len(task.feature) > 1:
                    direction_str = str(task.feature[1])
                    try:
                        direction = Direction(direction_str)
                    except ValueError:
                        # Handle numeric direction indices (0-8 for 9-grid)
                        direction_map = {
                            0: Direction.REAR,
                            1: Direction.REAR_LEFT,
                            2: Direction.LEFT,
                            3: Direction.FRONT_LEFT,
                            4: Direction.FRONT,
                            5: Direction.FRONT_RIGHT,
                            6: Direction.RIGHT,
                            7: Direction.REAR_RIGHT,
                            8: Direction.CENTER,
                        }
                        direction = direction_map.get(
                            int(direction_str), Direction.CENTER
                        )

        elif task.type == TaskType.MOVE_AROUND_OBJECT:
            # Format 3: move (item) to (platform) around (target_item)
            relation = SpatialRelation.AROUND
            if hasattr(task, "feature") and len(task.feature) > 0:
                target_object0 = (
                    task.feature[0].name
                    if hasattr(task.feature[0], "name")
                    else str(task.feature[0])
                )

        elif task.type == TaskType.MOVE_TO_MIDDLE_OF_OBJECTS:
            # Format 5: move (item) to (platform), between (target_itema) and (target_itemb)
            relation = SpatialRelation.BETWEEN
            if hasattr(task, "feature") and len(task.feature) >= 2:
                target_object0 = (
                    task.feature[0].name
                    if hasattr(task.feature[0], "name")
                    else str(task.feature[0])
                )
                target_object1 = (
                    task.feature[1].name
                    if hasattr(task.feature[1], "name")
                    else str(task.feature[1])
                )

        elif task.type == TaskType.MOVE_TO_EMPTY_PLATFORM_9_GRID:
            # Format 2: move (item) to (platform) (direction) part
            relation = SpatialRelation.AT
            if hasattr(task, "feature") and len(task.feature) > 0:
                direction_str = str(task.feature[0])
                try:
                    direction = Direction(direction_str)
                except ValueError:
                    # Handle numeric direction indices
                    direction_map = {
                        0: Direction.REAR,
                        1: Direction.REAR_LEFT,
                        2: Direction.LEFT,
                        3: Direction.FRONT_LEFT,
                        4: Direction.FRONT,
                        5: Direction.FRONT_RIGHT,
                        6: Direction.RIGHT,
                        7: Direction.REAR_RIGHT,
                        8: Direction.CENTER,
                    }
                    direction = direction_map.get(int(direction_str), Direction.CENTER)

                if len(task.feature) > 1:
                    part = str(task.feature[1])

        elif task.type == TaskType.MOVE_TO_EMPTY_PLATFORM:
            # Format 1: move (item) to (platform)
            relation = SpatialRelation.ON

        # Handle task direction if available
        if (
            hasattr(task, "direction")
            and task.direction is not None
            and direction is None
        ):
            if isinstance(task.direction, str):
                try:
                    direction = Direction(task.direction)
                except ValueError:
                    pass
            elif isinstance(task.direction, int):
                direction_map = {
                    0: Direction.REAR,
                    1: Direction.REAR_LEFT,
                    2: Direction.LEFT,
                    3: Direction.FRONT_LEFT,
                    4: Direction.FRONT,
                    5: Direction.FRONT_RIGHT,
                    6: Direction.RIGHT,
                    7: Direction.REAR_RIGHT,
                    8: Direction.CENTER,
                }
                direction = direction_map.get(task.direction, Direction.CENTER)

        import ipdb

        ipdb.set_trace()

        return TaskPrimitive(
            action="move",
            relation=relation,
            target_platform=target_platform,
            object=object_name,
            target_object0=target_object0,
            target_object1=target_object1,
            direction=direction,
            part=part,
        )

    def to_natural_language(self) -> str:
        """Convert to natural language description"""
        base = f"{self.action} {self.object} to {self.target_platform}"

        if self.relation == SpatialRelation.ON:
            # Format 1: move (item) to (platform)
            return base

        elif self.relation == SpatialRelation.AT:
            # Format 2: move (item) to (platform) (direction) part
            if self.direction and self.part:
                return f"{base} {self.direction.value} {self.part}"
            elif self.direction:
                return f"{base} {self.direction.value} area"
            else:
                return base

        elif self.relation == SpatialRelation.AROUND:
            # Format 3: move (item) to (platform) around (target_item)
            if self.target_object0:
                return f"{base} around {self.target_object0}"
            else:
                return f"{base} around some object"

        elif self.relation == SpatialRelation.FREESPACE:
            # Format 4: move (item) to (platform), (target_item)'s (direction) freespace
            if self.target_object0 and self.direction:
                return (
                    f"{base}, {self.target_object0}'s {self.direction.value} freespace"
                )
            elif self.target_object0:
                return f"{base}, {self.target_object0}'s freespace"
            else:
                return f"{base}, freespace"

        elif self.relation == SpatialRelation.BETWEEN:
            # Format 5: move (item) to (platform), between (target_itema) and (target_itemb)
            if self.target_object0 and self.target_object1:
                return (
                    f"{base}, between {self.target_object0} and {self.target_object1}"
                )
            elif self.target_object0:
                return f"{base}, near {self.target_object0}"
            else:
                return f"{base}, between objects"

        return base

    def to_pddl_goal(self) -> str:
        """Convert to PDDL goal representation"""
        if self.relation == SpatialRelation.ON:
            # Simple placement
            return f"(at {self.object} {self.target_platform})"

        elif self.relation == SpatialRelation.AT:
            # At specific direction/part
            if self.direction and self.part:
                return f"(and (at {self.object} ?loc) (on ?loc {self.target_platform}) (direction ?loc {self.direction.value}) (part ?loc {self.part}))"
            elif self.direction:
                return f"(and (at {self.object} ?loc) (on ?loc {self.target_platform}) (direction ?loc {self.direction.value}))"
            else:
                return f"(at {self.object} {self.target_platform})"

        elif self.relation == SpatialRelation.AROUND:
            # Around an object
            if self.target_object0:
                return f"(and (at {self.object} ?loc) (on ?loc {self.target_platform}) (around ?loc {self.target_object0}))"
            else:
                return f"(at {self.object} {self.target_platform})"

        elif self.relation == SpatialRelation.FREESPACE:
            # In freespace of an object
            if self.target_object0 and self.direction:
                return f"(and (at {self.object} ?loc) (on ?loc {self.target_platform}) (freespace ?loc {self.target_object0} {self.direction.value}))"
            elif self.target_object0:
                return f"(and (at {self.object} ?loc) (on ?loc {self.target_platform}) (freespace ?loc {self.target_object0}))"
            else:
                return f"(at {self.object} {self.target_platform})"

        elif self.relation == SpatialRelation.BETWEEN:
            # Between two objects
            if self.target_object0 and self.target_object1:
                return f"(and (at {self.object} ?loc) (on ?loc {self.target_platform}) (between ?loc {self.target_object0} {self.target_object1}))"
            elif self.target_object0:
                return f"(and (at {self.object} ?loc) (on ?loc {self.target_platform}) (near ?loc {self.target_object0}))"
            else:
                return f"(at {self.object} {self.target_platform})"

        return f"(at {self.object} {self.target_platform})"

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "action": self.action,
            "relation": self.relation.value,
            "target_platform": self.target_platform,
            "object": self.object,
            "target_object0": self.target_object0,
            "target_object1": self.target_object1,
            "direction": self.direction.value if self.direction else None,
            "part": self.part,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskPrimitive":
        """Create TaskPrimitive from dictionary"""
        return cls(
            action=data["action"],
            relation=SpatialRelation(data["relation"]),
            target_platform=data["target_platform"],
            object=data["object"],
            target_object0=data.get("target_object0"),
            target_object1=data.get("target_object1"),
            direction=Direction(data["direction"]) if data.get("direction") else None,
            part=data.get("part"),
        )

    def __str__(self) -> str:
        """String representation"""
        return self.to_natural_language()

    def __repr__(self) -> str:
        """Detailed representation"""
        return f"TaskPrimitive({self.action}, {self.relation.value}, {self.object} -> {self.target_platform})"

    def is_valid(self) -> bool:
        """Check if the primitive is valid"""
        if not self.action or not self.object or not self.target_platform:
            return False

        # Check relation-specific requirements
        if self.relation == SpatialRelation.BETWEEN:
            return self.target_object0 is not None and self.target_object1 is not None
        elif self.relation == SpatialRelation.AROUND:
            return self.target_object0 is not None
        elif self.relation == SpatialRelation.FREESPACE:
            return self.target_object0 is not None

        return True

    def get_required_objects(self) -> List[str]:
        """Get list of all objects required for this primitive"""
        objects = [self.object, self.target_platform]

        if self.target_object0:
            objects.append(self.target_object0)
        if self.target_object1:
            objects.append(self.target_object1)

        return [obj for obj in objects if obj is not None]


# Utility functions
def tasks_to_primitives(tasks: List[Task]) -> List[TaskPrimitive]:
    """Convert a list of tasks to task primitives"""
    primitives = []
    for task in tasks:
        try:
            primitive = TaskPrimitive.from_task(task)
            if primitive.is_valid():
                primitives.append(primitive)
        except Exception as e:
            print(f"Warning: Failed to convert task to primitive: {e}")

    return primitives


def primitives_to_pddl_domain(
    primitives: List[TaskPrimitive], domain_name: str = "task_domain"
) -> str:
    """Generate PDDL domain from primitives"""
    predicates = set()
    actions = set()

    for primitive in primitives:
        # Extract predicates from PDDL goals
        goal = primitive.to_pddl_goal()
        # This is a simplified extraction - you might want to improve this
        if "at" in goal:
            predicates.add("(at ?obj ?loc)")
        if "on" in goal:
            predicates.add("(on ?loc ?platform)")
        if "direction" in goal:
            predicates.add("(direction ?loc ?dir)")
        if "around" in goal:
            predicates.add("(around ?loc ?obj)")
        if "between" in goal:
            predicates.add("(between ?loc ?obj1 ?obj2)")
        if "freespace" in goal:
            predicates.add("(freespace ?loc ?obj ?dir)")

        actions.add(primitive.action)

    domain = f"""(define (domain {domain_name})
  (:requirements :strips :typing)
  (:types object location platform direction part)
  (:predicates
    {chr(10).join(f"    {pred}" for pred in sorted(predicates))}
  )
  
  ; Actions would be defined here based on the primitives
  ; This is a template - specific actions need to be implemented
)"""

    return domain


def export_primitives_summary(primitives: List[TaskPrimitive]) -> dict:
    """Export summary statistics of primitives"""
    summary = {
        "total_primitives": len(primitives),
        "relations": {},
        "complexity_distribution": {},
        "objects_involved": set(),
        "platforms_used": set(),
    }

    for primitive in primitives:
        # Count relations
        rel = primitive.relation.value
        summary["relations"][rel] = summary["relations"].get(rel, 0) + 1

        # Collect objects and platforms
        summary["objects_involved"].update(primitive.get_required_objects())
        summary["platforms_used"].add(primitive.target_platform)

    # Convert sets to lists for JSON serialization
    summary["objects_involved"] = list(summary["objects_involved"])
    summary["platforms_used"] = list(summary["platforms_used"])

    return summary
