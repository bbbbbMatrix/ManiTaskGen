"""Coverage analysis for process-based tasks (ManiTaskGen vs GPT taskgen).

Normalizes both sources to a common TaskRefs and computes, per dimension,
per-instance appearance counts plus distinct-covered / scene-total ratios.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TaskRefs:
    moving_objects: List[str] = field(default_factory=list)
    anchor_objects: List[str] = field(default_factory=list)
    target_platforms: List[str] = field(default_factory=list)
    source_platforms: List[str] = field(default_factory=list)


def node_id(node) -> Optional[str]:
    """Return the node's name id, or None for None / plain strings (directions)."""
    if node is None or isinstance(node, str):
        return None
    return getattr(node, "name", None)


def manitasken_task_to_refs(task) -> TaskRefs:
    """Extract refs from a ManiTaskGen process Task (one subtask)."""
    moving = node_id(task.item) or []
    target = node_id(task.destination) or []
    source = None
    try:
        source = node_id(task.item.get_bel_ground_platform())
    except Exception:
        source = None
    anchors = [node_id(f) for f in (task.feature or []) if not isinstance(f, str)]
    anchors = [a for a in anchors if a]
    return TaskRefs(
        moving_objects=[moving] if moving else [],
        anchor_objects=anchors,
        target_platforms=[target] if target else [],
        source_platforms=[source] if source else [],
    )


def taskchain_to_refs(chain) -> List[TaskRefs]:
    """One TaskRefs per subtask in a TaskChain."""
    return [manitasken_task_to_refs(st) for st in getattr(chain, "subtask_list", [])]


def gpt_task_to_refs(gpt_task: dict) -> List[TaskRefs]:
    """Normalize one GPT task (example.md schema) to TaskRefs per step."""
    out = []
    for step in gpt_task.get("steps", []) or []:
        def _opt(v):
            return v if isinstance(v, str) and v else None
        moving = _opt(step.get("moving_object_id"))
        src = _opt(step.get("source_platform_id"))
        tgt = _opt(step.get("target_platform_id"))
        anchors = [a for a in (step.get("anchor_object_ids") or []) if isinstance(a, str) and a]
        out.append(TaskRefs(
            moving_objects=[moving] if moving else [],
            anchor_objects=anchors,
            target_platforms=[tgt] if tgt else [],
            source_platforms=[src] if src else [],
        ))
    return out
