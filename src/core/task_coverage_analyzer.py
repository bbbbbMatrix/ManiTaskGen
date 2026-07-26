"""Coverage analysis for process-based tasks (ManiTaskGen vs GPT taskgen).

Normalizes both sources to a common TaskRefs and computes, per dimension,
per-instance appearance counts plus distinct-covered / scene-total ratios.
"""
import json
import os
from collections import Counter
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


DIMENSIONS = ("moving_objects", "anchor_objects", "target_platforms", "source_platforms")
_TOTAL_KEY = {
    "moving_objects": "objects",
    "anchor_objects": "objects",
    "target_platforms": "platforms",
    "source_platforms": "platforms",
}


def scene_totals(scene_graph) -> dict:
    """Total movable objects (children of sensible platforms) and total platforms."""
    platforms = set()
    objects = set()
    try:
        sensible = scene_graph.get_sensible_platform_list()
    except Exception:
        sensible = []
    for p in sensible:
        pid = node_id(p)
        if pid:
            platforms.add(pid)
        for child in getattr(p, "children", []) or []:
            cid = node_id(child)
            if cid:
                objects.add(cid)
    return {"objects": objects, "platforms": platforms}


def compute_coverage(refs_list, totals) -> dict:
    """Per-dimension counts + distinct-covered/total ratio."""
    counters = {d: Counter() for d in DIMENSIONS}
    for r in refs_list:
        for d in DIMENSIONS:
            for x in getattr(r, d, []) or []:
                counters[d][x] += 1
    out = {}
    for d in DIMENSIONS:
        total_set = totals[_TOTAL_KEY[d]]
        counts = dict(counters[d])
        covered = set(counts.keys())
        total = len(total_set)
        out[d] = {
            "counts": counts,
            "distinct_covered": len(covered),
            "total": total,
            "ratio": (len(covered) / total) if total else 0.0,
            "uncovered": sorted(total_set - covered),
        }
    return out


def write_coverage_report(manitasken_cov, gpt_cov, out_dir, meta) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Convert sets in meta for JSON serialization
    serializable_meta = {}
    for k, v in meta.items():
        if isinstance(v, set):
            serializable_meta[k] = sorted(v)
        elif isinstance(v, dict) and "totals" in meta and v is meta["totals"]:
            serializable_meta[k] = {dk: sorted(dv) if isinstance(dv, set) else dv for dk, dv in v.items()}
        else:
            serializable_meta[k] = v
    payload = {"meta": serializable_meta, "manitaskgen": manitasken_cov, "gpt": gpt_cov}
    with open(os.path.join(out_dir, "coverage_report.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    lines = [
        "# Process-based task coverage: ManiTaskGen vs GPT",
        "",
        f"sample_size={meta.get('sample_size')} seed={meta.get('seed')} "
        f"total_objects={len(meta.get('totals', {}).get('objects', []))} "
        f"total_platforms={len(meta.get('totals', {}).get('platforms', []))}",
        "",
        "| Dimension | ManiTaskGen covered/total (ratio) | GPT covered/total (ratio) |",
        "|---|---|---|",
    ]
    for d in DIMENSIONS:
        m = manitasken_cov[d]; g = gpt_cov[d]
        lines.append(
            f"| {d} | {m['distinct_covered']}/{m['total']} ({m['ratio']:.3f}) "
            f"| {g['distinct_covered']}/{g['total']} ({g['ratio']:.3f}) |"
        )
    lines.append("")
    lines.append("## Per-instance counts")
    for label, cov in (("ManiTaskGen", manitasken_cov), ("GPT", gpt_cov)):
        lines.append(f"### {label}")
        for d in DIMENSIONS:
            lines.append(f"- {d}: {cov[d]['counts']}")
    with open(os.path.join(out_dir, "coverage_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
