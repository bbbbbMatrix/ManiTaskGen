# Process-based Coverage vs GPT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample 100 generated process-based tasks; compute a multi-dimension coverage breakdown (per item / per receptacle counts + covered-vs-total ratios) for ManiTaskGen and for a GPT-produced task set (manual run via `demand/example.md`), and write a side-by-side comparison report.

**Architecture:** A pure `task_coverage_analyzer.py` normalizes both sources to a common `TaskRefs` and computes coverage against scene totals. Fully unit-tested with synthetic duck-typed objects (no pkl/network needed for unit tests). The CLI `05` loads the pkl + GPT json and writes the report.

**Tech Stack:** Python 3.10, pytest, existing `src.core.process_based_task_generation` (pkl shape), scene graph nodes.

## Global Constraints
- `import src...` works under `PYTHONPATH=.` (conftest.py from Plan 1 already bootstraps sys.path).
- Scene totals are the **same denominators** for both ManiTaskGen and GPT (fair comparison): total movable objects = direct children of sensible platforms; total platforms = `scene_graph.get_sensible_platform_list()`.
- IDs: use node `.name` (matches `SceneExporter`'s export ids → directly comparable across sources).
- `feature` entries that are `str` are directions (ignore); non-str entries are anchor object nodes.
- Install pytest once (done in Plan 1): `python -m pip install pytest`.

---

## File Structure
- Create: `src/core/task_coverage_analyzer.py`
- Create: `tests/core/test_coverage_analyzer.py`
- Create: `src/scripts/05_coverage_analysis.py`
- Create: `scripts/run_05_coverage_analysis.sh`

---

### Task 1: `TaskRefs` + ManiTaskGen normalization

**Files:**
- Create: `src/core/task_coverage_analyzer.py`
- Create: `tests/core/test_coverage_analyzer.py`

**Interfaces:**
- Produces: `TaskRefs` dataclass; `node_id(node) -> str|None`; `manitasken_task_to_refs(task) -> TaskRefs`; `taskchain_to_refs(chain) -> list[TaskRefs]`.

- [ ] **Step 1: Write the failing tests**

`tests/core/test_coverage_analyzer.py`:
```python
from src.core.task_coverage_analyzer import (
    TaskRefs, node_id, manitasken_task_to_refs, taskchain_to_refs,
)


class _Node:
    def __init__(self, name): self.name = name
    def get_bel_ground_platform(self): return self._platform
    def set_platform(self, p): self._platform = p; return self


class _Plat:
    def __init__(self, name): self.name = name


class _Task:
    def __init__(self, item, destination, feature=None, type=None):
        self.item = item; self.destination = destination; self.feature = feature or []; self.type = type


class _Chain:
    def __init__(self, subtasks): self.subtask_list = subtasks


def test_node_id():
    assert node_id(_Node("book_04_37")) == "book_04_37"
    assert node_id(None) is None
    assert node_id("rear") is None  # plain strings (directions) are not ids


def test_manitasken_task_to_refs_extracts_all_dimensions():
    src = _Plat("sofa_10_platform_0")
    item = _Node("book_04_37").set_platform(src)
    dst = _Plat("stool_02_platform_0")
    anchor = _Node("plate_01_50")
    t = _Task(item, dst, feature=[anchor, "rear-left"], type=None)
    refs = manitasken_task_to_refs(t)
    assert refs.moving_objects == ["book_04_37"]
    assert refs.target_platforms == ["stool_02_platform_0"]
    assert refs.source_platforms == ["sofa_10_platform_0"]
    assert refs.anchor_objects == ["plate_01_50"]  # direction string dropped


def test_taskchain_to_refs_one_per_subtask():
    src = _Plat("p_src"); dst = _Plat("p_dst")
    item = _Node("o1").set_platform(src)
    chain = _Chain([_Task(item, dst), _Task(item, dst, feature=[_Node("o2")])])
    refs = taskchain_to_refs(chain)
    assert len(refs) == 2
    assert refs[0].moving_objects == ["o1"]
    assert refs[1].anchor_objects == ["o2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/core/test_coverage_analyzer.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement module**

`src/core/task_coverage_analyzer.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/core/test_coverage_analyzer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/task_coverage_analyzer.py tests/core/test_coverage_analyzer.py
git commit -m "feat(coverage): TaskRefs + ManiTaskGen task normalization"
```

---

### Task 2: GPT JSON normalization

**Files:**
- Modify: `src/core/task_coverage_analyzer.py`
- Test: `tests/core/test_coverage_analyzer.py` (append)

**Interfaces:**
- Produces: `gpt_task_to_refs(gpt_task: dict) -> list[TaskRefs]` reading `steps[].{moving_object_id, source_platform_id, target_platform_id, anchor_object_ids}` per `demand/example.md`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_coverage_analyzer.py`:
```python
from src.core.task_coverage_analyzer import gpt_task_to_refs


def test_gpt_task_to_refs_reads_steps():
    gpt_task = {
        "task_id": "P0001",
        "steps": [
            {"moving_object_id": "book_04_37", "source_platform_id": "sofa_10_platform_0",
             "target_platform_id": "stool_02_platform_0", "anchor_object_ids": []},
            {"moving_object_id": "plate_01_50", "source_platform_id": "stool_02_platform_0",
             "target_platform_id": "cabinet_4_body_platform_0",
             "anchor_object_ids": ["bowl_06_54", None]},
        ],
    }
    refs = gpt_task_to_refs(gpt_task)
    assert len(refs) == 2
    assert refs[0].moving_objects == ["book_04_37"]
    assert refs[0].target_platforms == ["stool_02_platform_0"]
    assert refs[0].source_platforms == ["sofa_10_platform_0"]
    assert refs[1].anchor_objects == ["bowl_06_54"]  # None dropped
    assert refs[1].source_platforms == ["stool_02_platform_0"]


def test_gpt_task_to_refs_empty_steps():
    assert gpt_task_to_refs({"steps": []}) == []
    assert gpt_task_to_refs({}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/core/test_coverage_analyzer.py -v`
Expected: FAIL (`gpt_task_to_refs` not defined).

- [ ] **Step 3: Implement `gpt_task_to_refs`**

Append to `src/core/task_coverage_analyzer.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/core/test_coverage_analyzer.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/task_coverage_analyzer.py tests/core/test_coverage_analyzer.py
git commit -m "feat(coverage): GPT task JSON normalization"
```

---

### Task 3: Scene totals + coverage computation + report writer

**Files:**
- Modify: `src/core/task_coverage_analyzer.py`
- Test: `tests/core/test_coverage_analyzer.py` (append)

**Interfaces:**
- Produces: `scene_totals(scene_graph) -> {objects:set, platforms:set}`; `compute_coverage(refs_list, totals) -> dict`; `DIMENSIONS = ("moving_objects","anchor_objects","target_platforms","source_platforms")`; `write_coverage_report(manitasken_cov, gpt_cov, out_dir, meta)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_coverage_analyzer.py`:
```python
from src.core.task_coverage_analyzer import (
    scene_totals, compute_coverage, write_coverage_report, DIMENSIONS,
)


class _Child:
    def __init__(self, name): self.name = name


class _Sg:
    def __init__(self, plats):
        self._plats = plats
    def get_sensible_platform_list(self):
        return self._plats


class _PlatWithChildren:
    def __init__(self, name, children):
        self.name = name; self.children = children


def test_scene_totals():
    sg = _Sg([_PlatWithChildren("p0", [_Child("o1"), _Child("o2")]), _PlatWithChildren("p1", [])])
    tot = scene_totals(sg)
    assert tot["platforms"] == {"p0", "p1"}
    assert tot["objects"] == {"o1", "o2"}


def test_compute_coverage_countss_and_ratios():
    totals = {"objects": {"o1", "o2", "o3"}, "platforms": {"p0", "p1"}}
    refs = [
        TaskRefs(moving_objects=["o1"], target_platforms=["p0"], source_platforms=["p1"], anchor_objects=["o2"]),
        TaskRefs(moving_objects=["o1"], target_platforms=["p0"], source_platforms=["p0"], anchor_objects=[]),
    ]
    cov = compute_coverage(refs, totals)
    # moving_objects: o1 appeared twice; distinct covered 1/3
    assert cov["moving_objects"]["counts"] == {"o1": 2}
    assert cov["moving_objects"]["distinct_covered"] == 1
    assert cov["moving_objects"]["total"] == 3
    assert abs(cov["moving_objects"]["ratio"] - 1/3) < 1e-9
    # target_platforms: p0 x2; covered 1/2
    assert cov["target_platforms"]["counts"] == {"p0": 2}
    assert cov["target_platforms"]["ratio"] == 0.5
    assert cov["anchor_objects"]["counts"] == {"o2": 1}
    # every DIMENSIONS key present
    assert set(cov.keys()) == set(DIMENSIONS)


def test_write_coverage_report(tmp_path):
    totals = {"objects": {"o1", "o2"}, "platforms": {"p0", "p1"}}
    cov_mn = compute_coverage([TaskRefs(moving_objects=["o1"], target_platforms=["p0"])], totals)
    cov_gpt = compute_coverage([TaskRefs(moving_objects=["o2"], target_platforms=["p1"])], totals)
    write_coverage_report(cov_mn, cov_gpt, str(tmp_path),
                          meta={"sample_size": 1, "seed": 0, "totals": totals})
    import json
    data = json.loads((tmp_path / "coverage_report.json").read_text())
    assert data["meta"]["sample_size"] == 1
    assert data["manitaskgen"]["moving_objects"]["counts"] == {"o1": 1}
    assert data["gpt"]["moving_objects"]["counts"] == {"o2": 1}
    md = (tmp_path / "coverage_report.md").read_text()
    assert "moving_objects" in md and "ManiTaskGen" in md and "GPT" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/core/test_coverage_analyzer.py -v`
Expected: FAIL (`scene_totals`/`compute_coverage`/`write_coverage_report`/`DIMENSIONS` not defined).

- [ ] **Step 3: Implement totals, coverage, report**

Append to `src/core/task_coverage_analyzer.py`:
```python
import json
import os
from collections import Counter

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
    payload = {"meta": meta, "manitaskgen": manitasken_cov, "gpt": gpt_cov}
    with open(os.path.join(out_dir, "coverage_report.json"), "w") as f:
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
    with open(os.path.join(out_dir, "coverage_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/core/test_coverage_analyzer.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/task_coverage_analyzer.py tests/core/test_coverage_analyzer.py
git commit -m "feat(coverage): scene totals, coverage metrics, json+md comparison report"
```

---

### Task 4: CLI `05` + bash wrapper + manual integration

**Files:**
- Create: `src/scripts/05_coverage_analysis.py`
- Create: `scripts/run_05_coverage_analysis.sh`

**Interfaces:**
- Produces: `python src/scripts/05_coverage_analysis.py --manitaskgen_pkl <pkl> --scene_graph_pkl <sg.pkl> --gpt_json <json> --sample_size 100 --seed 0 --out <dir>` writes `coverage_report.{json,md}`.

- [ ] **Step 1: Implement the CLI**

`src/scripts/05_coverage_analysis.py`:
```python
# -*- coding: utf-8 -*-
"""Coverage analysis: ManiTaskGen process tasks (sampled) vs GPT taskgen (manual json).

Loads a TaskGeneration pkl (has .tasks = list[TaskChain]) and a scene_graph pkl,
samples N tasks, computes coverage; does the same for a GPT JSON
(demand/example.md schema) and writes a comparison report.
"""
import os, sys, argparse, pickle, json, random, logging

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import glog
from src.core.task_coverage_analyzer import (
    taskchain_to_refs, gpt_task_to_refs, scene_totals, compute_coverage,
    write_coverage_report,
)


def parse_args():
    p = argparse.ArgumentParser(description="Process-based coverage: ManiTaskGen vs GPT")
    p.add_argument("--manitaskgen_pkl", required=True, help="TaskGeneration pkl with .tasks")
    p.add_argument("--scene_graph_pkl", required=True, help="scene_graph pkl for scene totals")
    p.add_argument("--gpt_json", default=None, help="GPT taskgen JSON (example.md schema); optional")
    p.add_argument("--sample_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output dir (default runs/output/coverage)")
    return p.parse_args()


def main(args):
    random.seed(args.seed)
    out_dir = args.out or os.path.join("runs", "output", "coverage")
    os.makedirs(out_dir, exist_ok=True)

    # scene totals
    with open(args.scene_graph_pkl, "rb") as f:
        scene_graph = pickle.load(f)
    totals = scene_totals(scene_graph)
    glog.info(f"scene totals: objects={len(totals['objects'])} platforms={len(totals['platforms'])}")

    # ManiTaskGen: sample N TaskChains -> refs
    with open(args.manitaskgen_pkl, "rb") as f:
        taskgen = pickle.load(f)
    all_tasks = list(getattr(taskgen, "tasks", []))
    n = min(args.sample_size, len(all_tasks))
    sampled = random.sample(all_tasks, n) if n > 0 else []
    mn_refs = [r for chain in sampled for r in taskchain_to_refs(chain)]
    glog.info(f"sampled {n} ManiTaskGen tasks -> {len(mn_refs)} subtask refs")
    mn_cov = compute_coverage(mn_refs, totals)

    # GPT (optional; manual run via demand/example.md produces this json)
    if args.gpt_json and os.path.exists(args.gpt_json):
        with open(args.gpt_json) as f:
            gpt_data = json.load(f)
        gpt_tasks = gpt_data.get("tasks", gpt_data) if isinstance(gpt_data, dict) else gpt_data
        gpt_refs = [r for t in gpt_tasks for r in gpt_task_to_refs(t)]
        gpt_cov = compute_coverage(gpt_refs, totals)
        glog.info(f"GPT tasks -> {len(gpt_refs)} step refs")
    else:
        from src.core.task_coverage_analyzer import DIMENSIONS, _TOTAL_KEY
        glog.warning("No GPT json provided; GPT coverage will be empty in the report.")
        gpt_cov = {d: {"counts": {}, "distinct_covered": 0,
                       "total": len(totals[_TOTAL_KEY[d]]), "ratio": 0.0, "uncovered": []}
                   for d in DIMENSIONS}

    meta = {"sample_size": n, "seed": args.seed, "totals": {k: sorted(v) for k, v in totals.items()}}
    write_coverage_report(mn_cov, gpt_cov, out_dir, meta)
    glog.info(f"coverage report written to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_args())
```

- [ ] **Step 2: Add the bash wrapper**

`scripts/run_05_coverage_analysis.sh`:
```bash
#!/bin/bash
# Process-based coverage analysis: ManiTaskGen (sampled) vs GPT taskgen.
# Usage:
#   MANITASKGEN_PKL=... SCENE_GRAPH_PKL=... [GPT_JSON=...] bash scripts/run_05_coverage_analysis.sh
source "$(dirname "$0")/config.sh"
print_config
log_step "Starting 05_coverage_analysis.py"
cd "$BASE_DIR"

: "${MANITASKGEN_PKL:=$CACHE_DIR/process_based_task.pkl}"
: "${SCENE_GRAPH_PKL:=$CACHE_DIR/scene_graph.pkl}"

args=(--manitaskgen_pkl "$MANITASKGEN_PKL" --scene_graph_pkl "$SCENE_GRAPH_PKL" --sample_size "${SAMPLE_SIZE:-100}" --seed "${SEED:-0}")
[[ -n "$GPT_JSON" ]] && args+=(--gpt_json "$GPT_JSON")
[[ -n "$OUT_DIR" ]] && args+=(--out "$OUT_DIR")

run_python_script "05_coverage_analysis.py" "${args[@]}"
[[ $? -eq 0 ]] && log_info "Coverage report -> ${OUT_DIR:-runs/output/coverage}" || { log_error "coverage failed"; exit 1; }
```

- [ ] **Step 3: Run the full unit suite**

Run: `PYTHONPATH=. python -m pytest tests/ -v`
Expected: all tests PASS (both plans' suites).

- [ ] **Step 4: Manual integration run**

Run against the existing pkl + a placeholder/real GPT json:
```
PYTHONPATH=. python src/scripts/05_coverage_analysis.py \
  --manitaskgen_pkl runs/cache/process_based_task.pkl \
  --scene_graph_pkl runs/cache/scene_graph.pkl \
  --sample_size 100 --seed 0
```
Expected: prints scene totals (objects/platforms), samples 100 TaskChains, writes `runs/output/coverage/coverage_report.{json,md}` with non-zero ManiTaskGen coverage and the four-dimension table. Confirm denominators match the scene (objects/platforms counts). Then re-run with a real `--gpt_json` (GPT tasks from `demand/example.md`) to populate the GPT column.

- [ ] **Step 5: Commit**

```bash
git add src/scripts/05_coverage_analysis.py scripts/run_05_coverage_analysis.sh
git commit -m "feat(coverage): 05 CLI + bash wrapper for ManiTaskGen vs GPT coverage"
```

---

## Self-Review notes
- Spec coverage: TaskRefs + ManiTaskGen norm (Task 1), GPT norm (Task 2), scene totals + metrics + report (Task 3), CLI + sampling + integration (Task 4). All spec Part-2 items covered.
- Type consistency: `TaskRefs` fields (`moving_objects`, `anchor_objects`, `target_platforms`, `source_platforms`) are used identically by `manitasken_task_to_refs`, `gpt_task_to_refs`, `compute_coverage` (iterates `DIMENSIONS`), and the report writer. `scene_totals` returns `{"objects","platforms"}` consumed via `_TOTAL_KEY` in `compute_coverage`. Verified consistent.
- No placeholders; every code step has runnable code and exact commands.
