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
        if not isinstance(gpt_tasks, list):
            glog.warning(f"GPT json 'tasks' is not a list (got {type(gpt_tasks).__name__}); treating as no tasks.")
            gpt_tasks = []
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
