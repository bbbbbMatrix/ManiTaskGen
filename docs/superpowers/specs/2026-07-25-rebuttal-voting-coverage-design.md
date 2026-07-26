# Rebuttal: Outcome-based VLM voting stats + Process-based coverage vs GPT

Date: 2026-07-25
Branch: `haina-rebuttal-nips2026` (from `cc28b05`)
Demand: `demand/rebuttal_demand.md`

## Goal

Two deliverables for the rebuttal, both runnable from `config/default_config.yml` + an OpenRouter key:

1. **Part 1 — Outcome-based VLM voting.** For a curated set of raw candidate tasks, have the configured VLMs vote, score each task 0–3, report the score histogram, emit a human-reviewable bundle (task text + per-VLM verdicts + voting images), and write the final kept tasks.
2. **Part 2 — Process-based coverage vs GPT.** Sample 100 generated process-based tasks; compute a multi-dimension coverage breakdown (per item / per receptacle counts + covered-vs-total ratios); do the same for a GPT-produced task set (manual run via `demand/example.md`) and compare.

## Decisions (confirmed with user)

- Part 1: generate `task_num_per_pattern` (=5) candidates per pattern; patterns come from a **curated subset `.txt`** pointed to by `manitaskot_pattern_file` (treated as input, not all 200). VLMs are the 3 in yml `vlm_list`. Score = #`Feasible` votes ∈ {0,1,2,3}. Kept = score ≥ `keep_min_score` (default 2 = majority of 3). Review output = **HTML gallery + JSON manifest + per-task image subfolders**.
- Part 2: coverage = **full breakdown table** across {moving objects, anchor objects, target platforms (receptacle), source platforms}; per-instance counts + (distinct covered / scene total) ratio; computed for ManiTaskGen (sample 100) and GPT. GPT side = option 1 now (analyzer ingests a manually-produced GPT JSON), shaped so an automated 4o driver can drop in later.

## Part 1 design

### Root cause: yml `vlm_list` / `task_num_per_pattern` do not take effect

`get_outcome_based_task_generation_config()` in `src/utils/config_manager.py` returns a **fresh `OutcomeBaseGenerationConfig()`** every call, ignoring the loaded config. Separately, the staged loader copies stage→`AppConfig` attributes by matching names, but the stage attr is `outcome_based_task` while the `AppConfig` attr is `outcome_based_task_generation` — a name mismatch, so `vlm_list` / `task_num_per_pattern` never reach `AppConfig`. This is why editing yml does nothing. Fix (honors "优先调整 yml"):

- `get_outcome_based_task_generation_config()` → return `config_manager.config.outcome_based_task_generation`.
- In `StageBasedConfigLoader.export_to_app_config` / `import_from_app_config`, add explicit wiring: `app_config.outcome_based_task_generation = stage2b.outcome_based_task` (and reverse on import), so yml fields propagate.
- Add `keep_min_score: int = 2` as a dataclass field on `OutcomeBaseGenerationConfig` and to `config/default_config.yml` under `stage2b_outcome_task_generation.outcome_based_task`.

### Scoring (`src/core/outcome_based_task_generation.py`)

- Replace `VLMVoter.is_task_feasible` with `vote_task(task, scene_graph, task_id) -> dict` returning `{task_id, task_description, pattern, score, verdicts:[{model, verdict}], feasible, image_paths, platforms, objects}`. `score = sum(verdict == "Feasible")`. `feasible = score >= keep_min_score`. Keep `is_task_feasible` as a thin bool wrapper for any other callers.
- **Per-task image isolation:** call the existing `TaskFeasibilityEvaluator.evaluate_task_feasibility` with `save_path = image4vote_path/task_<task_id>` so different tasks no longer overwrite each other's voting images (today images are saved per-platform-name and collide). The evaluator already keys files by platform name, so scoping the directory is sufficient; no geometry/render changes.

### Candidate generation

- `OutcomeBasedTaskGenerator.generate_task_with_all_patterns()` generates `task_num_per_pattern` candidates per pattern by calling `generate_task_description` repeatedly (it already re-samples platforms/objects via `random.shuffle`/`random.sample`). Dedup identical descriptions; assign each surviving candidate a stable `task_id`. Skip patterns the scene cannot fulfil (existing behavior returns `None`).

### Orchestration

- New `OutcomeVotingRunner` in the same module: generate candidates → `vote_task` each → build histogram `{0:n,1:n,2:n,3:n}` + kept set → write all outputs. `src/scripts/02b_gen_outcome_based_tasks.py` becomes a thin caller (build generator, run, done) mirroring how `04_export_scene_for_taskgen.py` wraps `SceneExporter`.

### Outputs (`runs/output/outcome_review/`)

- `vote_results.json` — `{histogram:{0,1,2,3}, keep_min_score, vlm_list, tasks:[{task_id, description, pattern, score, verdicts[], image_paths, platforms, objects, kept}], kept_tasks:[...]}`.
- `outcome_based_task.txt` — kept tasks only (existing path/contract unchanged).
- `review_gallery.html` — one page embedding each task's voting images + per-VLM verdicts, grouped by score 0/1/2/3 (uses relative paths into `task_<id>/`).
- `task_<id>/` — per-task voting images.

### Files

- `src/core/outcome_based_task_generation.py` — `vote_task`, candidate-gen loop, `OutcomeVotingRunner`, gallery+json writers.
- `src/utils/config_manager.py` — getter fix + staged-loader wiring + `keep_min_score`.
- `config/default_config.yml` — add `keep_min_score`; keep `vlm_list`/`task_num_per_pattern`.
- `src/scripts/02b_gen_outcome_based_tasks.py` — thin caller.

## Part 2 design

### New module `src/core/task_coverage_analyzer.py`

- `CoverageAnalyzer` normalizes both sources to a common per-task `TaskRefs`:
  - ManiTaskGen (`Task`/`TaskChain` from `runs/cache/process_based_task.pkl`): `moving_objects=[subtask.item.name]`, `target_platforms=[subtask.destination.name]`, `source_platforms=[subtask.item.get_bel_ground_platform().name]`, `anchor_objects=[f.name for f in subtask.feature if object]`.
  - GPT (JSON per `demand/example.md`): from `steps[]` → `moving_object_id`, `source_platform_id`, `target_platform_id`, `anchor_object_ids`.
- ID comparability: `SceneExporter` uses `platform.name`/`node.name` as export ids = same namespace as the scene graph, so the two sets are directly comparable. A `StringConvertor`-based name normalizer guards against rename-dict drift, with an optional category rollup as a robustness fallback.
- Scene totals (denominators): total movable objects = direct children of sensible platforms; total platforms = `get_sensible_platform_list()`. Same denominators for both sources → fair comparison. Read from `scene_graph.pkl`, with the export `manifest.json` as an alternative source.

### Metrics (full breakdown, per dimension)

For each of {moving objects, anchor objects, target platforms, source platforms}: per-instance appearance counts; `distinct_covered / scene_total` ratio. Computed for (a) a seeded random sample of 100 ManiTaskGen tasks and (b) the GPT task set.

### New script `src/scripts/05_coverage_analysis.py` + `scripts/run_05_coverage_analysis.sh`

Args: `--manitaskgen_pkl`, `--gpt_json`, `--scene_graph_pkl`, `--sample_size 100`, `--seed`, `--out`. Designed so a future GPT-via-OpenRouter (`4o`) generator can feed the same `--gpt_json` path.

### Outputs (`runs/output/coverage/`)

- `coverage_report.json` + `coverage_report.md` — ManiTaskGen-vs-GPT side-by-side across the four dimensions, with per-instance counts and coverage ratios.

### Files

- `src/core/task_coverage_analyzer.py` (new).
- `src/scripts/05_coverage_analysis.py` (new).
- `scripts/run_05_coverage_analysis.sh` (new).

## Verification

- **Part 1:** `CONFIG_FILE=config/default_config.yml bash scripts/run_02b_gen_outcome_tasks.sh` with a real OpenRouter key. Confirm `vote_results.json` histogram has 0/1/2/3 buckets, kept tasks land in `outcome_based_task.txt`, `review_gallery.html` renders with images. If a full run is too costly during verification, tune yml (`vlm_list` → 1 cheap model, lower `task_num_per_pattern`) — that is the intended "adjust yml" knob, now that plumbing is fixed.
- **Part 2:** run `05` against the existing `runs/cache/process_based_task.pkl` (sample 100) + a placeholder/real GPT JSON; confirm the breakdown table and ratios compute and denominators match the scene.

## Out of scope

- Automated GPT-via-OpenRouter task generation (deferred; analyzer accepts manual GPT JSON now).
- Changing vote/feasibility prompts or rendering geometry (reuse as-is).
- Category-level coverage as a primary metric (instance-level is primary; category rollup is a fallback only).
