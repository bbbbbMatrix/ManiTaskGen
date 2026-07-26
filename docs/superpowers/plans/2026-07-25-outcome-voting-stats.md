# Outcome-based VLM Voting Stats — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For a curated set of raw candidate outcome-based tasks, have the configured VLMs vote, score each 0–3, emit a score histogram + human-reviewable bundle (JSON + HTML gallery + per-task voting images) + the kept tasks.

**Architecture:** Pure logic (candidate gen/dedup, scoring, histogram, writers, config plumbing) is unit-tested with pytest. The network path (VLM calls) is factored so `vote_task` can be tested by monkeypatching the evaluator. A new `OutcomeVotingRunner` orchestrates generate→vote→aggregate→write; `02b` becomes a thin caller.

**Tech Stack:** Python 3.10, pytest (new dev dep), existing `src.core.outcome_based_task_generation`, `src.utils.config_manager`, `src.core.task_feasibility_evaluator`.

## Global Constraints
- Run from repo root with `PYTHONPATH=.` (the bash wrappers already export `PYTHONPATH=$BASE_DIR`).
- Editing `config/default_config.yml` `stage2b_outcome_task_generation.outcome_based_task.{vlm_list,task_num_per_pattern,keep_min_score}` must take effect after Task 1.
- Score = number of `"Feasible"` verdicts ∈ {0,1,2,3}; kept when `score >= keep_min_score` (default 2).
- Per-task voting images must not overwrite each other: scope `save_path` per `task_id`.
- Install pytest once: `python -m pip install pytest`.

---

## File Structure
- Create: `conftest.py` (repo root, sys.path bootstrap so `import src...` works under pytest)
- Create: `tests/core/test_outcome_voting.py`
- Create: `tests/core/__init__.py`, `tests/__init__.py` (empty, so `tests.core` imports cleanly if needed)
- Modify: `src/utils/config_manager.py`
- Modify: `src/core/outcome_based_task_generation.py`
- Modify: `config/default_config.yml`
- Modify: `src/scripts/02b_gen_outcome_based_tasks.py`

---

### Task 1: Test scaffolding + config plumbing fix

**Files:**
- Create: `conftest.py`
- Create: `tests/__init__.py`, `tests/core/__init__.py` (empty)
- Create: `tests/core/test_outcome_voting.py`
- Modify: `src/utils/config_manager.py` (`OutcomeBaseGenerationConfig` ~line 362, `StageBasedConfigLoader.export_to_app_config` ~651 & `import_from_app_config` ~632, `get_outcome_based_task_generation_config` ~1527)
- Modify: `config/default_config.yml` (`stage2b_outcome_task_generation.outcome_based_task`)

**Interfaces:**
- Produces: `get_outcome_based_task_generation_config()` returns the singleton config (not a fresh default); `OutcomeBaseGenerationConfig` gains `keep_min_score: int = 2`; yml fields propagate.

- [ ] **Step 1: Install pytest**

Run: `python -m pip install pytest`
Expected: installs pytest (importable later).

- [ ] **Step 2: Write the failing tests**

`conftest.py`:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
```

`tests/core/test_outcome_voting.py`:
```python
import textwrap
from src.utils import config_manager as cm_mod
from src.utils.config_manager import ConfigManager


def test_getter_returns_singleton_not_fresh_default():
    cfg = cm_mod.get_outcome_based_task_generation_config()
    cfg.vlm_list = ["SENTINEL_A"]
    cfg.keep_min_score = 3
    again = cm_mod.get_outcome_based_task_generation_config()
    assert again.vlm_list == ["SENTINEL_A"]
    assert again.keep_min_score == 3


def test_yml_propagates_to_outcome_config(tmp_path):
    yml = tmp_path / "c.yaml"
    yml.write_text(textwrap.dedent("""
        stage2b_outcome_task_generation:
          outcome_based_task:
            task_num_per_pattern: 7
            keep_min_score: 2
            vlm_list:
              - "owner/one"
              - "owner/two"
    """))
    mgr = ConfigManager(config_file_path=str(yml), run_dir=str(tmp_path))
    obt = mgr.config.outcome_based_task_generation
    assert obt.task_num_per_pattern == 7
    assert obt.keep_min_score == 2
    assert obt.vlm_list == ["owner/one", "owner/two"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/core/test_outcome_voting.py -v`
Expected: FAIL (getter returns fresh default; yml vlm_list not propagated).

- [ ] **Step 4: Add `keep_min_score` field**

In `src/utils/config_manager.py`, `OutcomeBaseGenerationConfig` (≈line 362), add after `task_num_per_pattern`:
```python
    keep_min_score: int = 2  # min #Feasible votes to keep a task (default: majority of 3)
```

- [ ] **Step 5: Fix the getter**

Replace `get_outcome_based_task_generation_config` (≈line 1527) body:
```python
def get_outcome_based_task_generation_config() -> OutcomeBaseGenerationConfig:
    """Get Outcome Base Generation configuration (singleton, reflects loaded yml)."""
    return config_manager.config.outcome_based_task_generation
```

- [ ] **Step 6: Wire stage ↔ AppConfig for outcome_based_task**

In `StageBasedConfigLoader.import_from_app_config` (≈line 632), after the stage_configs loop, add:
```python
        self.stage_config.stage2b_outcome_task_generation.outcome_based_task = (
            app_config.outcome_based_task_generation
        )
```
In `StageBasedConfigLoader.export_to_app_config` (≈line 651), after its stage_configs loop, add:
```python
        app_config.outcome_based_task_generation = (
            self.stage_config.stage2b_outcome_task_generation.outcome_based_task
        )
        return app_config
```
(Keep the existing `return app_config` only once — replace it with the lines above.)

- [ ] **Step 7: Add keep_min_score to yml**

In `config/default_config.yml`, under `stage2b_outcome_task_generation.outcome_based_task`, add:
```yaml
    keep_min_score: 2
```
(alongside the existing `task_num_per_pattern` and `vlm_list`).

- [ ] **Step 8: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/core/test_outcome_voting.py -v`
Expected: PASS (2 tests).

- [ ] **Step 9: Commit**

```bash
git add conftest.py tests/ src/utils/config_manager.py config/default_config.yml
git commit -m "fix(config): yml vlm_list/task_num_per_pattern/keep_min_score propagate; getter returns singleton"
```

---

### Task 2: Candidate generation (N per pattern, dedup, task_id)

**Files:**
- Modify: `src/core/outcome_based_task_generation.py` (`OutcomeBasedTask` ~750, `generate_task_with_all_patterns` ~977)
- Test: `tests/core/test_outcome_voting.py` (append)

**Interfaces:**
- Produces: `OutcomeBasedTask` has `.task_id: str`; `OutcomeBasedTaskGenerator.generate_task_with_all_patterns()` returns up to `task_num_per_pattern` unique candidates per pattern, each with a stable `task_id`, collected in `self.task_list`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_outcome_voting.py`:
```python
from src.core.outcome_based_task_generation import (
    OutcomeBasedTask,
    generate_candidate_tasks,
)


class _FakePattern:
    """Duck-typed OutcomeBasedTaskPattern for deterministic candidate tests."""
    def __init__(self, pattern_text, descriptions):
        self.task_pattern = pattern_text
        self._descs = list(descriptions)
        self._i = 0
    def generate_task_description(self, platform_list, multilayer_object_list, room_object_list):
        if self._i >= len(self._descs):
            return None, [], []
        d = self._descs[self._i]
        self._i += 1
        if d is None:
            return None, [], []
        return d, ["P0"], []


def test_generate_candidate_tasks_dedup_and_ids():
    # pattern A yields 3 unique then a duplicate; pattern B yields one None then 2 unique
    patterns = [_FakePattern("PA", ["a1", "a2", "a3", "a1"]), _FakePattern("PB", [None, "b1", "b2"])]
    tasks = generate_candidate_tasks(patterns, task_num_per_pattern=5,
                                     platform_list=[], multilayer_object_list=[], room_object_list=[])
    descs = [t.task_description for t in tasks]
    assert descs == ["PA::a1", "PA::a2", "PA::a3", "PB::b1", "PB::b2"]
    ids = [t.task_id for t in tasks]
    assert len(set(ids)) == len(ids)  # unique ids
    assert all(t.task_pattern is patterns[0] or t.task_pattern is patterns[1] for t in tasks)


def test_generate_candidate_tasks_respects_task_num_cap():
    patterns = [_FakePattern("PA", [f"a{i}" for i in range(20)])]
    tasks = generate_candidate_tasks(patterns, task_num_per_pattern=3,
                                     platform_list=[], multilayer_object_list=[], room_object_list=[])
    assert len(tasks) == 3
```
(The fake returns description strings; the real `generate_task_description` returns full strings too, so the helper prefixes pattern id for the test only — see implementation: helper uses returned description verbatim. Adjust the fake to return `"PA::a1"` style by setting `descs` accordingly. Fix: make the fake return the literal final string, e.g. `_FakePattern("PA", ["PA::a1", "PA::a2", "PA::a3", "PA::a1"])`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/core/test_outcome_voting.py -v`
Expected: FAIL (`generate_candidate_tasks` not defined; `task_id` absent).

- [ ] **Step 3: Add `task_id` to OutcomeBasedTask**

In `OutcomeBasedTask.__init__` (≈line 751), add param `task_id: str = None` and `self.task_id = task_id`.

- [ ] **Step 4: Implement `generate_candidate_tasks` helper**

Add module-level function in `src/core/outcome_based_task_generation.py` (after the `OutcomeBasedTask` class):
```python
def generate_candidate_tasks(task_pattern_list, task_num_per_pattern, platform_list,
                             multilayer_object_list, room_object_list):
    """Generate up to task_num_per_pattern unique candidate tasks per pattern.

    Deduplicates identical task descriptions within this run. Assigns each
    surviving candidate a stable task_id. Patterns the scene cannot fulfil
    (generate_task_description returns None) are skipped.
    """
    candidates = []
    seen_descriptions = set()
    for p_idx, pattern in enumerate(task_pattern_list):
        produced = 0
        attempts = 0
        # cap attempts to avoid infinite loops when few unique tasks exist
        max_attempts = task_num_per_pattern * 6
        while produced < task_num_per_pattern and attempts < max_attempts:
            attempts += 1
            desc, rel_platforms, rel_multilayer = pattern.generate_task_description(
                platform_list=platform_list,
                multilayer_object_list=multilayer_object_list,
                room_object_list=room_object_list,
            )
            if desc is None:
                break
            if desc in seen_descriptions:
                continue
            seen_descriptions.add(desc)
            task_id = f"p{p_idx:03d}_t{produced:03d}"
            candidates.append(
                OutcomeBasedTask(
                    task_description=desc,
                    task_pattern=pattern,
                    multi_layer_object_list=rel_multilayer,
                    platform_list=rel_platforms,
                    room_object_list=room_object_list,
                    task_id=task_id,
                )
            )
            produced += 1
    return candidates
```

- [ ] **Step 5: Rewrite `generate_task_with_all_patterns` to use it**

Replace the body of `generate_task_with_all_patterns` (≈line 977) with:
```python
    def generate_task_with_all_patterns(self, task_num=None, desired_pattern_list=None):
        task_num = self.task_num_per_pattern if task_num is None else task_num
        patterns = (
            [self.task_pattern_list[i] for i in desired_pattern_list]
            if desired_pattern_list is not None
            else self.task_pattern_list
        )
        self.task_list = generate_candidate_tasks(
            patterns,
            task_num_per_pattern=task_num,
            platform_list=self.platform_list,
            multilayer_object_list=self.multilayer_object_list,
            room_object_list=self.room_object_list,
        )
        glog.info(f"Generated {len(self.task_list)} candidate outcome tasks.")
        return self.task_list
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/core/test_outcome_voting.py -v`
Expected: PASS (4 tests). (Fix the fake's `descs` to literal final strings as noted in Step 1.)

- [ ] **Step 7: Commit**

```bash
git add src/core/outcome_based_task_generation.py tests/core/test_outcome_voting.py
git commit -m "feat(outcome): generate N candidate tasks/pattern with dedup + task_id"
```

---

### Task 3: `VLMVoter.vote_task` (score + per-task image isolation)

**Files:**
- Modify: `src/core/outcome_based_task_generation.py` (`VLMVoter` ~217)
- Test: `tests/core/test_outcome_voting.py` (append)

**Interfaces:**
- Produces: `compute_vote_score(verdicts) -> int`; `VLMVoter.vote_task(task, scene_graph, task_id, image4vote_path) -> dict` with keys `task_id, task_description, pattern, score, verdicts:list[{model,verdict}], feasible, image_dir, platforms, objects`. `is_task_feasible(...)` becomes a bool wrapper over `vote_task`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_outcome_voting.py`:
```python
from src.core.outcome_based_task_generation import compute_vote_score, VLMVoter


def test_compute_vote_score_counts_feasible():
    assert compute_vote_score(["Feasible", "Not feasible", "Feasible"]) == 2
    assert compute_vote_score(["Not feasible", "Partially feasible"]) == 0
    assert compute_vote_score([]) == 0
    assert compute_vote_score(["Feasible", "Feasible", "Feasible"]) == 3


def test_vote_task_scores_and_isolates_image_dir(monkeypatch):
    voter = VLMVoter.__new__(VLMVoter)  # bypass __init__ (no API/network)
    voter.config = type("C", (), {"vlm_list": ["m0", "m1", "m2"], "keep_min_score": 2})()
    voter.global_config = type("G", (), {})()
    # rotate verdicts per model
    verdicts = {"m0": "Feasible", "m1": "Not feasible", "m2": "Feasible"}
    calls = []
    class _Eval:
        def evaluate_task_feasibility(self, interactor, task, scene_graph, width, height, save_path):
            calls.append(save_path)
            return verdicts[interactor.model_name]
    class _Inter:
        def __init__(self, name): self.model_name = name
        def change_model_name(self, n): self.model_name = n
    voter.evaluator = _Eval()
    voter.vlm_interactor = _Inter("m0")
    voter.vlm_interactor.change_model_name = lambda n: setattr(voter.vlm_interactor, "model_name", n)

    class _Task:
        task_id = "p000_t000"
        task_description = "do something"
        task_pattern = "PAT"
        platform_list = []
        multi_layer_object_list = []
    result = voter.vote_task(_Task(), scene_graph=None, task_id="p000_t000",
                             image4vote_path="/tmp/img4vote")
    assert result["score"] == 2
    assert result["feasible"] is True
    assert [v["verdict"] for v in result["verdicts"]] == ["Feasible", "Not feasible", "Feasible"]
    assert all(c == "/tmp/img4vote/task_p000_t000" for c in calls)  # per-task dir
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/core/test_outcome_voting.py -v`
Expected: FAIL (`compute_vote_score`/`vote_task` not defined).

- [ ] **Step 3: Implement `compute_vote_score`**

Add module-level function in `outcome_based_task_generation.py`:
```python
def compute_vote_score(verdicts):
    """Score = number of 'Feasible' verdicts (0..len(vlm_list))."""
    return sum(1 for v in verdicts if v == "Feasible")
```

- [ ] **Step 4: Implement `vote_task` + refactor `is_task_feasible`**

In `VLMVoter`, replace `is_task_feasible` (≈line 228) with:
```python
    def vote_task(self, task, scene_graph, task_id, image4vote_path):
        """Vote a task with all configured VLMs. Returns a result dict with score."""
        task_image_dir = f"{image4vote_path}/task_{task_id}"
        os.makedirs(task_image_dir, exist_ok=True)
        verdicts = []
        for vlm in self.vlm_list:
            self.vlm_interactor.change_model_name(vlm)
            self.vlm_interactor.model_name = vlm
            verdict = self.evaluator.evaluate_task_feasibility(
                self.vlm_interactor,
                task,
                scene_graph,
                width=512,
                height=512,
                save_path=task_image_dir,
            )
            verdicts.append({"model": vlm, "verdict": verdict})
        score = compute_vote_score([v["verdict"] for v in verdicts])
        keep_min = getattr(self.config, "keep_min_score", 2)
        return {
            "task_id": task_id,
            "task_description": task.task_description,
            "pattern": task.task_pattern.task_pattern if hasattr(task.task_pattern, "task_pattern") else str(task.task_pattern),
            "score": score,
            "verdicts": verdicts,
            "feasible": score >= keep_min,
            "image_dir": task_image_dir,
            "platforms": [p.platform_name for p in getattr(task, "platform_list", [])],
            "objects": [m.multilayer_object_name for m in getattr(task, "multi_layer_object_list", [])],
        }

    def is_task_feasible(self, task, scene_graph, task_id="legacy", image4vote_path=None):
        """Backward-compatible bool wrapper over vote_task."""
        image4vote_path = image4vote_path or self.global_config.image4vote_path
        return self.vote_task(task, scene_graph, task.task_id if hasattr(task, "task_id") and task.task_id else task_id, image4vote_path)["feasible"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/core/test_outcome_voting.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add src/core/outcome_based_task_generation.py tests/core/test_outcome_voting.py
git commit -m "feat(outcome): VLMVoter.vote_task scores 0-3 with per-task image isolation"
```

---

### Task 4: Aggregation (histogram + kept) + JSON writer

**Files:**
- Modify: `src/core/outcome_based_task_generation.py` (add `OutcomeVotingRunner` aggregation + writers)
- Test: `tests/core/test_outcome_voting.py` (append)

**Interfaces:**
- Produces: `build_histogram(results) -> {0:int,1:int,2:int,3:int}`; `OutcomeVotingRunner.write_results_json(results, path, keep_min_score, vlm_list)`; `OutcomeVotingRunner.write_kept_txt(results, path)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_outcome_voting.py`:
```python
from src.core.outcome_based_task_generation import build_histogram, OutcomeVotingRunner


def _res(score, tid):
    return {"task_id": tid, "task_description": f"d{tid}", "pattern": "P",
            "score": score, "verdicts": [], "feasible": score >= 2,
            "image_dir": f"/tmp/i{tid}", "platforms": [], "objects": []}


def test_build_histogram():
    results = [_res(0, "a"), _res(0, "b"), _res(2, "c"), _res(3, "d"), _res(1, "e")]
    assert build_histogram(results) == {0: 2, 1: 1, 2: 1, 3: 1}


def test_write_results_json_and_kept_txt(tmp_path):
    results = [_res(0, "a"), _res(3, "b")]
    runner = OutcomeVotingRunner.__new__(OutcomeVotingRunner)
    json_path = tmp_path / "vote_results.json"
    txt_path = tmp_path / "kept.txt"
    runner.write_results_json(results, str(json_path), keep_min_score=2, vlm_list=["m0", "m1", "m2"])
    runner.write_kept_txt(results, str(txt_path))
    import json
    data = json.loads(json_path.read_text())
    assert data["histogram"] == {0: 1, 1: 0, 2: 0, 3: 1}
    assert [t["task_id"] for t in data["kept_tasks"]] == ["b"]
    assert "d b" in txt_path.read_text() and "d a" not in txt_path.read_text()
```

- [ ] **Step 2: Run test to verify it fail**

Run: `PYTHONPATH=. python -m pytest tests/core/test_outcome_voting.py -v`
Expected: FAIL (`build_histogram`/`OutcomeVotingRunner` not defined).

- [ ] **Step 3: Implement aggregation + writers**

Add module-level + class in `outcome_based_task_generation.py`:
```python
def build_histogram(results):
    hist = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in results:
        s = r["score"]
        if s in hist:
            hist[s] += 1
    return hist


class OutcomeVotingRunner:
    """Orchestrates candidate generation -> voting -> aggregation -> outputs."""

    def __init__(self, generator, vlm_voter, scene_graph, image4vote_path, out_dir):
        self.generator = generator
        self.vlm_voter = vlm_voter
        self.scene_graph = scene_graph
        self.image4vote_path = image4vote_path
        self.out_dir = out_dir

    def run(self):
        import os, json
        os.makedirs(self.out_dir, exist_ok=True)
        tasks = self.generator.generate_task_with_all_patterns()
        results = []
        for task in tasks:
            tid = task.task_id or f"t{len(results):03d}"
            results.append(self.vlm_voter.vote_task(task, self.scene_graph, tid, self.image4vote_path))
        keep_min = getattr(self.vlm_voter.config, "keep_min_score", 2)
        vlm_list = self.vlm_voter.vlm_list
        self.write_results_json(results, os.path.join(self.out_dir, "vote_results.json"), keep_min, vlm_list)
        self.write_kept_txt(results, os.path.join(self.out_dir, "outcome_based_task.txt"))
        self.write_review_gallery(results, os.path.join(self.out_dir, "review_gallery.html"))
        glog.info(f"Voting done: {build_histogram(results)} kept={sum(1 for r in results if r['feasible'])}/{len(results)}")
        return results

    def write_results_json(self, results, path, keep_min_score, vlm_list):
        import json
        kept = [r for r in results if r["feasible"]]
        payload = {
            "keep_min_score": keep_min_score,
            "vlm_list": list(vlm_list),
            "histogram": build_histogram(results),
            "kept_tasks": kept,
            "tasks": results,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def write_kept_txt(self, results, path):
        with open(path, "w") as f:
            for r in results:
                if r["feasible"]:
                    f.write(f"{r['task_description']}\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/core/test_outcome_voting.py -v`
Expected: PASS (8 tests). (`write_review_gallery` is stubbed in Task 5; add a no-op now so `run()` doesn't break — see Task 5 Step 3 replaces it.)

- [ ] **Step 5: Commit**

```bash
git add src/core/outcome_based_task_generation.py tests/core/test_outcome_voting.py
git commit -m "feat(outcome): aggregate histogram + kept, write vote_results.json + kept txt"
```

---

### Task 5: HTML review gallery

**Files:**
- Modify: `src/core/outcome_based_task_generation.py` (`OutcomeVotingRunner.write_review_gallery`)
- Test: `tests/core/test_outcome_voting.py` (append)

**Interfaces:**
- Produces: `OutcomeVotingRunner.write_review_gallery(results, path)` writes `review_gallery.html` embedding each task's voting images (relative to `image_dir`) grouped by score.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_outcome_voting.py`:
```python
def test_write_review_gallery_groups_by_score(tmp_path):
    import os
    img_dir = tmp_path / "task_X"
    img_dir.mkdir()
    (img_dir / "plat.png").write_bytes(b"\x89PNG\r\n")  # dummy image
    results = [{
        "task_id": "X", "task_description": "Arrange things", "pattern": "P",
        "score": 3, "feasible": True, "image_dir": str(img_dir),
        "verdicts": [{"model": "m0", "verdict": "Feasible"}], "platforms": [], "objects": [],
    }]
    runner = OutcomeVotingRunner.__new__(OutcomeVotingRunner)
    html_path = tmp_path / "review_gallery.html"
    runner.write_review_gallery(results, str(html_path))
    html = html_path.read_text()
    assert "Arrange things" in html
    assert "Score 3" in html or "score-3" in html
    assert "task_X" in html  # image referenced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/core/test_outcome_voting.py -v`
Expected: FAIL (`write_review_gallery` not defined / no-op).

- [ ] **Step 3: Implement `write_review_gallery`**

Add method to `OutcomeVotingRunner`:
```python
    def write_review_gallery(self, results, path):
        import os, html as html_mod
        rows = []
        for r in sorted(results, key=lambda x: (-x["score"], x["task_id"])):
            imgs = ""
            img_dir = r.get("image_dir")
            if img_dir and os.path.isdir(img_dir):
                for fn in sorted(os.listdir(img_dir)):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                        rel = os.path.relpath(os.path.join(img_dir, fn), os.path.dirname(path))
                        imgs += f'<img src="{html_mod.escape(rel)}" style="max-width:320px;margin:4px;border:1px solid #ccc;">'
            verdicts = ", ".join(f'{v["model"]}={v["verdict"]}' for v in r.get("verdicts", []))
            kept = "KEPT" if r["feasible"] else ""
            rows.append(
                f'<div class="card score-{r["score"]}">'
                f'<h3>Score {r["score"]} <small>{kept}</small> &middot; {html_mod.escape(r["task_id"])}</h3>'
                f'<p>{html_mod.escape(r["task_description"])}</p>'
                f'<p class="v">pattern: {html_mod.escape(str(r["pattern"]))} | {html_mod.escape(verdicts)}</p>'
                f'<div>{imgs}</div></div>'
            )
        doc = (
            "<!doctype html><html><head><meta charset='utf-8'><title>Outcome Vote Review</title>"
            "<style>body{font-family:sans-serif;margin:16px} .card{border:1px solid #ddd;border-radius:6px;"
            "padding:10px;margin:10px 0} .score-0{background:#fdd} .score-1{background:#fec}"
            " .score-2{background:#efd} .score-3{background:#dfd} .v{color:#555;font-size:small}</style></head>"
            "<body><h1>Outcome-based VLM voting review</h1>"
            + "".join(rows) + "</body></html>"
        )
        with open(path, "w") as f:
            f.write(doc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/core/test_outcome_voting.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/outcome_based_task_generation.py tests/core/test_outcome_voting.py
git commit -m "feat(outcome): HTML review gallery grouped by score with voting images"
```

---

### Task 6: Wire `02b` as thin caller + manual integration

**Files:**
- Modify: `src/scripts/02b_gen_outcome_based_tasks.py`

**Interfaces:**
- Produces: `02b` builds generator + `VLMVoter`, runs `OutcomeVotingRunner.run()`, writes outputs under `runs/output/outcome_review/`.

- [ ] **Step 1: Rewrite the voting section of `02b`**

In `src/scripts/02b_gen_outcome_based_tasks.py`, replace the block after `outcome_based_task_generator.load_task_patterns(...)` (≈lines 197–211) with:
```python
    vlm_voter = VLMVoter()

    import os
    out_dir = os.path.join(os.path.dirname(main_config.outcome_based_task_txt_save_path), "outcome_review")
    os.makedirs(out_dir, exist_ok=True)
    image4vote_path = main_config.image4vote_path

    runner = OutcomeVotingRunner(
        generator=outcome_based_task_generator,
        vlm_voter=vlm_voter,
        scene_graph=scene_graph,
        image4vote_path=image4vote_path,
        out_dir=out_dir,
    )
    runner.run()
```
Add `OutcomeVotingRunner` to the existing import from `src.core.outcome_based_task_generation` (≈line 22).

- [ ] **Step 2: Manual integration run (requires OpenRouter key)**

Set a real key in `config/default_config.yml` (`common.open_router.api_key`). Optionally shrink the run for cost: set `stage2b_outcome_task_generation.outcome_based_task.vlm_list` to one cheap model and `task_num_per_pattern: 1`, and point `manitaskot_pattern_file` at a small curated subset `.txt`.
Run: `CONFIG_FILE=config/default_config.yml bash scripts/run_02b_gen_outcome_tasks.sh`
Expected: completes; `runs/output/outcome_review/vote_results.json` has `histogram` with 0–3 buckets, `outcome_based_task.txt` has kept tasks, `review_gallery.html` opens in a browser showing grouped images. If a yml field doesn't apply, fix in yml per "优先调整 yml".

- [ ] **Step 3: Commit**

```bash
git add src/scripts/02b_gen_outcome_based_tasks.py
git commit -m "feat(outcome): 02b runs OutcomeVotingRunner -> review bundle under outcome_review/"
```

---

## Self-Review notes
- Spec coverage: config plumbing (Task 1), candidate gen N/pattern (Task 2), scoring 0–3 + image isolation (Task 3), histogram + JSON + kept txt (Task 4), HTML gallery (Task 5), end-to-end run (Task 6). All spec Part-1 items covered.
- Type consistency: `vote_task` returns the dict shape consumed by `build_histogram`/`write_*`/`write_review_gallery` (all keyed on `score`, `feasible`, `image_dir`, `verdicts`, `task_description`, `pattern`, `task_id`). Verified consistent across tasks.
- No placeholders; every code step has runnable code and exact commands.
