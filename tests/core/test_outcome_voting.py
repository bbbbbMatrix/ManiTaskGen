import textwrap
from src.utils import config_manager as cm_mod
from src.utils.config_manager import ConfigManager
from src.core.outcome_based_task_generation import (
    OutcomeBasedTask,
    VLMVoter,
    compute_vote_score,
    generate_candidate_tasks,
)


def test_getter_returns_singleton_not_fresh_default():
    cfg = cm_mod.get_outcome_based_task_generation_config()
    cfg.vlm_list = ['SENTINEL_A']
    cfg.keep_min_score = 3
    again = cm_mod.get_outcome_based_task_generation_config()
    assert again.vlm_list == ['SENTINEL_A']
    assert again.keep_min_score == 3


def test_yml_propagates_to_outcome_config(tmp_path):
    yml = tmp_path / 'c.yaml'
    yml.write_text(textwrap.dedent("""
        stage2b_outcome_task_generation:
          outcome_based_task:
            task_num_per_pattern: 7
            keep_min_score: 2
            vlm_list:
              - 'owner/one'
              - 'owner/two'
    """))
    mgr = ConfigManager(config_file_path=str(yml), run_dir=str(tmp_path))
    obt = mgr.config.outcome_based_task_generation
    assert obt.task_num_per_pattern == 7
    assert obt.keep_min_score == 2
    assert obt.vlm_list == ['owner/one', 'owner/two']


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
    # The real generate_task_description returns None deterministically when a
    # pattern cannot be fulfilled (no matching platform/multilayer); the helper
    # treats None as terminal (break), not a transient retry.
    # PA: 3 unique then a duplicate, then exhausts (None) -> 3 produced.
    # PB: unfulfillable (None on first call) -> 0 produced.
    # PC: 2 unique then exhausts -> 2 produced.
    patterns = [
        _FakePattern("PA", ["PA::a1", "PA::a2", "PA::a3", "PA::a1"]),
        _FakePattern("PB", []),
        _FakePattern("PC", ["PC::c1", "PC::c2"]),
    ]
    tasks = generate_candidate_tasks(patterns, task_num_per_pattern=5,
                                     platform_list=[], multilayer_object_list=[], room_object_list=[])
    descs = [t.task_description for t in tasks]
    assert descs == ["PA::a1", "PA::a2", "PA::a3", "PC::c1", "PC::c2"]
    ids = [t.task_id for t in tasks]
    assert len(set(ids)) == len(ids)  # unique ids
    assert all(t.task_pattern in patterns for t in tasks)


def test_generate_candidate_tasks_respects_task_num_cap():
    patterns = [_FakePattern("PA", [f"PA::a{i}" for i in range(20)])]
    tasks = generate_candidate_tasks(patterns, task_num_per_pattern=3,
                                     platform_list=[], multilayer_object_list=[], room_object_list=[])
    assert len(tasks) == 3


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


from src.core.outcome_based_task_generation import build_histogram, OutcomeVotingRunner


def _res(score, tid):
    return {"task_id": tid, "task_description": f"d {tid}", "pattern": "P",
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
    assert data["histogram"] == {"0": 1, "1": 0, "2": 0, "3": 1}
    assert [t["task_id"] for t in data["kept_tasks"]] == ["b"]
    assert "d b" in txt_path.read_text() and "d a" not in txt_path.read_text()
