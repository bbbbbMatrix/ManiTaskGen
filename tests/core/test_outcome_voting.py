import textwrap
from src.utils import config_manager as cm_mod
from src.utils.config_manager import ConfigManager


def test_getter_returns_singleton_not_fresh_default():
    cfg = cm_mod.get_outcome_based_task_generation_config()
    cfg.vlm_list = ['SENTINEL_A']
    cfg.keep_min_score = 3
    again = cm_mod.get_outcome_based_task_generation_config()
    assert again.vlm_list == ['SENTINEL_A']
    assert again.keep_min_score == 3


def test_yml_propagates_to_outcome_config(tmp_path):
    yml = tmp_path / 'c.yaml'
    yml.write_text(textwrap.dedent("\n        stage2b_outcome_task_generation:\n          outcome_based_task:\n            task_num_per_pattern: 7\n            keep_min_score: 2\n            vlm_list:\n              - 'owner/one'\n              - 'owner/two'\n    "))
    mgr = ConfigManager(config_file_path=str(yml), run_dir=str(tmp_path))
    obt = mgr.config.outcome_based_task_generation
    assert obt.task_num_per_pattern == 7
    assert obt.keep_min_score == 2
    assert obt.vlm_list == ['owner/one', 'owner/two']


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
