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
