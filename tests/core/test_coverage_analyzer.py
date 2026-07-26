from src.core.task_coverage_analyzer import (
    TaskRefs, node_id, manitasken_task_to_refs, taskchain_to_refs,
    gpt_task_to_refs,
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
