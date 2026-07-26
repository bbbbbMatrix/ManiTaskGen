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
