"""test_pathfind.py -- BFS next-step over the video graph's transition edges. No network."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import pathfind


def _t(label, fr, to):
    return {"id": "phineas/%s/v0" % label, "kind": "transition", "label": label,
            "from": "phineas:" + fr, "to": "phineas:" + to}


def _i(label, at):
    return {"id": "phineas/%s/v0" % label, "kind": "idle", "label": label,
            "from": "phineas:" + at, "to": "phineas:" + at}


# anchor is the hub; glower / swoon / behind_chair hang off it; island is disconnected.
EDGES = [
    _i("breathe", "anchor"),
    _t("a2g", "anchor", "glower"), _t("g2a", "glower", "anchor"), _i("glare", "glower"),
    _t("a2s", "anchor", "swoon"), _t("s2a", "swoon", "anchor"), _i("heave", "swoon"),
    _t("a2b", "anchor", "behind_chair"), _t("b2a", "behind_chair", "anchor"), _i("declaim", "behind_chair"),
    _i("island_idle", "island"),   # a pose with no transitions in or out
]


def test_direct_neighbour():
    e = pathfind.next_step(EDGES, "phineas:anchor", "phineas:glower")
    assert e is not None and e["label"] == "a2g"


def test_multi_hop_returns_first_edge():
    # glower -> swoon must route through the hub; first hop is g2a
    e = pathfind.next_step(EDGES, "phineas:glower", "phineas:swoon")
    assert e is not None and e["label"] == "g2a"
    # behind_chair -> glower likewise goes via anchor; first hop is b2a
    e2 = pathfind.next_step(EDGES, "phineas:behind_chair", "phineas:glower")
    assert e2 is not None and e2["label"] == "b2a"


def test_already_there_is_none():
    assert pathfind.next_step(EDGES, "phineas:anchor", "phineas:anchor") is None


def test_unreachable_is_none():
    assert pathfind.next_step(EDGES, "phineas:anchor", "phineas:island") is None
    assert pathfind.next_step(EDGES, "phineas:anchor", "phineas:ghost") is None


def test_idle_self_loops_are_not_paths():
    # an idle edge never counts as a step between two distinct poses
    e = pathfind.next_step(EDGES, "phineas:anchor", "phineas:swoon")
    assert e["kind"] == "transition"


def test_reachable_poses():
    r = pathfind.reachable_poses(EDGES, "phineas:anchor")
    assert r == {"phineas:glower", "phineas:swoon", "phineas:behind_chair"}
    assert "phineas:island" not in r
    assert "phineas:anchor" not in r   # excludes the start itself


def test_shortest_path_node_sequence():
    p = pathfind.shortest_path(EDGES, "phineas:glower", "phineas:swoon")
    assert p == ["phineas:glower", "phineas:anchor", "phineas:swoon"]
    assert pathfind.shortest_path(EDGES, "phineas:anchor", "phineas:island") == []
