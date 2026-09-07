"""
test_clip_graph.py -- the clip library graph (runtime/clip_graph.py).

Pure stdlib under test: no numpy / cv2 / pygame, so every test here runs on a
bare box. Covers shortest-path traversal (reachable / same-node / unreachable),
the synthetic-placeholder accounting that drives missing_edges(), the cost model
that prefers real footage over a stub, Clip validation, and the JSON manifest
round-trip (the KV-index the player reads).
"""
from __future__ import annotations

import json

import pytest

from clip_graph import Clip, ClipGraph, POSES, default_graph


# --------------------------------------------------------------------------- #
# path()
# --------------------------------------------------------------------------- #
def _linear_graph():
    """idle -> address_house -> leaving, plus an idle self-loop. The canonical
    fixture the module's own __main__ smoke uses."""
    g = ClipGraph()
    g.add("idle", "address_house", frames=10, tags=["transition"])
    g.add("address_house", "idle", frames=10, tags=["transition"])
    g.add("address_house", "leaving", frames=14, tags=["transition"])
    g.add("idle", "idle", frames=18, loopable=True, tags=["hold"])
    return g


def test_path_reachable_returns_clip_sequence():
    g = _linear_graph()
    p = g.path("idle", "leaving")
    assert p is not None
    assert [c.key() for c in p] == ["idle->address_house", "address_house->leaving"]


def test_path_same_node_is_empty_list():
    # [] (already there) is DISTINCT from None (unreachable). The player branches
    # on this: [] => hold current pose, None => ask the pipeline to generate.
    g = _linear_graph()
    assert g.path("idle", "idle") == []


def test_path_unreachable_returns_none():
    g = _linear_graph()
    # 'leaving' has no outgoing edge, so nothing is reachable FROM it.
    assert g.path("leaving", "idle") is None


def test_path_unknown_node_returns_none():
    g = _linear_graph()
    # A node never declared / with no edge dict at all is treated as unreachable.
    assert g.path("idle", "not_a_pose") is None
    assert g.path("not_a_pose", "idle") is None


def test_path_prefers_real_footage_over_synthetic_penalty():
    """A direct synthetic edge (cost = frames + 1000) must lose to a longer route
    of real edges, because synthetic edges carry the 1000-penalty in .cost."""
    g = ClipGraph()
    # direct but synthetic: idle->leaving (asset=None)
    g.add("idle", "leaving", asset=None, frames=12)            # cost 1012
    # longer but real: idle->address_house->leaving (assets set)
    g.add("idle", "address_house", asset="a.mp4", frames=12)   # cost 12
    g.add("address_house", "leaving", asset="b.mp4", frames=12)  # cost 12 (sum 24)
    p = g.path("idle", "leaving")
    assert [c.key() for c in p] == ["idle->address_house", "address_house->leaving"], \
        "planner took the cheap synthetic shortcut instead of the real-footage route"


def test_path_picks_shortest_by_frames_among_real():
    g = ClipGraph()
    g.add("idle", "ponder", asset="x.mp4", frames=30)             # direct, 30
    g.add("idle", "lean_in", asset="y.mp4", frames=5)
    g.add("lean_in", "ponder", asset="z.mp4", frames=5)           # via lean_in, 10
    p = g.path("idle", "ponder")
    assert [c.key() for c in p] == ["idle->lean_in", "lean_in->ponder"]


# --------------------------------------------------------------------------- #
# missing_edges() / is_fully_baked()
# --------------------------------------------------------------------------- #
def test_missing_edges_lists_only_synthetic():
    g = ClipGraph()
    g.add("idle", "ponder", asset=None, frames=10)         # synthetic
    g.add("idle", "lean_in", asset="real.mp4", frames=10)  # real
    assert g.missing_edges() == [("idle", "ponder")]


def test_asset_set_edge_is_not_missing():
    g = ClipGraph()
    g.add("idle", "ponder", asset="some_clip.mp4", frames=10)
    # A clip claiming real footage (even if it would fail to decode) is NOT
    # 'missing' -- that's a different failure mode from never-generated.
    assert ("idle", "ponder") not in g.missing_edges()


def test_missing_edges_required_subset():
    g = ClipGraph()
    g.add("idle", "ponder", asset="some_clip.mp4", frames=10)  # real
    # required: a pair that is absent entirely + a pair present-but-real.
    gaps = g.missing_edges([("idle", "address_house"), ("idle", "ponder")])
    assert gaps == [("idle", "address_house")]


def test_missing_edges_required_flags_synthetic_present():
    g = ClipGraph()
    g.add("idle", "ponder", asset=None, frames=10)  # present but synthetic
    gaps = g.missing_edges([("idle", "ponder")])
    assert gaps == [("idle", "ponder")]


def test_is_fully_baked():
    g = ClipGraph()
    g.add("idle", "ponder", asset="real.mp4", frames=10)
    assert g.is_fully_baked() is True
    g.add("idle", "lean_in", asset=None, frames=10)
    assert g.is_fully_baked() is False


# --------------------------------------------------------------------------- #
# Clip dataclass: validation + derived properties
# --------------------------------------------------------------------------- #
def test_clip_rejects_unknown_pose():
    with pytest.raises(ValueError):
        Clip(from_node="idle", to_node="not_a_pose")
    with pytest.raises(ValueError):
        Clip(from_node="bogus", to_node="idle")


def test_clip_rejects_nonpositive_frames_fps():
    with pytest.raises(ValueError):
        Clip(from_node="idle", to_node="idle", frames=0)
    with pytest.raises(ValueError):
        Clip(from_node="idle", to_node="idle", fps=0)


def test_clip_derived_properties():
    synth = Clip(from_node="idle", to_node="idle", asset=None, frames=24, fps=24)
    assert synth.is_synthetic is True
    assert synth.is_hold is True                 # self-loop
    assert synth.duration_s == pytest.approx(1.0)
    assert synth.cost == pytest.approx(24 + 1000)  # synthetic penalty

    real = Clip(from_node="idle", to_node="ponder", asset="x.mp4", frames=12, fps=24)
    assert real.is_synthetic is False
    assert real.is_hold is False
    assert real.cost == pytest.approx(12.0)      # no penalty


def test_clip_key_is_ordered_pair():
    assert Clip(from_node="idle", to_node="ponder").key() == "idle->ponder"


def test_add_clip_replaces_existing_edge():
    # last write wins -- a verified clip cleanly supersedes a synthetic placeholder.
    g = ClipGraph()
    g.add("idle", "ponder", asset=None, frames=12)
    g.add("idle", "ponder", asset="real.mp4", frames=20)
    clip = g.get_clip("idle", "ponder")
    assert clip.asset == "real.mp4" and clip.frames == 20
    assert len([c for c in g.edges() if c.key() == "idle->ponder"]) == 1


# --------------------------------------------------------------------------- #
# Manifest save / load round-trip (the JSON KV-index)
# --------------------------------------------------------------------------- #
def test_manifest_round_trip_preserves_edges(tmp_path):
    g = _linear_graph()
    g.add("idle", "ponder", asset="clip.mp4", frames=15, fps=30,
          loopable=True, tags=["verified", "transition"])
    mpath = tmp_path / "manifest.json"
    g.save(mpath)

    g2 = ClipGraph.load(mpath)
    assert {c.key() for c in g2.edges()} == {c.key() for c in g.edges()}
    # field-level fidelity on a representative edge
    rt = g2.get_clip("idle", "ponder")
    assert rt.asset == "clip.mp4"
    assert rt.frames == 15 and rt.fps == 30 and rt.loopable is True
    assert rt.tags == ["verified", "transition"]


def test_manifest_is_atomic_and_well_formed(tmp_path):
    g = _linear_graph()
    mpath = tmp_path / "manifest.json"
    g.save(mpath)
    # No stray temp files left beside the manifest (atomic replace cleaned up).
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    # well-formed: {version, nodes, clips:[...]}
    data = json.loads(mpath.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert set(POSES).issubset(set(data["nodes"]))
    assert isinstance(data["clips"], list) and data["clips"]


def test_load_absent_manifest_is_empty_graph(tmp_path):
    g = ClipGraph.load(tmp_path / "does_not_exist.json")
    assert g.edges() == []
    # still a usable graph with the default pose nodes
    assert g.path("idle", "idle") == []


def test_from_dict_tolerates_extra_keys():
    # forward-compat: a row carrying keys the verify gate stamped on must load.
    row = {"from_node": "idle", "to_node": "ponder", "asset": "c.mp4",
           "frames": 10, "fps": 24, "loopable": False, "tags": [],
           "ssim_score": 0.93, "verified_at": "2026-01-01"}
    clip = Clip.from_dict(row)
    assert clip.key() == "idle->ponder" and clip.asset == "c.mp4"


# --------------------------------------------------------------------------- #
# default_graph(): the bootstrap fully-connected synthetic library
# --------------------------------------------------------------------------- #
def test_default_graph_fully_connected_all_synthetic():
    dg = default_graph()
    assert len(dg.edges()) == len(POSES) * len(POSES)
    # every edge is a placeholder => the whole grid is the generation worklist
    assert len(dg.missing_edges()) == len(POSES) * len(POSES)
    assert dg.is_fully_baked() is False


def test_default_graph_self_loops_are_loopable_holds():
    dg = default_graph()
    for pose in POSES:
        hold = dg.hold_clip(pose)
        assert hold is not None and hold.is_hold and hold.loopable


def test_default_graph_any_pair_reachable():
    dg = default_graph()
    assert dg.path("leaving", "arriving") is not None
    assert dg.path("arriving", "ponder") is not None
