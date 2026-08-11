"""test_provenance.py -- bi-temporal provenance over the REAL production graph.

Half of this file runs against `data/_realdata/video_graph.json`, the 259-node / 1472-edge
snapshot pulled off `hil`, and asserts real numbers. Synthetic fixtures appear only for the
two things the snapshot cannot contain: a `build()` actually running, and a store that has
been corrupted.

The real-data tests SKIP (never silently pass) when the snapshot is absent, so a fresh
clone with no data/ is still green.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from runtime import graph_provenance as gp
from runtime import video_graph as vg

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "_realdata" / "video_graph.json"

pytestmark = []


def _real():
    if not REAL.exists():
        pytest.skip("real production snapshot missing: %s (fresh clone -- data/ is gitignored)"
                    % REAL)
    return json.loads(REAL.read_text(encoding="utf-8"))


# ============================================================ REAL DATA: the snapshot
def test_real_snapshot_is_the_graph_we_think_it_is():
    d = _real()
    assert d.get("schema") == "living-portrait.video-graph/v3"
    assert len(d["nodes"]) == 259
    assert len(d["edges"]) == 1472
    kinds = {}
    for e in d["edges"]:
        kinds[e.get("kind")] = kinds.get(e.get("kind"), 0) + 1
    assert kinds == {"idle": 768, "transition": 704}
    # the gap this whole module exists to close: not one time field anywhere in it
    assert not any("created" in n or "invalidated" in n for n in d["nodes"].values())
    assert not any("created" in e or "invalidated" in e for e in d["edges"])


def test_real_created_stamps_only_where_an_artifact_actually_exists(capsys):
    """The load-bearing honesty property: a node/edge is dated ONLY when its still or gif is
    on this disk. Cross-checked against a completely different mechanism (os.path.exists per
    path) rather than against the scandir table that produced the answer."""
    d = _real()
    g = vg.VideoGraph(dict(d["nodes"]), list(d["edges"]))
    stats = gp.stamp(g.nodes, g.edges, root=ROOT)

    truth_n = {nid for nid, n in g.nodes.items()
               if n.get("image") and os.path.exists(str(ROOT / n["image"]))}
    truth_e = {e["id"] for e in g.edges
               if e.get("gif") and os.path.exists(str(ROOT / e["gif"]))}
    got_n = {nid for nid, n in g.nodes.items() if "created" in n}
    got_e = {e["id"] for e in g.edges if "created" in e}

    assert got_n == truth_n
    assert got_e == truth_e
    assert stats["nodes"] == 259 and stats["edges"] == 1472
    assert stats["nodes_created"] == len(truth_n)
    assert stats["edges_created"] == len(truth_e)
    assert stats["nodes_created"] > 0 and stats["edges_created"] > 0

    now = time.time()
    for n in g.nodes.values():
        if "created" in n:
            assert 0 < n["created"] <= now + 1
            assert n["created"] == os.stat(str(ROOT / n["image"])).st_mtime
    for e in g.edges:
        if "created" in e:
            assert e["created"] == os.stat(str(ROOT / e["gif"])).st_mtime

    stamped = sorted(n["created"] for n in g.nodes.values() if "created" in n)
    stamped += sorted(e["created"] for e in g.edges if "created" in e)
    span = ((max(stamped) - min(stamped)) / 86400.0) if stamped else 0.0
    with capsys.disabled():
        print("\n  REAL provenance coverage on this box:")
        print("    nodes dated %4d / %4d  (%.1f%% of stills resolvable)"
              % (stats["nodes_created"], stats["nodes"],
                 100.0 * stats["nodes_created"] / stats["nodes"]))
        print("    edges dated %4d / %4d  (%.1f%% of gifs resolvable)"
              % (stats["edges_created"], stats["edges"],
                 100.0 * stats["edges_created"] / stats["edges"]))
        print("    unresolvable: %d stills + %d gifs (production artifacts that live on hil)"
              % (stats["nodes"] - stats["nodes_created"], stats["edges"] - stats["edges_created"]))
        print("    dated artifacts span %.1f days, oldest %s, newest %s"
              % (span,
                 time.strftime("%Y-%m-%d", time.localtime(min(stamped))) if stamped else "-",
                 time.strftime("%Y-%m-%d", time.localtime(max(stamped))) if stamped else "-"))


def test_real_a_stale_stamp_cannot_outlive_its_artifact():
    """Provenance must never assert something the filesystem does not support. Pre-poison
    every record with a fabricated date; stamping has to REMOVE it wherever the artifact is
    gone, not leave it standing."""
    d = _real()
    nodes = {k: dict(v, created=1.0) for k, v in d["nodes"].items()}
    edges = [dict(e, created=1.0) for e in d["edges"]]
    gp.stamp(nodes, edges, root=ROOT)
    for n in nodes.values():
        assert n.get("created") != 1.0
        if "created" in n:
            assert os.path.exists(str(ROOT / n["image"]))
    for e in edges:
        assert e.get("created") != 1.0


def test_real_scandir_and_per_file_stat_agree(monkeypatch):
    """`mtimes` takes a directory-listing shortcut over the two artifact dirs (23x faster,
    measured). If that ever disagreed with plain os.stat the whole time dimension would be
    quietly wrong, so both paths are run over all 1731 real references and compared."""
    d = _real()
    refs = [n.get("image") for n in d["nodes"].values()] + [e.get("gif") for e in d["edges"]]
    fast = gp.mtimes(refs, root=ROOT)
    monkeypatch.setattr(gp, "SCANDIR_MIN_REFS", 10 ** 9)   # force the per-file stat path
    slow = gp.mtimes(refs, root=ROOT)
    assert fast == slow
    assert len(fast) > 0


def test_real_stamp_is_fast_enough_for_a_tick(capsys):
    d = _real()
    g = vg.VideoGraph(dict(d["nodes"]), list(d["edges"]))
    gp.stamp(g.nodes, g.edges, root=ROOT)             # warm the OS dir cache
    t0 = time.perf_counter()
    gp.stamp(g.nodes, g.edges, root=ROOT)
    dt = time.perf_counter() - t0
    with capsys.disabled():
        print("\n  REAL stamp over 259 nodes + 1472 edges: %.1f ms" % (dt * 1000.0))
    assert dt < 1.0, "%.0f ms is too slow to sit in a graph load" % (dt * 1000.0)


# ============================================================ REAL DATA: invalidation
def test_real_lost_edges_become_history_and_survive_a_reload(tmp_path, capsys):
    """The audit's silent-forgetting bug, reproduced on the real graph: 25 real clips go
    missing, plus a whole pose. The rebuilt graph must not contain them, the store must,
    and a fresh process reading the store back must be able to answer what the character
    could do before."""
    d = _real()
    prev_nodes, prev_edges = d["nodes"], d["edges"]

    doomed_edges = [e["id"] for e in prev_edges[300:325]]
    doomed_node = sorted(prev_nodes)[-1]
    new_nodes = {k: v for k, v in prev_nodes.items() if k != doomed_node}
    new_edges = [e for e in prev_edges if e["id"] not in set(doomed_edges)]
    assert len(new_edges) == 1472 - 25 and len(new_nodes) == 258

    t_death = 1_754_000_000.0
    store, stats = gp.reconcile(prev_nodes, prev_edges, new_nodes, new_edges,
                                store=gp.load_store(tmp_path / "nope.json"), now=t_death)
    assert stats["edges_invalidated"] == 25
    assert stats["nodes_invalidated"] == 1
    assert stats["edges_revived"] == 0

    p = tmp_path / "provenance.json"
    assert gp.save_store(store, path=p) is True
    reread = gp.load_store(p)                          # a different process would see this
    assert set(gp.invalidated_edge_ids(reread)) == set(doomed_edges)

    live = vg.VideoGraph(dict(new_nodes), list(new_edges))
    assert {e["id"] for e in live.edges} & set(doomed_edges) == set()

    h_nodes, h_edges = gp.merge_history(new_nodes, new_edges, reread)
    assert len(h_edges) == 1472 and len(h_nodes) == 259
    back = {e["id"]: e for e in h_edges if e.get("invalidated")}
    assert set(back) == set(doomed_edges)
    original = {e["id"]: e for e in prev_edges}
    for eid, e in back.items():
        assert e["invalidated"] == t_death
        # the FULL record came back, not just the id -- history can reconstruct the clip
        for k in ("from", "to", "kind", "gif", "motion_prompt", "character"):
            assert e.get(k) == original[eid].get(k)
    with capsys.disabled():
        print("\n  REAL invalidation: 25 edges + 1 node removed from a 1472/259 graph ->"
              " live view %d edges, history view %d edges" % (len(new_edges), len(h_edges)))


def test_real_history_view_is_never_walkable(tmp_path):
    """The hard safety property. Even holding a history graph, every walk helper refuses the
    dead: they all funnel through `edges_from`."""
    d = _real()
    prev_nodes, prev_edges = d["nodes"], d["edges"]
    # kill every edge leaving one busy node, so the walk MUST notice
    victim = max({e["from"] for e in prev_edges},
                 key=lambda n: sum(1 for e in prev_edges if e["from"] == n))
    doomed = {e["id"] for e in prev_edges if e["from"] == victim}
    assert len(doomed) > 5
    new_edges = [e for e in prev_edges if e["id"] not in doomed]
    store, _ = gp.reconcile(prev_nodes, prev_edges, prev_nodes, new_edges,
                            store=gp.load_store(tmp_path / "x.json"), now=1.0)

    h_nodes, h_edges = gp.merge_history(prev_nodes, new_edges, store)
    g = vg.VideoGraph(h_nodes, h_edges)
    assert len(g.edges) == 1472                  # the dead ARE in the object...
    assert len(g.dead_edges()) == len(doomed)
    assert g.edges_from(victim) == []            # ...and none of them is reachable
    assert g.idle_edges(victim) == [] and g.transition_edges(victim) == []
    assert g.random_edge(victim) is None
    assert g.next_edge(victim) is None
    for e in g.live_edges():
        assert e["id"] not in doomed


def test_real_load_defaults_to_the_live_view_and_reads_no_store(monkeypatch):
    """Every existing caller does `VideoGraph.load()` with no arguments. It must keep
    getting exactly the live graph -- and must not even consult the tombstone store, which
    is what keeps the default path free of extra I/O."""
    _real()
    called = []
    monkeypatch.setattr(gp, "load_store", lambda *a, **k: called.append(1) or gp._empty_store())
    g = vg.VideoGraph.load(REAL)                  # positional path, the pre-existing signature
    assert called == []
    assert g.history is False
    assert len(g.nodes) == 259 and len(g.edges) == 1472
    assert g.live_edges() == g.edges and g.dead_edges() == []
    assert g.provenance["nodes"] == 259 and g.provenance["edges_created"] > 0

    g2 = vg.VideoGraph.load(path=REAL, provenance=False)
    assert called == []
    assert not any("created" in n for n in g2.nodes.values())


def test_real_load_history_opt_in_merges_the_dead(tmp_path, monkeypatch):
    d = _real()
    doomed = [e["id"] for e in d["edges"][:7]]
    store, _ = gp.reconcile(d["nodes"], d["edges"], d["nodes"],
                            [e for e in d["edges"] if e["id"] not in set(doomed)],
                            store=gp._empty_store(), now=99.0)
    p = tmp_path / "prov.json"
    gp.save_store(store, path=p)
    # the snapshot on disk is the FULL graph, so the tombstoned ids are also live in it:
    # merge must not duplicate them.
    g = vg.VideoGraph.load(REAL, history=True, store_path=p)
    assert g.history is True
    assert len(g.edges) == 1472
    assert g.dead_edges() == []


# ============================================================ SYNTHETIC: build() twice
def _mini(tmp_path, monkeypatch):
    """A walk-safe two-pose character on a throwaway ROOT. Synthetic because the real
    snapshot cannot contain a running `build()` -- it is the OUTPUT of one."""
    root = tmp_path / "proj"
    proto = root / "data" / "clips" / "_proto"
    gen = root / "data" / "gen"
    proto.mkdir(parents=True)
    gen.mkdir(parents=True)
    for pose in ("anchor", "away"):
        (gen / ("testy_%s.png" % pose)).write_bytes(b"png")
    for label in ("t_breathe", "t_out", "t_back", "t_pace"):
        (proto / ("testy_%s_v0.gif" % label)).write_bytes(b"gif")

    monkeypatch.setattr(vg, "ROOT", root)
    monkeypatch.setattr(vg, "PROTO", proto)
    monkeypatch.setattr(vg, "GRAPH_PATH", root / "data" / "clips" / "video_graph.json")
    monkeypatch.setattr(gp, "STORE_PATH", root / "data" / "graph" / "provenance.json")
    monkeypatch.setattr(vg, "NODE_SPECS", {"testy": {
        "anchor": {"image": "data/gen/testy_anchor.png", "gen_prompt": "an anchor"},
        "away": {"image": "data/gen/testy_away.png", "gen_prompt": "away"}}})
    monkeypatch.setattr(vg, "EDGE_SPECS", [
        ("testy", "idle", "t_breathe", "anchor", "anchor", "breathes", True, ""),
        ("testy", "transition", "t_out", "anchor", "away", "walks off", False, ""),
        ("testy", "transition", "t_back", "away", "anchor", "walks back", False, ""),
        ("testy", "idle", "t_pace", "away", "away", "paces", True, ""),
    ])
    return root, proto, gen


def test_build_twice_invalidation_survives_the_rebuild(tmp_path, monkeypatch, capsys):
    """The requirement in one test: build, lose a clip, rebuild -- and the edge is still
    knowable afterwards even though video_graph.json was regenerated from scratch."""
    root, proto, gen = _mini(tmp_path, monkeypatch)

    g1 = vg.build()
    assert len(g1.edges) == 4 and len(g1.nodes) == 2
    assert g1.provenance["edges_created"] == 4 and g1.provenance["nodes_created"] == 2
    assert gp.load_store() == gp._empty_store() or not gp.load_store()["edges"]

    (proto / "testy_t_pace_v0.gif").unlink()               # a clip disappears from disk
    g2 = vg.build()
    assert len(g2.edges) == 3
    assert "testy/t_pace/v0" not in {e["id"] for e in g2.edges}   # gone from the LIVE graph

    on_disk = json.loads((root / "data" / "clips" / "video_graph.json").read_text("utf-8"))
    assert len(on_disk["edges"]) == 3                       # ...and gone from the file the
    assert not any("created" in e or "invalidated" in e     # walker reads raw, with no
                   for e in on_disk["edges"])               # derived fields leaked into it

    store = gp.load_store()
    assert set(store["edges"]) == {"testy/t_pace/v0"}       # but remembered here
    dead = store["edges"]["testy/t_pace/v0"]
    assert dead["record"]["from"] == "testy:away" and dead["record"]["kind"] == "idle"
    t_death = dead["invalidated"]
    assert t_death > 0

    g3 = vg.build()                                          # a third build, still missing
    assert gp.load_store()["edges"]["testy/t_pace/v0"]["invalidated"] == t_death, \
        "a still-missing edge must keep its ORIGINAL death date, not be re-dated every build"
    assert len(g3.edges) == 3

    hist = vg.VideoGraph.load(history=True)
    ids = {e["id"] for e in hist.edges}
    assert ids == {"testy/t_breathe/v0", "testy/t_out/v0", "testy/t_back/v0", "testy/t_pace/v0"}
    assert [e["id"] for e in hist.dead_edges()] == ["testy/t_pace/v0"]
    assert "testy/t_pace/v0" not in {e["id"] for e in hist.edges_from("testy:away")}
    with capsys.disabled():
        print("\n  build->lose->rebuild x2: live 3 edges, history 4, death date stable")


def test_build_revives_an_edge_whose_clip_comes_back(tmp_path, monkeypatch):
    root, proto, gen = _mini(tmp_path, monkeypatch)
    vg.build()
    (proto / "testy_t_pace_v0.gif").unlink()
    vg.build()
    assert gp.load_store()["edges"]

    (proto / "testy_t_pace_v0.gif").write_bytes(b"gif")     # regenerated
    g = vg.build()
    assert len(g.edges) == 4
    assert gp.load_store()["edges"] == {}, "a live edge must not also be a tombstone"
    assert vg.VideoGraph.load(history=True).dead_edges() == []


def test_build_that_refuses_to_save_records_no_deaths(tmp_path, monkeypatch):
    """Provenance follows the COMMITTED state. A build that trips walk-safety and refuses to
    save must not tombstone anything -- the graph that is actually running is the old one."""
    root, proto, gen = _mini(tmp_path, monkeypatch)
    vg.build()
    before = json.loads((root / "data" / "clips" / "video_graph.json").read_text("utf-8"))

    (proto / "testy_t_back_v0.gif").unlink()        # away: now no transition OUT -> E3 ERROR
    with pytest.raises(SystemExit):
        vg.build()
    assert gp.load_store()["edges"] == {}
    after = json.loads((root / "data" / "clips" / "video_graph.json").read_text("utf-8"))
    assert after == before


def test_build_stamps_created_from_the_real_gif_mtimes(tmp_path, monkeypatch):
    root, proto, gen = _mini(tmp_path, monkeypatch)
    old = time.time() - 10 * 86400
    os.utime(proto / "testy_t_out_v0.gif", (old, old))
    g = vg.build()
    by_id = {e["id"]: e for e in g.edges}
    assert abs(by_id["testy/t_out/v0"]["created"] - old) < 2
    assert by_id["testy/t_breathe/v0"]["created"] > old + 86400


def test_nodes_are_invalidated_too(tmp_path, monkeypatch):
    """A pose can vanish as well as a clip (a bedtime character regressing to partly
    generated drops its whole node set), and it is tombstoned the same way."""
    root, proto, gen = _mini(tmp_path, monkeypatch)
    vg.build()
    monkeypatch.setattr(vg, "NODE_SPECS", {"testy": {
        "anchor": {"image": "data/gen/testy_anchor.png", "gen_prompt": "an anchor"}}})
    monkeypatch.setattr(vg, "EDGE_SPECS", [
        ("testy", "idle", "t_breathe", "anchor", "anchor", "breathes", True, "")])
    g = vg.build()
    assert set(g.nodes) == {"testy:anchor"}
    store = gp.load_store()
    assert set(store["nodes"]) == {"testy:away"}
    assert store["nodes"]["testy:away"]["record"]["pose"] == "away"
    assert set(store["edges"]) == {"testy/t_out/v0", "testy/t_back/v0", "testy/t_pace/v0"}
    h = vg.VideoGraph.load(history=True)
    assert h.nodes["testy:away"]["invalidated"] > 0
    assert h.poses("testy") == ["testy:anchor", "testy:away"]     # history knows both


# ============================================================ SYNTHETIC: fail-soft edges
def test_missing_artifact_never_gets_a_fabricated_timestamp(tmp_path):
    nodes = {"a:x": {"image": "data/gen/not_here.png"}}
    edges = [{"id": "a/x/v0", "gif": "data/clips/_proto/not_here.gif"}, {"id": "a/y/v0"}]
    stats = gp.stamp(nodes, edges, root=tmp_path)
    assert "created" not in nodes["a:x"]
    assert all("created" not in e for e in edges)
    assert stats == {"nodes": 1, "nodes_created": 0, "edges": 2, "edges_created": 0}
    assert gp.mtimes(["data/gen/not_here.png"], root=tmp_path) == {}


def test_a_corrupt_store_degrades_to_no_history_not_an_exception(tmp_path):
    p = tmp_path / "prov.json"
    p.write_text("{ this is not json", encoding="utf-8")
    assert gp.load_store(p) == gp._empty_store()
    p.write_text('["a list, not an object"]', encoding="utf-8")
    assert gp.load_store(p) == gp._empty_store()
    p.write_text('{"edges": "not a dict", "nodes": {"n": "not a record"}}', encoding="utf-8")
    s = gp.load_store(p)
    assert s["edges"] == {} and s["nodes"] == {}
    n, e = gp.merge_history({"a": {}}, [{"id": "x"}], s)
    assert n == {"a": {}} and e == [{"id": "x"}]


def test_load_survives_a_provenance_module_that_is_not_there(monkeypatch, tmp_path):
    """The import is defensive for a reason: video_graph is imported two different ways.
    With no provenance module at all, a graph load must still return a graph."""
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"nodes": {"a:x": {"image": "i.png"}},
                             "edges": [{"id": "a/x/v0", "from": "a:x", "to": "a:x"}]}),
                 encoding="utf-8")
    monkeypatch.setattr(vg, "_prov", None)
    g = vg.VideoGraph.load(p, history=True)
    assert len(g.nodes) == 1 and len(g.edges) == 1
    assert g.provenance == {} and g.history is False
    assert g.edges_from("a:x")


def test_load_of_a_missing_file_is_still_an_empty_graph(tmp_path):
    g = vg.VideoGraph.load(tmp_path / "nothing.json")
    assert g.nodes == {} and g.edges == []
    assert g.live_edges() == []


def test_save_strips_derived_fields_even_from_a_history_view(tmp_path):
    g = vg.VideoGraph({"a:x": {"image": "i.png", "created": 5.0}},
                      [{"id": "a/x/v0", "created": 5.0, "invalidated": 6.0}])
    out = tmp_path / "g.json"
    g.save(out)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["nodes"]["a:x"] == {"image": "i.png"}
    assert d["edges"][0] == {"id": "a/x/v0"}
    assert g.nodes["a:x"]["created"] == 5.0        # the in-memory view is not mutated


def test_tombstones_are_bounded(tmp_path):
    prev = {"n%d" % i: {"pose": str(i)} for i in range(50)}
    store, stats = gp.reconcile(prev, [], {}, [], store=gp._empty_store(),
                                now=1000.0, max_tombstones=10)
    assert len(store["nodes"]) == 10
    assert stats["nodes_invalidated"] == 50 and stats["nodes_dead"] == 10


def test_prune_keeps_the_most_recent_deaths():
    tombs = {"a": {"invalidated": 1.0}, "b": {"invalidated": 9.0}, "c": {"invalidated": 5.0}}
    assert set(gp._prune(dict(tombs), 2)) == {"b", "c"}
    assert set(gp._prune(dict(tombs), 99)) == {"a", "b", "c"}


def test_summary_of_an_empty_store_is_zeroes_not_none():
    s = gp.summary(gp._empty_store(), now=100.0)
    assert s == {"nodes_dead": 0, "edges_dead": 0,
                 "oldest_death_age_days": 0.0, "newest_death_age_days": 0.0}
    store, _ = gp.reconcile({"a": {}}, [], {}, [], store=gp._empty_store(), now=0.0)
    s2 = gp.summary(store, now=86400.0)
    assert s2["nodes_dead"] == 1 and abs(s2["oldest_death_age_days"] - 1.0) < 1e-9
