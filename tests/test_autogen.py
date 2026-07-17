"""test_autogen.py -- the no-cost autogen layer: proposal queue, budget, lint+labels,
and the atomic graph merge. No MJ, no network (the MJ client is imported lazily)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import autogen
from runtime import video_graph


def test_add_proposal_dedup_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(autogen, "PROPOSALS", tmp_path / "proposals.json")
    p = {"character": "phineas", "label": "reading", "still_prompt": "x", "proposed_at": 100}
    id1 = autogen.add_proposal(p)
    id2 = autogen.add_proposal(p)                 # same (char,label) while pending -> dedup
    assert id1 == id2
    assert len(autogen.load_proposals()["proposals"]) == 1
    autogen.set_status(id1, "approved")
    assert [x["id"] for x in autogen._by_status("approved")] == [id1]


def test_budget_caps_per_day(tmp_path, monkeypatch):
    monkeypatch.setattr(autogen, "BUDGET", tmp_path / "budget.json")
    monkeypatch.setattr(autogen, "DAILY_CAP", 2)
    assert autogen.budget_left("phineas") == 2
    autogen._spend("phineas")
    assert autogen.budget_left("phineas") == 1
    autogen._spend("phineas")
    assert autogen.budget_left("phineas") == 0


def test_plan_lints_and_derives_labels():
    v, lab = autogen.plan({
        "character": "phineas", "label": "reading", "hub": "anchor",
        "still_prompt": "a tragedian reading a letter by candlelight",
        "transition_motion": "he lifts and unfolds a letter",
        "idles": [{"id": "reading_0", "motion": "he reads intently"}],
        "negatives": ["text", "naked"]})
    assert v["ok"] is True
    assert "naked" not in v["safe_negatives"]            # body word stripped from negatives
    assert lab["fwd"] == "anchor_to_reading" and lab["rev"] == "reading_to_anchor"


def test_plan_rejects_unsafe_still():
    v, _ = autogen.plan({"character": "phineas", "label": "x", "hub": "anchor",
                         "still_prompt": "a nude figure reclining", "transition_motion": "", "idles": []})
    assert v["ok"] is False


def test_autogen_additions_is_atomic(monkeypatch):
    spec = {"characters": {"phineas": {"poses": {
        "reading": {"node_image": "data/gen/phineas_reading.png", "still_prompt": "x", "hub": "anchor",
                    "transition": {"label": "anchor_to_reading", "reverse_label": "reading_to_anchor", "motion": "m"},
                    "idles": [{"id": "reading_0", "motion": "i"}]}}}}}
    monkeypatch.setattr(video_graph, "_load_autogen", lambda: spec)
    # every declared edge has a gif -> pose is READY and merges
    monkeypatch.setattr(video_graph, "_variant_gifs",
                        lambda c, l: [(0, "data/clips/_proto/%s_%s_v0.gif" % (c, l))])
    nodes, edges, skipped = video_graph._autogen_additions()
    assert "reading" in nodes.get("phineas", {})
    assert {"anchor_to_reading", "reading_to_anchor", "reading_0"} <= {e[2] for e in edges}
    assert skipped == []
    # a missing gif -> the WHOLE pose is skipped (never half-merged -> never strands the walk)
    monkeypatch.setattr(video_graph, "_variant_gifs", lambda c, l: [])
    nodes2, edges2, skipped2 = video_graph._autogen_additions()
    assert nodes2 == {} and edges2 == [] and "phineas:reading" in skipped2


# --------------------------------------------------------------- web-not-star connectivity

def _gif_all(c, l):
    return [(0, "data/clips/_proto/%s_%s_v0.gif" % (c, l))]


def _reading_spec(extra=None):
    pose = {"node_image": "data/gen/phineas_reading.png", "still_prompt": "x", "hub": "anchor",
            "transition": {"label": "anchor_to_reading", "reverse_label": "reading_to_anchor", "motion": "m"},
            "idles": [{"id": "reading_0", "motion": "i"}]}
    if extra is not None:
        pose["extra_links"] = extra
    return {"characters": {"phineas": {"poses": {"reading": pose}}}}


def test_plan_lints_extra_link_motions():
    """An unsafe motion in the OPTIONAL second path must fail the whole proposal (same mj_safe
    pass as the hub motions) -- the web link can't smuggle an unsafe submit past the gate."""
    v, _ = autogen.plan({
        "character": "phineas", "label": "reading", "hub": "anchor",
        "still_prompt": "a tragedian reading a letter", "transition_motion": "he lifts a letter",
        "idles": [{"id": "reading_0", "motion": "he reads"}],
        "extra_links": [{"sibling": "glower", "motion": "he strips off his nude clothing", "reverse_motion": ""}]})
    assert v["ok"] is False


def test_autogen_additions_emits_extra_links(monkeypatch):
    """A merged pose with a present, non-dangling sibling link emits BOTH extra transition edges
    -- this is the star->web mesh (glower<->reading, on top of the anchor<->reading spoke)."""
    extra = [{"sibling": "glower", "label": "glower_to_reading",
              "reverse_label": "reading_to_glower", "motion": "lm"}]
    monkeypatch.setattr(video_graph, "_load_autogen", lambda: _reading_spec(extra))
    monkeypatch.setattr(video_graph, "_variant_gifs", _gif_all)   # every label has a gif
    nodes, edges, skipped = video_graph._autogen_additions()
    labels = {e[2] for e in edges}
    assert {"anchor_to_reading", "reading_to_anchor", "glower_to_reading", "reading_to_glower"} <= labels
    # the extra edges run between the real sibling pose and the new pose (both directions)
    by_label = {e[2]: (e[3], e[4]) for e in edges}
    assert by_label["glower_to_reading"] == ("glower", "reading")
    assert by_label["reading_to_glower"] == ("reading", "glower")
    assert skipped == []


def test_autogen_additions_extra_link_optional_when_gif_missing(monkeypatch):
    """If the sibling-link clips never landed (gif missing) the pose STILL merges on its core
    hub edges -- a failed second path never costs us the whole pose."""
    extra = [{"sibling": "glower", "label": "glower_to_reading",
              "reverse_label": "reading_to_glower", "motion": "lm"}]
    core = {"anchor_to_reading", "reading_to_anchor", "reading_0"}
    monkeypatch.setattr(video_graph, "_load_autogen", lambda: _reading_spec(extra))
    monkeypatch.setattr(video_graph, "_variant_gifs", lambda c, l: _gif_all(c, l) if l in core else [])
    nodes, edges, skipped = video_graph._autogen_additions()
    labels = {e[2] for e in edges}
    assert "reading" in nodes.get("phineas", {})          # core pose merged
    assert core <= labels                                  # core edges present
    assert "glower_to_reading" not in labels and "reading_to_glower" not in labels


def test_autogen_additions_extra_link_dangling_sibling_dropped(monkeypatch):
    """A link to a sibling that is NOT a real node is dropped -- one bad link can never emit a
    dangling edge that would fail build()'s walk-safety and sink the whole graph."""
    extra = [{"sibling": "ghost", "label": "ghost_to_reading",
              "reverse_label": "reading_to_ghost", "motion": "lm"}]
    monkeypatch.setattr(video_graph, "_load_autogen", lambda: _reading_spec(extra))
    monkeypatch.setattr(video_graph, "_variant_gifs", _gif_all)
    nodes, edges, skipped = video_graph._autogen_additions()
    assert "reading" in nodes.get("phineas", {})          # core pose still merges
    assert not any("ghost" in (e[3], e[4]) for e in edges)   # no edge touches the phantom node


def test_extra_link_is_walk_safe():
    """Two leaves cross-linked by an extra edge stay walk-safe (no ERRORs): the mesh only ADDS
    edges, so reach + return invariants still hold."""
    g = video_graph.VideoGraph()
    for pose in ("anchor", "reading", "glower"):
        g.add_node("phineas:%s" % pose, character="phineas", pose=pose,
                   image="x.png", gen_prompt="p")
    def tr(label, fr, to):
        g.add_edge(id="phineas/%s/v0" % label, character="phineas", kind="transition",
                   label=label, **{"from": "phineas:%s" % fr, "to": "phineas:%s" % to})
    def idle(label, at):
        g.add_edge(id="phineas/%s/v0" % label, character="phineas", kind="idle",
                   label=label, **{"from": "phineas:%s" % at, "to": "phineas:%s" % at})
    # anchor hub <-> two leaf poses
    tr("anchor_to_reading", "anchor", "reading"); tr("reading_to_anchor", "reading", "anchor")
    tr("anchor_to_glower", "anchor", "glower");   tr("glower_to_anchor", "glower", "anchor")
    for n in ("anchor", "reading", "glower"):
        idle("%s_idle" % n, n)
    # the WEB edge: reading <-> glower directly
    tr("glower_to_reading", "glower", "reading"); tr("reading_to_glower", "reading", "glower")
    errs = [m for lvl, m in g.validate() if lvl == "ERROR"]
    assert errs == []


def test_record_pose_persists_extra_links(tmp_path, monkeypatch):
    monkeypatch.setattr(autogen, "AUTOGEN", tmp_path / "autogen_poses.json")
    p = {"character": "phineas", "label": "reading", "still_prompt": "x",
         "transition_motion": "m", "idles": [{"id": "reading_0", "motion": "i"}]}
    links = [{"sibling": "glower", "label": "glower_to_reading",
              "reverse_label": "reading_to_glower", "motion": "lm"}]
    autogen._record_pose(p, "anchor", "anchor_to_reading", "reading_to_anchor", links)
    saved = autogen._load(autogen.AUTOGEN, {})["characters"]["phineas"]["poses"]["reading"]
    assert saved["extra_links"] == links


# ------------------------------------------------------------ proposal-side connectivity

def _fake_graph():
    """anchor hub + two leaf poses (each degree 1) so siblings exist and are leaves-first."""
    g = video_graph.VideoGraph()
    for pose in ("anchor", "glower", "swoon"):
        g.add_node("phineas:%s" % pose, character="phineas", pose=pose, image="x.png")
    for label, fr, to in (("a2g", "anchor", "glower"), ("g2a", "glower", "anchor"),
                          ("a2s", "anchor", "swoon"), ("s2a", "swoon", "anchor")):
        g.add_edge(id="phineas/%s/v0" % label, character="phineas", kind="transition",
                   label=label, **{"from": "phineas:%s" % fr, "to": "phineas:%s" % to})
    return g


def test_sibling_candidates_leaves_first_and_excludes():
    from director import heartbeat
    g = _fake_graph()
    cands = heartbeat._sibling_candidates(g, "phineas", hub="anchor")
    assert "anchor" not in cands                       # hub excluded
    assert set(cands) == {"glower", "swoon"}           # the two leaves offered
    cands2 = heartbeat._sibling_candidates(g, "phineas", hub="anchor", exclude={"swoon"})
    assert cands2 == ["glower"]                         # bedtime/excluded pose dropped


def test_propose_pose_authors_idle_count_and_link(tmp_path, monkeypatch):
    from director import heartbeat
    monkeypatch.setattr(autogen, "PROPOSALS", tmp_path / "proposals.json")
    monkeypatch.setattr(heartbeat.video_graph.VideoGraph, "load", staticmethod(_fake_graph))
    monkeypatch.setattr(heartbeat, "_current_pose", lambda g, c: ("phineas:anchor", 0))
    monkeypatch.setattr(heartbeat, "_load_char_spec", lambda c: {})
    monkeypatch.setattr(heartbeat.circadian, "bedtime_poses", lambda spec, c: set())
    monkeypatch.setattr(heartbeat.mj_safe, "check",
                        lambda *a, **k: {"ok": True, "hard": [], "soft": [], "safe_negatives": []})
    monkeypatch.setattr(heartbeat.llm, "complete_json", lambda *a, **k: {
        "label": "reading", "still_prompt": "a tragedian reading", "transition_motion": "t",
        "reverse_motion": "r", "idles": ["i1", "i2", "i3", "i4", "i5"],   # over-supplied
        "link_to": "glower", "link_motion": "lm", "link_reverse_motion": "lr", "reason": "why"})
    pid = heartbeat.propose_pose("phineas", model="m", dry_run=False, log=lambda *a: None)
    assert pid
    saved = autogen.load_proposals()["proposals"][0]
    assert len(saved["idles"]) == heartbeat.IDLE_COUNT          # capped to the named constant
    assert len(saved["extra_links"]) == 1 and saved["extra_links"][0]["sibling"] == "glower"
    assert saved["extra_links"][0]["motion"] == "lm"


def test_propose_pose_falls_back_to_hub_only_on_bad_link(tmp_path, monkeypatch):
    """If the LLM names a link_to that isn't an offered candidate, we ship hub-only (fail-safe to
    today's behavior) instead of inventing an edge to a pose that doesn't exist."""
    from director import heartbeat
    monkeypatch.setattr(autogen, "PROPOSALS", tmp_path / "proposals.json")
    monkeypatch.setattr(heartbeat.video_graph.VideoGraph, "load", staticmethod(_fake_graph))
    monkeypatch.setattr(heartbeat, "_current_pose", lambda g, c: ("phineas:anchor", 0))
    monkeypatch.setattr(heartbeat, "_load_char_spec", lambda c: {})
    monkeypatch.setattr(heartbeat.circadian, "bedtime_poses", lambda spec, c: set())
    monkeypatch.setattr(heartbeat.mj_safe, "check",
                        lambda *a, **k: {"ok": True, "hard": [], "soft": [], "safe_negatives": []})
    monkeypatch.setattr(heartbeat.llm, "complete_json", lambda *a, **k: {
        "label": "reading", "still_prompt": "a tragedian reading", "transition_motion": "t",
        "reverse_motion": "r", "idles": ["i1", "i2"],
        "link_to": "does_not_exist", "link_motion": "lm", "reason": "why"})
    pid = heartbeat.propose_pose("phineas", model="m", dry_run=False, log=lambda *a: None)
    assert pid
    saved = autogen.load_proposals()["proposals"][0]
    assert saved["extra_links"] == []
