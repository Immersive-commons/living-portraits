"""test_edge_style.py -- manner/valence typing, and its wiring into the policy walk.

The rule this file is built on: the REAL graph is the fixture. `data/_realdata/video_graph.json`
is the production snapshot (259 nodes, 1,472 edges, every edge carrying the prose that was
actually sent to the video generator), and every coverage claim below is asserted against it
and PRINTED, because a lexicon that types 70% and says so is worth more than one that types
100% by calling everything neutral. Synthetic edges appear only for cases the snapshot does
not contain (a prompt with no manner at all, a lone leaf pose).

Real-data tests SKIP with a reason when the snapshot is absent -- they never silently pass.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from runtime import edge_style as es
from runtime import policy

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "_realdata" / "video_graph.json"

# The snapshot's own shape, asserted so a swapped-out fixture is caught rather than
# silently changing every coverage number below.
REAL_EDGES = 1472
REAL_TRANSITIONS = 704
REAL_IDLES = 768


def _real_edges():
    if not REAL.exists():
        pytest.skip("real graph snapshot missing: %s (run against data/_realdata/, not a stub)"
                    % REAL)
    try:
        graph = json.loads(REAL.read_text(encoding="utf-8"))
    except Exception as exc:                                    # pragma: no cover
        pytest.skip("real graph snapshot unreadable (%s): %s" % (REAL, exc))
    edges = graph.get("edges") or []
    if not edges:                                               # pragma: no cover
        pytest.skip("real graph snapshot has no edges: %s" % REAL)
    return edges


@pytest.fixture(scope="module")
def real_edges():
    return _real_edges()


@pytest.fixture(scope="module")
def real_index(real_edges):
    return es.index(real_edges)


# --------------------------------------------------------------------------- derive


def test_derive_reads_the_manner_out_of_real_prose():
    """Two clips that are both `kind: transition` and identical to every other term in
    policy.weigh. The prose is the only thing that tells them apart."""
    storm = es.derive("Body snaps forward from the relaxed slump of sleep, hand jerking up "
                      "to claw at the air as the head whips up in a sudden gasp")
    saunter = es.derive("the tragedian slowly settles back, easing gently into the seated "
                        "rest pose as the candle gutters, a soft unhurried sway")
    assert storm.manner == "surge"
    assert saunter.manner == "drift"
    assert storm.manner != saunter.manner


def test_unknown_prose_is_none_not_a_bucket():
    # real untyped lines from the snapshot: micro-gestures that name no manner at all
    for text in ("", None, "A single tear drips from my cheek",
                 "Checking the bottom of the mug for coffee grounds with a shake.",
                 "the frame sits there"):
        assert es.derive(text).manner is None
    assert es.derive("").valence == 0.0


def test_affect_words_move_valence_off_the_manner_base():
    plain = es.derive("the figure slumps, shoulders sagging, spine wilting")
    bitter = es.derive("the figure slumps with resentful fury, shoulders sagging in "
                       "bitter despair, spine wilting")
    assert plain.manner == bitter.manner == "collapse"
    assert bitter.valence < plain.valence
    assert -1.0 <= bitter.valence <= 1.0


def test_valence_signs_match_the_manner():
    assert es.derive("the spine straightens, the body rises, standing upright").valence > 0
    assert es.derive("the body slumps, sagging, crumpling downward").valence < 0


def test_reverse_marker_never_reaches_the_cues():
    body = "the tragedian slowly settles back, easing gently into the seated rest pose"
    assert es.derive("(reverse) " + body) == es.derive(body)
    assert es.derive("(reverse of move-behind) " + body) == es.derive(body)
    assert es.strip_reverse("(reverse) x") == ("x", True)
    assert es.strip_reverse("x") == ("x", False)


def test_invert_flips_only_the_directional_manners():
    rise = es.derive("the spine straightens as the body rises, standing upright")
    assert es.invert(rise).manner == "collapse"
    assert es.invert(es.invert(rise)).manner == "rise"          # involutive
    assert es.invert(rise).valence < rise.valence               # tone follows the manner
    flourish = es.derive("a grand theatrical bow, sweeping arms, dramatic and deliberate")
    assert flourish.manner == "flourish"
    assert es.invert(flourish) == flourish                      # its own inverse
    assert es.invert(es.NEUTRAL).manner is None


# ------------------------------------------------------- REAL DATA: coverage + histogram


def test_real_snapshot_is_the_graph_we_measured(real_edges):
    kinds = {}
    for e in real_edges:
        kinds[e.get("kind")] = kinds.get(e.get("kind"), 0) + 1
    assert len(real_edges) == REAL_EDGES
    assert kinds.get("transition") == REAL_TRANSITIONS
    assert kinds.get("idle") == REAL_IDLES


def test_real_coverage_is_reported_honestly(real_edges, capsys):
    """The headline number, printed so a regression is visible in the test output rather
    than only in an assertion message."""
    cov = es.coverage(real_edges)
    with capsys.disabled():
        print("\n  REAL COVERAGE  %d/%d edges typed (%.1f%%), %d untyped"
              % (cov["typed"], cov["n"], 100.0 * cov["share"], cov["untyped"]))
        for kind in sorted(cov["by_kind"]):
            k = cov["by_kind"][kind]
            print("    %-11s %4d edges, %4d typed (%.1f%%)"
                  % (kind, k["n"], k["typed"], 100.0 * k["share"]))
        print("  MANNER DISTRIBUTION")
        for manner, n in sorted(cov["manners"].items(), key=lambda t: -t[1]):
            print("    %-9s %4d  (%.1f%% of typed)" % (manner, n, 100.0 * n / cov["typed"]))
        print("    %-9s %4d  (untypeable by this lexicon)" % ("(none)", cov["untyped"]))

    # Measured 2026-08-10: 1030/1472 = 70.0% overall, 636/704 = 90.3% of transitions,
    # 394/768 = 51.3% of idles. Idles are micro-gestures ("a single tear drips from my
    # cheek") that frequently name no manner at all, and that is reported, not padded.
    assert cov["n"] == REAL_EDGES
    assert cov["typed"] >= 1000, "manner coverage regressed below the measured 1030"
    assert cov["share"] >= 0.65
    assert cov["by_kind"]["transition"]["share"] >= 0.88, "transitions are the ones that move"
    assert cov["by_kind"]["idle"]["share"] >= 0.45
    assert cov["untyped"] > 0, "100% coverage would mean the lexicon stopped saying 'I don't know'"


def test_every_manner_in_the_vocabulary_earns_its_place(real_index, real_edges):
    """A closed vocabulary chosen from the real corpus: each entry must actually occur in
    it, or it is invention rather than derivation."""
    hist = es.histogram(real_index)
    assert set(hist) <= set(es.MANNERS)
    missing = [m for m in es.MANNERS if not hist.get(m)]
    assert not missing, "manners that never fire on the real graph: %s" % missing


def test_real_manner_distribution_is_not_one_dominant_bucket(real_index):
    hist = es.histogram(real_index)
    top = max(hist.values())
    assert top / float(sum(hist.values())) < 0.35, \
        "one manner swallowing the graph means the lexicon is not discriminating: %s" % hist


def test_real_valences_stay_in_range_and_actually_spread(real_index):
    vals = [s.valence for s in real_index.values()]
    assert all(-1.0 <= v <= 1.0 for v in vals)
    assert min(vals) < -0.4 and max(vals) > 0.4
    assert len({round(v, 2) for v in vals}) > 10, "valence collapsed onto the manner bases"


def test_histogram_accepts_raw_real_edges(real_edges):
    assert sum(es.histogram(real_edges).values()) > 0


# ------------------------------------------------------- REAL DATA: the reverse-prose trap


def test_reverse_prose_is_detected_and_corrected_on_the_real_graph(real_edges, capsys):
    """`video_graph.py:423,480,501` writes a return leg's prompt as `"(reverse) " + motion`
    -- the FORWARD clip's text. Reading it literally types a rise as a collapse."""
    flipped = es._copied_prose_ids(real_edges)
    styles = es.index(real_edges)
    typed = [i for i in flipped if i in styles]
    changed = [i for i in typed if styles[i].manner in es.INVERSE_MANNER]
    with capsys.disabled():
        print("\n  REVERSE-PROSE  %d edges provably carry their twin's text; %d of those "
              "are typed" % (len(flipped), len(typed)))
        print("    %d landed on a directional manner (inversion changed it)" % len(changed))
        print("    %d landed on a direction-neutral manner (inversion is a no-op; the "
              "type is abruptness-correct, direction-approximate)" % (len(typed) - len(changed)))
    assert len(flipped) >= 300, "the trap is real and large; detection regressed"

    # detection is by PROOF (marked + an unmarked opposite-direction twin with identical
    # text), never by the marker alone -- hand-authored "(reverse)" lines describe their
    # own direction and must survive untouched.
    by_id = {e.get("id"): e for e in real_edges}
    for eid in flipped:
        text, marked = es.strip_reverse(by_id[eid].get("motion_prompt"))
        assert marked and text


def test_index_inversion_differs_from_reading_the_prose_literally(real_edges):
    """If index() were just derive() per edge, this whole correction would be a no-op."""
    naive = {e.get("id"): es.for_edge(e) for e in real_edges if es.for_edge(e).manner}
    corrected = es.index(real_edges)
    differing = [i for i in corrected if naive.get(i) and naive[i].manner != corrected[i].manner]
    assert len(differing) >= 100, "index() must correct the copied-prose edges, not echo derive()"


def test_index_keys_only_typed_edges(real_edges, real_index):
    ids = {e.get("id") for e in real_edges}
    assert set(real_index) <= ids
    assert len(real_index) < len(ids)               # untyped edges are absent, not "neutral"
    assert all(s.manner for s in real_index.values())


# ------------------------------------------------------- affinity: a preference, never a veto


def test_affinity_is_exactly_one_when_there_is_no_opinion():
    styled = es.derive("the figure slumps, sagging heavily downward")
    assert es.affinity(None, "weary") == 1.0
    assert es.affinity(es.NEUTRAL, "weary") == 1.0
    assert es.affinity(styled, None) == 1.0
    assert es.affinity(styled, "not-a-band") == 1.0
    assert es.affinity(styled, "weary", strength=0.0) == 1.0


def test_affinity_prefers_the_manner_the_band_would_choose():
    storm = es.derive("the body lunges, snapping forward with a sudden violent thrust")
    saunter = es.derive("slowly and gently the figure drifts, easing into a soft sway")
    assert storm.manner == "surge" and saunter.manner == "drift"
    assert es.affinity(saunter, "weary") > es.affinity(storm, "weary")
    assert es.affinity(storm, "restless") > es.affinity(saunter, "restless")


def test_affinity_never_reaches_zero_on_any_real_edge(real_index):
    """The load-bearing safety property: styling is a weight, so the worst any manner can
    do under the worst band is lose share -- it can never eliminate an edge."""
    worst = 1.0
    for style in real_index.values():
        for band in es.STYLE_AFFINITY:
            a = es.affinity(style, band)
            assert a >= es.STYLE_FLOOR > 0.0
            assert a <= es.STYLE_CEIL
            worst = min(worst, a)
    assert worst < 1.0, "if nothing is ever discouraged the layer does nothing"


def test_affinity_strength_fades_the_whole_layer_out():
    storm = es.derive("the body lunges, snapping forward with a sudden violent thrust")
    full = es.affinity(storm, "weary", 1.0)
    half = es.affinity(storm, "weary", 0.5)
    assert full < half < 1.0
    assert es.affinity(storm, "weary", 0.0) == 1.0


def test_every_band_policy_can_produce_has_an_affinity_table():
    """The two modules must agree on the band vocabulary or the layer silently no-ops."""
    bands = set(policy.MOOD_BIAS) | set(policy._BAND)
    assert bands <= set(es.STYLE_AFFINITY)
    assert bands <= set(es.BAND_VALENCE)


# ------------------------------------------------------- policy wiring


def _pair():
    """One storming exit and one sauntering exit from the same pose -- identical in every
    term policy.weigh has ever had, so any difference is the manner."""
    return [
        {"id": "x/storm", "kind": "transition", "label": "h2a", "from": "hub", "to": "a",
         "motion_prompt": "the body lunges, snapping forward with a sudden violent thrust"},
        {"id": "x/saunter", "kind": "transition", "label": "h2b", "from": "hub", "to": "b",
         "motion_prompt": "slowly and gently the figure drifts, easing into a soft sway"},
    ]


def test_no_styles_argument_leaves_weights_byte_identical():
    out = _pair()
    for kwargs in ({}, {"mood": "weary"}, {"mood": "restless", "band": "restless"},
                   {"mood": "weary", "goal": "a", "route": "beeline"},
                   {"mood": "calm", "context_energy": "high"}):
        base = policy.weigh("hub", out, out, **kwargs)
        for styles in (None, {}, ()):
            same = policy.weigh("hub", out, out, styles=styles, **kwargs)
            assert [w for _, w in same] == [w for _, w in base], kwargs


def test_unstyleable_edges_leave_weights_byte_identical():
    """The default the mission pins: an edge whose prose names no manner behaves exactly
    as it does today even with the layer switched on."""
    out = [{"id": "n/1", "kind": "transition", "label": "h2a", "from": "hub", "to": "a",
            "motion_prompt": "a single tear drips from my cheek"},
           {"id": "n/2", "kind": "idle", "label": "hold", "from": "hub", "to": "hub",
            "motion_prompt": "the frame sits a moment"}]
    styles = es.index(out)
    assert styles == {}
    base = policy.weigh("hub", out, out, mood="weary")
    styled = policy.weigh("hub", out, out, mood="weary", styles=styles)
    assert [w for _, w in styled] == [w for _, w in base]


def test_a_mood_with_no_band_leaves_weights_byte_identical():
    out = _pair()
    styles = es.index(out)
    assert len(styles) == 2                                # the edges ARE typeable
    for mood in (None, "", "beige", "thinking about tuesday"):
        assert policy._style_band(mood, None) is None
        base = policy.weigh("hub", out, out, mood=mood)
        styled = policy.weigh("hub", out, out, mood=mood, styles=styles)
        assert [w for _, w in styled] == [w for _, w in base], mood


def test_weary_prefers_the_saunter_and_restless_prefers_the_storm():
    out = _pair()
    styles = es.index(out)
    weary = dict((e["id"], w) for e, w in policy.weigh("hub", out, out, mood="weary",
                                                       styles=styles))
    restless = dict((e["id"], w) for e, w in policy.weigh("hub", out, out, mood="restless",
                                                          styles=styles))
    assert weary["x/saunter"] > weary["x/storm"]
    assert restless["x/storm"] > restless["x/saunter"]


def test_free_text_mood_reaches_the_style_layer_through_the_energy_bucket():
    """The heartbeat writes "feral devotion", not "restless". _style_band must land it in
    the same bucket _mood_bias does, or the body reasons from two readings of one mood."""
    assert policy._style_band("feral devotion", None) == "high"
    assert policy._style_band("a hollow, guarded tenderness", None) == "low"
    assert policy._style_band("weary triumph", None) == "weary"        # exact token wins
    assert policy._style_band(None, "fixated") == "fixated"            # declared band wins
    out = _pair()
    styles = es.index(out)
    w = dict((e["id"], x) for e, x in policy.weigh("hub", out, out, mood="feral devotion",
                                                   styles=styles))
    assert w["x/storm"] > w["x/saunter"]


def test_style_never_overturns_the_anti_reverse_fix():
    """The bug fixes are load-bearing. A storming edge straight back where we came from,
    under the band that loves storming, must still lose to the sauntering alternative."""
    out = _pair()
    styles = es.index(out)
    w = dict((e["id"], x) for e, x in policy.weigh(
        "hub", out, out, mood="restless", styles=styles, prev_node="a"))
    assert w["x/storm"] < w["x/saunter"]


def test_style_never_overturns_the_goal_gradient():
    out = _pair()
    graph = out + [{"id": "x/b2goal", "kind": "transition", "label": "b2g",
                    "from": "b", "to": "goalP", "motion_prompt": ""}]
    styles = es.index(graph)
    # goal is reachable only through the SAUNTER edge; the storm-loving band must still walk it
    w = dict((e["id"], x) for e, x in policy.weigh(
        "hub", out, graph, mood="restless", goal="goalP", route="beeline", styles=styles))
    assert w["x/saunter"] > w["x/storm"]


def test_style_never_overturns_novelty():
    out = _pair()
    styles = es.index(out)
    w = dict((e["id"], x) for e, x in policy.weigh(
        "hub", out, out, mood="restless", styles=styles,
        recent_nodes=["a", "a", "a", "a"]))
    assert w["x/storm"] < w["x/saunter"]


def test_excluded_edges_stay_at_zero_and_a_leaf_never_strands():
    out = _pair()
    styles = es.index(out)
    w = policy.weigh("hub", out, out, mood="weary", styles=styles, exclude={"h2a"})
    assert dict((e["id"], x) for e, x in w)["x/storm"] == 0.0
    # everything masked -> the never-strand fallback still fires, untouched by styling
    flat = policy.weigh("hub", out, out, mood="weary", styles=styles,
                        exclude={"h2a", "h2b"})
    assert all(x == 1.0 for _, x in flat)
    lone = [out[0]]
    assert all(x > 0 for _, x in policy.weigh("hub", lone, lone, mood="weary", styles=styles))


def test_choose_passes_styles_through_and_returns_a_real_edge():
    out = _pair()
    styles = es.index(out)
    rng = random.Random(4)
    picks = [policy.choose("hub", out, out, mood="weary", styles=styles, rng=rng)["id"]
             for _ in range(400)]
    assert set(picks) == {"x/storm", "x/saunter"}          # preference, never a veto
    assert picks.count("x/saunter") > picks.count("x/storm")


# ------------------------------------------------------- REAL DATA: policy over the real graph


def test_styling_strands_nothing_on_the_real_graph(real_edges, real_index):
    """Walk every real pose that has exits, under every band, and assert styling alone
    never drives a candidate set to all-zero. This is the property that would take the
    panels down if it were false."""
    out_by_node = {}
    for e in real_edges:
        out_by_node.setdefault(e.get("from"), []).append(e)
    checked = 0
    for node, out in out_by_node.items():
        for band in ("weary", "restless", "fixated", "serene", "high", "low"):
            base = policy.weigh(node, out, real_edges, band=band)
            styled = policy.weigh(node, out, real_edges, band=band, styles=real_index)
            assert len(styled) == len(base)
            assert any(w > 0 for _, w in styled), (node, band)
            for (e, sw), (_, bw) in zip(styled, base):
                if bw == 0.0:
                    assert sw == 0.0                       # only exclusion zeroes an edge
                else:
                    assert sw > 0.0
                    assert es.STYLE_FLOOR <= sw / bw <= es.STYLE_CEIL
            checked += 1
    assert checked > 200


def test_styling_actually_changes_the_real_walk(real_edges, real_index, capsys):
    """A layer that never changes a weight is decoration. Measure how much of the real
    graph it touches, and print it."""
    out_by_node = {}
    for e in real_edges:
        out_by_node.setdefault(e.get("from"), []).append(e)
    moved = total = 0
    for node, out in out_by_node.items():
        base = policy.weigh(node, out, real_edges, band="weary")
        styled = policy.weigh(node, out, real_edges, band="weary", styles=real_index)
        for (_, sw), (_, bw) in zip(styled, base):
            total += 1
            if abs(sw - bw) > 1e-12:
                moved += 1
    with capsys.disabled():
        print("\n  POLICY IMPACT  band=weary re-weighted %d/%d real candidate edges (%.1f%%)"
              % (moved, total, 100.0 * moved / total))
    assert moved > 0.5 * total


def test_real_graph_index_is_cheap_enough_to_build_at_load(real_edges):
    import time
    t0 = time.time()
    es.index(real_edges)
    elapsed = time.time() - t0
    assert elapsed < 5.0, "index() is built once at graph load, but 1472 edges took %.2fs" % elapsed
