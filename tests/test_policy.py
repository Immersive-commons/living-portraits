"""test_policy.py -- the unified weighted walk policy. Asserts on the transparent
weight vector (weigh) so the behaviour is pinned, not the RNG.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import policy


def _graph():
    """Hub-and-spoke star with a forward+reverse pair per spoke (mirrors the real graph):
    hub <-> leafA, hub <-> leafB, hub <-> goalP, hub <-> bedP, plus a few idles."""
    T = lambda lbl, a, b: {"id": lbl + "#0", "label": lbl, "kind": "transition", "from": a, "to": b}
    I = lambda lbl, a: {"id": lbl + "#0", "label": lbl, "kind": "idle", "from": a, "to": a}
    return [
        T("h2a", "hub", "leafA"), T("a2h", "leafA", "hub"),
        T("h2b", "hub", "leafB"), T("b2h", "leafB", "hub"),
        T("h2g", "hub", "goalP"), T("g2h", "goalP", "hub"),
        T("h2bed", "hub", "bedP"), T("bed2h", "bedP", "hub"),
        I("idle_hub", "hub"), I("idle_hub2", "hub"),
        I("idle_a", "leafA"), I("idle_g", "goalP"),
    ]


def _w(weights, label):
    return next(w for e, w in weights if e["label"] == label)


def test_anti_reverse_beats_backtrack():
    """At a leaf we arrived at from hub, idling outweighs immediately reversing to hub."""
    g = _graph()
    out = [e for e in g if e["from"] == "leafA"]   # a2h (back to hub) + idle_a
    w = policy.weigh("leafA", out, g, prev_node="hub", last_id="h2a#0",
                     reverse_of=policy._reverse_of_last("h2a"))
    assert _w(w, "idle_a") > _w(w, "a2h"), w   # don't run the clip backwards the instant you arrive


def test_goal_pull_prefers_toward_and_arrival():
    g = _graph()
    out = [e for e in g if e["from"] == "hub"]
    w = policy.weigh("hub", out, g, goal="goalP", route="beeline")
    # h2g lands ON the goal -> must beat every away-from-goal transition
    assert _w(w, "h2g") > _w(w, "h2a")
    assert _w(w, "h2g") > _w(w, "h2bed")


def test_idles_pin_at_goal():
    g = _graph()
    out = [e for e in g if e["from"] == "goalP"]   # g2h + idle_g
    w = policy.weigh("goalP", out, g, goal="goalP")
    assert _w(w, "idle_g") > _w(w, "g2h")   # hold at what you wanted, don't wander off


def test_exclude_mask_zeroes_bedtime_edges():
    g = _graph()
    out = [e for e in g if e["from"] == "hub"]
    w = policy.weigh("hub", out, g, exclude={"h2bed"})
    assert _w(w, "h2bed") == 0.0


def test_deadlock_fix_goal_behind_masked_pose():
    """THE BUG: the goal is only reachable through a bedtime pose, but that pose is
    excluded by day. The shared mask must stop the policy from forcing into it -> no
    idle<->bed reverse-pendulum. (goalP reachable only via bedP here.)"""
    g = [e for e in _graph() if e["label"] not in ("h2g", "g2h")]   # drop the direct hub->goal link
    g += [{"id": "bed2g#0", "label": "bed2g", "kind": "transition", "from": "bedP", "to": "goalP"},
          {"id": "g2bed#0", "label": "g2bed", "kind": "transition", "from": "goalP", "to": "bedP"}]
    out = [e for e in g if e["from"] == "hub"]
    w = policy.weigh("hub", out, g, goal="goalP", exclude={"h2bed", "bed2h"})
    assert _w(w, "h2bed") == 0.0   # never shoved into the bedtime pose chasing a goal
    rng = random.Random(0)
    picks = [policy.choose("hub", out, g, goal="goalP", exclude={"h2bed", "bed2h"},
                           rng=rng)["label"] for _ in range(200)]
    assert "h2bed" not in picks   # the deadlock can't form


def test_novelty_downweights_recent():
    g = _graph()
    out = [e for e in g if e["from"] == "hub"]
    fresh = policy.weigh("hub", out, g)
    seen = policy.weigh("hub", out, g, recent_nodes=["leafA", "leafA", "leafA"])
    assert _w(seen, "h2a") < _w(fresh, "h2a")   # bouncing onto a just-seen pose is discouraged


def test_mood_bends_the_body():
    g = _graph()
    out = [e for e in g if e["from"] == "hub"]
    restless = policy.weigh("hub", out, g, mood="restless")
    weary = policy.weigh("hub", out, g, mood="weary")
    # transition-vs-idle balance flips with mood
    assert _w(restless, "h2a") / _w(restless, "idle_hub") > _w(weary, "h2a") / _w(weary, "idle_hub")


def test_free_text_mood_buckets_by_energy():
    """Real heartbeat moods are free-text ('feral devotion'), not band names. They must still
    land in the right energy band, not silently fall to neutral."""
    t_high, i_high, _ = policy._mood_bias("feral devotion")
    t_low, i_low, _ = policy._mood_bias("a quiet, mournful ache")
    t_def, i_def, _ = policy._mood_bias("inscrutable")          # no energy word -> default
    assert t_high / i_high > t_def / i_def    # high-energy phrase -> more transitions
    assert t_low / i_low < t_def / i_def      # low-energy phrase  -> more idle/dwell
    assert policy._mood_bias("restless")[0] == policy.MOOD_BIAS["restless"][0]   # exact band still wins


def test_context_energy_shifts_transition_idle_balance():
    """The live weather/time nudge leans the walk: "high" (storm/wind) -> more transitions,
    "low" (clear-calm / deep night) -> more dwell. Asserted on the transition-vs-idle ratio."""
    g = _graph()
    out = [e for e in g if e["from"] == "hub"]
    high = policy.weigh("hub", out, g, context_energy="high")
    low = policy.weigh("hub", out, g, context_energy="low")
    # a stormy verdict moves more than a calm one
    assert _w(high, "h2a") / _w(high, "idle_hub") > _w(low, "h2a") / _w(low, "idle_hub")


def test_context_energy_layers_on_mood():
    """context_energy combines WITH the LLM mood (it does not replace it): a "high" weather tilt
    pushes a given mood toward more transitions vs the same mood with no tilt."""
    g = _graph()
    out = [e for e in g if e["from"] == "hub"]
    base = policy.weigh("hub", out, g, mood="weary")
    stormy = policy.weigh("hub", out, g, mood="weary", context_energy="high")
    assert _w(stormy, "h2a") / _w(stormy, "idle_hub") > _w(base, "h2a") / _w(base, "idle_hub")


def test_context_energy_none_is_backward_compatible():
    """SACRED: context_energy=None (and an unknown band) must leave the weights byte-identical
    to the pre-weather behaviour -- the whole feature is opt-in."""
    g = _graph()
    out = [e for e in g if e["from"] == "hub"]
    baseline = policy.weigh("hub", out, g, mood="restless", goal="goalP", route="beeline")
    explicit_none = policy.weigh("hub", out, g, mood="restless", goal="goalP", route="beeline",
                                 context_energy=None)
    unknown = policy.weigh("hub", out, g, mood="restless", goal="goalP", route="beeline",
                           context_energy="bogus")
    assert [w for _, w in explicit_none] == [w for _, w in baseline]
    assert [w for _, w in unknown] == [w for _, w in baseline]   # unrecognised band -> no tilt


def test_context_energy_mapping():
    """The director.context condition -> energy mapping: storm -> high (agitated), clear night ->
    low (settled, here via the SETTLING word 'clear'), deep dark night -> low, unknown -> None."""
    from director import context as ctx
    assert ctx._energy_from_conditions("thunderstorm", False, "night") == "high"  # storm wins, even at night
    assert ctx._energy_from_conditions("rain", True, "afternoon") == "high"
    assert ctx._energy_from_conditions("clear", False, "night") == "low"          # clear calm night
    assert ctx._energy_from_conditions("clear sky", True, "morning") == "low"     # calm clear morning
    assert ctx._energy_from_conditions("overcast", False, "night") == "low"       # deep dark night settles
    assert ctx._energy_from_conditions("overcast", True, "afternoon") is None     # neutral -> defer to mood
    assert ctx._energy_from_conditions("", None, "") is None
    # _signal_from_context distils the raw /api/context JSON into the walker's signal dict
    sig = ctx._signal_from_context({"local": {"daypart": "afternoon"},
                                    "weather": {"temp_f": 58, "condition": "Windy"},
                                    "sun": {"is_daylight": True}})
    assert sig["energy"] == "high" and sig["temp_f"] == 58.0 and sig["condition"] == "windy"
    assert ctx.context_signal.__doc__   # the public verb exists and is documented


def test_never_strands():
    g = _graph()
    out = [e for e in g if e["from"] == "hub"]
    # mask everything -> must still return an edge (uniform fallback), never None/freeze
    w = policy.weigh("hub", out, g, exclude={e["label"] for e in out})
    assert any(x > 0 for _, x in w)
    rng = random.Random(1)
    assert policy.choose("hub", out, g, exclude={e["label"] for e in out}, rng=rng) is not None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
