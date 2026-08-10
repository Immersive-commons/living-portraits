"""test_journal_score.py -- scored retrieval over the journal. Pure, offline, no LLM.

Pins the three terms (recency / importance / relevance) and the budget, so a future
tweak to the formula has to be deliberate rather than accidental.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import journal_score as js

HOUR = 3600.0
DAY = 86400.0


def _e(ts, pose, goal, reason="", mood="flat"):
    return {"ts": ts, "pose": pose, "goal": goal, "mood": mood, "reason": reason}


def test_load_skips_garbage_and_bounds_the_scan(tmp_path):
    p = tmp_path / "j.jsonl"
    lines = [json.dumps(_e(i, "phineas:anchor", "phineas:glower")) for i in range(10)]
    lines.insert(3, "{ half a written line")          # a torn append must not raise
    lines.insert(6, "")                               # blank line
    p.write_text("\n".join(lines), encoding="utf-8")

    assert len(js.load(p)) == 10                      # the two junk lines are dropped
    assert len(js.load(p, max_scan=4)) == 4           # bounded scan keeps the NEWEST
    assert js.load(p, max_scan=4)[-1]["ts"] == 9
    assert js.load(tmp_path / "nope.jsonl") == []     # missing file -> [], never raises


def test_recency_decays_per_hour():
    now = time.time()
    assert js.recency(_e(now, "a", "b"), now) == 1.0
    one_h = js.recency(_e(now - HOUR, "a", "b"), now)
    assert abs(one_h - js.RECENCY_DECAY) < 1e-9
    assert js.recency(_e(now - 30 * DAY, "a", "b"), now) < one_h


def test_importance_is_surprisal_of_the_want():
    entries = [_e(0, "p", "phineas:jealous_glare") for _ in range(99)]
    entries.append(_e(0, "p", "phineas:curtain_call"))         # done exactly once
    counts = js.goal_counts(entries)
    total = sum(counts.values())
    common = js.importance(entries[0], counts, total)
    rare = js.importance(entries[-1], counts, total)
    assert rare == 1.0                                          # a once-in-a-life want
    assert 0.0 < common < 0.2                                   # the rut is not surprising
    assert rare > common


def test_relevance_is_transition_hops_from_here():
    dist = {"phineas:anchor": 0, "phineas:glower": 1, "phineas:swoon": 4}
    here = js.relevance(_e(0, "phineas:anchor", "x"), dist)
    near = js.relevance(_e(0, "phineas:glower", "x"), dist)
    far = js.relevance(_e(0, "phineas:swoon", "x"), dist)
    off = js.relevance(_e(0, "phineas:island", "x"), dist)      # unreachable
    assert here == 1.0 and here > near > far > 0.0
    assert off == 0.0
    assert js.relevance(_e(0, "phineas:anchor", "x"), {}) == 0.0   # no map -> term drops out


def test_selection_beats_tail_five_on_the_thing_that_matters():
    """The rut-breaking memory is the OLDEST entry. `lines[-5:]` cannot see it; scored
    retrieval must, because surprisal outweighs 60 days of decay."""
    now = time.time()
    entries = [_e(now - 60 * DAY, "phineas:anchor", "phineas:curtain_call",
                  "the night I finally took a bow")]
    entries += [_e(now - (500 - i) * HOUR, "phineas:anchor", "phineas:jealous_glare", "again")
                for i in range(500)]

    tail5 = entries[-5:]
    assert not any("bow" in (e["reason"] or "") for e in tail5)

    picked = js.select(entries, now=now, distmap={"phineas:anchor": 0})
    assert any("bow" in (e["reason"] or "") for e in picked)
    assert picked == sorted(picked, key=lambda e: e["ts"])       # chronological for the prompt


def test_pins_reach_past_the_decay_horizon():
    """Measured on hil: raw recency+importance+relevance retrieved nothing older than 9
    days from a 66-day life. The reserved pins are what make long-term recall real."""
    now = time.time()
    old_standout = _e(now - 60 * DAY, "phineas:anchor", "phineas:curtain_call",
                      "the night I finally took a bow")
    entries = [old_standout]
    # a busy recent month: many DIFFERENT wants, so recent entries also score high on
    # importance -- exactly the real-journal shape that buried the distant memory.
    for i in range(800):
        entries.append(_e(now - (800 - i) * HOUR, "phineas:anchor",
                          "phineas:want%d" % (i % 100), "chatter %d" % i))

    unpinned = js.select(entries, now=now, distmap={"phineas:anchor": 0}, pin=0)
    assert old_standout not in unpinned                    # decay buries it
    assert min(js._ts(e) for e in unpinned) > now - 30 * DAY

    picked = js.select(entries, now=now, distmap={"phineas:anchor": 0})
    assert old_standout in picked                          # ...the pin reaches it
    assert picked == sorted(picked, key=lambda e: e["ts"])


def test_a_pin_is_never_a_recent_line():
    now = time.time()
    entries = [_e(now - i * 60, "p", "g%d" % i, "just now %d" % i) for i in range(40)]
    assert js.pinned(entries, now=now) == []                # nothing older than PIN_MIN_AGE


def test_token_budget_bounds_the_prompt_as_the_journal_grows():
    now = time.time()
    small = [_e(now - i * HOUR, "p", "g%d" % i, "x" * 60) for i in range(50)]
    huge = [_e(now - i * HOUR, "p", "g%d" % i, "x" * 60) for i in range(20000)]
    a = js.select(small, now=now, token_budget=300)
    b = js.select(huge, now=now, token_budget=300)
    assert len(b) <= len(a) + 2                                  # does not grow with history
    assert sum(len(x) for x in js.render(b, now)) <= 300 * js.CHARS_PER_TOKEN * 1.6


def test_aggregate_line_states_the_rut_in_one_sentence():
    now = time.time()
    entries = [_e(now - 60 * DAY + i, "p", "phineas:jealous_glare") for i in range(200)]
    entries += [_e(now - i * HOUR, "p", "phineas:swoon") for i in range(50)]
    line = js.aggregate_line(entries, now=now)
    assert "jealous_glare" in line and "200" in line and "60 days" in line
    assert js.aggregate_line([], now=now) == ""                  # empty life -> no claim


def test_visited_poses_is_where_it_has_stood_not_what_it_wanted():
    entries = [_e(0, "phineas:anchor", "phineas:swoon"), _e(1, "phineas:glower", "phineas:swoon")]
    assert js.visited_poses(entries) == {"phineas:anchor", "phineas:glower"}


def test_retrieval_over_a_real_sized_journal_is_fast():
    """25k entries is the live size on hil (2026-08-10). The heartbeat's budget is ~4 min."""
    now = time.time()
    entries = [_e(now - i * 400, "phineas:p%d" % (i % 120), "phineas:g%d" % (i % 120), "line %d" % i)
               for i in range(25000)]
    dist = {"phineas:p%d" % i: i % 7 for i in range(120)}
    t0 = time.time()
    picked = js.select(entries, now=now, distmap=dist)
    js.render(picked, now)
    js.aggregate_line(entries, now=now)
    elapsed = time.time() - t0
    assert elapsed < 5.0, "retrieval took %.2fs over 25k entries" % elapsed
