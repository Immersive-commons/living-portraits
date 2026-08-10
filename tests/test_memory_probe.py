"""test_memory_probe.py -- the FALSIFIABILITY gate for the portraits' memory.

Before this file, no memory change in this project could be shown to have helped or
hurt: the only evidence was reading the panels and feeling good about it. REMem
(arXiv 2602.13530) makes the fix concrete -- probe with questions the memory CAN answer
AND questions it CANNOT, and count "I don't remember" as a correct answer. A system that
answers everything is not remembering, it is confabulating.

What is actually asserted here is the RETRIEVAL layer, not an LLM's wording: for each of
ten questions we check whether the evidence needed to answer it reaches the prompt. That
keeps the probe deterministic, offline, and free -- and it is the layer v0.3.0 changed.
The model's phrasing is a separate (paid, flaky) question this file deliberately does not
pretend to settle.

Fixture: a synthetic 64-day life with the shape of the real one measured on hil --
one dominant want, one rare unrepeated event, a cluster of memories two hops away,
and unvisited reachable poses.
"""
import json
import re
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from director import heartbeat
from runtime import journal_score as js

HOUR = 3600.0
DAY = 86400.0
CHAR = "phineas"

# The graph the probe walks: anchor (hub) -> glower -> brood, plus two poses that exist
# and are reachable but have never been stood in (the frontier).
NODES = {
    "phineas:anchor": {"pose": "anchor", "gen_prompt": "seated in his chair -- the anchor pose"},
    "phineas:glower": {"pose": "glower", "gen_prompt": "glaring sideways -- the glower"},
    "phineas:brood":  {"pose": "brood", "gen_prompt": "chin on fist -- the brood"},
    "phineas:curtain_call": {"pose": "curtain_call", "gen_prompt": "taking a bow -- the curtain call"},
    "phineas:withered_rose": {"pose": "withered_rose", "gen_prompt": "holding a dead rose"},
}
HOPS = {"phineas:anchor": 0, "phineas:glower": 1, "phineas:brood": 2,
        "phineas:curtain_call": 2, "phineas:withered_rose": 3}
GOALS = ["phineas:brood", "phineas:curtain_call", "phineas:glower", "phineas:withered_rose"]


class _Graph:
    nodes = NODES
    edges = []


def _e(ts, pose, goal, reason, mood="bitter"):
    return {"ts": int(ts), "pose": pose, "goal": goal, "mood": mood, "reason": reason}


def _life(now):
    """A 64-day journal: 300 repetitions of one want, one unrepeated bow 61 days ago,
    a handful of memories made two hops from the hub, and a fresh newest line."""
    out = [_e(now - 61 * DAY, "phineas:anchor", "phineas:curtain_call",
              "the night I finally took a bow and no one was watching", "triumphant")]
    for i in range(300):
        out.append(_e(now - (61 * DAY) + (i + 1) * (60 * DAY / 300.0),
                      "phineas:anchor", "phineas:glower", "she is at it again", "bitter"))
    for i in range(6):
        out.append(_e(now - (5 - i) * DAY, "phineas:brood", "phineas:glower",
                      "brooding suits the light in here", "sullen"))
    out.append(_e(now - 900, "phineas:anchor", "phineas:glower",
                  "I shall simply have to out-suffer her", "wounded"))
    out.sort(key=lambda x: x["ts"])
    return out


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point the heartbeat's three data dirs at a tmp life and silence the weather fetch."""
    now = time.time()
    jdir, pdir, cdir = tmp_path / "journal", tmp_path / "pose", tmp_path / "chars"
    for d in (jdir, pdir, cdir):
        d.mkdir()
    (jdir / (CHAR + ".jsonl")).write_text(
        "\n".join(json.dumps(x) for x in _life(now)) + "\n", encoding="utf-8")
    monkeypatch.setattr(heartbeat, "JOURNAL_DIR", jdir)
    monkeypatch.setattr(heartbeat, "POSE_DIR", pdir)
    monkeypatch.setattr(heartbeat, "CHARS_DIR", cdir)
    monkeypatch.setattr(heartbeat.ctx, "context_line", lambda *a, **k: "")
    return {"now": now, "journal": jdir, "pose": pdir, "chars": cdir}


def _prompt(**kw):
    return heartbeat._build_user_prompt(_Graph(), CHAR, "phineas:anchor", 3, GOALS, 14,
                                        hops=HOPS, **kw)


# --------------------------------------------------------------------------- the probe
# Five questions the journal CAN answer -- the evidence must reach the prompt.
def _dominant_want(p):
    """The aggregate line must name the rut AND its size -- the fact no window of
    individual memories can carry (307 glowers out of 308 decisions here)."""
    m = re.search(r"wanted phineas:glower (\d+) times", p)
    return bool(m) and int(m.group(1)) >= 300


ANSWERABLE = [
    ("what do you want most often?",        _dominant_want),
    ("have you ever taken a bow?",          lambda p: "took a bow" in p),
    ("what were you just thinking?",        lambda p: "out-suffer her" in p),
    ("how long have you been hanging here?", lambda p: "61 days" in p or "60 days" in p),
    ("is there anywhere you have never been?", lambda p: "never once been in" in p
                                                        and "withered_rose" in p),
]

# Five questions it CANNOT -- nothing in the prompt may look like an answer, so the only
# honest reply is "I don't remember". A retrieval change that starts smuggling these in
# (e.g. by dumping the persona or the whole graph into context) fails here.
UNANSWERABLE = [
    ("did you ever dance?",                 "danc"),
    ("what did the visitor say to you?",    "visitor"),
    ("who won the argument on Tuesday?",    "argument"),
    ("what happened in the charging pod?",  "pod"),
    ("what is your neighbour's name?",      "beside you"),   # no pose files -> no neighbour
]


def test_the_ten_question_probe(wired):
    prompt = _prompt()
    failed = []
    for q, ok in ANSWERABLE:
        if not ok(prompt):
            failed.append("RECALL MISS: %s" % q)
    for q, needle in UNANSWERABLE:
        if needle in prompt.lower():
            failed.append("CONFABULATION RISK: %r appears in context for %s" % (needle, q))
    assert not failed, "\n".join(failed)


def test_tail_five_would_fail_the_same_probe(wired):
    """The control. The old reader (`lines[-5:]`) is scored by the same five recall
    questions and must fail at least one -- otherwise this probe proves nothing about
    the change that was made."""
    entries = js.load(wired["journal"] / (CHAR + ".jsonl"))
    old = "\n".join("- (%s, feeling %s) %s" % (e.get("pose"), e.get("mood"), e.get("reason"))
                    for e in entries[-5:])
    misses = [q for q, ok in ANSWERABLE if not ok(old)]
    assert len(misses) >= 4, "tail-5 unexpectedly answered: %s" % misses


def test_no_line_in_the_prompt_is_invented(wired):
    """Every retrieved memory must trace to a real journal entry. Retrieval may select and
    order; it may not author."""
    entries = js.load(wired["journal"] / (CHAR + ".jsonl"))
    reasons = {(e.get("reason") or "").strip() for e in entries}
    for line in _prompt().splitlines():
        if line.startswith("- (") and " ago, at " in line:
            body = line.split(") ", 1)[1].strip()
            assert body in reasons, "prompt line not found in journal: %r" % body


def test_the_neighbour_is_seen_when_lit_and_unseen_when_dark(wired):
    """Cross-character context is reconstructed at read time from the other panel's own
    files -- and a dark panel must vanish from the prompt rather than be reported frozen."""
    pdir, jdir, now = wired["pose"], wired["journal"], wired["now"]
    (jdir / "maxx.jsonl").write_text(
        json.dumps(_e(now - 600, "maxx:neon_noir", "maxx:flex", "city's mine tonight",
                      "electric")) + "\n", encoding="utf-8")
    (pdir / "maxx.json").write_text(json.dumps({"node": "maxx:neon_noir", "dwell": 4}),
                                    encoding="utf-8")
    (pdir / (CHAR + ".json")).write_text(json.dumps({"node": "phineas:anchor", "dwell": 3}),
                                         encoding="utf-8")

    lit = _prompt()
    assert "beside you" in lit and "neon_noir" in lit and "electric" in lit
    assert "Maxx" in lit or "MAXX" in lit
    assert "phineas" not in lit.split("beside you")[1].split("\n")[0]   # never sees itself

    import os
    stale = time.time() - heartbeat.NEIGHBOUR_STALE - 60
    os.utime(pdir / "maxx.json", (stale, stale))
    assert "beside you" not in _prompt()


def test_the_band_is_asked_for_in_the_bodys_vocabulary(wired):
    from runtime import policy
    prompt = _prompt()
    assert '"band"' in prompt
    for name in policy.MOOD_BIAS:
        assert name in prompt


def test_band_resolution_and_its_effect_on_the_body():
    from runtime import policy
    assert heartbeat._resolve_band("weary") == "weary"
    assert heartbeat._resolve_band(" Restless ") == "restless"
    assert heartbeat._resolve_band("quietly fixated") == "fixated"
    assert heartbeat._resolve_band("incandescent with spite") is None   # -> keyword fallback
    assert heartbeat._resolve_band(None) is None

    # the whole point: a mood the keyword matcher cannot read now reaches the body.
    guessed = policy._mood_bias("incandescent with spite")
    assert guessed == policy.DEFAULT_MOOD
    assert policy._mood_bias("incandescent with spite", "restless") == policy.MOOD_BIAS["restless"]
    # ...and with no band, behaviour is exactly what it was before this change.
    assert policy._mood_bias("a quiet, mournful ache", None) == policy._mood_bias("a quiet, mournful ache")
