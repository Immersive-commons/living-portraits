"""test_mind.py -- the LLM-intent decide() layer (pure; no network, no LLM). No network."""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import mind


def _t(label, fr, to):
    return {"id": "phineas/%s/v0" % label, "kind": "transition", "label": label,
            "from": "phineas:" + fr, "to": "phineas:" + to}


def _i(label, at):
    return {"id": "phineas/%s/v0" % label, "kind": "idle", "label": label,
            "from": "phineas:" + at, "to": "phineas:" + at}


EDGES = [
    _i("breathe", "anchor"), _i("settle", "anchor"),
    _t("a2g", "anchor", "glower"), _t("g2a", "glower", "anchor"), _i("glare", "glower"),
    _t("a2s", "anchor", "swoon"), _t("s2a", "swoon", "anchor"), _i("heave", "swoon"),
]
RNG = random.Random(0)


def _intent(goal, age=0.0):
    return {"characters": {"phineas": {"goal": goal, "set_at": time.time() - age}}}


def _out(node):
    return [e for e in EDGES if e.get("from") == node]


def test_no_intent_yields_none():
    d = mind.decide("phineas", "phineas:anchor", _out("phineas:anchor"), EDGES, None, {}, RNG)
    assert d["action"] == "none"


def test_en_route_forces_transition_toward_goal():
    d = mind.decide("phineas", "phineas:anchor", _out("phineas:anchor"), EDGES, None,
                    _intent("phineas:glower"), RNG)
    assert d["action"] == "force"
    assert d["edge"]["label"] == "a2g"          # first hop toward glower


def test_multi_hop_forces_first_hop():
    # at glower, want swoon -> route via anchor -> first forced edge is g2a
    d = mind.decide("phineas", "phineas:glower", _out("phineas:glower"), EDGES, None,
                    _intent("phineas:swoon"), RNG)
    assert d["action"] == "force" and d["edge"]["label"] == "g2a"


def test_at_goal_pins_on_an_idle():
    d = mind.decide("phineas", "phineas:glower", _out("phineas:glower"), EDGES, None,
                    _intent("phineas:glower"), RNG)
    assert d["action"] == "force" and d["edge"]["kind"] == "idle"
    assert d["edge"]["from"] == "phineas:glower"


def test_at_goal_idle_avoids_immediate_repeat():
    out = _out("phineas:anchor")            # two idles: breathe, settle
    last = out[0]["id"]
    seen = set()
    for _ in range(20):
        d = mind.decide("phineas", "phineas:anchor", out, EDGES, last,
                        _intent("phineas:anchor"), RNG)
        seen.add(d["edge"]["id"])
    assert last not in seen                  # never replays the exact last clip


def test_stale_goal_is_ignored():
    d = mind.decide("phineas", "phineas:anchor", _out("phineas:anchor"), EDGES, None,
                    _intent("phineas:glower", age=99999), RNG, max_age=1800.0)
    assert d["action"] == "none"             # dead heartbeat -> release to normal walk


def test_unreachable_goal_degrades_to_none():
    d = mind.decide("phineas", "phineas:anchor", _out("phineas:anchor"), EDGES, None,
                    _intent("phineas:ghost"), RNG)
    assert d["action"] == "none"
