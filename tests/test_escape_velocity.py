"""test_escape_velocity.py -- a character must be able to leave a cul-de-sac. No network.

The shape being pinned: on 2026-08-10 Phineas spent an hour at `commanding_aether`, a pose
with ONE exit and three idles, under band `fixated` (idles x1.5, transitions x0.4). The exit
held 13.4% against the idles' 86.6%, so the wall showed the same three clips for an hour.
56 of his 120 poses have one exit or none.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import policy

CH = "phineas"


def _idle(label):
    return {"id": "%s/%s/v0" % (CH, label), "kind": "idle", "label": label,
            "from": CH + ":spur", "to": CH + ":spur"}


def _exit():
    return {"id": "%s/spur_out/v0" % CH, "kind": "transition", "label": "spur_out",
            "from": CH + ":spur", "to": CH + ":hub"}


# the real shape: one exit, three idles -- what autogen builds by default
OUT = [_idle("spur_0"), _idle("spur_1"), _idle("spur_2"), _exit()]
ALL = [*OUT, {"id": "%s/hub_back/v0" % CH, "kind": "transition", "label": "hub_back", "from": CH + ":hub", "to": CH + ":spur"}]


def _exit_share(dwell, band="fixated"):
    w = policy.weigh(CH + ":spur", OUT, ALL, band=band, dwell=dwell)
    total = sum(x for _, x in w)
    return sum(x for e, x in w if e.get("kind") == "transition") / total


def _picks_to_leave(dwell, band="fixated"):
    """Expected number of picks before the character takes an exit -- the metric that maps
    onto what a viewer sees. Each pick is one full idle clip, roughly 6-15s on the wall."""
    s = _exit_share(dwell, band)
    return (1.0 / s) if s else float("inf")


def test_the_measured_trap_is_reproduced_at_zero_dwell():
    """Guard the premise. In this bare fixture the exit holds 8% (the incident measured
    13.4% because live goal-pull was also boosting it) -- either way, ~12 idle clips of
    expected wait, which is minutes of the same three animations."""
    assert 0.05 < _exit_share(0) < 0.15
    assert _picks_to_leave(0) > 10


def test_ordinary_lingering_is_untouched():
    """A character is allowed to dwell. The pull must not start on arrival, or every pose
    turns into a bus stop."""
    assert _exit_share(policy.ESCAPE_AFTER) == pytest.approx(_exit_share(0))
    assert _exit_share(3) == pytest.approx(_exit_share(0))


def test_the_exit_gets_more_attractive_the_longer_we_are_stuck():
    shares = [_exit_share(d) for d in (0, 12, 20, 40, 60)]
    assert shares == sorted(shares), "escape pull must be monotonic in dwell"
    assert shares[1] > shares[0]
    # at the dwell that actually happened on the wall, he must be leaving within a few
    # clips -- under a minute, not under an hour.
    assert _picks_to_leave(51) < 4, "still %.1f picks to leave at the real incident dwell" % _picks_to_leave(51)
    assert _picks_to_leave(20) < _picks_to_leave(0) / 2


def test_it_is_a_nudge_not_a_forced_march():
    """Bounded: a stuck character leaves, it does not teleport. The idles keep real weight
    so the pose still reads as inhabited on the way out."""
    assert _exit_share(10_000) < 0.95
    w = policy.weigh(CH + ":spur", OUT, ALL, band="fixated", dwell=10_000)
    assert all(x > 0 for _, x in w), "no edge may be zeroed by the escape term"


def test_it_helps_most_where_the_mood_hurts_most():
    """`fixated` and `weary` are the bands that create the trap; `restless` barely needs it."""
    stuck_fixated = _exit_share(0, "fixated")
    stuck_restless = _exit_share(0, "restless")
    assert stuck_fixated < stuck_restless
    assert _exit_share(30, "fixated") > stuck_fixated * 2


def test_default_arguments_leave_every_weight_byte_identical():
    """No caller that omits `dwell` may see any change at all."""
    before = policy.weigh(CH + ":spur", OUT, ALL, band="fixated")
    after = policy.weigh(CH + ":spur", OUT, ALL, band="fixated", dwell=0)
    assert [x for _, x in before] == [x for _, x in after]


def test_a_sleeping_character_is_never_pulled_out_of_bed():
    """The walker passes dwell=0 at the sleep pose. Verified here as the CONTRACT: with
    dwell=0 an eight-hour sleep dwell produces exactly the resting weights."""
    resting = policy.weigh(CH + ":spur", OUT, ALL, band="weary", dwell=0)
    all_night = policy.weigh(CH + ":spur", OUT, ALL, band="weary", dwell=0)
    assert [x for _, x in resting] == [x for _, x in all_night]


# --------------------------------------------------------------------------- pendant poses
# The trap that survived the first fix: a PENDANT pose has one neighbour, reached and left by
# the same edge, so its only exit is always a backtrack. Escape velocity lifted the exit to
# 45% and anti-reverse cut it to 3.2% -- 31 expected clips of the same three idles. Measured
# live on phineas:commanding_aether (in from sleep, out to sleep) on 2026-08-10.
PENDANT_OUT = [_idle("spur_0"), _idle("spur_1"), _idle("spur_2"),
               {"id": "%s/back/v0" % CH, "kind": "transition", "label": "back",
                "from": CH + ":spur", "to": CH + ":hub"}]


def _pendant_share(dwell):
    w = policy.weigh(CH + ":spur", PENDANT_OUT, PENDANT_OUT, band="weary", dwell=dwell,
                     prev_node=CH + ":hub")          # we ARRIVED from the only neighbour
    total = sum(x for _, x in w)
    return sum(x for e, x in w if e.get("kind") == "transition") / total


def test_a_fresh_arrival_still_does_not_bounce_straight_back():
    """The pendulum guard is the point of anti-reverse and must survive at short dwell."""
    assert _pendant_share(0) < 0.05
    assert policy._reverse_penalty(0) == policy.REVERSE_PENALTY
    assert policy._reverse_penalty(policy.ESCAPE_AFTER) == policy.REVERSE_PENALTY


def test_a_pendant_pose_is_not_a_cell():
    """After a long dwell the only door must open -- otherwise 'do not backtrack' silently
    means 'never leave'."""
    assert _pendant_share(0) < 0.05                                  # trapped on arrival
    assert _pendant_share(62) > 0.30, "still %.1f%% at the measured incident dwell" % (100 * _pendant_share(62))
    assert _reverse_is_monotonic()


def _reverse_is_monotonic():
    vals = [policy._reverse_penalty(d) for d in (0, 8, 12, 20, 28, 40, 100)]
    return vals == sorted(vals) and vals[-1] == 1.0


def test_forgiveness_is_complete_but_not_early():
    assert policy._reverse_penalty(policy.ESCAPE_AFTER + policy.REVERSE_FORGIVE) == 1.0
    assert policy._reverse_penalty(policy.ESCAPE_AFTER + 1) < 0.2   # barely moved yet
