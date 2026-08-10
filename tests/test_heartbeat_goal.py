"""test_heartbeat_goal.py -- the brain names a pose; the walker needs a node id.

Regression cover for the FROZEN-PORTRAITS bug (found on hil 2026-07-31): the goal
menu lists "phineas:withered_rose" but the model answers "withered_rose", the old
exact-membership test scored that a miss, and the character stayed put. 1170 of
1179 rejected decisions in the live log were this single mismatch.

Pure function, no graph / no network / no LLM.
"""
import pytest

from director.heartbeat import _resolve_goal

ALLOWED = {
    "phineas:shattered_mirror",
    "phineas:withered_rose",
    "phineas:withered_poise",
    "phineas:soliloquy",
}


def test_exact_node_id_passes_through():
    assert _resolve_goal("phineas:withered_rose", "phineas", ALLOWED) == (
        "phineas:withered_rose", "exact")


def test_bare_pose_label_is_prefixed():
    """THE bug: the model drops the 'phineas:' the menu used."""
    assert _resolve_goal("withered_rose", "phineas", ALLOWED) == (
        "phineas:withered_rose", "prefixed")


def test_quoted_and_padded_label_still_resolves():
    assert _resolve_goal('  "withered_rose" ', "phineas", ALLOWED)[0] == "phineas:withered_rose"


def test_case_and_space_insensitive_match():
    assert _resolve_goal("Withered_Poise", "phineas", ALLOWED) == (
        "phineas:withered_poise", "normalized")


def test_unknown_pose_is_novel_not_a_walk_target():
    """A name we genuinely do not own is proposal material -- the caller must NOT walk there."""
    assert _resolve_goal("phineas:rising_from_corpse", "phineas", ALLOWED) == (None, "novel")
    assert _resolve_goal("tarot_spread", "phineas", ALLOWED) == (None, "novel")


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_empty_goal_is_rejected(bad):
    assert _resolve_goal(bad, "phineas", ALLOWED) == (None, "empty")


def test_resolution_is_deterministic_across_calls():
    """Same input, same answer -- the walker must not oscillate between equal candidates."""
    first = _resolve_goal("withered_rose", "phineas", ALLOWED)
    for _ in range(20):
        assert _resolve_goal("withered_rose", "phineas", ALLOWED) == first
