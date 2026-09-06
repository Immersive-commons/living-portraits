"""The clip budget is the only thing standing between an autonomous loop and a spent
month, so it gets tested at its edges: the per-character cap, the whole-system cap, and
the day rollover."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import autogen


@pytest.fixture()
def budget(tmp_path, monkeypatch):
    monkeypatch.setattr(autogen, "CLIP_BUDGET", tmp_path / "clip_budget.json")
    return autogen


def test_starts_at_the_per_character_cap(budget):
    assert budget.clip_budget_left("phineas") == budget.CLIP_CHAR_CAP
    assert budget.clips_used() == 0


def test_per_character_cap_is_independent(budget):
    for _ in range(budget.CLIP_CHAR_CAP):
        budget._spend_clip("phineas")
    assert budget.clip_budget_left("phineas") == 0
    # maxx is untouched by phineas exhausting itself
    assert budget.clip_budget_left("maxx") == budget.CLIP_CHAR_CAP


def test_system_cap_binds_before_the_character_cap(budget):
    """Two characters at 6 each is 12, so the 13th clip is the system's to give -- but the
    14th must be refused even though that character still has personal headroom on paper."""
    for _ in range(6):
        budget._spend_clip("phineas")
    for _ in range(6):
        budget._spend_clip("maxx")
    assert budget.clips_used() == 12
    # a third character would have 1 left, not its full 6: the system cap is what remains
    assert budget.clip_budget_left("seraphina") == budget.CLIP_DAILY_CAP - 12 == 1
    budget._spend_clip("seraphina")
    assert budget.clip_budget_left("seraphina") == 0
    assert budget.clips_used() == budget.CLIP_DAILY_CAP


def test_never_returns_negative(budget):
    for _ in range(50):
        budget._spend_clip("phineas")
    assert budget.clip_budget_left("phineas") == 0
    assert budget.clip_budget_left("maxx") == 0     # system cap blown too


def test_rolls_over_on_a_new_day(budget, monkeypatch):
    for _ in range(budget.CLIP_CHAR_CAP):
        budget._spend_clip("phineas")
    assert budget.clip_budget_left("phineas") == 0
    monkeypatch.setattr(budget, "_today", lambda: "2099-01-01")
    assert budget.clip_budget_left("phineas") == budget.CLIP_CHAR_CAP
    assert budget.clips_used() == 0


def test_caps_match_the_plan_arithmetic(budget):
    """3000 credits/month at 7.5 per sound-off kling clip is ~400/month ~= 13/day. If
    someone raises a cap without redoing this sum, this test is the tripwire."""
    from pipeline import hf_gen
    monthly_credits = 3000
    per_month = monthly_credits / hf_gen.CREDITS_PER_CLIP
    assert per_month / 30 == pytest.approx(budget.CLIP_DAILY_CAP, abs=1.5)
    # two characters must fit inside the system cap without starving each other
    assert budget.CLIP_CHAR_CAP * 2 <= budget.CLIP_DAILY_CAP
