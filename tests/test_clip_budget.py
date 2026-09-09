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


# --------------------------------------------------------------- credits, not clips
# The cap counts CLIPS because that is what governs how fast the wall grows. The grant
# is spent in CREDITS, and stills are billed in credits and were counted nowhere: 4 each
# at the quality generate_still() actually requests, ~230 per grant period on the
# measured pose rate. These pin the half the old ledger could not see.

def test_a_still_is_billed_and_counted(budget):
    from pipeline import hf_gen
    assert budget.credits_used() == 0
    budget._spend_still("phineas")
    assert budget.credits_used() == hf_gen.CREDITS_PER_STILL


def test_a_still_does_not_consume_the_clip_cap(budget):
    """Deliberate: the clip cap is an artwork decision about growth rate, and a still is
    not a clip. If a still ever starts eating the cap, poses stop for the wrong reason."""
    for _ in range(4):
        budget._spend_still("phineas")
    assert budget.clip_budget_left("phineas") == budget.CLIP_CHAR_CAP
    assert budget.clips_used() == 0


def test_the_period_survives_the_daily_rollover(budget, monkeypatch):
    """THE bug. The daily record reset at midnight and took the period accumulator with
    it, so the 3000-credit monthly guarantee lived only in a comment and nothing could
    check it. Day counters must reset; the period must not."""
    monkeypatch.setattr(budget, "_today", lambda: "2026-09-01")
    budget._spend_clip("phineas")
    budget._spend_still("phineas")
    spent = budget.credits_used()
    assert spent > 0

    monkeypatch.setattr(budget, "_today", lambda: "2026-09-02")
    assert budget.clips_used() == 0            # the DAY resets
    assert budget.credits_used() == spent      # the PERIOD does not


def test_the_period_resets_when_a_new_grant_lands(budget, monkeypatch):
    """Credits arrive on the 23rd and do not roll over, so crossing that date is the one
    time the accumulator SHOULD go back to zero."""
    monkeypatch.setattr(budget, "_today", lambda: "2026-09-22")
    budget._spend_clip("phineas")
    assert budget.credits_used() > 0
    monkeypatch.setattr(budget, "_today", lambda: "2026-09-23")
    assert budget.credits_used() == 0


def test_a_period_is_as_long_as_the_month_it_starts_in(budget):
    """Not 30 days. Seven months of twelve are 31, which the old comment's fixed-30
    assumption quietly dropped. Small, but it was drift nothing could measure."""
    assert budget._period_start("2026-09-08") == "2026-08-23"   # before the 23rd -> last month
    assert budget._period_start("2026-09-23") == "2026-09-23"   # on the 23rd -> this month
    assert budget._period_start("2026-09-30") == "2026-09-23"
    assert budget._period_start("2026-01-05") == "2025-12-23"   # across a year boundary
    assert budget._period_start("2026-03-01") == "2026-02-23"   # across a short month


def test_an_old_ledger_file_still_loads(budget):
    """Boxes have a clip_budget.json written before any of this existed. It must read as
    a valid record with a zeroed period, not crash the generation loop on a KeyError."""
    budget._save(budget.CLIP_BUDGET, {"date": budget._today(), "total": 3,
                                      "chars": {"phineas": 3}})
    assert budget.clips_used() == 3
    assert budget.credits_used() == 0
    budget._spend_clip("phineas")
    assert budget.clips_used() == 4
