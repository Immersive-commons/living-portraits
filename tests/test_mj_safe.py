"""test_mj_safe.py -- the MJ moderation guard. Pure; no network. Encodes the real
banned-word + sleepwear lessons from the Living Portraits build."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from director import mj_safe


def test_clothing_negative_landmine_is_dropped():
    # the exact incident: `--no modern clothing` -> "clothing" flagged -> account block
    kept, dropped = mj_safe.sanitize_negatives(["modern clothing", "hat", "text", "watermark"])
    assert "text" in kept and "watermark" in kept
    assert any("clothing" in d for d in dropped)
    assert all("clothing" not in k for k in kept)


def test_negatives_are_allow_listed():
    kept, dropped = mj_safe.sanitize_negatives(
        "picture frame, two people, deformed, naked, nightgown, oxblood robe")
    assert "picture frame" in kept and "two people" in kept and "deformed" in kept
    # body/garment terms never survive
    for bad in ("naked", "nightgown", "oxblood robe"):
        assert bad in dropped


def test_safe_negative_string_roundtrip():
    s = mj_safe.safe_negative_string(["text", "watermark", "nude", "garment"])
    assert "text" in s and "watermark" in s
    assert "nude" not in s and "garment" not in s


def test_hard_block_rejects_positive():
    assert not mj_safe.is_safe("a nude figure reclining")
    hard, _ = mj_safe.scan("an erotic seductive pose")
    assert "erotic" in hard


def test_clean_phineas_prompt_passes():
    v = mj_safe.check(
        positive="oil painting, Rembrandt manner, a 62 year old tragedian in an oxblood "
                 "velvet doublet, reading a letter by candlelight, deep chiaroscuro",
        motion="he unfolds a letter and reads it intently, brow furrowing, then lowers it",
        negatives=["picture frame", "text", "watermark", "two people", "deformed"])
    assert v["ok"] is True
    assert v["hard"] == []
    assert set(v["safe_negatives"]) == {"picture frame", "text", "watermark", "two people", "deformed"}


def test_sleepwear_motion_is_soft_not_hard():
    v = mj_safe.check(positive="a figure in a high-necked quilted dressing robe",
                      motion="he yawns and stretches before settling to sleep",
                      negatives=["text"])
    assert v["ok"] is True              # allowed...
    assert "yawn" in v["soft"] and "stretch" in v["soft"]   # ...but flagged for rewording


def test_hard_block_in_motion_fails_whole_check():
    v = mj_safe.check(positive="a portrait", motion="she undresses slowly", negatives=[])
    assert v["ok"] is False
    assert "undress" in v["hard"]
