"""test_llm_extract.py -- JSON extraction from real model replies. No network.

Every shape below was taken from `_heartbeat.log` on hil, where 62 replies over 14,748
ticks failed to parse and each failure cost the character a turn (it stands still and the
previous intent is held). The old extractor spanned the first `{` to the LAST `}`, which
is a single object only when the model emits exactly one.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from director import llm

ONE = '{"goal": "maxx:analog_brew", "mood": "thirsty", "reason": "brew time"}'


def test_plain_object():
    assert llm._extract_json(ONE)["goal"] == "maxx:analog_brew"


def test_repeated_objects_take_the_first():
    """glm-5.1, observed 05:48:03 on hil: the same answer emitted four times in a row.
    The outermost span covered all four and parsed as nothing."""
    assert llm._extract_json(ONE * 4)["goal"] == "maxx:analog_brew"


def test_prose_around_the_object():
    assert llm._extract_json("Here you go:\n" + ONE + "\nHope that helps.")["mood"] == "thirsty"


def test_fenced_json():
    assert llm._extract_json("```json\n" + ONE + "\n```")["goal"] == "maxx:analog_brew"


def test_raw_control_character_inside_a_string():
    """A model writing dialogue puts real newlines in its prose. Strict JSON refuses them;
    the reply is still perfectly readable."""
    got = llm._extract_json('{"goal": "phineas:swoon", "mood": "torn", "reason": "a line\nand another"}')
    assert got["goal"] == "phineas:swoon"


def test_a_brace_inside_a_string_does_not_open_an_object():
    got = llm._extract_json('{"goal": "maxx:flex", "reason": "she said {nothing} at all"}')
    assert got["goal"] == "maxx:flex"


def test_a_genuinely_cut_off_reply_still_fails():
    """The honest negative: an unfinished object must NOT be salvaged into a half-truth.
    Recovering it would mean inventing the part the model never said."""
    assert llm._extract_json('```json\n{\n"label": "synth_cafeteria",\n"still_prompt": "Standing in a fl') is None
    assert llm._extract_json("no json here at all") is None


def test_reply_carries_why_the_model_stopped():
    r = llm._Reply("half a sen", "max_tokens")
    assert r == "half a sen"                 # every existing caller is unaffected
    assert r.stop_reason == "max_tokens"
    assert llm._Reply("done").stop_reason is None
