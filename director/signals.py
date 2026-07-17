"""
signals.py -- the "important data" the stage-manager reacts to.

v1 covers: clock + part-of-day, a drifting house mood (persisted), and an
external "material" feed (stubbed; swap _feed() for THE SIGNAL / Google
Calendar / weather later -- the director consumes whatever this returns).
"""
import datetime
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MOODS = ["wistful", "theatrical", "conspiratorial", "petulant", "grand", "bored", "mischievous"]


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def part_of_day(hour):
    if hour < 6:
        return "the small hours"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def _feed():
    # Real material: THE SIGNAL headlines + SF weather + today's calendar,
    # merged/deduped/cached by feeds.get_material() (each source guarded with a
    # short timeout). If feeds itself is unavailable, fall back to the original
    # hardcoded stub so the director never starves.
    try:
        try:
            import feeds  # director/ on sys.path (run as a script, like stage_manager)
        except ImportError:
            from director import feeds  # run as a package
        return feeds.get_material(max_items=5)
    except Exception:
        feed = _load(DATA / "feed.json", {"items": [
            "the humans released yet another frontier model today",
            "a startup three floors down pivoted again",
            "rain is forecast over the city tonight",
        ]})
        return feed.get("items", [])


def gather():
    DATA.mkdir(exist_ok=True)
    now = datetime.datetime.now()
    state = _load(DATA / "mood.json", {"mood": "theatrical"})
    if random.random() < 0.5:  # internal drift
        state["mood"] = random.choice(MOODS)
    (DATA / "mood.json").write_text(json.dumps(state), encoding="utf-8")
    return {
        "now": now.strftime("%A %H:%M"),
        "part_of_day": part_of_day(now.hour),
        "mood": state["mood"],
        "material": _feed(),
    }
