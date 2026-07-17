"""mind.py -- LLM-INTENT layer over the video-graph walk (daytime goal-seeking).

Sibling of circadian.py, SAME decide() shape. Where circadian forces the walker
toward a fixed nightly goal (sleep), the mind forces it toward a goal POSE chosen
by the heartbeat brain (director/heartbeat.py, GLM) and published to
data/mind/intent.json. The walker consults the mind ONLY by day -- circadian wins
at night (sleep is non-negotiable), and if the mind has no opinion the caller's
normal random walk runs unchanged. Precedence in the walker:

    circadian (clock)  >  mind (LLM intent)  >  random walk (fallback)

How a "want" becomes motion: the heartbeat writes {goal: "phineas:glower"}; on each
pick the mind runs pathfind(current -> goal) and FORCES the next transition; the
character lands one pose closer and asks again -- a visible step-by-step walk. At
the goal it PINS on idle loops until the heartbeat sets a new goal.

Design contract (avoids cross-writer races):
  * intent.json  -- the HEARTBEAT is the sole writer; the walker only reads.
  * pose/<char>.json -- each panel's walker is the sole writer; the heartbeat reads.

Resilience: a goal older than `max_age` (a dead/blocked heartbeat) is IGNORED, so the
portraits degrade to the lively random walk instead of freezing at a stale goal.

Pure + import-safe (stdlib only). The network/LLM lives in director/, never here:
this module is imported by the 10fps render loop and must stay fast + side-effect free.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from runtime import pathfind

ROOT = Path(__file__).resolve().parent.parent
MIND_DIR = ROOT / "data" / "mind"
INTENT_PATH = MIND_DIR / "intent.json"

DEFAULT_MAX_AGE = 1800.0   # 30 min: ignore a goal older than this (stale brain -> degrade)


def load_intent(path=INTENT_PATH):
    """Read intent.json -> dict (atomic single-file; tolerant of a missing/half-written
    file -> {}). Cheap enough to call per pick; the caller may mtime-cache it."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def goal_for(intent, character, now=None, max_age=DEFAULT_MAX_AGE):
    """The fresh goal node for `character`, or None. A goal with no/old `set_at` is
    treated as stale and ignored (so a stopped heartbeat releases the character)."""
    c = (intent or {}).get("characters", {}).get(character) or {}
    goal = c.get("goal")
    if not goal:
        return None
    set_at = c.get("set_at")
    if max_age and set_at is not None:
        now = time.time() if now is None else now
        try:
            if now - float(set_at) > max_age:
                return None
        except (TypeError, ValueError):
            pass
    return goal


def decide(character, node, out_edges, all_edges, last_id, intent, rng,
           now=None, max_age=DEFAULT_MAX_AGE):
    """Return one of:
        {"action": "force", "edge": e}   play exactly this edge (a step toward, or an
                                         idle hold at, the goal)
        {"action": "none"}               no/stale/unreachable goal -> caller walks normally

    The mind never excludes edges or strands the walk; "none" hands full control back."""
    goal = goal_for(intent, character, now=now, max_age=max_age)
    if not goal:
        return {"action": "none"}

    if goal == node:
        # arrived: PIN on a fresh idle here until the heartbeat picks a new goal, so the
        # character dwells at what it wanted instead of wandering off on its own.
        idles = [e for e in out_edges if e.get("kind") == "idle"]
        fresh = [e for e in idles if e.get("id") != last_id] or idles
        if fresh:
            return {"action": "force", "edge": rng.choice(fresh)}
        return {"action": "none"}       # no idle to hold on -> let normal walk run

    step = pathfind.next_step(all_edges, node, goal)
    if step is not None:
        return {"action": "force", "edge": step}   # walk one transition toward the goal
    return {"action": "none"}           # goal unreachable from here -> degrade to normal walk
