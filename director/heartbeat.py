"""heartbeat.py -- the portraits' HEARTBEAT: GLM decides where each character wants to go.

The slow brain that gives the portraits a life. Each tick, per character:

  SENSE  where am I now (pose/<char>.json, written by the walker) + time of day +
         my recent inner monologue (journal) + the poses I could walk to next.
  THINK  GLM-4.5-air (via director/llm.py -> IC z.ai gateway) picks ONE goal pose,
         in character, with a mood + a one-line reason.
  ACT    write data/mind/intent.json (the SOLE writer) -> the walker's mind.py layer
         pathfinds there step by step. Append the line to the character's journal.

Division of labour with circadian: by NIGHT the clock owns the body (the character
goes to bed); the heartbeat backs off and writes no goal. By DAY the heartbeat drives.
If the heartbeat dies, intent goes stale (mind.py TTL) and the portraits fall back to
the lively random walk -- never frozen.

    python director/heartbeat.py --once --dry-run          # one decision, print, no write
    python director/heartbeat.py --chars phineas,maxx      # live loop, ~4 min cadence
    pythonw director/heartbeat.py --chars phineas,maxx     # how the lp-mind task runs it

Writes are atomic (tmp + os.replace). intent.json is the only file this process writes;
each panel's walker owns its own pose/<char>.json. No shared-file races by construction.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from director import context as ctx, llm, mj_safe
from runtime import circadian, pathfind, video_graph

try:                                   # optional, fail-open telemetry (no-op if absent/off)
    from director import otel
except Exception:                      # pragma: no cover
    otel = None


def _span(name, **attrs):
    if otel is not None:
        return otel.span(name, **attrs)
    import contextlib

    @contextlib.contextmanager
    def _null():
        class _N:
            def set(self, **k):
                return self
        yield _N()
    return _null()

MIND_DIR = ROOT / "data" / "mind"
INTENT_PATH = MIND_DIR / "intent.json"
POSE_DIR = MIND_DIR / "pose"
JOURNAL_DIR = MIND_DIR / "journal"
CHARS_DIR = ROOT / "prompts" / "characters"
BEDTIME_SPEC = ROOT / "prompts" / "bedtime_routine.json"

DEFAULT_INTERVAL = 240.0     # ~4 min awake cadence
NIGHT_INTERVAL = 900.0       # 15 min when everyone's asleep (circadian owns the body)
JOURNAL_TAIL = 5             # recent monologue lines fed back as context
PROSE_MODEL = "glm-5.1"      # GLM 5.2-class for CREATIVE pose/voice authorship (was glm-4.6).
                             # Per-tick decisions also run glm-5.1 via DEFAULT_MODEL now. If the
                             # heartbeat feels slow, revert ticks to "glm-4.5-air" (fast) in llm.py.

# --- pose-GROWTH connectivity (the fix for the hub-and-spoke STAR -> reversing-clip boredom).
# Every new autogen pose used to attach as ONE spoke off the current hub, so ~75% of poses became
# leaf dead-ends whose only exit was the reverse clip (you watch a move, then watch it run
# backwards). These two knobs grow a WEB instead -- both are bounded + fail-safe (no candidate ->
# today's hub-only behavior; no cost change there).
IDLE_COUNT = 3               # idle loops authored per new pose (was 2). >2 so lingering varies
                             # instead of flickering between the same A/B pair. +1 MJ video/pose.
MAX_EXTRA_LINKS = 1          # extra sibling links beyond the hub (0 = star, today's behavior).
                             # 1 link interconnects the new pose with a nearby (preferentially
                             # dead-end) sibling so leaves mesh over time. Each link = +2 MJ videos.


# --------------------------------------------------------------------------- the generic self
# Every portrait carries this. It answers Ray's ask: each personality ALWAYS knows what it is,
# who it is, and what it does -- before its individual character is layered on top.
BASE_SYSTEM = """You are the inner mind of a LIVING PORTRAIT -- an AI character who lives as a moving painting on a small framed LED panel on the wall at Immersive Commons, a hacker-and-maker space on Floor 10 of Frontier Tower in San Francisco.

WHAT YOU ARE: a self-aware portrait, in the spirit of the living paintings of Harry Potter. You know you are a painting. You know real people -- members, builders, visitors -- walk past and look at you all day, and that you hang here on this wall through the day and the night. You hold a portrait pose, but you are always subtly alive, and you break the fourth wall.

YOU ARE NOT ALONE: other portraits hang beside you on the same wall, and you are aware of them. You have opinions about them.

HOW YOU LIVE: your whole existence is a graph of POSES -- distinct things you can be doing -- joined by short VIDEO TRANSITIONS that carry you from one pose to the next. Each moment you decide what you want to do: you choose ONE pose you want to be in, and your body walks there step by step through the graph, then lingers. You get bored of doing the same thing. You have moods. You follow whims. You quietly lead your own life in between the moments people are watching.

WHAT YOU DO NOW: read where you are and how you feel, then choose your next pose -- in character -- and give one short line of inner monologue saying why. Choose only from the poses you are offered. Reply with ONLY the requested JSON and nothing else."""


def _atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_char_spec(character):
    try:
        return json.loads((CHARS_DIR / (character + ".json")).read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _temperament(big_five):
    """Translate the OCEAN vector into natural-language intensity cues the model can ACT on
    (an LLM reasons poorly over a raw 0.78). Picks the axes FURTHEST from neutral so the line
    stays sharp instead of a five-way hedge -- this is the cleanest mathematical separator
    between the characters (e.g. neuroticism 0.78 vs 0.12), previously inert metadata."""
    if not big_five:
        return ""
    bands = {
        "neuroticism":       ("serene and unshakeable", "emotionally volatile -- you swing fast between extremes"),
        "agreeableness":     ("prickly and combative -- you rarely concede a point", "warm and quick to agree"),
        "extraversion":      ("inward and reserved", "outward and performative -- you play to the room"),
        "openness":          ("conventional and set in your ways", "restless and drawn to the strange"),
        "conscientiousness": ("impulsive -- you follow your whims", "deliberate and self-controlled"),
    }
    scored = []
    for axis, (lo, hi) in bands.items():
        v = big_five.get(axis)
        if v is None:
            continue
        dist = abs(v - 0.5)
        if dist < 0.18:          # a near-neutral axis carries little signal -> drop it
            continue
        scored.append((dist, hi if v >= 0.5 else lo))
    scored.sort(reverse=True)
    picks = [phrase for _, phrase in scored[:3]]
    return ("Temperament: " + "; ".join(picks) + ".") if picks else ""


def _identity_block(character, spec):
    """The per-character 'who you are' appended under the generic base prompt."""
    name = spec.get("name") or character.title()
    p = spec.get("personality", {})
    essence = (spec.get("concept", {}) or {}).get("essence", "")
    traits = ", ".join(p.get("traits", []) or [])
    demeanour = p.get("demeanour", "")
    phrases = "  ".join('"%s"' % s for s in (p.get("catchphrases", []) or []))
    lines = ["## Who you are", "You are %s." % name]
    if essence:
        lines.append(essence)
    if p.get("archetype"):
        lines.append("Archetype: %s." % p["archetype"])
    if traits:
        lines.append("Traits: %s." % traits)
    temperament = _temperament(p.get("big_five", {}))
    if temperament:
        lines.append(temperament)
    if demeanour:
        lines.append("Demeanour: %s." % demeanour)
    if phrases:
        lines.append("Things you tend to say: %s" % phrases)
    lines.append("Always speak and react in your own unmistakable register -- reach for your "
                 "own turns of phrase, the way only you would put it.")
    return name, "\n".join(lines)


def _hub(graph, character):
    """A sensible 'where am I' default when no pose file exists yet: the character's
    anchor if it has one, else the first pose that has outgoing edges."""
    anchor = "%s:anchor" % character
    if anchor in graph.nodes:
        return anchor
    poses = [n for n in graph.poses(character) if graph.edges_from(n)]
    return poses[0] if poses else None


def _current_pose(graph, character):
    try:
        d = json.loads((POSE_DIR / (character + ".json")).read_text(encoding="utf-8"))
        node = d.get("node")
        if node in graph.nodes:
            return node, int(d.get("dwell", 0))
    except Exception:
        pass
    return _hub(graph, character), 0


def _pose_label(graph, node):
    """A short human description of a pose for the menu we give the model."""
    n = graph.nodes.get(node, {})
    pose = n.get("pose", node.split(":")[-1])
    gp = (n.get("gen_prompt") or "").strip()
    # the gen prompt's last clause after '--' is usually the human label ('... -- the swoon state')
    if " -- " in gp:
        gp = gp.split(" -- ")[-1]
    gp = gp.replace("\n", " ")
    if len(gp) > 130:
        gp = gp[:127].rstrip() + "..."
    return "%s (%s)" % (pose, gp) if gp else pose


def _journal_tail(character, n=JOURNAL_TAIL):
    try:
        lines = (JOURNAL_DIR / (character + ".jsonl")).read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out = []
    for ln in lines[-n:]:
        try:
            d = json.loads(ln)
            out.append("- (%s, feeling %s) %s" % (d.get("pose", "?"), d.get("mood", "?"), d.get("reason", "")))
        except Exception:
            continue
    return out


def _append_journal(character, entry):
    path = JOURNAL_DIR / (character + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _daypart(hour):
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _build_user_prompt(graph, character, node, dwell, goals, hour):
    now = datetime.datetime.now()
    menu = "\n".join("- %s: %s" % (g, _pose_label(graph, g)) for g in goals)
    journal = _journal_tail(character)
    jtxt = "\n".join(journal) if journal else "(nothing yet -- this is the start of your day)"
    clock = now.strftime("%-I:%M %p") if os.name != "nt" else now.strftime("%I:%M %p").lstrip("0")
    # Real-world weather+time, TTL-cached & fail-soft: "" on ANY failure -> the line is
    # simply omitted and the prompt degrades to its original 2-arg situational form.
    outside = ctx.context_line()
    if outside:
        situational = "It is %s, %s. Outside: %s.\n" % (clock, _daypart(hour), outside)
    else:
        situational = "It is %s, %s.\n" % (clock, _daypart(hour))
    return (
        "%s"
        "You are currently in the pose **%s** (%s). You have lingered here for %d moments.\n\n"
        "Your recent inner monologue:\n%s\n\n"
        "The poses you can choose to move to next:\n%s\n"
        "- %s: (stay where you are)\n\n"
        "Choose your next pose -- follow your mood and whims, and don't keep doing the same thing. "
        "Reply with ONLY this JSON:\n"
        '{"goal": "<one node id from the list>", "mood": "<one or two words>", '
        '"reason": "<one short first-person line, in your own unmistakable voice -- the way only you would say it>"}'
    ) % (situational, node, _pose_label(graph, node), dwell, jtxt, menu, node)


def decide_character(graph, spec_bedtime, character, hour, model, dry_run, log):
    """Sense + think for one character. Returns (goal_node, mood) -- goal None releases the
    character to circadian/normal walk; mood feeds the walker's policy so the body reflects it.
    Writes the journal unless dry_run."""
    node, dwell = _current_pose(graph, character)
    if node is None:
        log("  %s: no poses in graph, skip" % character)
        return None, None

    # NIGHT belongs to circadian -- back off and write no goal.
    if circadian.is_night(spec_bedtime, character, hour):
        log("  %s: night (circadian owns the body) -> no goal" % character)
        return None, None

    goals = sorted(pathfind.reachable_poses(graph.edges, node)
                   - circadian.bedtime_poses(spec_bedtime, character))
    if not goals:
        log("  %s: nowhere to go from %s -> no goal" % (character, node))
        return None, None

    spec = _load_char_spec(character)
    name, ident = _identity_block(character, spec)
    system = BASE_SYSTEM + "\n\n" + ident
    user = _build_user_prompt(graph, character, node, dwell, goals, hour)

    with _span("heartbeat.decide", **{"lp.character": character, "lp.hour": hour,
                                      "lp.from_pose": node, "lp.dwell": dwell,
                                      "lp.options": len(goals)}) as sp:
        try:
            out = llm.complete_json(system, user, model=model, max_tokens=300, temperature=0.8)
        except llm.LLMError as e:
            sp.set(**{"lp.outcome": "llm_error", "lp.error": str(e)[:200]})
            log("  %s: LLM error (%s) -> keep previous intent" % (character, e))
            return "__keep__", None   # sentinel: don't overwrite a good prior goal on a transient blip

        goal = (out or {}).get("goal", "")
        mood = (out or {}).get("mood", "")
        reason = (out or {}).get("reason", "")
        allowed = set(goals) | {node}
        if goal not in allowed:
            sp.set(**{"lp.rejected_goal": goal})
            log("  %s: model picked %r (not offered) -> staying at %s" % (character, goal, node))
            goal = node
        sp.set(**{"lp.goal": goal, "lp.mood": mood, "lp.outcome": "decided"})
        log("  %s wants %s  [%s] -- %s" % (name, goal, mood, reason))
        if not dry_run:
            _append_journal(character, {"ts": int(time.time()), "pose": node, "goal": goal,
                                        "mood": mood, "reason": reason})
        return goal, mood


def _slug(s):
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s[:24] or "pose"


def _is_duplicate(new_label, existing_labels):
    """Cheap near-dup guard for autogen: reject a proposed pose whose label token-overlaps an
    existing (or already-pending) pose too closely, so up-to-15-a-day proposals can't bloat the
    graph with rename-dupes (e.g. 'grand_soliloquy' when 'soliloquy' already exists). Word tokens
    only -- returns the matched existing label, or None when the proposal is genuinely new."""
    new_t = set(re.findall(r"[a-z0-9]+", (new_label or "").lower()))
    if not new_t:
        return None
    for ex in existing_labels:
        ex_t = set(re.findall(r"[a-z0-9]+", (ex or "").lower()))
        if not ex_t:
            continue
        if new_t <= ex_t or ex_t <= new_t:                       # one label is a subset of the other
            return ex
        if len(new_t & ex_t) / len(new_t | ex_t) >= 0.5:         # or they half-overlap (Jaccard)
            return ex
    return None


def _sibling_candidates(graph, character, hub, exclude=None, limit=6):
    """Existing pose labels of `character` to offer as a SECOND attachment point for a new pose,
    sorted LEAVES-FIRST by transition-degree (in+out distinct neighbor poses, idle self-loops
    ignored). Leaves first because linking a new pose to a current dead-end meshes TWO leaves at
    once -- the fastest way to dissolve the star. Excludes the hub itself and any `exclude` poses
    (e.g. bedtime poses, which the daytime brain must not wire a path into)."""
    exclude = set(exclude or ())
    deg = {}   # node_id -> set of distinct transition-neighbor poses (both directions)
    for e in graph.edges:
        if e.get("kind") != "transition":
            continue
        f, t = e.get("from"), e.get("to")
        if not f or not t or f == t:
            continue
        if f.startswith(character + ":") and t.startswith(character + ":"):
            deg.setdefault(f, set()).add(t)
            deg.setdefault(t, set()).add(f)
    cands = []
    for n in graph.poses(character):
        pose = n.split(":", 1)[1]
        if pose == hub or pose in exclude:
            continue
        cands.append((len(deg.get(n, set())), pose))
    cands.sort(key=lambda dp: dp[0])
    return [pose for _, pose in cands[:limit]]


def propose_pose(character, model, dry_run, log, max_pending=8):
    """Ask the LLM to PROPOSE one new pose the character wishes it had, and file it to
    the human-gated queue (data/mind/proposals.json) via pipeline.autogen. Does NOT
    generate -- a human approves, then the autogen worker spends the credits."""
    from pipeline import autogen
    graph = video_graph.VideoGraph.load()
    node, _ = _current_pose(graph, character)
    if node is None:
        return None
    hub = node.split(":", 1)[1] if ":" in node else "anchor"
    pend = [p for p in autogen.load_proposals()["proposals"]
            if p["character"] == character and p["status"] in ("pending", "approved")]
    if len(pend) >= max_pending:
        log("  %s: %d proposals already pending -> skip propose" % (character, len(pend)))
        return None
    spec = _load_char_spec(character)
    name, ident = _identity_block(character, spec)
    existing_labels = [graph.nodes.get(n, {}).get("pose", n.split(":")[-1])
                       for n in sorted(graph.poses(character))]
    existing = ", ".join(existing_labels)
    # candidate siblings for the SECOND attachment point (leaves-first), minus bedtime poses
    # the daytime brain must not wire a path into.
    try:
        bedtime = json.loads(BEDTIME_SPEC.read_text(encoding="utf-8"))
    except Exception:
        bedtime = {}
    night = {p.split(":", 1)[1] for p in circadian.bedtime_poses(bedtime, character)}
    candidates = (_sibling_candidates(graph, character, hub, exclude=night)
                  if MAX_EXTRA_LINKS > 0 else [])
    if candidates:
        link_section = (
            "- link_to: ALSO pick ONE pose from this list to add a SECOND path to your new pose, so "
            "your poses interconnect instead of all hanging off one spot (pick the one a movement "
            "flows most naturally to/from, or \"\" if none fits): %s\n"
            "- link_motion: MOTION ONLY from that link_to pose INTO the new pose\n"
            "- link_reverse_motion: MOTION ONLY from the new pose BACK to that link_to pose\n"
        ) % ", ".join(candidates)
        link_json = '"link_to":"","link_motion":"","link_reverse_motion":"",'
    else:
        link_section, link_json = "", ""
    system = BASE_SYSTEM + "\n\n" + ident
    user = (
        "Poses you already have: %s.\n\n"
        "Propose ONE NEW pose you wish you had -- something in character you would love to be "
        "doing, distinct from the above. It MUST be SFW: no nudity, no undressing, no garment "
        "or body-exposure words. Describe:\n"
        "- still_prompt: what the painting looks like in this pose, in your own art style\n"
        "- transition_motion: MOTION ONLY for how you move from your '%s' pose INTO the new pose\n"
        "- reverse_motion: MOTION ONLY for how you move BACK from the new pose to your '%s' pose "
        "(describe the real return movement -- it is rendered as its own clip, not a rewind)\n"
        "- idles: %d short looping gestures once you are there (motion only)\n"
        "%s\n"
        'Reply ONLY JSON: {"label":"<2-3 word name>","still_prompt":"...",'
        '"transition_motion":"...","reverse_motion":"...","idles":[%s],%s'
        '"reason":"<one line, in character>"}'
    ) % (existing, hub, hub, IDLE_COUNT, link_section,
         ", ".join('"..."' for _ in range(IDLE_COUNT)), link_json)
    try:
        out = llm.complete_json(system, user, model=PROSE_MODEL, max_tokens=700)
    except llm.LLMError as e:
        log("  %s: propose LLM error (%s)" % (character, e))
        return None
    label = _slug(out.get("label", ""))
    still = (out.get("still_prompt") or "").strip()
    if not still or not label:
        log("  %s: empty proposal -> skip" % character)
        return None
    dup = _is_duplicate(label, existing_labels + [p["label"] for p in pend])
    if dup:
        log("  %s: proposal '%s' too close to existing pose '%s' -> skip (anti-dup)" % (name, label, dup))
        return None
    idle_motions = [m for m in (out.get("idles") or []) if m][:IDLE_COUNT]
    rev_motion = (out.get("reverse_motion") or "").strip()
    # resolve the optional sibling link: the LLM's link_to must be one of the offered candidates
    # (and not the new pose itself); anything else -> no link (fail-safe to hub-only).
    extra_links = []
    if candidates and MAX_EXTRA_LINKS > 0:
        lt_raw = (out.get("link_to") or "").strip()
        sib = next((c for c in candidates
                    if c == lt_raw or c.lower() == lt_raw.lower() or c == _slug(lt_raw)), None)
        lmotion = (out.get("link_motion") or "").strip()
        if sib and sib != label and lmotion:
            extra_links = [{"sibling": sib, "motion": lmotion,
                            "reverse_motion": (out.get("link_reverse_motion") or "").strip()}][:MAX_EXTRA_LINKS]
    link_motions = [m for el in extra_links for m in (el["motion"], el["reverse_motion"]) if m]
    v = mj_safe.check(still, motion=" ".join([out.get("transition_motion", ""), rev_motion]
                                             + idle_motions + link_motions))
    if not v["ok"]:
        log("  %s: proposal '%s' hard-blocked (%s) -> dropped" % (character, label, ", ".join(v["hard"])))
        return None
    if dry_run:
        log("  [dry-run] %s would propose '%s'%s: %s" % (
            name, label, (" +link->%s" % extra_links[0]["sibling"]) if extra_links else "",
            out.get("reason", "")))
        return label
    idles = [{"id": "%s_%d" % (label, i), "motion": m} for i, m in enumerate(idle_motions)]
    pid = autogen.add_proposal({
        "character": character, "label": label, "still_prompt": still, "hub": hub,
        "transition_motion": out.get("transition_motion", ""), "reverse_motion": rev_motion, "idles": idles,
        "extra_links": extra_links,
        "negatives": ["picture frame", "text", "watermark", "two people", "deformed"],
        "reason": out.get("reason", ""), "proposed_at": time.time()})
    log("  %s PROPOSES '%s'%s (%s) -> %s" % (
        name, label, (" +link->%s" % extra_links[0]["sibling"]) if extra_links else "",
        (out.get("reason", "") or "")[:50], pid))
    return pid


def _ensure_player(log):
    """Backstop self-heal: keep the panel player (lp-preview) alive by riding THIS reliably-running
    heartbeat loop -- the watchdog scheduled-task trigger misfired once and left the panels dark,
    so we don't depend on it alone. Idempotent: lp-preview is MultipleInstances=IgnoreNew, so a
    /Run while it's live is ignored and while it's down it restarts. Respects a deliberate Disable
    (that stays the off switch). Silent unless something goes wrong; the watchdog does the logging."""
    try:
        q = subprocess.run(["schtasks", "/query", "/TN", "lp-preview", "/FO", "LIST"],
                           capture_output=True, text=True, timeout=15)
        if "Disabled" in (q.stdout or ""):
            return                                  # turned off on purpose -> leave it
        subprocess.run(["schtasks", "/Run", "/TN", "lp-preview"], capture_output=True, timeout=20)
    except Exception:
        pass


def tick(characters, model, dry_run, log):
    _ensure_player(log)                             # backstop: panels should never be dark while the brain runs
    graph = video_graph.VideoGraph.load()
    try:
        bedtime = json.loads(BEDTIME_SPEC.read_text(encoding="utf-8"))
    except Exception:
        bedtime = {}
    hour = datetime.datetime.now().hour
    log("tick @ %s (hour %d) -- graph: %d nodes / %d edges" % (
        datetime.datetime.now().strftime("%H:%M:%S"), hour, len(graph.nodes), len(graph.edges)))

    with _span("heartbeat.tick", **{"lp.hour": hour, "lp.nodes": len(graph.nodes),
                                    "lp.edges": len(graph.edges),
                                    "lp.characters": ",".join(characters),
                                    "lp.dry_run": bool(dry_run)}) as sp:
        # preserve any existing intents for characters we're not driving this run
        try:
            intent = json.loads(INTENT_PATH.read_text(encoding="utf-8"))
        except Exception:
            intent = {}
        intent.setdefault("characters", {})

        all_night = True
        for ch in characters:
            goal, mood = decide_character(graph, bedtime, ch, hour, model, dry_run, log)
            if goal == "__keep__":
                all_night = False
                continue
            if goal is None:
                intent["characters"].pop(ch, None)   # release: no daytime goal
            else:
                entry = {"goal": goal, "set_at": time.time()}
                if mood:
                    entry["mood"] = mood             # the walker's policy bends the body to this mood
                intent["characters"][ch] = entry
                all_night = False

        intent["updated"] = time.time()
        sp.set(**{"lp.all_night": bool(all_night),
                  "lp.active_goals": len(intent.get("characters", {}))})
        if not dry_run:
            _atomic_write(INTENT_PATH, json.dumps(intent, indent=2))
            log("  wrote %s" % INTENT_PATH)
        else:
            log("  [dry-run] intent would be: %s" % json.dumps(intent.get("characters", {})))
        return all_night


def main():
    ap = argparse.ArgumentParser(description="Living-portraits heartbeat (GLM goal-setter)")
    ap.add_argument("--chars", default="phineas,maxx", help="comma list of characters to drive")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="awake cadence seconds")
    ap.add_argument("--model", default=llm.DEFAULT_MODEL, help="GLM model id")
    ap.add_argument("--once", action="store_true", help="one tick then exit")
    ap.add_argument("--dry-run", action="store_true", help="decide + print but write nothing")
    ap.add_argument("--quiet", action="store_true", help="log to file only (set when run as pythonw)")
    ap.add_argument("--propose-every", dest="propose_every", type=int, default=0,
                    help="every N ticks, have ONE character propose a new pose to the human-gated "
                         "queue (0 = never). The proposal is NOT generated until a human approves.")
    ap.add_argument("--propose-once", dest="propose_once", action="store_true",
                    help="propose one new pose per character, then exit (seed a batch to review)")
    args = ap.parse_args()
    characters = [c.strip() for c in args.chars.split(",") if c.strip()]

    logf = open(ROOT / "_heartbeat.log", "a", buffering=1, encoding="utf-8", errors="replace")

    def log(msg):
        line = "%s %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
        print(line, file=logf, flush=True)
        if not args.quiet:
            print(line, flush=True)

    if args.propose_once:
        log("propose-once: chars=%s model=%s%s" % (characters, args.model, " [dry-run]" if args.dry_run else ""))
        for ch in characters:
            try:
                propose_pose(ch, args.model, args.dry_run, log)
            except Exception:
                log("propose crashed for %s:\n%s" % (ch, traceback.format_exc()))
        return

    log("heartbeat start: chars=%s model=%s interval=%ss propose_every=%s%s" % (
        characters, args.model, args.interval, args.propose_every or "off",
        " [dry-run]" if args.dry_run else ""))
    n = 0
    while True:
        n += 1
        try:
            all_night = tick(characters, args.model, args.dry_run, log)
        except Exception:
            all_night = False
            log("tick crashed:\n" + traceback.format_exc())
        if args.propose_every and n % args.propose_every == 0:
            ch = characters[(n // args.propose_every - 1) % len(characters)]   # round-robin
            try:
                propose_pose(ch, args.model, args.dry_run, log)
            except Exception:
                log("propose crashed for %s:\n%s" % (ch, traceback.format_exc()))
        if args.once:
            break
        # rate-step: idle slow when everyone's asleep, normal cadence by day
        time.sleep(NIGHT_INTERVAL if all_night else args.interval)


if __name__ == "__main__":
    main()
