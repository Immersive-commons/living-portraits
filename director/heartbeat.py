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
from runtime import circadian, journal_score, pathfind, policy, video_graph

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
JOURNAL_TOKENS = 700         # token budget for RETRIEVED monologue (runtime/journal_score.py).
                             # Replaced JOURNAL_TAIL = 5, which showed the character ~35 minutes
                             # of a 64-day life and made a 2,060-repetition rut unnoticeable.
NEIGHBOUR_STALE = 1800.0     # ignore a neighbour's published pose older than this (dark panel)
FRONTIER_SHOWN = 2           # unvisited poses named per tick ("a version of you you've never been")
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


def _journal_entries(character):
    return journal_score.load(JOURNAL_DIR / (character + ".jsonl"))


def _display_name(character):
    return _load_char_spec(character).get("name") or character.title()


def _neighbour_lines(character, now=None):
    """What this character can ACTUALLY SEE of the others on the wall: each neighbour's
    current pose and the mood behind its last decision.

    Reconstructed at READ time from files that already exist and are already
    single-writer (`pose/<char>.json` written by that panel's walker, the neighbour's own
    journal). Nothing is stored: this is a cross-character edge computed per reader, which
    is also how Generative Agents handles relationships. Asymmetric on purpose -- each
    character's read of the other is built inside its own prompt.

    A neighbour whose panel has gone dark (pose file older than NEIGHBOUR_STALE) is simply
    not seen, rather than reported as frozen in place. Fail-soft: any error -> no line."""
    now = time.time() if now is None else now
    out = []
    try:
        others = sorted(p.stem for p in POSE_DIR.glob("*.json"))
    except Exception:
        return []
    for other in others:
        if other == character:
            continue
        try:
            path = POSE_DIR / (other + ".json")
            if now - path.stat().st_mtime > NEIGHBOUR_STALE:
                continue                                   # dark panel: unseen, not "still"
            d = json.loads(path.read_text(encoding="utf-8"))
            node = d.get("node")
            if not node:
                continue
            pose = str(node).split(":")[-1]
            mood = ""
            entries = _journal_entries(other)
            if entries:
                mood = (entries[-1].get("mood") or "").strip()
            if mood:
                out.append('On the wall beside you, %s is at %s, feeling "%s".'
                           % (_display_name(other), pose, mood))
            else:
                out.append("On the wall beside you, %s is at %s." % (_display_name(other), pose))
        except Exception:
            continue
    return out


def _frontier_line(graph, node, goals, visited, hops):
    """The poses reachable from here that this character has NEVER been in.

    3D-Mem (arXiv 2411.17735) calls this frontier memory: what has not been explored is
    itself retrievable content. It turns the graph's least flattering statistic -- the
    long tail of near-dead-end poses nothing ever walks to -- into a want that is grounded
    in real topology instead of persona prose. Empty (all explored) -> ""."""
    unseen = [g for g in goals if g not in visited]
    if not unseen:
        return ""
    unseen.sort(key=lambda g: (hops.get(g, 10 ** 6), g))
    named = []
    for g in unseen[:FRONTIER_SHOWN]:
        d = hops.get(g)
        step = "%d steps away" % d if d else "right beside you"
        named.append("%s (%s)" % (_pose_label(graph, g), step))
    return ("There are %d poses you have never once been in. The nearest: %s."
            % (len(unseen), "; ".join(named)))


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


def _build_user_prompt(graph, character, node, dwell, goals, hour, hops=None):
    now = datetime.datetime.now()
    ts = time.time()
    menu = "\n".join("- %s: %s" % (g, _pose_label(graph, g)) for g in goals)

    # --- MEMORY: scored retrieval over the whole journal, not the last five lines.
    # recency + importance(surprisal of the want) + relevance(transition hops from here),
    # under a token budget, rendered chronologically with relative ages.
    entries = _journal_entries(character)
    picked = journal_score.select(entries, now=ts, distmap=hops, token_budget=JOURNAL_TOKENS)
    journal = journal_score.render(picked, now=ts)
    jtxt = "\n".join(journal) if journal else "(nothing yet -- this is the start of your day)"
    aggregate = journal_score.aggregate_line(entries, now=ts)
    frontier = _frontier_line(graph, node, goals, journal_score.visited_poses(entries), hops or {})
    neighbours = _neighbour_lines(character, now=ts)

    extra = "\n".join(x for x in ([aggregate, frontier] + neighbours) if x)
    extra = ("\n" + extra + "\n") if extra else ""
    bands = ", ".join(sorted(policy.MOOD_BIAS))
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
        "You are currently in the pose **%s** (%s). You have lingered here for %d moments.\n"
        "%s\n"
        "What you remember, from across your whole life here:\n%s\n\n"
        "The poses you can choose to move to next:\n%s\n"
        "- %s: (stay where you are)\n\n"
        "Choose your next pose -- follow your mood and whims, and don't keep doing the same thing. "
        "Reply with ONLY this JSON:\n"
        '{"goal": "<one node id from the list>", "mood": "<one or two words, in your own words>", '
        '"band": "<the ONE word from this list closest to your mood: %s>", '
        '"reason": "<one short first-person line, in your own unmistakable voice -- the way only you would say it>"}'
    ) % (situational, node, _pose_label(graph, node), dwell, extra, jtxt, menu, node, bands)


def _resolve_goal(goal, character, allowed):
    """Map the model's raw `goal` string onto one of the offered node ids.

    The menu lists FULLY-QUALIFIED ids ("phineas:withered_rose") but the brain
    routinely answers with the BARE pose label ("withered_rose") -- it names the
    right pose and drops the prefix. An exact-membership test scores that a miss,
    so the character stays put and the tick is wasted.

    Measured on hil 2026-07-31 over 20k heartbeat log lines: 1179 rejected
    decisions, of which 1170 (99.2%) were exactly this and nothing else. Both
    characters had been parked on their hub pose for hours as a result.

    Order: exact id -> re-prefixed with this character -> case/space-insensitive
    match on the bare pose label. Returns (node_id, how); (None, "novel") when the
    name is genuinely not a pose we own -- that is proposal material, not a walk
    target, and the caller keeps the character where it is.
    """
    raw = (goal or "").strip().strip('"').strip("'").strip()
    if not raw:
        return None, "empty"
    if raw in allowed:
        return raw, "exact"
    bare = raw.split(":", 1)[-1].strip()
    qualified = "%s:%s" % (character, bare)
    if qualified in allowed:
        return qualified, "prefixed"
    want = bare.lower()
    for cand in sorted(allowed):                     # sorted -> deterministic on a tie
        if cand.split(":", 1)[-1].strip().lower() == want:
            return cand, "normalized"
    return None, "novel"


def _resolve_band(raw):
    """Map the model's declared band onto policy's vocabulary, or None.

    The mood stays free text (it is the character's voice, and flattening it to eight
    words would flatten the personality). The BAND is the same feeling said once in the
    only vocabulary the body understands. Measured over the real journals, keyword-guessing
    the band from free text failed for 59% of MAXX's moods and 88% of Phineas's -- they
    were authored, journaled, and dropped at the boundary. Asking for it costs one field.
    None here is not a failure: policy._mood_bias falls back to exactly today's guessing."""
    if not raw:
        return None
    text = str(raw).strip().strip('"').strip("'").lower()
    if text in policy.MOOD_BIAS:
        return text
    for tok in text.replace("-", " ").replace(",", " ").split():
        if tok in policy.MOOD_BIAS:
            return tok
    return None


def decide_character(graph, spec_bedtime, character, hour, model, dry_run, log):
    """Sense + think for one character. Returns (goal_node, mood, band) -- goal None releases
    the character to circadian/normal walk; mood feeds the walker's policy so the body reflects
    it, and band is the mood mapped ONCE onto policy's vocabulary (88% of free-text moods used
    to reach the body as no signal at all). Writes the journal unless dry_run."""
    node, dwell = _current_pose(graph, character)
    if node is None:
        log("  %s: no poses in graph, skip" % character)
        return None, None, None

    # NIGHT belongs to circadian -- back off and write no goal.
    if circadian.is_night(spec_bedtime, character, hour):
        log("  %s: night (circadian owns the body) -> no goal" % character)
        return None, None, None

    hops = pathfind.hops_from(graph.edges, node)     # one BFS: memory relevance + frontier
    goals = sorted(pathfind.reachable_poses(graph.edges, node)
                   - circadian.bedtime_poses(spec_bedtime, character))
    if not goals:
        log("  %s: nowhere to go from %s -> no goal" % (character, node))
        return None, None, None

    spec = _load_char_spec(character)
    name, ident = _identity_block(character, spec)
    system = BASE_SYSTEM + "\n\n" + ident
    user = _build_user_prompt(graph, character, node, dwell, goals, hour, hops=hops)

    with _span("heartbeat.decide", **{"lp.character": character, "lp.hour": hour,
                                      "lp.from_pose": node, "lp.dwell": dwell,
                                      "lp.options": len(goals)}) as sp:
        try:
            out = llm.complete_json(system, user, model=model, max_tokens=300, temperature=0.8)
        except llm.LLMError as e:
            sp.set(**{"lp.outcome": "llm_error", "lp.error": str(e)[:200]})
            log("  %s: LLM error (%s) -> keep previous intent" % (character, e))
            return "__keep__", None, None   # sentinel: don't overwrite a good prior goal on a blip

        goal = (out or {}).get("goal", "")
        mood = (out or {}).get("mood", "")
        reason = (out or {}).get("reason", "")
        band = _resolve_band((out or {}).get("band"))
        allowed = set(goals) | {node}
        resolved, how = _resolve_goal(goal, character, allowed)
        if resolved is None:
            sp.set(**{"lp.rejected_goal": goal, "lp.goal_match": how})
            log("  %s: model picked %r (no such pose) -> staying at %s" % (character, goal, node))
            goal = node
        else:
            if how != "exact":
                sp.set(**{"lp.goal_raw": goal})
                log("  %s: repaired %r -> %s (%s)" % (character, goal, resolved, how))
            goal = resolved
        sp.set(**{"lp.goal": goal, "lp.mood": mood, "lp.band": band or "",
                  "lp.goal_match": how, "lp.outcome": "decided"})
        log("  %s wants %s  [%s%s] -- %s" % (name, goal, mood,
                                             ("/" + band) if band else "", reason))
        if not dry_run:
            entry = {"ts": int(time.time()), "pose": node, "goal": goal,
                     "mood": mood, "reason": reason}
            if band:
                entry["band"] = band
            _append_journal(character, entry)
        return goal, mood, band


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
            goal, mood, band = decide_character(graph, bedtime, ch, hour, model, dry_run, log)
            if goal == "__keep__":
                all_night = False
                continue
            if goal is None:
                intent["characters"].pop(ch, None)   # release: no daytime goal
            else:
                entry = {"goal": goal, "set_at": time.time()}
                if mood:
                    entry["mood"] = mood             # the walker's policy bends the body to this mood
                if band:
                    entry["band"] = band             # ...and the band says it in the body's vocabulary
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
