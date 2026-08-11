"""policy.py -- ONE weighted edge-selection policy for the living-portrait walk.

Replaces the old layered tug-of-war (circadian FORCES out / mind FORCES in / random
walk with a separate anti-pendulum guard the forced branches bypassed) with a single
softmax over the candidate edges, where every concern is a WEIGHT evaluated together:

  * anti-reverse   -- the reverse of the clip just played, and a transition straight
                      back to the pose we came from, are heavily down-weighted -> no
                      "play a move then run it backwards" pendulum (the #1 visible bug).
                      Applied in ALL cases, so a goal-walk or a leaf pose can't force it.
  * novelty        -- decaying recency over recently-PLAYED clips and recently-VISITED
                      poses -> the walk spreads across the graph instead of orbiting a hub
                      and cycling the same 1-2 idles.
  * goal-pull      -- a SOFT gradient toward the heartbeat's goal pose (BFS distance):
                      'beeline' pulls hard (~the old forced single step), 'wander' pulls
                      gently (drifts there scenically). Shares the circadian EXCLUDE mask,
                      so a daytime goal-walk is never routed THROUGH a bedtime pose (the
                      deadlock that pinned MAXX in an idle<->empty bounce).
  * mood           -- the heartbeat's chosen mood finally bends the BODY: restless -> more
                      transitions + novelty; weary -> more idles + longer dwell; fixated ->
                      fewer transitions (lingers). Personality becomes visible, not just
                      journaled.
  * style          -- OPT-IN (`styles=`). The mood picks HOW MANY transitions; this picks
                      WHICH ONE. `runtime/edge_style.py` reads each clip's own
                      `motion_prompt` into a manner (storm / saunter / collapse / rise /
                      ...) and a valence, and a weary character then prefers the clip that
                      sags over the clip that lunges -- both of which are `kind:
                      transition` and indistinguishable to every other term here.
                      Preference only: bounded well above zero (edge_style.STYLE_FLOOR),
                      applied AFTER anti-reverse / novelty / goal so it can lean those
                      terms but never overturn them, and absent (or unstyled) it leaves
                      every weight byte-identical to the pre-style behaviour.

Pure + import-safe (stdlib only). Imported by the 10fps render loop, so it must stay
fast and side-effect free. Never strands: if every candidate masks out, it falls back to
a uniform pick over the raw out-edges, so a panel can never freeze.

    from runtime import policy
    edge = policy.choose(node, out_edges, all_edges, goal="maxx:cat_whisper",
                         mood="restless", route="wander", exclude=bedtime_labels,
                         prev_node=prev, last_id=last, recent_clips=rc, recent_nodes=rn,
                         styles=edge_style.index(all_edges), rng=random)
"""
from __future__ import annotations

import math
from collections import deque

# Sibling lookup mirrors how the render loop is wired: player.py puts runtime/ on
# sys.path (bare import), everything else imports the package. Styling is a preference
# layer, so a missing module degrades to exactly the pre-style behaviour instead of
# taking the panels down.
try:                                     # pragma: no cover - wiring, not behaviour
    import edge_style as _edge_style
except ImportError:                      # pragma: no cover
    try:
        from runtime import edge_style as _edge_style
    except ImportError:
        _edge_style = None

# --- tunables (kept as module constants so the tests pin the behaviour, not magic numbers).
REVERSE_PENALTY = 0.04     # multiply a transition that backtracks to prev_node / reverses last clip
LAST_CLIP_PENALTY = 0.02   # never replay the exact clip just shown
GOAL_ARRIVE_BOOST = 8.0    # a transition that lands ON the goal pose
AWAY_PENALTY = 0.15        # a transition that increases BFS distance to the goal
GOAL_HOLD_BOOST = 6.0      # idles AT the goal pose (pin there until a new goal)
ROUTE_PULL = {"beeline": 7.0, "wander": 2.0}   # toward-goal multiplier by route style

# --- ESCAPE VELOCITY: the longer a character has been somewhere, the more its exits are
# worth. Without this, a pose with ONE exit and three idles is a trap under a low-energy
# band: measured 2026-08-10, `fixated` weights idles 1.5x and transitions 0.4x, so a single
# exit held 13% against three idles' 87%, and Phineas looped the same three clips for an
# hour at commanding_aether. 56 of his 120 poses have one exit or none.
#
# This is deliberately NOT a fix to the graph. Generating return clips for all 112 one-exit
# poses across both characters is ~840 Higgsfield credits, 28% of a month, to reclaim time
# measured at 18% of dwell spread across a long tail whose worst single member is 1.1%. A
# weight costs nothing and covers poses that do not exist yet.
#
# Ramps only AFTER a genuine dwell so ordinary lingering is untouched, and the caller passes
# dwell=0 at the sleep pose and at night -- a character is SUPPOSED to stay put for hours
# there, and boosting its exits would wake it up at 2am.
ESCAPE_AFTER = 8           # idle units of honest dwell before the pull starts
ESCAPE_PER_UNIT = 0.35     # added transition multiplier per unit beyond that
ESCAPE_MAX = 6.0           # ceiling: a strong nudge, never a forced march
REVERSE_FORGIVE = 20       # idle units over which the anti-reverse penalty relaxes to none.
                           # Anti-reverse is a SHORT-timescale guard -- it exists to stop
                           # "play a move, then play it backwards", which is only ugly when
                           # it happens immediately. Held forever it turns a pendant pose
                           # (one neighbour, in and out) into a cell, because the only exit
                           # IS the backtrack. Both guards were right; together they trapped.


def _reverse_penalty(dwell=0):
    """REVERSE_PENALTY at a fresh arrival, relaxing to 1.0 (no penalty) by
    ESCAPE_AFTER + REVERSE_FORGIVE idle units. The pendulum stays fixed; the cell opens."""
    over = (dwell or 0) - ESCAPE_AFTER
    if over <= 0:
        return REVERSE_PENALTY
    t = min(1.0, over / float(REVERSE_FORGIVE))
    return REVERSE_PENALTY + (1.0 - REVERSE_PENALTY) * t

# mood -> (transition multiplier, idle multiplier, novelty exponent).
# novelty exponent >1 sharpens the novelty preference (more exploratory).
MOOD_BIAS = {
    "restless":  (1.8, 0.7, 1.4),
    "curious":   (1.5, 0.8, 1.5),
    "playful":   (1.5, 0.9, 1.3),
    "weary":     (0.5, 1.6, 0.9),
    "calm":      (0.7, 1.3, 1.0),
    "serene":    (0.7, 1.3, 1.0),
    "fixated":   (0.4, 1.5, 0.8),
    "wistful":   (0.7, 1.2, 1.0),
}
# "alive but calm" baseline: the heartbeat's moods are free-text and often won't name an
# exact band, so the default still moves enough that a long dwell on a 2-idle pose doesn't
# turn into visible A/B idle flicker.
DEFAULT_MOOD = (1.3, 0.85, 1.3)

# Energy buckets for FREE-TEXT moods (the LLM says "feral devotion", not "restless"). We scan
# the phrase for energy words and fall into a band; only then the neutral default.
_HIGH_ENERGY = ("restless", "feral", "frantic", "manic", "wild", "fierce", "eager", "electric",
                "charged", "excited", "defiant", "playful", "curious", "agitated", "hungry",
                "burning", "furious", "giddy", "alert", "jittery", "ravenous", "vengeful")
_LOW_ENERGY = ("weary", "tired", "calm", "serene", "somber", "sombre", "melancholy", "wistful",
               "drowsy", "heavy", "quiet", "still", "sleepy", "subdued", "hollow", "pensive",
               "resigned", "tender", "mournful", "languid", "wary", "guarded")
_BAND = {"high": (1.7, 0.75, 1.4), "low": (0.6, 1.4, 0.95)}

# Live weather/time NUDGE (director.context.context_signal -> "high"|"low"|None), layered ON TOP
# of the LLM mood. A gentle (transition_mul, idle_mul) tilt so the mood still LEADS and the weather
# only leans the balance: stormy/wind -> more transitions (agitated, alive); clear-calm or deep
# night -> more dwell (settled, serene). None = no tilt = exactly the pre-weather behaviour.
_CONTEXT_NUDGE = {"high": (1.35, 0.75), "low": (0.75, 1.35)}


def _mood_bias(mood, band=None):
    """`band` is the heartbeat's OWN mapping of its free-text mood onto MOOD_BIAS, decided
    once by the brain that authored the mood (heartbeat._resolve_band) instead of guessed
    here by keyword. When present it wins; when absent this is byte-identical to the
    keyword path it replaces, so an old intent.json (or a brain that omits the field)
    behaves exactly as before."""
    if band:
        hit = MOOD_BIAS.get(str(band).strip().lower())
        if hit:
            return hit
    if not mood:
        return DEFAULT_MOOD
    text = str(mood).strip().lower()
    if not text:
        return DEFAULT_MOOD
    # 1) exact band name on any token (restless / weary / ...)
    for tok in text.replace("-", " ").split():
        if tok in MOOD_BIAS:
            return MOOD_BIAS[tok]
    # 2) free-text energy bucket
    if any(w in text for w in _HIGH_ENERGY):
        return _BAND["high"]
    if any(w in text for w in _LOW_ENERGY):
        return _BAND["low"]
    return DEFAULT_MOOD


def _style_band(mood, band):
    """The band name `edge_style` should key its manner preference on, or None.

    Deliberately the SAME precedence `_mood_bias` uses -- declared band first, then an
    exact band token in the free text, then the energy bucket -- so the body's choice of
    WHICH clip and its choice of HOW MANY clips are never reasoning from two different
    readings of one mood. None means "no opinion", which is a multiplier of exactly 1.0
    and not a neutral bucket."""
    if band:
        b = str(band).strip().lower()
        if b in MOOD_BIAS or b in _BAND:
            return b
    text = str(mood or "").strip().lower()
    if not text:
        return None
    for tok in text.replace("-", " ").split():
        if tok in MOOD_BIAS:
            return tok
    if any(w in text for w in _HIGH_ENERGY):
        return "high"
    if any(w in text for w in _LOW_ENERGY):
        return "low"
    return None


def dist_to_goal(all_edges, goal):
    """BFS distance (in transition hops) from every node TO `goal`, over the transition
    sub-graph reversed. {node: hops}. Cheap; computed once per pick when there's a goal."""
    if not goal:
        return {}
    radj = {}
    for e in all_edges:
        if e.get("kind") == "transition":
            radj.setdefault(e.get("to"), []).append(e.get("from"))
    dist = {goal: 0}
    q = deque([goal])
    while q:
        n = q.popleft()
        for prev in radj.get(n, []):
            if prev not in dist:
                dist[prev] = dist[n] + 1
                q.append(prev)
    return dist


def _count(seq, item):
    # recency multiplicity without importing Counter for one lookup
    return sum(1 for x in seq if x == item)


def weigh(node, out_edges, all_edges, *, goal=None, mood=None, band=None, route="wander",
          exclude=None, prev_node=None, last_id=None, recent_clips=(), recent_nodes=(),
          reverse_of=None, distmap=None, context_energy=None, styles=None,
          style_strength=1.0, dwell=0):
    """Return [(edge, weight)] for every candidate (transparent -> the tests assert on it).
    `reverse_of(edge) -> bool` optionally marks an edge as the reverse of the last clip
    (label heuristic by default). Excluded edges get weight 0 unless they're the only exits.
    `context_energy` ("high"|"low"|None) is the live weather/time tilt layered on the mood;
    None leaves the weights byte-identical to the pre-weather behaviour.
    `styles` is `edge_style.index(all_edges)` -- {edge_id: Style} built ONCE at graph load,
    never per tick. Omitted (or an edge missing from it, or a mood naming no band) leaves
    the weights byte-identical to the pre-style behaviour; present, it is a bounded
    multiplier that can lean the pick toward manner-consistent movement but never zero an
    edge. `style_strength` 0..1 fades the whole layer out (0 == off)."""
    exclude = exclude or set()
    tmul, imul, nov_exp = _mood_bias(mood, band)
    # escape velocity: exits get worth more the longer we have honestly been here.
    if dwell and dwell > ESCAPE_AFTER:
        tmul *= min(ESCAPE_MAX, 1.0 + ESCAPE_PER_UNIT * (dwell - ESCAPE_AFTER))
    # manner preference is OFF unless the caller supplies an INDEX (a mapping, per
    # edge_style.index) AND the mood names a band. Anything else is silently no-styling
    # rather than a traceback out of the render loop.
    sband = (_style_band(mood, band)
             if (styles and _edge_style is not None and hasattr(styles, "get")) else None)
    # weather/time leans the transition/idle balance on TOP of the mood (None -> no change)
    if context_energy:
        ctmul, cimul = _CONTEXT_NUDGE.get(context_energy, (1.0, 1.0))
        tmul *= ctmul
        imul *= cimul
    if distmap is None and goal:
        distmap = dist_to_goal(all_edges, goal)
    here = (distmap or {}).get(node)
    pull = ROUTE_PULL.get(route, ROUTE_PULL["wander"])

    weights = []
    for e in out_edges:
        label = e.get("label")
        if label in exclude:
            weights.append((e, 0.0))
            continue
        w = 1.0
        kind = e.get("kind")
        eid = e.get("id")
        # clip-level novelty (both kinds) + hard anti-exact-repeat
        w *= (1.0 / (1.0 + _count(recent_clips, eid))) ** nov_exp
        if eid == last_id:
            w *= LAST_CLIP_PENALTY
        if kind == "transition":
            to = e.get("to")
            w *= tmul
            # pose-level novelty: avoid bouncing onto a recently-seen pose
            w *= (1.0 / (1.0 + _count(recent_nodes, to))) ** nov_exp
            # anti-reverse: straight back where we came from, or the reverse of the last clip.
            # FORGIVEN WITH DWELL. Going back immediately is a pendulum; going back after an
            # hour of standing still is the only door. On a PENDANT pose -- one in-edge, one
            # out-edge, the same neighbour both ways -- the two guards fight and the trap
            # wins: measured live 2026-08-10, Phineas at commanding_aether (in from sleep,
            # out to sleep) had escape velocity lift his exit to 45%, and anti-reverse cut it
            # straight back to 3.2%, which is 31 expected clips of the same three idles.
            if (prev_node is not None and to == prev_node) or (reverse_of and reverse_of(e)):
                w *= _reverse_penalty(dwell)
            # goal gradient (shares the exclude mask -> never routes through a masked pose)
            if goal and distmap:
                if to == goal:
                    w *= GOAL_ARRIVE_BOOST
                elif here is not None:
                    there = distmap.get(to)
                    if there is None:
                        w *= AWAY_PENALTY            # leaves the goal's reachable set
                    elif there < here:
                        w *= pull                    # toward the goal
                    elif there > here:
                        w *= AWAY_PENALTY            # away from the goal
        else:  # idle
            w *= imul
            if goal and node == goal:
                w *= GOAL_HOLD_BOOST                 # pin at the goal
        # manner preference LAST: it leans a choice the load-bearing terms above have
        # already shaped, and edge_style.affinity is floored above zero, so no styled
        # edge is ever eliminated and the anti-reverse / novelty / goal fixes still win.
        if sband is not None:
            w *= _edge_style.affinity(styles.get(eid), sband, style_strength)
        weights.append((e, w))

    # never strand: if everything masked/zeroed, fall back to a flat pick over raw exits
    if not any(w > 0 for _, w in weights) and out_edges:
        weights = [(e, 1.0) for e in out_edges]
    return weights


def _reverse_of_last(last_label):
    """Heuristic: edges are labelled like 'a2g' (anchor->glower) and 'g2a' (the reverse).
    Mark an edge as the reverse of the last clip if its label is the last label with the
    two sides of '2' swapped. Cheap, no graph metadata needed."""
    if not last_label or "2" not in last_label:
        return lambda e: False
    a, _, b = last_label.partition("2")
    target = b + "2" + a
    return lambda e: e.get("label") == target


def choose(node, out_edges, all_edges, *, goal=None, mood=None, band=None, route="wander",
           exclude=None, prev_node=None, last_id=None, last_label=None,
           recent_clips=(), recent_nodes=(), rng, distmap=None, context_energy=None,
           styles=None, style_strength=1.0, dwell=0):
    """Weighted-sample one edge from `out_edges`. Returns the chosen edge, or None only
    if there are no candidates at all. `context_energy` ("high"|"low"|None) is the live
    weather/time tilt layered on the mood; None = exactly the pre-weather behaviour.
    `styles` (edge_style.index) turns on the manner preference; None = exactly the
    pre-style behaviour."""
    if not out_edges:
        return None
    weights = weigh(node, out_edges, all_edges, goal=goal, mood=mood, band=band, route=route,
                    exclude=exclude, prev_node=prev_node, last_id=last_id,
                    recent_clips=recent_clips, recent_nodes=recent_nodes,
                    reverse_of=_reverse_of_last(last_label), distmap=distmap,
                    context_energy=context_energy, styles=styles,
                    style_strength=style_strength, dwell=dwell)
    total = math.fsum(w for _, w in weights)
    if total <= 0:
        return rng.choice(out_edges)
    r = rng.random() * total
    upto = 0.0
    for e, w in weights:
        upto += w
        if r <= upto:
            return e
    return weights[-1][0]
