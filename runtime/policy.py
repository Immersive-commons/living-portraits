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

Pure + import-safe (stdlib only). Imported by the 10fps render loop, so it must stay
fast and side-effect free. Never strands: if every candidate masks out, it falls back to
a uniform pick over the raw out-edges, so a panel can never freeze.

    from runtime import policy
    edge = policy.choose(node, out_edges, all_edges, goal="maxx:cat_whisper",
                         mood="restless", route="wander", exclude=bedtime_labels,
                         prev_node=prev, last_id=last, recent_clips=rc, recent_nodes=rn,
                         rng=random)
"""
from __future__ import annotations

import math
from collections import deque

# --- tunables (kept as module constants so the tests pin the behaviour, not magic numbers).
REVERSE_PENALTY = 0.04     # multiply a transition that backtracks to prev_node / reverses last clip
LAST_CLIP_PENALTY = 0.02   # never replay the exact clip just shown
GOAL_ARRIVE_BOOST = 8.0    # a transition that lands ON the goal pose
AWAY_PENALTY = 0.15        # a transition that increases BFS distance to the goal
GOAL_HOLD_BOOST = 6.0      # idles AT the goal pose (pin there until a new goal)
ROUTE_PULL = {"beeline": 7.0, "wander": 2.0}   # toward-goal multiplier by route style

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


def _mood_bias(mood):
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


def weigh(node, out_edges, all_edges, *, goal=None, mood=None, route="wander",
          exclude=None, prev_node=None, last_id=None, recent_clips=(), recent_nodes=(),
          reverse_of=None, distmap=None, context_energy=None):
    """Return [(edge, weight)] for every candidate (transparent -> the tests assert on it).
    `reverse_of(edge) -> bool` optionally marks an edge as the reverse of the last clip
    (label heuristic by default). Excluded edges get weight 0 unless they're the only exits.
    `context_energy` ("high"|"low"|None) is the live weather/time tilt layered on the mood;
    None leaves the weights byte-identical to the pre-weather behaviour."""
    exclude = exclude or set()
    tmul, imul, nov_exp = _mood_bias(mood)
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
            # anti-reverse: straight back where we came from, or the reverse of the last clip
            if (prev_node is not None and to == prev_node) or (reverse_of and reverse_of(e)):
                w *= REVERSE_PENALTY
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


def choose(node, out_edges, all_edges, *, goal=None, mood=None, route="wander",
           exclude=None, prev_node=None, last_id=None, last_label=None,
           recent_clips=(), recent_nodes=(), rng, distmap=None, context_energy=None):
    """Weighted-sample one edge from `out_edges`. Returns the chosen edge, or None only
    if there are no candidates at all. `context_energy` ("high"|"low"|None) is the live
    weather/time tilt layered on the mood; None = exactly the pre-weather behaviour."""
    if not out_edges:
        return None
    weights = weigh(node, out_edges, all_edges, goal=goal, mood=mood, route=route,
                    exclude=exclude, prev_node=prev_node, last_id=last_id,
                    recent_clips=recent_clips, recent_nodes=recent_nodes,
                    reverse_of=_reverse_of_last(last_label), distmap=distmap,
                    context_energy=context_energy)
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
