"""circadian.py -- time-of-day layer over the video-graph random walk.

The walker (_preview_graph.py) random-walks a character's poses by day. This module
makes NIGHT special: when the wall-clock crosses a character's bedtime window it stops
wandering and instead follows the scripted BEDTIME ROUTINE (prompts/bedtime_routine.json)
-- hub -> (nightcap) -> exit offscreen -> wait -> return in nightclothes -> prep -> sleep --
dwelling a few idle loops at each beat, then DWELLS at the sleep pose until wake_hour, when
it walks the chain in reverse (the free morning wake-up) and resumes the daytime walk.

Pure + import-safe (stdlib only). One entry point the walker calls per pick:

    decide(spec, character, node, out_edges, pose_dwell, last_id, hour, rng) -> dict
      {"action": "force",  "edge": <edge>}            play exactly this edge (routine step)
      {"action": "normal", "exclude": {labels...}}    caller runs its normal random pick,
                                                       skipping these (bedtime) edges
      {"action": "none"}                              no routine for this char -> fully normal

Graceful degrade: if a character's routine isn't generated yet (its bedtime edges aren't in
the graph), decide falls back to "normal" so the character simply stays awake -- it never
strands the walk.
"""
from __future__ import annotations


def in_window(hour, bedtime_hour, wake_hour):
    """Is `hour` within [bedtime, wake), wrapping past midnight? e.g. 21->5 means
    21,22,23,0,1,2,3,4 are night; 0->5 means 0,1,2,3,4 are night."""
    hour %= 24
    if bedtime_hour == wake_hour:
        return False
    if bedtime_hour < wake_hour:
        return bedtime_hour <= hour < wake_hour
    return hour >= bedtime_hour or hour < wake_hour   # wraps midnight


def _routine(spec, character):
    return (spec or {}).get("characters", {}).get(character)


def is_night(spec, character, hour):
    cs = _routine(spec, character)
    if not cs:
        return False
    c = cs.get("circadian", {})
    nw = spec.get("night_window", {})
    bt = c.get("bedtime_hour", nw.get("start_hour", 21))
    wk = c.get("wake_hour", nw.get("end_hour", 6))
    return in_window(hour, bt, wk)


def _chain(cs):
    """From the routine, build the linear transition chain + per-pose idle/dwell info.
    Returns (fwd, back, idles_at, dwell_at, sleep_pose, labels):
      fwd[pose]      = forward transition LABEL leaving `pose` toward sleep
      back[pose]     = reverse transition LABEL leaving `pose` toward the hub
      idles_at[pose] = [idle ids declared for the beat at `pose`]
      dwell_at[pose] = idle loops to dwell before advancing (None == until wake, i.e. sleep)
      sleep_pose     = terminal pose to dwell at overnight
      labels         = every bedtime label/id (to EXCLUDE from the daytime walk)
    """
    fwd, back, idles_at, dwell_at, labels = {}, {}, {}, {}, set()
    sleep_pose = None
    for beat in cs.get("routine", []):
        if beat.get("kind") == "transition":
            fwd[beat["from"]] = beat["label"]
            back[beat["to"]] = beat["reverse_label"]
            labels.add(beat["label"])
            labels.add(beat["reverse_label"])
            sleep_pose = beat["to"]            # last transition's destination = the sleep pose
        else:
            at = beat["at"]
            ids = [i["id"] for i in beat.get("idles", [])]
            idles_at[at] = ids
            labels.update(ids)
            d = beat.get("dwell", beat.get("dwell_loops", 2))
            dwell_at[at] = None if d == "until_wake" else int(d)
    return fwd, back, idles_at, dwell_at, sleep_pose, labels


def bedtime_labels(spec, character):
    cs = _routine(spec, character)
    return _chain(cs)[5] if cs else set()


def bedtime_poses(spec, character):
    """Full node-ids of the poses that belong to the bedtime routine, so the DAYTIME
    brain (heartbeat) can keep them off its goal menu (you don't decide to go to bed at
    noon -- the clock decides that). Excludes the hub, which is a normal daytime pose."""
    cs = _routine(spec, character)
    if not cs:
        return set()
    _, _, idles_at, dwell_at, _, _ = _chain(cs)
    poses = (set(idles_at) | set(dwell_at))
    poses.discard(cs.get("hub_pose"))
    return {character + ":" + p for p in poses}


def sleep_node(spec, character):
    """The '<char>:<sleep_pose>' the routine dwells at overnight (e.g. phineas:sleep,
    maxx:pod), or None. The walker uses this to give sleep idles a longer hold."""
    cs = _routine(spec, character)
    if not cs:
        return None
    sp = _chain(cs)[4]
    return (character + ":" + sp) if sp else None


def _find(out_edges, label):
    """First edge whose `label` matches (build() sets label=<label> on every edge).
    With variants there may be several; pick one at random-ish (the first) -- the caller's
    rng handles variety across picks since pose_dwell advances."""
    if not label:
        return None
    for e in out_edges:
        if e.get("label") == label:
            return e
    return None


def _pick_idle(out_edges, ids, last_id, rng):
    """A random idle edge whose label is in `ids`, avoiding an immediate repeat."""
    pool = [e for e in out_edges if e.get("kind") == "idle" and e.get("label") in ids]
    fresh = [e for e in pool if e.get("id") != last_id] or pool
    return rng.choice(fresh) if fresh else None


def decide(spec, character, node, out_edges, pose_dwell, last_id, hour, rng):
    cs = _routine(spec, character)
    if not cs:
        return {"action": "none"}
    fwd, back, idles_at, dwell_at, sleep_pose, labels = _chain(cs)
    pose = node.split(":", 1)[1] if ":" in node else node
    night = is_night(spec, character, hour)

    if night:
        if pose == sleep_pose:
            # dwell overnight: only the sleep-pose idles (breathe / turn / mutter), no exit
            e = _pick_idle(out_edges, idles_at.get(pose, []), last_id, rng)
            if e:
                return {"action": "force", "edge": e}
            return {"action": "normal", "exclude": set()}        # sleep idles missing -> degrade
        # an intermediate beat (hub / empty / nightclothes): dwell a few loops, then advance
        want = dwell_at.get(pose, 2)
        fwd_edge = _find(out_edges, fwd.get(pose, ""))
        if fwd_edge is not None and (want is not None and pose_dwell >= want):
            return {"action": "force", "edge": fwd_edge}          # advance toward sleep
        e = _pick_idle(out_edges, idles_at.get(pose, []), last_id, rng)
        if e is not None:
            return {"action": "force", "edge": e}                 # play this beat's idle
        if fwd_edge is not None:
            return {"action": "force", "edge": fwd_edge}          # no idle here -> just advance
        return {"action": "normal", "exclude": set()}             # routine not generated -> stay awake

    # DAY: if we're mid-routine (a bedtime pose), walk back toward the hub (wake up)
    if pose in labels_poses(idles_at, dwell_at) and pose != cs.get("hub_pose"):
        back_edge = _find(out_edges, back.get(pose, ""))
        if back_edge is not None:
            return {"action": "force", "edge": back_edge}
    # otherwise: normal daytime walk, but never wander INTO the bedtime chain
    return {"action": "normal", "exclude": labels}


def labels_poses(idles_at, dwell_at):
    """The set of poses that belong to the routine (have an idle/dwell beat)."""
    return set(idles_at) | set(dwell_at)
