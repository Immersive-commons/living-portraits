"""video_graph.py -- the living-portraits VIDEO KNOWLEDGE GRAPH (v3, multi-node).

A graph of FRAMES (nodes) and VIDEO CLIPS (edges), tied by start/end frame, with the
generation PROMPTS carried as semantic metadata so the whole thing is agent-queryable:

  * node = a frame/POSE of a character, id "<char>:<pose>" (e.g. phineas:anchor,
           phineas:standing). Carries the still IMAGE + its `gen_prompt`.
  * edge = a video CLIP from a start-frame node to an end-frame node. Carries the
           panel `gif` + its `motion_prompt` + `kind`:
             - kind="idle"        : self-loop on a pose (MJ `--end loop`). Interchangeable;
                                    the player traverses them at random for varied idle.
             - kind="transition"  : pose A -> pose B (MJ `--end <frame_url>`). A *stand-up*
                                    is anchor->standing; the *sit-down* is its reverse,
                                    standing->anchor, closing the cycle pixel-perfectly.

Phineas now has TWO poses (anchor seated + standing) joined by stand-up/sit-down
transitions; the player random-walks the graph (idle a while, transition, idle a while,
transition back). Seraphina is still single-pose (anchor) until she gets her own standing set.

Variants are DISCOVERED by globbing data/clips/_proto/<char>_<label>_v*.gif, so derived
clips (the reversed sit-down, which has no MJ job) are first-class. Stored at
data/clips/video_graph.json. Import-safe (stdlib only).
    python runtime/video_graph.py build
    python runtime/video_graph.py show
    python runtime/video_graph.py provenance

PROVENANCE (added 0.3.2). Every node and edge carries `created` -- the moment the SYSTEM
learned it exists, derived from its artifact's mtime, so it survives `build()` overwriting
this file every ~20 minutes. And an edge whose clip disappears is no longer SKIPPED into
oblivion: `build()` diffs against the last committed graph and tombstones what vanished, so
the graph can still answer "what could this character do last month". The tombstones live in
data/graph/provenance.json, NOT in video_graph.json -- that file stays a pure LIVE view
because `_preview_graph.py` reads it raw, and a dead edge in it is a dead clip handed to the
walker. `VideoGraph.load()` therefore defaults to the live view; `load(history=True)` is the
explicit opt-in that reassembles the dead. See runtime/graph_provenance.py.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "clips" / "video_graph.json"
CHARS = ROOT / "prompts" / "characters"
PROTO = ROOT / "data" / "clips" / "_proto"
_VAR_RE = re.compile(r"_v(\d+)\.gif$")


def _load_provenance_module():
    """This module is imported BOTH as `from runtime import video_graph` (repo root on the
    path) and run as `python runtime/video_graph.py build` (runtime/ on the path -- see
    pipeline/autogen.py:482), so the sibling import has to work either way. If neither
    works the graph still loads, simply without a time dimension: provenance is never
    allowed to be the reason a panel goes dark."""
    try:
        from runtime import graph_provenance as m
        return m
    except Exception:
        pass
    try:
        import graph_provenance as m
        return m
    except Exception:
        return None


_prov = _load_provenance_module()


_DERIVED_KEYS = ("created", "invalidated")


def _committed(d):
    """What actually goes on disk: the record minus its DERIVED provenance fields.

    `created` is recomputed from artifact mtime on every load and `invalidated` belongs to
    the tombstone store, so writing either into video_graph.json would persist a value that
    the next rebuild is free to contradict -- and would put a dead edge in front of the raw-
    JSON reader in _preview_graph.py. Stripping here means a load->save round trip cannot
    leak provenance into the file no matter which view it went through."""
    if not isinstance(d, dict) or not any(k in d for k in _DERIVED_KEYS):
        return d
    return {k: v for k, v in d.items() if k not in _DERIVED_KEYS}


class VideoGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes: dict = nodes or {}
        self.edges: list = edges or []
        self.provenance: dict = {}    # {nodes, nodes_created, edges, edges_created} from the last stamp
        self.history: bool = False    # True only if this view was loaded WITH the dead

    @classmethod
    def load(cls, path=None, *, history=False, provenance=True, store_path=None):
        """The LIVE graph by default -- invalidated edges are not in it and cannot be walked.

        `history=True` is the explicit opt-in that merges the tombstones back in; every
        entry that came from history carries `invalidated`, so even a history view can be
        filtered back down (`g.live_edges()`). `provenance=False` skips the mtime scan for
        a caller that wants raw speed (measured 1.7 ms over the 259/1472 production graph).

        The signature is additive and keyword-only past `path`, so every existing caller --
        `VideoGraph.load()` in heartbeat.py and autogen.py -- behaves exactly as before."""
        p = Path(GRAPH_PATH if path is None else path)
        if not p.exists():
            g = cls()
        else:
            d = json.loads(p.read_text(encoding="utf-8"))
            g = cls(d.get("nodes", {}), d.get("edges", []))
        if history and _prov is not None:
            # `merged` is what makes g.history honest. The old shape set it True
            # unconditionally after a merge_history() that swallows its own failure
            # and hands back the LIVE graph -- so a failed merge produced a graph
            # that reported having history and had none. History is a nice-to-have;
            # a graph that lies about having it is not.
            merged = {"ok": True}

            def _failed(why, _m=merged):
                _m["ok"] = False
                print("  provenance: history unavailable -- %s" % why, flush=True)

            try:
                store = _prov.load_store(store_path)
                g.nodes, g.edges = _prov.merge_history(g.nodes, g.edges, store,
                                                       on_error=_failed)
                g.history = merged["ok"]
            except Exception as e:
                print("  provenance: history unavailable -- %r" % (e,), flush=True)
        if provenance and _prov is not None:
            try:
                g.provenance = _prov.stamp(g.nodes, g.edges, root=ROOT)
            except Exception as e:
                # Stamping is decoration; the graph is not. But a stamp that never
                # lands means every downstream provenance view is silently empty.
                print("  provenance: stamp failed -- %r" % (e,), flush=True)
        return g

    def save(self, path=None):
        p = Path(GRAPH_PATH if path is None else path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(
            {"schema": "living-portrait.video-graph/v3",
             "nodes": {k: _committed(v) for k, v in self.nodes.items()},
             "edges": [_committed(e) for e in self.edges]},
            indent=2), encoding="utf-8")
        tmp.replace(p)   # atomic: a hot-reloading player never reads a half-written graph

    def live_edges(self):
        """Edges that are actually playable. Identical to `self.edges` in the default view;
        the point is that it stays correct in a history view too, so no caller can reach a
        dead clip by holding the wrong object."""
        return [e for e in self.edges if not e.get("invalidated")]

    def dead_edges(self):
        return [e for e in self.edges if e.get("invalidated")]

    def add_node(self, node_id, **attrs):
        self.nodes[node_id] = attrs

    def add_edge(self, **edge):
        self.edges.append(edge)

    def edges_from(self, node_id):
        """Outgoing PLAYABLE edges. The `invalidated` filter is the single choke point every
        walk helper below funnels through (idle/transition/random/next_edge, reachability,
        validate), so a history view cannot hand anyone a clip that is no longer on disk.
        In the default live view nothing is invalidated and this is the original one-liner."""
        return [e for e in self.edges
                if e.get("from") == node_id and not e.get("invalidated")]

    def idle_edges(self, node_id):
        return [e for e in self.edges_from(node_id) if e.get("kind") == "idle"]

    def transition_edges(self, node_id):
        return [e for e in self.edges_from(node_id) if e.get("kind") == "transition"]

    def random_edge(self, node_id, rng=None, exclude_id=None):
        cands = self.edges_from(node_id)
        if not cands:
            return None
        pool = [e for e in cands if e.get("id") != exclude_id] or cands
        return (rng or random).choice(pool)

    def next_edge(self, node_id, rng=None, exclude_id=None, transition_prob=0.18):
        """Walk policy: usually pick a fresh IDLE on this pose; occasionally take a
        TRANSITION to another pose. Returns an edge (its `to` is the next node)."""
        rng = rng or random
        idles = [e for e in self.idle_edges(node_id) if e.get("id") != exclude_id]
        trans = self.transition_edges(node_id)
        if trans and (not idles or rng.random() < transition_prob):
            return rng.choice(trans)
        if idles:
            return rng.choice(idles)
        return self.random_edge(node_id, rng=rng, exclude_id=exclude_id)

    @staticmethod
    def anchor(character):
        return "%s:anchor" % character

    def poses(self, character):
        return sorted(n for n in self.nodes if n.startswith(character + ":"))

    # --- walk-safety: guarantee the player's random walk always works + follows the graph ---
    def _reachable_from(self, start):
        seen, stack = {start}, [start]
        while stack:
            for e in self.edges_from(stack.pop()):
                t = e.get("to")
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return seen

    def validate(self):  # noqa: PLR0912  -- walk-safety is a checklist, and each branch is one way a walker can be stranded. Collapsing them would make a refusal harder to explain.
        """Return [(level, msg)]. The walk is random + graph-following BY CONSTRUCTION
        (uniform random choice over a node's out-edges, advance to edge.to). These invariants
        guarantee it STAYS that way no matter what states/loops are added; build() raises on
        any ERROR, so a state that would break the random graph-walk fails loudly:
          E1 every edge endpoint is a real node      -> walk never follows a non-existent edge
          E2 every node has >=1 outgoing edge          -> walk never dead-ends / strands
          E3 in a multi-pose character, every pose     -> walk can always LEAVE a state, so it
             has >=1 transition OUT                       keeps roaming (never trapped in one)
          W1 every pose reachable from its char's hub  -> it actually gets visited
          W2 every pose can reach back to the hub      -> not a one-way trap
          W3 every edge's gif exists                   -> no missing-clip stall
        """
        issues, node_ids = [], set(self.nodes)
        char_nodes = {}
        live = self.live_edges()      # the dead are history, not walk-safety problems
        for nid, n in self.nodes.items():
            char_nodes.setdefault(n.get("character"), []).append(nid)
        for e in live:
            for end in ("from", "to"):
                if e.get(end) not in node_ids:
                    issues.append(("ERROR", "edge %s .%s points at unknown node %r" % (e.get("id"), end, e.get(end))))
        for nid in self.nodes:
            outs = self.edges_from(nid)
            if not outs:
                issues.append(("ERROR", "node %s has NO outgoing edges -> the walk would strand here" % nid))
            multi = len(char_nodes.get(self.nodes[nid].get("character"), [])) > 1
            if multi and not any(e.get("kind") == "transition" for e in outs):
                issues.append(("ERROR", "node %s has NO transition OUT -> the walk gets trapped in this state" % nid))
        for _char, ids in char_nodes.items():
            starts = [i for i in ids if self.edges_from(i)]
            if not starts:
                continue
            hub = starts[0]
            reach = self._reachable_from(hub)
            for i in ids:
                if i not in reach:
                    issues.append(("WARN", "node %s is unreachable from hub %s -> never visited" % (i, hub)))
                elif hub not in self._reachable_from(i):
                    issues.append(("WARN", "node %s cannot reach back to hub %s -> one-way trap" % (i, hub)))
        for e in live:
            gif = e.get("gif")
            if gif and not (ROOT / gif).exists():
                issues.append(("WARN", "edge %s gif missing: %s" % (e.get("id"), gif)))
        return issues


# --------------------------------------------------------------------------- specs
_BREATHE = "the figure breathes slowly and naturally, chest and shoulders gently rising and falling, a faint sway, an occasional slow blink, otherwise still"
_LOOK = "the figure slowly turns their head to glance to one side and then the other, eyes calmly scanning the room, then settles back to face forward"
_DEFAULT = "Subtle motion, idle animation of the character in default position."
_MOVE_BEHIND = "the seated figure rises from his chair, steps out beside it, and stands at his full height beside the chair facing forward"
_MOVE_BACK = "(reverse of move-behind) the standing figure steps back to his chair and sits down, ending exactly on the seated anchor frame"

# behind_chair node gen prompt (semantic metadata). The node IMAGE is the ACTUAL
# landing frame of the move-behind transition (where the rise ends), not a separate
# still -- so idles loop on it and the transitions splice seamlessly. Wide enough that
# the figure is NOT cropped (the bug that killed the earlier phineas:standing pose).
_BEHIND_CHAIR_GEN = (
    "WIDE full-length oil painting, Dutch Golden Age Rembrandt manner, the oxblood velvet "
    "armchair in the foreground with the 62 year old tragedian standing at full height beside "
    "it, deep chiaroscuro, warm-black void, no picture frame -- the realized end frame of the "
    "seated->standing move-behind transition (identity carried from phineas:anchor)"
)

# MAXX-9 cyberpunk state-node gen prompts (semantic metadata). idle reads maxx.json
# positive_template; weapon/mech are the picked state stills.
_MAXX_WEAPON_GEN = (
    "cyberpunk neon-noir render, MAXX-9 in a combat firing stance, his oversized cybernetic arm "
    "reconfigured into a glowing arm-cannon with targeting reticles, rain-slick neon street with a "
    "hover-bus behind -- the 'capability mode' state still"
)
_MAXX_MECH_GEN = (
    "cyberpunk neon-noir render, MAXX-9 jumped back inside a towering exo-frame battle-rig that "
    "dwarfs him with deployed wings/plates/thrusters filling the frame, neon street with a hover-bus "
    "behind -- the 'big cyborg thing takes more space' state still"
)
_PH_GLOWER_GEN = "oil painting Rembrandt manner, the faded tragedian leaning to the frame's edge glaring resentfully off-frame at the brighter frame next door, dismissive hand -- the 'glower at the rival' state"
_PH_SWOON_GEN = "oil painting Rembrandt manner, the faded tragedian in a melodramatic Act-V swoon, hand to forehead, collapsed backward in tragic despair -- the 'swoon' state"
_MX_GLITCH_GEN = "cyberpunk neon-noir, MAXX-9 mid digital glitch, body fragmenting into datamosh shards and RGB error blocks -- the 'glitch / malfunction' state"
_MX_HACK_GEN = "cyberpunk neon-noir, MAXX-9 kneeling projecting a holographic console + keyboard, code streaming, smug -- the 'hack' state"
_MX_FLEX_GEN = "cyberpunk neon-noir, MAXX-9 in a cocky victory flex with neon trails and lens flares -- the 'flex / victory' state"

# node_id pose specs: gen_prompt None => read the character JSON positive_template.
NODE_SPECS = {
    "phineas": {
        "anchor":       {"image": "data/gen/phineas_anchor.png",       "gen_prompt": None},
        "behind_chair": {"image": "data/gen/phineas_behind_chair.png", "gen_prompt": _BEHIND_CHAIR_GEN},
        "glower":       {"image": "data/gen/phineas_glower.png",       "gen_prompt": _PH_GLOWER_GEN},
        "swoon":        {"image": "data/gen/phineas_swoon.png",        "gen_prompt": _PH_SWOON_GEN},
    },
    "seraphina": {
        "anchor":   {"image": "data/gen/seraphina_anchor.png", "gen_prompt": None},
    },
    "maxx": {
        "idle":   {"image": "data/gen/maxx_idle.png",   "gen_prompt": None},
        "weapon": {"image": "data/gen/maxx_weapon.png", "gen_prompt": _MAXX_WEAPON_GEN},
        "mech":   {"image": "data/gen/maxx_mech.png",   "gen_prompt": _MAXX_MECH_GEN},
        "glitch": {"image": "data/gen/maxx_glitch.png", "gen_prompt": _MX_GLITCH_GEN},
        "hack":   {"image": "data/gen/maxx_hack.png",   "gen_prompt": _MX_HACK_GEN},
        "flex":   {"image": "data/gen/maxx_flex.png",   "gen_prompt": _MX_FLEX_GEN},
    },
}

# (character, kind, label, from_pose, to_pose, motion_prompt, loop, mj_job)
# variants are globbed from data/clips/_proto/<char>_<label>_v*.gif
EDGE_SPECS = [
    # --- phineas: seated-anchor idles (self-loops) ---
    ("phineas", "idle", "breathe",       "anchor", "anchor", _BREATHE, True, "a01afceb-c4da-419d-8806-1613702ec3cf"),
    ("phineas", "idle", "look_around",   "anchor", "anchor", _LOOK,    True, "475f404d-1a2b-43e5-99e1-28bd9124fcc2"),
    ("phineas", "idle", "settle",        "anchor", "anchor", "the figure shifts their weight and settles, lifting one hand to adjust their collar then lowering it again, a small comfortable readjustment, then stillness", True, "0712df74-1ae7-46da-ace6-a6c7e4bc1b99"),
    ("phineas", "idle", "ponder",        "anchor", "anchor", "the figure tilts their head slightly as if in quiet thought, a slow blink and a faint nod, brow furrowing then easing, a subtle breath", True, "3bd6d284-a949-4cf1-83aa-49792de32354"),
    ("phineas", "idle", "default",       "anchor", "anchor", _DEFAULT, True, "f8929ecd-1a08-4305-b93d-f72a919c213c"),
    ("phineas", "idle", "wait",          "anchor", "anchor", "NPC idle wait", True, "0babb44e-69ad-4a2b-ac78-b877467349db"),
    ("phineas", "idle", "microgestures", "anchor", "anchor", "Subtle, NPC idle animation, the character is looking straight into the camera, with microgestures", True, "9c940b60-e7a3-4437-9ede-34eee6197c5d"),
    # --- phineas: TRANSITIONS (seated anchor <-> standing beside the chair) ---
    ("phineas", "transition", "move_behind", "anchor", "behind_chair", _MOVE_BEHIND, False, "33ca3591-b1b6-4fa8-b677-f3bab09d52d0"),
    ("phineas", "transition", "move_back",   "behind_chair", "anchor", _MOVE_BACK, False, ""),
    # --- phineas: BEHIND-CHAIR idles (self-loops on the standing-by-chair pose) ---
    ("phineas", "idle", "bc_declaim",    "behind_chair", "behind_chair", "lifts both arms in a grand theatrical declamation toward an unseen upper balcony, his oxblood robe shifting, then lowers his arms to rest beside the chair", True, ""),
    ("phineas", "idle", "bc_grip_chair", "behind_chair", "behind_chair", "grips the carved chair back with both hands and leans forward over it as if addressing the room, then straightens back up to his full height", True, ""),
    ("phineas", "idle", "bc_survey",     "behind_chair", "behind_chair", "turns his head and shoulders slowly to survey the gallery to one side then the other with cool contempt, then faces forward again", True, ""),
    ("phineas", "idle", "bc_gesture",    "behind_chair", "behind_chair", "sweeps one hand outward in a slow theatrical flourish then brings it back, the other resting near the chair, settling to stillness", True, ""),
    ("phineas", "idle", "bc_settle",     "behind_chair", "behind_chair", "draws himself up to his full grand height beside the chair with a slow breath and a faint regal sway, otherwise standing still facing forward", True, ""),
    # --- seraphina: anchor idles (single-pose for now) ---
    ("seraphina", "idle", "breathe",     "anchor", "anchor", _BREATHE, True, "c4addf08-ccd1-4679-a8f3-ccfd39771561"),
    ("seraphina", "idle", "look_around", "anchor", "anchor", _LOOK,    True, "949a2ab9-e96e-4fa7-be74-7179e90794ac"),
    ("seraphina", "idle", "default",     "anchor", "anchor", _DEFAULT, True, "3c84a919-d7b7-446d-9811-dcabbe29e9ca"),
    ("seraphina", "idle", "wait_thought","anchor", "anchor", "NPC idle wait, deep in thought, waiting", True, "c48a80bd-dbdc-476c-9306-086e5bf05163"),
    # --- MAXX-9: idle-state loops (home state) ---
    ("maxx", "idle", "idle_whir", "idle", "idle", "MAXX shifts his weight cockily, his cybernetic arm servos whirring and circuit veins pulsing, his visor HUD scanning, a neon hover-bus gliding past behind, then settles", True, ""),
    ("maxx", "idle", "idle_flag", "idle", "idle", "MAXX points a finger-gun at the viewer and a tiny holographic flag pops from his fingertip, he smirks as his arm servos whir, a neon hover-bus gliding past behind", True, ""),
    ("maxx", "idle", "idle_scan", "idle", "idle", "MAXX's cyan visor sweeps a glowing targeting scan, holographic HUD glyphs flickering around his head, his cyber-arm twitching, a neon hover-bus gliding past behind", True, ""),
    # --- MAXX-9: TRANSITIONS (fully connected: idle<->weapon, idle<->mech, weapon<->mech) ---
    ("maxx", "transition", "i2w", "idle",   "weapon", "MAXX snaps into a combat firing stance, his cybernetic arm whirring and reconfiguring into a glowing arm-cannon as targeting reticles flare, a neon hover-bus gliding past behind", False, ""),
    ("maxx", "transition", "w2i", "weapon", "idle",   "(reverse) MAXX's arm-cannon folds back into a hand as he eases out of the firing stance to a cocky idle, a neon hover-bus gliding past behind", False, ""),
    ("maxx", "transition", "i2m", "idle",   "mech",   "MAXX leaps backward as a towering exo-frame rig unfolds around him, thrusters igniting and armor plates expanding to fill the frame, a neon hover-bus gliding past behind", False, ""),
    ("maxx", "transition", "m2i", "mech",   "idle",   "(reverse) MAXX's exo-frame folds away as he drops back down to a cocky idle stance, a neon hover-bus gliding past behind", False, ""),
    ("maxx", "transition", "w2m", "weapon", "mech",   "MAXX's arm-cannon folds away as he leaps back and a massive exo-frame rig deploys around him, thrusters firing, a neon hover-bus gliding past behind", False, ""),
    ("maxx", "transition", "m2w", "mech",   "weapon", "(reverse) MAXX's exo-frame retracts as his arm reconfigures back into the arm-cannon firing stance, a neon hover-bus gliding past behind", False, ""),
    # --- MAXX-9: weapon-state loops ---
    ("maxx", "idle", "wpn_charge", "weapon", "weapon", "MAXX braces and charges his oversized arm-cannon, magenta energy building and targeting reticles sweeping across the glowing barrel, a neon hover-bus gliding past behind", True, ""),
    ("maxx", "idle", "wpn_twirl",  "weapon", "weapon", "MAXX twirls his arm-cannon and blows imaginary smoke off the glowing barrel with a smug smirk, servos whirring, a neon hover-bus gliding past behind", True, ""),
    ("maxx", "idle", "wpn_brace",  "weapon", "weapon", "MAXX holds a firing brace, recoil rippling through his cyber-arm as the cannon pulses and ejects bright shell-casings of light, a neon hover-bus gliding past behind", True, ""),
    # --- MAXX-9: mech-state loops ---
    ("maxx", "idle", "mech_thrust","mech",   "mech",   "MAXX's towering exo-frame flexes, thrusters pulsing and armor plates shifting, energy panels glowing brighter, a neon hover-bus gliding past behind", True, ""),
    ("maxx", "idle", "mech_point", "mech",   "mech",   "MAXX throws a dramatic point forward as his exo-frame rears up, thrusters flaring and HUD panels spinning, a neon hover-bus gliding past behind", True, ""),
    ("maxx", "idle", "mech_vent",  "mech",   "mech",   "MAXX's exo-frame comically overheats, venting jets of steam as warning lights flash and he shrugs, a neon hover-bus gliding past behind", True, ""),
    # --- phineas: glower (glare at the rival next door) + swoon (melodramatic collapse) ---
    ("phineas", "transition", "a2g", "anchor", "glower", "the seated tragedian turns and leans to one side, his face curdling into a resentful glare off-frame at the brighter frame next door, one hand rising in a dismissive flick", False, ""),
    ("phineas", "transition", "g2a", "glower", "anchor", "(reverse) the tragedian unbends from his glare and settles back square into the seated rest pose", False, ""),
    ("phineas", "transition", "a2s", "anchor", "swoon",  "the seated tragedian is overcome, the back of one hand flying to his forehead as he collapses backward into a melodramatic theatrical swoon", False, ""),
    ("phineas", "transition", "s2a", "swoon",  "anchor", "(reverse) the tragedian gathers himself and rises out of the swoon back to the upright seated rest pose", False, ""),
    ("phineas", "idle", "glower_glare", "glower", "glower", "holds his resentful sideways glare, lip curling, a slow contemptuous sniff, then narrows his eyes further", True, ""),
    ("phineas", "idle", "glower_flick", "glower", "glower", "flicks one hand dismissively as if shooing away the brighter frame next door, then folds his arms with a huff", True, ""),
    ("phineas", "idle", "swoon_heave",  "swoon",  "swoon",  "his chest heaves with a tragic sigh, the back of his hand trembling at his forehead, his head lolling deeper into despair", True, ""),
    ("phineas", "idle", "swoon_falter", "swoon",  "swoon",  "he half-lifts his head as if to recover, then sinks back into the melodramatic swoon with a shudder", True, ""),
    # --- MAXX-9: glitch / hack / flex (hub-and-spoke on idle) ---
    ("maxx", "transition", "i2g", "idle", "glitch", "MAXX suddenly glitches, his body fragmenting into datamosh shards and RGB-split error blocks with corrupted glyphs, a neon hover-bus passing behind", False, ""),
    ("maxx", "transition", "g2i", "glitch", "idle", "(reverse) MAXX's glitch resolves and he snaps back together into his cocky idle, a neon hover-bus passing behind", False, ""),
    ("maxx", "transition", "i2h", "idle", "hack", "MAXX drops to one knee and projects a glowing holographic console and keyboard from his cyber-arm, code streaming up, a neon hover-bus passing behind", False, ""),
    ("maxx", "transition", "h2i", "hack", "idle", "(reverse) MAXX dismisses the holo-console and rises back to his cocky idle, a neon hover-bus passing behind", False, ""),
    ("maxx", "transition", "i2f", "idle", "flex", "MAXX bursts into a cocky victory flex, arms pumping with neon energy trails and lens flares, a neon hover-bus passing behind", False, ""),
    ("maxx", "transition", "f2i", "flex", "idle", "(reverse) MAXX eases out of the flex back to his cocky idle, a neon hover-bus passing behind", False, ""),
    ("maxx", "idle", "glitch_flicker", "glitch", "glitch", "his body flickers and fragments into datamosh shards and RGB-split errors, glitch glyphs popping, then snaps back together, a neon hover-bus passing behind", True, ""),
    ("maxx", "idle", "glitch_stutter", "glitch", "glitch", "he stutters and judders with corrupted scanlines and error blocks, his visor strobing, then stabilizes, a neon hover-bus passing behind", True, ""),
    ("maxx", "idle", "hack_type",    "hack", "hack", "his fingers fly across the holographic keyboard, code streaming upward, his visor scanning the data with a smug smirk, a neon hover-bus passing behind", True, ""),
    ("maxx", "idle", "hack_execute", "hack", "hack", "he jabs a finger to execute and the holo-console flares and resolves with a confirmation pulse, he nods smugly, a neon hover-bus passing behind", True, ""),
    ("maxx", "idle", "flex_double",  "flex", "flex", "he pumps both arms into a double victory flex with neon energy trails and lens flares bursting, beaming, a neon hover-bus passing behind", True, ""),
    ("maxx", "idle", "flex_point",   "flex", "flex", "he throws a cocky finger-point and a wink as holographic confetti rains down, supremely pleased, a neon hover-bus passing behind", True, ""),
]


def _gen_prompt(character):
    p = CHARS / (character + ".json")
    try:
        spec = json.loads(p.read_text(encoding="utf-8-sig"))
        return spec.get("generation", {}).get("positive_template", "")
    except Exception:
        return ""


def _variant_gifs(character, label):
    """Discover variant gifs -> [(variant_int, project_relative_posix), ...] sorted."""
    out = []
    for g in PROTO.glob("%s_%s_v*.gif" % (character, label)):
        m = _VAR_RE.search(g.name)
        if m:
            out.append((int(m.group(1)), g.relative_to(ROOT).as_posix()))
    return sorted(out)


# --------------------------------------------------------------------------- bedtime routine
# Night-time bedtime routines are authored in prompts/bedtime_routine.json (the single source
# of truth the generator + this graph share). We DERIVE the extra nodes + edges from it and
# merge them into NODE_SPECS/EDGE_SPECS -- but only for a character whose routine is FULLY
# generated (every declared edge already has >=1 variant gif). So running `build` before the
# clips exist leaves the existing graph untouched instead of stranding orphan poses (which
# would trip the walk-safety ERRORs). The forward transition's REVERSE is its own edge whose
# gif is the forward clip flipped (the generator writes it) -- the free morning wake-up.
BEDTIME_SPEC = ROOT / "prompts" / "bedtime_routine.json"


def _load_bedtime():
    try:
        return json.loads(BEDTIME_SPEC.read_text(encoding="utf-8"))
    except Exception:
        return {"characters": {}}


def _bedtime_additions():
    """(nodes, edges, skipped) derived from the bedtime spec. A character is included only
    if EVERY edge it declares already has >=1 variant gif (atomic: partial -> skip the char)."""
    spec = _load_bedtime()
    nodes, edges, skipped = {}, [], []
    for char, cs in spec.get("characters", {}).items():
        char_edges = []   # (char, kind, label, from, to, motion, loop, job)
        for beat in cs.get("routine", []):
            if beat.get("kind") == "transition":
                m = beat["motion_prompt"]
                char_edges.append((char, "transition", beat["label"], beat["from"], beat["to"], m, False, ""))
                char_edges.append((char, "transition", beat["reverse_label"], beat["to"], beat["from"], "(reverse) " + m, False, ""))
            else:
                for idle in beat.get("idles", []):
                    char_edges.append((char, "idle", idle["id"], beat["at"], beat["at"], idle["motion_prompt"], True, ""))
        ready = bool(char_edges) and all(_variant_gifs(char, lbl) for (_, _, lbl, *_r) in char_edges)
        if not ready:
            if char_edges:
                skipped.append(char)
            continue
        for pose, n in cs.get("nodes", {}).items():
            nodes.setdefault(char, {})[pose] = {"image": n["node_image"], "gen_prompt": n.get("still_prompt")}
        edges.extend(char_edges)
    return nodes, edges, skipped


# --------------------------------------------------------------------------- autogen poses
# LLM-PROPOSED poses the portrait grew FOR ITSELF (Phase 2 autonomy). pipeline/autogen.py
# writes data/mind/autogen_poses.json once a pose's clips are generated. Same DATA-DRIVEN +
# ATOMIC contract as the bedtime routine: a pose merges only when every edge it declares
# already has >=1 variant gif, so a half-generated pose never strands the walk. A new pose
# hangs off a hub (default the character's anchor) by a transition + its free reverse, plus
# idle self-loops -- which satisfies the walk-safety invariants (reachable + can return) by
# construction.
AUTOGEN_SPEC = ROOT / "data" / "mind" / "autogen_poses.json"


def _load_autogen():
    try:
        return json.loads(AUTOGEN_SPEC.read_text(encoding="utf-8"))
    except Exception:
        return {"characters": {}}


def _autogen_additions():
    """(nodes, edges, skipped) derived from data/mind/autogen_poses.json (atomic per pose).

    Two passes per character so the WEB-meshing extra links are safe:
      1. find the poses whose CORE edges (hub transition + idles) all have gifs -- only those
         merge (atomic, exactly as before -- a half-generated pose never strands the walk).
      2. emit the core edges, then APPEND each optional sibling<->new link, but only when the
         sibling is a real node AND both link clips exist on disk. A dangling/half-done link is
         simply dropped, so one bad link can never break the build or trap the walk -- the core
         already guarantees reach + return; extra links only ADD ways across the graph.
    """
    spec = _load_autogen()
    nodes, edges, skipped = {}, [], []
    for char, cs in spec.get("characters", {}).items():
        poses = cs.get("poses", {}) or {}
        # pass 1: which poses are core-ready
        ready = {}
        for pose, p in poses.items():
            hub = p.get("hub", "anchor")
            tr = p.get("transition", {}) or {}
            fwd, rev, motion = tr.get("label"), tr.get("reverse_label"), tr.get("motion", "")
            pose_edges = []
            if fwd and rev:
                pose_edges.append((char, "transition", fwd, hub, pose, motion, False, ""))
                pose_edges.append((char, "transition", rev, pose, hub, "(reverse) " + motion, False, ""))
            for idle in p.get("idles", []) or []:
                pose_edges.append((char, "idle", idle["id"], pose, pose, idle.get("motion", ""), True, ""))
            core_ready = bool(pose_edges) and all(_variant_gifs(char, lbl) for (_, _, lbl, *_r) in pose_edges)
            if not core_ready:
                if pose_edges:
                    skipped.append("%s:%s" % (char, pose))
                continue
            ready[pose] = (p, pose_edges)
        # valid link endpoints = this char's base/bedtime nodes + the autogen poses that merged
        valid = set(NODE_SPECS.get(char, {}).keys()) | set(ready.keys())
        # pass 2: emit core edges + the optional, gif-gated, non-dangling extra links
        for pose, (p, pose_edges) in ready.items():
            nodes.setdefault(char, {})[pose] = {"image": p["node_image"], "gen_prompt": p.get("still_prompt")}
            edges.extend(pose_edges)
            for el in p.get("extra_links", []) or []:
                sib = el.get("sibling")
                lfwd, lrev, lmot = el.get("label"), el.get("reverse_label"), el.get("motion", "")
                if (sib in valid and sib != pose and lfwd and lrev
                        and _variant_gifs(char, lfwd) and _variant_gifs(char, lrev)):
                    edges.append((char, "transition", lfwd, sib, pose, lmot, False, ""))
                    edges.append((char, "transition", lrev, pose, sib, "(reverse) " + lmot, False, ""))
    return nodes, edges, skipped


_BT_NODES, _BT_EDGES, _BT_SKIPPED = _bedtime_additions()
for _ch, _poses in _BT_NODES.items():
    NODE_SPECS.setdefault(_ch, {}).update(_poses)
EDGE_SPECS.extend(_BT_EDGES)

_AG_NODES, _AG_EDGES, _AG_SKIPPED = _autogen_additions()
for _ch, _poses in _AG_NODES.items():
    NODE_SPECS.setdefault(_ch, {}).update(_poses)
EDGE_SPECS.extend(_AG_EDGES)


def _previous_committed():
    """The last graph this build actually committed -- the only honest baseline for "what
    disappeared". Read raw, fail-soft: no previous file (first ever build) means nothing has
    disappeared, which is exactly ({}, [])."""
    try:
        d = json.loads(Path(GRAPH_PATH).read_text(encoding="utf-8"))
        return (d.get("nodes") or {}), (d.get("edges") or [])
    except Exception:
        return {}, []


def _record_provenance(prev_nodes, prev_edges, g):
    """Tombstone whatever the new graph lost, stamp what it has, print one honest line.

    Called ONLY AFTER a successful save, so a build that refuses to commit (walk-safety
    errors) never records deaths for a graph nobody is running. Returns the stats dict for
    the caller/tests; every failure degrades to no provenance rather than no graph."""
    if _prov is None:
        return {}
    try:
        store = _prov.load_store()
        store, stats = _prov.reconcile(prev_nodes, prev_edges, g.nodes, g.edges, store=store)
        wrote = _prov.save_store(store) if (stats["edges_invalidated"] or stats["nodes_invalidated"]
                                            or stats["edges_revived"] or stats["nodes_revived"]
                                            or store["edges"] or store["nodes"]) else True
        g.provenance = _prov.stamp(g.nodes, g.edges, root=ROOT)
        print("  provenance: %d/%d nodes + %d/%d edges dated from their artifact" % (
            g.provenance.get("nodes_created", 0), g.provenance.get("nodes", 0),
            g.provenance.get("edges_created", 0), g.provenance.get("edges", 0)))
        if stats["edges_invalidated"] or stats["nodes_invalidated"]:
            print("  provenance: INVALIDATED %d edge(s) + %d node(s) this build "
                  "(kept as history, not deleted)" % (
                      stats["edges_invalidated"], stats["nodes_invalidated"]))
        if stats["edges_revived"] or stats["nodes_revived"]:
            print("  provenance: revived %d edge(s) + %d node(s)" % (
                stats["edges_revived"], stats["nodes_revived"]))
        if stats["edges_dead"] or stats["nodes_dead"]:
            print("  provenance: %d edge(s) + %d node(s) now live only in history%s" % (
                stats["edges_dead"], stats["nodes_dead"], "" if wrote else " (STORE WRITE FAILED)"))
        if stats.get("error"):
            # reconcile() is fail-open and returns zeroed counters on a crash, so
            # without this the line above reports "0 invalidated" for a step that
            # did not run. See graph_provenance.reconcile.
            print("  provenance: RECONCILE FAILED -- %s (counts above are not real)"
                  % stats["error"], flush=True)
        return stats
    except Exception as e:
        print("  provenance: FAILED -- %r (the build itself is unaffected)" % (e,),
              flush=True)
        return {"error": repr(e)}


def build():
    prev_nodes, prev_edges = _previous_committed()
    g = VideoGraph()
    for char, poses in NODE_SPECS.items():
        for pose, spec in poses.items():
            g.add_node("%s:%s" % (char, pose), character=char, pose=pose,
                       image=spec["image"], gen_prompt=spec["gen_prompt"] or _gen_prompt(char))
    missing = []
    for (char, kind, label, fr, to, motion, loop, job) in EDGE_SPECS:
        from_id, to_id = "%s:%s" % (char, fr), "%s:%s" % (char, to)
        variants = _variant_gifs(char, label)
        if not variants:
            missing.append("%s/%s" % (char, label))
            continue
        for v, gif in variants:
            g.add_edge(
                id="%s/%s/v%d" % (char, label, v),
                character=char, kind=kind, label=label, variant=v,
                **{"from": from_id, "to": to_id},
                start_frame=g.nodes.get(from_id, {}).get("image"),
                end_frame=g.nodes.get(to_id, {}).get("image"),
                gif=gif, motion_prompt=motion, loop=loop, mj_job=job,
                tags=[kind, label] + (["transition"] if kind == "transition" else ["idle"]),
            )
    issues = g.validate()
    errs = [m for lvl, m in issues if lvl == "ERROR"]
    warns = [m for lvl, m in issues if lvl == "WARN"]
    print("video_graph v3: %d nodes, %d edges (walk-safety: %d error, %d warn)" % (
        len(g.nodes), len(g.edges), len(errs), len(warns)))
    for char in NODE_SPECS:
        for node in g.poses(char):
            ie, te = len(g.idle_edges(node)), len(g.transition_edges(node))
            outs = sorted({e["to"] for e in g.transition_edges(node)})
            print("  %-18s %2d idle + %d transition%s" % (node, ie, te, (" -> " + ", ".join(outs)) if outs else ""))
    if missing:
        print("  (no gifs yet for: %s)" % ", ".join(missing))
    if _BT_SKIPPED:
        print("  (bedtime routine not fully generated, char skipped: %s)" % ", ".join(_BT_SKIPPED))
    for m in warns:
        print("  WARN:", m)
    if errs:
        for m in errs:
            print("  ERROR:", m)
        raise SystemExit("GRAPH NOT SAVED -- %d walk-safety error(s). Every pose needs a way out "
                         "(idle loops + at least one transition back toward the hub). Fix + rebuild." % len(errs))
    g.save()
    print("  walk-safe -> saved %s" % GRAPH_PATH)
    _record_provenance(prev_nodes, prev_edges, g)
    return g


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "show":
        g = VideoGraph.load()
        for nid, n in g.nodes.items():
            print("NODE", nid, "| image=" + str(n.get("image")))
        for e in g.edges:
            print("EDGE %-22s %-10s %s -> %s | %s" % (
                e["id"], e.get("kind"), e.get("from"), e.get("to"), (e.get("motion_prompt") or "")[:50]))
    elif cmd == "provenance":
        import datetime as _dt
        g = VideoGraph.load(history=True)
        p = g.provenance
        print("created (transaction time, from artifact mtime): %d/%d nodes, %d/%d edges" % (
            p.get("nodes_created", 0), p.get("nodes", 0),
            p.get("edges_created", 0), p.get("edges", 0)))
        dead_n = [(nid, n) for nid, n in g.nodes.items() if n.get("invalidated")]
        dead_e = g.dead_edges()
        print("invalidated (still in history, never playable): %d nodes, %d edges" % (
            len(dead_n), len(dead_e)))
        for nid, n in sorted(dead_n)[:20]:
            print("  DEAD NODE %-24s since %s" % (
                nid, _dt.datetime.fromtimestamp(n["invalidated"]).strftime("%Y-%m-%d %H:%M")))
        for e in sorted(dead_e, key=lambda x: x.get("id") or "")[:20]:
            print("  DEAD EDGE %-24s %s -> %s since %s" % (
                e.get("id"), e.get("from"), e.get("to"),
                _dt.datetime.fromtimestamp(e["invalidated"]).strftime("%Y-%m-%d %H:%M")))
        if len(dead_e) > 20:
            print("  ... and %d more" % (len(dead_e) - 20))
    elif cmd == "validate":
        g = VideoGraph.load()
        issues = g.validate()
        for lvl, m in issues:
            print(lvl + ":", m)
        n_err = sum(1 for lvl, _ in issues if lvl == "ERROR")
        print("walk-safety: %d error(s), %d warning(s)" % (n_err, len(issues) - n_err))
        sys.exit(1 if n_err else 0)
    else:
        print("usage: video_graph.py build | show | validate | provenance")
