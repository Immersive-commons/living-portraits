"""_preview_graph.py -- random-traversal living-portrait player driven by the VIDEO GRAPH.

Each panel sits on a character's anchor node and plays a RANDOM outgoing idle edge (its gif);
at the loop boundary it picks another random edge (avoiding an immediate repeat) -> endless,
non-repeating, varied idle pulled straight from data/clips/video_graph.json.

    pythonw _preview_graph.py --a phineas --b seraphina [--loops 1]

Borderless topmost at (0,0), session 1, same window pin as player.py. Reads gifs via PIL.
"""
import argparse
import datetime
import json
import os
import random
import sys
import traceback
from collections import deque
from pathlib import Path

os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
os.environ.setdefault("SDL_VIDEO_CENTERED", "0")

ROOT = Path(__file__).resolve().parent           # project root (C:\living-portraits on hil)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.stdout = sys.stderr = open(ROOT / "_preview.log", "a", buffering=1,
                               encoding="utf-8", errors="replace")

import pygame
from PIL import Image, ImageSequence

from runtime import circadian, mind, policy   # circadian=clock; mind=LLM goal; policy=unified weighted walk (all stdlib, import-safe)
try:
    from director import context as _context   # live weather/time -> movement-energy nudge (fail-soft; optional)
except Exception:
    _context = None                            # a missing/broken context module must NEVER stop the walker

PANELS = {"A": (0, 0, 256, 256), "B": (256, 0, 192, 192)}
WINDOW_W, WINDOW_H = 448, 256
FPS = 10
TOPMOST_EVERY = 100        # re-assert window topmost every N frames (~10s @ FPS) so a transient cover never buries us
DISPLAY_RETRY_MS = 3000    # wait between window-rebuild attempts while a fullscreen app still owns the screen
GRAPH = ROOT / "data" / "clips" / "video_graph.json"
MIND_DIR = ROOT / "data" / "mind"
POSE_DIR = MIND_DIR / "pose"            # this walker publishes <char>.json here; the heartbeat reads it
INTENT_PATH = MIND_DIR / "intent.json"  # the heartbeat writes goals here; the mind layer reads them


def load_gif(path, size):
    im = Image.open(path)
    out = []
    for fr in ImageSequence.Iterator(im):
        rgb = fr.convert("RGB").resize(size, Image.LANCZOS)
        out.append(pygame.image.fromstring(rgb.tobytes(), rgb.size, "RGB"))
    return out


class GraphCycler:
    """Random-WALKS the video graph for one character: dwells on a pose playing fresh
    idles, occasionally takes a TRANSITION edge to another pose (plays it once, then
    advances to that node). So phineas walks anchor -> stand up -> standing idles ->
    sit down -> anchor, on a randomized non-repeating path."""
    def __init__(self, graph, character, size, loops, transition_prob=0.2, start_pose="anchor",
                 spec=None, hour_override=None, max_idle_secs=0.0, sleep_idle_secs=12.0,
                 mind_on=False, policy_on=False):
        self.size = size
        self.loops = max(1, loops)
        self.tp = transition_prob
        self.character = character
        self.spec = spec or {}            # bedtime_routine.json (drives the circadian layer)
        self.mind_on = mind_on            # consult the LLM-intent layer (mind.py) by day?
        self.policy_on = policy_on        # unified weighted policy (novelty+anti-reverse+goal+mood)?
        self.recent_clips = deque(maxlen=8)   # recency memory -> novelty weighting kills repetition
        self.recent_nodes = deque(maxlen=6)   # recently-VISITED poses -> stop orbiting one hub
        self._last_ctx_energy = None          # last weather/time energy band applied (traced in the pick log)
        self._intent = {}                 # cached intent.json, re-read on mtime change
        self._intent_mtime = -1.0
        self._pub_node = None             # last (node, dwell) published -> re-publish when either changes
        self._pub_dwell = -1              # so the heartbeat sees dwell ADVANCE while a character lingers
        try:
            self._graph_mtime = GRAPH.stat().st_mtime   # for autogen hot-reload
        except OSError:
            self._graph_mtime = -1.0
        self.hour_override = hour_override  # int 0-23 to force a clock hour (test/demo), else live
        self.pose_dwell = 0               # idle loops completed at the current pose (routine pacing)
        # Each IDLE clip plays/loops until its effective DURATION, then advances to the next
        # animation. The MJ clips are ~6s (61 frames @ FPS) with motion up front + a static settle.
        #   * day / going-to-bed beats: max_frames (~5s) -> snappy, trims the tail.
        #   * SLEEPING (dwelling at the sleep pose): sleep_idle_frames (~12s, loops the clip) ->
        #     each animation holds longer so it reads restful, not flickering.
        self.max_frames = int(max_idle_secs * FPS) if max_idle_secs and max_idle_secs > 0 else 0
        self.sleep_idle_frames = int(sleep_idle_secs * FPS) if sleep_idle_secs and sleep_idle_secs > 0 else 0
        self.sleep_node = circadian.sleep_node(self.spec, character)   # "<char>:sleep" / ":pod" dwell pose
        self.shown = 0                    # frames displayed for the current clip
        self.all = [e for e in graph.get("edges", []) if e.get("character") == character
                    or (e.get("from") or "").startswith(character + ":")]
        self.node = character + ":" + start_pose
        if not self._from(self.node):   # home pose absent (e.g. maxx has no :anchor) -> first pose with edges
            froms = [e.get("from") for e in self.all if self._from(e.get("from"))]
            if froms:
                self.node = froms[0]
        self.cache = {}
        self.cur = None
        self.fi = 0
        self.lc = 0
        self.last_id = None
        self.prev_node = None   # the pose we arrived FROM -> anti-pendulum (no immediate backtrack)
        self.last_label = None
        npose = len({e.get("from") for e in self.all})
        ntr = len([e for e in self.all if e.get("kind") == "transition"])
        if self.all:
            self._pick()
        print("panel char=%s edges=%d poses=%d transitions=%d mind=%s policy=%s" % (
            character, len(self.all), npose, ntr, "on" if mind_on else "off",
            "on" if policy_on else "off"), flush=True)
        if self.mind_on or self.policy_on:
            self._publish_pose()

    def _from(self, node):
        return [e for e in self.all if e.get("from") == node]

    def _hour(self):
        return self.hour_override if self.hour_override is not None else datetime.datetime.now().hour

    def _read_intent(self):
        """Re-read intent.json only when it changes on disk (cheap per-pick mtime poll)."""
        try:
            m = INTENT_PATH.stat().st_mtime
        except OSError:
            return {}
        if m != self._intent_mtime:
            try:
                self._intent = json.loads(INTENT_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._intent = {}
            self._intent_mtime = m
        return self._intent

    def _publish_pose(self):
        """Tell the heartbeat where this character is now AND how long it has lingered.
        Sole writer of its own pose file (per-character path -> no cross-panel race);
        atomic tmp+replace. Re-publishes when the node OR the dwell count changes, so the
        heartbeat's 'you have lingered N moments' reflects real time spent at the pose
        (publishing on node-change alone pinned dwell at 0 -> the brain was pacing blind)."""
        if self._pub_node == self.node and self._pub_dwell == self.pose_dwell:
            return
        try:
            POSE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = POSE_DIR / (self.character + ".json.tmp")
            tmp.write_text(json.dumps({
                "node": self.node, "dwell": self.pose_dwell, "last_label": self.last_label,
                "updated": datetime.datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
            os.replace(tmp, POSE_DIR / (self.character + ".json"))
            self._pub_node = self.node
            self._pub_dwell = self.pose_dwell
        except Exception as e:
            print("pose publish failed: %r" % e, flush=True)

    def _maybe_reload(self):
        """Hot-reload the video graph when it changes on disk (autogen grew a pose) so the
        new pose appears WITHOUT a player restart. Rebuilds this character's edge list and
        preserves the current node (re-homes only if it vanished). Cheap mtime poll; the
        gif frame-cache is keyed by path so existing clips stay cached, new ones lazy-load."""
        try:
            m = GRAPH.stat().st_mtime
        except OSError:
            return
        if m == self._graph_mtime:
            return
        self._graph_mtime = m
        try:
            graph = json.loads(GRAPH.read_text(encoding="utf-8"))
        except Exception:
            return
        new_all = [e for e in graph.get("edges", []) if e.get("character") == self.character
                   or (e.get("from") or "").startswith(self.character + ":")]
        if not new_all:
            return
        self.all = new_all
        if not self._from(self.node):                 # current node gone -> re-home
            froms = [e.get("from") for e in self.all if self._from(e.get("from"))]
            if froms:
                self.node = froms[0]
        print("[%s] graph hot-reloaded: %d edges, %d poses" % (
            self.character, len(self.all), len({e.get("from") for e in self.all})), flush=True)

    def _pick(self):
        """Pick the next clip. By DAY this is the random walk (transition with prob `tp`,
        else a fresh idle, anti-repeating only the exact last clip). At NIGHT the circadian
        layer takes over: it forces the next bedtime-routine step (advance toward sleep, dwell,
        or -- at the sleep pose -- only sleep idles), and by day it excludes the bedtime edges
        so the walk never wanders into the bedroom. circadian falls back to 'normal' whenever a
        character has no (or not-yet-generated) routine, so the daytime behaviour is unchanged."""
        if self.mind_on or self.policy_on:
            self._maybe_reload()              # pick up autogen-grown poses without a restart
        out = self._from(self.node)
        if not out:   # stranded (malformed graph) -> re-home to a pose that has edges; never freeze
            homes = [e.get("from") for e in self.all if self._from(e.get("from"))]
            if homes:
                self.node = homes[0]
                out = self._from(self.node)
        if self.policy_on:
            return self._pick_policy(out)     # unified weighted walk (replaces the layered branches)
        dec = circadian.decide(self.spec, self.character, self.node, out,
                               self.pose_dwell, self.last_id, self._hour(), random)
        if dec.get("action") == "force":
            self.cur = dec.get("edge")
            return self._after_pick()
        # DAY: the LLM-intent layer gets the next say -- walk toward (or hold at) the
        # heartbeat's chosen goal. No goal / stale / unreachable -> mind yields "none"
        # and the normal random walk below runs unchanged.
        if self.mind_on:
            mdec = mind.decide(self.character, self.node, out, self.all, self.last_id,
                               self._read_intent(), random)
            if mdec.get("action") == "force":
                self.cur = mdec.get("edge")
                return self._after_pick()
        cands = out
        if dec.get("action") == "normal":
            excl = dec.get("exclude") or set()
            cands = [e for e in out if e.get("label") not in excl] or out
        idles = [e for e in cands if e.get("kind") == "idle" and e.get("id") != self.last_id]
        trans = [e for e in cands if e.get("kind") == "transition"]
        # anti-pendulum: never take the transition straight back to where we just came from.
        # X->Y then Y->X plays a move and then its reverse -> reads as a reversing loop / a
        # repeated state. Fall back to all transitions ONLY if the backtrack is the lone exit,
        # so the walk never strands.
        fwd = [e for e in trans if e.get("to") != self.prev_node] or trans
        if fwd and (not idles or random.random() < self.tp):
            self.cur = random.choice(fwd)
        elif idles:
            self.cur = random.choice(idles)
        elif cands:
            self.cur = random.choice(cands)
        return self._after_pick()

    def _pick_policy(self, out):
        """Unified weighted pick. NIGHT still belongs to circadian (sleep is non-negotiable);
        by DAY one softmax (runtime/policy) weighs novelty + anti-reverse + soft goal-pull +
        mood together, sharing circadian's bedtime mask so a goal-walk is never routed through
        a bedtime pose (the deadlock that pinned the panels in a reverse-pendulum)."""
        hour = self._hour()
        if circadian.is_night(self.spec, self.character, hour):
            dec = circadian.decide(self.spec, self.character, self.node, out,
                                   self.pose_dwell, self.last_id, hour, random)
            if dec.get("action") == "force":        # bedtime chain / sleep idles win at night
                self._last_ctx_energy = None         # circadian owns the night -> no weather tilt
                self.cur = dec.get("edge")
                return self._after_pick()
        # DAY (or night with no routine): the LLM goal + mood shape the policy, never force a fight.
        intent = self._read_intent()
        cobj = (intent.get("characters", {}) or {}).get(self.character) or {}
        goal = mind.goal_for(intent, self.character) if (self.mind_on or self.policy_on) else None
        mask = circadian.bedtime_labels(self.spec, self.character)   # keep the daytime walk out of the bedroom
        self._last_ctx_energy = self._context_energy()   # cache for the pick log + feed the policy
        self.cur = policy.choose(
            self.node, out, self.all, goal=goal, mood=cobj.get("mood"), band=cobj.get("band"),
            route=cobj.get("route", "wander"), exclude=mask, prev_node=self.prev_node,
            last_id=self.last_id, last_label=self.last_label,
            recent_clips=self.recent_clips, recent_nodes=self.recent_nodes, rng=random,
            context_energy=self._last_ctx_energy)
        return self._after_pick()

    def _context_energy(self):
        """Live weather/time energy band ("high"|"low"|None) tilting the walk on top of the LLM
        mood: storm/wind -> high (agitated, alive); clear-calm or deep night -> low (settled).
        context_signal() is TTL-cached + fail-soft, so a per-pick call is cheap; the bare-except
        plus the optional-import guard guarantee a dead source NEVER breaks the 10fps loop."""
        if _context is None:
            return None
        try:
            sig = _context.context_signal()
            return sig.get("energy") if sig else None
        except Exception:
            return None

    def _after_pick(self):
        if self.cur:
            self.last_id = self.cur.get("id")
            self.last_label = self.cur.get("label")
            self.recent_clips.append(self.last_id)   # novelty memory: this clip was just played
            ce = (" ctx=%s" % self._last_ctx_energy) if self._last_ctx_energy else ""
            print("[%s] %-12s @ %s (h%d d%d)%s" % (self.character, self.cur.get("label"),
                  self.node, self._hour(), self.pose_dwell, ce), flush=True)
        self.fi = 0
        self.lc = 0
        self.shown = 0
        if self.mind_on or self.policy_on:
            self._publish_pose()

    def _frames(self):
        gif = self.cur.get("gif")
        if gif not in self.cache:
            try:
                self.cache[gif] = load_gif(str(ROOT / gif), (self.size[0], self.size[1]))
            except Exception as e:
                print("gif load failed %s: %r" % (gif, e), flush=True)
                self.cache[gif] = []
        return self.cache[gif]

    def frame(self):
        if not self.cur:
            return None
        fr = self._frames()
        if not fr:
            self._pick()
            return None
        s = fr[self.fi % len(fr)]
        self.fi += 1
        self.shown += 1
        if self.cur.get("kind") == "transition":
            if self.fi >= len(fr):                  # a transition plays ONCE, then lands on its pose
                self.prev_node = self.node          # remember where we came from (anti-pendulum)
                self.node = self.cur.get("to", self.node)
                self.recent_nodes.append(self.node)  # novelty memory: this pose was just visited
                self.pose_dwell = 0                 # fresh pose -> reset dwell counter
                self._pick()
            return s
        # IDLE: loop the clip; advance to the next animation after the effective DURATION.
        # At the sleep pose, sleep_idle_frames (longer = more time, restful); else max_frames (snappy).
        # Either cap counts as a dwell unit so the circadian chain still advances; cap 0 -> use loops.
        if self.fi >= len(fr):
            self.fi = 0
            self.lc += 1
        at_sleep = self.sleep_node is not None and self.node == self.sleep_node
        cap = self.sleep_idle_frames if at_sleep else self.max_frames
        done = (self.shown >= cap) if cap else (self.lc >= self.loops)
        if done:
            self.pose_dwell += 1                    # one dwell unit at this pose
            self._pick()
        return s


def _assert_topmost():
    """Pin the borderless window on top at (0,0). Called at startup, after a display rebuild,
    and periodically -- so a transient cover or a display reset never leaves the panels buried.
    SWP_NOACTIVATE so re-pinning never steals focus from whatever is active. Best-effort; the
    bare-except keeps a topmost hiccup from ever reaching the render loop."""
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        SWP_SHOWWINDOW, SWP_NOACTIVATE = 0x0040, 0x0010
        u.SetWindowPos(pygame.display.get_wm_info()["window"], ctypes.c_void_p(-1),
                       0, 0, WINDOW_W, WINDOW_H, SWP_SHOWWINDOW | SWP_NOACTIVATE)
    except Exception as e:
        print("topmost failed:", e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="phineas", help="character on panel A")
    ap.add_argument("--b", default="seraphina", help="character on panel B")
    ap.add_argument("--loops", type=int, default=1, help="loops of an edge before picking another")
    ap.add_argument("--a-pose", dest="a_pose", default="anchor", help="start pose/node for panel A")
    ap.add_argument("--b-pose", dest="b_pose", default="anchor", help="start pose/node for panel B")
    ap.add_argument("--a-tp", dest="a_tp", type=float, default=0.2, help="panel A transition prob (state-hop rate, 0-1)")
    ap.add_argument("--b-tp", dest="b_tp", type=float, default=0.2, help="panel B transition prob (state-hop rate, 0-1)")
    ap.add_argument("--now-hour", dest="now_hour", type=int, default=None,
                    help="override the clock hour 0-23 to force day/night (test/demo); default = live clock")
    ap.add_argument("--max-idle-secs", dest="max_idle_secs", type=float, default=0.0,
                    help="day/going-to-bed: hold each IDLE clip N seconds then cut to the next. "
                         "DEFAULT 0 = play the FULL clip so its MJ --end loop closes seamlessly. A "
                         "positive value cuts mid-clip BEFORE the loop closes -> a jump-cut, glaring "
                         "on big-motion characters (MAXX) though invisible on subtle ones (Phineas).")
    ap.add_argument("--sleep-idle-secs", dest="sleep_idle_secs", type=float, default=12.0,
                    help="SLEEPING (at the sleep pose): hold each animation N seconds before the next -- bigger = more restful (loops the ~6s clip). default 12")
    ap.add_argument("--mind", action="store_true",
                    help="consult the LLM-intent layer (heartbeat goals from data/mind/intent.json) by day; off = pure random walk (current production default)")
    ap.add_argument("--policy", action="store_true",
                    help="use the UNIFIED weighted walk (runtime/policy): novelty + anti-reverse + "
                         "soft goal-pull + mood, one softmax. Fixes the reverse-pendulum + repetition "
                         "+ the goal-through-bedtime deadlock. Implies --mind for goals/mood.")
    args = ap.parse_args()
    if args.policy:
        args.mind = True   # the policy consumes the heartbeat's goal + mood from intent.json

    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    try:
        spec = json.loads((ROOT / "prompts" / "bedtime_routine.json").read_text(encoding="utf-8"))
    except Exception:
        spec = {}
    print("video graph: %d nodes, %d edges | bedtime: %s | hour=%s" % (
        len(graph.get("nodes", {})), len(graph.get("edges", [])),
        ",".join(spec.get("characters", {})) or "none",
        args.now_hour if args.now_hour is not None else "live"), flush=True)

    pygame.init()
    pygame.display.set_caption("lp-preview-graph")

    def _open_window():
        scr = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.NOFRAME)
        _assert_topmost()
        return scr

    screen = _open_window()

    ra, rb = PANELS["A"], PANELS["B"]
    ca = GraphCycler(graph, args.a, (ra[2], ra[3]), args.loops, transition_prob=args.a_tp,
                     start_pose=args.a_pose, spec=spec, hour_override=args.now_hour,
                     max_idle_secs=args.max_idle_secs, sleep_idle_secs=args.sleep_idle_secs,
                     mind_on=args.mind, policy_on=args.policy)
    cb = GraphCycler(graph, args.b, (rb[2], rb[3]), args.loops, transition_prob=args.b_tp,
                     start_pose=args.b_pose, spec=spec, hour_override=args.now_hour,
                     max_idle_secs=args.max_idle_secs, sleep_idle_secs=args.sleep_idle_secs,
                     mind_on=args.mind, policy_on=args.policy)

    clock = pygame.time.Clock()
    running = True
    n = 0
    while running:
        try:
            for e in pygame.event.get():
                # A wall kiosk must NOT quit because a fullscreen app (a Steam game) destroyed its
                # window -- that QUIT used to clean-exit the player and leave the panels dark until
                # the 3-min watchdog. Only the operator's ESC exits now; QUIT is ignored and a lost
                # display is rebuilt in the except below.
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
            screen.fill((0, 0, 0))
            sa = ca.frame()
            if sa:
                screen.blit(sa, (ra[0], ra[1]))
            sb = cb.frame()
            if sb:
                screen.blit(sb, (rb[0], rb[1]))
            pygame.display.flip()
            n += 1
            if n % TOPMOST_EVERY == 0:        # re-pin on top so a transient cover never buries us
                _assert_topmost()
        except pygame.error as ex:
            # Display/surface lost -- almost always a fullscreen app grabbing the screen. Don't die:
            # drop the gif caches, wait, and rebuild the window so the panels return the instant the
            # screen is free again. The same process persists (watchdog sees it UP -> no restart churn).
            print("display lost (%r) -> recovering" % ex, flush=True)
            try:
                ca.cache.clear(); cb.cache.clear()
                pygame.display.quit()
            except Exception:
                pass
            pygame.time.wait(DISPLAY_RETRY_MS)
            try:
                pygame.display.init()
                screen = _open_window()
                print("display recovered", flush=True)
            except pygame.error as ex2:
                print("recover deferred (%r)" % ex2, flush=True)   # screen still owned -> retry next loop
            continue
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
