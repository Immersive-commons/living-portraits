"""lived.py -- what a character has actually LIVED, accumulated at runtime.

The audit's most damning structural finding was not that the graph was small. It was
that the graph is an ASSET MANIFEST: nodes are poses a generator made, edges are clips
it rendered, and **nothing a character experiences was ever written back into it**. The
graph accreted from a credit card, not from living.

This is the write-back half. Per character, per node: how many times it has stood there,
when it first did, when it last did, how long it has spent, and which mood-bands it
brought. Per edge: how many times that clip has actually played. That turns "255 nodes"
into a lived topography with hot paths and cold corners -- and it is the VALID-TIME half
of a bi-temporal graph (when a thing was true of the world), the sibling of the graph's
own provenance (when the system learned the thing exists).

WHY A SEPARATE FILE, not fields on the graph: `video_graph.build()` REGENERATES
video_graph.json from NODE_SPECS/EDGE_SPECS and overwrites it, and Phase-2 autogen rebuilds
roughly every 20 minutes. Anything written into that file by the runtime is destroyed on the
next pose the characters grow. So lived experience is its own store, single-writer per
character (that panel's walker), merged by readers. Same discipline as intent.json and
pose/<char>.json.

Written from the 10fps render loop, so: in-memory counters, a bounded flush interval, one
atomic tmp+replace per flush, and every failure swallowed. A panel must never go dark
because a statistics file could not be written.

    from runtime import lived
    book = lived.Lived("phineas")        # walker side
    book.visit("phineas:glower", mood_band="fixated")
    book.play("phineas/a2g/v0")
    book.tick_dwell("phineas:glower")
    book.flush()                          # no-op unless FLUSH_EVERY has elapsed

    seen = lived.load("phineas")          # reader side (heartbeat)
    seen.visits("phineas:glower")         # -> 88
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVED_DIR = ROOT / "data" / "mind" / "lived"
SCHEMA = "living-portrait.lived/v1"

FLUSH_EVERY = 60.0     # seconds; the walker updates memory constantly, disk rarely
MAX_MOODS = 12         # per node: keep the most-brought bands, not an unbounded histogram


def path_for(character, base=None):
    return (Path(base) if base else LIVED_DIR) / ("%s.json" % character)


class Lived:
    """One character's lived record. The WALKER owns the instance and is its sole writer."""

    def __init__(self, character, base=None, now=None):
        self.character = character
        self.path = path_for(character, base)
        self._last_flush = (time.time() if now is None else now)
        self._dirty = False
        d = _read(self.path)
        self.nodes = d.get("nodes") or {}
        self.edges = d.get("edges") or {}
        # Carry through any key this class does not own. A writer that silently drops what it
        # does not recognise destroys other tools' data: the first flush after the 76-day
        # backfill erased its `backfill` provenance block and `label_plays` while keeping the
        # counts, so the record kept the numbers and lost where they came from.
        self._extra = {k: v for k, v in d.items()
                       if k not in ("schema", "character", "updated", "nodes", "edges")}

    # ---------------------------------------------------------------- writes
    def visit(self, node, mood_band=None, now=None):
        """Arrived at a pose. First arrival stamps `first` -- the moment this pose entered
        the character's life, which the graph itself cannot express."""
        now = time.time() if now is None else now
        rec = self.nodes.setdefault(node, {"visits": 0, "first": now, "dwell": 0, "moods": {}})
        rec["visits"] = int(rec.get("visits", 0)) + 1
        rec["last"] = max(now, rec.get("last") or now)
        # earliest wins: a clock step back (or a replayed record) must not rewrite the origin
        # of a pose's history to something later than a visit already on file.
        rec["first"] = min(rec.get("first", now), now)
        if mood_band:
            moods = rec.setdefault("moods", {})
            moods[mood_band] = int(moods.get(mood_band, 0)) + 1
            if len(moods) > MAX_MOODS:      # bounded: drop the least-brought band
                rec["moods"] = dict(sorted(moods.items(), key=lambda kv: -kv[1])[:MAX_MOODS])
        self._dirty = True

    def tick_dwell(self, node, n=1, now=None):
        """One more idle unit spent here. Cheap: no timestamp write on the hot path."""
        rec = self.nodes.setdefault(
            node, {"visits": 0, "first": (time.time() if now is None else now), "dwell": 0, "moods": {}})
        rec["dwell"] = int(rec.get("dwell", 0)) + int(n)
        self._dirty = True

    def play(self, edge_id, now=None):
        """This clip actually rolled on a panel -- as opposed to merely existing."""
        if not edge_id:
            return
        now = time.time() if now is None else now
        rec = self.edges.setdefault(edge_id, {"plays": 0, "first": now})
        rec["plays"] = int(rec.get("plays", 0)) + 1
        rec["last"] = now
        self._dirty = True

    def flush(self, force=False, now=None):
        """Persist at most every FLUSH_EVERY seconds. Returns True if it wrote.

        Fail-soft by contract: a statistics file is never worth a dark panel."""
        now = time.time() if now is None else now
        if not self._dirty and not force:
            return False
        if not force and (now - self._last_flush) < FLUSH_EVERY:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            body = dict(self._extra)                       # foreign keys first...
            body.update({"schema": SCHEMA, "character": self.character, "updated": now,
                         "nodes": self.nodes, "edges": self.edges})   # ...ours always win
            tmp.write_text(json.dumps(body), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            return False
        self._last_flush = now
        self._dirty = False
        return True

    # ---------------------------------------------------------------- reads
    def visits(self, node):
        return int((self.nodes.get(node) or {}).get("visits", 0))

    def dwell(self, node):
        return int((self.nodes.get(node) or {}).get("dwell", 0))

    def plays(self, edge_id):
        return int((self.edges.get(edge_id) or {}).get("plays", 0))

    def visited(self):
        """Every pose actually stood in. Ground truth for the frontier -- the journal only
        records decision POINTS, so a pose walked THROUGH never appeared there."""
        return {n for n, r in self.nodes.items() if int(r.get("visits", 0)) > 0}

    def unplayed(self, all_edge_ids):
        """Clips that exist and have never once rolled. Rendered, paid for, never seen."""
        return {e for e in all_edge_ids if self.plays(e) == 0}

    def dominant_mood(self, node):
        moods = (self.nodes.get(node) or {}).get("moods") or {}
        return max(moods.items(), key=lambda kv: kv[1])[0] if moods else None

    def age(self, node, now=None):
        """(seconds since first stood here, seconds since last) or (None, None)."""
        rec = self.nodes.get(node) or {}
        now = time.time() if now is None else now
        first, last = rec.get("first"), rec.get("last")
        return ((now - first) if first else None, (now - last) if last else None)

    def totals(self):
        v = sum(int(r.get("visits", 0)) for r in self.nodes.values())
        return {"poses_lived": len(self.visited()), "visits": v,
                "clips_played": sum(int(r.get("plays", 0)) for r in self.edges.values())}


def _read(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def load(character, base=None):
    """Reader-side snapshot. Readers never write, so a torn read degrades to fewer facts,
    never to a crash."""
    return Lived(character, base=base)
