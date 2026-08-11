#!/usr/bin/env python3
"""backfill_lived.py -- reconstruct 76 days of traversal from the walker's own pick-log.

`runtime/lived.py` started recording on 2026-08-10 at 14:51, so the characters' lived
record began 70 minutes old while the installation itself is months old. But the walker has
been printing every single pick since 2026-05-26 into `_preview.log` -- 849,107 lines, 434
restarts -- and that log is a complete, if lossy, history of every pose either character has
stood in. This mines it back into the record.

WHAT IS RECOVERABLE, and what is not:
  * VISITS and DWELL per pose -- exact. The log prints the node at every pick, so a node
    CHANGE is an arrival and a repeat is an idle unit. This is the thing that was asked for:
    which nodes do the paintings actually traverse.
  * PLAYS per clip -- at LABEL level only. 83 labels have 2-4 rendered variants sharing one
    name, and the log prints the label, so attributing a play to `.../v2` rather than
    `.../v0` would be invention. Label counts are exact; they go in their own field and the
    id-keyed `edges` map is left to the live recorder, which knows the id for certain.
  * TIMESTAMPS -- NOT recoverable. The log carries hour-of-day and dwell, never a date. So
    backfilled poses get NO `first`/`last`. That follows the same rule as graph provenance:
    no evidence, no field, never a fabricated date. `_lived_line` already omits the "first
    seen" clause when the stamp is absent, so this degrades to counts, which is honest.
  * MOOD per pose -- not in the log. The live record's `moods` are preserved untouched.

ORDERING HAZARD: the running walker holds its counters in memory and flushes every 60s, so
it will overwrite anything written underneath it. Stop lp-preview first. --apply refuses
to run while a walker process is alive unless --force is passed.

    python scripts/backfill_lived.py                 # dry run: what would be recovered
    python scripts/backfill_lived.py --apply         # merge into data/mind/lived/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG = ROOT / "_preview.log"
LIVED_DIR = ROOT / "data" / "mind" / "lived"

# [phineas] a2s          @ phineas:anchor (h16 d17)   -- hour/dwell absent in the oldest lines
PICK = re.compile(r"^\[(?P<char>[a-z0-9_]+)\]\s+(?P<label>\S+)\s+@\s+(?P<node>\S+)"
                  r"(?:\s+\(h(?P<hour>\d+)\s+d(?P<dwell>\d+)\))?")


def parse(log_path):
    """-> {char: {"nodes": {node: {visits, dwell}}, "labels": {label: plays},
                  "picks": n, "first_node": str}}"""
    out = {}
    last_node = {}
    try:
        f = open(str(log_path), "r", encoding="utf-8", errors="replace")
    except OSError as e:
        raise SystemExit("cannot read %s: %s" % (log_path, e))
    with f:
        for line in f:
            if not line.startswith("["):
                continue                      # restart banners, pygame noise, tracebacks
            m = PICK.match(line)
            if not m:
                continue
            ch, node, label = m.group("char"), m.group("node"), m.group("label")
            rec = out.setdefault(ch, {"nodes": {}, "labels": {}, "picks": 0, "first_node": node})
            rec["picks"] += 1
            rec["labels"][label] = rec["labels"].get(label, 0) + 1
            n = rec["nodes"].setdefault(node, {"visits": 0, "dwell": 0})
            if last_node.get(ch) != node:
                n["visits"] += 1              # a node CHANGE is an arrival
            else:
                n["dwell"] += 1               # a repeat is an idle unit spent here
            last_node[ch] = node
    return out


def walkers_alive():
    """Refuse to write under a running walker -- it would flush its own counters over us."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "@(Get-Process pythonw -ErrorAction SilentlyContinue).Count"],
                           capture_output=True, text=True, timeout=30)
        return int((r.stdout or "0").strip() or 0)
    except Exception:
        return -1                              # unknown: treated as "not proven safe"


def merge(char, back, live_path, log_span):
    """Backfill counts SUPERSEDE the live ones (the log covers the live window too, so the
    live counters are a subset). Everything the log cannot know -- moods, per-id clip plays,
    timestamps already earned -- is preserved."""
    live = {}
    if live_path.exists():
        try:
            live = json.loads(live_path.read_text(encoding="utf-8"))
        except Exception:
            live = {}
    live_nodes = live.get("nodes") or {}
    merged = {}
    for node, b in back["nodes"].items():
        keep = dict(live_nodes.get(node) or {})
        keep["visits"] = b["visits"]
        keep["dwell"] = b["dwell"]
        keep["backfilled"] = True             # counts came from the log, not from a live visit
        merged[node] = keep
    for node, l in live_nodes.items():        # a pose the live record saw and the log did not
        if node not in merged:
            merged[node] = l
    out = dict(live)
    out.update({
        "schema": live.get("schema") or "living-portrait.lived/v1",
        "character": char,
        "nodes": merged,
        "edges": live.get("edges") or {},     # id-keyed plays stay the live recorder's job
        "label_plays": back["labels"],        # exact, but label-level: variants share a name
        "backfill": {"source": "_preview.log", "picks": back["picks"],
                     "span": log_span,
                     "note": "visits/dwell recovered from the pick-log; no timestamps exist "
                             "in it, so backfilled poses carry no first/last date"},
    })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the merged records")
    ap.add_argument("--force", action="store_true", help="write even if a walker is running")
    ap.add_argument("--log", default=str(LOG))
    args = ap.parse_args()

    log_path = Path(args.log)
    st = log_path.stat()
    span = {"log_created": st.st_ctime, "log_modified": st.st_mtime,
            "bytes": st.st_size}
    print("mining %s (%.1f MB)" % (log_path, st.st_size / 1e6))
    data = parse(log_path)
    if not data:
        raise SystemExit("no pick lines found -- is this the right log?")

    for ch, rec in sorted(data.items()):
        nodes = rec["nodes"]
        visits = sum(n["visits"] for n in nodes.values())
        dwell = sum(n["dwell"] for n in nodes.values())
        print("\n%s: %d picks | %d poses stood in | %d arrivals | %d idle units | %d clip labels"
              % (ch.upper(), rec["picks"], len(nodes), visits, dwell, len(rec["labels"])))
        top = sorted(nodes.items(), key=lambda kv: -kv[1]["dwell"])[:5]
        for node, n in top:
            print("    %-34s visits %-5d dwell %-6d" % (node.split(":")[-1], n["visits"], n["dwell"]))

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply (stop lp-preview first).")
        return 0

    alive = walkers_alive()
    if alive != 0 and not args.force:
        raise SystemExit(
            "\nREFUSING: %s walker process(es) look alive. A running walker flushes its own\n"
            "counters every 60s and would overwrite this. Stop lp-preview, then re-run\n"
            "(or pass --force if you know better)." % (alive if alive >= 0 else "an unknown number of"))

    LIVED_DIR.mkdir(parents=True, exist_ok=True)
    for ch, rec in sorted(data.items()):
        p = LIVED_DIR / ("%s.json" % ch)
        if p.exists():
            bak = p.with_suffix(".json.pre-backfill")
            bak.write_bytes(p.read_bytes())      # production data: keep the old one
        merged = merge(ch, rec, p, span)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged), encoding="utf-8")
        os.replace(tmp, p)
        print("wrote %s (%d poses)" % (p, len(merged["nodes"])))
    print("\nRestart lp-preview; it will load the merged record and continue from it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
