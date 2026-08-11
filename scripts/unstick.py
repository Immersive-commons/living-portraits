#!/usr/bin/env python3
"""unstick.py -- give a cul-de-sac pose a second way out, chosen by measured cost.

56 of Phineas's 120 poses have one exit or none, and the same for 56 of MAXX's 138. That is
autogen's star topology: every new pose is a spur off a hub, so its only exit is the reverse
of the clip that got you there. Under a low-energy band the walk then sits there.

`runtime/policy.ESCAPE_VELOCITY` already fixes the SYMPTOM for free -- exits get more
attractive the longer a character has been somewhere, which covers all 112 spurs including
ones that do not exist yet. This script fixes the TOPOLOGY, which is a different thing: a
second real edge means somewhere else to go, not just a stronger pull toward the one door.

IT SPENDS REAL CREDITS, so it spends them where the traversal record says the time actually
goes. Fixing all 112 is ~840 credits, 28% of a month, to reclaim dwell whose worst single
member is 1.1% of a life. Ranked by measured dwell x visits, the top handful is ~5%.

Mechanism is autogen's own: append an `extra_links` entry to the pose's record in
data/mind/autogen_poses.json and generate the pair. `video_graph._autogen_additions` already
emits link edges in BOTH directions, gated on both gifs existing, dropping anything dangling
-- so a failed generation cannot strand the walk. No new graph code.

    python scripts/unstick.py                      # ranked plan + credit cost (SAFE default)
    python scripts/unstick.py --apply --limit 1    # generate for the worst one
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CREDITS_PER_CLIP = 7.5      # kling3_0 @ 1:1 / 5s / sound-off, the autogen default
CLIPS_PER_LINK = 2          # a link is a PAIR: out and back


def _load(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def rank(character, graph, lived_dir, autogen):
    """One-exit AUTOGEN poses, worst first. Base/bedtime poses are excluded -- they are not
    in the autogen record and are hand-authored topology, not accidental spurs."""
    edges = [e for e in graph["edges"] if (e.get("from") or "").startswith(character + ":")]
    outs = {}
    for e in edges:
        if e.get("kind") == "transition":
            outs.setdefault(e["from"], set()).add(e["to"])
    lived = _load(Path(lived_dir) / ("%s.json" % character), {}).get("nodes") or {}
    total_dwell = sum(int(v.get("dwell", 0)) for v in lived.values()) or 1
    owned = (autogen.get("characters", {}).get(character, {}).get("poses", {}) or {})

    rows = []
    for pose in owned:
        nid = "%s:%s" % (character, pose)
        exits = outs.get(nid, set()) - {nid}
        if len(exits) > 1:
            continue                                  # already has somewhere to go
        rec = lived.get(nid) or {}
        dwell, visits = int(rec.get("dwell", 0)), int(rec.get("visits", 0))
        if visits == 0:
            continue                                  # never been there: a link buys nothing
        rows.append({"pose": pose, "node": nid, "dwell": dwell, "visits": visits,
                     "share": dwell / float(total_dwell),
                     "exits": sorted(x.split(":")[-1] for x in exits)})
    rows.sort(key=lambda r: -r["dwell"])
    return rows


def best_sibling(character, graph, pose):
    """Link to the most-connected pose that is NOT the spur's existing exit and NOT part of
    the bedtime chain -- the point is a daytime way out, and hanging a spur off the bedroom
    is what created these in the first place."""
    from runtime import circadian
    spec = _load(ROOT / "prompts" / "bedtime_routine.json", {})
    night = circadian.bedtime_poses(spec, character)
    edges = [e for e in graph["edges"] if (e.get("from") or "").startswith(character + ":")]
    deg = {}
    for e in edges:
        if e.get("kind") == "transition":
            deg[e["from"]] = deg.get(e["from"], 0) + 1
    nid = "%s:%s" % (character, pose)
    cur = {e["to"] for e in edges if e.get("from") == nid and e.get("kind") == "transition"}
    cands = [(d, n) for n, d in deg.items()
             if n != nid and n not in cur and n not in night and n.startswith(character + ":")]
    if not cands:
        return None
    return max(cands)[1].split(":", 1)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="actually generate (spends credits)")
    ap.add_argument("--limit", type=int, default=1, help="how many poses to unstick")
    ap.add_argument("--character", default=None)
    ap.add_argument("--graph", default=str(ROOT / "data" / "clips" / "video_graph.json"))
    ap.add_argument("--lived", default=str(ROOT / "data" / "mind" / "lived"))
    ap.add_argument("--autogen", default=str(ROOT / "data" / "mind" / "autogen_poses.json"))
    args = ap.parse_args()

    graph = _load(args.graph, {"nodes": {}, "edges": []})
    autogen = _load(args.autogen, {})
    if not graph["edges"]:
        raise SystemExit("no graph at %s -- run this on the host" % args.graph)

    chars = [args.character] if args.character else sorted(
        autogen.get("characters", {}).keys())
    plan = []
    for ch in chars:
        rows = rank(ch, graph, args.lived, autogen)
        print("\n%s: %d one-exit autogen poses that have actually been visited" % (ch.upper(), len(rows)))
        for r in rows[:8]:
            sib = best_sibling(ch, graph, r["pose"])
            print("   %-26s %4.1f%% of life  %5d visits  only exit -> %s   | propose link -> %s"
                  % (r["pose"], 100 * r["share"], r["visits"], ",".join(r["exits"]) or "NONE", sib))
        for r in rows[:args.limit]:
            sib = best_sibling(ch, graph, r["pose"])
            if sib:
                plan.append((ch, r, sib))

    clips = len(plan) * CLIPS_PER_LINK
    print("\nPLAN: %d link(s) = %d clips = %.0f credits (%.1f%% of a 3000/month allowance)"
          % (len(plan), clips, clips * CREDITS_PER_CLIP, 100.0 * clips * CREDITS_PER_CLIP / 3000))
    for ch, r, sib in plan:
        print("   %s: %s <-> %s   (reclaims ~%.1f%% of dwell)" % (ch, r["pose"], sib, 100 * r["share"]))
    if not args.apply:
        print("\nDRY RUN -- nothing generated. Re-run with --apply to spend.")
        return 0

    from pipeline import autogen as ag
    store = _load(args.autogen, {})
    for ch, r, sib in plan:
        rec = store["characters"][ch]["poses"][r["pose"]]
        links = rec.setdefault("extra_links", [])
        if any(el.get("sibling") == sib for el in links):
            print("  %s:%s already links to %s -- skipping" % (ch, r["pose"], sib))
            continue
        # HIGGSFIELD, not Midjourney. autogen._generate_extra_links (and _client) are the
        # RETIRED MJ path -- it has returned 403 on every call since 2026-07-02, which is
        # exactly what the first version of this script walked into. hf_gen takes two local
        # stills and needs no upload at all.
        from pipeline import hf_gen
        ok, detail = hf_gen.healthy()
        if not ok:
            print("  Higgsfield unavailable (%s) -- nothing generated, nothing spent" % detail)
            return 1
        lfwd, lrev = ag._edge_labels(sib, r["pose"])
        pretty = lambda s: s.replace("_", " ")
        node_img = (graph["nodes"].get(r["node"]) or {}).get("image")
        sib_img = (graph["nodes"].get("%s:%s" % (ch, sib)) or {}).get("image")
        if not node_img or not sib_img:
            print("  missing a still for %s or %s -- skipping" % (r["pose"], sib))
            continue
        pairs = [(lfwd, ROOT / sib_img, ROOT / node_img,
                  "the figure moves from the %s pose into the %s pose and settles naturally"
                  % (pretty(sib), pretty(r["pose"]))),
                 (lrev, ROOT / node_img, ROOT / sib_img,
                  "the figure moves from the %s pose back into the %s pose and settles naturally"
                  % (pretty(r["pose"]), pretty(sib)))]
        made = []
        for label, start, end, motion in pairs:
            gif = ag.PROTO / ("%s_%s_v0.gif" % (ch, label))
            if gif.exists() and gif.stat().st_size > 0:
                print("  %s already on disk -- reusing, not re-spending" % gif.name)
                made.append(label)
                continue
            if ag.clip_budget_left(ch) <= 0:
                print("  daily clip budget exhausted for %s -- stopping cleanly" % ch)
                break
            mp4 = ag.PROTO / ("%s_%s_v0.mp4" % (ch, label))
            hf_gen.generate_clip(start, end, motion, mp4)
            ag._spend_clip(ch)
            ag._mp4_to_gif(mp4, gif)
            made.append(label)
            print("  generated %s (%d clips left today)" % (gif.name, ag.clip_budget_left(ch)))
        # BOTH directions or neither: _autogen_additions drops a half-link anyway, and a
        # recorded link whose clips are missing is a lie in the record.
        if len(made) == 2:
            links.append({"sibling": sib, "label": lfwd, "reverse_label": lrev,
                          "motion": pairs[0][3]})
            Path(args.autogen).write_text(json.dumps(store, indent=2), encoding="utf-8")
            print("  recorded link %s <-> %s" % (sib, r["pose"]))
        else:
            print("  only %d/2 clips landed for %s <-> %s -- record untouched" % (len(made), sib, r["pose"]))

    # rebuild in a FRESH process: video_graph merges autogen records at IMPORT time, so an
    # in-process rebuild would use a stale snapshot and silently drop the new edges.
    subprocess.run([sys.executable, str(ROOT / "runtime" / "video_graph.py"), "build"], cwd=str(ROOT))
    return 0


class _G:
    """_generate_extra_links only needs .nodes for the sibling's still image."""
    def __init__(self, graph):
        self.nodes = graph["nodes"]


if __name__ == "__main__":
    sys.exit(main())
