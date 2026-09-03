#!/usr/bin/env python3
"""seed_demo_media.py -- make a bare checkout walkable without any of our art.

WHY THIS EXISTS
---------------
`data/` is gitignored, and rightly so: the anchor stills and the motion clips are
generated art, hundreds of megabytes of it, and they are not what this repo is
about. But `runtime/video_graph.py build` discovers edges by globbing for those
clips on disk, so on a fresh clone it finds none, every pose ends up with no way
out, and the build refuses to save:

    ERROR: node phineas:anchor has NO outgoing edges -> the walk would strand here
    GRAPH NOT SAVED -- 21 walk-safety error(s).

That is the build being correct. It is also a wall: with no graph there is no
walk, no lived record, no context view, and the viewer opens on nothing. A
newcomer cannot see the system they were asked to work on.

This script writes PLACEHOLDER media -- flat colour cards with the pose name on
them, and short procedurally-animated loops -- one per still and per clip the
graph declares. It is enough for `build` to produce a real, walk-safe graph and
for every downstream surface to have something to show. It is not art and does
not pretend to be: every frame says PLACEHOLDER on it.

WHAT IT WILL NOT DO
-------------------
It never overwrites a file that already exists. That is the whole safety story,
and it is why this is safe to run on the production host by accident: real media
is simply skipped. `--force` overwrites. It writes only under `data/gen/` and
`data/clips/_proto/`, both gitignored.

WHERE THE SPECS COME FROM
-------------------------
NODE_SPECS and EDGE_SPECS are imported from `runtime.video_graph`, not restated
here -- a second copy of that table would drift the first time someone adds a
pose, and the seeder would then quietly under-seed the graph it exists to fill.

The bedtime chain is read separately from `prompts/bedtime_routine.json`. It has
to be: `video_graph` merges a character's bedtime beats only once every clip that
character declares already exists, so on a first run those edges are not yet in
EDGE_SPECS to be seen. Seeding straight from the spec closes that loop in one
pass -- which matters, because the circadian layer is what the Decision lens is
looking at after dark.

Usage
-----
    python scripts/seed_demo_media.py            # seed what is missing
    python scripts/seed_demo_media.py --dry-run  # say what it would write
    python scripts/seed_demo_media.py --force    # overwrite existing media

Then:
    python runtime/video_graph.py build
    python scripts/export_context_view.py

Requires Pillow (in requirements.txt). ASCII only, no network.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from PIL import Image, ImageDraw
except ImportError:                                             # pragma: no cover
    sys.exit("Pillow is required: python -m pip install -r requirements.txt")

GEN = ROOT / "data" / "gen"
PROTO = ROOT / "data" / "clips" / "_proto"
BEDTIME = ROOT / "prompts" / "bedtime_routine.json"

# Panel A is 256x256 and panel B 192x192 (panels.yaml). Seed at 256 square: the
# player scales to the panel rect, so one size covers both and nothing is cropped.
SIZE = 256
FRAMES = 12           # a 1.2s loop at the player's 10 fps
DURATION_MS = 100

# One hue per character so the placeholder walk is legible at a glance -- you can
# see a transition happen on the panel without reading the pose name.
PALETTE = {
    "phineas":   ((38, 14, 16), (210, 170, 90)),      # maroon / gold, panel A
    "seraphina": ((12, 26, 28), (120, 200, 190)),     # teal / mint, panel B
    "maxx":      ((16, 14, 34), (150, 130, 235)),     # violet
}
DEFAULT_PALETTE = ((22, 22, 24), (170, 170, 175))


def _palette(character):
    return PALETTE.get(character, DEFAULT_PALETTE)


def _card(character, top, bottom, phase=0.0):
    """One placeholder frame: flat ground, breathing accent bar, two text lines.

    `phase` in [0,1) drives the only moving element, so a sequence of frames is a
    genuine loop rather than a stack of identical images -- the render path is
    then exercising real frame advance, which a still would not test.
    """
    bg, accent = _palette(character)
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)

    # Breathing bar: a triangle wave, so frame 0 and frame N meet without a jump.
    tri = 2 * phase if phase < 0.5 else 2 * (1.0 - phase)
    half = int((SIZE * 0.12) + (SIZE * 0.26) * tri)
    cy = int(SIZE * 0.46)
    d.rectangle([SIZE // 2 - half, cy - 3, SIZE // 2 + half, cy + 3], fill=accent)

    d.rectangle([0, 0, SIZE - 1, SIZE - 1], outline=accent)
    d.text((10, 12), top, fill=accent)
    d.text((10, 26), bottom, fill=accent)
    d.text((10, SIZE - 22), "PLACEHOLDER - not real art", fill=accent)
    return img


def _write_still(path, character, pose, *, force, dry):
    if path.exists() and not force:
        return "skip"
    if dry:
        return "would-write"
    path.parent.mkdir(parents=True, exist_ok=True)
    _card(character, character, pose, phase=0.0).save(path)
    return "wrote"


def _write_clip(path, character, label, kind, *, force, dry):
    if path.exists() and not force:
        return "skip"
    if dry:
        return "would-write"
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [_card(character, character, "%s (%s)" % (label, kind), phase=i / FRAMES)
              for i in range(FRAMES)]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=DURATION_MS, loop=0, optimize=False)
    return "wrote"


def _bedtime_targets():
    """(stills, clips) the bedtime spec declares, read straight from the JSON.

    Returns ([(image_path_str, char, pose)], [(char, label, kind)]). Every
    transition beat contributes its reverse as well -- in production that clip is
    the forward one played backwards, and the graph treats it as a first-class
    edge, so a seeder that skipped it would leave each night pose one-way.
    """
    try:
        spec = json.loads(BEDTIME.read_text(encoding="utf-8"))
    except Exception:
        return [], []
    stills, clips = [], []
    for char, cs in (spec.get("characters") or {}).items():
        for pose, n in (cs.get("nodes") or {}).items():
            if n.get("node_image"):
                stills.append((n["node_image"], char, pose))
        for beat in cs.get("routine") or []:
            if beat.get("kind") == "transition":
                for lbl in (beat.get("label"), beat.get("reverse_label")):
                    if lbl:
                        clips.append((char, lbl, "transition"))
            for idle in beat.get("idles") or []:
                if idle.get("id"):
                    clips.append((char, idle["id"], "idle"))
    return stills, clips


def main(argv=None):
    ap = argparse.ArgumentParser(description="seed placeholder media so a bare checkout can build a walk-safe graph")
    ap.add_argument("--force", action="store_true",
                    help="overwrite media that already exists (destructive)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be written and exit")
    args = ap.parse_args(argv)

    import video_graph as vg                                     # noqa: E402

    stills = [(spec["image"], char, pose)
              for char, poses in vg.NODE_SPECS.items()
              for pose, spec in poses.items()]
    clips = [(char, label, kind)
             for (char, kind, label, *_rest) in vg.EDGE_SPECS]

    bt_stills, bt_clips = _bedtime_targets()
    stills += bt_stills
    clips += bt_clips

    # De-duplicate: a label carries many variants in production but one is enough
    # to make the edge exist, and the bedtime pass overlaps EDGE_SPECS on a rerun.
    seen_s, uniq_stills = set(), []
    for rel, char, pose in stills:
        if rel not in seen_s:
            seen_s.add(rel)
            uniq_stills.append((rel, char, pose))
    seen_c, uniq_clips = set(), []
    for char, label, kind in clips:
        if (char, label) not in seen_c:
            seen_c.add((char, label))
            uniq_clips.append((char, label, kind))

    tally = {"wrote": 0, "skip": 0, "would-write": 0}

    for rel, char, pose in uniq_stills:
        path = Path(rel) if Path(rel).is_absolute() else ROOT / rel
        r = _write_still(path, char, pose, force=args.force, dry=args.dry_run)
        tally[r] += 1
        if r != "skip":
            print("  still  %-46s %s" % (rel, r))

    for char, label, kind in uniq_clips:
        path = PROTO / ("%s_%s_v1.gif" % (char, label))
        r = _write_clip(path, char, label, kind, force=args.force, dry=args.dry_run)
        tally[r] += 1
        if r != "skip":
            print("  clip   %-46s %s" % (path.relative_to(ROOT).as_posix(), r))

    print("\n%d stills + %d clips declared; wrote %d, skipped %d (already present)%s" % (
        len(uniq_stills), len(uniq_clips), tally["wrote"], tally["skip"],
        ", %d would be written" % tally["would-write"] if args.dry_run else ""))

    if args.dry_run:
        return 0
    if tally["wrote"]:
        print("\nnext:  python runtime/video_graph.py build")
        print("       python scripts/export_context_view.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
