"""_preview_cycle.py -- TEMP on-panel VARIETY demo.

Each panel plays one loop clip a couple times, then jumps to a different clip from
its list at random -- the clip_graph idea: many idle loops, all anchored, chained for
endless non-repeating motion. Borderless + topmost at (0,0), same pin as player.py.
Reads GIFs via PIL. ESC quits. Stop lp-player/lp-preview first.

    pythonw _preview_cycle.py --a a1.gif a2.gif a3.gif [--b b1.gif ...] [--loops 2]
"""
import argparse
import os
import random
import sys
import traceback
from pathlib import Path

os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
os.environ.setdefault("SDL_VIDEO_CENTERED", "0")

ROOT = Path(__file__).resolve().parent
sys.stdout = sys.stderr = open(ROOT / "_preview.log", "a", buffering=1,
                               encoding="utf-8", errors="replace")

import pygame
from PIL import Image, ImageSequence

PANELS = {"A": (0, 0, 256, 256), "B": (256, 0, 192, 192)}
WINDOW_W, WINDOW_H = 448, 256
FPS = 10


def load_clip(path, size):
    im = Image.open(path)
    out = []
    for fr in ImageSequence.Iterator(im):
        rgb = fr.convert("RGB").resize(size, Image.LANCZOS)
        out.append(pygame.image.fromstring(rgb.tobytes(), rgb.size, "RGB"))
    return out


class Cycler:
    """Plays a clip `loops` times, then jumps to a random different clip."""
    def __init__(self, paths, size, loops, start=0):
        self.clips = [load_clip(p, size) for p in paths]
        self.loops = loops
        self.ci = min(start, len(self.clips) - 1)
        self.fi = 0
        self.lc = 0

    def frame(self):
        clip = self.clips[self.ci]
        surf = clip[self.fi]
        self.fi += 1
        if self.fi >= len(clip):
            self.fi = 0
            self.lc += 1
            if self.lc >= self.loops and len(self.clips) > 1:
                self.lc = 0
                nxt = self.ci
                while nxt == self.ci:
                    nxt = random.randrange(len(self.clips))
                self.ci = nxt
        return surf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="*", default=[])
    ap.add_argument("--loops", type=int, default=2)
    args = ap.parse_args()
    print("cycle preview:", args.a, "|", args.b, flush=True)

    pygame.init()
    pygame.display.set_caption("lp-preview-cycle")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.NOFRAME)
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        u.SetWindowPos(pygame.display.get_wm_info()["window"], ctypes.c_void_p(-1),
                       0, 0, WINDOW_W, WINDOW_H, 0x0040)
    except Exception as e:
        print("topmost failed:", e, flush=True)

    ra, rb = PANELS["A"], PANELS["B"]
    ca = Cycler(args.a, (ra[2], ra[3]), args.loops, start=0)
    cb = Cycler(args.b or args.a, (rb[2], rb[3]), args.loops, start=1)
    print("loaded A=%d clips, B=%d clips" % (len(ca.clips), len(cb.clips)), flush=True)

    clock = pygame.time.Clock()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        screen.fill((0, 0, 0))
        screen.blit(ca.frame(), (ra[0], ra[1]))
        screen.blit(cb.frame(), (rb[0], rb[1]))
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
