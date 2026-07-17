"""_preview_panels.py -- TEMP on-panel compare: clip A in panel A, clip B in panel B.

Plays two looped GIFs side by side in the real panel geometry, borderless + topmost
at desktop (0,0) -- the SAME window pin as player.py -- so they render on console
session 1 and reach the LED card. Reads GIFs via PIL (no imageio-ffmpeg needed).

Throwaway: the production player keeps consuming stage_state.json. Stop lp-player
before running this (two pinned (0,0) windows would fight). ESC quits.

    pythonw _preview_panels.py --a <gifA> --la SVD --b <gifB> --lb AnimateDiff
"""
import argparse
import os
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


def load_gif_surfaces(path, size):
    im = Image.open(path)
    surfs = []
    for fr in ImageSequence.Iterator(im):
        rgb = fr.convert("RGB").resize(size, Image.LANCZOS)
        surfs.append(pygame.image.fromstring(rgb.tobytes(), rgb.size, "RGB"))
    return surfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--la", default="A")
    ap.add_argument("--b", required=True)
    ap.add_argument("--lb", default="B")
    args = ap.parse_args()
    print("preview starting:", args.a, "|", args.b, flush=True)

    pygame.init()
    pygame.display.set_caption("lp-preview")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.NOFRAME)
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        hwnd = pygame.display.get_wm_info()["window"]
        user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), 0, 0, WINDOW_W, WINDOW_H, 0x0040)
    except Exception as exc:
        print("topmost failed:", exc, flush=True)

    ra, rb = PANELS["A"], PANELS["B"]
    fa = load_gif_surfaces(args.a, (ra[2], ra[3]))
    fb = load_gif_surfaces(args.b, (rb[2], rb[3]))
    print("frames A=%d B=%d" % (len(fa), len(fb)), flush=True)
    font = pygame.font.SysFont("consolas", 12, bold=True)
    la = font.render(args.la, True, (255, 240, 180))
    lb = font.render(args.lb, True, (180, 240, 230))

    clock = pygame.time.Clock()
    ia = ib = 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        screen.fill((0, 0, 0))
        if fa:
            screen.blit(fa[ia % len(fa)], (ra[0], ra[1])); ia += 1
        if fb:
            screen.blit(fb[ib % len(fb)], (rb[0], rb[1])); ib += 1
        screen.blit(la, (ra[0] + 4, ra[1] + 4))
        screen.blit(lb, (rb[0] + 4, rb[1] + 4))
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
