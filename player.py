"""
living-portraits -- frameless dual-panel player (v2)

One borderless window at desktop (0,0), sized to the bounding box of the two
LED panels, with two stages drawn into their exact sub-rects:

    Panel A : (0,   0) 256 x 256   <- LED screen 1
    Panel B : (256, 0) 192 x 192   <- LED screen 2

v2 consumes data/stage_state.json (written by the director) and performs the
current beat per panel: character name + action + the fourth-wall aside. Falls
back to a "waiting" card if no state yet. A small dot + clock prove liveness.

This text rendering is the dev view of the pipe; later versions swap it for
Live2D rig + baked clips while keeping this same state-consumption shell.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

# Set before pygame video init so SDL places the window at the desktop top-left.
os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
os.environ.setdefault("SDL_VIDEO_CENTERED", "0")

import pygame

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "data" / "stage_state.json"

# Stage renderer (real portrait + aside) lives in runtime/. Guarded so a broken
# or missing renderer falls back to the text dev-view rather than killing the show.
sys.path.insert(0, str(ROOT / "runtime"))
try:
    from stage_render import render_stage
    _HAVE_RENDER = True
except Exception as _e:
    print("stage_render import failed; text fallback:", _e, flush=True)
    _HAVE_RENDER = False

# Rig animator (blink / breathe / gaze). Guarded: if rig_loop or its deps are
# missing, RIGLOOP stays None and stage_render uses the static cutout/portrait.
PLAYER_T0 = time.monotonic()
try:
    from rig_loop import RigLoop
    RIGLOOP = RigLoop()
except Exception as _re:
    print("rig_loop import failed; static portrait, no animation:", _re, flush=True)
    RIGLOOP = None

# Cross-frame walk (the "walk out of one portrait, into the next" effect).
# Guarded exactly like the rig: if crossframe or numpy is missing, XSTATE stays
# None and the move branch in the loop is skipped entirely -- the show renders
# every panel through the normal draw_stage path, unchanged.
GEN_DIR = ROOT / "data" / "gen"
try:
    import crossframe
    XSTATE = crossframe.CrossFrameState()
except Exception as _xe:
    print("crossframe import failed; no cross-frame walk:", _xe, flush=True)
    crossframe = None
    XSTATE = None

# Decoded mover cutouts (native-res RGBA), keyed by slug. crossframe.exit_frame /
# enter_frame cover-resize this to each panel themselves, so we hold the native
# array and never re-decode per frame. None is cached too (missing/unreadable
# cutout) so a slug with no asset is probed at most once.
_CUTOUT_CACHE: dict[str, object] = {}


def _mover_cutout(slug):
    """Native-res RGBA for `slug`'s cutout, decoded once via stage_render's loader.

    Reuses stage_render._imread_rgba (the same BGRA->RGBA decode the static
    composite uses) rather than a private decoder, and caches the result -- the
    None miss included -- so the walk never re-reads the PNG each frame. Returns
    None when the cutout is missing/unreadable (caller then renders normally)."""
    if slug in _CUTOUT_CACHE:
        return _CUTOUT_CACHE[slug]
    rgba = None
    try:
        import stage_render as _sr  # sibling on sys.path; same module render_stage uses
        rgba = _sr._imread_rgba(GEN_DIR / (slug + "_cutout.png"))
    except Exception as exc:
        print("mover cutout decode failed; move falls back to normal:", exc, flush=True)
        rgba = None
    _CUTOUT_CACHE[slug] = rgba
    return rgba


def _move_signature(move):
    """(slug, from_panel, to_panel) for a beat's top-level `move`, or None if the
    move is absent/malformed. The slug is derived the SAME way as panels[*].char
    (stage_render.slugify) so the walk and the static cutout key on one slug."""
    if not isinstance(move, dict):
        return None
    char = move.get("char")
    frm = move.get("from")
    to = move.get("to")
    if not char or not frm or not to or frm == to:
        return None
    try:
        from stage_render import slugify
        slug = slugify(char)
    except Exception:
        return None
    if not slug or frm not in PANELS or to not in PANELS:
        return None
    return slug, frm, to


# Tracks the move signature XSTATE last began, so a `move` beat that lingers in
# stage_state.json across reloads is walked ONCE: a new signature starts a fresh
# walk; an already-walked one renders normally instead of looping forever.
_LAST_MOVE_SIG = None

PANELS = {
    "A": {"rect": (0, 0, 256, 256), "bg": (38, 14, 16), "accent": (210, 170, 90)},
    "B": {"rect": (256, 0, 192, 192), "bg": (12, 26, 28), "accent": (120, 200, 190)},
}
WINDOW_W = 448
WINDOW_H = 256
FPS = 30


def wrap_text(text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        test = (cur + " " + word).strip()
        if font.size(test)[0] <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def load_state(cache):
    """Reload stage_state.json only when its mtime changes."""
    try:
        m = STATE_PATH.stat().st_mtime
        if m != cache.get("mtime"):
            cache["mtime"] = m
            cache["data"] = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return cache.get("data")


def draw_stage_text(screen, name, panel, beat, frame, fonts):
    rect = pygame.Rect(*panel["rect"])
    sub = screen.subsurface(rect)
    w, h = rect.width, rect.height
    sub.fill(panel["bg"])
    pygame.draw.rect(sub, panel["accent"], sub.get_rect(), 2)
    pad = 8

    p = beat.get("panels", {}).get(name) if beat else None
    if p:
        title = fonts["bold"].render(p.get("char", "PANEL " + name), True, panel["accent"])
        sub.blit(title, (pad, pad))
        act = fonts["tiny"].render("[" + p.get("action", "idle") + "]", True, (150, 150, 150))
        sub.blit(act, (pad, pad + title.get_height() + 1))
        y = pad + title.get_height() + act.get_height() + 6
        for line in wrap_text(p.get("aside", ""), fonts["small"], w - 2 * pad):
            if y > h - 20:
                break
            sub.blit(fonts["small"].render(line, True, (235, 235, 235)), (pad, y))
            y += fonts["small"].get_linesize()
    else:
        msg = fonts["small"].render("PANEL " + name + " waiting...", True, (150, 150, 150))
        sub.blit(msg, (pad, pad))

    # liveness marker + clock
    cx = pad + int((w - 2 * pad) * (0.5 + 0.5 * math.sin(frame * 0.08)))
    pygame.draw.circle(sub, panel["accent"], (cx, h - 10), 3)
    clk = fonts["tiny"].render(time.strftime("%H:%M:%S"), True, (110, 110, 110))
    sub.blit(clk, (w - clk.get_width() - pad, h - clk.get_height() - 4))


def draw_stage(screen, name, panel, beat, frame, fonts):
    # Try the real renderer (portrait + aside); fall back to the text dev-view
    # on any error so the show never goes blank.
    if _HAVE_RENDER:
        try:
            beat_panel = beat.get("panels", {}).get(name) if beat else None
            rf = None
            if RIGLOOP is not None:
                try:
                    rf = RIGLOOP.rig_frame_for_beat(beat_panel, time.monotonic() - PLAYER_T0)
                except Exception as rexc:
                    print("rig_frame error; static portrait:", rexc, flush=True)
                    rf = None
            render_stage(screen, name, panel, beat_panel, frame, fonts, rig_frame=rf)
            return
        except Exception as exc:
            print("render_stage error; text fallback:", exc, flush=True)
    draw_stage_text(screen, name, panel, beat, frame, fonts)


def draw_move(screen, beat, frame, fonts, dt):
    """Render ALL panels for a cross-frame walk, if the current beat asks for one.

    The director adds a top-level ``move`` ({"char","from","to"}) to a beat to make
    a character walk OUT of one panel and IN to the other (see crossframe.py). When
    that's present and decodable this drives the whole frame: the from-panel plays
    EXIT, both panels show the empty GAP, the to-panel plays ENTER -- each as a
    figure layer handed to render_stage via ``rig_frame=`` (exactly the slot the rig
    uses), so the bg plate / alpha-over / text / liveness all come from stage_render
    unchanged. Panels not currently showing the mover get an empty (transparent)
    layer so they composite to bg-plate-only -- the "empty portrait" that sells the
    illusion.

    A new ``move`` signature starts a fresh walk (seeding occupancy from the beat);
    a signature that's already been walked keeps the mover suppressed on the FROM
    panel (it left there) while rendering the TO panel normally, until the director
    writes a new beat -- so the walk plays once, never loops.

    Returns True iff it handled the frame (caller then skips the normal per-panel
    draw); False when there is no usable move (no/invalid ``move``, crossframe
    unavailable, or the mover's cutout can't be decoded) so the caller renders
    normally. Fully guarded: any error returns False so the working show never
    breaks on a move bug.
    """
    global _LAST_MOVE_SIG
    if XSTATE is None or crossframe is None or not _HAVE_RENDER or not beat:
        return False
    try:
        sig = _move_signature(beat.get("move"))
        if sig is None:
            return False
        slug, frm, to = sig
        cutout = _mover_cutout(slug)
        if cutout is None:
            return False  # no figure to walk -> let the normal path render

        if sig != _LAST_MOVE_SIG:
            # A new move beat: seed who lives where (from the beat's own panel
            # chars) so panel_state can suppress correctly, then start the walk.
            from stage_render import slugify
            panels = beat.get("panels", {})
            fc = slugify((panels.get(frm) or {}).get("char", "")) or slug
            tc = slugify((panels.get(to) or {}).get("char", "")) or None
            XSTATE.set_occupant(frm, fc)
            XSTATE.set_occupant(to, tc)
            XSTATE.begin_transition(slug, frm, to)
            _LAST_MOVE_SIG = sig

        active = XSTATE.active
        if active:
            XSTATE.advance(dt)  # advance the walk clock by real elapsed seconds

        geom = {nm: PANELS[nm]["rect"] for nm in PANELS}
        exit_dir, enter_dir = crossframe.edge_for_move(frm, to, geom)

        for nm, panel in PANELS.items():
            rect = panel["rect"]
            beat_panel = beat.get("panels", {}).get(nm)
            ph, prog = XSTATE.panel_state(nm)
            if ph == crossframe.PHASE_EXIT:
                layer = crossframe.exit_frame(rect, cutout, prog, exit_dir)
                render_stage(screen, nm, panel, beat_panel, frame, fonts, rig_frame=layer)
            elif ph == crossframe.PHASE_ENTER:
                layer = crossframe.enter_frame(rect, cutout, prog, enter_dir)
                render_stage(screen, nm, panel, beat_panel, frame, fonts, rig_frame=layer)
            elif ph in (crossframe.PHASE_GAP, crossframe.PHASE_PRE):
                # The mover has left / not-yet-arrived here: empty (transparent)
                # layer -> bg plate only, no figure (the empty portrait).
                render_stage(screen, nm, panel, beat_panel, frame, fonts,
                             rig_frame=crossframe.empty_layer(rect))
            elif (not active) and nm == frm:
                # Walk finished but this move still lingers in the beat: the mover
                # has departed the FROM panel, so keep it empty there (it now lives
                # in TO, rendered normally below) until the director moves on.
                render_stage(screen, nm, panel, beat_panel, frame, fonts,
                             rig_frame=crossframe.empty_layer(rect))
            else:
                # PHASE_DONE on the to-panel / a panel not in the move: render
                # normally (this preserves the rig wiring in draw_stage).
                draw_stage(screen, nm, panel, beat, frame, fonts)
        return True
    except Exception as exc:
        print("draw_move error; normal per-panel render:", exc, flush=True)
        return False


def main():
    # Self-log so we can run under pythonw.exe (no console window on the desktop).
    sys.stdout = sys.stderr = open(ROOT / "player.log", "a", buffering=1, encoding="utf-8", errors="replace")
    print("living-portraits player v2 starting", flush=True)
    pygame.init()
    pygame.display.set_caption("living-portraits")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.NOFRAME)
    # Pin to (0,0) at exact size and keep above any console/desktop so the LED
    # region always shows the stage, never window chrome behind it.
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        hwnd = pygame.display.get_wm_info()["window"]
        user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), 0, 0, WINDOW_W, WINDOW_H, 0x0040)
    except Exception as exc:
        print("topmost failed:", exc, flush=True)
    fonts = {
        "bold": pygame.font.SysFont("georgia", 16, bold=True),
        "small": pygame.font.SysFont("georgia", 12),
        "tiny": pygame.font.SysFont("consolas", 10),
    }
    clock = pygame.time.Clock()
    print("window created at (0,0) size", (WINDOW_W, WINDOW_H), flush=True)

    cache = {}
    frame = 0
    last_tick = time.monotonic()  # for the cross-frame walk's real-seconds dt
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        beat = load_state(cache) if frame % 15 == 0 else cache.get("data")
        # dt = wall-clock seconds since the previous frame (monotonic). Capped so a
        # GC/IO stall can't skip a whole walk in one tick; crossframe also clamps.
        now = time.monotonic()
        dt = min(0.25, max(0.0, now - last_tick))
        last_tick = now
        screen.fill((0, 0, 0))
        # If the beat carries a cross-frame `move`, draw_move renders every panel
        # (EXIT / GAP / ENTER layers + the empty portrait). Otherwise -- and on any
        # move error -- it returns False and we render each panel normally.
        if not draw_move(screen, beat, frame, fonts, dt):
            for nm, panel in PANELS.items():
                draw_stage(screen, nm, panel, beat, frame, fonts)
        pygame.display.flip()
        frame += 1
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
