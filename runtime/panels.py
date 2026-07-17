"""panels.py -- load the LED-panel layout from panels.yaml for player.py.

This is the "panel config" box: the one place player.py learns its panel
geometry + palette + window size, so changing the physical layout (a different
desktop resolution, a different LED sending-card grab origin on a new host) is a
panels.yaml edit, not a code change. See ../panels.yaml for the file's shape and
the migration rationale.

The single public verb is:

    load_panels(path=None) -> (panels, window_w, window_h)

where `panels` is an ordered dict in the EXACT shape player.py's hardcoded
PANELS used -- {name: {"rect": (x, y, w, h), "bg": (r, g, b),
"accent": (r, g, b)}} -- with `rect`/`bg`/`accent` as tuples (player.py does
`pygame.Rect(*panel["rect"])`, crossframe unpacks `x, y, w, h = rect`, and the
draw path indexes panel["bg"]/["accent"]). `window_w`/`window_h` are DERIVED:
the bounding box (max x+w, max y+h) over every rect -- never read from the file.

The conductor wires it into player.py with one line, replacing the hardcoded
PANELS / WINDOW_W / WINDOW_H block:

    from panels import load_panels           # runtime/ already on sys.path
    PANELS, WINDOW_W, WINDOW_H = load_panels()

Degrade-to-today contract
-------------------------
`pyyaml` is import-guarded and EVERY failure path -- yaml missing, file missing,
file unreadable, malformed yaml, or a structurally-invalid layout -- returns the
hardcoded supercommons2 defaults (_DEFAULT_PANELS below, the literal values
player.py shipped with). So a broken or absent panels.yaml renders exactly
today's layout rather than crashing the show; a problem is announced on stderr
but never fatal. Per-panel validation is all-or-nothing: a single bad panel
falls the WHOLE file back to defaults rather than shipping a half-parsed layout
that would silently drop a portrait off a screen.

This module is pure-stdlib + (optional) pyyaml -- no numpy, no pygame, no
network -- so it imports anywhere. `python runtime/panels.py` runs the offline
self-test (shape match vs player.py + missing-file fallback).
"""
from __future__ import annotations

import sys
from pathlib import Path

# pyyaml is import-guarded: a box without it (or any yaml import error) falls
# back to the hardcoded layout below rather than failing to import. gallery.py
# treats yaml as a hard dep, but this loader's whole job is to degrade to a
# usable layout, so it must not hard-require it.
try:
    import yaml  # type: ignore
except Exception as _ye:  # pragma: no cover - pyyaml is present in the real repo
    yaml = None  # type: ignore
    print(f"[panels] pyyaml unavailable; using hardcoded layout: {_ye}", flush=True)

# Project root is one level up from runtime/ (mirrors stage_render/rig_loop ROOT).
ROOT = Path(__file__).resolve().parent.parent
PANELS_PATH = ROOT / "panels.yaml"

# The literal supercommons2 layout player.py shipped with. This is the fallback
# for EVERY failure path, and the self-test asserts a fresh panels.yaml round-
# trips back to exactly these values. Keep in lock-step with ../panels.yaml's
# seed block and (historically) player.py's hardcoded PANELS.
#   A  0,0   256x256  maroon/gold   <- LED screen 1 (larger frame)
#   B  256,0 192x192  teal/mint     <- LED screen 2 (smaller, brighter)
_DEFAULT_PANELS: "list[dict]" = [
    {"name": "A", "rect": (0, 0, 256, 256), "bg": (38, 14, 16),
     "accent": (210, 170, 90), "char": "phineas"},
    {"name": "B", "rect": (256, 0, 192, 192), "bg": (12, 26, 28),
     "accent": (120, 200, 190), "char": "seraphina"},
]


# --------------------------------------------------------------------------- #
# Validation helpers (each raises ValueError on a malformed field; the caller
# turns ANY raise into a wholesale fall-back to _DEFAULT_PANELS).
# --------------------------------------------------------------------------- #
def _as_rect(value) -> "tuple[int, int, int, int]":
    """Coerce a [x, y, w, h] sequence to a 4-int tuple. w/h must be positive
    (a zero/negative panel can't be drawn and would break the window bounds)."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"rect must be [x, y, w, h], got {value!r}")
    try:
        x, y, w, h = (int(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"rect values must be integers, got {value!r}")
    if w <= 0 or h <= 0:
        raise ValueError(f"rect w/h must be positive, got w={w} h={h}")
    if x < 0 or y < 0:
        raise ValueError(f"rect x/y must be >= 0, got x={x} y={y}")
    return (x, y, w, h)


def _as_rgb(value, field: str) -> "tuple[int, int, int]":
    """Coerce a [r, g, b] sequence to a 3-int tuple, each clamped to 0..255."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must be [r, g, b], got {value!r}")
    try:
        rgb = tuple(int(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} values must be integers, got {value!r}")
    if any(c < 0 or c > 255 for c in rgb):
        raise ValueError(f"{field} channels must be 0..255, got {value!r}")
    return rgb  # type: ignore[return-value]


def _normalize_panels(rows) -> "list[dict]":
    """Validate a list of raw panel dicts into the normalized internal shape.

    Returns a list of {name, rect (tuple), bg (tuple), accent (tuple), char}.
    Raises ValueError on ANY structural problem (not a list, empty, a row that
    isn't a dict, a missing/blank name, a duplicate name, or a bad rect/colour)
    so the caller falls the whole file back to the hardcoded layout.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("panels must be a non-empty list")
    out: "list[dict]" = []
    seen: "set[str]" = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"panel #{i} must be a mapping, got {type(row).__name__}")
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError(f"panel #{i} is missing a name")
        if name in seen:
            raise ValueError(f"duplicate panel name {name!r}")
        seen.add(name)
        char = row.get("char")
        out.append({
            "name": name,
            "rect": _as_rect(row.get("rect")),
            "bg": _as_rgb(row.get("bg"), "bg"),
            "accent": _as_rgb(row.get("accent"), "accent"),
            # char is an optional per-panel default slug (lowercased to match the
            # pipeline's <slug> asset keys); None when the file omits it.
            "char": (str(char).strip().lower() or None) if char else None,
        })
    return out


def _window_box(panels: "list[dict]") -> "tuple[int, int]":
    """The window size = the bounding box of every panel rect: (max x+w, max y+h).

    The player pins its window at the desktop top-left (0,0) and sizes it to this
    box, so each panel rect is a sub-rect a sending card grabs. Mirrors player.py
    shipping WINDOW_W=448 / WINDOW_H=256 for the A(256)+B(256,0,192,192) layout.
    """
    w = max(x + rw for (x, _y, rw, _rh) in (p["rect"] for p in panels))
    h = max(y + rh for (_x, y, _rw, rh) in (p["rect"] for p in panels))
    return int(w), int(h)


def _build(panels: "list[dict]") -> "tuple[dict, int, int]":
    """Turn the normalized list into player.py's (PANELS dict, WINDOW_W, WINDOW_H).

    The dict preserves list order (insertion-ordered) so the draw order matches
    the file; each value carries rect/bg/accent (the keys player.draw_stage
    reads) plus char (the per-panel default, ignored by the existing draw path).
    """
    window_w, window_h = _window_box(panels)
    panels_dict = {
        p["name"]: {
            "rect": p["rect"],
            "bg": p["bg"],
            "accent": p["accent"],
            "char": p["char"],
        }
        for p in panels
    }
    return panels_dict, window_w, window_h


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_panels(path: "Path | str | None" = None) -> "tuple[dict, int, int]":
    """Load the panel layout, returning (panels, window_w, window_h).

    `panels` is the dict player.py used as PANELS:
        {name: {"rect": (x, y, w, h), "bg": (r, g, b), "accent": (r, g, b),
                "char": slug | None}}
    with rect/bg/accent as TUPLES (player.py + crossframe rely on tuples), and
    `char` an extra per-panel default slug the existing draw path ignores.
    `window_w`/`window_h` are the bounding box over all rects.

    `path` defaults to ../panels.yaml. ANY failure -- pyyaml missing, file
    missing/unreadable, malformed yaml, or a structurally-invalid layout --
    falls back to the hardcoded supercommons2 defaults (a stderr note is
    printed, but the call never raises), so the show always gets a usable layout.
    """
    p = Path(path) if path is not None else PANELS_PATH

    # No yaml, or the file is absent: silent-but-logged fall back to defaults.
    if yaml is None:
        return _build(_DEFAULT_PANELS)
    if not p.exists():
        print(f"[panels] {p} not found; using hardcoded layout", flush=True)
        return _build(_DEFAULT_PANELS)

    try:
        raw = p.read_text(encoding="utf-8-sig")  # tolerate a BOM, like gallery.py
        data = yaml.safe_load(raw) or {}
        # Accept either {panels: [...]} (canonical) or a bare top-level list.
        rows = data.get("panels") if isinstance(data, dict) else data
        panels = _normalize_panels(rows)
        return _build(panels)
    except Exception as exc:
        # Malformed file or invalid layout: announce it, then degrade to today's
        # hardcoded layout rather than crash the player on a config typo.
        print(f"[panels] {p} unreadable/invalid ({type(exc).__name__}: {exc}); "
              f"using hardcoded layout", flush=True)
        return _build(_DEFAULT_PANELS)


# --------------------------------------------------------------------------- #
# Self-test -- OFFLINE. Asserts the default-file values round-trip, the returned
# dict matches player.py's PANELS shape (keys A/B, rect tuples, bg/accent), the
# window box is computed right, and a missing file falls back to the same dict.
# No pygame, no numpy, no network.
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    import tempfile

    print("=" * 78)
    print("panels.py self-test  (OFFLINE; no pygame / numpy / network)")
    print(f"  pyyaml={'yes' if yaml is not None else 'no(guarded)'}")
    print("=" * 78)
    failures: "list[str]" = []

    def check(cond, msg):
        print(f"  [{'OK ' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    # The shape player.py's hardcoded block had, asserted field-by-field below.
    expected = {
        "A": {"rect": (0, 0, 256, 256), "bg": (38, 14, 16), "accent": (210, 170, 90)},
        "B": {"rect": (256, 0, 192, 192), "bg": (12, 26, 28), "accent": (120, 200, 190)},
    }

    # --- 1. the SHIPPED panels.yaml loads to exactly player.py's values --------
    print("\n-- load the project's panels.yaml --")
    panels, ww, wh = load_panels()
    check(list(panels.keys()) == ["A", "B"],
          f"panel order is A, B (insertion-ordered), got {list(panels.keys())}")
    for nm, exp in expected.items():
        pan = panels.get(nm, {})
        check(isinstance(pan.get("rect"), tuple) and pan.get("rect") == exp["rect"],
              f"panel {nm} rect is a tuple == {exp['rect']} (got {pan.get('rect')!r})")
        check(isinstance(pan.get("bg"), tuple) and pan.get("bg") == exp["bg"],
              f"panel {nm} bg is a tuple == {exp['bg']}")
        check(isinstance(pan.get("accent"), tuple) and pan.get("accent") == exp["accent"],
              f"panel {nm} accent is a tuple == {exp['accent']}")
    # rect must be a 4-int tuple (pygame.Rect(*rect) / x,y,w,h unpack rely on it)
    check(all(len(p["rect"]) == 4 and all(isinstance(v, int) for v in p["rect"])
              for p in panels.values()),
          "every rect is a 4-int tuple (pygame.Rect(*rect) / x,y,w,h unpack)")
    # per-panel default char came through from the file
    check(panels["A"].get("char") == "phineas" and panels["B"].get("char") == "seraphina",
          "per-panel default char: A=phineas, B=seraphina")

    # --- 2. window box == bounding box of the rects (player.py: 448 x 256) -----
    print("\n-- derived window box --")
    check((ww, wh) == (448, 256),
          f"WINDOW_W, WINDOW_H == (448, 256) from the A+B bounds (got {(ww, wh)})")

    # --- 3. missing file -> hardcoded fallback (degrade-to-today contract) -----
    print("\n-- missing file falls back to the hardcoded layout --")
    missing = Path(tempfile.gettempdir()) / "lp_panels_does_not_exist_zzz.yaml"
    if missing.exists():  # paranoia: never test against a real file
        missing = missing.with_suffix(".unique.yaml")
    fp, fw, fh = load_panels(missing)
    check(list(fp.keys()) == ["A", "B"], "fallback still yields panels A, B")
    check((fw, fh) == (448, 256), "fallback window box == (448, 256)")
    check(fp["A"]["rect"] == (0, 0, 256, 256) and fp["B"]["rect"] == (256, 0, 192, 192),
          "fallback rects == player.py's hardcoded values")
    # the fallback dict is shape-identical to the on-disk load (rect/bg/accent)
    same_shape = all(
        fp[nm]["rect"] == panels[nm]["rect"]
        and fp[nm]["bg"] == panels[nm]["bg"]
        and fp[nm]["accent"] == panels[nm]["accent"]
        for nm in ("A", "B")
    )
    check(same_shape, "missing-file fallback matches the on-disk load (rect/bg/accent)")

    # --- 4. malformed / invalid files also fall back (not crash) ---------------
    if yaml is not None:
        print("\n-- malformed + invalid files degrade to the hardcoded layout --")
        with tempfile.TemporaryDirectory() as td:
            bad_yaml = Path(td) / "broken.yaml"
            bad_yaml.write_text("panels: [this is : not : valid", encoding="utf-8")
            bp, bw, bh = load_panels(bad_yaml)
            check(list(bp.keys()) == ["A", "B"] and (bw, bh) == (448, 256),
                  "malformed yaml -> hardcoded layout (no crash)")

            # a structurally-invalid layout (bad rect length) must also fall back
            bad_rect = Path(td) / "badrect.yaml"
            bad_rect.write_text(
                "version: 1\npanels:\n  - name: A\n    rect: [0, 0, 256]\n"
                "    bg: [1,2,3]\n    accent: [4,5,6]\n", encoding="utf-8")
            rp, rw, rh = load_panels(bad_rect)
            check(rp["A"]["rect"] == (0, 0, 256, 256),
                  "invalid rect (len 3) -> whole file falls back to defaults")

            # a VALID custom layout loads as-is + derives the right window box ---
            print("\n-- a valid custom layout loads + derives its own window box --")
            custom = Path(td) / "custom.yaml"
            custom.write_text(
                "version: 1\npanels:\n"
                "  - name: L\n    rect: [0, 0, 300, 200]\n"
                "    bg: [10, 20, 30]\n    accent: [200, 200, 200]\n    char: phineas\n"
                "  - name: R\n    rect: [300, 0, 100, 100]\n"
                "    bg: [40, 50, 60]\n    accent: [90, 90, 90]\n", encoding="utf-8")
            cp, cw, ch = load_panels(custom)
            check(list(cp.keys()) == ["L", "R"], "custom panel names + order honoured (L, R)")
            check(cp["L"]["rect"] == (0, 0, 300, 200) and cp["R"]["rect"] == (300, 0, 100, 100),
                  "custom rects parsed as tuples")
            check((cw, ch) == (400, 200),
                  f"custom window box == (400, 200) from bounds (got {(cw, ch)})")
            check(cp["R"].get("char") is None, "omitted char defaults to None")

    print("=" * 78)
    if failures:
        print(f"SELF-TEST FAILED -- {len(failures)} check(s) failed:")
        for f in failures:
            print(f"   - {f}")
        print("=" * 78)
        return 1
    print("self-test OK -- panels.yaml loads to player.py's exact shape (A/B, rect")
    print("tuples, bg/accent), window box derived (448x256), and missing/malformed/")
    print("invalid files all degrade to the hardcoded layout.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
