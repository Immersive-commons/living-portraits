#!/usr/bin/env python3
"""e2e_viewer.py -- drive graph_viewer.html in a real browser and report what breaks.

Run with polyfetch's interpreter so patchright + its Chromium are on hand:

    uv run --directory <polyfetch> python /abs/path/e2e_viewer.py <base-url> <out-dir>

WHY THIS EXISTS. graph_viewer.html is the one surface nothing tested. It holds no
Python, so pytest never sees it, and `curl` reports a healthy 200 on a page whose
JavaScript died before drawing anything. It did exactly that: one TypeError on a
fresh clone killed the boot, taking the detail pane and three of the four tabs
with it, and several sessions of 200s went past without noticing.

AGENTS.md already says "probe the DOM, do not eyeball a screenshot." That argues
against TRUSTING a screenshot, not against taking one -- taking one is how you
learn there is something to probe. This does both: it drives the page, then
asserts on what the browser itself reports.

WHAT IT DOES, per viewport: loads the page, clicks every lens, every tab, every
checkbox and every button, drags the provenance scrubber and clicks a node on the
canvas -- screenshotting each state and draining the browser's console errors,
uncaught JS exceptions and failed requests after every interaction.

NOT part of `pytest tests/`. It needs a browser and a running server, so it is an
operator/CI tool in the shape of `health/`, run against a URL you already serve.

    python scripts/live_view.py --port 8000 &
    uv run --directory <polyfetch> python "$PWD/scripts/e2e_viewer.py" \
        http://127.0.0.1:8000 ./e2e_out

Exits non-zero when it finds anything, so CI can gate on it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from polyfetch_scrape.render_session import render_session

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "./e2e_out")
OUT.mkdir(parents=True, exist_ok=True)

VIEWPORTS = [
    ("desktop-landscape", 1440, 900),
    ("tablet-portrait", 834, 1112),
    ("phone-portrait", 390, 844),
    ("phone-landscape", 844, 390),
]
LENSES = ["Structure", "Lived", "Provenance", "Manner", "Frontier", "Decision"]
TABS = ["graph", "memory", "codemap", "honesty"]

findings: list[dict] = []


def note(vp, step, kind, detail):
    findings.append({"viewport": vp, "step": step, "kind": kind, "detail": str(detail)[:300]})
    print(f"  [{kind:<9}] {vp}/{step}: {str(detail)[:150]}")


def shoot(page, vp, name):
    p = OUT / f"{vp}__{name}.png"
    try:
        page.screenshot(path=str(p))
    except Exception as e:      # a screenshot failing is itself a finding
        note(vp, name, "shot-fail", e)


def drain(s, vp, step, seen):
    """Report console errors / JS exceptions / failed requests new since last check."""
    for e in s.console_errors[seen["c"]:]:
        note(vp, step, "console", e)
    seen["c"] = len(s.console_errors)
    for f in s.network_failures[seen["n"]:]:
        note(vp, step, "network", f)
    seen["n"] = len(s.network_failures)


def _check_furniture(page, vp):
    for sel, label in [("#lenses", "lens rail"), ("#net", "graph canvas"),
                       ("#detail", "detail pane"), ("#now", "now strip")]:
        try:
            if page.locator(sel).count() == 0:
                note(vp, "structure", "missing", f"{label} ({sel}) not in DOM")
        except Exception as e:
            note(vp, "structure", "error", f"{label}: {e}")


def _check_detail_pane(page, vp):
    """An error message RENDERED into the pane is the failure a 200 hides."""
    try:
        txt = page.locator("#detail").inner_text(timeout=3000)
        for bad in ("TypeError", "ReferenceError", "undefined is not", "No context view"):
            if bad in txt:
                note(vp, "detail-pane", "page-error", txt.strip().splitlines()[0:3])
                return
    except Exception as e:
        note(vp, "detail-pane", "error", e)


def _click_lenses(s, page, vp, seen):
    for i, lens in enumerate(LENSES, 1):
        try:
            page.get_by_text(lens, exact=True).first.click(timeout=4000)
            page.wait_for_timeout(700)
            drain(s, vp, f"lens:{lens}", seen)
            shoot(page, vp, f"{i:02d}-lens-{lens.lower()}")
        except Exception as e:
            note(vp, f"lens:{lens}", "click-fail", e)


def _click_tabs(s, page, vp, seen):
    for t in TABS:
        try:
            page.locator(f'[data-tab="{t}"]').first.click(timeout=4000)
            page.wait_for_timeout(700)
            drain(s, vp, f"tab:{t}", seen)
            shoot(page, vp, f"10-tab-{t}")
            panel = page.locator(f"#tab-{t}")
            body = panel.inner_text(timeout=3000) if panel.count() else ""
            if body.strip() == "":
                note(vp, f"tab:{t}", "empty", "tab panel rendered no text")
        except Exception as e:
            note(vp, f"tab:{t}", "click-fail", e)


def _toggle_checkboxes(s, page, vp, seen):
    for sel, name in [("#showidles", "show-idle-self-loops"), ("#showlabels", "edge-labels")]:
        try:
            page.locator(sel).click(timeout=4000)
            page.wait_for_timeout(600)
            drain(s, vp, name, seen)
            shoot(page, vp, f"20-{name}")
            page.locator(sel).click(timeout=4000)          # restore
            page.wait_for_timeout(300)
        except Exception as e:
            note(vp, name, "click-fail", e)


def _drag_scrubber(s, page, vp, seen):
    """The provenance scrubber replays the graph accreting node by node."""
    try:
        page.get_by_text("Provenance", exact=True).first.click(timeout=3000)
        page.wait_for_timeout(500)
        page.locator("#scrub").fill("400")
        page.locator("#scrub").dispatch_event("input")
        page.wait_for_timeout(800)
        drain(s, vp, "scrubber", seen)
        shoot(page, vp, "30-scrubber-mid")
    except Exception as e:
        note(vp, "scrubber", "fail", e)


def _click_buttons(s, page, vp, seen):
    for sel, name in [("#play", "grow-button"), ("#relayout", "re-layout-button")]:
        try:
            page.locator(sel).click(timeout=4000)
            page.wait_for_timeout(1200)
            drain(s, vp, name, seen)
            shoot(page, vp, f"40-{name}")
        except Exception as e:
            note(vp, name, "click-fail", e)


def _click_canvas(s, page, vp, seen):
    """vis-network draws to a canvas, so a node is only clickable by position."""
    try:
        box = page.locator("#net").bounding_box()
        if box:
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.wait_for_timeout(700)
            drain(s, vp, "canvas-click", seen)
            shoot(page, vp, "50-canvas-click")
    except Exception as e:
        note(vp, "canvas-click", "fail", e)


def _to_graph_tab(page, vp):
    try:
        page.locator('[data-tab="graph"]').first.click(timeout=4000)
        page.wait_for_timeout(400)
    except Exception as e:
        note(vp, "tab:graph", "click-fail", e)


def exercise(s, vp):
    page = s.page
    seen = {"c": 0, "n": 0}
    page.wait_for_timeout(1800)                    # let vis-network settle
    drain(s, vp, "load", seen)
    shoot(page, vp, "00-load")

    _check_furniture(page, vp)
    _check_detail_pane(page, vp)
    _click_lenses(s, page, vp, seen)
    _click_tabs(s, page, vp, seen)
    _to_graph_tab(page, vp)
    _toggle_checkboxes(s, page, vp, seen)
    _to_graph_tab(page, vp)
    _drag_scrubber(s, page, vp, seen)
    _click_buttons(s, page, vp, seen)
    _click_canvas(s, page, vp, seen)


for vp_name, w, h in VIEWPORTS:
    print(f"\n=== {vp_name}  {w}x{h} ===")
    try:
        vid = str(OUT / "video") if vp_name == "desktop-landscape" else None
        with render_session(f"{BASE}/graph_viewer.html",
                            viewport=(w, h),
                            wait_until="networkidle",
                            record_video_dir=vid,
                            record_video_size=(w, h) if vid else None) as s:
            exercise(s, vp_name)
        if vid:
            print(f"  video -> {getattr(s, 'video_path', None)}")
    except Exception as e:
        note(vp_name, "session", "fatal", e)

(OUT / "findings.json").write_text(json.dumps(findings, indent=2))
print(f"\n{'=' * 60}\n{len(findings)} findings -> {OUT}/findings.json")
kinds: dict[str, int] = {}
for f in findings:
    kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
    print(f"  {v:>3}  {k}")
print(f"  screenshots: {len(list(OUT.glob('*.png')))}")

# Non-zero on any finding: this is a gate, not a report.
sys.exit(1 if findings else 0)
