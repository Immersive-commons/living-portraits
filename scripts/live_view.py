#!/usr/bin/env python3
"""live_view.py -- watch the walker think, from a browser, on any OS.

The viewer already renders live state. `context_view.json` carries a `now`
section per character -- current node, dwell, hop distances, the frontier, and
the Decision lens's real weights and shares -- and `graph_viewer.html` draws it.
What was missing is that nothing re-ran the export while the walker walked, so
the page showed whatever was true the last time someone typed a command. That is
the whole reason it reads as static.

This runs the three pieces together:

    walker (headless)  ->  writes data/mind/pose/<char>.json + the lived record
    exporter (on a timer) ->  reads those into data/graph/context_view.json
    http server        ->  serves the repo root, which the page polls

WHAT THIS IS NOT. It does not replace the wall. The panel player is Windows-first
because it pins a borderless SDL window at desktop origin for an LED sending card
to grab sub-rects out of -- that is hardware plumbing, and it is the only part of
this system that needs Windows. The WALKER is portable, and `SDL_VIDEODRIVER=dummy`
runs the real `_preview_graph.py` with the same flags the `lp-preview` scheduled
task uses. What you see here is production logic, not a simulation of it.

THE INVARIANT THIS RESPECTS. Every shared file under `data/` has exactly one
writer. The walker writes `pose/<char>.json` and the lived record; this script
never touches them. The exporter is read-only over production by contract -- it
never calls `build()`, never writes back into `video_graph.json`, never touches a
journal -- so running it on a timer adds no writer. If you change this file, keep
it that way: the concurrency design has no lock, and it is safe only because of
that rule.

    python scripts/live_view.py                  # walker + export every 2s + :8000
    python scripts/live_view.py --seconds 120    # stop after two minutes
    python scripts/live_view.py --no-serve       # if you are already serving

Ctrl-C stops everything it started, and nothing else.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / "scripts" / "export_context_view.py"
WALKER = ROOT / "_preview_graph.py"
VIEW = ROOT / "data" / "graph" / "context_view.json"


def _walker_cmd(a: str, b: str) -> list[str]:
    """The production flags, verbatim from the lp-preview task (ARCHITECTURE.md §1).

    `--policy` is what makes this the real decision path rather than a random
    walk: it routes every pick through runtime/policy.weigh.
    """
    return [sys.executable, str(WALKER),
            "--a", a, "--b", b,
            "--loops", "1", "--a-tp", "0.3", "--b-pose", "idle", "--b-tp", "0.4",
            "--policy"]


def _start_walker(a: str, b: str) -> subprocess.Popen | None:
    """Launch the real walker headless. Returns None with a printed reason on failure."""
    if not WALKER.exists():
        print(f"walker not found at {WALKER}")
        return None
    # The dummy video driver is what makes this portable. pygame still opens a
    # "display" and blits every frame -- the render path is fully exercised, it
    # just goes to a memory buffer instead of an LED sending card.
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    try:
        w = subprocess.Popen(_walker_cmd(a, b), cwd=str(ROOT), env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError as e:
        print(f"could not start the walker: {e}")
        return None
    time.sleep(1.5)                          # let it build its graph and first frame
    if w.poll() is not None:
        err = (w.stderr.read() or b"").decode(errors="replace").strip()
        print("the walker exited immediately:\n" + (err[-1200:] or "(no output)"))
        print("\nIf that is a pygame/SDL error, the walker needs `pip install pygame`.")
        return None
    print(f"walker      {a} | {b}   (headless, --policy)")
    return w


# `python -m http.server` answers / with a directory listing of the repo root,
# which is a poor front door: the viewer is one path deeper and nothing says so.
# This is the same server with one redirect, so the bare URL lands on the page.
_SERVER = """
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/graph_viewer.html")
            self.end_headers()
            return
        SimpleHTTPRequestHandler.do_GET(self)

    def log_message(self, *a):
        pass

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def _start_server(port: int) -> subprocess.Popen | None:
    """Start the viewer server, or say why it could not bind.

    stderr is kept, not discarded: the common failure here is `Address already in
    use` from a server an earlier run orphaned, and swallowing that turns a
    one-line fix into a mystery -- the script just exits with nothing to go on.
    """
    s = subprocess.Popen([sys.executable, "-c", _SERVER, str(port)],
                         cwd=str(ROOT), stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    time.sleep(0.6)
    if s.poll() is not None:
        err = (s.stderr.read() or b"").decode(errors="replace").strip()
        print(f"could not serve on port {port}:\n  {err.splitlines()[-1] if err else '(no output)'}")
        if "Address already in use" in err:
            print(f"  something is already on {port}. Free it, or pass --port <other>.")
        return None
    print(f"server      http://127.0.0.1:{port}/   (/ redirects to the viewer)")
    return s


def _export_once(n: int) -> None:
    """Re-read the walker's state into the viewer's JSON. Read-only by contract."""
    r = subprocess.run([sys.executable, str(EXPORT)], cwd=str(ROOT),
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print(f"  export failed: {(r.stderr or '').strip()[-200:]}")
        return
    lines = (r.stdout or "").strip().splitlines()
    tail = lines[-1] if lines else ""
    # Overwrite in place at a terminal; one line each when piped to a file or a
    # CI log, where \r would collapse the whole run onto one line.
    if sys.stdout.isatty():
        print(f"\r  export #{n}  {tail[-96:]:<96}", end="", flush=True)
    else:
        print(f"  export #{n}  {tail}", flush=True)


def _died(procs: list[tuple[str, subprocess.Popen]]) -> str | None:
    for name, proc in procs:
        if proc.poll() is not None:
            return f"{name} exited ({proc.returncode})"
    return None


def _stop(procs: list[tuple[str, subprocess.Popen]]) -> None:
    for name, proc in procs:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"stopped {name}")


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", default="phineas", help="character on panel A")
    ap.add_argument("--b", default="maxx", help="character on panel B")
    ap.add_argument("--every", type=float, default=2.0,
                    help="seconds between exports (default 2)")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after N seconds (default: run until Ctrl-C)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-serve", action="store_true",
                    help="do not start an http server; something else is serving the root")
    ap.add_argument("--no-walker", action="store_true",
                    help="only re-export; a walker is already running elsewhere")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    VIEW.parent.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[str, subprocess.Popen]] = []
    if not args.no_walker:
        w = _start_walker(args.a, args.b)
        if w is None:
            return 1
        procs.append(("walker", w))
    if not args.no_serve:
        srv = _start_server(args.port)
        if srv is None:
            _stop(procs)
            return 1
        procs.append(("server", srv))

    print(f"exporting   every {args.every:g}s -- reload the page to see it move\n")

    started, n = time.time(), 0
    try:
        while not (args.seconds and time.time() - started >= args.seconds):
            gone = _died(procs)
            if gone:
                print(f"\n{gone}; stopping.")
                break
            n += 1
            _export_once(n)
            time.sleep(args.every)
        else:
            print("\nreached --seconds; stopping.")
    except KeyboardInterrupt:
        print()
    finally:
        _stop(procs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
