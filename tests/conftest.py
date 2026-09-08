"""
conftest.py -- shared fixtures + import wiring for the living-portraits test suite.

GOAL: `python -m pytest tests/` works from the project root, fully OFFLINE, and
degrades (skip, never error) on a bare box missing numpy / cv2 / yaml.

sys.path
--------
The runtime modules import their siblings with BARE imports the way player.py
wires them at run time:
    runtime/stage_render.py  ->  from clip_player import ...
    runtime/rig_loop.py      ->  from rig import Rig ; from stage_render import slugify
    director/signals.py      ->  import feeds            (director/ on sys.path)
    director/feeds.py         ->  from gcal.client import ...  (repo root on sys.path)
So we prepend the project root, runtime/, and director/ to sys.path here, once,
before any test module imports them. This mirrors player.py's
`sys.path.insert(0, str(ROOT / "runtime"))` discipline.

Dependency degradation
-----------------------
numpy / cv2 / yaml are probed once and exposed as booleans (HAVE_NUMPY etc.) plus
ready-made skip markers (needs_numpy, needs_cv2, needs_yaml). A module that needs
a dep it lacks is SKIPPED, so the suite reports honest skip counts instead of
erroring on a box without the heavy deps. pygame is intentionally NOT required --
the whole render path is exercised headless via clip_player.DummySurface.

Isolation
---------
The feeds / signals fixtures repoint each module's data-dir + cache CONSTANTS at a
per-test tmp dir, so no test ever reads or writes the real data/feed.json or
data/mood.json, and the network fetchers are monkeypatched to deterministic stubs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# sys.path: project root + runtime/ + director/ (idempotent; front of path).
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
DIRECTOR_DIR = PROJECT_ROOT / "director"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"

for _p in (PROJECT_ROOT, RUNTIME_DIR, DIRECTOR_DIR, PIPELINE_DIR):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


# --------------------------------------------------------------------------- #
# Dependency probes + skip markers. Probed once at collection.
# --------------------------------------------------------------------------- #
def _probe(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except Exception:
        return False


HAVE_NUMPY = _probe("numpy")
HAVE_CV2 = _probe("cv2")
HAVE_YAML = _probe("yaml")

needs_numpy = pytest.mark.skipif(not HAVE_NUMPY, reason="numpy not installed")
needs_cv2 = pytest.mark.skipif(not HAVE_CV2, reason="cv2 (opencv) not installed")
needs_yaml = pytest.mark.skipif(not HAVE_YAML, reason="pyyaml not installed")


# --------------------------------------------------------------------------- #
# Synthetic image fixtures (numpy). Reuse each module's OWN synth helpers where
# they exist so the tests exercise the true schema/shape, not a re-implementation.
# --------------------------------------------------------------------------- #
@pytest.fixture
def synth_cutout_rgba():
    """A (256,256,4) RGBA cutout from rig.py's own _synth_cutout (figure on
    transparency, distinct dark eyes + mouth). Requires cv2 (rig.py draws with it)."""
    if not (HAVE_NUMPY and HAVE_CV2):
        pytest.skip("synth cutout needs numpy + cv2")
    import rig  # runtime/ is on sys.path
    return rig._synth_cutout(256)


@pytest.fixture
def crossframe_cutout():
    """crossframe.py's own off-centre synth cutout (centroid left-of-centre, so
    direction asserts are unambiguous). numpy-only."""
    if not HAVE_NUMPY:
        pytest.skip("needs numpy")
    import crossframe
    return crossframe._synth_cutout(256)


@pytest.fixture
def synth_face():
    """verify.py's deterministic synthetic 'portrait' (gradient bg + face blobs).
    Returns a builder so a test can request shifted/scaled poses."""
    if not (HAVE_NUMPY and HAVE_CV2):
        pytest.skip("synth face needs numpy + cv2")
    import verify
    return verify._synthetic_face


@pytest.fixture
def gen_dir(tmp_path):
    """An empty per-test data/gen-shaped dir. Tests that want assets write
    <slug>_*.png into it; the StageRenderer/Rig are pointed here, never at the
    real data/gen."""
    d = tmp_path / "gen"
    d.mkdir()
    return d


@pytest.fixture
def written_rig(gen_dir):
    """Write rig.py's synth cutout + rig-spec under a slug into gen_dir and return
    (slug, gen_dir). Lets rig_loop / Rig.from_slug build a real Rig offline.
    Skips cleanly if numpy/cv2 are absent (Rig cannot decode without cv2)."""
    if not (HAVE_NUMPY and HAVE_CV2):
        pytest.skip("rig fixture needs numpy + cv2")
    import json

    import cv2
    import rig

    slug = "synthhero"
    cut = rig._synth_cutout(256)
    spec = rig._synth_spec_for(cut)
    # Cutout written BGRA (rig.py reads it back and converts BGRA->RGBA on load).
    cv2.imwrite(str(gen_dir / f"{slug}_cutout.png"),
                cv2.cvtColor(cut, cv2.COLOR_RGBA2BGRA))
    (gen_dir / f"{slug}_rig.json").write_text(json.dumps(spec), encoding="utf-8")
    return slug, gen_dir


@pytest.fixture
def written_portrait(gen_dir):
    """Write stage_render.py's synth portrait + cutout under a slug into gen_dir.
    Returns (slug, gen_dir). For composite_art / StageRenderer art-present tests."""
    if not (HAVE_NUMPY and HAVE_CV2):
        pytest.skip("portrait fixture needs numpy + cv2")
    import cv2
    import stage_render as sr

    slug = "_test"
    portrait = sr._synthetic_portrait(512)
    cutout = sr._synthetic_cutout(512)
    cv2.imwrite(str(gen_dir / f"{slug}_portrait.png"),
                cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(gen_dir / f"{slug}_cutout.png"),
                cv2.cvtColor(cutout, cv2.COLOR_RGBA2BGRA))
    return slug, gen_dir
