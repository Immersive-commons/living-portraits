#!/usr/bin/env python3
"""
run_all.py -- a no-pytest fallback runner for the living-portraits test suite.

`python -m pytest tests/` is the primary entry point. This script exists so the
suite is still runnable on a box where pytest is NOT installed: it discovers the
test_*.py modules, collects their `test_*` functions, supplies the handful of
fixtures those tests depend on (the same ones conftest.py defines), and runs them
with a minimal monkeypatch shim.

It is deliberately small and covers the SAME tests pytest runs -- it is NOT a
second test framework. If pytest is available, prefer it (richer output, marks).

    python tests/run_all.py            # from the project root
    python run_all.py                  # from inside tests/

Exit code 0 == all collected tests passed (skips are not failures); 1 otherwise.
ASCII only; no network; degrades (skips) on missing numpy/cv2/yaml exactly like
the pytest run.
"""
from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

# --- sys.path: tests/ (so test_*.py + conftest import) + project root + the
# runtime/ director/ pipeline/ package dirs (mirror conftest). HERE goes on too
# so this works under `python tests/run_all.py`, `python run_all.py`, AND runpy. ---
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for _p in (HERE, PROJECT_ROOT, PROJECT_ROOT / "runtime", PROJECT_ROOT / "director",
           PROJECT_ROOT / "pipeline"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


# --------------------------------------------------------------------------- #
# A tiny pytest shim: only the surface the suite actually uses.
#   - Skipped(...) exception + a `pytest` module stand-in with .raises / .skip /
#     .approx / .fail / .mark.skipif.
# This lets the test modules `import pytest` and run unmodified under this runner.
# --------------------------------------------------------------------------- #
class Skipped(Exception):
    pass


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc!r}")
        return issubclass(et, self.exc)  # swallow the expected exception


class _Approx:
    def __init__(self, value, rel=None, abs=None):
        self.value = value
        self.rel = rel
        self.abs = abs if abs is not None else (1e-6 if rel is None else None)

    def __eq__(self, other):
        a, b = float(other), float(self.value)
        tol = self.abs if self.abs is not None else abs(b) * (self.rel or 1e-6)
        return abs(a - b) <= max(tol, 1e-12)

    def __repr__(self):
        return f"approx({self.value})"


class _Mark:
    @staticmethod
    def skipif(condition, reason=""):
        def deco(fn):
            if condition:
                fn.__skip__ = reason or "skipif"
            return fn
        return deco

    def __getattr__(self, _name):
        # any other mark (e.g. @pytest.mark.foo) is a no-op decorator
        return lambda *a, **k: (lambda fn: fn)


class _PytestShim:
    Skipped = Skipped
    mark = _Mark()

    @staticmethod
    def raises(exc):
        return _Raises(exc)

    @staticmethod
    def skip(reason=""):
        raise Skipped(reason)

    @staticmethod
    def fail(reason=""):
        raise AssertionError(reason)

    @staticmethod
    def approx(value, rel=None, abs=None):
        return _Approx(value, rel=rel, abs=abs)

    @staticmethod
    def fixture(*a, **k):
        # When used with arguments: @pytest.fixture(...) -> returns the decorator.
        if len(a) == 1 and callable(a[0]) and not k:
            a[0].__is_fixture__ = True
            return a[0]

        def deco(fn):
            fn.__is_fixture__ = True
            return fn
        return deco


# Install the shim as the importable `pytest` BEFORE importing the test modules,
# but only if real pytest is not present (then prefer the real thing via -m pytest).
if "pytest" not in sys.modules:
    try:
        import pytest as _real  # noqa: F401
        # Real pytest exists; recommend it but still allow this runner to proceed
        # with the real module (its fixtures won't auto-wire here, so we use the
        # shim regardless for the manual collection below).
        sys.modules["pytest"] = _PytestShim()  # type: ignore[assignment]
    except Exception:
        sys.modules["pytest"] = _PytestShim()  # type: ignore[assignment]

pytest = sys.modules["pytest"]


# --------------------------------------------------------------------------- #
# Dependency probes (mirror conftest) -- exposed for collection-time skips.
# --------------------------------------------------------------------------- #
def _probe(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


HAVE_NUMPY = _probe("numpy")
HAVE_CV2 = _probe("cv2")
HAVE_YAML = _probe("yaml")


# --------------------------------------------------------------------------- #
# A minimal monkeypatch (setattr-only, auto-undo) + tmp_path. Enough for the
# fixtures the suite uses. Reset between every test for isolation.
# --------------------------------------------------------------------------- #
class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=None):
        # support both setattr(obj, "name", value) and setattr("mod.attr", value)
        if value is None and isinstance(name, object) and not isinstance(name, str):
            raise TypeError("string-target form not supported in this shim")
        old = getattr(target, name)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def undo(self):
        for target, name, old in reversed(self._undo):
            setattr(target, name, old)
        self._undo.clear()


def _make_fixture_value(name, mp, tmp_root):  # noqa: PLR0912  -- the stdlib fallback runner: one branch per pytest feature it stands in for
    """Provide the small set of fixtures the test functions request by name.
    Mirrors conftest.py. Returns the value (or raises Skipped to mark a skip)."""
    if name == "monkeypatch":
        return mp
    if name == "tmp_path":
        return Path(tempfile.mkdtemp(dir=str(tmp_root)))
    if name == "gen_dir":
        d = Path(tempfile.mkdtemp(dir=str(tmp_root))) / "gen"
        d.mkdir()
        return d
    if name == "gpath":
        d = Path(tempfile.mkdtemp(dir=str(tmp_root)))
        return d / "gallery.yaml"
    if name == "synth_face":
        if not (HAVE_NUMPY and HAVE_CV2):
            raise Skipped("synth face needs numpy + cv2")
        import verify
        return verify._synthetic_face
    if name == "crossframe_cutout":
        if not HAVE_NUMPY:
            raise Skipped("needs numpy")
        import crossframe
        return crossframe._synth_cutout(256)
    if name == "synth_cutout_rgba":
        if not (HAVE_NUMPY and HAVE_CV2):
            raise Skipped("needs numpy + cv2")
        import rig
        return rig._synth_cutout(256)
    if name == "written_rig":
        if not (HAVE_NUMPY and HAVE_CV2):
            raise Skipped("rig fixture needs numpy + cv2")
        import json as _json
        import cv2
        import rig
        gd = Path(tempfile.mkdtemp(dir=str(tmp_root))) / "gen"
        gd.mkdir()
        slug = "synthhero"
        cut = rig._synth_cutout(256)
        spec = rig._synth_spec_for(cut)
        cv2.imwrite(str(gd / f"{slug}_cutout.png"), cv2.cvtColor(cut, cv2.COLOR_RGBA2BGRA))
        (gd / f"{slug}_rig.json").write_text(_json.dumps(spec), encoding="utf-8")
        return slug, gd
    if name == "written_portrait":
        if not (HAVE_NUMPY and HAVE_CV2):
            raise Skipped("portrait fixture needs numpy + cv2")
        import cv2
        import stage_render as sr
        gd = Path(tempfile.mkdtemp(dir=str(tmp_root))) / "gen"
        gd.mkdir()
        slug = "_test"
        cv2.imwrite(str(gd / f"{slug}_portrait.png"),
                    cv2.cvtColor(sr._synthetic_portrait(512), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(gd / f"{slug}_cutout.png"),
                    cv2.cvtColor(sr._synthetic_cutout(512), cv2.COLOR_RGBA2BGRA))
        return slug, gd
    if name in ("feeds_tmp", "feeds_paths"):
        import feeds
        d = Path(tempfile.mkdtemp(dir=str(tmp_root))) / "data"
        d.mkdir()
        cache = d / "feed.json"
        mp.setattr(feeds, "DATA", d)
        mp.setattr(feeds, "CACHE", cache)
        if name == "feeds_tmp":
            mp.setattr(feeds, "signal_headlines", lambda *a, **k: [])
            mp.setattr(feeds, "weather_today", lambda *a, **k: [])
            mp.setattr(feeds, "calendar_today", lambda *a, **k: [])

            def _no_net(*a, **k):
                raise AssertionError("network call attempted in an offline test")

            mp.setattr(feeds, "_get_json", _no_net)
            mp.setattr(feeds, "_get_text", _no_net)
        return cache
    if name == "faces":
        # test_verify's `faces` fixture: build from synth_face directly.
        if not (HAVE_NUMPY and HAVE_CV2):
            raise Skipped("needs numpy + cv2")
        import numpy as np
        import verify
        base = verify._synthetic_face()
        trunc = base.copy()
        trunc[base.shape[0] // 2:, :, :] = 0
        return {
            "base": base,
            "same": base.copy(),
            "jittered": np.clip(base.astype(np.int16) + 4, 0, 255).astype(np.uint8),
            "pose": verify._synthetic_face(cx=104, cy=96, scale=1.25),
            "trunc": trunc,
            "noise": np.random.default_rng(7).integers(0, 256, base.shape).astype(np.uint8),
        }
    if name == "tmp_chars":
        import gallery
        d = Path(tempfile.mkdtemp(dir=str(tmp_root))) / "characters"
        mp.setattr(gallery, "CHARS_DIR", d)
        return d
    raise Skipped(f"runner has no fixture provider for {name!r}")


# --------------------------------------------------------------------------- #
# Collection + execution
# --------------------------------------------------------------------------- #
def _module_skip_reason(mod):
    """Honor a module-level pytestmark = pytest.mark.skipif(...) by detecting a
    __skip__ on the marker the shim set. (The shim's skipif sets fn.__skip__; for
    a module-level mark we re-derive from the booleans the modules use.)"""
    # The test modules gate on HAVE_NUMPY/HAVE_CV2/HAVE_YAML via pytestmark; we read
    # those names from the module's globals when present (they import from conftest).
    g = vars(mod)
    mark = g.get("pytestmark")
    # the shim's skipif returns a decorator, so a module-level pytestmark is that
    # decorator object; we can't introspect its condition. Instead, re-evaluate the
    # well-known gates the suite uses, by module name.
    name = mod.__name__
    if name in ("test_crossframe", "test_rig_loop") and not HAVE_NUMPY:
        return "needs numpy"
    if name == "test_stage_render" and not HAVE_NUMPY:
        return "needs numpy"
    if name == "test_verify" and not (HAVE_NUMPY and HAVE_CV2):
        return "verify needs numpy + cv2"
    if name == "test_gallery" and not HAVE_YAML:
        return "gallery needs pyyaml"
    _ = mark  # silence unused
    return None


def main():
    test_files = sorted(HERE.glob("test_*.py"))
    passed = failed = skipped = 0
    failures = []

    tmp_root = Path(tempfile.mkdtemp(prefix="lp_runall_"))

    for tf in test_files:
        modname = tf.stem
        try:
            mod = importlib.import_module(modname)
        except Skipped as exc:
            print(f"[SKIP module] {modname}: {exc}")
            skipped += 1
            continue
        except Exception:
            print(f"[ERROR import] {modname}")
            traceback.print_exc()
            failed += 1
            failures.append(f"{modname} (import)")
            continue

        mod_skip = _module_skip_reason(mod)
        funcs = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction)
                 if n.startswith("test_") and f.__module__ == modname]
        for fn_name, fn in sorted(funcs):
            label = f"{modname}::{fn_name}"
            if mod_skip:
                print(f"[SKIP] {label} ({mod_skip})")
                skipped += 1
                continue
            if getattr(fn, "__skip__", None):
                print(f"[SKIP] {label} ({fn.__skip__})")
                skipped += 1
                continue
            mp = _MonkeyPatch()
            try:
                kwargs = {}
                for pname in inspect.signature(fn).parameters:
                    kwargs[pname] = _make_fixture_value(pname, mp, tmp_root)
                fn(**kwargs)
                passed += 1
            except Skipped as exc:
                print(f"[SKIP] {label} ({exc})")
                skipped += 1
            except Exception:
                failed += 1
                failures.append(label)
                print(f"[FAIL] {label}")
                traceback.print_exc()
            finally:
                mp.undo()

    print("\n" + "=" * 60)
    print(f"run_all.py: {passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print("failures:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
