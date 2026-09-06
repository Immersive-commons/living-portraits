"""
test_verify.py -- the QA gate (pipeline/verify.py).

Covers the SSIM primitive's discrimination (identical ~1.0, recolored-but-aligned
high, shifted-pose mid, truncated low, noise ~0), the loopability / continuity
threshold logic, the identity degraded-path floor, the register skip-degrades-open
contract, and the Verdict aggregation (skips don't veto; AND of real pass/fails).

verify.py hard-requires cv2 + numpy at IMPORT (top-level `import cv2`), so the
whole file skips on a box without them. InsightFace + Ollama are NOT required:
identity runs degraded, and register is forced to a deterministic outcome via a
monkeypatched _ollama_yes_no so no socket is ever opened.

The skip has to happen at MODULE level, and neither of the two obvious ways
works here. A `pytestmark = pytest.mark.skipif(...)` gates the tests pytest has
already collected -- it does not stop this module's own body from running, so
`import verify` below still executed at COLLECTION time and took the whole
session down with `Interrupted: 1 error during collection`: zero tests run, in
the entire suite, not just this file. And `pytest.importorskip("cv2")` does not
catch it either, because importorskip defaults to ModuleNotFoundError while the
failure we actually get is a plain ImportError -- opencv-python is installed and
importable everywhere except that libGL.so.1 is missing, which is the normal
state of a headless container.

So: conftest's HAVE_* probes, which already `__import__` inside `except
Exception` and therefore see this correctly, plus an explicit module-level skip.
Absence is a skip with a reason, never an error (AGENTS.md rule 4).
"""
from __future__ import annotations

import pytest

from conftest import HAVE_CV2, HAVE_NUMPY

if not (HAVE_NUMPY and HAVE_CV2):
    pytest.skip(
        "verify.py imports cv2 + numpy at module load",
        allow_module_level=True,
    )

import numpy as np  # noqa: E402

import verify  # noqa: E402
from verify import (  # noqa: E402
    CONTINUITY_THRESHOLD, LOOP_THRESHOLD,
    Verdict, continuity, identity, loopability, register, ssim, verify_clip, verify_portrait,
)


@pytest.fixture
def faces(synth_face):
    """A family of deterministic synthetic poses derived from verify._synthetic_face."""
    base = synth_face()
    return {
        "base": base,
        "same": base.copy(),
        "jittered": np.clip(base.astype(np.int16) + 4, 0, 255).astype(np.uint8),
        "pose": synth_face(cx=104, cy=96, scale=1.25),       # shifted+scaled (structure moved)
        "trunc": _truncate(base),                            # bottom half black (corrupt render)
        "noise": np.random.default_rng(7).integers(0, 256, base.shape).astype(np.uint8),
    }


def _truncate(img):
    out = img.copy()
    out[img.shape[0] // 2:, :, :] = 0
    return out


# --------------------------------------------------------------------------- #
# SSIM primitive: monotone discrimination
# --------------------------------------------------------------------------- #
def test_ssim_identical_is_one(faces):
    assert ssim(faces["base"], faces["base"]) == pytest.approx(1.0, abs=1e-6)


def test_ssim_jitter_high_pose_mid_trunc_low_noise_near_zero(faces):
    s_same = ssim(faces["base"], faces["same"])
    s_jit = ssim(faces["base"], faces["jittered"])
    s_pose = ssim(faces["base"], faces["pose"])
    s_trunc = ssim(faces["base"], faces["trunc"])
    s_noise = ssim(faces["base"], faces["noise"])
    # ordering: identical >= jitter > shifted-pose > truncated > noise
    assert s_same >= s_jit > s_pose > s_trunc > s_noise
    assert s_jit > 0.9, f"a +4-level recolor should stay structurally high, got {s_jit:.3f}"
    assert s_noise < 0.2, f"noise should score near 0, got {s_noise:.3f}"


def test_ssim_handles_size_mismatch(faces):
    big = np.repeat(np.repeat(faces["base"], 2, axis=0), 2, axis=1)  # 2x in both dims
    # shape-matched internally; a 2x-upscaled copy is still structurally ~identical
    assert ssim(faces["base"], big) > 0.8


# --------------------------------------------------------------------------- #
# loopability: seam threshold
# --------------------------------------------------------------------------- #
def test_loopability_identical_passes(faces):
    passed, score, _ = loopability(faces["base"], faces["same"])
    assert passed is True and score >= LOOP_THRESHOLD


def test_loopability_moved_seam_fails(faces):
    passed, _, reason = loopability(faces["base"], faces["pose"])
    assert passed is False
    assert "pop" in reason.lower() or "<" in reason


def test_loopability_unreadable_frame_hard_fails():
    # an unscorable input -> a (False, 0.0, reason) verdict, never an exception
    passed, score, reason = loopability("nonexistent_file.png", "also_missing.png")
    assert passed is False and score == 0.0
    assert "could not score" in reason


# --------------------------------------------------------------------------- #
# continuity: lands-on-target vs drifts
# --------------------------------------------------------------------------- #
def test_continuity_lands_on_target(faces):
    passed, score, _ = continuity(faces["base"], faces["same"])
    assert passed is True and score >= CONTINUITY_THRESHOLD


def test_continuity_drifts_fails(faces):
    # clip ends on the shifted pose but claims to arrive at the base node
    passed, _, _ = continuity(faces["pose"], faces["base"])
    assert passed is False


# --------------------------------------------------------------------------- #
# identity (degraded path -- no InsightFace required)
# --------------------------------------------------------------------------- #
def test_identity_same_image_passes_even_degraded(faces, monkeypatch):
    # force the degraded path so the test is host-independent (no InsightFace)
    monkeypatch.setattr(verify, "_insightface_app", lambda: None)
    passed, _, reason = identity(faces["base"], faces["same"])
    assert passed is True


def test_identity_different_image_fails_even_degraded(faces, monkeypatch):
    monkeypatch.setattr(verify, "_insightface_app", lambda: None)
    passed, _, _ = identity(faces["base"], faces["noise"])
    assert passed is False


def test_identity_degraded_reason_marks_coarse(faces, monkeypatch):
    monkeypatch.setattr(verify, "_insightface_app", lambda: None)
    _, _, reason = identity(faces["base"], faces["same"])
    assert "degraded" in reason.lower(), "degraded identity must flag itself as coarse"


# --------------------------------------------------------------------------- #
# register: skip-degrades-open + clear pass/fail when Ollama answers
# --------------------------------------------------------------------------- #
_CONTRACT = ("Theme: a dim alchemist's study.\nVoice: dry patrician baritone.\n"
             "A washed-up tragedian who knows he is a painting and plays to the house.")


def test_register_skips_open_when_ollama_undecided(monkeypatch):
    # Ollama unreachable/undecided -> (None, reason); register degrades OPEN.
    monkeypatch.setattr(verify, "_ollama_yes_no", lambda *a, **k: (None, "ollama unreachable (URLError)"))
    passed, score, reason = register("any line", _CONTRACT)
    assert passed is True, "register must degrade OPEN when the check can't run"
    assert score == -1.0, "skipped register must carry the -1.0 sentinel"
    assert "SKIPPED" in reason


def test_register_passes_when_in_voice(monkeypatch):
    monkeypatch.setattr(verify, "_ollama_yes_no", lambda *a, **k: (True, '{"in_register": true}'))
    passed, score, _ = register("You there, behind the glass -- witness Act Five.", _CONTRACT)
    assert passed is True and score == 1.0


def test_register_fails_when_out_of_voice(monkeypatch):
    monkeypatch.setattr(verify, "_ollama_yes_no", lambda *a, **k: (False, '{"in_register": false}'))
    passed, score, reason = register("Here's Tuesday's weather: sunny, high of 71.", _CONTRACT)
    assert passed is False and score == 0.0
    assert "OUT of voice" in reason


# --------------------------------------------------------------------------- #
# Verdict aggregation: skips don't veto; AND of real pass/fails
# --------------------------------------------------------------------------- #
def test_verdict_and_of_real_checks():
    v = Verdict()
    v.add("a", (True, 0.9, "ok"))
    v.add("b", (True, 0.95, "ok"))
    assert v.passed is True
    v.add("c", (False, 0.1, "drifts"))
    assert v.passed is False


def test_verdict_skipped_does_not_veto():
    v = Verdict()
    v.add("continuity", (True, 0.9, "lands"))
    v.add("register", (True, -1.0, "register: SKIPPED (ollama unreachable)"))
    assert v.passed is True
    assert "register" in v.skipped
    assert "register" not in v.degraded


def test_verdict_degraded_tracked_but_counts():
    v = Verdict()
    v.add("identity", (True, 0.7, "identity[degraded] hist+ORB 0.700 >= 0.62 (coarse)"))
    assert v.passed is True
    assert "identity" in v.degraded


# --------------------------------------------------------------------------- #
# Orchestrators: verify_clip / verify_portrait
# --------------------------------------------------------------------------- #
def test_verify_clip_idle_loop_passes(faces, monkeypatch):
    monkeypatch.setattr(verify, "_insightface_app", lambda: None)
    v = verify_clip([faces["base"], faces["same"], faces["same"]], kind="idle",
                    canonical_portrait=faces["same"])
    assert v.passed is True


def test_verify_clip_broken_seam_fails(faces):
    v = verify_clip([faces["base"], faces["base"], faces["trunc"]], kind="idle")
    assert v.passed is False


def test_verify_clip_bridge_lands(faces):
    v = verify_clip([faces["pose"], faces["base"], faces["same"]],
                    target_node_frame=faces["same"], kind="bridge")
    assert v.passed is True


def test_verify_clip_bridge_drifts_fails(faces):
    v = verify_clip([faces["base"], faces["base"], faces["pose"]],
                    target_node_frame=faces["base"], kind="bridge")
    assert v.passed is False


def test_verify_clip_bridge_requires_target(faces):
    # kind=bridge with no target_node_frame is a hard fail, not a crash
    v = verify_clip([faces["base"], faces["same"]], kind="bridge")
    assert v.passed is False
    assert any("requires target_node_frame" in r for r in v.reasons)


def test_verify_clip_needs_two_frames(faces):
    v = verify_clip([faces["base"]], kind="idle")
    assert v.passed is False
    assert any("at least 2 frames" in r for r in v.reasons)


def test_verify_clip_unknown_kind_fails(faces):
    v = verify_clip([faces["base"], faces["same"]], kind="pirouette")
    assert v.passed is False


def test_verify_portrait_readable_first_portrait_passes(faces, monkeypatch):
    monkeypatch.setattr(verify, "_ollama_yes_no", lambda *a, **k: (None, "skipped"))
    v = verify_portrait(faces["base"], character_md=_CONTRACT)
    # readable (self-SSIM ~1) passes; register skipped (open). No reference -> identity skipped.
    assert v.passed is True


def test_verify_portrait_blank_render_fails_identity(faces, monkeypatch):
    monkeypatch.setattr(verify, "_insightface_app", lambda: None)
    blank = np.zeros_like(faces["base"])
    # blank-vs-blank 'readable' passes, but identity vs a real portrait fails
    v = verify_portrait(blank, canonical_portrait=faces["base"])
    assert v.passed is False
    assert "identity" in v.checks
