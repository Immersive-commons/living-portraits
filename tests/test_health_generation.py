"""The generation detector has to FIRE, not just pass.

A health check nobody has seen fail is an unfalsifiable success signal: it looks like
coverage and provides none. This one was written for a specific incident, so the first
test replays that incident's exact shape and asserts the check catches it.

It also pins the two ways the check is allowed to stay quiet, because the first draft
got one of them wrong and FAILED live against a wall that was fine.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# health/ is a plain directory with no __init__.py -- it is an operator tool, not a
# package, and the oracle runs `python checks.py <id>` from inside it. Load it by path
# rather than adding another sys.path entry the rest of the suite would inherit.
_spec = importlib.util.spec_from_file_location("lp_health_checks", ROOT / "health" / "checks.py")
checks = importlib.util.module_from_spec(_spec)
sys.modules["lp_health_checks"] = checks
_spec.loader.exec_module(checks)


@pytest.fixture()
def host(monkeypatch):
    """Answer for the host, so the check is tested and the network is not."""
    def _set(payload, *, err=None, task_enabled=True):
        monkeypatch.setattr(checks, "_remote_json", lambda _s: (payload, err))
        monkeypatch.setattr(checks, "_task_enabled", lambda _t: task_enabled)
    return _set


def _verdict_ok(rc):
    """`_verdict` returns a PROCESS EXIT CODE -- 0 pass, 1 fail -- because the oracle runs
    these as `python checks.py <id>` and gates on rc. So the truthy value means FAILURE,
    and `assert checks.generation_healthy()` would assert the opposite of what it reads
    like. Naming it once here is cheaper than getting it backwards per test."""
    return rc == 0


def test_it_catches_the_outage_it_was_written_for(host):
    """2026-07-01 to 2026-08-08: every generation failed on the same upload 403, 886
    times, and nothing paged for 39 days. This is that week, and it must FAIL."""
    host({"window_days": 7, "queued": 210, "done": 0, "failed": 210, "pending": 0,
          "top_reason": "POST /api/storage-upload-file -> 403", "top_count": 208})
    v = checks.generation_healthy()
    assert not _verdict_ok(v)


def test_it_names_the_dominant_reason(host, capsys):
    """Naming the reason is what turns a page into a diagnosis. A check that says only
    'generation is unhealthy' costs the reader the same five weeks.

    The reason is EVIDENCE, which `_verdict` prints; the return value is only the exit
    code. So this reads stdout, which is also what the oracle greps.
    """
    host({"window_days": 7, "queued": 210, "done": 0, "failed": 210, "pending": 0,
          "top_reason": "POST /api/storage-upload-file -> 403", "top_count": 208})
    checks.generation_healthy()
    out = capsys.readouterr().out
    assert "403" in out
    assert "storage-upload-file" in out
    assert "CHECK generation_healthy: FAIL" in out


def test_a_deep_queue_is_not_a_fault(host):
    """The live false alarm. CLIP_CHAR_CAP exists to make the queue back up, so pending
    work with nothing attempted is the DESIGNED steady state. Counting `pending` as an
    attempt failed against a healthy wall sitting at 12/13 clips for the day."""
    host({"window_days": 7, "queued": 12, "done": 0, "failed": 0, "pending": 12,
          "top_reason": None, "top_count": 0})
    assert _verdict_ok(checks.generation_healthy())


def test_off_by_choice_is_a_pass(host):
    """Same rule as every other detector here: a wall deliberately not growing owes
    nothing. Without this the check pages every time someone disables lp-gen on purpose."""
    host({"window_days": 7, "queued": 0, "done": 0, "failed": 99, "pending": 0,
          "top_reason": "anything", "top_count": 99}, task_enabled=False)
    assert _verdict_ok(checks.generation_healthy())


def test_a_healthy_week_passes(host):
    """The measured steady state on 2026-09-08: about 3 poses a day, nothing failing."""
    host({"window_days": 7, "queued": 21, "done": 21, "failed": 0, "pending": 4,
          "top_reason": None, "top_count": 0})
    assert _verdict_ok(checks.generation_healthy())


def test_a_minority_of_failures_is_tolerated(host):
    """Transient vendor errors are normal and must not page. The bar is 60%, so a bad
    but working week stays quiet."""
    host({"window_days": 7, "queued": 20, "done": 12, "failed": 8, "pending": 0,
          "top_reason": "HTTP 503", "top_count": 5})
    assert _verdict_ok(checks.generation_healthy())


def test_a_majority_of_failures_is_not(host):
    host({"window_days": 7, "queued": 20, "done": 5, "failed": 15, "pending": 0,
          "top_reason": "HTTP 503", "top_count": 15})
    assert not _verdict_ok(checks.generation_healthy())


def test_an_unreachable_host_fails_rather_than_passing_quietly(host):
    """Cannot-see is not the same as fine. Returning PASS here would make every other
    outage invisible the moment ssh broke."""
    host(None, err="host unreachable")
    assert not _verdict_ok(checks.generation_healthy())
