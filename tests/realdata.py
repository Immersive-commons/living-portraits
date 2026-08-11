"""realdata.py -- the PRODUCTION snapshot, loaded once, for tests that must not run on fixtures.

Every other test file in this suite builds its own synthetic life. That is the right default:
fixtures are fast, deterministic, and they let a test say exactly what shape it is asserting.
It is also how a harness ends up passing by construction -- a fixture built to demonstrate a
capability will demonstrate it.

This module is the counterweight. It loads the real graph and the real journals captured from
`hil` into `data/_realdata/`, so a measurement can come back with a number nobody chose:

    import realdata                                 # tests/ is on sys.path under pytest
    from realdata import realdata_snapshot           # noqa: F401 -- the skip-if-absent fixture

    def test_something(realdata_snapshot):
        g = realdata.graph()                        # 259 nodes / 1472 edges
        j = realdata.journal("phineas")             # 11,601 entries over 66 days

READ-ONLY BY CONTRACT. Nothing here writes to `data/_realdata/`, and callers must not either:
the snapshot is evidence, and evidence that a test can edit proves nothing. Every loader
returns a fresh top-level container each call for the same reason -- one test mutating a list
must not silently change what the next one measures. (The parsed entries themselves are shared
for speed; do not mutate the dicts.)

Absence is a SKIP with a reason, never an error: a bare checkout has no snapshot, and the rest
of the suite must still be green there.

Dependency-free: stdlib + pytest. Deliberately does NOT import any module under test, so a
broken `journal_score` cannot silently change what the fixture reports as ground truth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                      # peers may import this before conftest runs
    sys.path.insert(0, str(ROOT))

SNAPSHOT = ROOT / "data" / "_realdata"
GRAPH_FILE = SNAPSHOT / "video_graph.json"
INTENT_FILE = SNAPSHOT / "intent.json"
PROPOSALS_FILE = SNAPSHOT / "proposals.json"
CHARACTERS = ("phineas", "maxx")

SKIP_REASON = (
    "production snapshot missing: %s. Real-data measurements are skipped, not faked. "
    "Copy the live graph + journals from hil into that directory to run them." % SNAPSHOT)

_cache: dict = {}


# --------------------------------------------------------------------------- availability
def journal_file(character):
    return SNAPSHOT / ("%s.jsonl" % character)


def available(*characters):
    """True when the graph AND every named journal are present (default: both characters)."""
    if not GRAPH_FILE.is_file():
        return False
    for c in (characters or CHARACTERS):
        if not journal_file(c).is_file():
            return False
    return True


def require(*characters):
    """Skip the calling test -- with the path in the reason -- when the snapshot is absent."""
    if not available(*characters):
        pytest.skip(SKIP_REASON)


@pytest.fixture(scope="session")
def realdata_snapshot():
    """Session fixture: skips the test when the snapshot is absent, else yields this module."""
    require()
    return sys.modules[__name__]


# --------------------------------------------------------------------------- loaders
def raw_graph():
    """The graph JSON exactly as production stores it -- schema string included, nothing
    normalised. Measurements about what the graph does NOT carry have to read it raw."""
    if "raw_graph" not in _cache:
        _cache["raw_graph"] = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
    d = _cache["raw_graph"]
    return {"schema": d.get("schema"), "nodes": dict(d.get("nodes") or {}),
            "edges": list(d.get("edges") or [])}


def graph():
    """The same graph as a `runtime.video_graph.VideoGraph`, for code that walks it.

    Imported lazily and only here, so a test that only wants the raw JSON never depends on
    the runtime package importing cleanly."""
    from runtime import video_graph
    d = raw_graph()
    return video_graph.VideoGraph(d["nodes"], d["edges"])


def journal(character):
    """Every parsed line of one character's real journal, oldest first.

    Unparseable lines are counted rather than dropped silently -- `journal_stats()` reports
    them, because "the file has 11,601 lines and retrieval sees 11,601 entries" is itself one
    of the things this harness checks."""
    key = "journal:%s" % character
    if key not in _cache:
        entries, lines, bad = [], 0, 0
        with journal_file(character).open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                lines += 1
                try:
                    d = json.loads(ln)
                except Exception:
                    bad += 1
                    continue
                if isinstance(d, dict):
                    entries.append(d)
                else:
                    bad += 1
        _cache[key] = (entries, {"lines": lines, "parsed": len(entries), "unparseable": bad})
    return list(_cache[key][0])


def journal_stats(character):
    """{lines, parsed, unparseable} for the raw file -- the retention denominator."""
    journal(character)
    return dict(_cache["journal:%s" % character][1])


def intent():
    try:
        return json.loads(INTENT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summary():
    """One dict of the snapshot's shape, for a test that wants to print what it measured."""
    g = raw_graph()
    out = {"nodes": len(g["nodes"]), "edges": len(g["edges"]), "schema": g["schema"],
           "characters": {}}
    for c in CHARACTERS:
        if journal_file(c).is_file():
            e = journal(c)
            stamps = [float(x["ts"]) for x in e if x.get("ts")]
            out["characters"][c] = {
                "entries": len(e),
                "days": ((max(stamps) - min(stamps)) / 86400.0) if stamps else 0.0,
                "first_ts": min(stamps) if stamps else None,
                "last_ts": max(stamps) if stamps else None}
    return out
