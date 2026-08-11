"""test_lived.py -- the lived-experience write-back. No network.

The audit's structural complaint was that the graph "accretes from a credit card, not from
experience" — nothing a character did was ever recorded against it. These tests pin the
store that fixes that, and the real-data test drives it with the ACTUAL production graph
(259 nodes / 1472 edges) rather than a toy one, because the failure modes that matter
(bounded moods, flush cost, a partial record) only show up at real size.
"""
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import lived

REAL = ROOT / "data" / "_realdata" / "video_graph.json"
DAY = 86400.0


def test_a_visit_is_counted_and_stamped(tmp_path):
    b = lived.Lived("phineas", base=tmp_path, now=0)
    b.visit("phineas:glower", mood_band="fixated", now=1000.0)
    b.visit("phineas:glower", mood_band="weary", now=2000.0)
    assert b.visits("phineas:glower") == 2
    rec = b.nodes["phineas:glower"]
    assert rec["first"] == 1000.0 and rec["last"] == 2000.0      # valid-time, both ends
    assert rec["moods"] == {"fixated": 1, "weary": 1}
    assert b.visits("phineas:never") == 0                        # unknown pose is 0, not KeyError


def test_dwell_and_plays_accumulate(tmp_path):
    b = lived.Lived("maxx", base=tmp_path, now=0)
    b.tick_dwell("maxx:idle", now=0)
    b.tick_dwell("maxx:idle", now=0)
    b.play("maxx/i2f/v0")
    b.play("maxx/i2f/v0")
    b.play("maxx/i2f/v1")
    assert b.dwell("maxx:idle") == 2
    assert b.plays("maxx/i2f/v0") == 2 and b.plays("maxx/i2f/v1") == 1
    b.play(None)                                                  # a pick with no id must not crash
    assert b.plays(None) == 0


def test_mood_histogram_stays_bounded(tmp_path):
    """A free-text-adjacent field on a loop that runs for months is an unbounded-growth
    trap. Keep the most-brought bands, drop the tail."""
    b = lived.Lived("phineas", base=tmp_path, now=0)
    for i in range(lived.MAX_MOODS * 3):
        b.visit("phineas:anchor", mood_band="band%02d" % i, now=float(i))
    for _ in range(5):
        b.visit("phineas:anchor", mood_band="band00", now=999.0)
    moods = b.nodes["phineas:anchor"]["moods"]
    assert len(moods) <= lived.MAX_MOODS
    assert "band00" in moods                                      # the most-brought one survives
    assert b.dominant_mood("phineas:anchor") == "band00"


def test_flush_is_rate_limited_then_atomic(tmp_path):
    b = lived.Lived("phineas", base=tmp_path, now=0.0)
    b.visit("phineas:anchor", now=1.0)
    assert b.flush(now=5.0) is False                              # too soon: the render loop calls constantly
    assert not lived.path_for("phineas", tmp_path).exists()
    assert b.flush(now=lived.FLUSH_EVERY + 1) is True
    on_disk = json.loads(lived.path_for("phineas", tmp_path).read_text(encoding="utf-8"))
    assert on_disk["schema"] == lived.SCHEMA and on_disk["character"] == "phineas"
    assert on_disk["nodes"]["phineas:anchor"]["visits"] == 1
    assert not list(tmp_path.glob("*.tmp"))                       # atomic: no temp left behind
    assert b.flush(now=lived.FLUSH_EVERY + 2) is False            # nothing dirty -> no write


def test_a_record_survives_a_restart(tmp_path):
    """The walker restarts often (watchdog, logon, a game stealing the display). A life
    that resets to zero on every restart is not a life."""
    b = lived.Lived("phineas", base=tmp_path, now=0.0)
    for i in range(7):
        b.visit("phineas:swoon", now=float(i))
    b.flush(force=True, now=10.0)
    again = lived.load("phineas", base=tmp_path)
    assert again.visits("phineas:swoon") == 7
    again.visit("phineas:swoon", now=20.0)
    assert again.visits("phineas:swoon") == 8                     # continues, does not restart


def test_writes_never_raise_when_the_disk_refuses(tmp_path, monkeypatch):
    """Fail-soft by contract: a statistics file is never worth a dark panel."""
    b = lived.Lived("phineas", base=tmp_path, now=0.0)
    b.visit("phineas:anchor", now=1.0)
    monkeypatch.setattr(lived.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    assert b.flush(force=True, now=100.0) is False                # reported, not raised
    assert b._dirty is True                                       # and the counters are kept for next time


def test_the_line_the_character_reads_is_grammatical_and_makes_no_false_claims(tmp_path):
    """This text goes into the character's own prompt. `stood here 1 times` is a tell, and
    claiming a "first visit 0m ago" teaches a portrait a past it does not have."""
    import time as _t
    from director import heartbeat as hb
    now = _t.time()
    b = lived.Lived("phineas", base=tmp_path, now=now)

    b.visit("phineas:anchor", mood_band="fixated", now=now - 30)          # brand new
    line = hb._lived_line(b, "phineas:anchor", now=now)
    assert "once" in line and "1 times" not in line
    assert "ago" not in line.split("Across")[0]                            # no invented history

    b2 = lived.Lived("maxx", base=tmp_path, now=now)                       # a real history
    for i in range(12):
        b2.visit("maxx:idle", mood_band="weary", now=now - 5 * DAY + i * 3600)
    line = hb._lived_line(b2, "maxx:idle", now=now)
    assert "12 times" in line and "the first 5d ago" in line and "weary" in line

    assert hb._lived_line(b, "phineas:never_been", now=now) == ""          # silence, not a zero
    assert hb._lived_line(None, "phineas:anchor", now=now) == ""           # no record -> no line


def test_flush_preserves_keys_this_class_does_not_own(tmp_path):
    """A writer that drops what it does not recognise destroys other tools' data. The
    76-day backfill wrote a `backfill` provenance block and `label_plays` into this file;
    the walker's next flush kept the counts and erased both."""
    p = lived.path_for("phineas", tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "schema": lived.SCHEMA, "character": "phineas",
        "nodes": {"phineas:anchor": {"visits": 19434, "dwell": 55434}},
        "edges": {},
        "label_plays": {"breathe": 900},
        "backfill": {"source": "_preview.log", "picks": 422868},
    }), encoding="utf-8")

    b = lived.Lived("phineas", base=tmp_path, now=0.0)
    assert b.visits("phineas:anchor") == 19434            # the backfilled history loads
    b.visit("phineas:anchor", now=1.0)
    b.flush(force=True, now=100.0)

    back = json.loads(p.read_text(encoding="utf-8"))
    assert back["backfill"]["picks"] == 422868            # ...and survives a live flush
    assert back["label_plays"]["breathe"] == 900
    assert back["nodes"]["phineas:anchor"]["visits"] == 19435   # while the count advances
    assert back["character"] == "phineas"                 # our own keys still win


def test_a_torn_file_degrades_to_an_empty_record(tmp_path):
    p = lived.path_for("phineas", tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"nodes": {"phineas:anchor": {"visits": 3}}, "edg', encoding="utf-8")
    b = lived.Lived("phineas", base=tmp_path)
    assert b.visits("phineas:anchor") == 0                        # no crash, no half-parsed lie


# --------------------------------------------------------------------------- real data
@pytest.mark.skipif(not REAL.exists(),
                    reason="production snapshot absent: data/_realdata/video_graph.json "
                           "(pull it from the host to run the real-data tests)")
def test_against_the_real_graph(tmp_path, capsys):
    """Drive the store with the ACTUAL production graph: walk every real node once, play
    every real clip once, and check the record answers real questions at real size."""
    g = json.loads(REAL.read_text(encoding="utf-8"))
    nodes, edges = g["nodes"], g["edges"]
    assert len(nodes) >= 250 and len(edges) >= 1400, "snapshot smaller than expected"

    per_char = {}
    t0 = time.time()
    for nid, n in nodes.items():
        ch = n.get("character") or nid.split(":")[0]
        per_char.setdefault(ch, lived.Lived(ch, base=tmp_path, now=0.0)).visit(nid, now=1000.0)
    for e in edges:
        ch = e.get("character") or (e.get("from") or "").split(":")[0]
        if ch in per_char:
            per_char[ch].play(e.get("id"), now=1000.0)
    elapsed = time.time() - t0

    total_nodes = sum(len(b.nodes) for b in per_char.values())
    total_edges = sum(len(b.edges) for b in per_char.values())
    assert total_nodes == len(nodes)
    assert total_edges <= len(edges)                       # ids may repeat across variants

    # the frontier question, asked against the real graph
    ph = per_char["phineas"]
    real_phineas = {n for n in nodes if n.startswith("phineas:")}
    assert ph.visited() == real_phineas
    assert ph.unplayed({e["id"] for e in edges if e.get("character") == "phineas"}) == set()

    # and the same store with NOTHING lived: every real pose is frontier
    empty = lived.Lived("phineas", base=tmp_path / "empty", now=0.0)
    assert empty.visited() == set()
    assert len(empty.unplayed({e["id"] for e in edges})) == len({e["id"] for e in edges})

    for b in per_char.values():
        assert b.flush(force=True, now=10.0) is True
    size = sum(lived.path_for(c, tmp_path).stat().st_size for c in per_char)

    with capsys.disabled():
        print("\n  REAL graph: %d nodes / %d edges across %d characters"
              % (len(nodes), len(edges), len(per_char)))
        for ch, b in sorted(per_char.items()):
            print("    %-9s %s" % (ch, b.totals()))
        print("    record cost: %.3fs to build, %.1f KB on disk" % (elapsed, size / 1024.0))
    assert elapsed < 5.0, "recording a full graph traversal took %.2fs" % elapsed
