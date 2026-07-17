"""test_prune_proposals.py -- the SAFE proposals.json janitor. No network, no MJ.

Covers: dry-run mutates nothing; --submit prunes ONLY failed; protected statuses survive;
a malformed record is never lost; backup is written on submit; the --older-than-days age
gate keeps recent failures; an undateable failed record is kept under an age gate.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import prune_proposals as pp


def _rec(status, label, epoch=None):
    """A minimal proposal record. id carries the trailing-epoch the tool dates on."""
    ep = int(epoch if epoch is not None else time.time())
    return {"character": "phineas", "label": label, "status": status,
            "id": "phineas_%s_%d" % (label, ep)}


def _write(tmp_path, records):
    p = tmp_path / "proposals.json"
    p.write_text(json.dumps({"proposals": records}, indent=2), encoding="utf-8")
    return p


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))["proposals"]


def test_dry_run_changes_nothing(tmp_path):
    p = _write(tmp_path, [_rec("failed", "a"), _rec("done", "b"), _rec("pending", "c")])
    before = p.read_bytes()
    res = pp.prune(p, submit=False, log=lambda *_: None)
    assert res["n_pruned"] == 1                      # it WOULD prune the one failed
    assert res["submitted"] is False
    assert res["backup"] is None
    assert p.read_bytes() == before                  # ... but the file is byte-identical


def test_submit_prunes_only_failed(tmp_path):
    recs = [_rec("failed", "a"), _rec("failed", "b"), _rec("done", "c"),
            _rec("pending", "d"), _rec("approved", "e"), _rec("generating", "f")]
    p = _write(tmp_path, recs)
    res = pp.prune(p, submit=True, log=lambda *_: None)
    assert res["n_pruned"] == 2
    left = _load(p)
    assert {r["status"] for r in left} == {"done", "pending", "approved", "generating"}
    assert len(left) == 4
    assert not any(r["status"] == "failed" for r in left)


def test_backup_written_on_submit(tmp_path):
    p = _write(tmp_path, [_rec("failed", "a"), _rec("done", "b")])
    res = pp.prune(p, submit=True, log=lambda *_: None)
    assert res["backup"] is not None
    bak = Path(res["backup"])
    assert bak.exists()
    # backup holds the ORIGINAL (failed still present); live file does not
    assert any(r["status"] == "failed" for r in _load(bak))
    assert not any(r["status"] == "failed" for r in _load(p))


def test_malformed_record_survives(tmp_path):
    # a non-dict, a dict with no status, and a status that isn't a string -- none are `failed`,
    # so all must survive even a --submit prune (and the run must not crash).
    recs = [_rec("failed", "a"), "i-am-not-a-dict", {"id": "x_y_1", "no": "status"},
            {"status": 123, "id": "z_z_2"}, _rec("done", "b")]
    p = _write(tmp_path, recs)
    res = pp.prune(p, submit=True, log=lambda *_: None)
    assert res["n_pruned"] == 1                      # only the real failed went
    left = _load(p)
    assert "i-am-not-a-dict" in left
    assert {"id": "x_y_1", "no": "status"} in left
    assert {"status": 123, "id": "z_z_2"} in left
    assert any(r == "done" for r in (x.get("status") if isinstance(x, dict) else None for x in left))


def test_older_than_days_keeps_recent(tmp_path):
    now = time.time()
    old = _rec("failed", "old", epoch=now - 30 * 86400)
    recent = _rec("failed", "recent", epoch=now - 1 * 86400)
    p = _write(tmp_path, [old, recent, _rec("done", "keep")])
    res = pp.prune(p, older_than_days=7, submit=True, log=lambda *_: None)
    assert res["n_pruned"] == 1                      # only the 30-day-old failed
    labels = {r["label"] for r in _load(p) if isinstance(r, dict)}
    assert labels == {"recent", "keep"}


def test_older_than_days_keeps_undateable_failed(tmp_path):
    # a failed record whose id has no trailing epoch can't be dated -> kept under an age gate.
    undateable = {"character": "phineas", "label": "weird", "status": "failed", "id": "no-epoch-here"}
    p = _write(tmp_path, [undateable])
    res = pp.prune(p, older_than_days=7, submit=True, log=lambda *_: None)
    assert res["n_pruned"] == 0
    assert len(_load(p)) == 1
