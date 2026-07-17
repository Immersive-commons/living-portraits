"""prune_proposals.py -- a SAFE standalone janitor for data/mind/proposals.json.

The autogen queue accumulates terminal `failed` records forever -- the state machine
never resurrects a `failed` proposal (run_generate only ever processes `approved`/`pending`
and resets crashed `generating` orphans back to `approved`; see pipeline/autogen.py
run_generate). So every `failed` row is pure dead weight that load_proposals() re-reads
on every queue/status/generate call. Most are historical: the 2026-06-17 Midjourney v8.1
break rejected `--oref` and dead-lettered ~166 stills in one stroke.

This tool prunes `failed` proposals. It NEVER removes pending / approved / generating /
done -- those are live or completed state. It is DRY-RUN by default and writes only with
--submit; before any write it drops a timestamped backup beside the file and then does an
atomic tmp+replace (the same write discipline as autogen._save). Every record is parsed
defensively: a malformed row (not a dict, missing/odd status, undateable id) is KEPT, never
pruned and never crashes the run.

Usage:
    python pipeline/prune_proposals.py <proposals.json>                  # dry-run, prune ALL failed
    python pipeline/prune_proposals.py <proposals.json> --submit         # actually write
    python pipeline/prune_proposals.py <proposals.json> --older-than-days 7
    python pipeline/prune_proposals.py <proposals.json> --older-than-days 7 --submit

stdlib only -- runs anywhere, including hil.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# A `failed` proposal is the ONLY prunable status. Live/terminal-good states are protected.
PRUNABLE = "failed"
PROTECTED = ("pending", "approved", "generating", "done")

# autogen builds ids as "<character>_<label>_<int(proposed_at)>" (see autogen.add_proposal),
# so the trailing 10-digit group is the proposal's creation epoch.
_ID_TS = re.compile(r"_(\d{9,11})$")


def _load(path):
    """Read proposals.json -> {"proposals": [...]}. Raises with a clear message on a file
    that isn't the expected shape (we refuse to operate on something we don't understand)."""
    raw = Path(path).read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("proposals"), list):
        raise ValueError("not a proposals file: expected {'proposals': [...]}, got %s"
                         % type(obj).__name__)
    return obj


def _save_atomic(path, obj):
    """Atomic write: tmp sibling + os.replace. Mirrors autogen._save so the on-disk shape
    (indent=2 json) is byte-for-byte the house style."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _backup(path):
    """Copy the file to a timestamped sibling BEFORE any write. Returns the backup path."""
    path = Path(path)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dst = path.with_name("%s.%s.bak" % (path.name, stamp))
    dst.write_bytes(path.read_bytes())
    return dst


def _status_of(rec):
    """Defensive status read. A non-dict or missing status -> None (treated as unknown =>
    PROTECTED, never pruned)."""
    if not isinstance(rec, dict):
        return None
    s = rec.get("status")
    return s if isinstance(s, str) else None


def _id_epoch(rec):
    """Best-effort creation epoch from the id's trailing timestamp. None if unparseable."""
    if not isinstance(rec, dict):
        return None
    m = _ID_TS.search(str(rec.get("id", "")))
    return int(m.group(1)) if m else None


def _should_prune(rec, cutoff_epoch):
    """True only for a `failed` record that also passes the age gate. Never prunes a
    protected/unknown status; never prunes an undateable record when an age gate is set."""
    if _status_of(rec) != PRUNABLE:
        return False
    if cutoff_epoch is None:
        return True                       # prune ALL failed
    ep = _id_epoch(rec)
    if ep is None:
        return False                      # can't date it -> keep it (conservative)
    return ep < cutoff_epoch


def _counts(props):
    """Counter of status across records (malformed/unknown bucketed as '<malformed>')."""
    c = Counter()
    for rec in props:
        s = _status_of(rec)
        c[s if s is not None else "<malformed>"] += 1
    return c


def prune(path, *, older_than_days=None, submit=False, log=print):
    """Run the prune. Returns a result dict: counts before/after, n_pruned, backup path.
    DRY-RUN unless submit=True."""
    obj = _load(path)
    props = obj["proposals"]
    before = _counts(props)

    cutoff = None
    if older_than_days is not None:
        cutoff = time.time() - older_than_days * 86400

    keep, pruned = [], []
    for rec in props:
        try:
            (pruned if _should_prune(rec, cutoff) else keep).append(rec)
        except Exception:
            keep.append(rec)              # any surprise -> KEEP; pruning must never lose live data

    after = _counts(keep)
    policy = ("failed older than %d days" % older_than_days) if older_than_days is not None \
        else "all failed"

    def _fmt(c):
        return ", ".join("%s=%d" % (k, c[k]) for k in sorted(c)) or "(empty)"

    log("file        : %s" % path)
    log("policy      : prune %s" % policy)
    log("before      : %d records  [%s]" % (len(props), _fmt(before)))
    log("would prune : %d  (status=failed%s)" % (
        len(pruned), "" if cutoff is None else " older than %dd" % older_than_days))
    log("after       : %d records  [%s]" % (len(keep), _fmt(after)))
    # invariant guard: protected statuses must be untouched
    for s in PROTECTED:
        if before.get(s, 0) != after.get(s, 0):
            raise AssertionError("BUG: protected status %r changed (%d -> %d); refusing to write"
                                 % (s, before.get(s, 0), after.get(s, 0)))

    backup = None
    if not submit:
        log("MODE        : DRY-RUN -- nothing written. Re-run with --submit to apply.")
    elif not pruned:
        log("MODE        : --submit, but nothing to prune. Leaving file untouched.")
    else:
        backup = _backup(path)
        log("backup      : %s" % backup)
        obj["proposals"] = keep
        _save_atomic(path, obj)
        log("MODE        : WROTE %d records (pruned %d)." % (len(keep), len(pruned)))

    return {"before": dict(before), "after": dict(after), "n_pruned": len(pruned),
            "policy": policy, "backup": str(backup) if backup else None, "submitted": submit}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prune terminal `failed` proposals from a "
                                             "living-portraits proposals.json (safe; dry-run default).")
    ap.add_argument("path", help="path to proposals.json")
    ap.add_argument("--older-than-days", type=int, default=None,
                    help="only prune failed proposals whose id-timestamp is older than N days "
                         "(default: prune ALL failed)")
    ap.add_argument("--submit", action="store_true",
                    help="actually write (with timestamped backup). Default is dry-run.")
    args = ap.parse_args(argv)

    p = Path(args.path)
    if not p.exists():
        print("error: no such file: %s" % p, file=sys.stderr)
        return 2
    try:
        prune(p, older_than_days=args.older_than_days, submit=args.submit)
    except (ValueError, json.JSONDecodeError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
