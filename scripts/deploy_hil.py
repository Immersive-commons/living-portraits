#!/usr/bin/env python3
"""deploy_hil.py -- ship this project to the production host, verifiably.

WHY THIS EXISTS. Deploys were `scp` of whichever files a session happened to touch, so
nobody could answer the only question that matters during an incident: *what is actually
running on hil?* The first audit that asked it (2026-08-10) found the production box had
been running for months WITHOUT `prompts/characters/maxx.json` -- so one of the two live
characters had no archetype, no traits and no big-five temperament reaching the model at
all -- plus a stale `gallery.yaml` and `bedtime_routine.json`. None of that was visible
from either side.

WHAT IT DOES.
  1. Takes the deployable set from `git ls-files` -- so the repo, not a human's memory,
     decides what production consists of.
  2. Hashes both sides and ships only what differs (the tree is 20 GB, 13 GB of it clips;
     a blind copy is not an option).
  3. Writes `DEPLOYED.json` on the host recording the life-repo SHA, when, by whom, and
     every file changed.
  4. Commits the result to a git repo ON the host, so the box carries its own history and
     `git status` there reveals any hand-edit made during an incident.

The host tree is NOT a clone of this monorepo -- it is the project subdirectory at its own
root, with 13 GB of gitignored clips. So the host repo is a deployment LEDGER (what is
running, when it arrived, what changed) rather than a mirror. That is the honest shape:
`git log` on the host answers "what changed and when", and `git diff` answers "did someone
edit production by hand".

    python scripts/deploy_hil.py --dry-run        # what would change (default: SAFE)
    python scripts/deploy_hil.py --apply          # ship it + commit on the host
    python scripts/deploy_hil.py --apply --restart  # ...and restart the live tasks
    python scripts/deploy_hil.py --status         # what is running, and has it been touched

CONSEQUENTIAL: --apply writes to a live art installation. Dry-run is the default and prints
the exact file list first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIFE = ROOT.parent.parent
HOST = os.environ.get("LP_HOST", "hil")
REMOTE = os.environ.get("LP_REMOTE_ROOT", "C:/living-portraits")

# Directories that exist in the repo but are NOT part of a running installation: talk
# material, audits, research, screenshots, and the 13 GB of runtime state that lives only
# on the host. Mirrors scripts/release.py's EXCLUDE_DIRS in spirit -- different question
# (what does production need) but the same discipline: a rule, not a hand-kept list.
EXCLUDE_DIRS = {"data", "sessions", "deck", "_article", "_audit", "_research", "_bakeoff",
                "_hosting", "_shots", "__pycache__", ".venv", ".venv-gen", ".claude", ".git"}
EXCLUDE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".svg", ".log"}
LEDGER = "DEPLOYED.json"


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def deployable():
    """The file list, from git. Sorted so a deploy is reproducible."""
    out = _run(["git", "-C", str(LIFE), "ls-files", "projects/living-portraits"]).stdout.split("\n")
    rels = []
    for line in out:
        line = line.strip()
        if not line:
            continue
        rel = line.split("projects/living-portraits/", 1)[-1]
        parts = Path(rel).parts
        if parts and parts[0] in EXCLUDE_DIRS:
            continue
        if Path(rel).suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        if rel == ".gitignore":
            continue        # host-managed: ensure_repo() writes repo rules + host-only ones
        if (ROOT / rel).is_file():
            rels.append(rel)
    return sorted(rels)


def local_hashes(rels):
    return {r: hashlib.sha256((ROOT / r).read_bytes()).hexdigest()[:16] for r in rels}


_REMOTE_HASH = r'''
import hashlib, json, io, os, sys
root = sys.argv[1]
rels = json.loads(io.open(sys.argv[2], encoding="utf-8").read())
out = {}
for rel in rels:
    p = os.path.join(root, rel.replace("/", os.sep))
    try:
        with open(p, "rb") as f:
            out[rel] = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        out[rel] = None
io.open(sys.argv[3], "w", encoding="utf-8").write(json.dumps(out))
print("OK")
'''


def remote_hashes(rels):
    """Hash the host's copy of the same paths. One round trip, not one per file."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "rels.json").write_text(json.dumps(rels), encoding="utf-8")
    (tmp / "_hash.py").write_text(_REMOTE_HASH, encoding="utf-8")
    for f in ("rels.json", "_hash.py"):
        r = _run(["scp", "-q", str(tmp / f), "%s:%s/_deploy_%s" % (HOST, REMOTE, f)])
        if r.returncode:
            raise SystemExit("scp to %s failed: %s" % (HOST, r.stderr.strip()))
    py = "%s/.venv/Scripts/python.exe" % REMOTE
    r = _run(["ssh", HOST, "%s %s/_deploy__hash.py %s %s/_deploy_rels.json %s/_deploy_out.json"
              % (py, REMOTE, REMOTE, REMOTE, REMOTE)])
    if "OK" not in r.stdout:
        raise SystemExit("remote hash failed: %s%s" % (r.stdout, r.stderr))
    _run(["scp", "-q", "%s:%s/_deploy_out.json" % (HOST, REMOTE), str(tmp / "out.json")])
    got = json.loads((tmp / "out.json").read_text(encoding="utf-8"))
    _run(["ssh", HOST, "cmd /c del %s\\_deploy_*.json %s\\_deploy__hash.py"
          % (REMOTE.replace("/", "\\"), REMOTE.replace("/", "\\"))])
    return got


def plan(rels, loc, rem):
    missing = [r for r in rels if rem.get(r) is None]
    changed = [r for r in rels if rem.get(r) is not None and rem[r] != loc[r]]
    return missing, changed


def push(files):
    """scp each file, creating parent dirs on the host as needed."""
    dirs = sorted({str(Path(f).parent).replace("\\", "/") for f in files if Path(f).parent != Path(".")})
    for d in dirs:
        _run(["ssh", HOST, 'cmd /c if not exist "%s\\%s" mkdir "%s\\%s"'
              % (REMOTE.replace("/", "\\"), d.replace("/", "\\"),
                 REMOTE.replace("/", "\\"), d.replace("/", "\\"))])
    sent = 0
    for f in files:
        r = _run(["scp", "-q", str(ROOT / f), "%s:%s/%s" % (HOST, REMOTE, f)])
        if r.returncode:
            print("  FAILED %s: %s" % (f, r.stderr.strip()))
        else:
            sent += 1
    return sent


def life_sha():
    return _run(["git", "-C", str(LIFE), "rev-parse", "--short", "HEAD"]).stdout.strip()


def ensure_repo():
    """git init on the host, once. The ledger is worthless if it starts after the drift.

    The .gitignore goes down BEFORE `git init`, and `.gitignore` is excluded from the
    deployable set so a later push can never clobber it. Get that order wrong and the very
    first `git add -A` commits 13 GB of clips -- which is exactly what the first draft of
    this function did."""
    r = _run(["ssh", HOST, 'cmd /c if exist "%s\\.git" (echo REPO) else (echo NONE)'
              % REMOTE.replace("/", "\\")])
    if "REPO" in r.stdout:
        return False
    # the repo's own ignore rules, plus what only exists on a running host.
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8") + (
        "\n# --- host-only (added by scripts/deploy_hil.py) ---\n"
        ".venv-gen/\nsessions/\n*.pyc\n_snap*/\n_deploy_*\n"
        "midjourney/\nshot_*.png\nplayer.log\n")
    tmp = Path(tempfile.mkdtemp()) / "gitignore"
    tmp.write_text(gi, encoding="utf-8")
    _run(["scp", "-q", str(tmp), "%s:%s/.gitignore" % (HOST, REMOTE)])
    for cmd in ("git init -q -b main",
                'git config user.email "deploy@living-portraits"',
                'git config user.name "hil deploy"',
                # a push into a checked-out branch would otherwise be refused; this makes the
                # host able to accept one later without a bare mirror.
                "git config receive.denyCurrentBranch updateInstead"):
        _run(["ssh", HOST, "cd /d %s && %s" % (REMOTE.replace("/", "\\"), cmd)])
    return True


def host_commit(message):
    cmds = "cd /d %s && git add -A && git commit -q -m \"%s\" || echo NOTHING_TO_COMMIT" % (
        REMOTE.replace("/", "\\"), message.replace('"', "'"))
    return _run(["ssh", HOST, cmds]).stdout.strip()


def status():
    print("host: %s   root: %s" % (HOST, REMOTE))
    r = _run(["ssh", HOST, "cd /d %s && git log --oneline -5" % REMOTE.replace("/", "\\")])
    print("\n-- deploy history on the host --\n" + (r.stdout.strip() or "(no repo yet)"))
    r = _run(["ssh", HOST, "cd /d %s && git status --porcelain" % REMOTE.replace("/", "\\")])
    dirty = [l for l in r.stdout.split("\n") if l.strip()]
    print("\n-- hand-edits on production since the last deploy --")
    print("\n".join(dirty) if dirty else "(none -- production matches its last deploy)")
    r = _run(["ssh", HOST, "cd /d %s && type %s" % (REMOTE.replace("/", "\\"), LEDGER)])
    if r.stdout.strip():
        try:
            d = json.loads(r.stdout)
            print("\n-- last deploy --\n  life sha %s at %s (%d files)"
                  % (d.get("life_sha"), d.get("at"), len(d.get("files", []))))
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="actually ship (default is dry-run)")
    ap.add_argument("--restart", action="store_true", help="restart lp-mind + lp-preview after")
    ap.add_argument("--status", action="store_true", help="what is running + hand-edit check")
    ap.add_argument("--message", default=None)
    args = ap.parse_args()

    if args.status:
        status()
        return 0

    rels = deployable()
    loc = local_hashes(rels)
    print("deployable set: %d files (from git ls-files)" % len(rels))
    rem = remote_hashes(rels)
    missing, changed = plan(rels, loc, rem)
    print("  identical : %d" % (len(rels) - len(missing) - len(changed)))
    print("  MISSING   : %d" % len(missing))
    for f in missing:
        print("      + %s" % f)
    print("  CHANGED   : %d" % len(changed))
    for f in changed:
        print("      ~ %s" % f)

    if not missing and not changed:
        print("\nproduction already matches the repo.")
        return 0
    if not args.apply:
        print("\nDRY RUN -- nothing sent. Re-run with --apply to ship.")
        return 0

    fresh = ensure_repo()
    if fresh:
        print("\ninitialised a git repo on %s (baseline = the tree as found, BEFORE this deploy)" % HOST)
        print("  " + (host_commit("baseline: production tree as found, before the first tracked deploy") or ""))

    sent = push(missing + changed)
    sha = life_sha()
    ledger = {"life_sha": sha, "at": _run(["git", "-C", str(LIFE), "log", "-1", "--format=%cI"]).stdout.strip(),
              "host": HOST, "files": sorted(missing + changed)}
    tmp = Path(tempfile.mkdtemp()) / LEDGER
    tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    _run(["scp", "-q", str(tmp), "%s:%s/%s" % (HOST, REMOTE, LEDGER)])
    print("\nsent %d/%d files" % (sent, len(missing + changed)))
    print(host_commit(args.message or "deploy %s: %d file(s)" % (sha, sent)))

    if args.restart:
        print("\nrestarting lp-mind + lp-preview...")
        _run(["ssh", HOST, "powershell -NoProfile -Command \"Stop-ScheduledTask lp-preview; "
                           "Stop-ScheduledTask lp-mind; Get-Process pythonw -EA SilentlyContinue | "
                           "Stop-Process -Force; Start-Sleep 3; Start-ScheduledTask lp-mind; "
                           "Start-ScheduledTask lp-preview\""])
        print("restarted. Verify the panels before walking away.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
