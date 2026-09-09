#!/usr/bin/env python3
"""release.py -- version control pipeline for Living Portraits.

The project lives INSIDE the life monorepo but publishes a curated subset to a
separate public repo. That split is the whole reason this script exists: before it,
the public subset was implicit (a one-off copy on 2026-07-16), nobody could say what
was in it, and it silently drifted 7 commits behind while continuing to document a
Midjourney path that now 403s on every call. Anyone who cloned it got a system that
could not run.

So the rules are HERE, in code, and `check` will tell you the truth about drift.

    python scripts/release.py manifest          # what IS public, per the rules
    python scripts/release.py check             # drift: private vs published
    python scripts/release.py sync              # copy the subset into the OSS tree
    python scripts/release.py release 0.3.0     # the gated pipeline

Nothing here pushes. Publishing to GitHub is a consequential act and stays a human
`git push` in the OSS repo, by design.
"""
from __future__ import annotations

import argparse
import filecmp
import fnmatch
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # projects/living-portraits
LIFE = ROOT.parent.parent                              # the monorepo root
OSS = Path(os.environ.get("LP_OSS_REPO", LIFE.parent / "living-portraits-oss"))
TAG_PREFIX = "living-portraits/v"                      # monorepo: tags MUST be namespaced

# ---------------------------------------------------------------- the public contract
# EXCLUDE wins over INCLUDE. Derived 2026-08-09 by diffing the existing public repo
# against the private tree, so these rules REPRODUCE what was already published rather
# than inventing a new policy -- `check` proves it.
EXCLUDE_DIRS = {
    "deck",        # talk material, not the system
    "_article",    # the published write-up + its assets
    "_audit",      # internal audit
    "_bakeoff",    # model comparison (5.6 MB page)
    "_hosting",    # hosting proposal
    "_shots",      # screenshots / montage experiments
    "data",        # runtime state, secrets, 7 GiB of clips
    "sessions",    # auth
    "midjourney",  # vendored client
    ".venv", ".venv-gen", "__pycache__", ".claude", ".git",
}
EXCLUDE_FILES = {
    # internal docs: roadmap/history/quality notes that describe unshipped intent
    "AUTONOMY.md", "MIGRATION.md", "PLAYER_INTEGRATION.md", "QUALITY_MILESTONES.md",
    # goals.yaml is the same class -- a rung ladder of unshipped intent, whose public
    # counterpart is ROADMAP.md. Nothing in the tree reads it, and it cites the private
    # project memory. It was published only because this list was derived on 2026-08-09
    # by diffing the repo as it then stood, and the file postdates that.
    "goals.yaml",
    # superseded diagrams -- only v4 is published
    "living-portraits-architecture.svg",
    "living-portraits-architecture-v2.svg",
    "living-portraits-architecture-v3.svg",
}
EXCLUDE_GLOBS = [
    "*.log", "*.bak", "*.tmp", "*.part", "*.session", ".env*",
    "*_captures.json", "install_genstack.done",
    "*.png", "*.jpg", "*.jpeg", "*.mp4",     # stills/renders are not source
]
# files that exist ONLY in the public repo and must never be clobbered or deleted by a
# sync. Verified 2026-08-09: LICENSE and NOTICE are the complete set -- .gitignore and
# README.md live in BOTH trees and ARE synced, so listing them here would make `check`
# report them as permanently missing.
OSS_ONLY = {"LICENSE", "NOTICE"}
# published even though a glob would drop them
# published even though a glob or an EXCLUDE_DIR would drop them
FORCE_INCLUDE = {"demo.gif", "living-portraits-architecture-v4.svg",
                 # hf_gen.py:16 cites this as the reason for every model choice, and
                 # _bakeoff/ is an EXCLUDE_DIR -- so the citation resolved for nobody
                 # reading the public repo. 4.8 KB of measurements, no credentials, no
                 # paths, no costs. The 5.6 MB rendered page stays excluded.
                 "_bakeoff/README.md"}


def _tracked() -> list[str]:
    """Git-tracked files under the project, repo-relative paths stripped to project-relative.
    Using git (not a walk) means .gitignore is honoured for free."""
    out = subprocess.run(["git", "-C", str(LIFE), "ls-files", "projects/living-portraits"],
                         capture_output=True, text=True, check=True).stdout
    prefix = "projects/living-portraits/"
    return sorted(p[len(prefix):] for p in out.splitlines() if p.startswith(prefix))


def is_public(rel: str) -> bool:
    parts = Path(rel).parts
    if rel in FORCE_INCLUDE:
        return True
    if parts[0] in EXCLUDE_DIRS:
        return False
    if rel in EXCLUDE_FILES or parts[-1] in EXCLUDE_FILES:
        return False
    return not any(fnmatch.fnmatch(parts[-1], g) for g in EXCLUDE_GLOBS)


def manifest() -> list[str]:
    return [p for p in _tracked() if is_public(p)]


# ---------------------------------------------------------------- commands
def cmd_manifest(_):
    m = manifest()
    for p in m:
        print(p)
    print(f"\n{len(m)} files would be published (of {len(_tracked())} tracked).",
          file=sys.stderr)


def cmd_check(_):
    """Drift report. Exit 1 if the public repo does not match the manifest."""
    if not OSS.exists():
        print(f"OSS repo not found at {OSS} (set LP_OSS_REPO)")
        return 2
    want = set(manifest())
    have = set(subprocess.run(["git", "-C", str(OSS), "ls-files"],
                              capture_output=True, text=True, check=True).stdout.split())
    have -= OSS_ONLY

    missing = sorted(want - have)          # in private, should be public, is not
    extra = sorted(have - want)            # published but no longer in the manifest
    changed = []
    for rel in sorted(want & have):
        a, b = ROOT / rel, OSS / rel
        if a.exists() and b.exists() and not filecmp.cmp(a, b, shallow=False):
            changed.append(rel)

    print(f"manifest: {len(want)} files | published: {len(have)}")
    for label, rows in (("MISSING from public", missing),
                        ("STALE in public (content differs)", changed),
                        ("EXTRA in public (not in manifest)", extra)):
        if rows:
            print(f"\n{label} ({len(rows)}):")
            for r in rows[:40]:
                print("  " + r)
            if len(rows) > 40:
                print(f"  ... and {len(rows) - 40} more")
    drift = len(missing) + len(changed) + len(extra)
    print(f"\n{'DRIFT: ' + str(drift) + ' file(s)' if drift else 'IN SYNC'}")
    return 1 if drift else 0


def cmd_sync(args):
    """Copy the manifest into the OSS working tree. Does not commit, does not push."""
    if not OSS.exists():
        print(f"OSS repo not found at {OSS}")
        return 2
    m = manifest()
    copied = 0
    for rel in m:
        src, dst = ROOT / rel, OSS / rel
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
            shutil.copy2(src, dst)
            copied += 1
            if args.verbose:
                print("  +", rel)
    # remove published files the manifest dropped (but never OSS-only files)
    have = set(subprocess.run(["git", "-C", str(OSS), "ls-files"],
                              capture_output=True, text=True, check=True).stdout.split())
    removed = 0
    for rel in sorted(have - set(m) - OSS_ONLY):
        p = OSS / rel
        if p.exists():
            p.unlink()
            removed += 1
            if args.verbose:
                print("  -", rel)
    print(f"synced {len(m)} files: {copied} written, {removed} removed")
    print(f"review with:  git -C {OSS} status")
    return 0


def _version() -> str:
    return (ROOT / "VERSION").read_text().strip()


def parse_version(v):
    """'0.4.1' -> (0, 4, 1). None if it is not three integers.

    `v.count(".") == 2` accepted "a.b.c" and "0.4.x", which then sorted as strings or
    not at all. Parsing is what lets the bump be compared rather than merely spelled.
    """
    parts = (v or "").split(".")
    if len(parts) != 3:
        return None
    try:
        return tuple(int(x) for x in parts)
    except ValueError:
        return None


def cmd_release(args):
    """Gated release: tests -> version -> changelog -> tag. Push stays human.

    EVERY gate runs BEFORE the first side effect. That ordering is the point: the tag
    check used to sit after the VERSION write, so `release 0.4.0` against an existing
    tag exited 1 having already set VERSION to 0.4.0 with no commit and no tag. The
    tree was then dirty for a release that never happened, `check` reported VERSION as
    STALE in public, and GATE 1 refused the retry -- a failure that made itself harder
    to recover from.
    """
    new = args.version
    parsed = parse_version(new)
    if parsed is None:
        print("version must be X.Y.Z, three integers")
        return 2

    # GATE 0: the bump must go UP. Only the tag's existence was checked, so `release
    # 0.1.0` on a 0.4.0 tree succeeded as long as v0.1.0 had never been cut -- and a
    # published version that moves backwards is very hard to undo.
    cur = ""
    try:
        cur = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    cur_parsed = parse_version(cur)
    if cur_parsed is not None and parsed <= cur_parsed:
        print(f"VERSION is {cur}; {new} is not an increase")
        return 2

    # GATE 1: the monorepo must be clean for THIS project (other projects may be dirty --
    # this tree runs several concurrent sessions, so a global clean check would never pass)
    dirty = subprocess.run(["git", "-C", str(LIFE), "status", "--porcelain",
                            "projects/living-portraits"],
                           capture_output=True, text=True, check=False).stdout.strip()
    if dirty and not args.allow_dirty:
        print("living-portraits has uncommitted changes:\n" + dirty)
        print("commit them first, or pass --allow-dirty")
        return 1

    # GATE 2: tests
    if not args.skip_tests:
        print("running tests...")
        r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT, check=False)
        if r.returncode != 0:
            print("tests failed -- not releasing")
            return 1

    # GATE 3: the changelog must actually mention this version. A release whose notes
    # say nothing is how a changelog becomes decoration.
    ch = (ROOT / "CHANGELOG.md")
    if ch.exists() and f"## {new}" not in ch.read_text(encoding="utf-8"):
        print(f"CHANGELOG.md has no '## {new}' section -- write the notes first")
        return 1

    # GATE 4: the tag must be free. Checked BEFORE anything is written -- see the
    # docstring. This was the one gate that ran after a side effect.
    tag = TAG_PREFIX + new
    if subprocess.run(["git", "-C", str(LIFE), "tag", "-l", tag],
                      capture_output=True, text=True, check=False).stdout.strip():
        print(f"tag {tag} already exists -- bump the version or delete the tag")
        return 1

    # ---------- first side effect below this line ----------
    (ROOT / "VERSION").write_text(new + "\n")
    subprocess.run(["git", "-C", str(LIFE), "add",
                    "projects/living-portraits/VERSION",
                    "projects/living-portraits/CHANGELOG.md"], check=True)
    # VERSION/CHANGELOG may already be committed (they are written before the release
    # is cut, so the notes can be reviewed). An empty commit is not an error here --
    # the tag is the artifact, the commit is just where it points.
    staged = subprocess.run(["git", "-C", str(LIFE), "diff", "--cached", "--quiet"], check=False).returncode
    if staged:
        subprocess.run(["git", "-C", str(LIFE), "commit", "-m",
                        f"living-portraits: release {new}"], check=True)
    else:
        print("VERSION/CHANGELOG already committed -- tagging the current commit")
    subprocess.run(["git", "-C", str(LIFE), "tag", "-a", tag, "-m",
                    f"Living Portraits {new}"], check=True)
    print(f"\ntagged {tag}")
    print("next:  python scripts/release.py sync   # then review + commit in the OSS repo")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Living Portraits version-control pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest", help="list the files the public repo should contain")
    sub.add_parser("check", help="report drift between private and published")
    s = sub.add_parser("sync", help="copy the manifest into the OSS working tree")
    s.add_argument("-v", "--verbose", action="store_true")
    r = sub.add_parser("release", help="gated version bump + tag")
    r.add_argument("version")
    r.add_argument("--skip-tests", action="store_true")
    r.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    return {"manifest": cmd_manifest, "check": cmd_check,
            "sync": cmd_sync, "release": cmd_release}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
