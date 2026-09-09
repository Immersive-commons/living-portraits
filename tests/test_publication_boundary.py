"""A published file must not point at something the reader cannot open.

This repo is a curated SUBSET of a private tree (scripts/release.py owns the rules), so
a path that resolves for the author can resolve for nobody else. Four internal documents
were cited from published files by nine sources before this test existed -- including
`_bakeoff/README.md`, which `pipeline/hf_gen.py` names as the justification for every
model choice, and which lived in an EXCLUDE_DIR.

Neither existing check catches this. `test_doc_citations` resolves `file.py:line`
against symbols; `test_codemap` checks the viewer's own rows. Neither looks at a bare
path in prose or a docstring.

WHEN A REFERENCE IS DELIBERATE, SAY SO. Some things genuinely are not here -- an internal
audit, a hosting proposal, a file that lives only on the wall host. Mark those with
`(internal)`, `(not in this repo)` or `(created at runtime)` on the same line and
this test accepts them. The
contract is not "never mention an absent file"; it is "never mention one silently".
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCAN_DIRS = ("runtime", "director", "pipeline", "health", "scripts", "tests",
             "install", "prompts")
SCAN_EXT = (".py", ".md", ".yaml", ".yml", ".ps1")

# Top-level names this repo owns or knowingly excludes. Restricting to these is what
# keeps the check from flagging `ControlNetModel/config.json` (a HuggingFace repo id),
# `sam3/__init__.py` (a package), or `path/to/file.py` (a docstring example).
OWNED = {"runtime", "director", "pipeline", "health", "scripts", "tests", "install",
         "prompts", "_research", "_bakeoff", "_audit", "_hosting", "deck", "research",
         "midjourney", "sessions"}

# `data/` is gitignored by design and seeded at runtime -- absence there is the
# documented state, not a broken reference.
IGNORE_FIRST = {"data", "sessions"}

DISCLOSED = re.compile(
    r"\(internal\)|\(not in this repo\)|not in\s+(?:the\s+)?repo|\(created at runtime\)",
    re.I)

PATH_REF = re.compile(
    r"(?<![\w:/.-])((?:\.\./)?[_a-zA-Z][\w.-]*(?:/[\w.-]+)+"
    r"\.(?:md|py|yaml|yml|json|txt|ps1|sh|bat|html|tsv|svg))")


def _files():
    out = []
    for d in SCAN_DIRS:
        out += [p for p in (ROOT / d).rglob("*")
                if p.suffix in SCAN_EXT and "vendor" not in p.parts]
    out += list(ROOT.glob("*.md")) + list(ROOT.glob("*.py"))
    return out


def dangling():
    """{referenced-path: {citing files}} for references this clone cannot resolve."""
    bad = {}
    for p in _files():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            for m in PATH_REF.finditer(line):
                raw = m.group(1)
                rel = raw[3:] if raw.startswith("../") else raw
                first = rel.split("/", 1)[0]
                if first in IGNORE_FIRST or first not in OWNED:
                    continue
                if (ROOT / rel).exists() or DISCLOSED.search(line):
                    continue
                bad.setdefault(rel, set()).add(p.relative_to(ROOT).as_posix())
    return bad


def test_no_published_file_points_at_something_absent():
    bad = dangling()
    assert not bad, "unresolvable references:\n" + "\n".join(
        "  %s  <- %s" % (t, ", ".join(sorted(s))) for t, s in sorted(bad.items()))


def test_the_scan_can_see_something():
    """A guard on the guard. If the regex or OWNED ever stops matching, this check
    passes vacuously and the whole class goes invisible again -- which is exactly the
    failure it was written for."""
    seen = set()
    for p in _files():
        for m in PATH_REF.finditer(p.read_text(encoding="utf-8", errors="replace")):
            rel = m.group(1)
            if rel.split("/", 1)[0] in OWNED:
                seen.add(rel)
    assert len(seen) > 50, "only %d owned path references found -- the scan is broken" % len(seen)
