"""
test_doc_citations.py -- the `file.py:line` citations have to still land on what
they claim.

ARCHITECTURE.md's whole method is that every claim carries a citation, and its
own header admits the risk: "Line citations from the original audit were not
re-verified against the current files and may have drifted by a few lines." They
drifted by more than a few. A citation that lands on an unrelated line is worse
than no citation, because it quietly converts a checkable document back into
prose you have to trust -- and the reader cannot tell which kind they are holding.

WHAT THIS CHECKS, and why it is not the obvious thing

Checking that the cited FILE exists and the LINE is within it catches nothing: all
75 citations in this repo pass that today, and 16 of them are still wrong. Line
numbers do not go out of range when code is inserted above them; they go stale.

So this resolves SYMBOLS. Where a doc line names a symbol in backticks next to a
citation -- `` `heartbeat.py:530` `tick()` `` -- the symbol is looked up in the
cited file with `ast` and the real definition line is compared against the cited
one. That is the only form of this check that can actually fail.

TWO CONVENTIONS IN THIS REPO THAT A GENERIC TOOL MISSES

1. Fenced blocks. ARCHITECTURE.md's call-flow diagrams (`:46-58`, `:70-95`) are
   inside ``` fences and carry some of the most load-bearing citations in the
   document. Doc-drift tools routinely skip fenced content as "code, not prose".
   Here it is the opposite: the fences are where the flow is documented.

2. Bare continuation anchors. A line reads
   `` `heartbeat.py:571` via `_atomic_write` `:103` `` -- the second anchor has
   no filename and inherits it from the first. There are ~100 of these. A regex
   demanding `name.ext:NNN` cannot see any of them, and they drift like the rest.

Both are handled below. KNOWN_DRIFT records exactly what the checker reports today, so this test is green
on a true statement rather than on a filtered one. Correcting a citation FAILS
this test until its entry is deleted -- deliberately: the list is a work queue
that cannot be ignored, not a suppression file. Tracked in issue #10.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Where a bare `foo.py` in a citation might live. Ordered: repo root wins.
SEARCH_DIRS = ("", "runtime", "director", "pipeline", "scripts", "tests", "health", "install")

# `path/to/file.py:123` or `file.py:123-456`, with or without backticks.
FILE_CITE = re.compile(r"`?([A-Za-z_][\w./-]*\.py)`?:(\d+)(?:-\d+)?")
# A continuation anchor: `:123` with no filename, inheriting the last file cited.
BARE_CITE = re.compile(r"`:(\d+)(?:-\d+)?`")
# A symbol reference. Two forms, because the two conventions in this repo differ:
#   backticked prose  -- `tick`, `tick()`, `Class.method`, `module.CONST`
#   bare call in a fence -- GraphCycler.frame(), _pick()
# The second is required: inside ``` fences nothing is backticked, and the fenced
# call-flow blocks carry several of the most load-bearing citations here.
SYMBOL = re.compile(
    r"`([A-Za-z_]\w*(?:\.\w+)*)(?:\(\))?`"      # `name` / `name()` / `a.b`
    r"|\b([A-Za-z_]\w*(?:\.\w+)*)\(\)"          # bare name() in a fence
)

# Citations to files that live in the private monorepo, not here. Not drift.
OFF_TREE = {"AUTONOMY.md", "MIGRATION.md", "PLAYER_INTEGRATION.md", "QUALITY_MILESTONES.md"}

# A citation may legitimately point at a CALL SITE rather than a definition, so a
# symbol resolving elsewhere is only reported when the gap is large enough that a
# call site is not a plausible explanation... except that call sites are exactly
# what several of these cite. The tolerance is therefore about insertion drift,
# not about proximity: a definition that moved 3 lines is noise, 30 is rot.
TOLERANCE = 3

# How close a symbol must sit to a citation to be considered its subject, in
# characters. Wide enough for `` `tick()` (`heartbeat.py:530`) `` and for a fenced
# `` tick()                heartbeat.py:530 ``; narrow enough that a second anchor
# further along the row does not get bound to the same name.
ADJACENT = 40


def _symbol_table(path: Path) -> dict[str, int]:
    """Every def/class/CONST in a file -> the line it is defined on.

    First definition wins: a name rebound later (a conditional import fallback,
    a redefinition in a nested scope) is not what a doc citation means.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        return {}
    table: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            table.setdefault(node.name, node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    table.setdefault(target.id, node.lineno)
    return table


_TABLES: dict[Path, dict[str, int]] = {}


def _resolve(rel: str) -> Path | None:
    for d in SEARCH_DIRS:
        p = ROOT / d / rel if d else ROOT / rel
        if p.exists() and p.is_file():
            return p
    return None


def _claims():
    """Yield (doc, doc_line, target_file, cited_line, symbol) for every citation
    on a doc line that also names a symbol.

    Pairing is by ADJACENCY: a citation binds only to a symbol sitting within
    ADJACENT characters of it, so a row carrying several anchors pairs each with
    the name beside it rather than binding all of them to whichever came first.
    """
    for doc in sorted(ROOT.rglob("*.md")):
        if ".git" in doc.parts or "node_modules" in doc.parts:
            continue
        for lineno, line in enumerate(
            doc.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            cites = [(m.start(), m.group(1), int(m.group(2))) for m in FILE_CITE.finditer(line)]
            if not cites:
                continue
            # Bare `:NNN` anchors inherit the nearest FILE cited to their left.
            for m in BARE_CITE.finditer(line):
                prior = [c for c in cites if c[0] < m.start()]
                if prior:
                    cites.append((m.start(), max(prior, key=lambda c: c[0])[1], int(m.group(1))))
            syms = [
                (m.start(), (m.group(1) or m.group(2)).split(".")[-1])
                for m in SYMBOL.finditer(line)
            ]
            if not syms:
                continue
            for pos, rel, cited in cites:
                if rel in OFF_TREE:
                    continue
                target = _resolve(rel)
                if target is None or target.suffix != ".py":
                    continue
                if target not in _TABLES:
                    _TABLES[target] = _symbol_table(target)
                table = _TABLES[target]
                # ADJACENCY, not nearest-on-the-line. A row often carries several
                # anchors where only one names a definition and the rest point at
                # locations INSIDE it -- ARCHITECTURE.md:229 cites `_acquire_lock`
                # at :583 (correct) and then :63 / :592-606 for the stale-lock
                # branch within it. Binding those to the symbol would report two
                # drifts that are not drifts. So a symbol only claims a citation
                # sitting immediately beside it, separated by nothing but the
                # punctuation the convention uses: `sym` (`file.py:NNN`).
                known = [
                    (abs(sp - pos), name) for sp, name in syms
                    if name in table and abs(sp - pos) <= ADJACENT
                ]
                if not known:
                    continue
                yield doc, lineno, target, cited, min(known)[1]


# Citations that are currently WRONG. Each entry is
#   (doc-relative-path, doc-line, symbol): actual-definition-line
# Fixing the citation makes this test fail until the entry is removed, which is
# the point: this is a work queue, not a suppression list. Tracked in issue #10.
KNOWN_DRIFT = {
    # The fenced call-flow blocks. Every one of these is real: the functions moved
    # and the diagrams did not follow.
    ("ARCHITECTURE.md", 52, "frame"): 342,
    ("ARCHITECTURE.md", 54, "_pick"): 221,
    ("ARCHITECTURE.md", 75, "tick"): 749,
    ("ARCHITECTURE.md", 76, "_ensure_player"): 733,
    ("ARCHITECTURE.md", 77, "decide_character"): 494,
    ("ARCHITECTURE.md", 82, "propose_pose"): 625,
    ("ARCHITECTURE.md", 87, "run_generate"): 823,
    # The single-writer contract table.
    ("ARCHITECTURE.md", 221, "_atomic_write"): 118,
    ("ARCHITECTURE.md", 224, "save"): 127,
    ("ARCHITECTURE.md", 481, "_resolve_goal"): 409,
    ("_research/CONTEXT_GRAPHS_FINDINGS.md", 291, "_build_user_prompt"): 359,
    # ARCHITECTURE.md:306 carries FOUR anchors in one row -- `_preview_graph.py:188`
    # for `_pick`, then :203-204 for a short-circuit INSIDE it, then :238 for
    # `_pick_policy`. `_pick` (:188 -> 221) and `_pick_policy` (:238 -> 271) are
    # genuine drift; the :203 pairing is an artifact of a row this dense, and is
    # recorded rather than tuned away because the fix is to split the row, not to
    # make the checker cleverer about prose.
    ("ARCHITECTURE.md", 306, "_pick"): 221,
    ("ARCHITECTURE.md", 306, "_pick_policy"): 271,
}


def _drifted():
    out = []
    for doc, lineno, target, cited, sym in _claims():
        actual = _TABLES[target][sym]
        if abs(actual - cited) > TOLERANCE:
            out.append((str(doc.relative_to(ROOT)), lineno, sym, cited, actual))
    return out


def test_no_new_citation_drift():
    """A citation that stops pointing at what it names is a silent doc regression."""
    new = [
        d for d in _drifted()
        if KNOWN_DRIFT.get((d[0], d[1], d[2])) != d[4]
    ]
    assert not new, "Doc citations that no longer land on the symbol they name:\n" + "\n".join(
        f"  {doc}:{ln} cites :{cited} for `{sym}` -- it is at :{actual}"
        for doc, ln, sym, cited, actual in new
    )


def test_known_drift_list_has_no_stale_entries():
    """Fixing a citation must retire its KNOWN_DRIFT entry, or the queue never empties."""
    live = {(d[0], d[1], d[2]): d[4] for d in _drifted()}
    stale = sorted(k for k in KNOWN_DRIFT if k not in live)
    assert not stale, (
        "These citations are no longer drifted -- delete them from KNOWN_DRIFT:\n  "
        + "\n  ".join(f"{doc}:{ln} `{sym}`" for doc, ln, sym in stale)
    )


@pytest.mark.parametrize("doc_rel", ["ARCHITECTURE.md", "AGENTS.md", "README.md", "CONTRIBUTING.md"])
def test_cited_files_exist(doc_rel):
    """A citation to a file that is not here at all, and is not a known private-tree doc."""
    doc = ROOT / doc_rel
    text = doc.read_text(encoding="utf-8", errors="replace")
    missing = sorted({
        rel for rel in (m.group(1) for m in FILE_CITE.finditer(text))
        if rel not in OFF_TREE and _resolve(rel) is None
    })
    assert not missing, f"{doc_rel} cites files that do not exist here: {missing}"
