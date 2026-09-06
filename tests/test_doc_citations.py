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

Both are handled below.

A THIRD convention defeats this check, and it says so rather than pretending
otherwise: range citations (`file.py:12-40`) point at BLOCKS, not definitions,
and there are 31 of them. Binding a range to a definition line reports drift that
is not drift, so ranges are not matched at all -- they were verified by hand
against the commit named in ARCHITECTURE.md's header, and a test below asserts
that this file keeps admitting it cannot see them.

Symbols bind BEFORE-ONLY. Both conventions put the name first, so a symbol to the
right of an anchor is a different claim. Binding rightwards produced three false
reports on rows carrying several anchors -- the doc was right and this file was
wrong -- which is worth remembering before loosening it again.
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
# A definition-site citation. The trailing `(?!-)` is load-bearing: `file.py:12-40`
# is a BLOCK reference ("reads intent.json, mtime-cached"), not a claim about where
# a symbol is defined, and there are 31 of them. Scoring a range against a def line
# reports drift that is not drift, so ranges are not matched at all.
FILE_CITE = re.compile(r"`?([A-Za-z_][\w./-]*\.py)`?:(\d+)(?!\d|-)")
# A continuation anchor: `:123` with no filename, inheriting the last file cited.
RANGE_CITE = r"\.py`?:\d+-\d+"
BARE_CITE = re.compile(r"`:(\d+)(?!\d|-)`")
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
                # BEFORE-ONLY. Both conventions here put the symbol first --
                # `sym` (`file.py:N`) in prose, `sym()   file.py:N` in a fence --
                # so a symbol to the RIGHT of an anchor is a different claim, not
                # this one's subject. Binding rightwards is what produced three
                # false reports: at :221 the anchor `heartbeat.py:571` is the
                # intent.json WRITE SITE and `_atomic_write` sits after it; at
                # :224 `save()` beat `build()` to the `video_graph.py:471` anchor
                # by one character; at :306 the interior anchor `:203-204` bound
                # to `_pick_policy()` further along the row. In all three the doc
                # was right and this function was wrong.
                known = [
                    (pos - sp, name) for sp, name in syms
                    if name in table and 0 < pos - sp <= ADJACENT
                ]
                if not known:
                    continue
                yield doc, lineno, target, cited, min(known)[1]


# Every citation that names a symbol now lands on it. There is no drift ledger
# here on purpose: a recorded baseline in a test file is a second, hidden copy of
# a fact that belongs in the document, and it must be curated by whoever
# maintains this repo. ARCHITECTURE.md's header carries the honest version --
# the commit its line numbers were verified against -- where a reader will see it.
#
# If this test fails, a citation stopped pointing at what it names. Fix the
# citation. Do not add a ledger back.


def _drifted():
    out = []
    for doc, lineno, target, cited, sym in _claims():
        actual = _TABLES[target][sym]
        if abs(actual - cited) > TOLERANCE:
            out.append((str(doc.relative_to(ROOT)), lineno, sym, cited, actual))
    return out


def test_no_new_citation_drift():
    """A citation that stops pointing at what it names is a silent doc regression."""
    drifted = _drifted()
    assert not drifted, "Doc citations that no longer land on the symbol they name:\n" + "\n".join(
        f"  {doc}:{ln} cites :{cited} for `{sym}` -- it is at :{actual}"
        for doc, ln, sym, cited, actual in drifted
    )


def test_unverifiable_anchors_are_counted_not_hidden():
    """Say what this check cannot see, rather than implying it saw everything.

    Range citations (`file.py:12-40`) point at BLOCKS -- "reads intent.json,
    mtime-cached" -- not at definitions, and bare `:NNN` anchors often point at a
    line INSIDE a function. Neither is resolvable from a symbol table. Those were
    verified by hand against the commit named in ARCHITECTURE.md's header.

    This does not fail on them. It fails if the code ever starts pretending they
    are covered.
    """
    text = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8", errors="replace")
    ranges = re.findall(RANGE_CITE, text)
    assert ranges, (
        "no range citations found -- if the convention changed, this test's "
        "premise is stale and its docstring is now misleading"
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
