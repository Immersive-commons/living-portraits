"""install/ENVIRONMENT.md has to stay honest in both directions.

A documentation file with no test is a snapshot of the day someone wrote it. Twenty-one
of the twenty-eight variables here were documented nowhere at all, which is what the
file fixes; this is what stops it happening again.

Both directions matter, and for different reasons. A variable the code reads and the
doc omits is the original problem. A variable the doc describes and the code no longer
reads is worse in a quiet way: it reads as current, and the reader has no way to tell.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "install" / "ENVIRONMENT.md"

# Explicit directories, never a walk-and-filter. The first version of this scan used
# `p.as_posix().lstrip("./")` to strip the leading path, which also strips the dot from
# `.venv` -- so the skip list never matched and it reported 85 variables, most of them
# numpy's and pytest's internals.
CODE_DIRS = ("runtime", "director", "pipeline", "health", "scripts", "tests", "install")

ENV_READ = re.compile(
    r"""os\.(?:environ\.get|getenv|environ\.setdefault)\(\s*["']([A-Z_0-9]+)["']"""
    r"""|os\.environ\[\s*["']([A-Z_0-9]+)["']\s*\]""")

# Names that appear in backticks in the doc but are not variables the code reads.
NOT_A_VARIABLE = {"SECRET"}


def _code_files():
    out = []
    for d in CODE_DIRS:
        out += [p for p in (ROOT / d).rglob("*.py") if "vendor" not in p.parts]
    return out + list(ROOT.glob("*.py"))


def read_by_code():
    """{VAR: {files}} for every variable the project's own code reads."""
    found = {}
    for p in _code_files():
        for m in ENV_READ.finditer(p.read_text(encoding="utf-8", errors="replace")):
            found.setdefault(m.group(1) or m.group(2), set()).add(
                p.relative_to(ROOT).as_posix())
    return found


def named_in_doc():
    text = DOC.read_text(encoding="utf-8")
    return set(re.findall(r"`([A-Z][A-Z0-9_]{2,})`", text)) - NOT_A_VARIABLE


def test_the_doc_exists():
    assert DOC.exists(), "install/ENVIRONMENT.md is the single place this is written down"


def test_every_variable_the_code_reads_is_documented():
    missing = sorted(set(read_by_code()) - named_in_doc())
    assert not missing, (
        "read by the code, absent from install/ENVIRONMENT.md: %s" % ", ".join(missing))


def test_the_doc_names_nothing_the_code_stopped_reading():
    """A stale row reads as current and the reader cannot tell. Delete it, or say it
    is historical outside backticks."""
    stale = sorted(named_in_doc() - set(read_by_code()))
    assert not stale, (
        "documented but read nowhere: %s" % ", ".join(stale))


def test_the_scan_did_not_wander_into_site_packages():
    """A guard on the guard. If this scan ever picks up the venv it will report ~85
    variables, the doc will look hopelessly incomplete, and the real signal is lost."""
    found = read_by_code()
    assert 20 <= len(found) <= 45, (
        "expected the project's own variables, got %d -- check CODE_DIRS" % len(found))
    for var, files in found.items():
        for f in files:
            assert not f.startswith(".venv"), "%s came from the venv (%s)" % (var, f)
