"""release.py is the only thing between the private tree and what the world sees, and
it had no tests. These cover the gates, not the git plumbing: what must be refused, and
-- the part the bug was about -- that a refusal leaves nothing behind.

Every subprocess call is answered by a stub, so nothing here touches a real repo, runs
pytest, or writes a tag.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# scripts/ is an operator directory, not a package. Load by path rather than adding
# another sys.path entry the whole suite would inherit (same reason as the health tests).
_spec = importlib.util.spec_from_file_location("lp_release", ROOT / "scripts" / "release.py")
release = importlib.util.module_from_spec(_spec)
sys.modules["lp_release"] = release
_spec.loader.exec_module(release)


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A fake project root with a VERSION and a CHANGELOG, and git answered by a stub."""
    (tmp_path / "VERSION").write_text("0.4.0\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("## 0.5.0\n- notes\n## 0.4.0\n", encoding="utf-8")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "LIFE", tmp_path)

    state = {"existing_tags": set(), "calls": []}

    def fake_run(cmd, **kw):
        state["calls"].append(cmd)
        out = ""
        if "tag" in cmd and "-l" in cmd:
            wanted = cmd[cmd.index("-l") + 1]
            out = wanted if wanted in state["existing_tags"] else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(release.subprocess, "run", fake_run)
    state["path"] = tmp_path
    return state


def _args(version, **kw):
    return types.SimpleNamespace(version=version, allow_dirty=True, skip_tests=True, **kw)


def _version(repo):
    return (repo["path"] / "VERSION").read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------- parse_version
@pytest.mark.parametrize("s,want", [
    ("0.4.0", (0, 4, 0)), ("1.0.0", (1, 0, 0)), ("0.10.2", (0, 10, 2)),
    ("0.4", None), ("0.4.0.1", None), ("a.b.c", None), ("0.4.x", None), ("", None),
])
def test_parse_version(s, want):
    """`v.count(".") == 2` accepted 'a.b.c' and '0.4.x'. Three integers, or nothing."""
    assert release.parse_version(s) == want


def test_ordering_is_numeric_not_lexical():
    """0.10.0 is above 0.9.0. As strings it is below, which is why this is parsed."""
    assert release.parse_version("0.10.0") > release.parse_version("0.9.0")


# ------------------------------------------------------------------------- the gates
def test_a_taken_tag_leaves_VERSION_alone(repo):
    """THE BUG. VERSION was written before the tag check, so a refused release still
    set it -- leaving the tree dirty for a release that never happened, which then
    tripped the clean-tree gate on the retry."""
    repo["existing_tags"].add(release.TAG_PREFIX + "0.5.0")
    assert release.cmd_release(_args("0.5.0")) == 1
    assert _version(repo) == "0.4.0", "a refused release modified VERSION"


def test_a_backwards_bump_is_refused(repo):
    """Only the tag's existence was checked, so 0.1.0 on a 0.4.0 tree succeeded as
    long as v0.1.0 had never been cut."""
    assert release.cmd_release(_args("0.1.0")) == 2
    assert _version(repo) == "0.4.0"


def test_the_same_version_again_is_refused(repo):
    assert release.cmd_release(_args("0.4.0")) == 2
    assert _version(repo) == "0.4.0"


def test_a_malformed_version_is_refused(repo):
    assert release.cmd_release(_args("0.4.x")) == 2
    assert _version(repo) == "0.4.0"


def test_missing_changelog_notes_leave_VERSION_alone(repo):
    """GATE 3 already ran before the write; this pins that it stays that way."""
    (repo["path"] / "CHANGELOG.md").write_text("## 0.4.0\n", encoding="utf-8")
    assert release.cmd_release(_args("0.5.0")) == 1
    assert _version(repo) == "0.4.0"


def test_a_good_release_writes_the_version_and_tags(repo):
    """The happy path still works. Without this the tests above are satisfied by a
    function that refuses everything."""
    assert release.cmd_release(_args("0.5.0")) == 0
    assert _version(repo) == "0.5.0"
    tagged = [c for c in repo["calls"] if "tag" in c and "-a" in c]
    assert tagged, "a successful release cut no tag"


def test_no_side_effect_precedes_a_refusal(repo):
    """The general form of the bug, rather than the one instance of it: a refused
    release must not have run `git add`, `git commit` or `git tag -a` either."""
    repo["existing_tags"].add(release.TAG_PREFIX + "0.5.0")
    release.cmd_release(_args("0.5.0"))
    for c in repo["calls"]:
        assert "add" not in c, "staged something before refusing"
        assert "commit" not in c, "committed before refusing"
        assert not ("tag" in c and "-a" in c), "tagged before refusing"
