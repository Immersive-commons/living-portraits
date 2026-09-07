"""
test_gallery.py -- the cast registry + add-character front door (gallery.py).

Covers the registry CRUD + yaml round-trip, the one-character-per-panel invariant
(collision raises), panel normalization, the character-md authoring (written once,
never clobbered), and add_character's build FENCE -- exercised offline by injecting
a fake orchestrate module (the module's own _FakeOrchestrate) so the heavy chain
never runs and the build is reported DEFERRED.

Needs pyyaml (gallery.py imports it at module load). Skips on a box without it.
All tests run against a tmp gallery.yaml + a tmp CHARS_DIR -- the real roster and
prompts/characters/ are never written.
"""
from __future__ import annotations

import pytest

from conftest import HAVE_YAML

pytestmark = pytest.mark.skipif(not HAVE_YAML, reason="gallery.py imports pyyaml at module load")

import gallery
from gallery import (
    CastEntry, Gallery, _FakeOrchestrate, _norm_panel,
    add_character, assign_panel, list_cast,
)


@pytest.fixture
def gpath(tmp_path):
    return tmp_path / "gallery.yaml"


@pytest.fixture
def tmp_chars(tmp_path, monkeypatch):
    """Redirect gallery.CHARS_DIR at a tmp dir so add_character's .md writes never
    litter the real prompts/characters/. Returns the dir."""
    d = tmp_path / "characters"
    monkeypatch.setattr(gallery, "CHARS_DIR", d)
    return d


# --------------------------------------------------------------------------- #
# CastEntry + panel normalization
# --------------------------------------------------------------------------- #
def test_norm_panel_accepts_a_b_none():
    assert _norm_panel("A") == "A"
    assert _norm_panel("b") == "B"          # case-insensitive
    assert _norm_panel(None) is None
    assert _norm_panel("none") is None
    assert _norm_panel("") is None


def test_norm_panel_rejects_garbage():
    with pytest.raises(ValueError):
        _norm_panel("C")
    with pytest.raises(ValueError):
        _norm_panel("left")


def test_cast_entry_round_trip_dict():
    e = CastEntry(slug="x", name="X", panel="A", theme="a theme",
                  persona="a persona", voice="en_GB-alan-medium", mood="grand",
                  created_at="2026-01-01T00:00:00")
    e2 = CastEntry.from_dict(e.to_dict())
    assert e2 == e


# --------------------------------------------------------------------------- #
# Registry load / save round-trip
# --------------------------------------------------------------------------- #
def test_load_absent_is_empty(gpath):
    g = Gallery.load(gpath)
    assert g.cast == []


def test_save_load_round_trip(gpath):
    g = Gallery(path=gpath)
    g.upsert(CastEntry(slug="phineas", name="Phineas", panel="A", theme="study"))
    g.upsert(CastEntry(slug="seraphina", name="Seraphina", panel="B", theme="loft"))
    g.save()

    g2 = Gallery.load(gpath)
    assert [e.slug for e in g2.cast] == ["phineas", "seraphina"]
    assert g2.get("phineas").panel == "A"
    # byte-stable across a second round-trip
    d1 = [e.to_dict() for e in g2.cast]
    g2.save()
    assert [e.to_dict() for e in Gallery.load(gpath).cast] == d1


def test_saved_yaml_is_well_formed(gpath):
    import yaml
    g = Gallery(path=gpath)
    g.upsert(CastEntry(slug="x", name="X", panel="A"))
    g.save()
    data = yaml.safe_load(gpath.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data.get("version") == 1 and "cast" in data


def test_save_is_atomic_no_temp_left(gpath):
    g = Gallery(path=gpath)
    g.upsert(CastEntry(slug="x", name="X"))
    g.save()
    leftovers = [p.name for p in gpath.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# The one-character-per-panel invariant
# --------------------------------------------------------------------------- #
def test_panel_collision_raises_on_upsert(gpath):
    g = Gallery(path=gpath)
    g.upsert(CastEntry(slug="phineas", name="Phineas", panel="A"))
    with pytest.raises(ValueError):
        g.upsert(CastEntry(slug="intruder", name="Intruder", panel="A"))


def test_reassign_same_slug_to_its_panel_is_ok(gpath):
    g = Gallery(path=gpath)
    g.upsert(CastEntry(slug="phineas", name="Phineas", panel="A"))
    # re-adding the SAME slug to its own panel is idempotent, not a collision
    g.upsert(CastEntry(slug="phineas", name="Phineas (edited)", panel="A"))
    assert g.get("phineas").name == "Phineas (edited)"
    assert sum(1 for e in g.cast if e.slug == "phineas") == 1


def test_assign_panel_collision_raises(gpath):
    g = Gallery(path=gpath)
    g.upsert(CastEntry(slug="phineas", name="Phineas", panel="A"))
    g.upsert(CastEntry(slug="seraphina", name="Seraphina", panel="B"))
    g.save()
    with pytest.raises(ValueError):
        assign_panel("seraphina", "A", path=gpath)  # A held by phineas


def test_assign_panel_frees_and_moves(gpath):
    g = Gallery(path=gpath)
    g.upsert(CastEntry(slug="phineas", name="Phineas", panel="A"))
    g.upsert(CastEntry(slug="seraphina", name="Seraphina", panel="B"))
    g.save()
    assign_panel("phineas", None, path=gpath)          # free A
    e = assign_panel("seraphina", "A", path=gpath)     # now A is open
    assert e.panel == "A"


def test_assign_panel_unknown_slug_raises(gpath):
    Gallery(path=gpath).save()
    with pytest.raises(KeyError):
        assign_panel("ghost", "A", path=gpath)


# --------------------------------------------------------------------------- #
# add_character: md authoring + build fence (DEFERRED via fake orchestrate)
# --------------------------------------------------------------------------- #
def test_add_character_writes_md_registers_and_defers(gpath, tmp_chars):
    fake = _FakeOrchestrate()
    res = add_character(
        "inkwell", name="The Inkwell Imp",
        theme="a cramped scriptorium at 3am",
        persona="A minor sprite who resents being dipped into. Knows it is paint.",
        voice="en_US-ryan-high", panel=None,
        path=gpath, _orchestrate=fake,
    )
    md_file = tmp_chars / "inkwell.md"
    assert res.md_written and md_file.exists(), "did not write prompts/characters/inkwell.md"
    md_text = md_file.read_text(encoding="utf-8")
    assert md_text.startswith("# The Inkwell Imp"), "md heading should use the display name"
    assert "Theme:" in md_text and "Voice:" in md_text
    assert res.registered is True
    assert res.build == "deferred", f"build should report DEFERRED (got {res.build!r})"
    assert fake.calls == ["inkwell"], "orchestrate.build_character invoked exactly once"
    assert res.build_result is not None and res.build_result.get("dry_run") is True


def test_add_character_registers_in_yaml(gpath, tmp_chars):
    add_character("inkwell", name="Imp", theme="t", persona="p",
                  path=gpath, _orchestrate=_FakeOrchestrate())
    cast = list_cast(gpath)
    assert "inkwell" in [e.slug for e in cast]


def test_re_add_does_not_clobber_md_or_duplicate(gpath, tmp_chars):
    fake = _FakeOrchestrate()
    add_character("inkwell", name="Imp", theme="t", persona="original",
                  path=gpath, _orchestrate=fake)
    res2 = add_character("inkwell", name="Imp", theme="t", persona="(edited)",
                         path=gpath, _orchestrate=fake)
    assert not res2.md_written, "re-add clobbered the existing md file"
    g = Gallery.load(gpath)
    assert sum(1 for e in g.cast if e.slug == "inkwell") == 1, "re-add duplicated the row"


def test_re_add_preserves_created_at(gpath, tmp_chars):
    fake = _FakeOrchestrate()
    add_character("inkwell", name="Imp", theme="t", persona="p",
                  path=gpath, _orchestrate=fake)
    first = Gallery.load(gpath).get("inkwell").created_at
    add_character("inkwell", name="Imp", theme="t", persona="p2",
                  path=gpath, _orchestrate=fake)
    assert Gallery.load(gpath).get("inkwell").created_at == first


def test_add_character_run_build_false_skips_orchestrate(gpath, tmp_chars):
    fake = _FakeOrchestrate()
    res = add_character("inkwell", name="Imp", theme="t", persona="p",
                        path=gpath, run_build=False, _orchestrate=fake)
    assert res.registered is True
    assert res.build == "deferred"
    assert fake.calls == [], "run_build=False must NOT call orchestrate"


def test_add_character_panel_collision_propagates(gpath, tmp_chars):
    add_character("phineas", name="Phineas", theme="t", persona="p", panel="A",
                  path=gpath, _orchestrate=_FakeOrchestrate())
    with pytest.raises(ValueError):
        add_character("intruder", name="Intruder", theme="t", persona="p", panel="A",
                      path=gpath, _orchestrate=_FakeOrchestrate())


def test_add_result_deferred_property(gpath, tmp_chars):
    res = add_character("inkwell", name="Imp", theme="t", persona="p",
                        path=gpath, _orchestrate=_FakeOrchestrate())
    assert res.deferred is True
    # the AddResult is serializable for logging
    d = res.to_dict()
    assert d["slug"] == "inkwell" and d["build"] == "deferred"


# --------------------------------------------------------------------------- #
# seed_from_md: reads the real seed character files (read-only) into a tmp roster
# --------------------------------------------------------------------------- #
def test_seed_from_md_reads_both_portraits(gpath):
    # seed_from_md reads the CHECKED-IN prompts/characters/*.md (read-only) but
    # WRITES only to our tmp gpath -- safe.
    if not (gallery.CHARS_DIR / "phineas.md").exists():
        pytest.skip("seed character md files not present in this checkout")
    g = gallery.seed_from_md(gpath, overwrite=True)
    slugs = [e.slug for e in g.cast]
    assert "phineas" in slugs and "seraphina" in slugs
    ph = g.get("phineas")
    assert ph.panel == "A"
    assert gallery.Gallery.load(gpath).get("seraphina").panel == "B"
