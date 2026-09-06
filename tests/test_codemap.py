"""
test_codemap.py -- the codebase map has to stay honest in BOTH directions.

`scripts/export_context_view.CODEMAP` is a hand-authored answer to "what produces
what", and its header says why it is hand-authored: a machine cannot say what a
module is FOR. So this file does not try to. It asserts the two things a machine
CAN check, and leaves the prose to a person.

FORWARD -- every row points at a file that is on disk. `codemap()` already writes
`exists: true/false` per row, and until now nothing read it: a moved file
degraded the viewer's map into a description of something that is not there,
silently, which is how it would stay broken.

BACKWARD -- every file in the mapped layers is either IN the map or in UNMAPPED
below with a reason. This is the direction that actually rots. Rows rarely break;
what happens is a module lands and nobody adds it, and the map decays by omission
while every row in it still resolves.

Scope is deliberately NOT the whole repo. A bidirectional check over all 90 files
would demand "what produces what" prose for 28 test modules and 18 pipeline
modules that do not contribute to the context view at all, which would push the
map toward being an inventory -- the one thing its header says it is not. The
scope is the layers that produce the artifact: runtime/, director/, scripts/, and
the repo root.

The point is not coverage. The point is that adding a module to those layers
forces a decision -- is this load-bearing for the artifact? -- and records the
answer where the next person can read it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# scripts/ is not on sys.path (conftest adds root, runtime/, director/, pipeline/),
# and export_context_view is a script rather than a package member, so load it by path.
_spec = importlib.util.spec_from_file_location(
    "_export_context_view", ROOT / "scripts" / "export_context_view.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CODEMAP = _mod.CODEMAP

MAPPED = {row[1] for row in CODEMAP}

# The layers whose files must be accounted for. pipeline/ and tests/ are out of
# scope by design -- see the module docstring.
SCOPED_DIRS = ("runtime", "director", "scripts")


# Deliberately not in the map, with the reason. Adding a file to runtime/,
# director/, scripts/ or the root means either writing it a CODEMAP row or
# adding it here -- and "I could not decide" is not one of the two options.
UNMAPPED = {
    # --- runtime/: the RENDERING half. The map covers what decides what plays
    # next; these turn that decision into pixels. They are HEAVY (numpy/cv2) and
    # sit outside the walker's pure import set.
    "runtime/stage_render.py": "renders a decision, does not make one",
    "runtime/clip_player.py": "renders a decision, does not make one",
    "runtime/crossframe.py": "renders a decision, does not make one",
    "runtime/rig.py": "renders a decision, does not make one",
    "runtime/rig_loop.py": "renders a decision, does not make one",
    "runtime/capture_demo.py": "offline demo renderer; produces a GIF, not the artifact",
    "runtime/behavior_select.py": (
        "no importers anywhere in the repo -- mapping it would describe a "
        "component that nothing runs. See the open issue on ARCHITECTURE.md:436, "
        "which currently lists it under the render stack."
    ),
    # --- director/: operator and telemetry surfaces, not producers of the view.
    "director/otel.py": "telemetry; fail-open by design and contributes no field",
    "director/stage_manager.py": "parked path (task lp-director, last run 2026-05-31)",
    "director/voice_eval.py": "offline voice grading; not on the live loop",
    "director/signals.py": "thin adapter over feeds.py; no field of its own",
    "director/feeds.py": (
        "ARGUABLY A GAP: context.py is mapped as 'the world seam' and this is "
        "what supplies it. Left unmapped only because the seam's row already "
        "describes the behaviour; revisit when the world layer is next touched."
    ),
    # --- scripts/: operator tools. backfill_lived.py IS mapped, because it
    # produced data the Lived lens still shows; these produce none.
    "scripts/deploy_hil.py": "operator tool; ships files, produces no view data",
    "scripts/release.py": "operator tool; publishes the OSS subset",
    "scripts/unstick.py": "operator tool; proposes topology spend, writes no view data",
    "scripts/seed_demo_media.py": "onboarding tool; writes placeholder media only",
    # --- repo root.
    "player.py": "parked v2 player (task lp-player, Disabled)",
    "gallery.py": "cast registry / add-character front door; upstream of the graph",
    "_preview_cycle.py": "throwaway on-panel demo, self-declared TEMP",
    "_preview_panels.py": "throwaway on-panel demo, self-declared TEMP",
}


def _scoped_files():
    """Every .py in the mapped layers, plus the root scripts, as repo-relative paths."""
    out = []
    for d in SCOPED_DIRS:
        out += [
            str(p.relative_to(ROOT)).replace("\\", "/")
            for p in sorted((ROOT / d).glob("*.py"))
            if p.name != "__init__.py"
        ]
    out += [p.name for p in sorted(ROOT.glob("*.py"))]
    return out


# --------------------------------------------------------------------------- #
# forward: the map does not describe files that are not there
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel", sorted(MAPPED))
def test_every_mapped_path_is_on_disk(rel):
    assert (ROOT / rel).exists(), (
        f"CODEMAP describes {rel}, which is not on disk. Either the file moved "
        f"and the row needs updating, or the row outlived the file."
    )


def test_no_duplicate_rows():
    paths = [row[1] for row in CODEMAP]
    dupes = {p for p in paths if paths.count(p) > 1}
    assert not dupes, f"CODEMAP lists these paths more than once: {sorted(dupes)}"


def test_every_row_is_well_formed():
    for row in CODEMAP:
        assert len(row) == 4, f"expected (layer, path, role, shows), got {row!r}"
        layer, path, role, shows = row
        assert layer and path and role and shows, f"empty field in {path!r}"
        assert len(role) > 40, (
            f"{path}: the role field is what makes this a map rather than a file "
            f"list -- {role!r} is too thin to be worth a row"
        )


# --------------------------------------------------------------------------- #
# backward: no file in the mapped layers is silently absent from the map
# --------------------------------------------------------------------------- #
def test_every_scoped_file_is_mapped_or_explicitly_not():
    unaccounted = [f for f in _scoped_files() if f not in MAPPED and f not in UNMAPPED]
    assert not unaccounted, (
        "These files are in a mapped layer but appear neither in CODEMAP nor in "
        "UNMAPPED:\n  " + "\n  ".join(unaccounted) + "\n\n"
        "Add a CODEMAP row saying what it produces and which lens shows it, or add "
        "it to UNMAPPED with the reason it produces nothing. Both are fine answers; "
        "leaving it undecided is what this test exists to prevent."
    )


def test_unmapped_entries_still_exist():
    """An UNMAPPED reason for a deleted file is stale bookkeeping, not documentation."""
    gone = [rel for rel in UNMAPPED if not (ROOT / rel).exists()]
    assert not gone, (
        f"UNMAPPED carries reasons for files that no longer exist: {sorted(gone)}. "
        f"Delete the entries."
    )


def test_unmapped_and_mapped_do_not_overlap():
    both = MAPPED & set(UNMAPPED)
    assert not both, (
        f"These are in CODEMAP and also listed as deliberately unmapped: {sorted(both)}"
    )
