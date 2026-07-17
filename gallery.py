"""gallery.py -- the cast registry + add-character front door for living-portraits.

This is the "gallery / cast" box: a single human-readable roster (gallery.yaml) of who
hangs in the gallery, plus the one verb that brings a NEW portrait into being end to end:

    add_character(slug, name, theme, persona, voice=None, panel=None)
        1. WRITE prompts/characters/<slug>.md   (if absent) -- a persona file in the exact
           shape of phineas.md / seraphina.md, so the stage_manager inherits it the same way
           it inherits the two seed characters (directives + every character file concatenated).
        2. REGISTER the cast entry in gallery.yaml                 (atomic write).
        3. BUILD via pipeline.orchestrate.build_character(slug)    (IMPORT-GUARDED).

Step 3 is the only heavy step and it is deliberately fenced. The producer's segment/rig/verify
stages need numpy+opencv and its generate stage needs torch+diffusers+SD-1.5 weights -- present
on supercommons2 (SC2), not on every box. We do NOT reimplement that fence here; orchestrate
already degrades to a dry-run PLAN on a box that cannot run the chain. gallery.py adds ONE more
layer outside it: if `pipeline.orchestrate` itself cannot even be imported (a truly bare box, no
numpy at all), we still write the .md + register the cast entry and report the build as DEFERRED
to SC2. So registration is always durable; the model run is the part that waits for the right host.

The registry is checked-in source-of-truth (a roster), NOT runtime state -- it lives at the
project root beside README.md, not under the gitignored data/. The character .md files it points
at are likewise checked-in under prompts/characters/.

    python gallery.py list
    python gallery.py add <slug> --name "..." --theme "..." --persona "..." [--voice ...] [--panel A|B]
    python gallery.py                 # self-test (offline; build guarded off; round-trips yaml)
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml  # pyyaml -- a hard dep of the parent `life` repo (used widely); see CLAUDE.md.

# --------------------------------------------------------------------------------------------
# Paths. gallery.yaml is a ROSTER (checked in at root); character md files are checked in too.
# --------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
GALLERY_PATH = ROOT / "gallery.yaml"
CHARS_DIR = ROOT / "prompts" / "characters"
DIRECTIVES_PATH = ROOT / "prompts" / "_stage-directives.md"

# The two panels the director maps characters onto (director/stage_manager.py: PANEL A / B).
# A == the larger frame; B == the smaller, brighter frame. None == cast member with no panel yet.
PANELS = ("A", "B")


# ==========================================================================================
# Cast entry
# ==========================================================================================
@dataclass
class CastEntry:
    """One portrait in the gallery. Mirrors the fields a character md file + the director need.

    `theme` is a single line that MUST match the "Theme:" line in prompts/characters/<slug>.md
    -- gallery.yaml is the index; the md file is the full brief; the Theme line is the join key
    a human eyeballs to confirm they describe the same portrait. `voice` is a piper voice id
    (e.g. "en_GB-alan-medium") consumed by pipeline/tts.py; None until a voice is chosen.
    `panel` is "A" / "B" / None.
    """

    slug: str
    name: str
    panel: Optional[str] = None
    theme: str = ""
    persona: str = ""
    voice: Optional[str] = None
    mood: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict:
        # Ordered for a readable yaml diff; identity fields first, then the descriptive body.
        return {
            "slug": self.slug,
            "name": self.name,
            "panel": self.panel,
            "theme": self.theme,
            "persona": self.persona,
            "voice": self.voice,
            "mood": self.mood,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CastEntry":
        return cls(
            slug=str(d["slug"]),
            name=str(d.get("name", d["slug"])),
            panel=_norm_panel(d.get("panel")),
            theme=str(d.get("theme") or ""),
            persona=str(d.get("persona") or ""),
            voice=(str(d["voice"]) if d.get("voice") else None),
            mood=(str(d["mood"]) if d.get("mood") else None),
            created_at=(str(d["created_at"]) if d.get("created_at") else None),
        )


def _norm_panel(panel) -> Optional[str]:
    """Normalize a panel value to 'A' / 'B' / None. Raises on anything else (fail loud:
    a typo'd panel would silently drop a portrait off both screens)."""
    if panel is None:
        return None
    p = str(panel).strip().upper()
    if p in ("", "NONE", "NULL"):
        return None
    if p not in PANELS:
        raise ValueError(f"panel must be one of {PANELS} or None, got {panel!r}")
    return p


# ==========================================================================================
# Registry (the roster file)
# ==========================================================================================
@dataclass
class Gallery:
    """The full cast roster. Thin wrapper over an ordered list of CastEntry.

    Source of truth is gallery.yaml; this is its in-memory form. load()/save() round-trip it;
    add()/assign_panel() mutate it. One panel may hold at most one character (the player has
    exactly two frames) -- assigning a panel that is already taken raises unless reassigning the
    same slug.
    """

    cast: list[CastEntry] = field(default_factory=list)
    path: Path = GALLERY_PATH

    # ---- load / save (atomic; mirrors clip_graph.save discipline) -------------------------
    @classmethod
    def load(cls, path: Path | str = GALLERY_PATH) -> "Gallery":
        """Load the roster from yaml. An absent/empty file yields an empty gallery."""
        path = Path(path)
        if not path.exists():
            return cls(cast=[], path=path)
        raw = path.read_text(encoding="utf-8-sig")  # tolerate a BOM, like the director does
        data = yaml.safe_load(raw) or {}
        rows = data.get("cast", []) if isinstance(data, dict) else (data or [])
        cast = [CastEntry.from_dict(r) for r in rows]
        return cls(cast=cast, path=path)

    def save(self, path: Optional[Path | str] = None) -> Path:
        """Persist the roster to yaml atomically (tmp in same dir -> os.replace).

        A reader never sees a half-written file -- same atomic discipline as
        runtime/clip_graph.py:save and director/stage_manager.py:write_state.
        """
        path = Path(path) if path is not None else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "cast": [e.to_dict() for e in self.cast]}
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    # ---- queries --------------------------------------------------------------------------
    def get(self, slug: str) -> Optional[CastEntry]:
        for e in self.cast:
            if e.slug == slug:
                return e
        return None

    def list_cast(self) -> list[CastEntry]:
        """The roster in registration order. (Free function list_cast() below is the public API.)"""
        return list(self.cast)

    def panel_holder(self, panel: str) -> Optional[CastEntry]:
        p = _norm_panel(panel)
        for e in self.cast:
            if e.panel == p and p is not None:
                return e
        return None

    # ---- mutation -------------------------------------------------------------------------
    def upsert(self, entry: CastEntry) -> CastEntry:
        """Insert a new cast entry, or replace the existing one with the same slug in place.

        Guards the two-frame invariant: a panel already held by a DIFFERENT slug raises, so we
        never silently knock a portrait off its frame. Reassigning the same slug to its own
        panel is fine (idempotent re-add)."""
        entry.panel = _norm_panel(entry.panel)
        if entry.panel is not None:
            holder = self.panel_holder(entry.panel)
            if holder is not None and holder.slug != entry.slug:
                raise ValueError(
                    f"panel {entry.panel} already held by {holder.slug!r}; "
                    f"free it or pick another panel before adding {entry.slug!r}"
                )
        for i, e in enumerate(self.cast):
            if e.slug == entry.slug:
                self.cast[i] = entry
                return entry
        self.cast.append(entry)
        return entry

    def assign_panel(self, slug: str, panel: Optional[str]) -> CastEntry:
        """Move a cast member onto panel A/B (or off-panel with None). Persists nothing -- the
        caller saves. Enforces the one-character-per-panel invariant."""
        entry = self.get(slug)
        if entry is None:
            raise KeyError(f"no cast member with slug {slug!r}")
        panel = _norm_panel(panel)
        if panel is not None:
            holder = self.panel_holder(panel)
            if holder is not None and holder.slug != slug:
                raise ValueError(f"panel {panel} already held by {holder.slug!r}")
        entry.panel = panel
        return entry


# ==========================================================================================
# Character md file authoring (the shape phineas.md / seraphina.md use)
# ==========================================================================================
def _char_md_text(name: str, panel: Optional[str], theme: str, voice_line: str, persona: str) -> str:
    """Render a character md file in the EXACT shape of the seed files.

    Layout (see prompts/characters/phineas.md):

        # <Name>  (PANEL <X>, <frame note>)

        Theme: <one line>.
        Voice: <one line>.

        <persona prose paragraph>

    The leading "# <Name> (PANEL ...)" + the "Theme:" / "Voice:" lines are the join the director
    and the verify gate read; the prose paragraph is the actable brief. The shared 4th-wall meta
    is NOT copied in -- it is inherited: stage_manager.build_system() concatenates
    _stage-directives.md ahead of every character file, so each character file stays persona-only.
    """
    if panel == "A":
        frame_note = "PANEL A, the larger frame"
    elif panel == "B":
        frame_note = "PANEL B, the smaller, brighter frame"
    else:
        frame_note = "no panel assigned"
    theme_line = theme.strip().rstrip(".")
    persona_body = persona.strip()
    return (
        f"# {name}  ({frame_note})\n"
        f"\n"
        f"Theme: {theme_line}.\n"
        f"{voice_line}\n"
        f"\n"
        f"{persona_body}\n"
    )


def _write_char_md(slug: str, name: str, panel: Optional[str], theme: str,
                   persona: str, voice: Optional[str]) -> tuple[Path, bool]:
    """Write prompts/characters/<slug>.md IF ABSENT. Returns (path, written).

    Never clobbers an existing persona file -- a hand-tuned character brief outranks anything
    add_character would synthesize, so an existing file is left untouched (written=False). The
    Voice: line carries the piper voice id when known, else a neutral placeholder so the file
    keeps the two-line Theme/Voice header the director expects.
    """
    path = CHARS_DIR / f"{slug}.md"
    if path.exists():
        return path, False
    if voice:
        voice_line = f"Voice: piper voice {voice}."
    else:
        voice_line = "Voice: unset (assign a piper voice id in gallery.yaml)."
    text = _char_md_text(name, panel, theme, voice_line, persona)
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic write, same discipline as the roster
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path, True


# ==========================================================================================
# The build fence -- pipeline.orchestrate.build_character, import-guarded.
# ==========================================================================================
@dataclass
class AddResult:
    """Structured outcome of add_character. Serializable; safe to log/print.

    `md_path`/`md_written`: the persona file and whether THIS call created it.
    `registered`: True iff the cast entry is now in gallery.yaml.
    `build`: one of 'registered' (clip row written), 'deferred' (heavy stages unavailable here ->
             run on SC2), 'quarantined' (built + failed the verify gate), 'error'.
    `build_detail`: a human one-liner (the orchestrate error/plan note, or the clip key).
    `build_result`: the raw orchestrate.BuildResult.to_dict() when the build ran, else None.
    """

    slug: str
    md_path: Optional[str] = None
    md_written: bool = False
    registered: bool = False
    build: str = "deferred"
    build_detail: str = ""
    build_result: Optional[dict] = None

    @property
    def deferred(self) -> bool:
        return self.build == "deferred"

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "md_path": self.md_path,
            "md_written": self.md_written,
            "registered": self.registered,
            "build": self.build,
            "build_detail": self.build_detail,
            "build_result": self.build_result,
        }


def _run_build_guarded(slug: str, *, allow_generate: bool, _orchestrate=None) -> tuple[str, str, Optional[dict]]:
    """Call pipeline.orchestrate.build_character(slug), fully fenced.

    Returns (build_status, detail, raw_result_dict). build_status is one of:
        'registered'   -- a clean/passing verdict wrote a clip row (only happens on SC2)
        'deferred'     -- the heavy chain cannot run on THIS host (dry-run / unavailable stages /
                          import failure); the cast entry stands, the model run waits for SC2
        'quarantined'  -- the chain ran and the verify gate FAILED (parked, not registered)
        'error'        -- the chain raised mid-run

    Fencing happens at TWO levels:
      1. import: a bare box without numpy can't even import orchestrate -> deferred.
      2. result: orchestrate itself degrades to dry_run on a box missing torch/opencv -> deferred.
    `_orchestrate` is a seam for the offline self-test to inject a fake module; production passes
    None and we import the real one.
    """
    orch = _orchestrate
    if orch is None:
        try:
            from pipeline import orchestrate as orch  # type: ignore
        except Exception as e:
            # Level 1: the producer module (or a transitive heavy import) is absent on this box.
            return "deferred", f"build deferred to SC2 (pipeline.orchestrate unavailable: {type(e).__name__}: {e})", None

    try:
        # allow_generate stays OFF by default even on SC2 callers unless explicitly opted in; the
        # producer's own guard makes an accidental SD download impossible regardless.
        res = orch.build_character(slug, allow_generate=allow_generate)
    except Exception as e:
        return "error", f"build raised: {type(e).__name__}: {e}", None

    rd = res.to_dict() if hasattr(res, "to_dict") else None

    # Level 2: orchestrate ran but its heavy stages were unavailable -> it hands back a dry-run
    # plan with an error string. That is the "needs SC2" signal -- treat as deferred, not failure.
    if getattr(res, "dry_run", False) or not getattr(res, "portrait", None):
        detail = getattr(res, "error", None) or "no portrait yet (generation belongs on SC2)"
        return "deferred", f"build deferred to SC2: {detail}", rd

    if getattr(res, "registered", False):
        return "registered", f"registered clip row {getattr(res, 'clip_key', None)!r}", rd

    if getattr(res, "quarantined", False):
        reasons = "; ".join(str(r) for r in getattr(res, "reasons", [])[:3])
        return "quarantined", f"built but verify gate withheld registration: {reasons}", rd

    # built, passed, but not registered (e.g. strict soft-pass) -> still a deferred registration.
    return "deferred", "built but not registered (soft gate) -- re-run on SC2 to register", rd


# ==========================================================================================
# Public API
# ==========================================================================================
def load(path: Path | str = GALLERY_PATH) -> Gallery:
    """Load the cast roster from gallery.yaml (or an empty roster if absent)."""
    return Gallery.load(path)


def save(gallery: Gallery, path: Optional[Path | str] = None) -> Path:
    """Persist a roster atomically. Thin free-function wrapper over Gallery.save."""
    return gallery.save(path)


def list_cast(path: Path | str = GALLERY_PATH) -> list[CastEntry]:
    """The full cast, in registration order."""
    return Gallery.load(path).list_cast()


def assign_panel(slug: str, panel: Optional[str], *, path: Path | str = GALLERY_PATH) -> CastEntry:
    """Assign (or clear, with None) a cast member's panel and persist. Returns the updated entry."""
    g = Gallery.load(path)
    entry = g.assign_panel(slug, panel)
    g.save(path)
    return entry


def add_character(
    slug: str,
    name: str,
    theme: str,
    persona: str,
    voice: Optional[str] = None,
    panel: Optional[str] = None,
    *,
    path: Path | str = GALLERY_PATH,
    allow_generate: bool = False,
    run_build: bool = True,
    _orchestrate=None,
) -> AddResult:
    """Bring a NEW character into the gallery end to end.

    Order is chosen so registration is durable even if the build can't run here:
      1. write prompts/characters/<slug>.md  (only if absent -- never clobber a tuned brief)
      2. register the cast entry in gallery.yaml  (atomic)
      3. run pipeline.orchestrate.build_character(slug), IMPORT-GUARDED -- on a box without the
         heavy deps this reports the build as DEFERRED to SC2 and the cast entry still stands.

    Args:
        slug:    character slug; the join key for <slug>.md, <slug>_portrait.png, the clip graph.
        name:    display name (the md heading + roster).
        theme:   one-line scene/theme; written as the md "Theme:" line and the roster's theme.
        persona: the actable brief paragraph (md body + roster persona, stored verbatim).
        voice:   piper voice id (optional; flows into the md "Voice:" line + tts.py).
        panel:   "A" / "B" / None. Enforced one-character-per-panel.
        allow_generate: opt in to the HEAVY SD render inside orchestrate (intended for SC2).
        run_build: set False to register only (skip step 3 entirely; build stays 'deferred').
        _orchestrate: test seam -- inject a fake orchestrate module to exercise the fence offline.

    Returns an AddResult.
    """
    panel = _norm_panel(panel)
    res = AddResult(slug=slug)

    # 1. persona file (idempotent: never clobbers an existing hand-tuned brief).
    md_path, written = _write_char_md(slug, name, panel, theme, persona, voice)
    res.md_path = str(md_path)
    res.md_written = written

    # 2. register the cast entry (atomic). created_at preserved on re-add.
    g = Gallery.load(path)
    existing = g.get(slug)
    created_at = existing.created_at if existing and existing.created_at else _now_iso()
    entry = CastEntry(
        slug=slug, name=name, panel=panel, theme=theme.strip(),
        persona=persona.strip(), voice=voice, mood=(existing.mood if existing else None),
        created_at=created_at,
    )
    g.upsert(entry)
    g.save(path)
    res.registered = True

    # 3. build (import-guarded). On a bare/local box this is 'deferred' and that is correct.
    if run_build:
        status, detail, raw = _run_build_guarded(slug, allow_generate=allow_generate, _orchestrate=_orchestrate)
        res.build, res.build_detail, res.build_result = status, detail, raw
    else:
        res.build = "deferred"
        res.build_detail = "build skipped (run_build=False) -- register-only; run on SC2 to bake"

    return res


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ==========================================================================================
# Seeding -- read the two existing character md files into a fresh roster.
# ==========================================================================================
def _collapse_wraps(text: str) -> str:
    """Collapse editor hard-wrap newlines into spaces, but keep paragraph breaks.

    A run of one-or-more blank lines becomes a single newline (a paragraph break); single
    newlines inside a paragraph become spaces; runs of spaces collapse to one. So a hard-wrapped
    one-paragraph persona becomes one clean line, while a genuine two-paragraph brief keeps its
    break."""
    paras = []
    for chunk in re.split(r"\n\s*\n", text.strip()):
        words = chunk.split()
        if words:
            paras.append(" ".join(words))
    return "\n".join(paras)


def _parse_char_md(slug: str, text: str) -> CastEntry:
    """Pull (name, panel, theme, persona) out of a phineas/seraphina-shaped md file.

    Heading:  '# <Name>  (PANEL <X>, ...)'  -> name + panel
    'Theme:'  line -> theme (period stripped)
    persona   -> the prose paragraph(s) after the Theme/Voice header block
    The shared 4th-wall directives are NOT in these files (they are inherited), so everything
    after the Voice: line is persona.
    """
    name = slug
    panel: Optional[str] = None
    theme = ""
    voice: Optional[str] = None
    body_lines: list[str] = []
    in_body = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not in_body and line.startswith("#"):
            head = line.lstrip("#").strip()
            # split off a trailing "(PANEL ...)" note
            if "(" in head:
                name_part, _, paren = head.partition("(")
                name = name_part.strip()
                up = paren.upper()
                if "PANEL A" in up:
                    panel = "A"
                elif "PANEL B" in up:
                    panel = "B"
            else:
                name = head
            continue
        if not in_body and line.lower().startswith("theme:"):
            theme = line.split(":", 1)[1].strip().rstrip(".")
            continue
        if not in_body and line.lower().startswith("voice:"):
            voice = line.split(":", 1)[1].strip().rstrip(".") or None
            in_body = True  # persona starts after the Voice line
            continue
        if in_body:
            body_lines.append(raw_line)
    # The md body is one paragraph hard-wrapped by the editor; collapse the wrap newlines into
    # single spaces so the roster holds clean prose (and yaml renders it as a tidy scalar). A
    # genuine paragraph BREAK (a blank line between paragraphs) is preserved as a single newline.
    persona = _collapse_wraps("\n".join(body_lines))
    return CastEntry(
        slug=slug, name=name, panel=panel, theme=theme, persona=persona,
        voice=None,  # the md "Voice:" line is a description, not a piper id; leave id unset
        mood=None, created_at=None,
    )


# Seed voices: piper ids matched to each portrait's described voice. Used only when seeding a
# fresh gallery.yaml from the md files (a real piper id, not the prose "Voice:" description).
_SEED_VOICES = {
    "phineas": "en_GB-alan-medium",      # dry patrician baritone
    "seraphina": "en_GB-jenny_dioco-medium",  # quick, dry, modern-feeling
}
_SEED_MOODS = {
    "phineas": "tragic-grandiose",
    "seraphina": "wry-complicit",
}


def seed_from_md(path: Path | str = GALLERY_PATH, *, overwrite: bool = False) -> Gallery:
    """Build (and save) gallery.yaml from the existing prompts/characters/*.md seed files.

    Reads phineas.md + seraphina.md, parses each into a CastEntry, stamps a piper voice id and a
    mood + created_at, and writes the roster. Idempotent: if gallery.yaml already lists both
    seeds and overwrite=False, returns the existing roster untouched.
    """
    path = Path(path)
    existing = Gallery.load(path)
    have = {e.slug for e in existing.cast}
    if not overwrite and {"phineas", "seraphina"}.issubset(have):
        return existing

    g = Gallery(cast=list(existing.cast), path=path) if not overwrite else Gallery(cast=[], path=path)
    now = _now_iso()
    for seed_slug in ("phineas", "seraphina"):
        md = CHARS_DIR / f"{seed_slug}.md"
        if not md.exists():
            continue
        entry = _parse_char_md(seed_slug, md.read_text(encoding="utf-8-sig"))
        entry.voice = _SEED_VOICES.get(seed_slug)
        entry.mood = _SEED_MOODS.get(seed_slug)
        entry.created_at = now
        g.upsert(entry)
    g.save(path)
    return g


def ensure_seeded(path: Path | str = GALLERY_PATH) -> Gallery:
    """Load the roster, seeding it from the md files on first use. The lazy-init the CLI uses."""
    g = Gallery.load(path)
    if not g.cast:
        return seed_from_md(path)
    return g


# ==========================================================================================
# CLI
# ==========================================================================================
def _fmt_entry(e: CastEntry) -> str:
    panel = e.panel or "-"
    voice = e.voice or "unset"
    return (f"  [{panel}] {e.slug:14s} {e.name:24s} voice={voice:24s} mood={e.mood or '-'}\n"
            f"        theme: {e.theme}")


def cmd_list(args) -> int:
    g = ensure_seeded(args.path)
    cast = g.list_cast()
    print(f"gallery.yaml -- {len(cast)} cast member(s)  [{args.path}]")
    by_panel = {"A": None, "B": None}
    for e in cast:
        if e.panel in by_panel:
            by_panel[e.panel] = e.slug
    print(f"  panel A: {by_panel['A'] or '(empty)'}   panel B: {by_panel['B'] or '(empty)'}")
    print()
    for e in cast:
        print(_fmt_entry(e))
    return 0


def cmd_add(args) -> int:
    res = add_character(
        args.slug,
        name=args.name,
        theme=args.theme,
        persona=args.persona,
        voice=args.voice,
        panel=args.panel,
        path=args.path,
        allow_generate=args.allow_generate,
        run_build=not args.no_build,
    )
    print(f"added '{res.slug}':")
    print(f"  md           : {res.md_path}  ({'written' if res.md_written else 'already existed'})")
    print(f"  registered   : {res.registered}  -> {args.path}")
    print(f"  build        : {res.build.upper()}  -- {res.build_detail}")
    if res.deferred:
        print("  NOTE: the heavy generate/segment/rig/verify chain runs on supercommons2 (SC2).")
        print("        The cast entry + persona file are committed; run `pipeline/orchestrate.py "
              f"{res.slug}` on SC2 to bake the portrait into a clip row.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="living-portraits cast registry + add-character")
    ap.add_argument("--path", default=str(GALLERY_PATH), help="gallery.yaml path (default: project root)")
    sub = ap.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="show the cast roster")
    p_list.set_defaults(func=cmd_list)

    p_add = sub.add_parser("add", help="add a new character (writes md, registers, builds via SC2)")
    p_add.add_argument("slug", help="character slug (lowercase, no spaces)")
    p_add.add_argument("--name", required=True, help="display name")
    p_add.add_argument("--theme", required=True, help="one-line scene/theme (the md Theme: line)")
    p_add.add_argument("--persona", required=True, help="actable brief paragraph")
    p_add.add_argument("--voice", default=None, help="piper voice id (e.g. en_GB-alan-medium)")
    p_add.add_argument("--panel", default=None, help="A | B | (omit for no panel)")
    p_add.add_argument("--allow-generate", action="store_true",
                       help="opt in to the HEAVY SD render in orchestrate (intended for SC2)")
    p_add.add_argument("--no-build", action="store_true",
                       help="register only; skip the orchestrate build step")
    p_add.set_defaults(func=cmd_add)

    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        return _selftest()
    return args.func(args)


# ==========================================================================================
# Self-test -- OFFLINE. Seeds the roster, adds a dummy character with the build GUARDED OFF,
# asserts the md got written + the cast entry registered + the build reported DEFERRED, lists
# it, and round-trips the whole roster through yaml. No SC2, no torch, no network.
# ==========================================================================================
class _FakeDryRunResult:
    """Stand-in for orchestrate.BuildResult on a box that cannot run the heavy chain: a dry-run
    with no portrait, exactly what build_character returns when generation is guarded off."""

    def __init__(self, slug):
        self.slug = slug
        self.dry_run = True
        self.portrait = None
        self.registered = False
        self.quarantined = False
        self.clip_key = None
        self.error = "no portrait and generation is off (belongs on SC2)"
        self.reasons = []
        self.plan = ["generate -> dummy_portrait.png (REQUIRED but guarded off on this host)"]

    def to_dict(self):
        return {"slug": self.slug, "dry_run": True, "portrait": None, "registered": False,
                "error": self.error, "plan": list(self.plan)}


class _FakeOrchestrate:
    """A fake `pipeline.orchestrate` module: build_character always returns a dry-run (build OFF),
    so the self-test exercises add_character's fence with zero heavy deps and zero risk of a real
    model run."""

    def __init__(self):
        self.calls = []

    def build_character(self, slug, allow_generate=False, **kw):
        assert allow_generate is False, "self-test must never opt into real generation"
        self.calls.append(slug)
        return _FakeDryRunResult(slug)


def _selftest() -> int:
    global CHARS_DIR  # the dummy-add step temporarily redirects this at the module level
    import shutil

    print("=" * 80)
    print("gallery.py self-test  (OFFLINE; build GUARDED OFF; no SC2 / torch / network)")
    print("=" * 80)

    work = Path(tempfile.mkdtemp(prefix="lp_gallery_"))
    gpath = work / "gallery.yaml"
    failures: list[str] = []

    def check(cond, msg):
        print(f"  [{'OK ' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    try:
        # --- 1. seed from the real md files -------------------------------------------------
        print("\n-- seed roster from prompts/characters/*.md --")
        g = seed_from_md(gpath, overwrite=True)
        slugs = [e.slug for e in g.cast]
        check("phineas" in slugs and "seraphina" in slugs, f"seeded both portraits: {slugs}")
        ph = g.get("phineas")
        se = g.get("seraphina")
        check(ph is not None and ph.panel == "A", "phineas on PANEL A")
        check(se is not None and se.panel == "B", "seraphina on PANEL B")
        # theme matches the md Theme: line (the documented join key)
        ph_md = (CHARS_DIR / "phineas.md").read_text(encoding="utf-8-sig")
        theme_in_md = next((ln.split(":", 1)[1].strip().rstrip(".")
                            for ln in ph_md.splitlines() if ln.lower().startswith("theme:")), "")
        check(ph.theme == theme_in_md, "phineas roster theme == md Theme: line")
        check(bool(ph.voice) and bool(se.voice), "seed voices assigned (piper ids)")
        check(bool(ph.persona) and "tragedian" in ph.persona, "phineas persona captured from md body")

        # --- 2. add a dummy character with the build GUARDED OFF ----------------------------
        print("\n-- add_character('inkwell', build guarded off) --")
        fake = _FakeOrchestrate()
        # point CHARS_DIR at a temp dir so the dummy .md never litters prompts/characters/
        real_chars = CHARS_DIR
        CHARS_DIR = work / "characters"
        try:
            res = add_character(
                "inkwell",
                name="The Inkwell Imp",
                theme="a cramped scriptorium at 3am, spilled ink, one guttering candle",
                persona=("A minor sprite who lives in the inkwell and resents being dipped into. "
                         "Heckles both portraits about their diction. Knows it is paint."),
                voice="en_US-ryan-high",
                panel=None,
                path=gpath,
                _orchestrate=fake,
            )
            md_file = CHARS_DIR / "inkwell.md"
            check(res.md_written and md_file.exists(), "add_character WROTE prompts/characters/inkwell.md")
            md_text = md_file.read_text(encoding="utf-8")
            check(md_text.startswith("# The Inkwell Imp"), "md heading uses the display name")
            check("Theme:" in md_text and "Voice:" in md_text, "md carries Theme: + Voice: header lines")
            check("inkwell" in res.build_detail.lower() or res.deferred, "build path resolved for inkwell")
            check(res.registered, "cast entry registered in gallery.yaml")
            check(res.build == "deferred", f"build reported DEFERRED to SC2 (got {res.build!r})")
            check(fake.calls == ["inkwell"], "orchestrate.build_character was invoked exactly once")
            check(res.build_result is not None and res.build_result.get("dry_run") is True,
                  "raw build result is a dry-run (no portrait baked locally)")

            # --- 2b. re-add is idempotent: md NOT clobbered, entry replaced in place --------
            print("\n-- re-add 'inkwell' (idempotent) --")
            res2 = add_character(
                "inkwell", name="The Inkwell Imp",
                theme="a cramped scriptorium at 3am, spilled ink, one guttering candle",
                persona="(edited persona)", voice="en_US-ryan-high",
                path=gpath, _orchestrate=fake,
            )
            check(not res2.md_written, "re-add did NOT clobber the existing md file")
            g_after = Gallery.load(gpath)
            check(sum(1 for e in g_after.cast if e.slug == "inkwell") == 1,
                  "re-add replaced the entry in place (no duplicate row)")
            check(g_after.get("inkwell").created_at == Gallery.load(gpath).get("inkwell").created_at,
                  "created_at preserved across re-add")
        finally:
            CHARS_DIR = real_chars

        # --- 3. list shows it ---------------------------------------------------------------
        print("\n-- list_cast() --")
        cast = list_cast(gpath)
        names = [e.slug for e in cast]
        check("inkwell" in names, f"list_cast includes the new character: {names}")
        check(len(cast) == 3, f"roster has 3 members (2 seed + 1 added), got {len(cast)}")

        # --- 4. round-trip through yaml -----------------------------------------------------
        print("\n-- yaml round-trip --")
        g1 = Gallery.load(gpath)
        d1 = [e.to_dict() for e in g1.cast]
        # save to a second path, reload, compare
        gpath2 = work / "gallery_rt.yaml"
        g1.save(gpath2)
        g2 = Gallery.load(gpath2)
        d2 = [e.to_dict() for e in g2.cast]
        check(d1 == d2, "roster is byte-stable across save -> load -> save -> load")
        # the file is valid yaml a human/other tool can read
        reparsed = yaml.safe_load(gpath2.read_text(encoding="utf-8"))
        check(isinstance(reparsed, dict) and reparsed.get("version") == 1 and "cast" in reparsed,
              "gallery.yaml is well-formed yaml {version, cast:[...]}")

        # --- 5. panel invariant + assign_panel ----------------------------------------------
        print("\n-- panel invariant + assign_panel --")
        try:
            # inkwell -> panel A should collide with phineas
            assign_panel("inkwell", "A", path=gpath)
            check(False, "assigning an occupied panel must raise")
        except ValueError:
            check(True, "one-character-per-panel enforced (A already held by phineas)")
        # free panel B's holder off, then move inkwell onto B
        assign_panel("seraphina", None, path=gpath)
        e = assign_panel("inkwell", "B", path=gpath)
        check(e.panel == "B", "assign_panel moved inkwell onto freed panel B")

        # --- print the resulting roster -----------------------------------------------------
        print("\n-- resulting gallery.yaml --")
        print(gpath.read_text(encoding="utf-8"))

    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("=" * 80)
    if failures:
        print(f"SELF-TEST FAILED -- {len(failures)} check(s) failed:")
        for f in failures:
            print(f"   - {f}")
        print("=" * 80)
        return 1
    print("self-test OK -- seeded 2 portraits, added 1 with build GUARDED-OFF (md written + cast")
    print("registered + build DEFERRED to SC2), list showed it, roster round-tripped through yaml.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
