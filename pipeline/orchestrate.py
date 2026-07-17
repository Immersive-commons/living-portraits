"""orchestrate.py -- the producer loop that drives one character from prompt to clip row.

This is the "orchestrate (producer)" box in living-portraits-architecture-v3.svg. It owns
no art and no rig math; it CHAINS the four stage modules and applies the verify gate's
register/quarantine policy, so a passing character lands in the clip library and a failing
one is parked with its reasons:

    generate.py  -> data/gen/<slug>_portrait.png   (SD 1.5 on SC2 -- HEAVY, guarded here)
    segment.py   -> <slug>_cutout.png + <slug>_bg.png + mask + backend + coverage
    rig_spec.py  -> <slug>_rig.json                (Live2D-style layer regions + warp)
    verify.py    -> Verdict {passed, checks, scores, reasons, degraded, skipped}
        |
        +-- pass  -> ClipGraph.add_clip(idle self-loop on the base pose) + graph.save()
        +-- fail  -> log the failing checks + quarantine (do NOT register)

Why a self-loop and not a transition: what the producer has actually MADE at this point is a
still portrait + a cutout + a rig spec -- NOT a baked video. The idle-on-pose held loop is the
one edge a character can claim immediately (its art exists; the player can render it
synthetically until img2vid bakes real footage). We register it with asset=None so the graph's
missing_edges() correctly keeps surfacing it as "owes real footage" -- exactly clip_graph's
SYNTHETIC-placeholder contract. The portrait/cutout/rig paths ride along as tags + the
character node's provenance, so clipsmith's manifest row carries who+where without inventing a
clip that has not been baked.

Pass policy is a FLAG (mirrors VERIFY_CONTRACT.md's kick-back loop):
    strict=True  (default) -> register only on a CLEAN pass: passed and NOT skipped and NOT
                              degraded. A degraded identity (no InsightFace) or a skipped
                              register (Ollama down) is a SOFT pass -> not-yet-registerable;
                              re-run on a host with full deps (SC2) or escalate.
    strict=False (lenient) -> register on any passed verdict, recording the soft-gate flags
                              (degraded/skipped) on the clip row so the soft gate is visible.

Import policy (mirrors the rest of the pipeline -- degrade, don't crash):
  * runtime.clip_graph is pure stdlib -> always imported at module load.
  * segment / rig_spec / verify need numpy+opencv (assumed present in the .venv); they are
    imported LAZILY so this module imports even where they are not, and a dry-run plan can
    still run and report which stages WOULD run.
  * generate needs torch + diffusers + the SD 1.5 weights (on SC2, not every box). It is
    HARD-guarded: we never import it at module load and never call its real renderer here.
    build_character only invokes generation when explicitly asked AND generate imports; the
    self-test never triggers a model download.

    python pipeline/orchestrate.py            # self-test: run the EXISTING phineas portrait
                                              # through segment+rig+verify, print the verdict +
                                              # whether it registered; dry-run plan if no portrait.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# --- pure-stdlib runtime layer: always available -------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "pipeline") not in sys.path:
    # so `import segment` / `import rig_spec` / `import verify` resolve the sibling modules
    # the same way their own self-tests do.
    sys.path.insert(0, str(ROOT / "pipeline"))

from runtime.clip_graph import POSES, Clip, ClipGraph, MANIFEST_PATH  # noqa: E402

# Where generation writes / where we read portraits + cutouts + rigs.
GEN_DIR = ROOT / "data" / "gen"
CHARS_DIR = ROOT / "prompts" / "characters"
QUARANTINE_DIR = ROOT / "data" / "quarantine"

# The character's base pose -- the held idle loop we register on a clean pass. POSES[0] is
# "idle" by the clip_graph vocabulary; kept as a name so a future per-character base pose
# (e.g. a portrait that opens mid-gesture) is a one-line change, not a magic index.
BASE_POSE = POSES[0]  # "idle"


# ==========================================================================================
# Capability probe -- which heavy stages can actually run on THIS box.
# ==========================================================================================
def _probe_stages() -> dict:
    """Best-effort import probe of every stage. Returns {stage: (available, detail)}.

    Pure introspection -- imports modules but invokes NOTHING (no model build, no network,
    no SD download). `generate` is import-probed (it imports torch+diffusers at module top);
    a failure here just means real generation is off-box, which is the normal local case.
    """
    stages: dict[str, tuple[bool, str]] = {}

    # generate: torch + diffusers at module top. Import alone is enough to know if a real
    # render is even possible on this host. We still never CALL its main() here.
    try:
        import importlib

        importlib.import_module("generate")
        stages["generate"] = (True, "torch+diffusers import OK (SD weights pulled on first run)")
    except Exception as e:
        stages["generate"] = (False, f"{type(e).__name__}: {e}")

    # segment: numpy+opencv required; SAM backend is itself import-guarded inside segment.
    try:
        import segment as _seg  # noqa: F401

        try:
            sam_ok = _seg.sam_available()
        except Exception:
            sam_ok = False
        stages["segment"] = (True, f"opencv OK; SAM backend={'available' if sam_ok else 'absent -> grabcut fallback'}")
    except Exception as e:
        stages["segment"] = (False, f"{type(e).__name__}: {e}")

    # rig_spec: numpy+opencv; Haar ships with opencv; dlib landmarks optional.
    try:
        import rig_spec as _rs  # noqa: F401

        try:
            lm = _rs._landmarks_available()
        except Exception:
            lm = False
        stages["rig_spec"] = (True, f"opencv Haar OK; 68pt landmark lib={'present' if lm else 'absent (proportional fallback)'}")
    except Exception as e:
        stages["rig_spec"] = (False, f"{type(e).__name__}: {e}")

    # verify: numpy+opencv required; insightface + ollama are the soft-degrade axes.
    try:
        import verify as _vf  # noqa: F401

        ins = False
        try:
            ins = _vf._insightface_app() is not None
        except Exception:
            ins = False
        stages["verify"] = (
            True,
            f"opencv SSIM OK; identity={'InsightFace' if ins else 'degraded hist+ORB'}; "
            f"register=Ollama-guarded (skips open if unreachable)",
        )
    except Exception as e:
        stages["verify"] = (False, f"{type(e).__name__}: {e}")

    return stages


# ==========================================================================================
# Result type
# ==========================================================================================
@dataclass
class BuildResult:
    """Structured outcome of build_character. Serializable; safe to log/aggregate.

    portrait/cutout/bg/rig are paths (str) of the artifacts that exist on disk; missing
    stages leave them None. `verdict` is the verify.Verdict (or None on a dry-run / pre-gate
    abort). `registered` is True iff a clip row was written to the graph. `clip_key` names the
    edge that was (or would be) written. `quarantined`/`reasons` capture a fail. `dry_run`/
    `plan` describe a no-op planning pass on a box that cannot run the chain.
    """

    slug: str
    portrait: Optional[str] = None
    cutout: Optional[str] = None
    bg: Optional[str] = None
    rig: Optional[str] = None
    backend: Optional[str] = None
    coverage: Optional[float] = None
    verdict: object = None  # verify.Verdict | None (kept loose so this module imports verify-free)
    registered: bool = False
    clip_key: Optional[str] = None
    quarantined: bool = False
    reasons: list[str] = field(default_factory=list)
    dry_run: bool = False
    plan: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        v = self.verdict
        verdict_d = None
        if v is not None:
            verdict_d = {
                "passed": getattr(v, "passed", None),
                "checks": list(getattr(v, "checks", [])),
                "scores": dict(getattr(v, "scores", {})),
                "degraded": list(getattr(v, "degraded", [])),
                "skipped": list(getattr(v, "skipped", [])),
                "reasons": list(getattr(v, "reasons", [])),
            }
        return {
            "slug": self.slug,
            "portrait": self.portrait,
            "cutout": self.cutout,
            "bg": self.bg,
            "rig": self.rig,
            "backend": self.backend,
            "coverage": self.coverage,
            "verdict": verdict_d,
            "registered": self.registered,
            "clip_key": self.clip_key,
            "quarantined": self.quarantined,
            "reasons": list(self.reasons),
            "dry_run": self.dry_run,
            "plan": list(self.plan),
            "error": self.error,
        }


# ==========================================================================================
# Policy: does this verdict earn a clip row?
# ==========================================================================================
def _clean_pass(verdict, strict: bool) -> bool:
    """Register decision from a Verdict under the chosen policy.

    strict  -> passed AND no skipped AND no degraded (a soft gate is not-yet-registerable).
    lenient -> passed (soft-gate flags are recorded on the row, not vetoes).
    """
    if verdict is None or not getattr(verdict, "passed", False):
        return False
    if strict:
        return not getattr(verdict, "skipped", None) and not getattr(verdict, "degraded", None)
    return True


def _char_md_path(slug: str) -> Optional[Path]:
    p = CHARS_DIR / f"{slug}.md"
    return p if p.exists() else None


def _portrait_path(slug: str) -> Path:
    """Canonical location generate.py writes to. The producer reads from here."""
    return GEN_DIR / f"{slug}_portrait.png"


# ==========================================================================================
# The producer loop for ONE character.
# ==========================================================================================
def build_character(
    slug: str,
    *,
    strict: bool = True,
    graph: Optional[ClipGraph] = None,
    manifest_path: Path | str = MANIFEST_PATH,
    portrait_path: Optional[Path | str] = None,
    out_dir: Optional[Path | str] = None,
    regenerate: bool = False,
    allow_generate: bool = False,
    save: bool = True,
    canonical_portrait: Optional[Path | str] = None,
    dry_run: bool = False,
) -> BuildResult:
    """Run one character from portrait -> segment -> rig_spec -> verify -> (maybe) clip row.

    Args:
        slug: character slug; matches prompts/characters/<slug>.md and <slug>_portrait.png.
        strict: pass policy (see _clean_pass). Default True = clean-pass-only registration.
        graph: ClipGraph to register into. None -> load (or start) the manifest at manifest_path.
        manifest_path: where the clip library lives / is saved.
        portrait_path: override the portrait location (default data/gen/<slug>_portrait.png).
        out_dir: where segment/rig write derivatives (default beside the portrait).
        regenerate: if True, re-run generation even if a portrait already exists (needs
            allow_generate AND a working generate stage).
        allow_generate: explicit opt-in to invoke the HEAVY SD 1.5 renderer. Default False --
            generation belongs on SC2; this guard makes a local model download impossible by
            accident. When False and no portrait exists, returns a dry-run plan.
        save: persist the graph to manifest_path after a registration.
        canonical_portrait: optional reference still for the identity re-render check. None for
            a character's FIRST portrait (identity simply does not run -- that pass defines
            canonical, per the contract).
        dry_run: do not run any stage; return the plan of what WOULD run. Useful on a box
            without numpy/opencv, or to preview before committing VRAM on SC2.

    Returns a BuildResult.
    """
    res = BuildResult(slug=slug)
    ppath = Path(portrait_path) if portrait_path else _portrait_path(slug)
    out = Path(out_dir) if out_dir else None
    char_md = _char_md_path(slug)
    stages = _probe_stages()

    # ---- plan (always computed, so dry-run / abort paths can report it) -------------------
    plan: list[str] = []
    portrait_exists = ppath.exists()
    will_generate = (regenerate or not portrait_exists) and allow_generate and stages["generate"][0]
    if will_generate:
        plan.append(f"generate -> {ppath.name} (SD 1.5; HEAVY)")
    elif not portrait_exists:
        plan.append(f"generate -> {ppath.name} (REQUIRED but {'guarded off' if not allow_generate else 'unavailable'} on this host)")
    else:
        plan.append(f"reuse existing portrait {ppath.name}")
    plan.append(f"segment -> <slug>_cutout.png + <slug>_bg.png  [{stages['segment'][1] if stages['segment'][0] else 'UNAVAILABLE'}]")
    plan.append(f"rig_spec -> <slug>_rig.json  [{stages['rig_spec'][1] if stages['rig_spec'][0] else 'UNAVAILABLE'}]")
    plan.append(f"verify_portrait -> Verdict  [{stages['verify'][1] if stages['verify'][0] else 'UNAVAILABLE'}]")
    pol = "clean-pass-only" if strict else "lenient (soft pass registers)"
    plan.append(f"on clean pass -> ClipGraph.add_clip({BASE_POSE}->{BASE_POSE}, asset=None, loopable) + save  [policy: {pol}]")
    res.plan = plan

    if dry_run:
        res.dry_run = True
        return res

    # ---- stage 1: portrait (guarded) -------------------------------------------------------
    if will_generate:
        try:
            import generate as _gen  # noqa: F401

            # generate.main() reads sys.argv; call its building blocks explicitly instead so we
            # never depend on argv state. It has no slug-arg API beyond main(), so we set argv.
            old = sys.argv
            try:
                sys.argv = ["generate.py", slug]
                _gen.main()
            finally:
                sys.argv = old
        except Exception as e:  # SD download/oom/import -> surface, do not crash the loop
            res.error = f"generate failed: {type(e).__name__}: {e}"
            return res
        portrait_exists = ppath.exists()

    if not portrait_exists:
        # No portrait and we are not allowed / able to make one -> hand back the plan as a
        # dry-run so the conductor knows this slug needs SC2's generate step first.
        res.dry_run = True
        res.error = f"no portrait at {ppath} and generation is {'off' if not allow_generate else 'unavailable'}"
        return res
    res.portrait = str(ppath)

    # ---- stages 2-4 need numpy+opencv. If absent, we can only plan. -----------------------
    if not (stages["segment"][0] and stages["rig_spec"][0] and stages["verify"][0]):
        res.dry_run = True
        missing = [s for s in ("segment", "rig_spec", "verify") if not stages[s][0]]
        res.error = f"cannot run chain -- unavailable stage(s): {', '.join(missing)} (see plan)"
        return res

    import segment as seg
    import rig_spec as rs
    import verify as vf

    # ---- stage 2: segment ------------------------------------------------------------------
    try:
        sres = seg.segment(ppath, out_dir=out)
    except Exception as e:
        res.error = f"segment failed: {type(e).__name__}: {e}"
        return res
    res.cutout = str(sres["cutout"])
    res.bg = str(sres["bg"])
    res.backend = sres["backend"]
    res.coverage = float(sres["coverage"])

    # ---- stage 3: rig_spec -----------------------------------------------------------------
    try:
        spec = rs.build_rig_spec(sres["cutout"], out_dir=out)
    except Exception as e:
        res.error = f"rig_spec failed: {type(e).__name__}: {e}"
        return res
    res.rig = spec.get("_path")

    # ---- stage 4: verify gate --------------------------------------------------------------
    try:
        verdict = vf.verify_portrait(
            ppath,
            canonical_portrait=Path(canonical_portrait) if canonical_portrait else None,
            character_md=char_md,
        )
    except Exception as e:
        res.error = f"verify failed: {type(e).__name__}: {e}"
        return res
    res.verdict = verdict

    # ---- decision: register on a (clean / passing) verdict, else quarantine ---------------
    if _clean_pass(verdict, strict):
        if graph is None:
            graph = ClipGraph.load(manifest_path)
        clip = _idle_clip_for(slug, res, verdict)
        graph.add_clip(clip)
        res.registered = True
        res.clip_key = clip.key()
        if save:
            graph.save(manifest_path)
    else:
        # Failing OR soft-pass-under-strict: park it, never register. The failing/soft reasons
        # are the kick-back signal the generator would tweak on (VERIFY_CONTRACT kick-back loop).
        res.quarantined = True
        why = []
        if not getattr(verdict, "passed", False):
            why.append("verify FAILED")
        if strict and getattr(verdict, "degraded", None):
            why.append("degraded=" + ",".join(verdict.degraded))
        if strict and getattr(verdict, "skipped", None):
            why.append("skipped=" + ",".join(verdict.skipped))
        res.reasons = (why or ["not a clean pass"]) + list(getattr(verdict, "reasons", []))
        _quarantine(slug, res, manifest_path)

    return res


def _idle_clip_for(slug: str, res: BuildResult, verdict) -> Clip:
    """The clip row a passing character earns: its held idle self-loop on the base pose.

    asset=None ON PURPOSE -- the producer has made the portrait+cutout+rig, NOT a baked mp4;
    img2vid/RIFE on SC2 bakes the real footage later. Leaving asset=None keeps this edge in
    ClipGraph.missing_edges() so the gen backfill knows it still owes real footage. The
    character + provenance ride as tags so clipsmith's manifest row carries who/where/how.
    loopable=True (a held pose must loop seamlessly). frames=18/fps=24 mirror clip_graph's
    default_graph() hold length.
    """
    tags = ["hold", "synthetic", f"character:{slug}", "stage:portrait+rig"]
    if res.backend:
        tags.append(f"seg:{res.backend}")
    # surface a soft-gate verdict on the row itself so a lenient registration is auditable.
    if verdict is not None:
        if getattr(verdict, "degraded", None):
            tags.append("verify:degraded=" + ",".join(verdict.degraded))
        if getattr(verdict, "skipped", None):
            tags.append("verify:skipped=" + ",".join(verdict.skipped))
    return Clip(
        from_node=BASE_POSE,
        to_node=BASE_POSE,
        asset=None,           # not baked yet -> stays a missing edge for the gen backfill
        frames=18,
        fps=24,
        loopable=True,
        tags=tags,
    )


def _quarantine_dir_for(manifest_path: Path | str) -> Path:
    """Quarantine sits beside the clip library it shadows. For the canonical manifest that is
    data/quarantine/; for a temp/alt manifest (self-test, SC2 scratch) it co-locates with that
    manifest so a test run never litters the repo's data/quarantine/."""
    mp = Path(manifest_path)
    if mp == Path(MANIFEST_PATH):
        return QUARANTINE_DIR
    return mp.parent / "quarantine"


def _quarantine(slug: str, res: BuildResult, manifest_path: Path | str = MANIFEST_PATH) -> Path:
    """Park a failing/soft-pass character's reasons for human review. Never registers.

    Writes a tiny text record under <manifest dir>/quarantine/<slug>.txt (data/ is gitignored
    runtime state). Best-effort -- a quarantine-write failure must not mask the verdict.
    """
    qdir = _quarantine_dir_for(manifest_path)
    rec = qdir / f"{slug}.txt"
    try:
        qdir.mkdir(parents=True, exist_ok=True)
        lines = [f"slug: {slug}", f"portrait: {res.portrait}", f"cutout: {res.cutout}", f"rig: {res.rig}", ""]
        lines += res.reasons
        rec.write_text("\n".join(str(x) for x in lines) + "\n", encoding="utf-8")
    except Exception:
        pass
    return rec


# ==========================================================================================
# Batch + worklist
# ==========================================================================================
def build_all(
    slugs: Iterable[str],
    *,
    strict: bool = True,
    manifest_path: Path | str = MANIFEST_PATH,
    save: bool = True,
    **kw,
) -> list[BuildResult]:
    """Run build_character over many slugs against ONE shared graph (one save at the end).

    Loads the manifest once, registers every clean pass into it, saves once (so a batch is a
    single atomic manifest write, not N races). Per-character options pass through via **kw
    (e.g. allow_generate, dry_run, out_dir). Returns the list of BuildResults in input order.
    """
    graph = ClipGraph.load(manifest_path)
    results: list[BuildResult] = []
    any_registered = False
    for slug in slugs:
        r = build_character(
            slug,
            strict=strict,
            graph=graph,
            manifest_path=manifest_path,
            save=False,            # defer: one write for the whole batch
            **kw,
        )
        any_registered = any_registered or r.registered
        results.append(r)
    if save and any_registered:
        graph.save(manifest_path)
    return results


def worklist(
    manifest_path: Path | str = MANIFEST_PATH,
    required: Optional[Iterable[tuple[str, str]]] = None,
) -> list[tuple[str, str]]:
    """What the gen backfill should make next: the graph's missing (synthetic) edges.

    Thin pass-through to ClipGraph.missing_edges() so the backfill (and the conductor's SC2
    command) has one source of truth for "which transitions still owe real footage." With
    `required`, narrow to the subset of a beat's needed pairs that are absent-or-synthetic.
    """
    graph = ClipGraph.load(manifest_path)
    return graph.missing_edges(required)


# ==========================================================================================
# Self-test  --  run the EXISTING phineas portrait through segment+rig+verify; dry-run if none.
# ==========================================================================================
def _find_phineas_portrait() -> Optional[Path]:
    """Locate a real phineas portrait to self-test against.

    Prefers the canonical pipeline location data/gen/phineas_portrait.png. Falls back to the
    repo-root phineas_portrait.png (an early real 512x512 render present on the build box) so
    the self-test exercises the live segment+rig+verify chain even before the pipeline has
    populated data/gen/. Returns None if neither exists -> caller runs a dry-run plan instead.
    """
    canonical = _portrait_path("phineas")
    if canonical.exists():
        return canonical
    root_render = ROOT / "phineas_portrait.png"
    if root_render.exists():
        return root_render
    return None


def _selftest() -> int:
    import tempfile

    print("=" * 80)
    print("orchestrate.py self-test  (LOCAL only; no SD download; no SC2)")
    print("=" * 80)

    # 1. capability probe -- what would run live on this box vs is guarded/absent.
    print("\n-- stage capability probe (which stages run live HERE) --")
    stages = _probe_stages()
    for name, (ok, detail) in stages.items():
        print(f"  {name:9s}: {'LIVE ' if ok else 'OFF  '} {detail}")

    portrait = _find_phineas_portrait()

    # 2a. no real portrait -> dry-run plan (the brief's no-portrait branch).
    if portrait is None:
        print("\n-- no phineas portrait on disk -> DRY-RUN PLAN --")
        r = build_character("phineas", dry_run=True)
        for step in r.plan:
            print("   -", step)
        print("\n[selftest] dry-run plan produced (generation belongs on SC2).")
        print("=" * 80)
        return 0

    # 2b. real portrait present -> run the LIVE chain into a temp manifest + out_dir, so the
    # self-test never pollutes the repo's data/clips or data/gen.
    where = "data/gen (canonical)" if portrait == _portrait_path("phineas") else "repo root (early render)"
    print(f"\n-- found phineas portrait: {portrait}  [{where}] --")
    print("   running segment + rig_spec + verify LIVE through the producer loop...")

    tmp = Path(tempfile.mkdtemp(prefix="lp_orchestrate_"))
    manifest = tmp / "manifest.json"

    # Strict policy first (the brief's default). On this box register is SKIPPED (Ollama VL
    # model not pulled) and identity does not run (first portrait, no canonical) -> a strict
    # pass will NOT register, which is the correct, honest behavior of the soft gate.
    print("\n-- build_character(strict=True)  [clean-pass-only] --")
    r_strict = build_character(
        "phineas",
        strict=True,
        portrait_path=portrait,
        out_dir=tmp,
        manifest_path=manifest,
    )
    _print_result(r_strict)

    # Lenient policy: a passing-but-soft verdict registers, with the soft-gate flags stamped
    # onto the clip row. This is the path that proves the exact ClipGraph row we write.
    print("\n-- build_character(strict=False)  [lenient: soft pass registers] --")
    manifest2 = tmp / "manifest_lenient.json"
    r_len = build_character(
        "phineas",
        strict=False,
        portrait_path=portrait,
        out_dir=tmp,
        manifest_path=manifest2,
    )
    _print_result(r_len)

    # 3. assertions -- the chain produced real artifacts + a verdict, and the policy gate did
    # what the contract says (lenient registers a passing verdict; strict gates the soft pass).
    assert r_strict.portrait and Path(r_strict.portrait).exists(), "portrait path missing"
    assert r_strict.cutout and Path(r_strict.cutout).exists(), "cutout not produced"
    assert r_strict.rig and Path(r_strict.rig).exists(), "rig json not produced"
    assert r_strict.verdict is not None, "no verdict from verify"
    assert getattr(r_strict.verdict, "passed", False) is True, "phineas portrait should pass the readable gate"

    if r_len.verdict.passed:
        assert r_len.registered, "lenient policy must register a passing verdict"
        assert r_len.clip_key == f"{BASE_POSE}->{BASE_POSE}", f"unexpected clip key {r_len.clip_key}"
        # confirm the row round-trips through the manifest exactly as clipsmith will read it
        g = ClipGraph.load(manifest2)
        c = g.get_clip(BASE_POSE, BASE_POSE)
        assert c is not None and c.is_hold and c.loopable and c.is_synthetic, "idle self-loop row malformed"
        assert (BASE_POSE, BASE_POSE) in g.missing_edges(), "asset=None idle loop must stay a missing edge"
        print(f"\n[selftest] clip row written + round-tripped: {c.key()}  tags={c.tags}")
        # under strict, the same soft verdict should NOT have registered
        if r_strict.verdict.skipped or r_strict.verdict.degraded:
            assert not r_strict.registered, "strict policy must NOT register a soft pass"
            print("[selftest] strict correctly withheld registration (soft gate: "
                  f"skipped={r_strict.verdict.skipped} degraded={r_strict.verdict.degraded})")

    # 4. worklist demo against the lenient manifest.
    print("\n-- worklist (what the gen backfill still owes -- synthetic/missing edges) --")
    wl = worklist(manifest2)
    print(f"   {len(wl)} missing edge(s): {wl}")

    print("\n" + "=" * 80)
    print("self-test OK -- producer loop ran the live segment+rig+verify chain on the real")
    print("phineas portrait; strict gated the soft pass, lenient registered the idle self-loop.")
    print("=" * 80)
    return 0


def _print_result(r: BuildResult) -> None:
    print(f"   slug={r.slug}")
    print(f"   portrait={r.portrait}")
    print(f"   cutout={r.cutout}  (backend={r.backend}, coverage={None if r.coverage is None else round(r.coverage,3)})")
    print(f"   rig={r.rig}")
    if r.verdict is not None:
        v = r.verdict
        print(f"   verdict: passed={v.passed}  checks={v.checks}  degraded={v.degraded}  skipped={v.skipped}")
        for reason in getattr(v, "reasons", []):
            print(f"     {reason}")
    print(f"   registered={r.registered}  clip_key={r.clip_key}  quarantined={r.quarantined}")
    if r.error:
        print(f"   error={r.error}")
    if r.reasons:
        print(f"   reasons={r.reasons[:3]}{' ...' if len(r.reasons) > 3 else ''}")
    if (r.dry_run or r.verdict is None) and r.plan:
        print("   plan (would run):")
        for step in r.plan:
            print(f"     - {step}")


def main() -> int:
    ap = argparse.ArgumentParser(description="living-portraits producer: portrait -> segment -> rig -> verify -> clip row")
    ap.add_argument("slug", nargs="?", help="character slug (omit for self-test)")
    ap.add_argument("--all", nargs="+", metavar="SLUG", help="build several characters into one manifest")
    ap.add_argument("--lenient", action="store_true", help="register a passing-but-soft verdict (default: clean-pass-only)")
    ap.add_argument("--allow-generate", action="store_true", help="opt in to the HEAVY SD 1.5 render (intended for SC2)")
    ap.add_argument("--regenerate", action="store_true", help="re-render even if a portrait exists (needs --allow-generate)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; run no stage")
    ap.add_argument("--worklist", action="store_true", help="print the gen backfill worklist (missing edges) and exit")
    args = ap.parse_args()

    if args.worklist:
        wl = worklist()
        print(f"{len(wl)} missing edge(s):")
        for fn, tn in wl:
            print(f"  {fn} -> {tn}")
        return 0

    strict = not args.lenient

    if args.all:
        results = build_all(
            args.all,
            strict=strict,
            allow_generate=args.allow_generate,
            regenerate=args.regenerate,
            dry_run=args.dry_run,
        )
        for r in results:
            _print_result(r)
            print()
        registered = sum(1 for r in results if r.registered)
        print(f"built {len(results)} character(s); {registered} registered.")
        return 0

    if args.slug:
        r = build_character(
            args.slug,
            strict=strict,
            allow_generate=args.allow_generate,
            regenerate=args.regenerate,
            dry_run=args.dry_run,
        )
        _print_result(r)
        return 0

    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
