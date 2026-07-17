# living-portraits -- SC2 full-local bring-up (install/)

Installs the **producer-side** generative deps on supercommons2
(`immer@<producer-host>`) and runs the producer backfill so every rendered
portrait lands in the clip library.

This is the **second** install step, run AFTER the portraits exist and AFTER
`install_tasks.ps1` has made the runtime show durable:

| Script | venv it wires | What it does |
|---|---|---|
| `install_tasks.ps1` | `.venv` (runtime) | player + director + watchdog scheduled tasks |
| `sc2_bringup.ps1` (this) | `.venv-gen` (generative) | piper TTS voices, insightface identity, SAM seg deps, then backfill the clip graph |

Built LOCALLY, deployed + run BY THE CONDUCTOR. Do not run from a dev machine --
it pip-installs into SC2's `.venv-gen` and writes the clip library on SC2.

## SC2 facts (the host this targets)

- Host: `immer@<producer-host>`
- Project: `C:\Users\immer\living-portraits`
- Generative venv: `.venv-gen` (torch **2.5.1+cu121 already installed**)
- Runtime venv: `.venv` (player/director -- not touched here)
- qwen3 + SD 1.5 already local. The Ollama **VL** model is the one optional pull
  (see "Known gaps" below).

## What the script does (5 wrapped steps, each prints OK / SKIP / FAIL)

Every step is wrapped so **one failure does not abort the rest**. Ordered:

1. **piper-tts install** -- `pip install piper-tts` into `.venv-gen`.
2. **piper voices download** -- `python -m piper.download_voices en_GB-alan-medium
   en_GB-jenny_dioco-medium --data-dir data\tts\voices`. **SKIP** when both
   `.onnx` files are already present (idempotent -- no re-fetch of ~60-120MB
   voices). These are the two voices `pipeline/tts.py` maps: `alan` = phineas
   (patrician baritone), `jenny_dioco` = seraphina (bright wit).
3. **insightface + onnxruntime install** -- `pip install insightface onnxruntime`
   (CPU build is fine). With insightface present, `verify.py`'s identity check
   upgrades from the degraded hist+ORB fallback to real ArcFace embeddings.
4. **SAM seg deps** -- `pip install iopath hydra-core omegaconf` into `.venv-gen`.
   The SAM **weights are gated on HuggingFace and are NOT fetched** -- `segment.py`
   import-guards the whole SAM stack and falls back to GrabCut if the
   weights/creds are absent. A pip failure here is recorded as **SKIP** (not
   FAIL): the GrabCut fallback needs none of these deps.
5. **producer backfill** -- `.venv-gen\Scripts\python.exe pipeline\orchestrate.py
   --all phineas seraphina` (run with cwd = project root). This runs
   segment -> rig -> verify -> clip-row over each character's **existing**
   portrait. **No `--allow-generate`** -- portraits must already be rendered; the
   conductor renders them separately (the HEAVY SD 1.5 step, deliberately guarded
   off here so a bring-up can never trigger a model download).

A 6th advisory step prints an import probe (`torch / cv2 / piper / insightface /
onnxruntime / iopath` AVAILABLE-or-MISSING) and the producer worklist (which
transitions still owe real footage). Advisory only -- never fails the bring-up.

The script ends with a per-step summary table (`OK` / `SKIP` / `FAIL` per step).

### The one hard stop

If `.venv-gen\Scripts\python.exe` does not exist on the host, the script prints a
`FATAL` line and `exit 1` before any step -- there is nothing to bring up. Every
other failure is per-step and non-fatal to the rest.

## The exact command the conductor runs

Same pattern as the watchdog deploy in `install/README.md` -- two files go to the
host (`watchdog.ps1` + `install_tasks.ps1` are already there from the runtime
deploy; this adds `sc2_bringup.ps1`), then run it on SC2 via `Invoke-RemotePS`.

```powershell
# 0. Dot-source the remote-PS helper (handles BOM + base64 EncodedCommand).
. C:\Users\jtole\Documents\2026\life\scripts\invoke-remote-ps.ps1

# 1. scp this script to SC2 (pure ASCII, no BOM -- straight scp is safe).
scp C:\Users\jtole\Documents\2026\life\projects\living-portraits\install\sc2_bringup.ps1 `
    immer@<producer-host>:C:/Users/immer/living-portraits/install/sc2_bringup.ps1

# 2. Run it on SC2. Streams every step's OK/SKIP/FAIL + the final summary back.
Invoke-RemotePS -Host_ immer@<producer-host> -Script @'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\immer\living-portraits\install\sc2_bringup.ps1
'@
```

> The backfill (step 5) needs `data\gen\<slug>_portrait.png` for each slug to
> already be on SC2. If the portraits are not rendered yet, run the conductor's
> separate render step first, then re-run this script (it is idempotent).

## Expected output

A healthy first run on a host that already has the portraits looks like:

```
STEP: piper-tts install                        => OK
STEP: piper voices download                    => OK     (or SKIP on a re-run)
STEP: insightface + onnxruntime install        => OK
STEP: SAM seg deps (iopath + friends)          => OK     (or SKIP if a dep won't pip)
STEP: producer backfill (orchestrate ...)      => OK
STEP: capability verify + worklist             => OK
...
bring-up summary
OK     piper-tts install
SKIP   piper voices download
OK     insightface + onnxruntime install
OK     SAM seg deps (iopath + friends)
OK     producer backfill (orchestrate --all, strict)
OK     capability verify + worklist
```

The producer backfill prints a per-character block (portrait / cutout+backend /
rig / verdict / registered) and a `built N character(s); M registered.` line.

**`M` may be 0 even on a fully-OK run -- that is expected, not a failure.** The
backfill runs in **strict** mode (clean-pass-only). A character registers its idle
clip only on a clean pass: `passed AND not degraded AND not skipped`. The
register check (`verify.py:register`) sends the **portrait image** to the Ollama
**VL** model; if that model is not pulled, register is **SKIPPED** (it degrades
open, never hard-fails), which makes the verdict a soft pass -- so strict withholds
registration. This is the honest behavior of the soft gate, documented in
`VERIFY_CONTRACT.md`.

To get clean strict registrations, the conductor pulls the VL model on SC2:

```
ollama pull qwen2.5vl:7b
```

(then re-run this script -- step 5 will register the clean passes). Alternatively,
a `--lenient` backfill registers a passing-but-soft verdict with the soft-gate
flags stamped on the clip row, but the default + intended SC2 path is strict + a
pulled VL model.

## Re-running safely (idempotent)

Safe to run any number of times:

- **pip steps** -- `pip install` is a no-op when the package is already satisfied.
- **voice download** -- SKIPs when both `.onnx` files exist; never re-fetches.
- **producer backfill** -- registers only NEW clean passes. `ClipGraph.add_clip`
  is keyed by the `from->to` edge, so re-running never duplicates the idle
  self-loop row; a character already registered stays a single row.

A failed step on a prior run is simply re-attempted on the next run -- fix the
underlying cause (e.g. pull the VL model, render the missing portrait) and re-run.

## Known gaps / the one thing that can still fail on SC2

- **SAM 3.1 weights are gated** (HuggingFace). This script installs the pip-able
  SAM deps (`iopath` + config libs) but does **not** fetch the checkpoint -- it
  needs gated-repo creds and is a separate, large download. Without the weights,
  `segment.py` runs **GrabCut** (deterministic, dependency-free, always
  available). The cutouts are good enough for the rig + verify chain, so this is a
  quality nicety, not a blocker. If SAM segmentation is wanted later, the
  conductor authenticates to the gated repo and pulls the checkpoint out-of-band;
  no code change is needed (segment.py picks SAM up automatically once it imports
  and a checkpoint is available).
- **0 registered under strict** is expected until `ollama pull qwen2.5vl:7b` runs
  (see "Expected output"). Not a failure of this script.
- **Missing portraits** make step 5 FAIL with a clear message -- render them via
  the conductor's separate step, then re-run.
- **onnxruntime** is installed as the CPU build by default. That is sufficient for
  the single-still identity check; no GPU onnxruntime is required.

## Files

- `sc2_bringup.ps1` -- this bring-up script (idempotent; run once, re-runnable).
- `BRINGUP.md` -- this file.
- (siblings, from the runtime deploy) `install_tasks.ps1`, `watchdog.ps1`,
  `README.md`.

## Notes / pitfalls

- ASCII only, straight quotes, `--` not em-dashes throughout -- this file and the
  script scp to a Windows-codepage host; smart punctuation would break the PS
  parser. (Same rule as the runtime install files.)
- This script does NOT render portraits and does NOT touch the runtime `.venv` or
  any scheduled task. It is purely producer-side: gen deps + clip-graph backfill.
- It installs into `.venv-gen`, not `.venv`. The two venvs are deliberately
  separate so the lean runtime show never imports the multi-GB gen stack.
