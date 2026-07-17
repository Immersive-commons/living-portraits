# VERIFY CONTRACT

The gate (`pipeline/verify.py`) between the **generative pipeline** and the **clip
library**. Living Portraits bakes its art offline, so the gate can be fussy: an asset
becomes a node/edge in the clip graph **only after it passes here**. A failure is not a
crash — it is a *verdict* that kicks the asset back to the generator with a tweaked prompt.

This file is the source of truth for the thresholds, what each check guarantees, and the
kick-back loop. The numeric constants live at the top of `verify.py`; tune them there and
keep this doc in sync.

---

## Where the gate sits

```
prompts/ -> generate.py (SDXL) -> SAM segment -> Live2D rig -> img2vid+RIFE -> [ VERIFY GATE ] -> CLIP LIBRARY
                  ^                                                                  |
                  |  kick-back: reject + regenerate with a tweaked prompt (<= N)     |
                  +------------------------------------------------------------------+
```

A still portrait is gated by `verify_portrait(...)` before it becomes the canonical node
art. A baked clip (idle loop, walk, transition/bridge) is gated by `verify_clip(...)`
before it becomes an edge. Only `passed == True` assets are registered.

---

## The four checks

Each is a pure function returning `(passed: bool, score: float, reason: str)`.

| Check | Signature | Guarantees | Threshold | Direction |
|---|---|---|---|---|
| **continuity** | `continuity(clip_end_frame, target_node_frame)` | A bridge/transition/walk clip's **last frame** matches the graph node it claims to arrive at, so the player can splice the next clip without a visible jump. | `CONTINUITY_THRESHOLD = 0.78` (SSIM) | `>=` passes |
| **loopability** | `loopability(first_frame, last_frame)` | An idle loop **closes**: first and last frame are near-identical, so the seam does not pop on repeat. | `LOOP_THRESHOLD = 0.92` (SSIM) | `>=` passes |
| **identity** | `identity(candidate_img, canonical_portrait)` | The asset is still the **same character** — a re-render did not quietly swap the actor's face. | `0.55` InsightFace cosine · `0.62` degraded hist+ORB | `>=` passes |
| **register** | `register(text_or_image, character_md)` | A spoken aside (or the portrait's look) stays in the character's **theatrical, fourth-wall voice** per its contract in `prompts/characters/<slug>.md`. | yes/no from local Ollama | yes passes |

### SSIM (continuity, loopability)

Gaussian-windowed structural similarity (Wang et al. 2004): 11×11 window, σ = 1.5, the same
formulation as `skimage.metrics.structural_similarity`, implemented with OpenCV's separable
Gaussian blur so the gate carries **no extra dependency** beyond numpy + opencv. Inputs are
converted to grayscale float `[0, 1]` and shape-matched (the smaller is resized to the
larger) before scoring. Output is the mean of the SSIM map, in `[-1, 1]`; `1.0` is identical.

SSIM measures **structure, not color**: a recolored-but-aligned frame scores ~1.0, a
shifted/scaled pose scores mid (~0.77 in the self-test), a half-truncated/corrupt frame
scores low (~0.49), and noise scores ~0. That is the intended behavior — continuity/loop
care about whether the *shape* lines up, not the exact palette (lighting drifts between
renders).

> **Threshold rationale.** `0.92` for loops is strict: a loop seam is watched thousands of
> times, any structural mismatch reads as a twitch. `0.78` for continuity is looser: a
> transition only needs to *land near* the target pose before the next clip takes over, and
> the source pose legitimately differs from the target.

### identity — degrades, never lies

Preferred path is **InsightFace** (`buffalo_l`, ArcFace): cosine of the largest detected
face's L2-normalized embedding, threshold `0.55`. If InsightFace is **absent**, the model
pack is missing, or no execution provider is available, the check **falls back** to a coarse
HSV-histogram-correlation + ORB-match-rate blend, the verdict is **marked `degraded`**, and
the threshold tightens to `0.62` to partly compensate for the cruder signal.

> The degraded path is **not a face matcher** — it cannot reliably tell two different faces
> apart. It exists to catch gross failures (blank/corrupt render, wholly different image) and
> to keep the gate runnable offline. The reason string says so explicitly. When identity
> *correctness* matters, install InsightFace on the gate host.

Under InsightFace, **no detectable face in either image is a FAIL** (with reason) — a
portrait the face model cannot even find a face in must not enter the library silently.

### register — soft check, degrades **open**

Asks a local Ollama model a yes/no, grounded in the character's `Theme:`/`Voice:` contract:
*"does this stay in the theatrical fourth-wall voice?"* Text asides go to a chat model
(`qwen3:8b`, the director's model); portrait images go to a vision model (`qwen2.5vl:7b`).
The HTTP call is guarded (urllib, `OLLAMA_TIMEOUT = 20s`, try/except over every failure
mode).

If Ollama is **unreachable**, the **model is not pulled** (404), the response is malformed,
or the model **cannot decide**, the check is **SKIPPED**: it returns `passed=True` with the
sentinel `score = -1.0` and a `"register: SKIPPED (...)"` reason.

> **register is the one check that degrades OPEN.** It is taste-level and soft; a hard infra
> failure (Ollama down) must not block the entire clip library. The other three checks
> degrade *closed* enough to stay safe (continuity/loop have no network dep; identity fails
> closed on unreadable input). The `-1.0` score sentinel lets callers/telemetry distinguish a
> *skip* from a real `0.0` judgement.

---

## Verdict semantics

`Verdict { passed, checks, scores, reasons, degraded, skipped }`.

- `passed` is the **AND of every check that actually ran with a real pass/fail**.
- A **SKIPPED** check (`score == -1.0`, or `"SKIPPED"` in its reason) does **not** veto the
  verdict — it is recorded in `skipped[]` only.
- A **degraded** check (e.g. identity without InsightFace) **still counts** toward `passed`;
  its name is also recorded in `degraded[]` so the caller and telemetry can see the gate ran
  soft.
- `Verdict.add(name, (ok, score, reason))` is the only mutator; `Verdict.summary()` renders a
  human line (`PASS`/`FAIL` + `degraded=…; skipped=…` tags + per-check reasons).

This means: **`passed=True` with a non-empty `skipped`/`degraded` list is a real pass under a
reduced gate, not a clean pass.** A consumer that wants the strict gate should refuse to
register assets whose verdict carries `skipped`/`degraded` (policy choice, left to the
caller).

---

## Orchestrators — what runs when

### `verify_portrait(portrait_img, canonical_portrait=None, character_md=None)`

| Condition | Checks run |
|---|---|
| always | `readable` (self-SSIM — catches blank/corrupt/half-written files) |
| `canonical_portrait` given | `identity` (re-render must match the established face) |
| `character_md` given | `register` (the portrait's look, via the VL model) |

The **first** portrait of a character has no reference, so identity is simply not run
(there is nothing to drift from yet — that first pass *defines* canonical).

### `verify_clip(clip_frames, target_node_frame=None, canonical_portrait=None, kind="bridge", aside=None, character_md=None)`

| `kind` | Checks run |
|---|---|
| `idle` / `loop` | `loopability(first, last)` |
| `bridge` / `transition` / `walk` | `continuity(last, target_node_frame)` — **requires** `target_node_frame` |
| (other) | hard FAIL — unknown kind |
| `canonical_portrait` given (any kind) | `+ identity(last, canonical_portrait)` |
| `aside` **and** `character_md` given | `+ register(aside, character_md)` |

`clip_frames` must have **≥ 2 frames** (first + last); fewer is a hard FAIL with reason.

---

## Kick-back loop (caller's responsibility)

The gate **judges**; it does not regenerate. The orchestrator that owns generation runs the
loop:

```
attempt = 0
while attempt < MAX_RETRIES:                 # MAX_RETRIES = 3 recommended
    asset  = generate(slug, prompt)          # generate.py (still) / clip baker
    v = verify_portrait(asset, canonical, char_md)   # or verify_clip(...)
    if v.passed and not v.skipped and not v.degraded:   # strict policy: clean pass only
        register(asset); break
    attempt += 1
    prompt = tweak(prompt, v)                 # nudge by the failing check (see below)
else:
    quarantine(asset, v.reasons)              # park for human review; do NOT register
```

**Tweak heuristics by failing check** (the prompt nudge depends on *why* it failed):

| Failed check | Prompt / param nudge |
|---|---|
| `loopability` | re-bake the loop with the **last frame pinned to the first** (or shorten the loop); for img2vid, lower motion strength. |
| `continuity` | re-bake the **transition** ending closer to the target node — seed/condition the final frame on the target pose; reduce travel distance. |
| `identity` | re-render the still with a **stronger identity anchor** (reference image / IP-Adapter / lower CFG drift); never accept a face the model can't find. |
| `register` | re-ask the stage-manager for the aside with a **stronger 4th-wall directive** (raise the theatrical register; ban the sincere-modern voice). This is the only one that loops on *text*, not pixels. |

**Invariants of the loop:**

- **Fail-then-tweak, never fail-then-lower-threshold.** Thresholds are fixed contract; the
  fix is a better asset, not a looser bar.
- **Max N retries (3), then quarantine** — park the asset + the failing reasons for human
  review. Never silently register a failed asset, and never loop forever (offline batch
  on a shared box; a runaway loop starves the catalog of VRAM).
- **A `degraded`/`skipped` verdict is a soft pass.** Strict callers (above) treat it as
  not-yet-registerable and either retry on a host with full deps or escalate; lenient
  callers may register but should record the soft-gate flags alongside the asset.

---

## Dependency / environment matrix

| Dependency | Required? | Absent ⇒ |
|---|---|---|
| numpy, opencv | **yes** | gate cannot run (assumed present in `.venv`) |
| insightface (+ model pack, provider) | no | `identity` runs **degraded** (hist+ORB, stricter threshold, marked) |
| Ollama + chat/VL model pulled | no | `register` is **SKIPPED** (degrades open) |

No network at **import** time. The module imports and the self-test runs with **neither**
InsightFace nor Ollama present.

**On the build host (this machine), as verified:** numpy 2.2.6 + opencv 4.12.0 present;
InsightFace **absent** (identity degraded); Ollama reachable but the chat/VL models are
**not pulled** (register skipped). **On SC2** (the production gate host) install InsightFace
and pull `qwen3:8b` (already present per the director) + a VL model to run the gate at full
strength.

---

## Self-test

```
python pipeline/verify.py
```

Builds synthetic numpy images (no files, no required network) and asserts every check's
PASS **and** FAIL path: identical frames pass loopability; a shifted pose fails it and fails
continuity; a totally different image fails identity even degraded; a truncated frame fails a
loop; the orchestrators AND correctly; skipped checks don't veto. Prints each verdict.
