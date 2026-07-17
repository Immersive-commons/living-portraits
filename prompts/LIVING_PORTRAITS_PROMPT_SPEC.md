# Living Portraits — dynamic prompt spec (the "literal personality" model)

How we make Harry-Potter-style moving portraits real on the LED panels, where the
people in the frames have **literal personalities** that drive how they look, move,
and speak. The per-character spec is a JSON file (`prompts/characters/<id>.json`);
this doc is the rationale + the pipeline that consumes it.

> Status 2026-05-27: schema + `phineas.json` authored. Tech stack chosen below.
> Generation/animation not yet wired (SD1.5 anchor proof exists; SDXL + InstantID +
> LivePortrait still to install). The old text/aside show is what's live on the panels now.

---

## 1. The vision

- **The LED bezel IS the frame.** No painted frame, no wall — the figure is edge to
  edge, filling the panel. (Earlier framed/walled attempts were wrong.)
- **They are people pretending to be paintings.** Not a still that subtly breathes —
  a real person holding a portrait pose who is actually, always, subtly moving, in a
  **seamless loop**, and who breaks the fourth wall.
- **Literal personalities.** Each portrait has a real personality that is the *primary
  spec*: it dictates the face, the costume, the idle motion, the gaze, and the lines.

## 2. How HP portraits actually work (canon → our design principle)

Per Wizarding World, a magical portrait **imitates its subject's demeanour and
favourite phrases**, and how interactive it is scales with the subject; the richest
ones (headmasters) are *trained* to behave exactly like the person
([source](https://www.harrypotter.com/features/how-do-magical-portraits-actually-work)).

**Design principle that falls out:** personality is not decoration applied after the
art — it is the seed. We encode a personality profile, then *derive* appearance,
motion, and dialogue from it. Sir Cadogan challenges everyone to a duel; the Fat Lady
guards her food and drink. Phineas declaims and sulks.

## 3. The tech stack (researched 2026-05-27)

| Layer | Pick | Why | Source |
|---|---|---|---|
| Anchor image | **SDXL** (not SD1.5) | SD1.5@512 gives melty faces; SDXL is sharp at panel scale | — |
| Identity lock | **InstantID** / IP-Adapter-FaceID-plusv2 | one face stays the SAME across re-gens + both panels from a single anchor, zero-shot | [InstantID](https://arxiv.org/pdf/2401.07519) |
| Motion | **LivePortrait** (Kuaishou) | keypoint retarget of expression/head/eye/gaze onto a still; **fast** (not diffusion-per-frame, so it dodges the LTX/Turing wall) and **directly controllable** — this is what makes "declaim vs sulk" literal | [LivePortrait](https://github.com/KlingTeam/LivePortrait) |
| Personality→motion | **Big-Five → gaze/blink/head** mapping | traits predict gaze targets, blink rate, head motion; lets a profile drive idle behaviour | [personality-driven gaze](https://arxiv.org/pdf/2012.02224) |
| Loop | LivePortrait neutral→behaviour→neutral (native loop); SVD+boomerang fallback | start==end so clips chain at the anchor | — |

**Ruled out on hil's 2080 Ti (Turing, fp16, no FlashAttention):** Wan FLF2V (Ampere-only)
and LTX-Video (loads, but ~9 min/step ≈ 5 h/clip). LivePortrait is the path *because*
it isn't a per-frame diffusion model.

## 4. The JSON schema (`living-portrait.character/v1`)

One file per character. Blocks:

- `concept` — the vision + the canon rule + the one-paragraph **essence**.
- `personality` — `big_five` (0–1) + `traits` + `demeanour` + `catchphrases`. **The seed.**
- `appearance` — face / hair / complexion / costume / default expression. Derived to read as the personality.
- `style` — medium / school / lighting / palette / brushwork / finish ("no craquelure — he's alive").
- `framing` — edge-to-edge crop, no frame, 1:1, eyeline.
- `generation` — `model`, `identity_lock`, `positive_template` (assembled from the fields), `negative`, `params`.
- `motion` — `engine`, `loop_contract`, `gaze`, `blink_rate_bpm`, and **`idle_behaviors`** (each with a weight + the traits that drive it).
- `voice` — register for the director (already wired).
- `dynamic` — **what makes it a *dynamic* prompt:** per loop, pick a behaviour by weight (variety); `mood_reweights` lets the director's current `data/mood.json` bend the body language; `render_pseudocode` is the consume path.

## 5. The pipeline (how the JSON becomes a living portrait)

```
character.json
   │  generation.positive_template.format(**spec)  + identity_lock
   ▼
anchor.png  ── SDXL + InstantID ──►  the canonical face (same every time)
   │
   │  for each idle_behavior (weighted by personality, re-weighted by mood):
   ▼
loop clip ── LivePortrait(anchor, driving=behavior, return_to=neutral) ──► seamless loop
   │
   ▼
clip_graph:  node = anchor pose,  edges = behaviour loops   (existing runtime/clip_graph.py)
   │
   ▼
player picks a random edge each cycle ──► endless, non-repeating, in-character idle
   │
director/stage_manager.py (qwen3)  ──► the fourth-wall line, in `voice`, re-weighting mood
```

The personality shows up **three** times: the **face** (appearance derived from it),
the **body** (idle_behaviors + gaze + mood reweights), and the **words** (voice + the
director). That triangulation is what sells "this painting has a real personality."

## 6. Characters

- **Phineas** (`phineas.json`) — faded tragedian, PANEL A. Authored.
- **Seraphina** (PANEL B) — the brighter, quicker wit next door. TODO: author `seraphina.json`
  from `seraphina.md` (high extraversion + openness, low neuroticism; behaviours like
  *amused-aside-to-viewer*, *delighted-at-his-sulk*, *quick-wit-flash*).

## 7. Next steps to make it real

1. Install on hil's `.venv-gen`: SDXL base, an InstantID (or IP-Adapter-FaceID) pack, LivePortrait.
2. Generate Phineas's anchor with SDXL + InstantID from `phineas.json.generation`.
3. Author `seraphina.json`; generate her anchor.
4. Wire `pipeline/_bake_liveportrait.py`: anchor + behaviour → loop clip; register on `clip_graph`.
5. Teach the player to pick a random behaviour edge per cycle, re-weighted by `data/mood.json`.

> GPU note: generation/animation contends with the director's qwen3 for the 2080 Ti's
> 11 GB (running two GPU jobs crashed SVD earlier). Pause `lp-director` + `lp-watchdog`
> during a bake, restore after — or bake while the show is parked.
