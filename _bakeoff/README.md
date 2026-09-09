# FLF2V bake-off — which Higgsfield model can walk between two graph nodes

Run 2026-08-09. Eight Higgsfield video models handed **three real edges** from
`data/clips/video_graph.live.json`, each at the cheapest settings it exposes.
Rendered page: `flf2v-bakeoff.html` (internal -- 5.6 MB of inlined clips, not in this repo;
the measurements below are the part that matters,
clips inlined at panel size + a 480px detail view).

The question this answers: the clip graph needs **one clip per edge**, so the metric
that matters is **credits per edge**, not credits per anything else.

## Findings that cost money to learn

**1. `hf generate cost` prices work the API cannot do.** It quoted a price for every
model in the set when handed a start AND an end frame. Two then refused the render:

| Model | Quoted | Refused with |
|---|---|---|
| `kling3_0_turbo` | 7.5 | `medias: List should have at most 1 item after validation, not 2` |
| `grok_video_v15` | 22.5 | `medias may only contain start_image, image, or audio` |

The cost endpoint does not validate the medias count. **Only a render proves support.**

**2. Accepted params are not honoured params.** `seedance_2_5` took an explicit
`--height 512 --width 512` without complaint and returned **1280×720 landscape**.
The panels are square, so it would need a centre-crop. Disqualified on format.

**3. `sound=off` is a 25% discount on Kling.** `kling3_0` @ 1:1 / 5s is 10 credits
with sound, **7.5 without**. The portraits speak through their own TTS
(`pipeline/tts.py`), so the baked audio track is dead weight. Every model returned
an audio stream unrequested.

**4. The plan caps CONCURRENT jobs at 8** (`error_type: rate_limit_reached`,
`concurrent_jobs_limit: 8`). Submitting more just bounces.

**5. Failed renders are refunded.** Confirmed in `hf account transactions` — both a
failed Seedance 2.5 (32.5) and two failed Mini jobs (5 each) came back.

## What each model actually delivered

| Model | Delivered | fps | Credits/edge | Verdict |
|---|---|---|---|---|
| `seedance_2_0_mini` | 640×640 | 24 | **5** | cheapest square option |
| `wan2_7` | 960×960 | **30** | 7.5 | only 30fps in the set |
| `kling3_0` | 960×960 | 24 | 7.5 (sound off) / 10 | fastest to return (124–190s) |
| `seedance_2_0` | 640×640 | 24 | 15 | same envelope as Mini at 3× price |
| `wan3_0` | — | — | 12.5 | **far too slow** — >20 min, never landed |
| `seedance_2_5` | 1280×720 | 24 | 32.5 | not square, ignored requested size |
| `kling3_0_turbo` | — | — | — | no end-frame slot |
| `grok_video_v15` | — | — | — | no end-frame slot |

## Budget — and what got wired in

Subscription grants **3000 credits/month** on the 23rd, and **unused credits are
wiped** — the 2026-07-23 rows show `Subscription Credits +3000` alongside
`Subscription Credits Reset -1953`. No rollover, so the allowance is a per-month
ceiling, not a bank.

**There is no daily cap.** The only limit the API enforces is **8 concurrent jobs**;
at Kling's ~150s that is ~190 renders/hour, so the whole month could be spent in an
afternoon. A per-day number is a *pace*, not a ceiling.

| Per edge | Credits | Edges/month | Even pace/day |
|---|---|---|---|
| Seedance 2.0 Mini | 5 | 600 | 20 |
| **Kling v3.0 · sound off** | **7.5** | **400** | **13** |
| Wan 2.7 · 30fps | 7.5 | 400 | 13 |
| Kling v3.0 · sound on | 10 | 300 | 10 |

**Chosen: Kling v3.0, sound off, 13 clips/day, 6 per character.** Implemented in
`pipeline/autogen.py` (`CLIP_DAILY_CAP` / `CLIP_CHAR_CAP`) and `pipeline/hf_gen.py`.
The system cap is checked before the per-character one.

**Both directions are generated.** The reverse edge could be had free by flipping the
forward clip, and it looks wrong — reversed playback runs the physics backwards, so
cloth settles upward and a figure rising from a chair reads as being pulled into it.
The graph is walked in both directions, so both get real motion. A pose is therefore
**4 clips** (forward + reverse + 2 idles) ≈ 30 credits with its still.

At 6 clips/character/day that is **~1.5 poses per character per day** — deliberately
slow. The characters generate autonomously with no approval step, so the budget is
the only thing bounding them; 13/day × 7.5 × 30 = 2925 against a 3000 allowance, so a
full-burn month lands just inside the plan.

`wan3_0` never finished a single edge — all three jobs sat `in_progress` past 43
minutes, on top of two earlier 20-minute timeouts under a blocking wait. Abandoned.

## Reproduce

`lp_fire.sh` submits without `--wait` (serial submits dodge the one-time-use token
rotation race); `lp_harvest.sh` polls and refills slots under the concurrency cap;
`build_compare.py` downloads + probes actual specs; `emit_page.py` inlines everything.
Do NOT re-run `hf` anywhere but the owner host (`node`) — see
`life/projects/j4me/infra/deploy/hf_proxy/README.md`.
