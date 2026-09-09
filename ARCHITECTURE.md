# Living Portraits — codebase map

Written 2026-08-09 against the repo and the live production host `hil`. Read-only audit;
no code changed. Every claim is cited `file.py:line`. Anything I inferred rather than
confirmed is marked **(inferred)**. Closing section lists what I did not verify.

**Updated 2026-08-11 for v0.4.0** — new modules (`lived`, `journal_score`,
`graph_provenance`, `edge_style`, `reflect`), two new single-writer data contracts, two
dwell-dependent policy terms, the `health/` detector suite, and the deploy ledger.

**Line citations are checked by a test, not by hope.** They had drifted 50-500 lines in
sections 1 and 8 — the v0.4.0 pass re-checked the module and contract claims but not the line
numbers, and `tick()` had moved 219 lines while still being cited at its old one.

`tests/test_doc_citations.py` now resolves every citation that names a symbol, with `ast`,
and fails when one stops pointing at what it names. So the honest statement about this
document is no longer a date or a commit — it is that the suite is green. Range citations (`file.py:12-40`) point at blocks
rather than definitions and cannot be checked mechanically; they were verified by hand in the
same pass.

---

## If you read nothing else

1. **There are two loops and they never touch each other's files.** A RUNTIME loop paints
   panels at 10 fps from pre-baked gifs and generates nothing. A GENERATION loop proposes
   and buys new clips on a 20-minute timer. They meet only through `data/clips/video_graph.json`.
2. The runtime loop is `_preview_graph.py` (scheduled task `lp-preview`), 10 fps
   (`_preview_graph.py:58`), playing gifs listed in the video graph.
3. The generation loop is `director/heartbeat.py` (`lp-mind`, ~4 min) proposing, and
   `pipeline/autogen.py` (`lp-gen`, every 20 min) building.
4. **The walker's import set must stay pure stdlib and import-safe** — `_preview_graph.py:33`
   imports `circadian`, `lived`, `mind`, `policy`; `:40` adds `edge_style` fail-soft; `mind`
   pulls `pathfind` (`runtime/mind.py:33`, `runtime/policy.py:34`). NOT `video_graph` — the
   walker reads the built JSON, which is what makes the two loops independent. And not the
   whole directory: the rendering half of `runtime/` imports numpy and cv2 at module level,
   and §2's table marks each file PURE or HEAVY. All network/LLM lives in `director/`.
5. Every shared file has exactly ONE writer. That is the whole concurrency design
   (`AUTONOMY.md:47-69` — that file lives in the private tree and is not published here;
   `scripts/release.py:55` lists it as a release-sync target). Adding a second writer to any
   of them is the way to break this system.
6. Precedence for what a portrait does: **circadian (clock) > mind (LLM goal) > walk**.
   Resolved in `_preview_graph.py:221-269`; in production (`--policy`) it is
   `_preview_graph.py:271-301`.
7. `data/clips/video_graph.json` is authoritative. `video_graph.live.json` is a stale local
   artifact that does not exist on hil — ignore it.
8. `runtime/video_graph.py` is the live graph. `runtime/clip_graph.py` is the older,
   parallel v2 graph — do not confuse them. It is NOT parked: `pipeline/orchestrate.py:64`,
   `_bake_liveportrait.py:419,506` and `_bake_animatediff_motion.py:131` all import it, so a
   change there reaches the generation stack.
9. The Midjourney generation path is **retired but intact**. It has been returning HTTP 403
   on every image upload since 2026-07-02. Higgsfield replaced it on 2026-08-08.
10. hil is NOT the Higgsfield session owner — `node` is, with one-time-use refresh tokens.
    **Never run `hf` on hil** (`pipeline/hf_gen.py:10-14`).

---

## 1. The two loops

### RUNTIME loop — plays clips, generates nothing

```
lp-preview  (scheduled task, AtLogon, MultipleInstances=IgnoreNew)
  pythonw _preview_graph.py --a phineas --b maxx --loops 1 --a-tp 0.3
          --b-pose idle --b-tp 0.4 --mind --max-idle-secs 0 --policy

  every frame (10 fps, _preview_graph.py:58, :413-451):
    GraphCycler.frame()                 _preview_graph.py:342
      -> blit one gif frame per panel
      -> at clip end: _pick()           _preview_graph.py:221
           reads data/mind/intent.json  (mtime-cached, :126-138)
           writes data/mind/pose/<char>.json when node or dwell changes (:140-158)
           hot-reloads video_graph.json on mtime change (:160-186)
```

Two panels: A 256x256 @ (0,0), B 192x192 @ (256,0), one 448x256 borderless
topmost window (`_preview_graph.py:39-40`), re-pinned every 100 frames (`:42`, `:431`).
It never quits on pygame QUIT — only operator ESC (`:463-466`) — and rebuilds the window
when a fullscreen app steals the display (`:433-450`).

**Nothing in this loop calls a network or an LLM.** The `--mind` and `--policy` flags only
make it *read* a file the other loop wrote.

### GENERATION loop — proposes and buys clips

```
lp-mind     (AtLogon, long-lived process; NIGHT_INTERVAL 900s / DEFAULT_INTERVAL 240s)
  pythonw director/heartbeat.py --chars phineas,maxx --quiet --propose-every 8

  every ~4 min (heartbeat.py:70, :615-631):
    tick()                              heartbeat.py:749
      _ensure_player()                  heartbeat.py:733   (schtasks /Run lp-preview — backstop)
      per character: decide_character() heartbeat.py:494
        read pose/<char>.json + journal tail (5 lines, :72)
        GLM call via director/llm.py    heartbeat.py:327
        write intent.json (atomic)      heartbeat.py:571
        append journal/<char>.jsonl     heartbeat.py:222
    every 8th tick: propose_pose()      heartbeat.py:625   -> proposals.json (status=pending)

lp-gen      (time trigger, repeat PT20M)
  pythonw pipeline/autogen.py generate --auto --limit 1

  run_generate()                        autogen.py:828
    single-flight lock                  autogen.py:583
    for each pending/approved proposal:
      generate_one_hf()                 autogen.py:616     (backend "hf", default, :89)
        clip budget check before EVERY clip (:702, :717, :731)
        hf_gen.generate_still / generate_clip -> hf-proxy on `node`
        _record_pose -> autogen_poses.json
        _rebuild_graph -> subprocess `video_graph.py build`  autogen.py:477
```

The walker then hot-reloads the rebuilt graph without a restart (`_preview_graph.py:160-186`).
That is the ONLY coupling between the loops.

**Cadences at a glance**

| Loop | Trigger | Cadence | Cost per cycle |
|---|---|---|---|
| Walker frame | continuous | 10 fps | zero |
| Walker pick | clip end | ~5-6 s | zero |
| Heartbeat tick | timer | 240 s day / 900 s all-asleep | 1 GLM call/character |
| Proposal | every 8th tick | ~32 min | 1 GLM call |
| Autogen run | schtasks | 20 min | 0-6 Higgsfield clips |

---

## 2. Module by module

Legend: **PURE** = stdlib-only, import-safe, no network/GPU/display (the WALKER's contract --
see item 4; it binds the modules the loop imports, not the whole directory). **HEAVY** = network, LLM, GPU or display.

### `runtime/` — deterministic playback layer

| File | Owns | Class |
|---|---|---|
| `__init__.py` | the layer's contract statement | PURE |
| `policy.py` | THE production edge picker: one softmax over anti-reverse / novelty / goal-pull / mood (`:124-186`) | PURE |
| `journal_score.py` | **scored retrieval** over the journal: recency + surprisal-of-the-want + relevance in transition HOPS, under a token budget, with reserved long-term pins (v0.3.0) | PURE |
| `lived.py` | **the write-back**: what a character has actually LIVED — visits / first / last / dwell / mood-band per pose, plays per clip. Sole writer is that panel's walker (v0.4.0) | PURE (writes `data/mind/lived/<char>.json`) |
| `graph_provenance.py` | **bi-temporal**: `created` from artifact mtime, and invalidation as a tombstoning diff so a vanished clip is retired rather than forgotten (v0.4.0) | PURE (writes `data/graph/provenance.json`) |
| `edge_style.py` | **typed edges**: manner + valence derived from each clip's own `motion_prompt`; 1030/1472 real edges typed. Preference layer only (v0.4.0) | PURE |
| `mind.py` | LLM-intent layer; reads `intent.json`, forces a step toward the goal, pins idles at it (`:69-93`) | PURE |
| `circadian.py` | clock layer; bedtime chain, sleep dwell, day-time bedtime-edge exclusion (`:126-159`) | PURE |
| `pathfind.py` | BFS over transition edges: `next_step` / `shortest_path` / `reachable_poses` | PURE |
| `video_graph.py` | **the live graph**: node/edge specs, bedtime + autogen merge, `validate()`, `build()` | PURE (writes `video_graph.json`) |
| `clip_graph.py` | the OLDER v2 clip library (poses/clips + shortest path + manifest). Not parked — `pipeline/` imports it in three places | PURE |
| `clip_player.py` | renders a clip into one panel surface with seam crossfade | HEAVY (opencv/pygame, guarded) |
| `stage_render.py` | composites portrait + bg + asides into a panel | HEAVY (numpy/cv2) |
| `crossframe.py` | walk-out-of-A / walk-into-B figure layer | HEAVY (numpy) |
| `rig.py` / `rig_loop.py` | animates a cutout from its rig spec; adapter to `stage_render`'s (rgb, alpha) | HEAVY (cv2) |
| `behavior_select.py` | personality-weighted idle-behaviour picker (mood reweights) | PURE (duck-typed, `:30-40`) |
| `panels.py` | loads panel geometry/palette from `panels.yaml` | PURE-ish (yaml) |
| `capture_demo.py` | headless GIF/PNG proof-of-motion renderer | HEAVY |

Imported by the live walker: `policy.py`, `mind.py`, `circadian.py`, `pathfind.py`,
`video_graph.py`, plus `lived.py` and `edge_style.py` (v0.4.0). `journal_score.py` and
`graph_provenance.py` are pure too but are read by the BRAIN and the graph builder, not the
render loop. The rest of `runtime/`
belongs to the parked `player.py` rendering path (see §7).

### `director/` — the brain (all network lives here)

| File | Owns | Class |
|---|---|---|
| `heartbeat.py` | the slow brain: sense → GLM → write intent + journal; also `propose_pose` | HEAVY (LLM, schtasks) |
| `llm.py` | GLM client via the IC z.ai gateway; key resolution env → `~/.config` → `data/mind` | HEAVY |
| `context.py` | one weather/time line for the prompt + an energy band for the walk; fail-soft to `""` | HEAVY (fail-soft) |
| `mj_safe.py` | the moderation linter between the LLM and any image generator; hard/soft verdicts | PURE |
| `otel.py` | optional OpenTelemetry spans; no-op unless `LP_OTEL=1` | PURE when off |
| `feeds.py` / `signals.py` | live external material (TTL-cached, fail-soft) + clock/house-mood, for the stage manager | HEAVY / PURE-ish |
| `stage_manager.py` | qwen3 beat writer → `data/stage_state.json` (the parked Track-1 show) | HEAVY |
| `reflect.py` | **nightly reflection** (v0.4.0): in the circadian dwell, reads the day via `journal_score` and writes what the character UNDERSTANDS back into the same journal as `kind: "reflection"` — no `goal` key, so it never pollutes the want histogram and scores at max importance | HEAVY (LLM, fail-soft) |
| `voice_eval.py` | measures per-character voice distinctness/fidelity via z.ai | HEAVY |

`mj_safe.check()` is called on every proposal in BOTH paths (`heartbeat.py:710`,
`autogen.py:307`) — it is the one guard that is not optional.

### `pipeline/` — asset production

| File | Owns | Class | Status |
|---|---|---|---|
| `autogen.py` | the pose-growth worker: proposals, budgets, lock, both backends, graph rebuild | HEAVY | LIVE |
| `hf_gen.py` | the Higgsfield backend via the hf-proxy on `node` (still + first→last clip) | HEAVY | LIVE |
| `prune_proposals.py` | janitor for terminal `failed` rows in `proposals.json` | PURE | LIVE (manual) |
| `verify.py` | QA gate: loopability / continuity / coverage checks before an asset registers | PURE-ish | reference |
| `orchestrate.py` | the producer loop prompt→clip-row for the v2 pipeline | HEAVY | parked |
| `generate.py` / `segment.py` / `rig_spec.py` | SD1.5 portrait → SAM cutout + bg plate → Live2D rig JSON | GPU | parked |
| `tts.py` | spoken asides + viseme track (Piper; falls back to a plan manifest) | HEAVY | parked |
| `_bake*.py` (5 files) | local-GPU img2vid experiments on hil's 2080 Ti | GPU | experiments |
| `_gen_anchor.py`, `_gen_anchor_sdxl.py` | frameless anchor stills (SD1.5 / SDXL+InstantID) | GPU | experiments |
| `_predownload.py`, `_probe_caps.py`, `_probe_lp.py` | one-shot probes | — | throwaway |

### Top-level

| File | Owns | Status |
|---|---|---|
| `_preview_graph.py` | **the production player.** Random/policy walk over the video graph | LIVE |
| `player.py` | the v2 `stage_state.json`-consuming player (task `lp-player`) | **Disabled** |
| `gallery.py` | the cast registry + `add_character()` front door → `gallery.yaml` | LIVE (manual) |
| `graph_viewer.html` | browser view of the graph, served by `lp-graph` on :8011 | LIVE |
| `lp_watchdog_preview.ps1` | scoped self-heal for `lp-preview` + `lp-mind` only | LIVE |
| `start_portraits.ps1` / `stop_portraits.ps1` / `shot.ps1` | desktop one-click up/down (3-phase teardown); panel screenshot | LIVE |
| `health/` | **nine deterministic detectors + `oracle.yaml`** (v0.4.0). `python main.py verify check projects/living-portraits/health/oracle.yaml`; `verify probe` proves they do not flap. Cron `living_portraits_health`, two-hourly | LIVE |
| `scripts/deploy_hil.py` | **the deploy ledger** (v0.4.0): file list from `git ls-files`, ships only diffs, writes `DEPLOYED.json` with the source SHA, commits to a git repo ON the host. `--status` answers what is running and whether anyone hand-edited it | LIVE |
| `scripts/backfill_lived.py` | mines `_preview.log` (76 days, 847k picks) back into the lived record; refuses to write under a live walker | LIVE (one-shot) |
| `scripts/unstick.py` | ranks one-exit poses by measured dwell and buys a second exit for the worst, on Higgsfield, inside the budget rail | LIVE (manual) |
| `scripts/release.py` | version/tag gate + public-subset manifest and sync | LIVE |

### `install/`, `prompts/`, `data/`, `tests/`

| Path | Owns |
|---|---|
| `install/deploy_host.ps1` | host-agnostic runtime bring-up on any Windows tailnet host |
| `install/bringup_host.ps1` / `sc2_bringup.ps1` | generalized vs SC2-specific producer-side bring-up |
| `install/_install_genstack.ps1` | SDXL + InstantID + LivePortrait into hil's `.venv-gen` |
| `install/install_tasks.ps1` | registers `lp-player` / `lp-director` / `lp-watchdog` (the v2 task set) |
| `install/sync_assets.ps1` | copy GPU-baked assets from the GPU host to a no-GPU player host |
| `install/watchdog.ps1` | the ORIGINAL blanket watchdog — deliberately **Disabled** on hil |
| `install/{README,BRINGUP,SYNC,GENSTACK_INSTALL,OTEL}.md` | the matching runbooks |
| `prompts/characters/<slug>.{json,md}` | per-character identity (JSON = spec incl. big_five; MD = persona prose) |
| `prompts/bedtime_routine.json` | the circadian routine spec (single source of truth for night) |
| `prompts/idle_animations.json`, `_stage-directives.md`, `LIVING_PORTRAITS_PROMPT_SPEC.md` | prompt library |
| `data/` | **entirely gitignored.** Runtime state; the local copy is stale, hil is real |
| `tests/` | 17 test modules + `conftest.py` + `run_all.py` (a no-pytest fallback runner) |

---

## 3. Data contracts (single-writer file protocol)

All of `data/` is gitignored (`.gitignore:2`). On hil the real files live at
`C:\living-portraits\data\`.

| File | Writer | Readers | Atomic? | If a second writer appears |
|---|---|---|---|---|
| `data/mind/intent.json` | **heartbeat only** (`heartbeat.py:571` via `_atomic_write` `:118`) | walker (`_preview_graph.py:126-138`), mtime-cached | yes (tmp + `os.replace`) | last-write-wins clobbers the other character's goal — `tick()` deliberately re-reads and merges (`heartbeat.py:546-550`) so it can drive a subset; a second *process* defeats that |
| `data/mind/pose/<char>.json` | **that character's walker only** (`_preview_graph.py:148-156`) | heartbeat (`heartbeat.py:182`) | yes | per-character path is why two panels don't race. One file for both characters would have raced; the split is the fix |
| `data/mind/journal/<char>.jsonl` | heartbeat, append-only — decisions AND (v0.4.0) nightly `kind: "reflection"` lines from `director/reflect.py` | heartbeat via `journal_score.select` (SCORED retrieval since v0.3.0, not the last 5) | **no** — plain append | concurrent appends can interleave a line; readers skip unparsable lines (`:217-218`) so it degrades rather than breaks |
| `data/clips/video_graph.json` | `video_graph.build()` (`video_graph.py:561`, `save()` `:54-61`) — driven by `lp-gen` | walker (hot-reload), heartbeat, pathfind | yes | a torn read is impossible by construction; two concurrent builds are prevented by the autogen lock, not by the file |
| `data/mind/proposals.json` | `autogen.add_proposal` / `set_status` (`:142`, `:157`) — heartbeat proposes, worker transitions | both | yes (`_save` `:130`) | **read-modify-write, not locked.** Heartbeat appending while the worker sets a status can lose one side's edit. The 20-min/4-min cadences make the collision rare, not impossible **(inferred: I found no lock on this file)** |
| `data/mind/clip_budget.json` | `_spend_clip` (`autogen.py:220`) | `clip_budget_left` (`:212`), status report | yes | the lock (`:583`) is what makes the read-modify-write safe — a second unlocked writer would let the daily cap be exceeded |
| `data/mind/autogen_poses.json` | `_record_pose` (`autogen.py:530`) | `video_graph._load_autogen` (`:360`) | yes | same class as proposals: RMW under the autogen lock only |
| `data/mind/gen_events.jsonl` | `_telemetry` (`autogen.py:69`), append-only | `autogen.py status` | no | best-effort by design; never raises into the gen loop (`:78-79`) |
| `data/mind/autogen.lock` | `_acquire_lock` (`autogen.py:583`) | — | O_EXCL create | this IS the mutex. Stale locks are stolen after a dead PID or 3h (`:63`, `:592-606`) |
| `data/mind/lived/<char>.json` | **that character's walker only** (`runtime/lived.py`) — same per-character split as `pose/` | heartbeat (habit line + frontier ground truth), `health/checks.py`, `scripts/unstick.py` | yes | v0.4.0. Flushed at most once a minute from the 10 fps loop. **`flush()` round-trips keys it does not own** — the first version dropped the backfill's provenance block while keeping its numbers |
| `data/graph/provenance.json` | `video_graph.build()` via `graph_provenance` | `VideoGraph.load(history=True)`, `health/checks.py` | yes | v0.4.0. Tombstones only; bounded at `MAX_TOMBSTONES`. NOT written by the runtime loop |
| `data/mind/{zai_key.txt,hf_proxy.json,ic_context_token.txt}` | human | `llm.py`, `hf_gen.py:68-69` | — | secrets; gitignored. **`ic_context_token.txt` silently expired 2026-07-14 and nothing noticed for 27 days** — `context_line()` is fail-soft by contract, which is why `health/checks.py:world_context` now exists |

**Atomicity primitive everywhere:** write `<path>.tmp`, then `os.replace`. Both loops use it
(`heartbeat.py:103-108`, `_preview_graph.py:150-154`, `autogen.py:130-134`, `video_graph.py:57-61`).

**Staleness guard:** a goal older than 30 minutes is ignored (`runtime/mind.py:39`,
`goal_for` `:51-66`). A dead heartbeat therefore releases the portraits back to the lively
walk instead of freezing them at a stale goal.

---

## 4. Entry points

### Scheduled tasks on hil (`desktop-hil08rd`), read live 2026-08-09

| Task | State | Action | Trigger |
|---|---|---|---|
| `lp-preview` | **Running** | `.venv\Scripts\pythonw.exe _preview_graph.py --a phineas --b maxx --loops 1 --a-tp 0.3 --b-pose idle --b-tp 0.4 --mind --max-idle-secs 0 --policy` | AtLogon |
| `lp-mind` | **Running** | `pythonw director\heartbeat.py --chars phineas,maxx --quiet --propose-every 8` | AtLogon |
| `lp-gen` | Ready | `pythonw pipeline\autogen.py generate --auto --limit 1` | time, repeat PT20M |
| `lp-watchdog-preview` | Ready | `powershell -File C:\living-portraits\lp_watchdog_preview.ps1` | time PT3M + AtLogon |
| `lp-graph` | Running | `python -m http.server 8011 --bind 0.0.0.0` | (no trigger; manual/left running) |
| `lp-graphserver` | Ready | `python -m http.server 8000 --bind 0.0.0.0` | AtBoot — last result `4294967295` (failed) |
| `lp-shot` | Ready | `powershell -File C:\living-portraits\shot.ps1` | manual |
| `lp-player` | **Disabled** | `pythonw player.py` | the parked v2 show |
| `lp-director` | **Disabled** | `pythonw director\stage_manager.py --loop 45` | the parked v2 show |
| `lp-watchdog` | **Disabled** | `install\hil_watchdog.ps1` | the blanket watchdog; disabled on purpose |

All tasks are `MultipleInstances=IgnoreNew`, and all live ones carry
`WorkingDirectory=C:\living-portraits`. `lp-shot` and `lp-watchdog` have an EMPTY working
directory (they pass absolute `-File` paths).

Two independent things keep the panels lit: the watchdog task (every 3 min) and
`heartbeat._ensure_player()` on every tick (`heartbeat.py:514-527`). Both respect a
deliberate `Disabled` state, which is how `stop_portraits.ps1` stays stopped.

### Human CLIs

```
# runtime
pythonw _preview_graph.py --a phineas --b maxx --mind --policy   # the show
python  runtime/video_graph.py build | show | validate           # rebuild / inspect / lint

# brain
python director/heartbeat.py --once --dry-run                    # one decision, writes nothing
python director/heartbeat.py --once --chars phineas,maxx         # one real tick
python director/heartbeat.py --propose-once --chars phineas      # seed proposals only
python director/voice_eval.py [--draws N]                        # voice distinctness score

# generation
python pipeline/autogen.py status                                # ONE-SCREEN HEALTH VIEW (:832)
python pipeline/autogen.py list | approve <id> | reject <id>
python pipeline/autogen.py generate [--dry-run] [--limit N] [--gated]
python pipeline/prune_proposals.py                               # drop terminal `failed` rows

# ops (on hil) — .\start_portraits.ps1  /  .\stop_portraits.ps1
# tests        — python -m pytest tests/   (or tests/run_all.py without pytest)
```

`autogen.py status` is the first thing to run when something looks wrong: it prints task
states, last successful generation, 24h outcome counts, top failure reasons, queue counts,
graph size, lock state, both budgets, and cooldown (`autogen.py:832-936`).

---

## 5. The precedence chain

**Stated contract:** `circadian (clock) > mind (LLM intent) > random walk`
(`runtime/mind.py:8-10`, `AUTONOMY.md:22`).

**Where it is actually resolved** — production runs `--policy`, so read the second column.

| | Legacy path (`--mind` only) | **Production path (`--policy`)** |
|---|---|---|
| entry | `_preview_graph.py:221` `_pick()` | `_pick()` short-circuits at `:203-204` → `_pick_policy()` `:271` |
| clock | `circadian.decide(...)` `:205`; `"force"` wins outright `:207-209` | night only: `circadian.is_night` `:244`, `decide` `:245`, `"force"` wins `:248-250` |
| LLM goal | `mind.decide(...)` `:214`; `"force"` wins `:216-218` | folded in as a *weight*: `goal` + `mood` + `route` passed to `policy.choose` `:257-262` |
| walk | random idle/transition with anti-pendulum `:219-236` | one softmax, `runtime/policy.py:200-222` |

**Two DWELL-dependent terms were added in v0.4.0, and they exist because the static weights
trapped a character between two correct guards:**

- **Escape velocity** — the transition multiplier grows with `dwell` past `ESCAPE_AFTER`,
  bounded by `ESCAPE_MAX`. A pose with one exit and three idles is a trap under a low-energy
  band (`fixated` = idles x1.5, transitions x0.4). The walker passes `dwell=0` at the sleep
  pose, because dwelling there for hours IS the behaviour.
- **Anti-reverse forgiveness** (`policy._reverse_penalty`) — `REVERSE_PENALTY` relaxes to
  1.0 by `ESCAPE_AFTER + REVERSE_FORGIVE`. On a **pendant** pose (one neighbour, in and out
  by the same edge) the only exit IS a backtrack, so anti-reverse at a flat 0.04 cut escape
  velocity's 45% straight back to 3.2%. Anti-reverse is a short-timescale guard — the
  pendulum is only ugly when it is immediate.

**Reachability is computed over the edges the body will actually use at this hour**
(`heartbeat.decide_character`). By day the bedtime chain is removed BEFORE the goal menu is
built. Before v0.4.0 the menu used the full edge set while the walk masked bedtime edges, so
the brain could want a pose that could not be reached until nightfall.
| bedtime mask | `circadian` returns `exclude` labels `:220-222` | `circadian.bedtime_labels` passed as `exclude` `:255` |

So in production the clock still has absolute priority at night, and by day the LLM goal is a
**soft gradient** (BFS distance to goal, `policy.py:99-116`, `:166-176`) rather than a forced
step. That change exists because forced steps fought the anti-pendulum guard and deadlocked a
goal-walk routed through a bedtime pose (`policy.py:16-19`).

The heartbeat honours the same precedence from the other side: it writes no goal at all at
night (`heartbeat.py:308-310`) and excludes bedtime poses from the LLM's menu (`:312-313`).

Weights that decide the body, all module constants so tests pin them
(`policy.py:40-46`): `REVERSE_PENALTY 0.04`, `LAST_CLIP_PENALTY 0.02`,
`GOAL_ARRIVE_BOOST 8.0`, `GOAL_HOLD_BOOST 6.0`, `AWAY_PENALTY 0.15`,
`ROUTE_PULL {beeline 7.0, wander 2.0}`. Mood is matched by exact band, then by a free-text
energy bucket, then a neutral default (`policy.py:81-96`).

---

## 6. Test coverage

`python -m pytest tests/` → **343 passed, 1 skipped in ~21 s** (verified locally 2026-08-11;
was 214 on 2026-08-09). Fully offline; `conftest.py` skips rather than errors when
numpy/cv2/yaml are absent.

**Since v0.3.0 the suite includes REAL-DATA tests.** A production snapshot lives at
`data/_realdata/` (gitignored): the live graph, both journals, the lived records. Those tests
`pytest.skip` with an explicit reason when the snapshot is absent — skip, never silently
pass — so a fresh clone still goes green. They exist because three defects in two days were
invisible to fixtures and obvious against production: a 9-day retrieval horizon, a walker
that never counted the pose it boots in, and a prompt line claiming a first visit "0m ago".
A fixture agrees with whatever you believed when you wrote it.

New modules: `test_journal_score.py`, `test_memory_probe.py`, `test_lived.py`,
`test_provenance.py`, `test_edge_style.py`, `test_reflection.py`, `test_escape_velocity.py`,
`test_llm_extract.py`, `test_context_graph_probe.py` (the scorecard — it asserts criterion
(e) FAILS, so a green suite is not a green system).

| Test module | Tests | Covers |
|---|---:|---|
| `test_verify.py` | 26 | `pipeline/verify.py` |
| `test_feeds.py` | 23 | `director/feeds.py` + `director/signals.py` |
| `test_crossframe.py` | 23 | `runtime/crossframe.py` |
| `test_clip_graph.py` | 23 | `runtime/clip_graph.py` |
| `test_gallery.py` | 20 | `gallery.py` |
| `test_autogen.py` | 14 | `pipeline/autogen.py` + `runtime/video_graph.py` merge |
| `test_stage_render.py` | 13 | `runtime/stage_render.py` |
| `test_policy.py` | 13 | `runtime/policy.py` |
| `test_rig_loop.py` | 12 | `runtime/rig_loop.py` (+ `rig.py` via fixtures) |
| `test_heartbeat_goal.py` | 9 | `director/heartbeat._resolve_goal` **only** |
| `test_pathfind.py` | 7 | `runtime/pathfind.py` |
| `test_mj_safe.py` | 7 | `director/mj_safe.py` |
| `test_mind.py` | 7 | `runtime/mind.py` |
| `test_prune_proposals.py` | 6 | `pipeline/prune_proposals.py` |
| `test_clip_budget.py` | 6 | `autogen` clip budget + `hf_gen` constants |
| `test_circadian.py` | 4 | `runtime/circadian.py` |
| `test_otel.py` | 2 | `director/otel.py` no-op behaviour |

**Uncovered, ranked by risk:**

1. **`_preview_graph.py` — 460 lines, zero tests.** The entire production render loop:
   `GraphCycler._pick`, dwell accounting, gif caching, pose publishing, graph hot-reload,
   display-loss recovery. Every behavioural bug the git log records (reverse pendulum,
   frozen walk, dark panels) lived here. The picker logic it delegates to *is* tested;
   the wiring around it is not.
2. **`director/heartbeat.py` — only `_resolve_goal` is tested.** `tick`, `decide_character`,
   `propose_pose`, `_is_duplicate`, `_sibling_candidates`, `_temperament`, `_ensure_player`
   are all untested, and `propose_pose` is what spends money downstream.
3. **`pipeline/hf_gen.py` generation paths** — only constants are touched. The proxy contract
   (file-key extensions, error→`kind` classification `:110-117`) is unverified in tests and
   has already broken once (commit `e96c65a5`).
4. **`runtime/video_graph.py:build()` / `validate()`** — exercised indirectly through
   `test_autogen.py`, but the walk-safety invariants E1-E3/W1-W3 (`:114-157`) that gate every
   graph write have no direct tests.
5. `director/llm.py`, `director/context.py`, `director/stage_manager.py`, `pipeline/orchestrate.py`,
   `segment.py`, `rig_spec.py`, `tts.py`, `generate.py`, `runtime/behavior_select.py`,
   `runtime/panels.py`, `runtime/clip_player.py`, `runtime/capture_demo.py`, `player.py` —
   no tests. Most are on the parked path, which is why this is lower risk than it looks.

---

## 7. Live / reference / dead

**LIVE (touch with care):** `_preview_graph.py`, `runtime/{policy,mind,circadian,pathfind,video_graph}.py`,
`director/{heartbeat,llm,context,mj_safe,otel}.py`, `pipeline/{autogen,hf_gen}.py`,
`lp_watchdog_preview.ps1`, `start_portraits.ps1`, `stop_portraits.ps1`, `gallery.py`.

**RETIRED BUT DELIBERATELY KEPT — the Midjourney path.**
`director/mj_safe.py` (still LIVE — both backends lint through it), `autogen.generate_one`
(`:314-452`), `_reverse_gif` (`:257`), `_client`/`_wait`/`_download`/`_video_to_gif`,
`_generate_extra_links` (`:490`), and the MJ-era constants (`OREF_WEIGHT`, `IMAGINE_VERSION`,
`DAILY_CAP`, `:97-117`). Kept because it is the only reference for the `--oref` identity lock
and the moderation guard, and it is the fallback if the Higgsfield plan goes away
(`autogen.py:86-89`). Selected with `--backend` / `LP_GEN_BACKEND`; default is `hf`.

**The MJ path is broken today, and I confirmed it on hil.** Telemetry
(`data/mind/gen_events.jsonl`) shows **886 failures reading `POST /api/storage-upload-file -> 403`,
first 2026-07-02T02:04:57, last 2026-08-08T20:28:57.** Last MJ-path success was
2026-07-02T00:57:18 — the 403 started within the hour and never stopped. That is the 37-day
outage the Higgsfield cutover fixed: first HF event 2026-08-08T20:44:25, 4 successes since,
and `hf_gen.healthy()` returns `(True, 'ok')` right now.

**LOCAL-GPU EXPERIMENTS (reference, not run):** the five `pipeline/_bake*.py` plus
`_gen_anchor{,_sdxl}.py`. Each documents an honest ceiling on hil's Turing 2080 Ti
(LTX FLF2V ~5 h/clip; AnimateDiff moderate motion magnitude). Nothing live imports them.

**PARKED (the v2 "Track-1" show):** `player.py`, `director/stage_manager.py`,
`director/signals.py`, and the `runtime/` render stack `player.py` actually imports
(`clip_player`, `stage_render`, `crossframe`, `rig_loop` at `player.py:37,47,59`). Their tasks
`lp-player` and `lp-director` are **Disabled**; `lp-director` last ran 2026-05-31. This code is
well-tested and coherent — it is parked, not rotten.

Three names that used to sit in that list do not belong there, each for a different reason:
`pipeline/orchestrate.py` and `clip_graph` are **live** (orchestrate imports clip_graph, and so
do two `_bake_*` modules), while `behavior_select` (604 lines) and `panels` (337) have **zero
importers anywhere in this repo** — `player.py` does not import either, and `panels.py:23`
documents a call site, `from panels import load_panels`, that does not exist. They are neither
parked nor live; nothing runs them at all.

**THROWAWAY:** `_shots/*` (27 one-off render scripts), `pipeline/_probe_*.py`,
`_predownload.py`. (`_preview_cycle.py` and `_preview_panels.py` were also on this list
and were deleted 2026-09-06 — self-declared TEMP, referenced by nothing.)

**Ambiguity resolved — two pairs that have bitten before:**

| Pair | Authoritative | Legacy / stale |
|---|---|---|
| `video_graph.json` vs `video_graph.live.json` | `data/clips/video_graph.json` — the only one on hil (257 nodes / 1462 edges, 2026-08-09) | `video_graph.live.json` is byte-identical to the local `video_graph.json.bak-20260609` and **does not exist on hil**. Dead file. |
| `runtime/video_graph.py` vs `runtime/clip_graph.py` | `video_graph.py` — imported by `_preview_graph.py`, `heartbeat.py`, `autogen.py`, `pathfind.py` | `clip_graph.py` — imported only by `orchestrate.py`, `clip_player.py`, the `_bake_*` scripts. Different schema, different vocabulary (`POSES`, `Clip`, `manifest.json`). |

Local `data/` is also stale (59 nodes / 500 edges vs hil's 257 / 1462). **Never reason about
production state from the local `data/` directory.**

---

## 8. Seams a newcomer trips on

1. **`runtime/` must stay import-safe.** It is imported by the 10 fps render loop
   (`runtime/policy.py:34`, `mind.py:24`). Adding a network call, a heavy import, or
   import-time file I/O to any of `policy` / `mind` / `circadian` / `pathfind` stalls frames.
   The walker even guards its optional `director.context` import (`_preview_graph.py:34-37`)
   so a broken brain module can never stop the panels.
2. **hil is NOT the Higgsfield session owner.** `node` is, and the session uses one-time-use
   refresh tokens, so running `hf` on hil steals and kills the owner's session. All generation
   goes through the hf-proxy (`pipeline/hf_gen.py:10-14`).
3. **The `Set-ScheduledTask -Action` WorkingDirectory wipe trap.** Replacing a task's action
   drops `WorkingDirectory` unless you set it in the same call. Every live task depends on
   `C:\living-portraits` being cwd (relative script paths, relative gif paths in the graph).
   Flagged in `AUTONOMY.md:138` and `:176`.
4. **A clean exit is not a failure.** `RestartOnFailure` never fires when pygame gets a QUIT
   from a fullscreen app, which left the panels dark for hours on 2026-06-06. Two fixes now
   coexist: the walker ignores QUIT entirely (`_preview_graph.py:463-466`) and the watchdog
   checks the *process*, not the task result (`lp_watchdog_preview.ps1`).
5. **`_rebuild_graph` must be a fresh subprocess.** `video_graph`'s NODE_SPECS/EDGE_SPECS are
   mutated at *import* time (`video_graph.py:509-514`), so an in-process `build()` would
   silently omit the pose just recorded (`autogen.py:477-487`).
6. **`build()` refuses to save a broken graph.** Any walk-safety ERROR raises SystemExit and
   the old graph stays (`video_graph.py:604`). A new pose merges only when *every* edge it
   declares already has a gif on disk (`:428`) — half-generated poses are skipped atomically,
   never stranded.
7. **The heartbeat's model names poses, not node ids.** 1170 of 1179 rejected decisions were
   the model answering `withered_rose` instead of `phineas:withered_rose`, and both characters
   sat parked for hours as a result. `_resolve_goal` (`heartbeat.py:409-439`) repairs it —
   don't "simplify" that back to an exact-membership test.
8. **The clip budget is the only thing standing between the loop and the credit plan.**
   Generation is autonomous since 2026-08-09 — `pending` proposals are built with no human
   approval (`autogen.py:764-776`). What replaced review is arithmetic: 13 clips/day system-wide,
   6 per character (`:94-95`), re-checked before **every** clip. Raising either number changes
   the money, not just the pace.
9. **Two writers to `proposals.json` are not locked** (heartbeat proposes while the worker
   transitions statuses). Cadence makes it rare. Treat it as a known sharp edge.
10. **`data/` is gitignored in full.** There is no repo copy of production state, and the local
    one is two months stale. Read hil.
11. **The journal grows without bound.** 13,432 lines for maxx, 11,272 for phineas, 7.1 MB
    combined. No rotation exists. ~~The brain reads the last 5 (`heartbeat.py:72`).~~
    **Corrected 2026-09-06:** that retrieval was replaced in v0.3.0 and the cited line now
    says so — `heartbeat.py:72-74` reads `JOURNAL_TOKENS = 700  # token budget for RETRIEVED
    monologue`, with the comment "Replaced JOURNAL_TAIL = 5, which showed the character ~35
    minutes". The live call is scored retrieval at `heartbeat.py:368`
    (`journal_score.select(...)`), as §2's `journal_score.py` row and the v0.3.0 CHANGELOG
    entry both already said. Unbounded growth is still real; the reading strategy was not.
12. **`proposals.json` is 2.7 MB with 905 `failed` rows** out of 1157 (mostly the MJ 403 era),
    and every queue/status/generate call re-reads all of it. `pipeline/prune_proposals.py`
    exists for exactly this and has not been run.

---

## What I did not verify

- **I did not run the show.** No panel was rendered, no clip generated, no scheduled task
  started or stopped. Claims about the walker's frame-level behaviour come from reading
  `_preview_graph.py` plus tailing `_preview.log` on hil.
- **The MJ 403's cause.** I confirmed the 403s exist, their string, their count and their date
  range from `gen_events.jsonl`. I did not reproduce one live, and I did not determine whether
  it is an expired session, a plan change, or an endpoint change. My own out-of-band probe hit
  a *different* error (`MidjourneyAPIAuthError: missing firebase.api_key or firebase.refresh_token`)
  because it did not resolve `sessions/midjourney.json` the way the real run does — so treat
  that probe as noise, and the telemetry as the fact.
- **The `proposals.json` race is inferred**, not observed. I found no lock around
  `add_proposal` / `set_status` and no interleaving in the data; I did not attempt to trigger one.
- **`director/` module behaviour under failure** (`llm.py`, `context.py`, `feeds.py`) is read
  from source and docstrings. I did not exercise their fail-soft paths.
- **`lp-graphserver`'s last result `4294967295`** — a failure, on a task nothing appears to
  depend on. I did not investigate; `lp-graph` on :8011 is the one that is up.
- **The parked v2 stack** (`player.py`, `stage_render`, `rig`, `crossframe`, `clip_graph`,
  `orchestrate`) I mapped from docstrings, imports and test files only. It has good test
  coverage; I did not read it line by line, and I did not confirm it still runs.
- **`install/hil_watchdog.ps1`** is referenced by the disabled `lp-watchdog` task but is not in
  the repo — I did not check whether it exists on hil.
- **Line numbers are from the repo working tree at 2026-08-09.** hil runs a deployed copy; I
  did not diff the two, so a hil-side hotfix would not be reflected here.
