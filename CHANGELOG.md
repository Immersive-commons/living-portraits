# Changelog

Living Portraits. Versions are marked in the life monorepo as
`living-portraits/vX.Y.Z` (the prefix matters — the repo holds many projects).

Pre-1.0: the shape of the system is still moving. A minor bump means a capability
changed; a patch means a fix. `scripts/release.py` enforces that a version cannot be
tagged unless it appears here.

---

## Unreleased

**The context graph is now something you can look at.** Every layer 0.3.x and 0.4.0 added lives
in a file the viewer never opened, so the only picture of this system was still the May one: a
pose graph and a clip list.

### Added
- **`scripts/export_context_view.py`** — joins the six single-writer stores and the two
  derivations that only existed in Python into one `data/graph/context_view.json` (≈1 MB, 0.5s):
  the graph, `graph_provenance.stamp`'s transaction time, the lived record's valid time,
  `edge_style.index`'s manner and valence, the journal with `journal_score.select` run for the
  pose each character is standing in *right now*, and the live `policy.weigh` surface over its
  exits. Read-only over production by contract — it never calls `build()`, never writes back into
  `video_graph.json`, and never touches a journal.
- **`graph_viewer.html` rebuilt as six lenses over one graph** — Structure (the old view),
  **Lived** (visits as heat, dwell as size, clips that have actually rolled vs merely exist),
  **Provenance** (age gradient plus a scrubber that replays the graph accreting, node by node,
  from 2026-05-26 to now), **Manner** (valence-coloured typed edges, untyped ones dotted and
  counted), **Frontier** (reachable poses never once entered, with their hop distance), and
  **Decision** (the live weight on every exit, the goal ring, the route between them). Plus a
  Memory tab showing what each character is remembering this minute and why it scored, a
  Codebase-map tab that checks every row against disk, and a tab for what the artifact does not
  know. The now-strip re-reads the walker's pose file every eight seconds.

### Fixed
- **The viewer would have drawn a softmax over a sleeping character.** `_pick_policy` asks
  circadian first and returns on a force, so at night the bedtime chain owns the body and the
  policy weights are not consulted at all. The export now carries `decision.consulted` and the
  artifact says so instead of illustrating a decision nothing is making.
- A CSS rule meant for the weight bar's fill also matched its label spans, stacking the label and
  the percentage at the same x. Caught by probing the two spans' bounding boxes rather than by
  looking at a screenshot, which is the only way that class of bug is falsifiable.

### Added — a clone you can actually run

Bringing a collaborator on found that a fresh checkout was a dead end. Everything below is that
gap, closed and then verified by cloning into a scratch directory and running it as a stranger.

- **`requirements.txt`** — there was no dependency manifest of any kind, so the six libraries
  this needs were knowable only by reading imports. Split from **`requirements-gen.txt`**, the
  CUDA clip-generation stack, which nothing outside `pipeline/` imports and nobody needs to
  develop here.
- **`scripts/seed_demo_media.py`** — `build` discovers edges by globbing clips on disk, so on a
  bare clone it found none, every pose had no way out, and the build correctly refused to save
  21 walk-safety errors. That is a wall: no graph, no walk, no lived record, nothing for the
  viewer to open. The seeder writes placeholder stills and procedurally-animated loops for every
  still and clip the graph declares, which turns a bare clone into **20 nodes, 98 edges,
  walk-safe** — the full bedtime chain included, because it reads `bedtime_routine.json` directly
  rather than waiting for edges that only appear once their clips exist. It never overwrites a
  file that already exists, so running it on the production host is a no-op.
- **`AGENTS.md`** — how to point a coding agent at this repo, built around the five invariants no
  test can catch: single-writer files, `runtime/` stays pure stdlib, a walk-safety refusal is
  correct behaviour, absence is a skip with a reason, detection never repairs.
- **`CONTRIBUTING.md`**, and a **README** that had described the May state — the wrong host, and
  a status list with the generative pipeline still unchecked, three months after it shipped.

### Fixed — while verifying the above
- **The viewer's own intro contradicted its header.** "the same 266 poses and 1,558 clips" was
  hardcoded prose; on any dataset but the production one it stated a number the chips directly
  above it disproved. It now reads the loaded view. Found by rendering the artifact against a
  seeded clone, which is the only configuration where a hardcoded production count is visibly
  wrong — and the first repair for it threw, because `V.nodes` is an object and only `V.edges`
  is an array.
- Documented that **`health/` is an operator surface, not a contributor one**: `checks.py` SSHes
  to the production host rather than inspecting your checkout, and `oracle.yaml` hardcodes that
  host's interpreter path. Both are fine; neither was written down, and both look like something
  a new contributor should be able to run.

### Corrected — four claims that were not true

An outside review of this repo checked the docs against the code rather than against each other.
These are what it found. Each one had a citation, and the citation is what made it findable.

- **The documented test count was off by one, and in the direction that stops work.** README,
  AGENTS and CONTRIBUTING all said `304 passed, 40 skipped on a bare clone`. That figure needs
  `opentelemetry-sdk`, which is deliberately not in `requirements.txt` — `director/otel.py` is
  fail-open and the system does not need it. A clone following the documented setup gets
  **303 / 41**. This mattered more than one test: AGENTS.md tells an agent that a changed count
  means a broken baseline and to *stop and say so before beginning the task*, so the wrong number
  was an instruction to halt on a healthy checkout. Now stated as 303 / 41, with the
  `opentelemetry-sdk` case named in README.
- **"`runtime/` is pure stdlib" was true of the walker, not the directory.** Stated in that
  over-broad form in README, AGENTS rule 2, CONTRIBUTING, `requirements.txt`'s own header, and
  ARCHITECTURE's "If you read nothing else" — while ARCHITECTURE's module table 110 lines later
  correctly marked 6 of 19 files HEAVY. `runtime/rig.py:64` imports cv2 and numpy at module
  level; `clip_player.py:44`, `stage_render.py:50` and `crossframe.py:111` import numpy. The
  real invariant is narrower and still absolute: the seven modules the 10 fps loop imports —
  `policy`, `mind`, `circadian`, `pathfind`, `video_graph`, `lived`, `edge_style` — are pure.
  All five statements now say that, and point at the table as the authority.
- **ARCHITECTURE said the brain reads the last 5 journal lines, citing the line that retracts
  it.** `ARCHITECTURE.md:498` cited `heartbeat.py:72`, which reads `JOURNAL_TOKENS = 700  #
  token budget for RETRIEVED monologue` and carries the comment "Replaced JOURNAL_TAIL = 5".
  Scored retrieval shipped in 0.3.0 and this entry's own §2 table already said so. Unbounded
  journal growth is still real; the reading strategy was not.
- **The stated blocker on v0.5.0 no longer exists.** ROADMAP said `graph_viewer.html:127`
  renders every node as a full-size PNG for a ~355 MB cold load. The viewer was rewritten in
  `dcd6fc3` and now draws nodes as `shape:"dot", size:12` (`graph_viewer.html:283`); the only
  two `<img>` uses are lazy, in the detail pane. The cost model was not updated with the
  rewrite. Also corrected there: `_preview_graph.py` is 506 lines, not 460.

Two more were found and are **not** fixed here, because both are behaviour rather than prose:
`director/voice_eval.py:47` still pins `JUDGE_MODEL = "glm-4.5-air"`, which `research/FACTS.md:60`
declares stale wherever it appears — but changing a live judge model is a behavioural change, not
a doc fix. And `AUTONOMY.md`, cited for the single-writer invariant in three places, lives in the
private tree and is not published here; ARCHITECTURE now says so rather than dangling.

### Fixed — the suite could stop running entirely without failing

- **`tests/test_verify.py` aborted collection for the whole session on any box where cv2 is
  unimportable**, which is the normal state of a headless container: `opencv-python` imports fine
  except that `libGL.so.1` is absent. Result was `Interrupted: 1 error during collection` and
  **zero tests run** — presenting as one broken file rather than 344 tests not running. The guard
  was a module-level `pytestmark`, which gates already-collected tests and does not stop the
  module body, so the `import verify` two lines below still executed. (`pytest.importorskip` does
  not fix it either: it defaults to catching `ModuleNotFoundError`, and this is a plain
  `ImportError`.) Now uses conftest's `HAVE_*` probes — which saw it correctly all along — with
  `pytest.skip(allow_module_level=True)`. Absence is a skip with a reason, never an error.

### Added — the four commands are now checked by a machine

- **`.github/workflows/ci.yaml`.** AGENTS.md's verification chain, run on every push and PR, with
  the assertion it makes in prose finally made mechanically: both test counts are floors, so a
  test that flips from pass to skip fails the build instead of hiding inside a green tick. A
  second job installs deliberately *without* opencv and requires the suite to still run and still
  skip cleanly — a permanent guard on the bug above. It also asserts all 23 codemap rows still
  resolve on disk, which nothing checked, and renders the dual panel headless. It runs on Linux
  on purpose: a green run means "you do not need our art, our GPU, or our hardware" is true on a
  box nobody involved has ever touched.
- **`requirements.lock`** (`uv pip compile --universal`), so CI installs the exact resolution the
  Windows host would get, `win32` markers included. `requirements.txt` stays the contract.
- **`.pylintrc`**, encoding what this project decided on purpose — the fail-open contract, the
  import-degradation design, the sys.path wiring — each with its reason written next to it, and
  with the genuine findings deliberately left switched on. 7.75 → 9.71.

---

## 0.4.0 — 2026-08-11

**The characters write back what they live, and the installation can be asked how it is.**

0.3.0 gave them retrieval. This gives them a *record* — and gives the system a way to say
"I am fine" that does not depend on a human looking at a wall.

### Added — memory that accretes from living
- **`runtime/lived.py`** — each walker is now the sole writer of its character's record:
  visits, when a pose was first and last stood in, dwell, which mood-band it brought, and
  which clips have *actually rolled* as opposed to merely existing. In-memory counters on
  the 10 fps path, one atomic flush a minute, every failure swallowed — a statistics file
  is never worth a dark panel. This makes the audit's sharpest line ("the graph accretes
  from a credit card, not from experience") false for the first time.
  It cannot live in the graph file: `video_graph.build()` regenerates that from specs every
  ~20 minutes and would erase it.
- **`scripts/backfill_lived.py`** — the walker has printed every pick since 2026-05-26, so
  76 days of traversal were recovered from `_preview.log`: **847,659 picks, 434 restarts.**
  Visits and dwell come back exactly (a node change is an arrival, a repeat is an idle
  unit). Clip plays come back at **label level only** — 83 labels have 2-4 rendered variants
  sharing a name, so attributing a play to `v2` over `v0` would be invention. **Timestamps do
  not come back at all**: the log carries hour-of-day and never a date, so backfilled poses
  carry no first/last rather than a plausible-looking one.
- **`runtime/graph_provenance.py`** — bi-temporal provenance. Every node and edge carries
  when the *system* learned it exists, derived from artifact mtime so it survives a rebuild.
  **259/259 nodes and 1472/1472 edges on the production host.** Invalidation is a diff
  against the last committed graph rather than a flag, so a vanished clip is tombstoned with
  its whole record; the live JSON stays a pure live view, making "the walker never plays a
  dead clip" true by construction.
- **`director/reflect.py`** — nightly reflection in the circadian dwell, where the heartbeat
  is idle anyway. Each character reads the day it just lived and says what it *understands*.
  Idempotent from the journal itself, so a restart mid-window is a no-op. Reflections land
  in the same JSONL with **no `goal` key**, which is the whole integration: they never
  pollute the want histogram, they score at maximum importance, and 0.3.0's retrieval picks
  them up for free.

### Added — the body reads the graph better
- **`runtime/edge_style.py`** — typed edges, finally true. The entire edge type system was
  one bit (`idle|transition`). A manner and valence are now derived from each clip's own
  `motion_prompt`: **1,030 of 1,472 real edges typed, 90.3% of transitions**, across ten
  manners. 442 stay untyped and are *reported* untyped rather than quietly called neutral.
  It also caught what nobody had noticed: **316 edges are reverses carrying their twin's
  prose**, and for 121 of them inverting the manner changes the answer.
  Preference only — bounded above zero, applied after anti-reverse/novelty/goal.
- **Escape velocity** (`policy.ESCAPE_*`) — exits get more attractive the longer a character
  has honestly been somewhere. A pose with one exit and three idles is a trap under a
  low-energy band; measured on the real incident, 12 expected clips before leaving became 3.
  `dwell=0` at the sleep pose, so nobody is pulled out of bed at 2am.
- **Anti-reverse is forgiven with dwell** (`policy._reverse_penalty`). The subtler half:
  on a **pendant** pose — one neighbour, reached and left by the same edge — the only exit
  *is* a backtrack, so escape velocity lifted it to 45% and anti-reverse cut it straight back
  to 3.2%. Two correct guards, trapping a character between them. Anti-reverse is a
  short-timescale guard by nature and now relaxes to none by 28 idle units. The pendulum
  stays fixed; the cell opens.

### Fixed
- **The brain was choosing goals on a graph the body does not walk.** The goal menu was
  built over *all* edges while the daytime walk masks the bedtime chain, so a goal routed
  through it could never arrive — the gradient pulled Phineas toward the bedroom at 4pm for
  an hour. It also explains a pose that went unvisited for its entire 76-day life:
  unreachable by day, and by night he is asleep.
- **The Higgsfield worker was discarding every link it was given.** `generate_one_hf` had no
  reference to `extra_links`; the meshing step was never ported from Midjourney. So
  `MAX_EXTRA_LINKS` was dead config and **982 of 1,163 proposals carried authored link
  motions that were never generated** — the star topology was not legacy debt being outgrown,
  it was being manufactured daily.
- **`IDLE_COUNT` 3 → 2**, to buy that link. The arithmetic is forced: 2 transitions +
  IDLE_COUNT idles + 2 link clips against `CLIP_CHAR_CAP=6`. At 3 idles the link is deferred
  every day forever, which is exactly how the star survived the knob written to prevent it.
  A third way to stand still is worth less than a second way out.
- **The walker was erasing what it did not write.** `lived.flush()` serialised only the keys
  it knew, so the first flush after the backfill kept the numbers and deleted the provenance
  block. Foreign keys now round-trip.
- **`prompts/characters/maxx.json` was absent from production for months** — one of two live
  characters running with no archetype, traits or big-five reaching the model at all. His
  voice had been surviving on his own journal feeding back. (Also stale on the box:
  `gallery.yaml`, `bedtime_routine.json`.)
- **The world seam had been dead for 27 days.** The IC context token went 401 on 2026-07-14
  and `context_line()` is fail-soft by contract, so every tick looked healthy while the
  characters were told nothing about the weather, the hour, or the building they hang in.

### Added — the installation can be asked how it is
- **`health/`** — eight deterministic detectors, one per incident a human had to notice.
  Runs on the repo's existing oracle contract, so `verify check` gates them and **`verify
  probe` proves them** — a health check that flaps is a pager that teaches you to ignore it.
  Registered as the two-hourly `living_portraits_health` cron.
  A detector never repairs; cannot-see is FAIL, never PASS; every verdict cites its numbers.
  **Off-by-choice is a PASS** — `stop_portraits.ps1` disables the tasks, and calling a
  human's deliberate decision a failure is how a monitor gets ignored.
- **`scripts/deploy_hil.py`** — deploys take their file list from `git ls-files`, ship only
  what differs, write the source SHA into `DEPLOYED.json` on the host, and commit to a git
  repo **on the host**. `--status` answers "what is running" and "did anyone hand-edit
  production" — the questions nobody could answer before.
- **`scripts/unstick.py`** — ranks one-exit poses by measured dwell and buys a second exit
  for the worst, on Higgsfield, inside the budget rail.

### Known / not done
- Criterion (e) of the context-graph scorecard still **FAILS**: zero stored facts about any
  event, person or room, while 5.3% of the monologue is about exactly those. `/api/context`
  is restored but it is weather and time — the world as *conditions*, cached and gone next
  tick, never journaled or scored. Declarative memory is still the honest gap.
- 47% of Phineas's poses still have one exit or none. The generator no longer manufactures
  them; the existing ones are a `unstick.py` job, metered.
- The cron escalation path blocks on an interactive Telegram prompt, so a failing detector
  cannot currently page anyone. Detection works; the alarm does not ring.

---

## 0.3.1 — 2026-08-10

**Tell the truth in public.** Two claims in the public record were false, one was stale, and
0.3.0 had just made the honest version of the first one *stronger* than the original.

### Fixed
- **`research/FACTS.md` [LP-CAST] was false.** It claimed *"Relationships are graph context,
  not just decoration"*; production has **zero cross-character edges** and the rivalry lived
  in `gallery.yaml` prose. Replaced with what is now true and is the better claim:
  **relationships are reconstructed at read time, per character, from what each one can
  actually see.** Generative Agents does not store relationship edges either. The correction
  is recorded in the file rather than quietly swapped.
- **[LP-HEARTBEAT] was false as phrased** — *"THE GRAPH IS THE MEMORY the loop reads and
  writes."* The loop reads the graph and writes two files that are not in it; the graph is
  read-only at runtime. Now: **the graph is the map the loop reads, the journal is the memory
  it writes** — and since 0.3.0 it reads that memory by score.
- **`AUTONOMY.md` described a code path production no longer takes.** Its precedence diagram
  (`circadian > mind > random walk`, where the mind *forces* each step) documents `--mind`;
  the live player runs `--policy`, where the goal is a **soft gradient** inside one softmax
  with novelty, anti-reverse and mood. Both paths are now documented, and which one is live
  is stated.
- **`glm-4.5-air` → `glm-5.1`** across FACTS.md, AUTONOMY.md, the deck chunk and speaker
  notes. It changed in July and nothing said so.
- **Deck slide 05 rebuilt** from its chunk (`deck/_build.py`, 12 slides) so the talk no longer
  carries the retracted line. The speaker note now says what to answer if someone asks why the
  wording changed. **Not deployed** — that stays a human step.

### Fixed — the parse failures that were costing live ticks
62 replies over 14,748 ticks failed as *"no JSON object in model reply"*, each one a turn where
the character stands still. The message named the least likely cause, and the log's own 300-char
truncation destroyed the evidence needed to tell the causes apart.

- **`_extract_json` now walks balanced objects left to right and takes the first that parses.**
  The old code spanned the first `{` to the **last** `}`, which is one object only if the model
  emits exactly one — and glm-5.1 has been observed emitting the same answer four times in a
  row, whereupon the span covers all four and parses as nothing. Quote- and escape-aware, so a
  brace inside a string never opens an object. Retries with `strict=False` for the raw control
  characters a model writing dialogue produces.
- **The reply now carries why the model stopped** (`llm._Reply`, a `str` subclass — no caller
  changes). A reply cut off at `max_tokens` is an unfinished sentence, not malformed JSON, and
  the error now says which one it was, with `repr()` of 600 chars so a stray control character
  or mojibake is visible instead of invisible.
- **Deliberately not "fixed":** the remaining truncated replies. A cut-off object must not be
  salvaged into a half-truth — that would mean inventing the part the model never said. The new
  message makes the next occurrence diagnosable instead.
- 8 tests (`tests/test_llm_extract.py`), every shape taken from the real log.

### Still open
The public repo has **29 files synced into its working tree awaiting review and commit there** —
including the whole 0.3.0 memory layer. Until that lands, the published code still documents a
Midjourney path that has returned 403 on every call since 2026-07-02. Push stays human, by design.

---

## 0.3.0 — 2026-08-10

**It remembers.** The characters had accumulated 25,373 journal entries over 66 days and
read the last five of them (`JOURNAL_TAIL = 5`) — about 35 minutes of remembered life,
0.04% of it. Phineas had wanted `jealous_glare` 2,147 times, 19% of every decision he has
ever made, with no mechanism that could notice. Diagnosed in
[`_audit/CONTEXT_GRAPH_AUDIT.md`](_audit/CONTEXT_GRAPH_AUDIT.md), method chosen against
[`_research/CONTEXT_GRAPHS_FINDINGS.md`](_research/CONTEXT_GRAPHS_FINDINGS.md).

### Added
- **Scored retrieval over the whole journal** (`runtime/journal_score.py`). Recency decay
  + importance as surprisal of the want (`-log₂ p(goal)`) + relevance in **transition
  hops**, all weights 1.0 ([Generative Agents, 2304.03442](https://arxiv.org/abs/2304.03442)).
  Hop-distance relevance is the part a vector store cannot do: the graph already knows how
  far a memory happened from where the character is standing. Selection is bounded by a
  **token budget**, not an entry count, so the prompt stays flat as the journal grows
  forever. The storage layer is unchanged — the JSONL keeps everything, verbatim, and only
  the reader got smarter ([2603.02473](https://arxiv.org/abs/2603.02473): retrieval method
  spans ~20 points of accuracy, write strategy 3–8).
- **Reserved long-term pins.** Found by running the new retrieval against the real
  journals rather than the fixture: pure decay retrieved nothing older than **9 days** from
  a 66-day life. Two slots are now reserved for distant memories that still stand out.
  Retrieval now reaches back **weeks — a median of 18–21 days, and past 50 days from some
  poses** (measured by replaying 20 real decisions per character in
  `tests/test_context_graph_probe.py`), against `tail -5`'s 20 minutes.
  **Corrected 2026-08-10:** this first read "Phineas now reaches back 51 days." That was
  true of the sample taken and not a property of the system — reach is *pose-dependent*,
  because hop-relevance pulls in memories made near where the body currently stands, and
  how old those are depends on where "here" is. Measuring both characters across many
  poses gives the range above; the single best number was the least honest one.
- **An aggregate line** — *"you have wanted phineas:jealous_glare 2,147 times (19% of
  everything you have ever chosen)"*. The shape of a life, which no window of individual
  memories can carry.
- **The neighbour line.** Each character's prompt now carries what it can actually see of
  the others: their current pose and the mood behind their last decision, read from files
  that already existed and were already single-writer. Phineas's dominant behaviour was
  jealousy of a neighbour whose state he could not perceive — a constant, not a response.
  Nothing is stored: this is a cross-character edge reconstructed per reader, which is how
  Generative Agents handles relationships too. A dark panel goes *unseen* rather than being
  reported frozen in place.
- **Frontier.** `reachable_poses(here) − visited_poses` in one line — the poses this
  character has never once been in ([3D-Mem, 2411.17735](https://arxiv.org/abs/2411.17735)).
  Turns the graph's least flattering statistic, its long tail of near-dead-end poses, into
  content: *there is a version of me I have never been, six steps away.*
- **`pathfind.hops_from`** — one BFS per tick, serving both the relevance term and the
  frontier.
- **The memory probe** (`tests/test_memory_probe.py`) — ten questions, five answerable from
  the journal and five not, where *"I don't remember"* counts as correct
  ([REMem, 2602.13530](https://arxiv.org/abs/2602.13530)). It asserts the retrieval layer,
  not an LLM's wording, so it is deterministic, offline, and free. It carries its own
  control: the old `lines[-5:]` reader is scored by the same questions and must fail. Before
  this file, no memory change in this project could be shown to have helped.

### Fixed
- **The mood no longer dies at the boundary.** `policy._mood_bias` guessed a band from the
  free-text mood by keyword and fell through to the default for **59% of MAXX's moods and
  88% of Phineas's** — authored, journaled, and discarded. The brain that writes the mood
  now maps it once (`band`, one of policy's eight), carried in `intent.json` and the
  journal. The mood stays free text; flattening it to eight words would flatten the
  character. With no band present, behaviour is byte-identical to before.

### Verified on `hil`, against production data
- 231 tests green, including the probe. Retrieval over 25k entries: **0.4–0.6s**, inside a
  240s tick.
- Replay of the last 20 decisions per character: the prompt would have differed on
  **20/20**. Retrieved 11–12 memories per tick, **all** of them outside `tail -5`'s reach.
- The first live tick on the new code, unprompted: *"Two thousand and forty-seven times I
  have glared at that luminous upstart — enough! ... the true tragedian does not seethe; he
  turns his BACK upon the rabble"* → goal `phineas:demanding_silence`, a pose he had **never
  once been in**, band `fixated`. Rut noticed, frontier taken, band plumbed, in one decision.
- Panels confirmed painting after restart (`_shots/v030_panels.png`).

### Not in this release
`FACTS.md [LP-CAST]` is still publicly false and the published repo still documents the dead
Midjourney path — both are v0.3.1, one edit and a human `git push`. Reflection nodes remain
deferred.

---

## 0.2.0 — 2026-08-09

**The portraits now grow their own gallery, unattended, on a budget.**

### Changed
- **Generation moved from Midjourney to Higgsfield** (`pipeline/hf_gen.py`).
  Midjourney could only animate forward from a single still and hope it landed near
  the target pose — which is why `pipeline/verify.py` carries a `continuity()` check
  at all. Higgsfield takes a first *and* last frame, so a graph edge is generated as
  the interpolation it actually is and landing on the target node is structural.
  Selected by measuring 8 models on 3 real graph edges (`_bakeoff/`).
- **The reverse edge is generated, not flipped.** The old path faked the return by
  playing the forward clip backwards, which runs the physics backwards: cloth settles
  upward, flame flickers in reverse, a figure rising from a chair reads as being
  pulled into it. Both directions are now real renders.
- **Generation is autonomous.** `pending` proposals build with no human approval
  step; the characters decide what they want and go get it. `--gated` restores review.
- **Sound is off** on every clip. The portraits speak through their own piper TTS, so
  a baked audio track was 2.5 credits an edge of dead weight.

### Added
- **A clip budget** (`CLIP_DAILY_CAP=13`, `CLIP_CHAR_CAP=6`) — the safety rail that
  replaces human review. Derived, not chosen: 3000 credits/month (granted the 23rd,
  not rolled over) at 7.5 credits per sound-off Kling clip is ~400/month, so 13/day
  spends the allowance evenly rather than exhausting it mid-month. Re-checked before
  *every* clip, and the resume logic skips artifacts already on disk, so hitting the
  cap mid-pose costs nothing.
- `scripts/release.py` — this pipeline. The public subset was previously implicit;
  it is now a rule set that `check` proves against the published repo.
- `VERSION`, `CHANGELOG.md`.
- `tests/test_clip_budget.py` — 6 tests over the cap edges and the day rollover.

### Fixed
- **The Midjourney path had been failing silently for weeks**
  (`MidjourneyAPIAuthError: POST /api/storage-upload-file -> 403`), which is the 905
  failed proposals in the queue. Retiring it fixed a pipeline that had stopped
  producing anything.
- Inlined-file keys now carry extensions — the render proxy derives its temp filename
  from the key's suffix, and bare keys produced a `.bin` the API refused to type.

### Known gaps
- `FACTS.md [LP-CAST]` claims relationships are graph context. Production has **zero
  cross-character edges**, so that is currently false (`_audit/`).
- The journal is read 5 entries at a time (`JOURNAL_TAIL = 5`) against 24,598 entries
  over 64 days.
- Midjourney code is retained as reference and fallback, but its session is dead.

---

## 0.1.0 — 2026-07-16

Initial public release — a curated subset published to
`github.com/Immersive-commons/living-portraits`. The runtime walker, the director /
heartbeat brain, the Midjourney generation pipeline, install scripts and tests.

Versioned retroactively: the release predates this changelog, and is recorded here so
`0.2.0` has something to be a successor to.
