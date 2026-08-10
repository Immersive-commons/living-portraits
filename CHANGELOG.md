# Changelog

Living Portraits. Versions are marked in the life monorepo as
`living-portraits/vX.Y.Z` (the prefix matters — the repo holds many projects).

Pre-1.0: the shape of the system is still moving. A minor bump means a capability
changed; a patch means a fix. `scripts/release.py` enforces that a version cannot be
tagged unless it appears here.

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
  Phineas now reaches back **51 days**.
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
