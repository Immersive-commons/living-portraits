# Living Portraits — roadmap

Written 2026-08-09 at `v0.2.0`. Built from three grounded inputs, not from intuition:

- [`_audit/CONTEXT_GRAPH_AUDIT.md (internal)`](_audit/CONTEXT_GRAPH_AUDIT.md (internal)) — what the system actually is
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — what is load-bearing and what is untested
- [`_research/CONTEXT_GRAPHS_FINDINGS.md`](_research/CONTEXT_GRAPHS_FINDINGS.md) — what the
  literature says about the audit's conclusions (via `life/research/`, 682 papers → 120 → 49 cards)

Milestones are versions (`scripts/release.py`). Effort is engineer-hours, honest.

---

## Where we are — v0.4.0, shipped and running

Two characters on `hil`, 262 nodes / 1532 edges, ~90 credits/day against a 3000/month
allowance that does not roll over. Retrieval is scored across a 25,000-entry journal, the
graph carries a time dimension and typed edges, and each character keeps a record of 76 days
of where it has actually stood — 847,659 picks, recovered from the walker's own pick-log.
Nine deterministic detectors say whether the installation is healthy, two-hourly, without
anyone looking at a wall.

**What is true and unflattering:** the world half does not exist. Zero stored facts about any
event, person or room, while 5.3% of what these characters say is about exactly those things.
`/api/context` gives them weather and time — the world as conditions, cached and gone next
tick, never journaled and never scored. And 47% of Phineas's poses still have one exit or
none; the generator no longer manufactures them, but the existing ones are a metered job.

**The line for a talk, now with numbers behind it:** a true context graph in its retrieval
half, a procedural asset manifest in its representation half, and blind to the world in its
content half.

---

## v0.3.0 — "It remembers" ✅ SHIPPED 2026-08-10

All five items landed, verified on `hil` against the real journals (see `CHANGELOG.md`).
One thing the plan did not anticipate, found only by running it on production data: pure
recency decay retrieved nothing older than **9 days** from a 66-day life, so the milestone
would have shipped a system that remembers a week and calls it a past. Two reserved
long-term pins fixed it; retrieval now reaches back a median of 18-21 days and past 50 from
some poses (reach is pose-dependent — hop-relevance pulls in memories made near where the body
currently stands, so a single number flatters whichever pose it was sampled at). The fixture could not have
surfaced that — only the real journal could.

The table below is the plan as written, kept as the record of what was estimated.



**Thesis:** every fix below makes the portraits behave as though they have a past and a
neighbour. None of them touches the render loop, none needs a database, and the whole
milestone is roughly one working day.

The research is unambiguous that this is where the points are: retrieval method accounts for
~20 points of accuracy span versus 3-8 for write strategy
([2603.02473](https://arxiv.org/abs/2603.02473)), and raw-storage-plus-good-retrieval beats
lossy extraction. We are currently at the worst point on that curve — perfect storage, no
retrieval.

| # | Work item | File | Effort | Why now |
|---|---|---|---|---|
| 1 | **Neighbour line.** Each character's prompt gains one line built from the *other's* `pose/<char>.json` + last journal entry. Asymmetric and per-character on purpose — Phineas's read of her need not match hers of him. | `director/heartbeat.py:239` | **1h** | Turns the jealousy driving 18.3% of his decisions from a constant into a contingency. Generative Agents reconstructs relations at retrieval too — this is the canonical design, not a shortcut. |
| 2 | **Scored retrieval.** Replace `JOURNAL_TAIL = 5` with recency decay + `-log2 p(goal)` importance + BFS-hop relevance, top-k by token budget, plus one aggregate line (*"you have chosen jealous_glare 2,060 times in 64 days"*). | new `runtime/journal_score.py`, `heartbeat.py:72,207` | **4h** | The single change that makes it remember. Park's recency/importance/relevance is a 2023 floor; the BFS-hop relevance term is the 2026 delta and costs ~2 extra hours. |
| 3 | **Frontier.** `reachable_poses(current) − visited_poses` → one prompt line. | `heartbeat.py`, `runtime/pathfind.py` | **30m** | The capability nobody considered ([3D-Mem, 2411.17735](https://arxiv.org/abs/2411.17735)). Converts the audit's most damning statistic — 109 of 255 nodes with ≤1 outgoing transition, many never visited — from a defect into content: *"there is a version of me I have never been, two steps away."* |
| 4 | **Mood-band plumbing.** Write `band` into `intent.json` so `policy._mood_bias` reads a real field. | `heartbeat.py`, `runtime/policy.py:81` | **2h** | 88% of authored interior state never reaches the body. This is a memory-fidelity fix, not a cosmetic one. |
| 5 | **The eval probe.** Ten questions, five answerable from the journal and five not; pass = correct recall AND correct *"I don't remember."* | new `tests/test_memory_probe.py` | **2h** | Today there is **no way to tell whether a memory change helped.** Ship this with 1-4 or they are unfalsifiable. |

**Definition of done**
- `python -m pytest tests/` green, including the new probe.
- A journal replay shows the same prompt would have differed on ≥1 of the last 20 decisions.
- Latency: the added scoring stays inside the heartbeat's ~7-minute budget, measured, not assumed
  ([LiCoMemory, 2511.01448](https://arxiv.org/abs/2511.01448) — update latency is a real axis).
- `CHANGELOG.md` 0.3.0 section written before the tag (the release gate enforces this).

**Explicitly NOT in this milestone**, each now with a citation rather than an opinion:
graph DB / Neo4j ([2506.05690](https://arxiv.org/abs/2506.05690) Obs.1), PPR over the pose graph
(BFS is exact at 257 nodes), embeddings over the journal (2603.02473), GraphRAG community
summaries (Obs.8: 40k vs 900 tokens), LLM relation-extraction for typed edges
([2510.10114](https://arxiv.org/abs/2510.10114) — "unstable and costly").

---

## v0.3.1 — Tell the truth in public ✅ SHIPPED 2026-08-10

Small, and it is currently wrong, which is why it is not deferred.

- **`FACTS.md [LP-CAST]` is false.** It claims relationships are graph context; there are zero
  cross-character edges. After v0.3.0 #1 the honest replacement is *stronger* than the original:
  **"relationships are reconstructed at read time, per character, from what each one can actually
  see."** Generative Agents does not store relationship edges either.
- **Sync the public repo.** `scripts/release.py check` reports **15 files of drift**: 3 missing
  (`hf_gen.py`, 2 test files), 12 stale. The published code still documents a Midjourney path that
  has returned 403 on every call since **2026-07-02** — anyone who clones it gets a system that
  cannot run.
- Push is a human `git push`, by design.

**Effort: ~1h.**

---

## v0.4.0 — "It writes back" ✅ SHIPPED 2026-08-11

**Not what this slot said it would be.** The plan here was hosting; what actually happened is
that "fix everything that needs fixing" turned into the memory-write-back half of the context
graph, and hosting moved down. Recorded rather than rewritten, because the divergence is the
interesting part: the roadmap was written from an audit, and production had more to say.

Shipped: the lived write-back + a 76-day backfill from the pick-log (847k picks), bi-temporal
provenance (259/259 nodes, 1472/1472 edges), nightly reflection, typed edges (1030/1472),
escape velocity and anti-reverse forgiveness, the daytime-reachability fix, the web-link port
that stopped the star topology being manufactured daily, an eight-detector health oracle, and
a deploy ledger. Full notes in `CHANGELOG.md`.

Three things it also uncovered, none of which were on any plan: a character spec missing from
production for months, the world-context token dead for 27 days, and 982 authored link motions
being discarded by the generator.

---

## v0.5.0 — A URL you can check ← THE NEXT MILESTONE

`_hosting/PROPOSAL.md (internal)`, already costed. Cloudflare R2 on a custom domain, **$0/month**, Vercel
never in the path.

~~The blocker is not the host: `graph_viewer.html:127` renders every node as a full-size PNG, so
a cold load pulls **~355 MB**.~~ **Retracted 2026-09-06 — this stopped being true when the viewer
was rewritten in `dcd6fc3` and the roadmap's cost model was not updated with it.** The viewer now
draws nodes as `shape:"dot", size:12` (`graph_viewer.html:283`); the only two `<img>` uses left
are in the *detail* pane for the selected node, both `loading="lazy"`
(`graph_viewer.html:543`, `:567`). So a cold load pulls the JSON and nothing else, and the
355 MB figure describes a viewer that no longer exists. The transcode below is still worth doing
for the per-clip previews, but it is no longer a blocker on publishing.

Transcode (GIF→h264 measured ~20× on 10 real clips) and the
published footprint is 332 MB with a 7.2 MB session. Freshness is a push from
`autogen._rebuild_graph()` with content-addressed keys, uploading `graph.json` **last** so the
index never points at an object that is not up yet.

**Effort: ~1 day.** Deferred past v0.4.0 for the same reason it was deferred past v0.3.0:
publishing a graph whose characters could not remember was the less interesting version. They
can now, and the graph has a time dimension and typed edges to show, so this is the next one
worth doing.

---

## v0.6.0 — Make the load-bearing parts provable

From the map's risk ranking, not from a desire for coverage.

1. **`_preview_graph.py`: 506 lines, zero tests** — the entire production render loop. Every
   behavioural bug the git log records (reverse pendulum, frozen walk, dark panels) lived here.
   Test `_pick`, dwell accounting, and graph hot-reload at minimum.
2. **`heartbeat.propose_pose`** — untested, and it is what spends money downstream.
3. **`hf_gen.py` proxy contract** — untested; it has already broken once (`e96c65a5`, the
   extensionless-key `.bin` failure).
4. **`proposals.json` has no lock** around `add_proposal` / `set_status`, with `lp-mind` and
   `lp-gen` both writing. The map inferred the race from reading, not from observing one — so
   step one is to *reproduce* it, then fix it.
5. **`lp-graphserver` last returned `4294967295`** — a failure on a task nothing appears to
   depend on. Either it matters and is broken, or it does not and should be deleted.

---

## Later, in rough order

- **Reflection nodes** (audit #3, research #4): nightly summaries appended to the same JSONL as
  `kind: "reflection"`, triggered by a mood-band shift or a broken repetition run rather than the
  clock ([2604.12285](https://arxiv.org/abs/2604.12285)).
- **Seraphina.** One node, no panel, and Phineas's designated rival. Either hang a third panel or
  admit MAXX is the rival — right now the persona prose and the topology disagree.
- **Re-seed `context-graphs`.** One query (`graph memory consolidation summarization node`) was
  lost to an S2 429, and it is exactly the consolidation literature the reflection work needs.
- **Fix `seed_papers` pinning.** GraphRAG (`2404.16130`) was a declared seed paper and did not
  survive the ranker into the snapshot. A declared seed should be pinned.

---

## Risks worth naming

- **The budget is the only thing between an autonomous loop and a spent month.** 13/day × 7.5 ×
  30 = 2925 against 3000. It has held twice. It has not been tested against a month.
- **`hil` is a single point of failure** — one Windows box, physical panels, and the console user
  must stay logged in.
- **The Higgsfield session is single-owner on `node`.** Running `hf` anywhere else kills it.
- **Credits do not roll over.** A quiet month is 3000 credits thrown away; a burst month cannot
  borrow from it.
