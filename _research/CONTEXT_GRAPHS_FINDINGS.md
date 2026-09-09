# Context Graphs: what the literature says about the Living Portraits audit

Run 2026-08-09 through `life/research/` (Semantic Scholar seed → LLM distill → cluster).
Tests `_audit/CONTEXT_GRAPH_AUDIT.md` (2026-08-08) against the corpus the pipeline returned.

**Headline: the audit's central call survives. Its ranking of the fixes does not, quite.**
The corpus supports "no graph DB at this size" more strongly than the audit argued, supports
"fix retrieval first" with a directly-on-point citation, and says the audit's #1 (cross-character
signal computed at read time) is not a compromise — it is what the canonical believable-character
paper actually did. The one place the audit is behind: recency/importance/relevance is a 2023
floor, not 2026 state of the art, and the delta is worth about two extra hours.

---

## 1. What the pipeline actually returned

**Seed** (`data/research/papers/context-graphs.jsonl`, `.run.json`):

| | |
|---|---|
| Queries run | 15 (1 failed: `graph memory consolidation summarization node` → S2 429) |
| Unique papers found | **682** |
| Ranked kept (`top_n`) | **120** |
| Wall time | 4m45s |

**Distill** (`data/research/distill/context-graphs/tldr.yaml`) — ran clean on `claude -p`, no key
problem: **49 cards** from the top 50 (the one skip is an off-topic molecular-properties paper).
Types: 35 method, 4 benchmark, 4 survey, 3 empirical, 1 position. Novelty: 9 high / 33 medium / 5 low.

**Clusters** (`clusters.yaml`): 7 themes. Three matter here — *episodic + semantic memory as separate
layers*, *agentic retrieval over one-shot top-k*, and *frontier memory for embodied agents*.

### Two pipeline bugs found and one worth fixing

1. **Fixed (committed to `research/topics.yaml`).** The new topic had `since: 2020-01-01` unquoted.
   PyYAML parses that as `datetime.date`, and `seed_topic._year_filter` does `since_iso[:4]` →
   `TypeError: 'datetime.date' object is not subscriptable`. Every other topic in the file quotes it.
2. **Not fixed — real bug.** GraphRAG (`2404.16130`), a declared `seed_papers` entry, is **absent
   from the 120-row snapshot**. Seed papers are not pinned past the ranker. Verified the paper exists
   (`scholar.get_paper ARXIV:2404.16130` → "From Local to Global: A Graph RAG Approach…", 2024).

### New vs the adjacent topics

Deduped against `agent-memory-frontier` (200) + `agent-forgetting` (200):

- **95 of 120 seeded papers are new.** **37 of 49 distilled cards are new.**
- What `context-graphs` uniquely surfaced: the **GraphRAG-skeptic literature**
  (`2506.05690`, `2508.06105`, `2510.10114`), the **embodied/spatial memory cluster**
  (`2411.17735`, `2505.22657`, `2305.17537`, `2502.10177`, `2603.19137` — 14 papers, none in the
  adjacent snapshots), and **Generative Agents itself** (`2304.03442`), which neither adjacent
  topic had. That last one is notable: the project's closest ancestor was not in Ray's corpus at all
  until this topic ran.
- What it did **not** surface: anything meaningful on social/relational memory (see §4).

### Audit citation check

Both of the audit's load-bearing 2026 citations verify against S2, though neither is in any of the
three snapshots:

- `2603.02473` **"Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory"** — real.
  Abstract confirms the audit's paraphrase and adds a clause the audit did not quote (see §3).
- `2605.04897` **"Storage Is Not Memory"** — real, and present in `agent-memory-frontier`.

**Top papers by distilled signal score:** Generative Agents (9) · REMem (8) · Zep (8) · HippoRAG (8) ·
Mem0 (8) · LoCoMo (8) · then a band of 7s: When-to-use-Graphs, HippoRAG 2, Hindsight, Agent-Native
Memory, LiCoMemory, GAM, RMM, AriGraph, LinearRAG, GraphRAG-R1, GraphRAG-Bench.

---

## 2. Q1 — Does the corpus back "don't build a knowledge graph at 257 nodes"?

**Yes, and for a better reason than size.** The audit argued from scale ("a dict and a deque beat a
database at this size"). The corpus argues from **task shape**, which is the more defensible line
on stage and does not stop being true if the graph grows to 2,000 nodes.

**GraphRAG-Bench / "When to use Graphs in RAG"** ([arXiv 2506.05690](https://arxiv.org/abs/2506.05690),
ICLR'26) is the direct answer. Its numbered observations:

- *Obs.1* — "Basic RAG is comparable to or outperforms GraphRAG in **simple fact retrieval tasks that
  do not require complex reasoning across connected concepts**." Vanilla RAG hit 83.2% evidence recall
  on single-passage questions.
- *Obs.2* — graphs win on "complex reasoning, contextual summarize, and creative generation."
- *Obs.8* — **cost**: MS-GraphRAG global reached ~40,000 tokens per query vs vanilla RAG's ~900.
- *Obs.9* — "excessive token accumulation often introduces redundant information, which in turn
  degrades context relevance."

Living Portraits' memory query is "what did I just do, and how do I feel about it" — Level-1 fact
retrieval over a single character's own log. That is precisely the regime where the benchmark says
the graph loses *and* costs 40×.

Three more, converging:

- **"Are We Ready For An Agent-Native Memory System?"** ([2606.24775](https://arxiv.org/abs/2606.24775),
  12 systems × 11 datasets): "**no single architecture dominates**; effectiveness depends heavily on
  how well the memory structure aligns with the **workload bottleneck**." Living Portraits' bottleneck
  is measured and unambiguous: it reads 5 of 24,598 entries. That is a *retrieval* bottleneck, and no
  change to representation touches it.
- **"Does Memory Need Graphs?"** ([2601.01280](https://arxiv.org/abs/2601.01280), LongMemEval +
  HaluMem): "many performance differences are driven by **foundational system settings rather than
  specific architectural innovations**."
- The field is actively retreating from expensive graph construction: **LinearRAG**
  ([2510.10114](https://arxiv.org/abs/2510.10114)) drops LLM relation extraction entirely as "unstable
  and costly," and **LogicRAG** ([2508.06105](https://arxiv.org/abs/2508.06105)) builds a query-time
  DAG instead of any pre-built graph. Both beat their GraphRAG baselines.

**Where the corpus pushes back.** **AriGraph** ([2407.04363](https://arxiv.org/abs/2407.04363)) is the
uncomfortable one: an LLM agent that *constructs and updates* a memory graph integrating semantic and
episodic memory while exploring a text world, and it "markedly outperforms other established memory
methods and strong RL baselines." That is the closest structural analogue to Living Portraits in the
whole corpus — an agent inhabiting a world, accumulating experience.

The distinction that saves the audit: AriGraph's agent accumulates **facts about a world it can
perceive** (objects, locations, state changes). Living Portraits has no world sensor — the audit's own
opportunity #6, correctly ranked last. Until the portraits can perceive the floor, there is nothing
for a declarative graph to hold. **The audit's "not worth doing" list is right, but the reason to put
in the deck is "we have no world to model yet," not "255 nodes is small."**

---

## 3. Q2 — Is recency/importance/relevance still SOTA? What would I implement?

**No. It is the 2023 floor.** It is still the right *first* move here, and the corpus says why, but it
is not where the field is.

### The case for doing it anyway, first

`2603.02473` (verified live via `scholar.get_paper`, abstract quoted):

> "On LoCoMo, **retrieval method is the dominant factor**: average accuracy spans 20 points across
> retrieval methods (57.1% to 77.2%) but only 3-8 points across write strategies. **Raw chunked
> storage, which requires zero LLM calls, matches or outperforms expensive lossy alternatives**,
> suggesting that current memory pipelines may discard useful context that downstream retrieval
> mechanisms fail to compensate for."

The second sentence is stronger than the audit used it. It is a direct endorsement of "**keep the
JSONL, change the reader**" — not a compromise, the measured-best option. The audit's instinct to
leave the 7 MB of raw first-person prose untouched is the paper's actual recommendation.

### Where the field went after Park et al.

Three axes, all in the distilled cards:

1. **Retrieval became iterative, not a scoring function.** **REMem**
   ([2602.13530](https://arxiv.org/abs/2602.13530)) uses an agentic retriever with tools that traverse
   a time-aware memory graph across multiple steps: **+3.4% recollection, +13.4% episodic reasoning
   over Mem0 and HippoRAG 2**. Same direction in GraphRAG-R1 (`2507.23581`) and OrcaLoca (`2502.00350`).
2. **Writes became multi-granular and retrieval became learned.** **RMM**
   ([2503.08026](https://arxiv.org/abs/2503.08026)) summarizes at utterance / turn / session
   granularity (prospective) and tunes the retriever online using the LLM's own cited evidence as
   reward (retrospective): >10% on LongMemEval.
3. **Evidence and inference got separated structurally.** **Hindsight**
   ([2512.12818](https://arxiv.org/abs/2512.12818)) splits memory into four networks — world facts,
   agent experiences, entity summaries, **evolving beliefs** — with `retain / recall / reflect` as the
   API. **GAM** ([2604.12285](https://arxiv.org/abs/2604.12285)) keeps an episodic event graph separate
   from a consolidated topic network and promotes between them **on semantic shift, not on a clock**.

### What I would actually implement in `heartbeat.py`

Replace `_journal_tail(character, n=5)` with a scorer. Concretely:

```
score(e) = w_r * 0.995 ** (hours_since(e.ts))          # recency, Park's decay
         + w_i * -log2( count(e.goal) / N )            # importance = self-information
         + w_v * relevance(e)                          # see below
```

Three deliberate choices, each earned from a paper rather than from taste:

- **importance = self-information of the goal, `-log2 p(goal)`,** over a `Counter` of the character's
  own history. This is the audit's "rarity" idea given a unit. Phineas has 106 distinct goals over
  11,227 lines; `jealous_glare` at 18.3% scores ~2.4 bits, a once-ever goal scores ~13.5. It needs no
  LLM call, which is what `2603.02473` says to prefer.
- **relevance uses the graph, and this is the part that is genuinely Living Portraits'.** Every other
  system in this corpus computes relevance by embedding cosine because it has no topology. This project
  has an exact 255-node adjacency and a working BFS. Score an entry by
  `1 / (1 + bfs_hops(e.pose, current_pose))` — memories from *here and nearby* outrank memories from
  across the graph, and it is free (`pathfind.py` already computes it). Combine with exact match on
  `goal` for the pose being considered. **This is the one retrieval term nobody else in the corpus can
  build, and it is the honest answer to "why does this project have a graph."**
- **`k` by token budget, not by count.** FinMem ([2311.13743](https://arxiv.org/abs/2311.13743))
  makes its "perceptual span" an explicit tunable knob rather than a constant. `JOURNAL_TAIL = 5`
  becomes `JOURNAL_BUDGET_TOKENS`, which also caps the prompt growth the audit flagged in §6.

Then two cheap upgrades over the audit's plan, ~2h total on top of its 4:

- **One aggregate line** the LLM can reason over — the audit already proposed this
  ("You have chosen jealous_glare 2,060 times in 64 days"). Keep it; it is the single line most likely
  to change behaviour, and it costs a `Counter`.
- **Consolidate on semantic shift, not at bedtime** (GAM). The audit's #3 fires nightly on the clock.
  GAM's result says gate it on a detected shift — for this codebase, a change in mood band or a broken
  repetition run. Bedtime is still the right *compute* window; the trigger should be the shift.

**Do not** build the RL retriever (RMM) or the agentic multi-step retriever (REMem). Both need traffic
and eval infrastructure this project does not have. They are the correct 2027 answer.

---

## 4. Q3 — Is there published work on social/relational memory between characters?

**Almost none, and that is the finding.** Being precise, because the audit called this the cheap
high-value gap and it deserves an honest answer rather than a padded one:

- Of the **120 seeded papers**, exactly **2** match any of `theory of mind | social memory |
  relational memory | interpersonal | NPC | relationship graph`. One is Generative Agents. The other
  (`2410.09824`, GraphAgent-Generator) is about *generating* social network topology at 100k-node
  scale, not about one character's memory of another.
- The **400 papers in the adjacent snapshots add nothing** on this axis.
- The topic file *asked* for it — two of the fifteen queries were `social relational memory
  multi-agent theory of mind` and `character believability persistent NPC memory`. They returned
  GraphRAG papers.

This is partly a real hole in the field. Agent-memory research is benchmarked almost entirely on
**LoCoMo and LongMemEval** — conversational QA — which has no notion of what A feels about B. And it
is partly a ranker artifact: the pipeline's `citation_velocity_weight: 0.5` buries 2026 papers with
zero citations, which is exactly where this work lives.

**Filling the gap by web search, after the pipeline ran** (all four verified via `scholar.get_paper`,
none in any snapshot):

- **Subjective-Graph LLM Agents for Simulating Uncertainty in Classroom Social Perception**
  ([2603.20750](https://arxiv.org/abs/2603.20750), 2026). The closest thing to what the audit wants.
  Each agent operates through an **individualized subjective graph** determining peer visibility and
  evidence access, with Bayesian belief states. The premise is stated as: *"Social actors do not
  observe a common social world: each individual forms judgments from a partial and potentially
  distorted view of the surrounding network."* On 482 students / 12 classrooms, collective ranking
  error **grows 0.066 → 0.124 over six epochs despite repeated objective performance signals** — a
  persistent distortion that objective evidence does not correct. That is Phineas exactly: a private,
  wrong, self-reinforcing model of a neighbour. The design implication is sharp — **there is no
  relationship graph, there are two subjective graphs**, and Phineas's edge to Seraphina should not be
  the same object as hers to him.
- **Ella: Embodied Social Agents with Lifelong Memory**
  ([2506.24019](https://arxiv.org/abs/2506.24019), 2025). 15 agents living socially in a 3D open world
  for days, on a **name-centric semantic memory** plus **spatiotemporal episodic memory**; the agents
  "build social relationships." This is the nearest existing *system* to Living Portraits and the right
  citation if the deck needs one for "characters that remember each other."
- **Profile-Graph Memory** ([2607.19359](https://arxiv.org/abs/2607.19359), 2026) — cross-entity
  traversal via substring-matched entity names appearing in LLM-written narrative profiles. A way to
  get cross-character edges out of prose without an extraction pipeline.

**And the strongest point for the audit, which the audit missed about its own recommendation:**
Generative Agents ([2304.03442](https://ar5iv.labs.arxiv.org/html/2304.03442)) — the paper the whole
believable-character line descends from, 5,062 citations — **does not store a relationship graph
either.** Relationships in Park et al. are emergent from the memory stream and reconstructed at
retrieval time. The audit's opportunity #1 ("a cross-character edge computed at read time rather than
stored") is not a shortcut that happens to be cheap. It is what the canonical system does, and
`2603.20750` says the per-character subjectivity is a feature to keep, not an approximation to fix later.

---

## 5. Q4 — The capability nobody in this project has considered

**Frontier memory: represent what the character has *never done* as a first-class, retrievable thing.**

**3D-Mem** ([2411.17735](https://arxiv.org/abs/2411.17735)) splits an embodied agent's memory into
*Memory Snapshots* (explored) and **Frontier Snapshots** (glimpses of unexplored regions), so planning
can "consider both known and potential new information." The pipeline's cluster #7 named this as its
own theme across three papers. Every memory system in the rest of the corpus stores only what happened.

Living Portraits is the ideal case for it and has the data sitting there:

```python
never_visited = set(graph.nodes) - {e["pose"] for e in journal}
frontier = never_visited & pathfind.reachable_poses(graph.edges, current_node)
```

Two lines. It converts the audit's most damning graph statistic — **109 of 255 nodes with ≤1 outgoing
transition, and an unmeasured number never visited in 64 days** — from a defect into content. A
character that can say *"there is a version of me I have never been, and it is two steps away"* has a
want that is not a mood string, and it is grounded in the real topology rather than the persona prose.
It also gives the softmax in `policy.weigh()` a principled novelty term over the *lifetime* instead of
`deque(maxlen=8)`, which is minutes.

Two smaller ones, same source, both nearly free:

- **Refusal-on-unanswerable as a memory-quality metric** (REMem, `2602.13530`). This project currently
  has **no way to tell whether a memory change helped**. A ten-question probe — five answerable from
  the journal, five not — is the smallest honest eval, and "the portrait correctly says *I don't
  remember*" is a better demo than any retrieval score.
- **Update latency, not just retrieval quality** (LiCoMemory, [2511.01448](https://arxiv.org/abs/2511.01448)).
  The heartbeat has a ~7-minute budget. Any scorer added to `_build_user_prompt` should be timed against
  it from day one.

---

## 6. What to borrow — concrete, tied to this codebase

Ordered by (value / effort), and where it differs from the audit's ranking, why.

| # | Change | File | Effort | Grounded in |
|---|---|---|---|---|
| 1 | Neighbour line in the prompt from the other character's `pose/*.json` + last journal entry. **Keep it per-character and asymmetric** — Phineas's read of Seraphina need not match hers of him. | `heartbeat.py:359` `_build_user_prompt` | ~1h | `2304.03442` (relations reconstructed at retrieval), `2603.20750` (subjective graphs) |
| 2 | Scored retrieval replacing `JOURNAL_TAIL = 5`: recency decay + `-log2 p(goal)` importance + **BFS-hop relevance** + `k` by token budget. Plus the one aggregate count line. | `heartbeat.py:72,207` new `journal_score.py` | ~4h | `2603.02473` (retrieval dominates; raw storage wins), `2304.03442`, `2311.13743` (span as a knob) |
| 3 | `frontier = reachable_poses − visited_poses`, one line in the prompt. | `heartbeat.py` + `pathfind.py` | ~30min | `2411.17735` |
| 4 | Nightly reflection appended to the same JSONL with `kind: "reflection"`, **triggered by mood-band shift or a broken repetition run, not the clock**. | `circadian.py` / `heartbeat.py` | ~4h | `2604.12285` (GAM, semantic-shift gating), `2304.03442` |
| 5 | Write `band` into `intent.json` at generation time so `policy._mood_bias` reads a field. Frame it as a memory-fidelity fix, not a bug fix: 88% of Phineas's authored interior state never reaches his body. | `heartbeat.py`, `policy.py:81` | ~2h | `2512.12818` (separate evidence from inference), `2501.10332` (emotional memory as a first-class store) |
| 6 | Ten-question answerable/unanswerable probe as the regression test for 2–4. | new `tests/` | ~2h | `2602.13530` |

**Explicitly do not build**, now with citations rather than assertion: graph DB / Neo4j
(`2506.05690` Obs.1, `2606.24775`), PPR over the pose graph (BFS is exact here; PPR earns its keep on
multi-hop entity association), embeddings over the journal (`2603.02473` — raw storage + good retrieval
beats lossy extraction), GraphRAG community summaries (`2506.05690` Obs.8: 40k vs 900 tokens),
LLM relation extraction to build typed edges (`2510.10114` calls it "unstable and costly").

**One deck correction the corpus forces.** The audit told Ray to fix `FACTS.md [LP-CAST]` because
zero cross-character edges makes "relationships are graph context" false. That stands. But the
replacement line should not be an apology — Generative Agents doesn't store relationship edges either.
The defensible claim is: *"relationships are reconstructed at read time, per character, from what each
one can actually see"* — which is both true after change #1 and a stronger point than the original.

---

## 7. What I did not verify

- **I did not read any full paper.** Every claim here is from the abstract, the distilled card, or —
  for `2506.05690` only — the arXiv HTML, where I read the numbered observations (Obs.1/2/8/9). The
  40,000-vs-900-token figure and the 83.2% recall come from that HTML rendering via WebFetch, not from
  the PDF.
- **`2601.01280` ("Does Memory Need Graphs?")**: a WebFetch of the PDF returned a summary with
  confident-sounding numbers I could not corroborate, so I discarded it and quote only the S2 abstract.
  Treat that paper's specific numbers as unread.
- **`2603.20750` and `2606.06036` have null abstracts in S2.** The Subjective-Graph details (482
  students, 12 classrooms, 0.066→0.124) come from a WebFetch of the arXiv abstract page, single source.
- **Node/edge counts.** The brief says 257/1462; the audit measured 255/1452 on `hil`. I measured
  neither — I did not touch the production graph, the journals, or any Living Portraits code. Every
  codebase figure in this document is quoted from the audit.
- **The BFS-relevance term is my proposal, not a published result.** No paper in the corpus scores
  memory relevance by graph-hop distance, because no system in the corpus has a topology to do it with.
  It is well-motivated and cheap, and it is untested.
- **Effort estimates are the audit's**, carried forward. I added ~2h for the GAM/frontier deltas by
  analogy, not by measurement.
- **The social-memory gap may be a search artifact rather than a real hole.** Web search found four
  on-point 2026 papers in one query that fifteen S2 queries missed. I did not re-seed the topic with
  better queries to find out how much more is there — that is the obvious next run, with
  `citation_velocity_weight` lowered so 2026 zero-citation work can rank.
- **`agent-forgetting` and `agent-memory-frontier` were deduped by `paper_id` only.** Same paper under
  two S2 ids would read as new.

---

**Artifacts:** `data/research/papers/context-graphs.{jsonl,run.json}` ·
`data/research/distill/context-graphs/{tldr.yaml,clusters.yaml}`
