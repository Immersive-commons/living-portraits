# FACTS.md - Context Graphs deck (B-1)

Every slide claim must cite an id below, or carry an explicit `UNVERIFIED` tag.
ids resolve in `citations.yaml`. Prose here is the source of truth; chunks paraphrase.
No em/en-dashes in any prose a reader sees (this file is internal; the chunks obey the ban).

Legend: [LP-*] = Living Portraits repo. [RAG-*] = life/research. [ASIMOV-*] = Reflection AI (web, verified). [DNA-*] = palette/motif.

---

## A. Living Portraits as a context graph

### [LP-GRAPH] The graph itself (graph_viewer.html)
- Living Portraits ships a real interactive graph view titled "Video Knowledge Graph" (`graph_viewer.html`, rendered with vis-network).
- NODES are frames: each node is a pose that carries its still image plus its gen prompt (the SDXL/MJ generation prompt) as semantic metadata.
- EDGES are video clips. Two kinds:
  - idle self-loops (a clip that loops on one pose); drawn as solid edges tinted to the character.
  - transitions (pose to pose): a clip that ties one frame to a different one and carries its MJ motion prompt; drawn as a dashed orange edge (`#ff5a3c`), plays once, has an arrow.
- The player random-walks this graph by day.
- Per-character node color: Phineas gold `#d4a85a`, Seraphina teal `#78ccc2`, MAXX-9 magenta `#ff3df0`. The bedtime / circadian layer is indigo `#8b6cff`.
- The viewer footer reports live counts from `data/clips/video_graph.json`: N frames, N clips, N idle + N transition, N characters. (Exact totals are runtime data; do NOT hardcode a number on a slide. Say "dozens of poses and clips" or show the graph, not a fabricated count.) [LP-GRAPH]

### [LP-CAST] The cast (gallery.yaml)
- Three characters, each a node-set with its own persona, voice, and node color:
  - Master Phineas Quill (panel A): a washed-up tragedian, mood "tragic-grandiose", voice en_GB-alan. Node gold.
  - Seraphina Vane: a Restoration-era wit, mood "wry-complicit", voice en_GB-jenny. Node teal.
  - MAXX-9 (panel B): a "legally-distinct discount action hero", voice en_US-ryan. Node magenta.
- The personas relate: Phineas resents the brighter frame next door (Seraphina); MAXX-9 is the loud neon anachronism Phineas loathes. **Relationships are reconstructed at READ time, per character, from what each one can actually see** — since v0.3.0 each prompt carries a line built from the other panel's own published pose and last mood, so the resentment is a response to a neighbour's actual state rather than a standing fact. Nothing relational is stored: there are **zero cross-character edges** in the graph, by design, which is also how Generative Agents handles relationships. A panel that has gone dark is simply unseen. [LP-CAST]
- CORRECTED 2026-08-10: this line previously read "(Relationships are graph context, not just decoration.)" That was false — the relationship lived in `gallery.yaml` prose and one clause of a `gen_prompt`, and the graph had no cross-character edges to carry it. It is now true in the stronger, read-time form above.

### [LP-HEARTBEAT] The heartbeat memory loop (AUTONOMY.md)
- An LLM "heartbeat" gives each portrait a life: `director/heartbeat.py` is the slow brain that runs sense, then think, then write.
  - sense: pulls real-world context (see [LP-CONTEXT]) plus each character's recent pose state and journal.
  - think: a single GLM decision (model `glm-5.1`) picks a goal pose for the character, plus a free-text mood and the mood-band the body understands.
  - write: writes `data/mind/intent.json` (the goal) and appends one line to `data/mind/journal/<char>.jsonl` (the character's inner monologue, fed back next tick).
- The walker reads the graph and the intent: `runtime/mind.py:decide()` runs `runtime/pathfind.py:next_step()` (a BFS over the transition edges) to find the next single hop toward the goal, forces that one transition, lands one pose closer, and asks again. A visible step-by-step walk. At the goal it pins on idle loops.
- **The graph is the MAP the loop reads. The journal is the MEMORY it writes.** The heartbeat reads pose state + journal and writes intent; the walker reads intent + the graph and writes pose state. The graph itself is read-only at runtime — it is an asset manifest grown offline by the generation pipeline, not something experience accretes into. Single-writer per file, no races. All writes atomic (tmp + os.replace). [LP-HEARTBEAT]
- Since v0.3.0 the journal is read by SCORE, not by recency: recency decay + importance as surprisal of the want + relevance in transition hops, under a token budget, with two slots reserved for distant memories that still stand out (`runtime/journal_score.py`). The hop-distance term is the part that needs the graph — a memory made two steps from where the character stands is nearer than one across the graph. [LP-HEARTBEAT]
- CORRECTED 2026-08-10: this previously read "THE GRAPH IS THE MEMORY the loop reads and writes." The loop reads the graph and writes two files that are not in it, so as stated it was false.

### [LP-PRECEDENCE] One graph, layered traversals (AUTONOMY.md + circadian.py)
- Three goal-setters drive the same body, in strict precedence:
  `circadian (the clock) > mind (the LLM intent) > random walk (fallback)`.
- circadian (`runtime/circadian.py:decide()`) is a hardcoded goal-seeker: when the wall clock crosses a character's bedtime window it stops wandering and walks a scripted BEDTIME ROUTINE (hub, nightcap, exit offscreen, return in nightclothes, prep, sleep), dwelling a few idle loops per beat, then dwells at the sleep pose until wake hour, then walks the chain in reverse.
- mind (`runtime/mind.py:decide()`) is the SAME decide() contract as circadian, but its goal is the LLM-chosen pose by day.
- So it is ONE graph with multiple traversals over it: a daytime LLM-goal walk, a daytime random walk, and a nightly circadian chain. Same nodes and edges, different ways to walk them. [LP-PRECEDENCE]

### [LP-STALE] Fail-soft staleness guard (mind.py + AUTONOMY.md)
- A goal older than `DEFAULT_MAX_AGE` (30 minutes, `1800.0` seconds in `runtime/mind.py`) is ignored. If the heartbeat dies or is blocked, the portraits degrade to the lively random walk, never frozen at a stale goal.
- `runtime/circadian.py` and `runtime/mind.py` and `runtime/pathfind.py` are pure, stdlib-only, import-safe (imported by the 10 fps render loop). All network and LLM code lives in `director/`. [LP-STALE]

### [LP-CONTEXT] Real-world context into the prompt (director/context.py)
- `director/context.py:context_line()` pulls IC `GET /api/context` (Frontier Tower SF local weather + time, Open-Meteo behind it) and renders ONE short natural phrase for the heartbeat prompt, e.g. "a chilly 63F overcast afternoon in SF; the sun sets at 8:33pm".
- Fail-soft by contract: it swallows ALL failure (no token, network down, non-2xx, bad JSON) and returns an empty string, never raising. No network at import time. TTL-cached to `data/mind/context.json` (15 min TTL) so almost every ~4-min tick is served from cache with zero network. A dead source just omits the clause; it can NEVER break a heartbeat tick. [LP-CONTEXT]

### [LP-BRAIN] The brain (AUTONOMY.md)
- Model: `glm-5.1` for both the per-tick decision and Phase-2 pose authorship (`heartbeat.PROSE_MODEL`, `llm.DEFAULT_MODEL`). It was `glm-4.5-air` through 2026-07; that name is stale wherever it still appears.
- Path: `director/llm.py` posts to the IC z.ai gateway (Anthropic Messages shape) with an IC-minted `agt_` proxy key that carries ZERO IC tool scopes, so it is safe on an unattended host. The gateway holds the real Z.ai org key and meters a weekly token budget. [LP-BRAIN]

### [LP-DEPLOY] Where it runs (README.md + AUTONOMY.md)
- Runs unattended on a dedicated host (referred to as "hil" in the autonomy deploy runbook); the original player box is supercommons2 (i9-9900K / 64GB / RTX 2080 Ti).
- The player is deterministic and plays pre-baked clips; nothing is generated in realtime in the show loop. A self-heal watchdog keeps it alive (README status + `lp_watchdog_preview.ps1` / `start_portraits.ps1`). [LP-DEPLOY]
- UNVERIFIED-LIVE: exact current host name and uptime are runtime facts; do not state a specific uptime number on a slide.

### [LP-PHASE1] What is actually built (AUTONOMY.md)
- Phase 1 ("the brain over existing poses") is marked BUILT: pathfind + mind + heartbeat + walker hook + 14 tests + a live-validated LLM. GLM picks among EXISTING poses; the graph walks there. No new image generation in this phase.
- Phase 2 (autonomous pose growth: the mind requests a brand-new pose, an `lp-gen` worker runs the MJ pipeline, the walker hot-reloads the graph) is designed, not the claim of this talk. Tag any Phase-2 statement as roadmap. [LP-PHASE1]

---

## B. RAG research grounding (life/research)

### [RAG-FARM] The pgvector embedding farm (research/embed/README.md)
- life/research runs a distributed RAG farm: a cluster-native paper embedder runs `nomic-embed-text` on each node and writes 768-dim vectors to a central Postgres 16 + pgvector instance (db `research`, on the `immersivecommons` box over the tailnet).
- Schema: a `papers` table (paper_id, title, abstract, year, authors, embedding vector(768), embedded_at). Work queue = rows WHERE embedded_at IS NULL; nodes pull jobs via FOR UPDATE SKIP LOCKED. Scale by adding boxes. [RAG-FARM]
- This is a real, running flat-vector store: it is exactly the "nearest chunks by similarity" baseline a context graph is contrasted against.

### [RAG-CROSS] cross_search (research/cross_search.py)
- `research/cross_search.py` (also `python -m research.cross_search`) is the retrieval entry point over that farm. [RAG-CROSS]

### [RAG-GRAPHRAG] GraphRAG vs flat RAG, as a named industry move (research/AGENTIC_SOTA_v1.md + SOTA_UPGRADE.md)
- The life/research SOTA scan names the graph-structured memory direction explicitly. Mechanisms catalogued include:
  - a temporal knowledge graph with edge invalidation (Zep),
  - a hippocampal-index knowledge graph traversed by Personalized PageRank for cheap multi-hop retrieval (HippoRAG / HippoRAG 2),
  - memory-stream + reflection synthesis with recency/importance/relevance retrieval (Generative Agents),
  - "preserve raw events verbatim, run extraction at retrieval time not ingestion" (Storage-is-not-Memory).
- Honest caveat the scan records: Microsoft GraphRAG community-summary indexing was a deliberate SKIP for this corpus because it caused a ~370x token blowup; LLM-as-reranker was skipped (~270x slower for ~2 nDCG points). So "graph beats flat" is a SHAPE claim (relationships, reachability, multi-hop), NOT a blanket "always cheaper" claim. State it that way. [RAG-GRAPHRAG]
- Provenance the scan cites: HippoRAG 2 (arXiv 2502.14802), Qwen3-Embed/Rerank (2506.05176), Anthropic Contextual Retrieval, Jina late-chunking (2409.04701). [RAG-GRAPHRAG]

### [RAG-CONTRAST] The core contrast (synthesis of LP-GRAPH + RAG-FARM + RAG-GRAPHRAG)
- Flat RAG: embed everything, retrieve the nearest chunks by cosine similarity. You get the most-similar pieces, but no relationships, no order, no reachability, no path between them.
- Context graph: memory as nodes plus TYPED edges. Retrieval becomes a WALK (traverse typed edges), not a similarity scan. You get reachability, ordering, and grounded multi-hop the model can actually traverse.
- Living Portraits is the toy that makes this literal: the "nearest pose" is meaningless; what matters is which pose you can REACH and the path the body takes to get there (pathfind BFS). [RAG-CONTRAST]

---

## C. Asimov / Reflection AI (verified via WebSearch 2026-06-26)

### [ASIMOV] Asimov is a real context-graph product
- Reflection AI (founded by alumni of Google DeepMind) launched Asimov in July 2025. [ASIMOV-SEQUOIA] [ASIMOV-DOCS]
- Asimov is a code-RESEARCH / code-comprehension agent: it helps engineers understand large existing codebases rather than primarily generating new code. Company framing: roughly 70% of engineering time is spent reading and understanding existing systems, not writing new code. [ASIMOV-SILICONANGLE] [ASIMOV-SEQUOIA]
- It builds a knowledge graph: it continuously indexes entire GitHub repositories AND architecture docs, chat threads (Slack / Teams), issue trackers (Jira), and commit histories, to construct a living knowledge graph of the codebase plus the institutional knowledge around it ("learns from more than just code"). [ASIMOV-SILICONANGLE] [ASIMOV-DOCS]
- Architecture: many small long-context retriever agents that pull relevant info from the large codebase, plus one large short-context reasoning agent (the combiner) that synthesizes a coherent answer. [ASIMOV-DOCS]
- Deploys inside the customer's own VPC so proprietary source stays in their cloud. [ASIMOV-DOCS]
- Why it belongs on the slide: it is the same pattern at industry scale. Model the domain (a codebase + its team knowledge) as a graph; reason by retrieving over that graph, not by dumping files into a flat window. Living Portraits is the small, visible version of the same idea. [ASIMOV]
- Reflection AI funding is web-verified but is NOT load-bearing for this talk; treat it as optional color, not a slide claim. Confirmed timeline: March 2025 raised $130M (a $25M seed plus a $105M Series A) at about a $545M valuation, led by Lightspeed and Sequoia; October 2025 raised $2B at an $8B valuation (investors include Nvidia, Eric Schmidt, Lightspeed, Sequoia); as of early 2026 reported to be raising a further round near a $20B valuation. [ASIMOV-TC] [ASIMOV-WIKI] [ASIMOV-SACRA]
- UNVERIFIED: any specific Asimov benchmark number, accuracy claim, customer count, or pricing. Do not put those on a slide. The "~70% of engineering time is reading code" figure is the company's own framing, not an independent measurement; attribute it to Reflection if used. State only the qualitative architecture facts above.

---

## D. DNA / palette (graph_viewer.html + intake.yaml)

### [DNA-PALETTE]
- bg near-black `#0d0b10`; panel `#16131c`; ink `#e9e4f0`; muted `#9a90ad`; border `#2a2433`.
- Node triad: gold `#d4a85a` (Phineas), teal `#78ccc2` (Seraphina), magenta `#ff3df0` (MAXX-9). Indigo `#8b6cff` = circadian / bedtime layer.
- Edge color: dashed transition orange `#ff5a3c`. Idle self-loops are solid, tinted to the node.
- These are stolen verbatim from `graph_viewer.html` (the real graph), not menu-shopped. Persona Sagmeister, mood gallery-noir, force vector LIVING / breathing-constellation. [DNA-PALETTE]

---

## Talk identity
- Title: "Context Graphs: How Living Portraits Remember". Tagline: "Memory you can walk. Nodes are moments, edges are how you move between them."
- Speaker: Rayyan Zahid. Venues: J4ME and VCN. The deck is shown in the room. [TALK]
- See the real graph yourself: `graph_viewer.html` in the living-portraits repo. [LP-GRAPH]
