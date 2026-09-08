# living-portraits

Two LED panels on a wall, each showing a character who is awake. They breathe,
they move between poses, they go to bed at night and get up in the morning, they
remember what they have been doing, and they occasionally break the fourth wall
about it.

Nothing is generated in realtime. A deterministic player blits pre-baked clips at
10 fps; a slow generation loop, running on its own timer, proposes and mints new
ones. The two loops never touch each other's files — they meet only through
`data/clips/video_graph.json`, and every shared file has exactly one writer. That
constraint is the whole concurrency design, and it is the first thing to
understand before changing anything here.

**Version 0.4.0.** See [CHANGELOG.md](CHANGELOG.md) for what each release added
and, as often, which earlier claim it had to retract.

---

## Quickstart

You do **not** need our art, our GPU, or our hardware. `data/` is gitignored —
the anchor stills and motion clips are hundreds of megabytes of generated media —
so a fresh clone seeds its own placeholders and runs everything from there.

```bash
git clone https://github.com/Immersive-commons/living-portraits.git
cd living-portraits
python -m pip install -r requirements.txt

python -m pytest tests/            # 304 passed, 40 skipped on a bare clone
python scripts/seed_demo_media.py  # placeholder stills + loops, 118 files
python runtime/video_graph.py build        # -> 20 nodes, 98 edges, walk-safe
python scripts/export_context_view.py      # -> data/graph/context_view.json
```

Then serve the repo root and open the viewer — it fetches JSON, so `file://`
will not work:

```bash
python -m http.server 8000
# http://localhost:8000/graph_viewer.html
```

That shows the graph as of the last export. To watch a character actually walk
it, run the walker and the exporter together:

```bash
python scripts/live_view.py          # ctrl-C stops it
```

This runs the real `_preview_graph.py` headless (`SDL_VIDEODRIVER=dummy`) with
the production flags, re-exports every two seconds, and serves the root. Reload
the page and the **Now** strip and **Decision** lens move. You do not need
Windows for this — what is Windows-first is the wall itself, below.

Six lenses over the same graph: **Structure**, **Lived** (visits as heat, dwell
as size), **Provenance** (a scrubber that replays the graph accreting node by
node), **Manner** (typed edges coloured by valence), **Frontier** (poses that are
reachable but have never once been entered), and **Decision** (the live weight on
every exit). Plus Memory, a codebase map checked against disk, and a tab for what
the artifact does not know.

The placeholder clips are flat colour cards that say `PLACEHOLDER` on them. They
are not art and they will not fool anyone — they exist so the graph is walkable
and every downstream surface has something real to show. The seeder never
overwrites a file that already exists, so it is safe to run on a host that has
the real media.

### Python

**3.13 is what the suite is actually run on.** 3.10+ should work — the tree uses
`from __future__ import annotations` throughout and contains no version-specific
syntax or stdlib — but that is an inspection, not a test result. Windows and
Linux both work for everything above; the panel player itself is Windows-first
(see *Running the wall*).

### Skipped tests are expected

40 of them. They fall into two groups, and both are honest skips rather than
hidden failures:

- **`test_real_*`** — these assert against a 66-day production snapshot in
  `data/_realdata/`, which is not in the repo. They exist so that measurements
  come back with numbers nobody chose; on a fresh clone they skip with that
  reason rather than quietly re-running on fixtures.
- **dependency degradation** — anything needing a library you did not install.

---

## Layout

| Path | What lives there |
|---|---|
| `runtime/` | The render loop's world: graph, walk, policy, circadian, lived record, provenance. **Pure stdlib by contract** — the 10 fps loop imports it, so nothing here may pull a heavy dependency. |
| `director/` | Everything with a network or an LLM in it: the stage manager, feeds, reflection, heartbeat, OTel. All of it lives on this side of the line. |
| `pipeline/` | Clip generation. Needs a CUDA GPU and `requirements-gen.txt`. Not needed to develop. |
| `health/` | Nine deterministic detectors for "this installation is fine". |
| `scripts/` | Operator tools: the context-view export, the demo seeder, deploy, backfill. |
| `tests/` | 344 tests. `pytest`, or `python tests/run_all.py` on a box without it. |
| `prompts/` | Character specs (`characters/*.json`), the bedtime routine, stage directives. |
| `graph_viewer.html` | The six-lens viewer. Reads one exported JSON; `scripts/live_view.py` keeps that JSON current while a walk is in progress. |
| `panels.yaml` | Panel geometry and palette. Moving a panel is a config change, not a code change. |

## Where to read next

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the codebase map, written against the
  live host, every claim cited `file.py:line`, with a section listing what was
  not verified. Start with its "If you read nothing else" list.
- **[AGENTS.md](AGENTS.md)** — how to point a coding agent at this repo without
  it breaking the invariants that are not expressible in code.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — setup, the rules that matter, and what
  a change has to prove before it lands.
- **[ROADMAP.md](ROADMAP.md)** — where this is going.
- **[health/README.md](health/README.md)** — what "fine" means, mechanically, and
  why every detector in there is the fossil of an incident a human had to notice.
  Note that `health/` is an *operator* surface: it SSHes to the production host
  rather than inspecting your checkout, so it is worth reading and not worth
  trying to run.

## Running the wall

The panel player is Windows-first: it pins a borderless SDL window to the desktop
origin and an LED sending card grabs sub-rects out of it. `install/` carries the
host bring-up, the scheduled-task definitions, and the watchdog.

You do not need any of that to work on the system. The entire render path is
exercised headless through `clip_player.DummySurface`, which is why the test
suite does not require `pygame` at all.

## Status

- [x] Dual-panel player, graph-driven, self-healing under a watchdog
- [x] Video knowledge graph: poses as nodes, clips as edges, walk-safety enforced at build
- [x] Generation pipeline (anchors + motion clips) and an autonomous propose→mint loop
- [x] Circadian bedtime chain — the characters actually sleep
- [x] Lived record, memory retrieval, and reflection written back by the characters
- [x] Provenance: the graph can answer what a character could do last month
- [x] Health oracle, nine deterministic detectors
- [x] Six-lens context viewer
- [ ] Full-body motion at production quality (the open one — see ROADMAP)

## Licence

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
