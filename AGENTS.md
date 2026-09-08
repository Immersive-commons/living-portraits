# AGENTS.md — working on this repo with a coding agent

For Claude Code, Codex, Cursor, Aider, or any agent you point at this tree. Read
this before the first edit. Humans: everything here is true for you too, it is
just written for something that will happily make 40 correct-looking changes
before anyone notices the invariant it broke.

This file is the short version. [ARCHITECTURE.md](ARCHITECTURE.md) is the long
one, written against the live production host with every claim cited
`file.py:line`.

---

## Orient first

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q                 # expect: 304 passed, 40 skipped
python scripts/seed_demo_media.py          # placeholder media, ~118 files
python runtime/video_graph.py build        # expect: 20 nodes, 98 edges, 0 errors
python scripts/export_context_view.py
```

If that chain is green you have a complete, walkable system with none of the
project's real art. If it is not green, stop and say so — do not begin the task
you were given on top of a broken baseline.

Then read ARCHITECTURE.md's **"If you read nothing else"** list. Ten numbered
facts; they are load-bearing and several of them are counterintuitive.

## The five rules that are not expressible in code

A test suite cannot catch these. They are the ones to hold in your head.

**1. Every shared file has exactly one writer.** That is the entire concurrency
design — two loops run on independent timers with no lock between them, and they
are safe only because no file is ever written from both sides. Adding a second
writer to any file under `data/` is the way to break this system, and it will not
fail immediately or reproducibly. If your change needs to write somewhere another
component already writes, that is a design conversation, not an implementation
detail.

**2. `runtime/` is pure stdlib and must stay import-safe.** The 10 fps render
loop imports it. A network call, an LLM client, or a heavy dependency introduced
anywhere under `runtime/` is a stutter on a physical wall. Everything with a
socket or a model in it lives in `director/`. This line is real and it is not
negotiable for convenience.

**3. Walk-safety is enforced at build, and the build refusing to save is correct
behaviour.** Every pose needs a way out: idle loops plus at least one transition
back toward its hub. `runtime/video_graph.py build` will refuse to write a graph
that would strand or trap the walker. Do not "fix" that error by relaxing the
check — a trapped walker is a character frozen in one pose on a wall in a public
space, and nobody will notice for a week.

**4. Absence is a skip with a reason, never a pass and never an error.** The
`test_real_*` tests measure against a 66-day production snapshot that is not in
this repo. On a bare clone they skip and say why. If you find yourself making one
of them pass by pointing it at a fixture, you have deleted the only tests in the
suite whose numbers nobody chose. Same rule in `health/`: cannot-see is FAIL,
never PASS.

**5. Detection never repairs.** A `health/` detector reports; something else,
under its own policy, acts. Merging the two lets a repair quietly redefine what
"healthy" means.

## Where things are

| You want to change… | Go to | Watch out for |
|---|---|---|
| How a character chooses its next pose | `runtime/policy.py`, `runtime/circadian.py` | Precedence is **circadian > mind (LLM goal) > walk**. Circadian answers first and *returns on a force* — after dark the bedtime chain owns the body and the policy weights are never consulted at all. |
| The graph itself — poses, clips, edges | `runtime/video_graph.py` | `NODE_SPECS` / `EDGE_SPECS` are the source of truth. Edges are **discovered by globbing clips on disk**, so a spec with no media silently produces no edge. |
| What a character remembers | `runtime/lived.py`, `runtime/journal_score.py`, `director/reflect.py` | Journals are append-only and single-writer. |
| The viewer | `graph_viewer.html`, `scripts/export_context_view.py` | The export is **read-only over production by contract**: never call `build()`, never write back into `video_graph.json`, never touch a journal. |
| Watching a walk happen | `scripts/live_view.py` | Runs the real walker headless plus the exporter on a timer. It adds no writer — that is why it is safe to loop. Keep it that way. |
| Clip generation | `pipeline/` | Needs a CUDA GPU and `requirements-gen.txt`. Almost certainly not your task. |
| Panel geometry or colour | `panels.yaml` | Config, not code. Do not hardcode a rect. |

`runtime/clip_graph.py` is an **older parallel v2 graph** used only by the parked
player path. `runtime/video_graph.py` is the live one. They look similar and they
are not interchangeable — check which one your caller actually uses.

## Verifying a change

1. `python -m pytest tests/ -q` — the count must not go down, and a test that
   flipped from pass to skip is a regression wearing a disguise.
2. `python runtime/video_graph.py build` — if you touched the graph, poses, or
   specs. Zero walk-safety errors.
3. `python scripts/export_context_view.py` then reload the viewer — if you
   touched anything the six lenses read.
4. For anything visual, **probe the DOM, do not eyeball a screenshot.** A CSS bug
   in this repo's history stacked a label and its percentage at the same x; it
   was found by comparing two elements' bounding boxes and would not have been
   found by looking. Screenshots confirm; they do not falsify.

## Things that will waste your time

- **`data/` is gitignored and mostly absent.** That is deliberate. Run
  `scripts/seed_demo_media.py` rather than inventing fixtures, so you are working
  against the same graph shape everyone else is.
- **`pipeline/vendor/` is gitignored too** — InstantID and LivePortrait are not
  in this repo. Never run `pip install -r requirements.txt` from inside either of
  them: both pin transformers 4.38, which breaks diffusers and takes the whole
  generation stack down.
- **`health/` is an operator tool, not a contributor tool.** `checks.py` SSHes to
  the production host (`HOST = "hil"`, `health/checks.py:29`) and reports on the
  live installation; `oracle.yaml` additionally hardcodes an absolute interpreter
  path from that host. Without access to `hil` both simply fail to connect —
  nothing here reads your local checkout. Read `health/README.md` for what the
  nine detectors mean; do not expect to run them.
- **The Midjourney generation path is retired but intact.** It has returned HTTP
  403 on every upload since 2026-07-02. Higgsfield replaced it. Do not debug it.
- **`data/clips/video_graph.live.json`**, if you ever see one, is a stale local
  artifact that does not exist in production. Ignore it.

## Writing

Docstrings in this repo say *why*, and several of them exist specifically to stop
someone re-deriving a wrong answer that already cost a day. The CHANGELOG retracts
its own earlier claims by name when they turn out to be false. Match that: if you
find something in a comment or a doc that is not true any more, correcting it is
part of the change, not a separate chore.

Do not add a claim you have not checked. "Should work" belongs in a commit
message, not in a docstring.
