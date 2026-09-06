# Contributing

## Setup

```bash
git clone https://github.com/Immersive-commons/living-portraits.git
cd living-portraits
python -m pip install -r requirements.txt
python -m pytest tests/ -q          # 303 passed, 41 skipped on a bare clone
```

That is the whole setup. Six libraries, no GPU, no hardware, none of the
project's art. If the suite is green you have a working development environment.

To get something to *look at*, seed placeholder media and build a graph:

```bash
python scripts/seed_demo_media.py
python runtime/video_graph.py build          # 20 nodes, 98 edges, walk-safe
python scripts/export_context_view.py
python -m http.server 8000                   # then open /graph_viewer.html
```

`requirements-gen.txt` is the clip-generation stack — torch, diffusers,
InstantID, LivePortrait, ~15 GB of weights and a CUDA toolchain. **You almost
certainly do not need it.** Nothing under `runtime/`, `director/`, `health/`,
`scripts/` or `tests/` imports any of it.

## Before you change anything

Read [AGENTS.md](AGENTS.md). It is written for coding agents but the five rules
in it are the ones a human gets wrong too, and they are not enforced by any test:
single-writer files, the walker's import set stays pure stdlib, walk-safety
refusals are correct, absence is a skip with a reason, detection never repairs.

Then read the "If you read nothing else" list at the top of
[ARCHITECTURE.md](ARCHITECTURE.md).

## What a change has to prove

- **The test count does not go down.** A test that flipped from pass to skip is a
  regression wearing a disguise — say so in the PR if you meant to do it.
- **The graph still builds** (`python runtime/video_graph.py build`, zero
  walk-safety errors) if you touched poses, specs, or the graph itself.
- **Visual changes are probed, not eyeballed.** Compare bounding boxes, assert on
  the DOM. A screenshot can confirm a fix; it cannot falsify one.
- **New behaviour arrives with a test**, and where the behaviour is about
  production reality rather than logic, it belongs in the `test_real_*` family
  that measures against the snapshot instead of a fixture you built to pass.

## How changes land

`main` is protected: **it requires a pull request**, so a direct push will be
rejected even with write access. Branch, push the branch, open a PR.

```bash
git checkout -b what-youre-doing
git push -u origin what-youre-doing
gh pr create --fill
```

## Commits and docs

Commit messages explain **why**, in prose. The CHANGELOG is written the same way
and it retracts its own earlier claims by name when they turn out to have been
wrong — that is a feature of this project, not an accident. If your change makes
a line in a doc or a docstring untrue, fixing that line is part of your change.

Do not add a claim you have not checked.

## Reporting something broken

Open an issue with the command you ran, its full output, and your Python version.
If it is about the graph or the viewer, `python runtime/video_graph.py show` and
the first twenty lines of `data/graph/context_view.json` are usually enough to
tell what shape the world was in.

## Licence

Contributions are accepted under Apache 2.0, matching [LICENSE](LICENSE).
