# living-portraits -- asset sync (GPU host -> no-GPU host)

`sync_assets.ps1` copies the **GPU-baked assets** from supercommons2 (SC2, the
RTX 2080 Ti box) to a **no-GPU player host** so that host can run the show
without the heavy generative stack. It is the asset half of a two-machine split;
the director-reaches-SC2's-Ollama half is a 2-line code edit documented at the
foot of this file.

Built **locally**, deployed + run **by the conductor**. It is just `ssh`/`scp`
glue -- it does not run on SC2 or on the target itself; it drives both from the
life-repo dev box (jtole), which holds the SSH keys to each host.

## What it syncs

The exact file set the runtime reads (and only that -- the `data\gen\_selftest\`
subdir and any `_test_* / bad_*` debris are NOT synced):

| File (relative to project root) | Produced by | Read at runtime by |
|---|---|---|
| `data\gen\<slug>_portrait.png` | `pipeline/generate.py` | `runtime/stage_render.py` (fallback plate) |
| `data\gen\<slug>_cutout.png` | `pipeline/segment.py` | `player.py` + `stage_render.py` (the figure) |
| `data\gen\<slug>_bg.png` | `pipeline/segment.py` | `stage_render.py` (background plate) |
| `data\gen\<slug>_rig.json` | `pipeline/rig_spec.py` | `runtime/rig_loop.py` (Live2D-style warp) |
| `data\clips\manifest.json` | `runtime/clip_graph.py` | `runtime/clip_player.py` (clip KV-index) |

Slugs are **discovered** from whatever `*_portrait.png` files exist on SC2
(excluding `_selftest`), or pinned explicitly with `-Slugs`.

## Topology -- stage-via-local (default) vs direct

```
                        SC2  (data\gen + data\clips)
                          |  scp pull   (leg 1)
   DEFAULT (staged):      v
                        local staging  ($env:TEMP\lp-asset-sync\)
                          |  scp push   (leg 2)
                          v
                        TARGET  (data\gen + data\clips)
```

Both scp legs originate from the dev box, where key auth to **each** host is
known to work. We do **not** assume SC2 can SSH to the target -- that is a
separate trust relationship the fleet does not generally grant. **Staged is the
safe default and works with the keys we already have.**

`-Direct` attempts a single host-to-host hop (SC2 -> target). Faster (no local
round-trip, no temp space) **but requires SC2 -> target SSH key trust to already
exist**. If that trust is absent it FAILs on the copy -- drop `-Direct` to fall
back to staging.

## The exact commands the conductor runs

Same shape as `install/README.md` and `BRINGUP.md` (dot-source the remote-PS
helper for the verify step, then run the script with `-File`):

```powershell
# 0. (only needed for the verify block below) dot-source the remote-PS helper.
. C:\Users\jtole\Documents\2026\life\scripts\invoke-remote-ps.ps1

# 1. The sync itself -- runs from the life repo / project root. Discovers slugs
#    from SC2, stages via local, pushes to the target. <TARGET> is user@host.
powershell -NoProfile -ExecutionPolicy Bypass `
  -File C:\Users\jtole\Documents\2026\life\projects\living-portraits\install\sync_assets.ps1 `
  -Target <user@target-host>

# Variations:
#   -Slugs phineas,seraphina      pin the slugs (skip SC2 discovery)
#   -Direct                       try SC2 -> target directly (needs SC2->target trust)
#   -DryRun                       list the worklist, copy nothing
#   -Source immer@<producer-host>  override the GPU source host (default = SC2)
#   -SourceProject / -TargetProject   override the per-host project roots
#   -IdentityFile <path>          hand a specific key to scp/ssh
```

The script prints one `OK`/`FAIL` line per file and a final tally, and exits
non-zero if anything failed -- so the conductor's transcript is honest about
what landed. Idempotent: scp overwrites the target every run (last write wins),
so a re-run after a fresh bake on SC2 just refreshes the target.

### Verify on the target (after a sync)

```powershell
. C:\Users\jtole\Documents\2026\life\scripts\invoke-remote-ps.ps1
Invoke-RemotePS -Host_ <user@target-host> -Script @'
$gen = "C:\Users\immer\living-portraits\data\gen"
Get-ChildItem -LiteralPath $gen -File |
  Where-Object { $_.Name -notlike "_*" } |
  Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
Test-Path "C:\Users\immer\living-portraits\data\clips\manifest.json"
'@
```

Expect the per-slug quartet (`_portrait/_cutout/_bg/_rig`) for each character and
`True` for the manifest.

## REMOTE DIRECTOR SWITCH (the 2-line edit -- conductor applies to stage_manager.py)

The no-GPU target has no local qwen3, so its director must call **SC2's** Ollama
over the tailnet. `director/stage_manager.py` currently hardcodes:

```python
OLLAMA = "http://127.0.0.1:11434/api/chat"
```

Make it env-overridable (the **2-line change** -- this is NOT done by
`sync_assets.ps1`):

```python
import os                                                         # near the other imports
OLLAMA = os.environ.get("LP_OLLAMA", "http://127.0.0.1:11434/api/chat")
```

Then on the **target**, point its director at SC2's Ollama by setting the env var
where the director's scheduled task can see it:

```
LP_OLLAMA = http://<producer-host>:11434/api/chat
```

e.g. as a Machine-scope env var:

```powershell
[Environment]::SetEnvironmentVariable(
    'LP_OLLAMA', 'http://<producer-host>:11434/api/chat', 'Machine')
```

With `LP_OLLAMA` **unset** (the SC2 full-local case) the const falls back to
`127.0.0.1` and nothing changes -- so the same code runs on both the GPU host and
the no-GPU host. The `<producer-host>:11434` address is correct for an off-box
client: the supercommons2 memory note records that the Windows-host Ollama is
reachable from another machine at its **tailnet IP**, not localhost.

**Two SC2/ACL-side preconditions** (the conductor verifies out-of-band; the sync
script neither checks nor changes them):

1. SC2's Ollama must listen **beyond loopback** -- set `OLLAMA_HOST=0.0.0.0` on
   SC2 (Ollama binds 127.0.0.1 by default, which an off-box client cannot reach).
2. The **tailnet ACL** must permit `target -> SC2 : 11434`.

## Files

- `sync_assets.ps1` -- the asset sync (this folder). Idempotent; re-runnable.
- `SYNC.md` -- this file.
- (siblings) `install_tasks.ps1` + `watchdog.ps1` (runtime durability),
  `sc2_bringup.ps1` (producer-side gen deps + clip-graph backfill).

## Known gaps / notes

- **Staged is the default for a reason.** `-Direct` needs SC2 -> target SSH key
  trust; staged only needs the dev box's existing keys to each host.
- **No delete / no prune.** Overwrite-only push. A slug RETIRED on SC2 leaves its
  stale copy on the target -- clean that up by hand if a character is dropped.
- **Slug discovery anchors on `*_portrait.png`.** A slug with a portrait but an
  unbaked cutout/bg/rig surfaces those as per-file FAILs (intentional -- it tells
  you the bake is incomplete, rather than silently shipping a half-set).
- **Clip media not yet pulled.** Only `data\clips\manifest.json` syncs. Today the
  clip library is synthetic-placeholder (`asset=None` per `clip_graph.py`), so the
  manifest IS the whole library. When real footage (mp4 / png-sequence dirs) starts
  landing, extend `$relPaths` in the script to also pull each row's non-null
  `asset` path.
- **ASCII only, straight quotes, `--` not em-dashes** throughout the script and
  this file -- they target a Windows-codepage host where smart punctuation would
  break the PS parser (same rule as the other `install/` files).
