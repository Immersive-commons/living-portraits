# living-portraits

Self-aware theatrical portraits on two LED panels, driven by an async generative
pipeline. Nothing runs in realtime: a deterministic player shows pre-baked clips;
a slow stage-manager (qwen3) composes the beats.

## Layout
- `player.py`       frameless dual-panel player (A 256x256 @ 0,0; B 192x192 @ 256,0)
- `run_player.bat`  launcher (interactive scheduled task `lp-player`)
- `prompts/`        `_stage-directives.md` (4th-wall meta) + `characters/*`
- `director/`       `stage_manager.py` (qwen3 -> beat -> data/stage_state.json) + `signals.py`
- `data/`           runtime state: stage_state.json, mood.json, feed.json (gitignored)

## Runs on
supercommons2 (`immer@<producer-host>`): i9-9900K / 64GB / RTX 2080 Ti 11GB.
Shares the box with the inference-engineering catalog. qwen3:8b via local Ollama.

## Status
- [x] Frameless dual-panel player (test card) launched on console session 1
- [x] Prompt library + qwen3 stage-manager (beats -> stage_state.json)
- [ ] Player consumes stage_state.json (clip playback + asides)
- [ ] Generative pipeline (SDXL/Flux + SAM + Live2D + img2vid) + verification gate
- [ ] Self-heal watchdog
