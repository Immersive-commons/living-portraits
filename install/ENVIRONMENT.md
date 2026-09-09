# Environment

Every variable the project's own code reads, what it does, and what happens when it
is unset. **28 of them; 21 appeared in no document before this file.**

Nothing here is required to run the test suite, build a graph, drive the walker
headless or open the viewer — the README's four commands need none of it. These matter
when you are pointing the system at a *service*: an LLM, the context feed, the
Higgsfield proxy, a trace collector, or a different host.

`tests/test_environment.py` asserts this file names every variable the code reads, so
a new one cannot land undocumented.

**Secrets are marked.** None of them belong in a committed file — each has a
file-based fallback under `~/.config/living-portraits/` or `data/mind/`, and `data/`
is gitignored.

---

## Inference — `director/llm.py`

The director's brain. Resolution order is env first, then a key file.

| variable | default | what it does |
|---|---|---|
| `ZAI_AGENT_KEY` | — | **SECRET.** Gateway key, checked **first**. Falls back to `~/.config/living-portraits/zai_key.txt`, then `data/mind/zai_key.txt` |
| `ANTHROPIC_AUTH_TOKEN` | — | **SECRET.** Alternate name for the same key, checked second |
| `ZAI_GATEWAY_BASE_URL` | `https://immersivecommons13.tail5da903.ts.net` | Overrides the tailnet gateway |
| `LP_LLM_LOCAL_FIRST` | `0` | `1` tries local Ollama before the gateway |
| `LP_LLM_LOCAL_ONLY` | `0` | `1` never calls the gateway at all |
| `LP_LLM_FALLBACK` | `1` | `0` disables falling back when the first choice fails |
| `LP_OLLAMA_URL` | `http://127.0.0.1:11434/api/chat` | Local inference endpoint |
| `LP_OLLAMA_MODEL` | `qwen3:8b` | Local model |
| `LP_OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama holds the model. Short values mean a cold load per call |

With no key and no key file, `resolve_key()` returns `None` and the caller degrades —
the characters keep walking, they just stop having new opinions.

## Context feed — `director/context.py`

What the characters are told about the weather, the hour, and the building.

| variable | default | what it does |
|---|---|---|
| `IC_CONTEXT_URL` | `https://www.immersivecommons.com` | Context feed base |
| `IC_CONTEXT_TOKEN` | — | **SECRET.** Feed auth |

This path is fail-soft by contract, which is how it went dead for 27 days in 2026-07
while every tick looked healthy. The `world_context` detector in `health/` exists
because of that.

## Generation — `pipeline/`

| variable | default | what it does |
|---|---|---|
| `LP_GEN_BACKEND` | `hf` | `hf` = Higgsfield (production), `mj` = the retired Midjourney path |
| `LP_HF_PROXY_URL` | — | Higgsfield proxy base. Falls back to `~/.config/living-portraits/hf_proxy.json`, then `data/mind/hf_proxy.json` |
| `LP_HF_PROXY_TOKEN` | — | **SECRET.** Proxy token, same fallback chain |
| `HF_HOME` | `~/.cache/huggingface` | Model cache, read by `_probe_caps.py` |

**Both proxy values are needed together** — one alone falls through to the file chain.
Missing credentials raise `HFGenError(kind="auth")` naming both routes.

**A second host running the `hf` backend steals and kills the first's session.** `node`
owns it and the refresh tokens are single-use. This is not a variable to point
somewhere new casually — see #26.

## Telemetry — `director/otel.py`

Fail-open by design; the whole subsystem is optional.

| variable | default | what it does |
|---|---|---|
| `LP_OTEL` | unset | `1` forces on, `0` forces off. **Unset** falls back to the sentinel file `data/mind/otel.enabled` |
| `LP_OTEL_FILE` | — | Write spans to this file instead of exporting |
| `LP_OTEL_CAPTURE_CONTENT` | `0` | `1` records prompt/response bodies in spans. **Off by default on purpose** |
| `OTEL_SERVICE_NAME` | `living-portraits` | Service name on exported spans |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | Standard OTLP endpoint |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | — | Traces-specific endpoint; either turns the exporter on |

`opentelemetry-sdk` is deliberately **not** in `requirements.txt`. Install it and the
suite reports 304/40 instead of a bare clone's 303/41.

## Operator tools — `scripts/`

| variable | default | what it does |
|---|---|---|
| `LP_HOST` | `hil` | Deploy target for `deploy_hil.py` |
| `LP_REMOTE_ROOT` | `C:/living-portraits` | Project root on that host |
| `LP_OSS_REPO` | `<life-parent>/living-portraits-oss` | Where `release.py sync` publishes |

## SDL — the window, and how it is not one

These are `setdefault`, so an existing value in the environment always wins.

| variable | set to | why |
|---|---|---|
| `SDL_VIDEO_WINDOW_POS` | `0,0` | Pins the borderless window at the desktop origin. **The LED sending card grabs sub-rects from a fixed screen region**, so the position is not cosmetic — see #34 |
| `SDL_VIDEO_CENTERED` | `0` | Centering would defeat the above |
| `SDL_VIDEODRIVER` | `dummy` | Set by `stage_render.py` for headless composition. Also the manual override that runs the real walker on any OS: `SDL_VIDEODRIVER=dummy python _preview_graph.py ...` |
| `SDL_AUDIODRIVER` | `dummy` | Same, for audio |

`SDL_VIDEODRIVER=dummy` is what `scripts/live_view.py` uses, and what makes the walker
testable off the Windows host at all.
