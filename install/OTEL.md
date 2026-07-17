# OpenTelemetry for the Living Portraits brain

Optional, fail-open instrumentation of the portraits' LLM brain. **OFF by default** —
if `opentelemetry` isn't installed or it isn't turned on, every hook is a no-op and the
show runs exactly as before. A telemetry bug can never blank the panels.

## What is instrumented

`director/otel.py` is the soft layer. Two call sites use it today:

- **`director/llm.py`** — one **GenAI-convention span per model call**. Because z.ai and
  the qwen3 fallback each get their own span, a gateway outage reads as two spans
  (`chat glm-5.1` ERROR → `chat qwen3:8b` OK). Attributes: `gen_ai.system`
  (`zai`/`ollama`), `gen_ai.request.model`, `gen_ai.response.model`,
  `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.response.length_chars`,
  latency, status.
- **`director/heartbeat.py`** — a `heartbeat.tick` span (graph size, char list, all-night,
  active-goal count) wrapping per-character `heartbeat.decide` spans (chosen
  `lp.goal` / `lp.mood`, from-pose, dwell, rejected/llm-error outcomes).

Prompt/completion **text is NOT captured** unless you opt in with
`LP_OTEL_CAPTURE_CONTENT=1`.

## Turn it on (hil)

1. Install the SDK into the venv that runs `lp-mind` (the brain):

   ```powershell
   & "C:\Users\Immersive Commons 1\.local\bin\uv.exe" pip install `
     --python C:\living-portraits\.venv\Scripts\python.exe opentelemetry-sdk
   ```

   (Only needed for the `lp-mind` venv — the walker/`lp-preview` makes no LLM calls.)

2. Enable + restart the brain (no env-var / scheduled-task surgery needed):

   ```powershell
   New-Item -ItemType File C:\living-portraits\data\mind\otel.enabled -Force
   Stop-ScheduledTask lp-mind; Get-Process pythonw -ErrorAction SilentlyContinue |
     Where-Object { $_.CommandLine -like '*heartbeat.py*' } | Stop-Process -Force
   Start-ScheduledTask lp-mind
   ```

   Spans land at `C:\living-portraits\data\mind\otel_spans.jsonl` (gitignored).

3. Turn it back off: delete `data/mind/otel.enabled` (or set `LP_OTEL=0`) and restart.

## Config (all env, all optional)

| var | default | meaning |
|---|---|---|
| `LP_OTEL` | unset | `1` = on, `0` = force off. Also on if `data/mind/otel.enabled` exists. |
| `LP_OTEL_FILE` | `data/mind/otel_spans.jsonl` | local JSONL sink; `0` disables the file sink |
| `LP_OTEL_CAPTURE_CONTENT` | `0` | also record prompt/completion text on spans |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | **opt-in** OTLP/HTTP export to a collector |
| `OTEL_EXPORTER_OTLP_HEADERS` | unset | auth header for the collector |
| `OTEL_SERVICE_NAME` | `living-portraits` | `service.name` on the resource |

### Optional: ship to Langfuse Cloud (the fleet's LLM-observability backend)

The local file sink stays self-contained. To ALSO export to Langfuse Cloud
(`ic-agent-surface` project), set, in the `lp-mind` process environment:

```
OTEL_EXPORTER_OTLP_ENDPOINT=https://us.cloud.langfuse.com/api/public/otel
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64(public_key:secret_key)>
```

and `uv pip install opentelemetry-exporter-otlp-proto-http` into the venv. Langfuse keys
live in `projects/immersive-commons-unified/.secrets/langfuse.json`. Brain prompts/
decisions then leave the box — leave this OFF for the self-contained default.

## Tests

`tests/test_otel.py` — off-path no-op (in-process) + on-path GenAI-span export
(subprocess, skipped if `opentelemetry-sdk` absent). Part of `tests/run_all.py`.
