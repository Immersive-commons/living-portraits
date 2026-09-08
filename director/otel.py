"""otel.py -- OPTIONAL OpenTelemetry instrumentation for the living portraits' brain.

Soft, fail-open, self-contained by default. If `opentelemetry` is not installed,
or `LP_OTEL` is not "1", every helper here is a no-op and the show runs EXACTLY as
before -- preserving hil's "self-contained, nothing leaves the box" ethos. The
instrumentation can never raise into the brain loop: a telemetry bug must never
blank the panels.

Spans follow the OpenTelemetry GenAI semantic conventions (`gen_ai.*`) so the LLM
brain decisions read as model calls in any OTel backend.

Config (all via env, all optional):
  LP_OTEL=1                          turn instrumentation ON (default OFF = no-op).
                                     LP_OTEL=0 force-OFF (overrides the sentinel).
                                     Instrumentation is also ON if the sentinel file
                                     data/mind/otel.enabled exists -- so on hil you
                                     just `touch` that file + restart lp-mind, no
                                     env-var/scheduled-task surgery needed.
  LP_OTEL_FILE=<path>                local JSONL span sink (default
                                     data/mind/otel_spans.jsonl when LP_OTEL=1;
                                     set to "0" to disable the file sink)
  LP_OTEL_CAPTURE_CONTENT=1          also record prompt/completion text on spans
                                     (default OFF -- spec-aligned; keeps prompts
                                     out of the trace unless you opt in)
  OTEL_EXPORTER_OTLP_ENDPOINT=...    OPT-IN OTLP/HTTP export (e.g. Langfuse Cloud
  OTEL_EXPORTER_OTLP_HEADERS=...     us.cloud.langfuse.com/api/public/otel,
                                     header "Authorization=Basic <b64(pub:secret)>")
                                     -- only wired if an endpoint is set.
  OTEL_SERVICE_NAME=living-portraits service.name on the resource

Only `opentelemetry-api` + `opentelemetry-sdk` are needed for the local file sink;
the OTLP/HTTP exporter package is imported lazily, only when an endpoint is set.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_INITED = False
_TRACER = None
_CAPTURE = os.environ.get("LP_OTEL_CAPTURE_CONTENT", "0") == "1"


def enabled() -> bool:
    """True once init() has stood up a real tracer (LP_OTEL=1 + SDK present)."""
    return _TRACER is not None


def _default_file():
    return str(ROOT / "data" / "mind" / "otel_spans.jsonl")


def _is_on():
    """ON if LP_OTEL=1 OR the sentinel data/mind/otel.enabled exists. LP_OTEL=0 forces OFF."""
    v = os.environ.get("LP_OTEL")
    if v == "1":
        return True
    if v == "0":
        return False
    try:
        return (ROOT / "data" / "mind" / "otel.enabled").exists()
    except Exception:
        return False


class _JsonlSpanExporter:
    """A tiny duck-typed SpanExporter that appends one JSON line per finished span.
    Self-contained local sink -- no collector, nothing leaves the box."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult
        try:
            rows = []
            for s in spans:
                ctx = s.get_span_context()
                start, end = s.start_time, s.end_time
                rows.append(json.dumps({
                    "name": s.name,
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                    "start_ns": start,
                    "end_ns": end,
                    "dur_ms": round((end - start) / 1e6, 2) if (start and end) else None,
                    "status": s.status.status_code.name if s.status else None,
                    "attributes": dict((s.attributes or {}).items()),
                }, default=str))
            with self.path.open("a", encoding="utf-8") as f:
                f.write("\n".join(rows) + "\n")
            return SpanExportResult.SUCCESS
        except Exception:
            # a telemetry write failure must never break the show
            return SpanExportResult.FAILURE

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


def init():
    """Idempotent. Stand up the TracerProvider on first call when LP_OTEL=1 and the
    OTel SDK is importable. Returns a tracer, or None (every helper then no-ops)."""
    global _INITED, _TRACER
    if _INITED:
        return _TRACER
    _INITED = True
    if not _is_on():
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return None
    try:
        resource = Resource.create({
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "living-portraits"),
            "service.namespace": "immersive-commons",
        })
        provider = TracerProvider(resource=resource)

        file_target = os.environ.get("LP_OTEL_FILE")
        if file_target != "0":
            provider.add_span_processor(
                BatchSpanProcessor(_JsonlSpanExporter(file_target or _default_file())))

        # OPT-IN OTLP/HTTP export (Langfuse Cloud / any collector) -- only if an endpoint is set.
        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or \
           os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            except Exception:
                pass  # OTLP exporter not installed -> file sink still works

        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("living-portraits")
        return _TRACER
    except Exception:
        _TRACER = None
        return None


# --------------------------------------------------------------------------- span handles
class _NullSpan:
    """No-op handle returned when telemetry is off. All methods swallow silently."""

    def set(self, **kw):
        return self

    def response(self, **kw):
        return self

    def content(self, prompt=None, completion=None):
        return self

    def error(self, msg=None):
        return self


def _safe_set(span, key, value):
    if value is None:
        return
    try:
        if isinstance(value, (bool, int, float, str)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, str(value))
    except Exception:
        pass


class _SpanHandle:
    def __init__(self, span):
        self._span = span

    def set(self, **kw):
        for k, v in kw.items():
            _safe_set(self._span, k, v)
        return self

    def response(self, model=None, input_tokens=None, output_tokens=None, text=None):
        _safe_set(self._span, "gen_ai.response.model", model)
        _safe_set(self._span, "gen_ai.usage.input_tokens", input_tokens)
        _safe_set(self._span, "gen_ai.usage.output_tokens", output_tokens)
        if text is not None:
            _safe_set(self._span, "gen_ai.response.length_chars", len(text))
            if _CAPTURE:
                _safe_set(self._span, "gen_ai.completion", text[:4000])
        return self

    def content(self, prompt=None, completion=None):
        if _CAPTURE:
            if prompt is not None:
                _safe_set(self._span, "gen_ai.prompt", str(prompt)[:8000])
            if completion is not None:
                _safe_set(self._span, "gen_ai.completion", str(completion)[:4000])
        return self

    def error(self, msg=None):
        try:
            from opentelemetry.trace import Status, StatusCode
            self._span.set_status(Status(StatusCode.ERROR, (str(msg)[:300]) if msg else ""))
        except Exception:
            pass
        return self


def shutdown():
    """Flush + stop the provider (drains the BatchSpanProcessor). Safe to call when
    telemetry is off (no-op). Call before a short-lived process exits if you want the
    last spans written; the long-running brain loop relies on the batch timer instead."""
    if _TRACER is None:
        return
    try:
        from opentelemetry import trace
        prov = trace.get_tracer_provider()
        if hasattr(prov, "shutdown"):
            prov.shutdown()
    except Exception:
        pass


@contextlib.contextmanager
def llm_span(system, model, op="chat", **attrs):
    """A GenAI-convention span around ONE model call. `system` = gen_ai.system
    ('zai' / 'ollama'); each fallback attempt gets its own span, so the z.ai->qwen3
    hop is visible as two spans. On exception, the span is marked ERROR + re-raised."""
    tracer = init()
    if tracer is None:
        yield _NullSpan()
        return
    try:
        from opentelemetry.trace import Status, StatusCode
        with tracer.start_as_current_span("%s %s" % (op, model)) as span:
            _safe_set(span, "gen_ai.system", system)
            _safe_set(span, "gen_ai.operation.name", op)
            _safe_set(span, "gen_ai.request.model", model)
            for k, v in attrs.items():
                _safe_set(span, k, v)
            try:
                yield _SpanHandle(span)
            except BaseException as e:
                try:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)[:300]))
                except Exception:
                    pass
                raise
    except BaseException:
        raise
    finally:
        pass


@contextlib.contextmanager
def span(name, **attrs):
    """A plain span for non-LLM work (a heartbeat tick, a character decision, a graph
    rebuild). Returns a handle; never raises out of the telemetry layer itself."""
    tracer = init()
    if tracer is None:
        yield _NullSpan()
        return
    with tracer.start_as_current_span(name) as sp:
        for k, v in attrs.items():
            _safe_set(sp, k, v)
        yield _SpanHandle(sp)
