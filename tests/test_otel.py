"""test_otel.py -- the OPTIONAL telemetry layer must be invisible when off, and emit
GenAI-convention spans to the local JSONL sink when on.

The on-path runs in a SUBPROCESS: an OTel TracerProvider can be set only once per
process, and director.otel.init() caches globally, so both states can't coexist in one
interpreter. The off-path (the default the show runs in) is tested in-process.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_offpath_is_noop_without_env():
    """LP_OTEL unset -> enabled() False, span handles are inert, nothing is written."""
    prev = os.environ.pop("LP_OTEL", None)
    try:
        # fresh import so module globals reflect the env
        for m in ("director.otel", "otel"):
            sys.modules.pop(m, None)
        from director import otel
        assert otel.init() is None
        assert otel.enabled() is False
        with otel.llm_span("zai", "glm-5.1", op="chat") as sp:
            sp.response(model="glm-5.1", input_tokens=10, output_tokens=5, text="hi")
            sp.set(**{"lp.x": 1})
        with otel.span("heartbeat.tick", **{"lp.nodes": 3}) as sp:
            sp.set(**{"lp.all_night": False})
        otel.shutdown()  # safe no-op
    finally:
        if prev is not None:
            os.environ["LP_OTEL"] = prev


def test_onpath_writes_genai_spans():
    """LP_OTEL=1 + a file sink -> finished spans land as JSONL with gen_ai.* attributes."""
    try:
        import opentelemetry.sdk  # noqa: F401
    except Exception:
        import pytest
        pytest.skip("opentelemetry-sdk not installed in this interpreter")

    with tempfile.TemporaryDirectory() as d:
        sink = Path(d) / "spans.jsonl"
        prog = (
            "import sys; sys.path.insert(0, %r)\n"
            "from director import otel\n"
            "assert otel.init() is not None and otel.enabled()\n"
            "with otel.llm_span('zai','glm-5.1',op='chat') as sp:\n"
            "    sp.response(model='glm-5.1', input_tokens=42, output_tokens=7, text='hello')\n"
            "with otel.span('heartbeat.tick', **{'lp.nodes': 20}) as sp:\n"
            "    sp.set(**{'lp.all_night': False})\n"
            "otel.shutdown()\n"
        ) % (str(ROOT),)
        env = dict(os.environ, LP_OTEL="1", LP_OTEL_FILE=str(sink))
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr

        rows = [json.loads(ln) for ln in sink.read_text(encoding="utf-8").splitlines() if ln.strip()]
        names = {row["name"] for row in rows}
        assert any(n.startswith("chat ") for n in names), names
        assert "heartbeat.tick" in names, names
        llm_row = next(row for row in rows if row["name"].startswith("chat "))
        a = llm_row["attributes"]
        assert a.get("gen_ai.system") == "zai"
        assert a.get("gen_ai.request.model") == "glm-5.1"
        assert a.get("gen_ai.usage.input_tokens") == 42
        assert a.get("gen_ai.usage.output_tokens") == 7


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
