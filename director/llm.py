"""llm.py -- the portraits' language brain: GLM (Z.ai) via the IC gateway.

Calls the IC->Z.ai gateway (Anthropic Messages API shape) with an IC-minted `agt_`
proxy key. The gateway holds the real Z.ai org key; our key only carries a weekly
token budget + a model allow-list (zero IC tool scopes), so it is safe on hil.

    base : https://immersivecommons13.tail5da903.ts.net   (tailnet; ZAI_GATEWAY_BASE_URL overrides)
    model: glm-4.5-air  (fast + cheap; right for a per-tick decision. glm-4.6 for prose)
    auth : Bearer agt_...   (resolved below, never committed)

Key resolution order (first hit wins):
    1. env ZAI_AGENT_KEY  or  ANTHROPIC_AUTH_TOKEN
    2. ~/.config/living-portraits/zai_key.txt        (out-of-repo stash)
    3. <project>/data/mind/zai_key.txt               (gitignored runtime stash on hil)

stdlib only (urllib) so it adds no dependency to the hil deploy. Non-streaming: the
heartbeat wants one JSON decision, not a token stream.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

try:                                   # optional, fail-open telemetry (no-op if absent/off)
    from director import otel
except Exception:                      # pragma: no cover - import-path fallback
    try:
        import otel
    except Exception:
        otel = None

ROOT = Path(__file__).resolve().parent.parent


def _otel_llm_span(system, model, **attrs):
    """Return a GenAI span context manager, or a no-op CM if otel is unavailable."""
    if otel is not None:
        return otel.llm_span(system, model, **attrs)
    import contextlib

    @contextlib.contextmanager
    def _null():
        class _N:
            def set(self, **k):
                return self

            def response(self, **k):
                return self
        yield _N()
    return _null()

DEFAULT_BASE = "https://immersivecommons13.tail5da903.ts.net"
DEFAULT_MODEL = "glm-5.1"   # GLM 5.2-class (gateway serves glm-5.1 -> glm-5.2). Was glm-4.5-air.
ANTHROPIC_VERSION = "2023-06-01"

_KEY_FILES = (
    Path.home() / ".config" / "living-portraits" / "zai_key.txt",
    ROOT / "data" / "mind" / "zai_key.txt",
)

# --- LOCAL FALLBACK BRAIN: qwen3 via Ollama on hil. When z.ai (the IC gateway / org key) is
# unavailable, the portraits keep thinking on the LOCAL model so an outage never blanks the
# show; it auto-recovers to GLM once z.ai answers again. qwen3:8b verified in-character on hil
# (Phineas/MAXX/Seraphina distinct + valid JSON, ~9-14s warm). Set LP_LLM_FALLBACK=0 to disable.
OLLAMA_URL = os.environ.get("LP_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.environ.get("LP_OLLAMA_MODEL", "qwen3:8b")
OLLAMA_KEEP_ALIVE = os.environ.get("LP_OLLAMA_KEEP_ALIVE", "30m")  # avoid per-call cold model loads
FALLBACK_ENABLED = os.environ.get("LP_LLM_FALLBACK", "1") != "0"

# --- BRAIN INDEPENDENCE: by default the brain is z.ai-first (GLM) with qwen3 as the fallback.
# On the physical hil box we want the personalities to be SELF-CONTAINED so the show never
# depends on the external IC->z.ai gateway (ic13) or IC's org key being alive:
#   LP_LLM_LOCAL_FIRST=1  -> local qwen3 is the PRIMARY brain; z.ai is used ONLY if local fails
#                           (independent in normal operation, keeps a remote safety net).
#   LP_LLM_LOCAL_ONLY=1   -> NEVER calls z.ai; fully air-gapped to hil's qwen3 (purist).
# Both unset = original z.ai-first behaviour (correct for the dev box / when GLM voice is wanted).
LOCAL_FIRST = os.environ.get("LP_LLM_LOCAL_FIRST", "0") == "1"
LOCAL_ONLY = os.environ.get("LP_LLM_LOCAL_ONLY", "0") == "1"
_ZAI_COOLDOWN = 120.0      # after a z.ai failure, go straight to local for this long
_zai_down_until = 0.0      # circuit-breaker timestamp
_zai_was_down = False      # transition flag, for one-line recover/fallback logging


class LLMError(RuntimeError):
    """Any failure talking to the gateway (no key / network / non-200 / bad body)."""


def resolve_key():
    k = os.environ.get("ZAI_AGENT_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if k and k.strip():
        return k.strip()
    for p in _KEY_FILES:
        try:
            t = p.read_text(encoding="utf-8").strip()
            if t:
                return t
        except Exception:
            continue
    return None


def base_url():
    return (os.environ.get("ZAI_GATEWAY_BASE_URL") or DEFAULT_BASE).rstrip("/")


def _zai_complete(system, user, model=DEFAULT_MODEL, max_tokens=600, temperature=1.0,
                  base=None, timeout=60):
    """One non-streaming z.ai (GLM) completion. Returns text; raises LLMError on any failure."""
    key = resolve_key()
    if not key:
        raise LLMError(
            "no Z.ai key found (set $ZAI_AGENT_KEY or write ~/.config/living-portraits/zai_key.txt)")
    url = (base or base_url()) + "/v1/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={
            "content-type": "application/json",
            "authorization": "Bearer " + key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )
    with _otel_llm_span("zai", model, op="chat",
                        **{"gen_ai.request.max_tokens": max_tokens,
                           "gen_ai.request.temperature": temperature,
                           "server.address": (base or base_url())}) as sp:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise LLMError("gateway HTTP %s: %s" % (e.code, body)) from e
        except urllib.error.URLError as e:
            raise LLMError("gateway unreachable: %r" % (e.reason,)) from e
        except Exception as e:
            raise LLMError("gateway call failed: %r" % (e,)) from e

        # Anthropic Messages shape: {"content": [{"type":"text","text": "..."}], ...}
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        text = "".join(parts).strip()
        usage = data.get("usage") or {}
        sp.response(model=data.get("model") or model,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"), text=text)
        if not text:
            raise LLMError("gateway returned no text content: %s" % (json.dumps(data)[:300],))
        return text


def _ollama_complete(system, user, max_tokens=600, temperature=1.0, timeout=60, json_mode=False):
    """One non-streaming LOCAL qwen3 completion via Ollama. `json_mode` asks Ollama for strict
    JSON (reliable for the heartbeat parser). Strips qwen3 <think> blocks. Raises LLMError."""
    body = {
        "model": OLLAMA_MODEL, "stream": False, "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,   # keep the model resident so each call isn't a cold load
        "messages": [{"role": "system", "content": (system or "") + "\n/no_think"},
                     {"role": "user", "content": user}],
        "options": {"temperature": temperature, "num_predict": max(64, int(max_tokens))},
    }
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
                                 method="POST", headers={"content-type": "application/json"})
    with _otel_llm_span("ollama", OLLAMA_MODEL, op="chat",
                        **{"gen_ai.request.temperature": temperature,
                           "server.address": OLLAMA_URL}) as sp:
        try:
            with urllib.request.urlopen(req, timeout=max(timeout, 180)) as r:   # local 8b can be slow
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise LLMError("local ollama unreachable: %r" % (e.reason,)) from e
        except Exception as e:
            raise LLMError("local ollama call failed: %r" % (e,)) from e
        text = re.sub(r"<think>.*?</think>", "", (data.get("message", {}) or {}).get("content", "") or "",
                      flags=re.S).strip()
        sp.response(model=OLLAMA_MODEL,
                    input_tokens=data.get("prompt_eval_count"),
                    output_tokens=data.get("eval_count"), text=text)
        if not text:
            raise LLMError("local ollama returned no content")
        return text


def complete(system, user, model=DEFAULT_MODEL, max_tokens=600, temperature=1.0,
             base=None, timeout=60, json_mode=False):
    """z.ai (GLM) FIRST; on any failure fall back to the LOCAL qwen3 (Ollama) so the portraits
    keep thinking through a gateway/org-key outage. A circuit breaker skips z.ai for a cooldown
    after a failure (no per-call failure latency) and auto-recovers to GLM. LP_LLM_FALLBACK=0
    forces z.ai-only. Returns text; raises LLMError only if BOTH paths fail.

    INDEPENDENT MODES (hil): LP_LLM_LOCAL_FIRST runs the LOCAL qwen3 as the primary brain and
    only reaches z.ai if the local model fails; LP_LLM_LOCAL_ONLY never touches z.ai at all."""
    global _zai_down_until, _zai_was_down

    # Independent-on-hil path: local qwen3 is the brain (no external gateway dependency).
    if LOCAL_ONLY or LOCAL_FIRST:
        try:
            return _ollama_complete(system, user, max_tokens=max_tokens, temperature=temperature,
                                    timeout=timeout, json_mode=json_mode)
        except LLMError:
            if LOCAL_ONLY:
                raise                       # air-gapped: local is the only brain
            return _zai_complete(system, user, model, max_tokens, temperature, base, timeout)

    if not (FALLBACK_ENABLED and time.time() < _zai_down_until):
        try:
            text = _zai_complete(system, user, model, max_tokens, temperature, base, timeout)
            if _zai_was_down:
                print("[llm] z.ai recovered -> back on %s" % model, flush=True)
                _zai_was_down = False
            return text
        except LLMError as e:
            if not FALLBACK_ENABLED:
                raise
            _zai_down_until = time.time() + _ZAI_COOLDOWN
            if not _zai_was_down:
                print("[llm] z.ai down (%s) -> local %s for %ds" % (
                    str(e)[:90], OLLAMA_MODEL, int(_ZAI_COOLDOWN)), flush=True)
                _zai_was_down = True
    return _ollama_complete(system, user, max_tokens=max_tokens, temperature=temperature,
                            timeout=timeout, json_mode=json_mode)


def complete_json(system, user, **kw):
    """complete() + tolerant JSON extraction. Models wrap JSON in prose / ```json fences;
    we grab the outermost {...}. Raises LLMError if nothing parses."""
    kw.setdefault("json_mode", True)   # the local-qwen3 fallback then uses Ollama strict-JSON
    text = complete(system, user, **kw)
    obj = _extract_json(text)
    if obj is None:
        raise LLMError("no JSON object in model reply: %s" % (text[:300],))
    return obj


def _extract_json(text):
    # fast path
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip a ```json ... ``` fence if present
    if "```" in text:
        seg = text.split("```", 2)
        if len(seg) >= 2:
            inner = seg[1]
            if inner.lstrip().lower().startswith("json"):
                inner = inner.split("\n", 1)[1] if "\n" in inner else inner
            try:
                return json.loads(inner.strip())
            except Exception:
                pass
    # outermost balanced {...}
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None
