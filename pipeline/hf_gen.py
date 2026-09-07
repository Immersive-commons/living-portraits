"""hf_gen.py -- the Higgsfield generation backend (replaces Midjourney for new poses).

WHY THIS EXISTS: the clip graph stores poses as nodes and transitions as EDGES, and both
endpoints of an edge are stills we already have. Midjourney could only animate FORWARD from
one image and hope it landed near the target, which is why `pipeline/verify.py` carries a
`continuity(clip_end_frame, target_node_frame)` check at all. Higgsfield takes a first AND a
last frame, so an edge is generated as the interpolation it actually is -- landing on the
target node is structural, not lucky.

ROUTING: hil is NOT the Higgsfield session owner -- `node` is, and the session uses
one-time-use refresh tokens, so a second host running `hf` steals and kills the first's
session. Everything here therefore goes through the hf-proxy on node
(`j4me/infra/deploy/hf_proxy/`), which inlines files as base64 and shells to the pinned CLI.
NEVER run `hf` on hil.

MODEL CHOICE (measured 2026-08-09, see ../_bakeoff/README.md):
  * stills  -> gpt_image_2 @ 1k       = 0.5 credits, and 1024x1024 is natively square
  * clips   -> kling3_0 @ 1:1/5s      = 7.5 credits with sound OFF (10 with it on)
`sound=off` is not a preference: the portraits speak through their own piper TTS
(`pipeline/tts.py`), so a baked audio track is dead weight that costs 2.5 credits an edge.

Credits are a MONTHLY pool (3000, granted on the 23rd, and NOT rolled over), so the caps in
autogen.py are the real safety net -- see CLIP_DAILY_CAP / CLIP_CHAR_CAP there.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# model + floor settings, measured not assumed
STILL_MODEL = "gpt_image_2"
CLIP_MODEL = "kling3_0"
CLIP_SECONDS = 5
CREDITS_PER_CLIP = 7.5      # kling3_0, 1:1, 5s, sound off
CREDITS_PER_STILL = 4.0     # gpt_image_2 @ quality=high, 1k -- BILLED, not estimated
                            # (low quality is 0.5; high is worth it here because this still
                            # is the anchor every clip of the pose is generated from, and a
                            # soft anchor makes every downstream edge soft)
# A whole pose is therefore ~41.5 credits: 1 still + forward + reverse + 3 idles.

_TIMEOUT_S = 900


class HFGenError(RuntimeError):
    """A generation failed. Carries `kind` so callers can branch:
    auth / rate_limit / capability / transient."""

    def __init__(self, msg, kind="transient"):
        super().__init__(msg)
        self.kind = kind


# ---------------------------------------------------------------- proxy credentials
def _proxy():
    """(url, token). Resolved env -> ~/.config -> data/mind, mirroring how the z.ai key is
    resolved in director/llm.py. Never committed; data/ is gitignored."""
    url = os.environ.get("LP_HF_PROXY_URL")
    tok = os.environ.get("LP_HF_PROXY_TOKEN")
    if url and tok:
        return url.rstrip("/"), tok
    for cand in (Path.home() / ".config" / "living-portraits" / "hf_proxy.json",
                 ROOT / "data" / "mind" / "hf_proxy.json"):
        try:
            d = json.loads(cand.read_text())
            u, t = d.get("url", "").rstrip("/"), d.get("token", "")
            if u and t:
                return u, t
        except Exception:
            continue
    raise HFGenError(
        "no hf proxy credentials: set LP_HF_PROXY_URL + LP_HF_PROXY_TOKEN, or write "
        "data/mind/hf_proxy.json {\"url\":..., \"token\":...}", kind="auth")


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _run_create(model, args, exts, files, *, timeout_s=_TIMEOUT_S):
    """POST one generation through the proxy. Returns the media URL.

    The proxy REFUSES a file flag whose value is not an inlined-file key (it would
    otherwise let a host path be uploaded off the box), so every file arg is passed as an
    opaque key present in `files`."""
    url, tok = _proxy()
    body = json.dumps({
        "model": model, "args": args, "exts": exts, "files": files,
        "timeout": "%dm" % max(1, timeout_s // 60), "attempts": 1,
    }).encode()
    req = urllib.request.Request(
        url + "/run_create", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s + 60) as r:
            out = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise HFGenError("proxy rejected our token (%s)" % e.code, kind="auth")
        raise HFGenError("proxy HTTP %s" % e.code, kind="transient")
    except Exception as e:
        raise HFGenError("proxy unreachable: %s" % e, kind="transient")
    if not out.get("ok"):
        err = str(out.get("error", ""))
        low = err.lower()
        if "medias" in low or "only contain" in low or "at most 1 item" in low:
            raise HFGenError(err, kind="capability")
        if "rate_limit" in low or "concurrent" in low:
            raise HFGenError(err, kind="rate_limit")
        raise HFGenError(err or "unknown proxy failure", kind="transient")
    return out["url"]


def _download(url, out_path, timeout_s=300):
    """Fetch a generated asset. The URL comes from the PROXY'S RESPONSE, not from us.

    That is a different trust level from the gateway URL in `_proxy()`, which is
    ours and comes from config. urllib dispatches on scheme, and it honours
    `file://` -- so a proxy that is compromised, spoofed, or simply buggy could
    return `file:///etc/passwd` and this function would read it and write it into
    `data/gen/` as a generated still, where the graph would then serve it. Nothing
    downstream re-checks: `_variant_gifs` globs, `video_graph.build()` trusts what
    is on disk, and the viewer renders it.

    So the scheme is checked here rather than assumed. http and https only.
    """
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        # NOT `transient`. autogen re-queues everything except a terminal kind as
        # "approved" (autogen.py:813), and _spend_clip runs only AFTER
        # generate_clip returns (autogen.py:707-709). So a retryable failure here
        # is the worst possible shape: _run_create has already billed Higgsfield
        # 7.5 credits, the local budget never increments, CLIP_DAILY_CAP never
        # trips, and the next run does it again. A proxy handing back a
        # non-http URL will keep doing so; retrying spends money to learn nothing.
        raise HFGenError(
            "proxy returned a %s:// URL; only http/https are fetched" % (scheme or "relative"),
            kind="refused")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout_s) as r, open(tmp, "wb") as f:
        f.write(r.read())
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise HFGenError("downloaded 0 bytes from %s" % url, kind="transient")
    os.replace(tmp, out_path)
    return out_path


# ---------------------------------------------------------------- public API
def generate_still(prompt, out_png, *, ref_png=None, negatives=""):
    """Generate a new pose still. `ref_png` is the character's CANONICAL anchor still --
    always the original, never the most recent generation, or the face drifts a little
    further every time (the identity-drift risk in AUTONOMY.md).

    Returns Path(out_png). ~0.5 credits.
    """
    text = prompt if not negatives else "%s. Avoid: %s" % (prompt, negatives)
    args = ["--prompt", text, "--quality", "high", "--resolution", "1k"]
    files = {}
    if ref_png:
        # the key MUST carry a real extension: the proxy names its temp file from the key's
        # suffix, and hf refuses a ".bin" it cannot type ("Cannot detect media type").
        args += ["--image", "REF.png"]
        files["REF.png"] = _b64(ref_png)
    url = _run_create(STILL_MODEL, args, ["png", "jpg", "webp"], files)
    return _download(url, out_png)


def generate_clip(start_png, end_png, motion_prompt, out_mp4, *, seconds=CLIP_SECONDS):
    """Generate ONE edge clip as a true first->last-frame interpolation.

    An IDLE loop is just the degenerate case where start_png == end_png: the clip departs
    from the pose and returns to it, so it chains seamlessly (the same trick
    `_bake_ltx.py` uses locally).

    sound is forced OFF -- see module docstring. Returns Path(out_mp4). ~7.5 credits.
    """
    # keys carry .png on purpose -- the proxy derives the temp filename from the key's
    # suffix and hf rejects an extensionless ".bin" it cannot type.
    files = {"START.png": _b64(start_png), "END.png": _b64(end_png)}
    args = ["--prompt", motion_prompt,
            "--start-image", "START.png", "--end-image", "END.png",
            "--aspect_ratio", "1:1", "--duration", str(seconds), "--sound", "off"]
    url = _run_create(CLIP_MODEL, args, ["mp4", "webm", "mov"], files)
    return _download(url, out_mp4)


def healthy():
    """(ok, detail) -- is the proxy up AND is the owner session alive? A present
    credentials file with a dead session renders nothing, so check `session` too."""
    try:
        url, tok = _proxy()
    except HFGenError as e:
        return False, str(e)
    req = urllib.request.Request(url + "/healthz", headers={"Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode())
    except Exception as e:
        return False, "proxy unreachable: %s" % e
    if not d.get("ok"):
        return False, "proxy unhealthy: %s" % d
    if d.get("session") is False:
        return False, "HF session is DEAD on the owner host -- re-auth ON NODE"
    return True, "ok"
