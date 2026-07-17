"""
tts.py -- spoken asides for the portraits (free, local TTS + a viseme track).

Given an aside line + a character id, this synthesizes speech to a cached wav
under data/tts/<char>_<hash>.wav and a viseme/timing track to
data/tts/<char>_<hash>.json (a list of {"t": seconds, "viseme": <code>}) so the
player's lip-sync can drive a rig mouth param (Live2D ParamMouthOpenY/Form) or a
LivePortrait mouth keyframe.

The engine is import-guarded. Piper (preferred) is NOT installed on this box, so
the path actually exercised here is the FALLBACK: we still write a plan manifest
(data/tts/<char>_<hash>.plan.json) describing what would be generated, plus a
synthetic even-timed viseme track derived from the text. Downstream lip-sync can
be built and tested against the synthetic track today; once the conductor pip-
installs Piper on SC2, the same synth() call produces a real wav + real timings
and the plan file is no longer written.

    python pipeline/tts.py            # self-test: synth a sample Phineas line

Caching is keyed by sha256(text + voice), so re-running synth() on the same line
is idempotent -- no duplicate audio is generated.
"""
import hashlib
import json
import re
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tts"

# --- engine import guard -----------------------------------------------------
# Prefer Piper (tiny, fast, fully offline, ONNX voices). Kokoro is the richer
# fallback. Neither is installed here, so ENGINE stays None and synth() takes the
# plan-manifest + synthetic-viseme path. No network or model load at import time.
ENGINE = None
try:  # pragma: no cover - exercised only once Piper is installed on SC2
    from piper import PiperVoice  # type: ignore

    ENGINE = "piper"
except Exception:
    try:  # pragma: no cover
        from kokoro import KPipeline  # type: ignore

        ENGINE = "kokoro"
    except Exception:
        ENGINE = None


# --- per-character voice map -------------------------------------------------
# Phineas: a dry patrician baritone -- a low, formal British male voice.
# Seraphina: quick, dry, modern-feeling wit -- a brighter, faster British female.
# `piper` ids are HuggingFace voice slugs (rhasspy/piper-voices); `kokoro` ids are
# Kokoro's built-in voice names. `rate`/`pitch` are advisory shaping hints the
# real engine path can apply (Piper: length_scale=1/rate; Kokoro: speed=rate).
VOICES = {
    "phineas": {
        "piper": "en_GB-alan-medium",        # mature British male, measured
        "kokoro": "bm_george",               # British male
        "rate": 0.92,                        # slower -- grandiloquent, Act-V weight
        "pitch": -2,                         # baritone
        "note": "dry patrician baritone; plays every line as the close of Act V",
    },
    "seraphina": {
        "piper": "en_GB-jenny_dioco-medium",  # bright British female
        "kokoro": "bf_emma",                  # British female
        "rate": 1.08,                         # quicker -- the better lines, faster
        "pitch": +1,
        "note": "quick dry wit; talks to the house like co-conspirators",
    },
}
DEFAULT_VOICE = "phineas"

# Where the .onnx voices live once downloaded (Piper). Kept out of git; the
# conductor seeds these on SC2 (see INSTALL at the bottom of this file).
PIPER_VOICE_DIR = ROOT / "data" / "tts" / "voices"

# --- viseme model ------------------------------------------------------------
# Compact Preston-Blair / Oculus-style viseme set. Each code is a mouth shape the
# rig (or LivePortrait) maps to a (MouthOpenY, MouthForm) pair. "sil" = closed.
# This is the lip-sync vocabulary downstream consumers key on -- keep it stable.
VISEMES = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR",
           "aa", "E", "I", "O", "U"]

# Phoneme-bucket -> viseme. We don't run a real phonemizer in the fallback; we map
# orthographic characters to the bucket their sound usually lands in. Good enough
# to animate a mouth convincingly at ~rig framerate; the real engine path replaces
# this with phoneme timings when available.
_CHAR_VISEME = {
    "a": "aa", "e": "E", "i": "I", "o": "O", "u": "U", "y": "I",
    "p": "PP", "b": "PP", "m": "PP",
    "f": "FF", "v": "FF",
    "t": "DD", "d": "DD", "n": "nn", "l": "nn",
    "k": "kk", "g": "kk", "c": "kk", "q": "kk",
    "s": "SS", "z": "SS", "x": "SS",
    "r": "RR",
    "w": "U", "h": "aa", "j": "CH",
    "th": "TH", "ch": "CH", "sh": "CH",
}
SECS_PER_PHONEME = 0.085  # ~12 mouth shapes/sec -- natural-looking speech cadence


def _hash(text: str, voice: str) -> str:
    return hashlib.sha256((text + "|" + voice).encode("utf-8")).hexdigest()[:16]


def _stem(char: str, text: str, voice: str) -> str:
    return char + "_" + _hash(text, voice)


def _phonemize(text: str):
    """Crude grapheme->viseme-bucket splitter for the synthetic track.

    Lowercases, drops everything but letters/spaces, greedily takes the digraphs
    th/ch/sh first, then single letters. A space becomes a brief "sil" (mouth
    rests between words). Returns a flat list of viseme codes.
    """
    visemes = []
    for word in re.findall(r"[a-z']+|\s+", text.lower()):
        if word.isspace():
            visemes.append("sil")
            continue
        i = 0
        w = word.replace("'", "")
        while i < len(w):
            pair = w[i:i + 2]
            if pair in _CHAR_VISEME:
                visemes.append(_CHAR_VISEME[pair])
                i += 2
                continue
            ch = w[i]
            visemes.append(_CHAR_VISEME.get(ch, "sil"))
            i += 1
        visemes.append("sil")  # close the mouth at the end of each word
    return visemes


def synthetic_visemes(text: str, rate: float = 1.0):
    """Evenly-timed viseme track derived from text (no audio engine needed).

    `rate` < 1 slows the cadence (more seconds/shape); > 1 speeds it. Returns
    [{"t": <start seconds>, "viseme": <code>}, ...], the same shape the real
    engine path will emit so the player's lip-sync code is engine-agnostic.
    """
    step = SECS_PER_PHONEME / max(rate, 0.1)
    track = [{"t": round(i * step, 3), "viseme": v}
             for i, v in enumerate(_phonemize(text))]
    if not track:  # empty/whitespace aside -> a single resting frame
        track = [{"t": 0.0, "viseme": "sil"}]
    track.append({"t": round(len(track) * step, 3), "viseme": "sil"})  # trailing rest
    return track


def _voice_for(char: str):
    spec = VOICES.get(char, VOICES[DEFAULT_VOICE])
    voice_id = spec.get(ENGINE) if ENGINE else (spec.get("piper") or spec.get("kokoro"))
    return spec, voice_id


def _synth_piper(text, voice_id, rate, wav_path):  # pragma: no cover - needs Piper on SC2
    """Real Piper path. Writes wav; returns phoneme/timing -> viseme track.

    Piper exposes per-phoneme audio via PiperVoice.phonemize + synthesize; here we
    keep it simple (full synth to wav) and reuse the synthetic-cadence track scaled
    to the wav's real duration so timings stay honest. Swap in true phoneme
    alignment (eSpeak phoneme stream) when wiring the rig for tighter sync.
    """
    model = PIPER_VOICE_DIR / (voice_id + ".onnx")
    voice = PiperVoice.load(str(model))
    with wave.open(str(wav_path), "wb") as wf:
        voice.synthesize(text, wf, length_scale=1.0 / max(rate, 0.1))
    with wave.open(str(wav_path), "rb") as wf:
        dur = wf.getnframes() / float(wf.getframerate() or 1)
    visemes = _phonemize(text)
    n = max(len(visemes), 1)
    track = [{"t": round(i * dur / n, 3), "viseme": v} for i, v in enumerate(visemes)]
    track.append({"t": round(dur, 3), "viseme": "sil"})
    return track


def _synth_kokoro(text, voice_id, rate, wav_path):  # pragma: no cover - needs Kokoro on SC2
    """Real Kokoro path. Streams 24kHz float chunks; we write a 16-bit PCM wav."""
    import numpy as np  # local import: only on the installed-engine path

    pipe = KPipeline(lang_code="b")  # 'b' = British English
    audio = np.concatenate([chunk.audio for chunk in pipe(text, voice=voice_id, speed=rate)])
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm.tobytes())
    dur = len(pcm) / 24000.0
    visemes = _phonemize(text)
    n = max(len(visemes), 1)
    track = [{"t": round(i * dur / n, 3), "viseme": v} for i, v in enumerate(visemes)]
    track.append({"t": round(dur, 3), "viseme": "sil"})
    return track


def synth(text: str, char: str = DEFAULT_VOICE, force: bool = False):
    """Synthesize an aside to a cached wav + viseme track for one character.

    Returns a dict:
        {"char", "voice", "engine", "text",
         "wav": <path or None>, "visemes": <path>, "plan": <path or None>,
         "track": [...], "cached": bool}

    Idempotent: keyed by sha256(text+voice). If the cache files already exist and
    force is False, nothing is regenerated. When no engine is installed, `wav` is
    None and a `.plan.json` manifest is written instead (so the conductor can see
    exactly what would have been spoken).
    """
    text = (text or "").strip()
    spec, voice_id = _voice_for(char)
    rate = float(spec.get("rate", 1.0))
    stem = _stem(char, text, voice_id or "novoice")

    OUT.mkdir(parents=True, exist_ok=True)
    wav_path = OUT / (stem + ".wav")
    vis_path = OUT / (stem + ".json")
    plan_path = OUT / (stem + ".plan.json")

    have_audio = wav_path.exists() and vis_path.exists()
    have_plan = plan_path.exists() and vis_path.exists()
    if not force and (have_audio or have_plan):
        track = json.loads(vis_path.read_text(encoding="utf-8"))
        return {
            "char": char, "voice": voice_id, "engine": ENGINE, "text": text,
            "wav": str(wav_path) if wav_path.exists() else None,
            "visemes": str(vis_path),
            "plan": str(plan_path) if plan_path.exists() else None,
            "track": track, "cached": True,
        }

    wav_out = None
    plan_out = None
    if ENGINE == "piper" and voice_id:  # pragma: no cover - needs Piper on SC2
        track = _synth_piper(text, voice_id, rate, wav_path)
        wav_out = str(wav_path)
    elif ENGINE == "kokoro" and voice_id:  # pragma: no cover - needs Kokoro on SC2
        track = _synth_kokoro(text, voice_id, rate, wav_path)
        wav_out = str(wav_path)
    else:
        # FALLBACK (the path tested here): no engine installed. Emit a plan
        # manifest + a synthetic, text-derived viseme track so lip-sync is buildable.
        track = synthetic_visemes(text, rate)
        plan = {
            "status": "planned",
            "reason": "no local TTS engine installed (import-guarded)",
            "engine_preferred": "piper",
            "char": char,
            "voice": voice_id,
            "voice_note": spec.get("note", ""),
            "rate": rate,
            "pitch": spec.get("pitch", 0),
            "text": text,
            "would_write": {
                "wav": str(wav_path),
                # Piper voices are 22050 Hz; Kokoro is 24000 Hz. The plan reports
                # the preferred (Piper) rate since that's what synth() will use first.
                "sample_rate": 24000 if ENGINE == "kokoro" else 22050,
            },
            "viseme_track": str(vis_path),
            "frames": len(track),
        }
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        plan_out = str(plan_path)

    vis_path.write_text(json.dumps(track, indent=2), encoding="utf-8")
    return {
        "char": char, "voice": voice_id, "engine": ENGINE, "text": text,
        "wav": wav_out, "visemes": str(vis_path), "plan": plan_out,
        "track": track, "cached": False,
    }


def synth_beat(state: dict, force: bool = False):
    """Convenience: synth both panels of a stage_state.json beat.

    `state` is the dict the stage-manager writes (panels.A / panels.B with .char
    and .aside). Returns {"A": <synth dict>, "B": <synth dict>} keyed by panel.
    The player can call this when a new beat lands and gate playback on the wav
    (or, in fallback, just drive the mouth off the viseme track silently).
    """
    out = {}
    for panel, pdata in state.get("panels", {}).items():
        char = (pdata.get("char") or DEFAULT_VOICE).lower()
        if char not in VOICES:
            char = DEFAULT_VOICE
        out[panel] = synth(pdata.get("aside", ""), char=char, force=force)
    return out


# --- self-test ---------------------------------------------------------------
# Runs synth() on a sample Phineas line and asserts the cache files exist. With no
# engine installed this exercises the .plan.json + synthetic-viseme fallback.
SAMPLE = ("You there, in the cheap seats -- yes, you. Do try to look as though "
          "you understand Marlowe.")

INSTALL = """\
To make these asides audible, the conductor installs ONE engine on SC2 (free, local):

  Piper (preferred -- tiny ONNX voices, fully offline, fast on CPU):
    pip install piper-tts
    # then fetch the two voices into data/tts/voices/ (one-time, ~60-120MB each):
    python -m piper.download_voices en_GB-alan-medium       --data-dir data/tts/voices
    python -m piper.download_voices en_GB-jenny_dioco-medium --data-dir data/tts/voices

  Kokoro (richer prosody, needs torch -- already on SC2 for the gen stack):
    pip install kokoro soundfile
    # voices bm_george / bf_emma download on first use.

No code change is needed after install: synth() detects the engine via the import
guard at the top of this file and produces a real wav + timings instead of a plan.
"""


def _self_test():
    res = synth(SAMPLE, char="phineas", force=True)
    print("engine:", ENGINE or "NONE (fallback: plan + synthetic visemes)")
    print("voice :", res["voice"])
    print("wav   :", res["wav"] or "(none -- engine not installed)")
    print("plan  :", res["plan"] or "(none -- real wav was produced)")
    print("track :", res["visemes"])
    assert Path(res["visemes"]).exists(), "viseme track was not written"
    if res["wav"]:
        assert Path(res["wav"]).exists(), "engine path: wav was not written"
    else:
        assert res["plan"] and Path(res["plan"]).exists(), "fallback: plan manifest missing"
    track = res["track"]
    assert isinstance(track, list) and track, "viseme track is empty"
    assert all({"t", "viseme"} <= set(f) for f in track), "viseme frames malformed"
    assert all(f["viseme"] in VISEMES for f in track), "unknown viseme code emitted"
    print("viseme track length:", len(track), "frames; duration ~",
          round(track[-1]["t"], 2), "s")
    print("first 8 frames:", track[:8])

    # idempotency: a second non-forced call must reuse the cache (cached=True).
    again = synth(SAMPLE, char="phineas")
    assert again["cached"], "second synth() did not hit the cache"
    print("idempotent re-run: cached =", again["cached"])
    print("\n" + INSTALL)


if __name__ == "__main__":
    _self_test()
