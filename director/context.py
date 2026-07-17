"""context.py -- one short real-world weather+time line for the heartbeat prompt.

The heartbeat (`director/heartbeat.py`) decides each character's next pose. This
module fetches IC's `GET /api/context` (Frontier Tower SF local weather + time,
Open-Meteo behind the scenes, `context:read` scope) and renders it into ONE short
natural phrase the prompt can drop in, e.g.:

    a chilly 63F overcast afternoon in SF; the sun sets at 8:33pm

Mirrors `director/feeds.py`'s fail-soft discipline EXACTLY -- this can NEVER break a
heartbeat tick:
- `context_line()` swallows ALL failure (no token, network, non-2xx, malformed JSON)
  and returns "" (empty string, never a raise). The caller appends the line only when
  it is non-empty, so a dead source just omits the clause.
- NO network at import time. The heartbeat ticks ~every 4 min in a 24/7 loop; a
  hanging fetch would freeze the show. Timeout is <= NET_TIMEOUT.
- TTL-cached to data/mind/context.json (weather changes hourly): a fresh-cache tick
  returns instantly with zero network. Only a missing/stale cache pays a round trip.

Token resolution mirrors `director/llm.py:resolve_key()` (first hit wins):
    1. env IC_CONTEXT_TOKEN
    2. ~/.config/living-portraits/ic_context_token.txt   (out-of-repo stash)
    3. <project>/data/mind/ic_context_token.txt           (gitignored runtime stash on hil)

stdlib only (urllib) so it adds no dependency to the hil deploy.

Self-test:  python -m director.context   (needs IC_CONTEXT_TOKEN set for a live line)
"""
import datetime
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIND = ROOT / "data" / "mind"
CACHE = MIND / "context.json"
SIGNAL_CACHE = MIND / "context_signal.json"   # cache for context_signal() (separate from the phrase cache)

# Fail-fast: a slow source must never block the ~4-min beat -- let the cache (or an
# omitted clause) cover it instead. Mirrors feeds.py NET_TIMEOUT.
NET_TIMEOUT = 5  # seconds

# Weather changes hourly; the heartbeat ticks every ~4 min, so a 15-min TTL keeps the
# line fresh while almost every tick is served from cache with zero network.
CONTEXT_TTL_SECS = 900  # 15 minutes

# The IC unified app the context token authenticates against. $IC_CONTEXT_URL overrides
# (the local-dev verify path points it at http://localhost:3000).
DEFAULT_BASE = "https://www.immersivecommons.com"

_TOKEN_FILES = (
    Path.home() / ".config" / "living-portraits" / "ic_context_token.txt",  # out-of-repo stash
    ROOT / "data" / "mind" / "ic_context_token.txt",                        # gitignored runtime stash on hil
)


def resolve_context_token():
    """First-hit-wins: env IC_CONTEXT_TOKEN -> ~/.config stash -> data/mind stash.

    Mirrors director/llm.py:resolve_key(). Returns the token string or None.
    """
    k = os.environ.get("IC_CONTEXT_TOKEN")
    if k and k.strip():
        return k.strip()
    for p in _TOKEN_FILES:
        try:
            t = p.read_text(encoding="utf-8").strip()
            if t:
                return t
        except Exception:
            continue
    return None


def _base_url():
    return (os.environ.get("IC_CONTEXT_URL") or DEFAULT_BASE).rstrip("/")


def _fmt_clock(iso_time):
    """'2026-06-17T20:33' (or '... 20:33') -> '8:33pm'. Returns '' on any parse miss."""
    try:
        # Accept both 'YYYY-MM-DDTHH:MM' and 'YYYY-MM-DD HH:MM' tails.
        s = str(iso_time).replace("T", " ")
        hhmm = s[11:16]
        h, m = int(hhmm[:2]), int(hhmm[3:5])
        ampm = "am" if h < 12 else "pm"
        h12 = h % 12 or 12
        return "%d:%02d%s" % (h12, m, ampm)
    except Exception:
        return ""


def _temp_feel(temp_f):
    """A one-word warmth adjective so the line reads naturally. '' if unknown."""
    try:
        t = float(temp_f)
    except Exception:
        return ""
    if t < 50:
        return "cold"
    if t < 60:
        return "chilly"
    if t < 72:
        return "mild"
    if t < 82:
        return "warm"
    return "hot"


def _phrase_from_context(data):
    """Build ONE short natural phrase from the /api/context JSON.

    Target shape: "a chilly 63F overcast afternoon in SF; the sun sets at 8:33pm".
    Degrades gracefully: weather may be null (IC fail-soft), so a time-only line is
    still emitted. Returns '' only if there is nothing useful to say.
    """
    local = data.get("local") or {}
    weather = data.get("weather") or {}
    sun = data.get("sun") or {}

    daypart = str(local.get("daypart") or "").strip()
    temp_f = weather.get("temp_f")
    condition = str(weather.get("condition") or "").strip().lower()

    # --- the weather+daypart clause ---
    clause = ""
    if temp_f is not None or condition or daypart:
        feel = _temp_feel(temp_f) if temp_f is not None else ""
        bits = []
        if feel:
            bits.append(feel)
        if temp_f is not None:
            bits.append("%dF" % round(float(temp_f)))
        if condition:
            bits.append(condition)
        if daypart:
            bits.append(daypart)
        desc = " ".join(bits).strip()
        if desc:
            article = "an" if desc[:1] in "aeiou" else "a"
            clause = "%s %s in SF" % (article, desc)

    # --- the sunset clause ---
    sunset = _fmt_clock(sun.get("sunset")) if sun.get("sunset") else ""
    daylight = sun.get("is_daylight")
    sun_clause = ""
    if sunset:
        # Past sunset reads oddly as "the sun sets at ..."; say "set at" instead.
        verb = "the sun sets at" if daylight is not False else "the sun set at"
        sun_clause = "%s %s" % (verb, sunset)

    if clause and sun_clause:
        return "%s; %s" % (clause, sun_clause)
    return clause or sun_clause


def _cache_fresh(ttl=CONTEXT_TTL_SECS):
    """Return the cached phrase if data/mind/context.json is younger than ttl; else None.

    Fast path -- no network. A cache with no/garbled `fetched_at` is treated as stale.
    """
    try:
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        phrase = cached.get("phrase")
        if not isinstance(phrase, str):
            return None
        stamped = cached.get("fetched_at")
        if not stamped:
            return None
        age = (datetime.datetime.now()
               - datetime.datetime.fromisoformat(stamped)).total_seconds()
        if 0 <= age < ttl:
            return phrase  # may be "" -- a cached empty phrase is still a valid fresh answer
    except Exception:
        pass
    return None


def _cache_write(phrase):
    """Persist phrase + timestamp. Best-effort; failure is non-fatal."""
    try:
        MIND.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(
            json.dumps({
                "phrase": phrase,
                "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }),
            encoding="utf-8",
        )
    except Exception:
        pass


def _fetch_context_json(token, timeout=NET_TIMEOUT):
    """Authenticated GET /api/context -> parsed JSON. Raises on any failure -- caller guards."""
    url = _base_url() + "/api/context"
    req = urllib.request.Request(
        url, method="GET",
        headers={
            "authorization": "Bearer " + token,
            "User-Agent": "living-portraits/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def context_line(force=False):
    """Return ONE short weather+time phrase, TTL-cached. NEVER raises -- "" on any failure.

    The heartbeat appends this line to its prompt only when non-empty (item-2 fail-soft
    rule): no token, dead network, non-2xx, or malformed JSON all degrade to "" and the
    weather clause is simply omitted. Almost every tick is served from the fresh cache.

    Args:
        force: bypass the TTL and force a live refresh.
    """
    try:
        if not force:
            fresh = _cache_fresh()
            if fresh is not None:
                return fresh

        token = resolve_context_token()
        if not token:
            return ""

        data = _fetch_context_json(token)
        if not isinstance(data, dict) or data.get("ok") is False:
            return ""

        phrase = _phrase_from_context(data)
        _cache_write(phrase)
        return phrase
    except Exception:
        return ""


# --- direct weather/time -> movement-energy signal (layered on the LLM mood by the walker) ---
#
# context_line() shapes the BODY only INDIRECTLY: it feeds a phrase into the heartbeat prompt,
# so weather bends movement only when the LLM's chosen mood word happens to echo it. This signal
# is the DIRECT, deterministic path -- the walker reads `energy` and nudges its transition/dwell
# balance every pick, regardless of what the LLM said. Mapping rationale:
#
#   storm/thunder/rain/wind/snow/heavy  -> "high"  agitated, alive: turbulent air = restless paintings
#   clear/sunny/calm/fair/fog/still     -> "low"   settled, serene: still skies = calm paintings
#   deep night (is_daylight False + daypart "night") -> "low"   the quiet middle of the night settles them
#   anything else (overcast, cloudy, unknown)        -> None    neutral: defer entirely to the LLM mood
#
# `high` is checked BEFORE `low`/night, so a stormy night still reads "high" (agitated wins over the
# settling night). None is a real verdict, not a failure -- it means "don't tilt, let the mood lead".
_AGITATING = ("storm", "thunder", "rain", "drizzle", "shower", "wind", "gust",
              "squall", "snow", "hail", "sleet", "blizzard", "heavy")
_SETTLING = ("clear", "sunny", "calm", "fair", "fog", "mist", "haze", "still")


def _energy_from_conditions(condition, is_daylight, daypart):
    """Map raw condition + daylight + daypart -> "high" | "low" | None. See the block above."""
    c = (condition or "").lower()
    dp = (daypart or "").lower()
    if any(w in c for w in _AGITATING):       # turbulent weather wins, even at night
        return "high"
    if any(w in c for w in _SETTLING):        # clear/calm/still skies settle them
        return "low"
    if is_daylight is False and dp == "night":  # the deep, dark middle of the night settles them
        return "low"
    return None                                # neutral -> defer to the LLM mood


def _signal_from_context(data):
    """Build the small movement-energy signal dict from the /api/context JSON.

    Same JSON shape _phrase_from_context() reads (local.daypart, weather.{temp_f,condition},
    sun.is_daylight) -- just distilled to the fields the walker's policy nudge needs. Every
    field degrades to None/"" so a partial payload still yields a usable (possibly energy=None)
    signal rather than a raise.
    """
    local = data.get("local") or {}
    weather = data.get("weather") or {}
    sun = data.get("sun") or {}

    daypart = str(local.get("daypart") or "").strip()
    condition = str(weather.get("condition") or "").strip()
    try:
        temp_f = float(weather.get("temp_f")) if weather.get("temp_f") is not None else None
    except Exception:
        temp_f = None
    is_daylight = sun.get("is_daylight")
    if not isinstance(is_daylight, bool):     # only trust a real bool; null/garbage -> unknown
        is_daylight = None

    return {
        "energy": _energy_from_conditions(condition, is_daylight, daypart),
        "is_daylight": is_daylight,
        "daypart": daypart,
        "temp_f": temp_f,
        "condition": condition.lower(),
    }


def _signal_cache_fresh(ttl=CONTEXT_TTL_SECS):
    """Cached signal dict if data/mind/context_signal.json is younger than ttl; else None.
    Mirrors _cache_fresh() -- fast path, no network, stale/garbled cache -> None."""
    try:
        cached = json.loads(SIGNAL_CACHE.read_text(encoding="utf-8"))
        sig = cached.get("signal")
        if not isinstance(sig, dict):
            return None
        stamped = cached.get("fetched_at")
        if not stamped:
            return None
        age = (datetime.datetime.now()
               - datetime.datetime.fromisoformat(stamped)).total_seconds()
        if 0 <= age < ttl:
            return sig
    except Exception:
        pass
    return None


def _signal_cache_write(signal):
    """Persist the signal dict + timestamp. Best-effort; failure is non-fatal."""
    try:
        MIND.mkdir(parents=True, exist_ok=True)
        SIGNAL_CACHE.write_text(
            json.dumps({
                "signal": signal,
                "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }),
            encoding="utf-8",
        )
    except Exception:
        pass


def context_signal(force=False):
    """Return the weather/time movement-energy signal dict, TTL-cached. NEVER raises.

    Shape: {"energy": "high"|"low"|None, "is_daylight": bool|None, "daypart": str,
            "temp_f": float|None, "condition": str}. Returns None on ANY failure (no token,
    dead network, non-2xx, malformed JSON) so the walker's per-pick call is fail-soft -- a
    dead source just means "no tilt, defer to the LLM mood". Reuses the same fetch/token
    helpers as context_line(); only the cache file differs.

    Args:
        force: bypass the TTL and force a live refresh.
    """
    try:
        if not force:
            fresh = _signal_cache_fresh()
            if fresh is not None:
                return fresh

        token = resolve_context_token()
        if not token:
            return None

        data = _fetch_context_json(token)
        if not isinstance(data, dict) or data.get("ok") is False:
            return None

        signal = _signal_from_context(data)
        _signal_cache_write(signal)
        return signal
    except Exception:
        return None


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    tok = resolve_context_token()
    print("token resolved: %s" % ("yes" if tok else "NO (set $IC_CONTEXT_TOKEN)"))
    print("base url: %s" % _base_url())
    print("\n=== context_line() -- call 1 (refresh or warm cache) ===")
    line = context_line()
    print("   -> %r" % (line,))
    print("cache file: %s  exists=%s" % (CACHE, CACHE.exists()))

    print("\n=== context_line() -- call 2 (should hit fresh-cache fast path) ===")
    fresh = _cache_fresh()
    print("   _cache_fresh() within TTL (%ss)? %s"
          % (CONTEXT_TTL_SECS, "YES -> served without network" if fresh is not None else "no -> would refresh"))
    line2 = context_line()
    print("   -> %r  (identical to call 1? %s)" % (line2, line2 == line))

    print("\n=== context_line(force=True) -- bypasses TTL ===")
    print("   -> %r" % (context_line(force=True),))

    print("\n=== context_signal() -- direct weather/time -> movement energy ===")
    print("   -> %r" % (context_signal(),))
