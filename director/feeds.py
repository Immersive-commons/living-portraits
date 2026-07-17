"""
feeds.py -- real external "material" for the stage-manager.

signals.py:_feed() used to return a hardcoded list. This module replaces that
with live fetchers (THE SIGNAL headlines, today's calendar, SF weather), each
defensively wrapped so a dead network never stalls the director.

Contract:
- Every fetcher returns list[str] of short lines an actor can riff on.
- Every fetcher swallows ALL failure (timeout, import, parse) -> [].
- NO network at import time. The director runs on supercommons2 every 45s;
  a hanging fetch would freeze the whole stage. Timeouts are <= NET_TIMEOUT.
- get_material() is TTL-cached. The director calls _feed() every 45s 24/7, but
  the underlying sources change slowly (THE SIGNAL weekly, weather hourly,
  calendar daily) -- so we only hit the network when data/feed.json is missing
  or older than FEED_TTL_SECS (default 30 min). On a fresh-cache tick it
  returns instantly with zero network; ~99% of ticks take this path. Pass
  force=True to bypass the TTL (conductor-triggered refresh).
- On a refresh tick it merges + dedupes + caps, caches to data/feed.json with a
  `fetched_at` stamp, and degrades: live -> cache -> the original hardcoded stub.

Self-test:  python -m director.feeds   (or  python feeds.py  from director/)
"""
import datetime
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "feed.json"

# Keep this short. The director loops every 45s; a slow source must fail fast
# and let the cache/stub cover it rather than block the beat.
NET_TIMEOUT = 5  # seconds, per request

# How long cached material is served before we re-fetch. The director ticks
# every 45s but the sources move slowly (SIGNAL weekly / weather hourly /
# calendar daily), so 30 min keeps the feed fresh without hammering the network
# on every beat. Bump down for livelier material, up to fetch less often.
FEED_TTL_SECS = 1800  # 30 minutes

# SF -- the portraits live at Frontier Tower.
_SF_LAT, _SF_LON = 37.7749, -122.4194

# Last-resort material if both the live sources AND the cache are unavailable.
# Mirrors the strings signals.py shipped with so behavior is unchanged on a
# cold box with no network.
STUB = [
    "the humans released yet another frontier model today",
    "a startup three floors down pivoted again",
    "rain is forecast over the city tonight",
]

# Open-Meteo WMO weather codes -> a short phrase. Partial map; unknown codes
# fall through to a generic line so we never emit a bare integer.
_WMO = {
    0: "clear skies",
    1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    95: "a thunderstorm", 96: "a thunderstorm with hail", 99: "a thunderstorm with hail",
}


def _get_json(url, timeout=NET_TIMEOUT):
    """GET a URL, parse JSON. Raises on any failure -- callers guard."""
    req = urllib.request.Request(url, headers={"User-Agent": "living-portraits/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url, timeout=NET_TIMEOUT):
    """GET a URL as text. Raises on any failure -- callers guard."""
    req = urllib.request.Request(url, headers={"User-Agent": "living-portraits/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _clean(s):
    """Collapse whitespace; strip stray markdown/control chars for a clean line."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s.strip("*#_ \t").strip()


def signal_headlines(max_lines=5):
    """Latest THE SIGNAL issue's story headlines.

    THE SIGNAL is a public Immersive Commons publication. We use the plain
    JSON Feed (one GET) to find the newest issue's slug + dek, then pull the
    per-story headlines from that issue's markdown (a second GET). If the
    markdown step fails we still return the issue title + dek from the feed.
    """
    try:
        feed = _get_json("https://www.immersivecommons.com/newsletter/feed.json")
        items = feed.get("items") or []
        if not items:
            return []
        latest = items[0]  # JSON Feed 1.1: newest first
        # slug lives in the item id/url tail, e.g. .../newsletter/week-06
        url = (latest.get("url") or latest.get("id") or "").rstrip("/")
        slug = url.split("/")[-1] if url else ""

        # Fallback line set: the issue title + its one-paragraph dek. Used if
        # the markdown pull below yields nothing.
        fallback = []
        title = _clean(latest.get("title") or "")
        if title:
            fallback.append(f"THE SIGNAL: {title}")
        dek = _clean(latest.get("summary") or latest.get("content_text") or "")
        if dek:
            fallback.append(dek[:200])

        # Preferred: per-story headlines from the issue markdown. Story
        # headlines render as "### NN — Headline." (level-3, numeric prefix).
        # Deeper '#' lines in the .md are receipt-card sub-steps, not stories,
        # so we match level-3 specifically and require the numeric prefix.
        headlines = []
        if slug:
            try:
                md = _get_text(f"https://www.immersivecommons.com/newsletter/{slug}.md")
                # Story headings look like "### 68 · Headline." -- the
                # separator after the number is a MIDDLE DOT (U+00B7) in
                # practice, but accept en/em dash + hyphen too for robustness.
                for m in re.finditer(
                    r"^###\s+\d+\s*[·–—-]\s*(.+)$", md, flags=re.M
                ):
                    line = _clean(m.group(1))
                    if line:
                        headlines.append(line)
            except Exception:
                headlines = []

        chosen = headlines or fallback
        return chosen[:max_lines]
    except Exception:
        return []


def calendar_today(max_lines=3):
    """Today's calendar events as short lines.

    Imports gcal lazily and guards everything: in the director's runtime
    (no TTY, possibly no OAuth token) the client raises rather than hangs,
    and we degrade to []. Never triggers an interactive OAuth flow -- the
    gcal client itself refuses that without a TTY.
    """
    try:
        import sys
        # Ensure the life/ repo root is importable so `gcal` resolves even when
        # the director runs from inside director/ (bare sys.path). Done here,
        # not at module import, to keep import-time side effects nil.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from gcal.client import CalendarClient  # lazy: import may fail on SC2
        cal = CalendarClient()
        events = cal.today()
        lines = []
        for e in events:
            summary = _clean(e.get("summary") or "")
            if not summary:
                continue
            if e.get("all_day"):
                lines.append(f"on the calendar today: {summary}")
            else:
                # start is ISO; grab HH:MM if present.
                start = e.get("start") or ""
                hhmm = start[11:16] if len(start) >= 16 and "T" in start else ""
                when = f" at {hhmm}" if hhmm else ""
                lines.append(f"on the calendar: {summary}{when}")
        return lines[:max_lines]
    except Exception:
        return []


def weather_today():
    """One line of current SF weather from Open-Meteo (no API key)."""
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={_SF_LAT}&longitude={_SF_LON}"
            "&current=temperature_2m,weather_code"
            "&temperature_unit=fahrenheit&timezone=America%2FLos_Angeles"
        )
        data = _get_json(url)
        cur = data.get("current") or {}
        temp = cur.get("temperature_2m")
        code = cur.get("weather_code")
        phrase = _WMO.get(int(code)) if code is not None else None
        if temp is None and not phrase:
            return []
        if phrase and temp is not None:
            return [f"outside it is {phrase}, {round(temp)} degrees over the city"]
        if phrase:
            return [f"outside: {phrase} over the city"]
        return [f"it is {round(temp)} degrees over the city"]
    except Exception:
        return []


def _cache_read():
    """Return cached items list, or None if absent/unreadable."""
    try:
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        items = cached.get("items")
        if isinstance(items, list) and items:
            return items
    except Exception:
        pass
    return None


def _cache_fresh(ttl=FEED_TTL_SECS):
    """Return cached items if the file exists and `fetched_at` is younger than
    ttl seconds; else None. This is the fast path -- no network involved.

    A cache with no/garbled `fetched_at` (e.g. an old-format file, or the
    fallback branch in signals.py that wrote a bare {"items": ...}) is treated
    as stale so the next tick refreshes and re-stamps it.
    """
    try:
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        items = cached.get("items")
        if not (isinstance(items, list) and items):
            return None
        stamped = cached.get("fetched_at")
        if not stamped:
            return None
        age = (datetime.datetime.now()
               - datetime.datetime.fromisoformat(stamped)).total_seconds()
        if 0 <= age < ttl:
            return items
    except Exception:
        pass
    return None


def _cache_write(items):
    """Persist items + timestamp. Best-effort; failure is non-fatal."""
    try:
        DATA.mkdir(exist_ok=True)
        CACHE.write_text(
            json.dumps({
                "items": items,
                "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }),
            encoding="utf-8",
        )
    except Exception:
        pass


def _refresh(max_items):
    """Live fetch all sources -> dedupe -> cap -> cache. Degrade gracefully.

    Degradation ladder (unchanged):
      1. Live fetch (any source that returns lines). Cached on success.
      2. Last good cache file (if every live source came back empty).
      3. The hardcoded STUB (cold box, no network, no prior cache).

    This is the network path -- only called by get_material() on a cache
    miss/stale tick (or force=True). Returns list[str]; never raises.
    """
    lines = []
    for fetch in (signal_headlines, weather_today, calendar_today):
        try:
            lines.extend(fetch() or [])
        except Exception:
            # Belt-and-suspenders: each fetcher already guards, but a bug in
            # one must never sink the whole feed.
            continue

    # Dedupe preserving order (case-insensitive).
    seen = set()
    deduped = []
    for ln in lines:
        ln = _clean(ln)
        key = ln.lower()
        if ln and key not in seen:
            seen.add(key)
            deduped.append(ln)

    if deduped:
        capped = deduped[:max_items]
        _cache_write(capped)
        return capped

    # No live material -> last known good cache, else the stub.
    cached = _cache_read()
    if cached:
        return cached[:max_items]
    return STUB[:max_items]


def get_material(max_items=5, force=False):
    """Return material lines, TTL-cached. Never raises.

    The director calls this every 45s in a 24/7 loop, but the sources move
    slowly. So: if data/feed.json exists and its `fetched_at` is younger than
    FEED_TTL_SECS, return the cached items immediately with NO network. Only on
    a missing/stale cache (or force=True) do we pay for a live refresh.

    Args:
        max_items: cap on the returned list length.
        force: bypass the TTL and force a live refresh (conductor-triggered).

    Returns list[str], length <= max_items.
    """
    if not force:
        fresh = _cache_fresh()
        if fresh is not None:
            return fresh[:max_items]
    return _refresh(max_items)


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=== per-source probe (network may fail here; [] is fine) ===")
    for name, fn in (
        ("signal_headlines", signal_headlines),
        ("weather_today", weather_today),
        ("calendar_today", calendar_today),
    ):
        try:
            out = fn()
        except Exception as exc:  # should never happen -- fetchers guard
            out = f"RAISED (bug): {exc!r}"
        print(f"\n[{name}] ->")
        if isinstance(out, list):
            for line in out:
                print(f"   - {line}")
            if not out:
                print("   (empty -- fell back)")
        else:
            print(f"   {out}")

    print("\n=== get_material(max_items=5) -- call 1 (refresh or warm cache) ===")
    material = get_material(max_items=5)
    for line in material:
        print(f"   - {line}")
    src = "STUB" if material == STUB[:5] else "cache-or-live"
    print(f"({len(material)} lines; source class: {src})")
    print(f"cache file: {CACHE}  exists={CACHE.exists()}")

    print("\n=== get_material() -- call 2 (should hit fresh-cache fast path) ===")
    fresh = _cache_fresh()
    print(f"   _cache_fresh() within TTL ({FEED_TTL_SECS}s)? "
          f"{'YES -> served without network' if fresh is not None else 'no -> would refresh'}")
    material2 = get_material(max_items=5)
    print(f"   returned {len(material2)} lines (identical to call 1? {material2 == material})")

    print("\n=== get_material(force=True) -- bypasses TTL, forces refresh ===")
    forced = get_material(max_items=5, force=True)
    print(f"   returned {len(forced)} lines")
