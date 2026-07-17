"""
test_feeds.py -- the director's material feed + TTL cache (director/feeds.py)
and the gather() shape contract (director/signals.py).

FULLY OFFLINE. feeds.py talks to the network via _get_json / _get_text and the
gcal client; we monkeypatch those to deterministic stubs (or force-raise them)
so no socket is ever opened. The module-level CACHE / DATA path constants are
repointed at a per-test tmp dir, so the real data/feed.json is never touched.

Covered:
  * _cache_fresh TTL gate -- fresh cache served with ZERO network; stale / unstamped
    cache ignored.
  * get_material -- fresh-cache fast path (no _refresh call), refresh on stale.
  * _refresh degradation ladder -- live -> last-good cache -> STUB.
  * dedupe + cap.
  * signals.gather() returns {now, part_of_day, mood, material} with material list[str].
"""
from __future__ import annotations

import datetime
import json

import pytest

import feeds


@pytest.fixture
def feeds_tmp(tmp_path, monkeypatch):
    """Repoint feeds.DATA + feeds.CACHE at a tmp dir, and stub all three network
    fetchers to return [] by default so a test that doesn't set up a live source
    never opens a socket. Returns the cache Path for the test to write/inspect."""
    data = tmp_path / "data"
    data.mkdir()
    cache = data / "feed.json"
    monkeypatch.setattr(feeds, "DATA", data)
    monkeypatch.setattr(feeds, "CACHE", cache)
    # Default: every live source empty + every network primitive raises if called,
    # so the only way to get live material is an explicit per-test override.
    monkeypatch.setattr(feeds, "signal_headlines", lambda *a, **k: [])
    monkeypatch.setattr(feeds, "weather_today", lambda *a, **k: [])
    monkeypatch.setattr(feeds, "calendar_today", lambda *a, **k: [])

    def _no_network(*a, **k):
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(feeds, "_get_json", _no_network)
    monkeypatch.setattr(feeds, "_get_text", _no_network)
    return cache


@pytest.fixture
def feeds_paths(tmp_path, monkeypatch):
    """Lighter sibling of feeds_tmp: repoints DATA/CACHE only, WITHOUT stubbing the
    high-level fetchers -- for tests that exercise signal_headlines / weather_today
    themselves (they monkeypatch the low-level _get_json/_get_text directly)."""
    data = tmp_path / "data"
    data.mkdir()
    cache = data / "feed.json"
    monkeypatch.setattr(feeds, "DATA", data)
    monkeypatch.setattr(feeds, "CACHE", cache)
    return cache


def _write_cache(cache, items, *, fetched_at=None, omit_stamp=False):
    payload = {"items": items}
    if not omit_stamp:
        payload["fetched_at"] = (fetched_at or datetime.datetime.now()).isoformat(timespec="seconds")
    cache.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# _cache_fresh: the TTL gate (the fast, no-network path)
# --------------------------------------------------------------------------- #
def test_cache_fresh_returns_recent_items(feeds_tmp):
    _write_cache(feeds_tmp, ["a fresh line", "another"], fetched_at=datetime.datetime.now())
    assert feeds._cache_fresh() == ["a fresh line", "another"]


def test_cache_fresh_none_when_stale(feeds_tmp):
    old = datetime.datetime.now() - datetime.timedelta(seconds=feeds.FEED_TTL_SECS + 60)
    _write_cache(feeds_tmp, ["stale line"], fetched_at=old)
    assert feeds._cache_fresh() is None


def test_cache_fresh_none_when_unstamped(feeds_tmp):
    # An old-format / fallback-written cache with no fetched_at is treated as stale.
    _write_cache(feeds_tmp, ["unstamped"], omit_stamp=True)
    assert feeds._cache_fresh() is None


def test_cache_fresh_none_when_absent(feeds_tmp):
    assert feeds._cache_fresh() is None  # file does not exist yet


def test_cache_fresh_none_when_empty_items(feeds_tmp):
    _write_cache(feeds_tmp, [], fetched_at=datetime.datetime.now())
    assert feeds._cache_fresh() is None


# --------------------------------------------------------------------------- #
# get_material: fast path vs refresh
# --------------------------------------------------------------------------- #
def test_get_material_uses_fresh_cache_without_refresh(feeds_tmp, monkeypatch):
    _write_cache(feeds_tmp, ["cached one", "cached two"], fetched_at=datetime.datetime.now())
    # _refresh must NOT be called on a fresh-cache tick.
    monkeypatch.setattr(feeds, "_refresh",
                        lambda *a, **k: pytest.fail("_refresh called on a fresh-cache tick"))
    out = feeds.get_material(max_items=5)
    assert out == ["cached one", "cached two"]


def test_get_material_caps_fresh_results(feeds_tmp):
    _write_cache(feeds_tmp, [f"line {i}" for i in range(10)], fetched_at=datetime.datetime.now())
    out = feeds.get_material(max_items=3)
    assert out == ["line 0", "line 1", "line 2"]


def test_get_material_force_bypasses_fresh_cache(feeds_tmp, monkeypatch):
    _write_cache(feeds_tmp, ["cached"], fetched_at=datetime.datetime.now())
    called = {"n": 0}

    def _fake_refresh(max_items):
        called["n"] += 1
        return ["refreshed"]

    monkeypatch.setattr(feeds, "_refresh", _fake_refresh)
    out = feeds.get_material(max_items=5, force=True)
    assert out == ["refreshed"] and called["n"] == 1


def test_get_material_refreshes_when_stale(feeds_tmp, monkeypatch):
    old = datetime.datetime.now() - datetime.timedelta(seconds=feeds.FEED_TTL_SECS + 60)
    _write_cache(feeds_tmp, ["stale"], fetched_at=old)
    monkeypatch.setattr(feeds, "signal_headlines", lambda *a, **k: ["a live headline"])
    out = feeds.get_material(max_items=5)
    assert "a live headline" in out


# --------------------------------------------------------------------------- #
# _refresh: the degradation ladder (live -> cache -> STUB) + dedupe + cap
# --------------------------------------------------------------------------- #
def test_refresh_live_merges_and_caches(feeds_tmp, monkeypatch):
    monkeypatch.setattr(feeds, "signal_headlines", lambda *a, **k: ["headline one"])
    monkeypatch.setattr(feeds, "weather_today", lambda *a, **k: ["it is 60 degrees"])
    out = feeds._refresh(max_items=5)
    assert "headline one" in out and "it is 60 degrees" in out
    # cached on success, stamped (so the next tick is a fast-path hit)
    cached = json.loads(feeds_tmp.read_text(encoding="utf-8"))
    assert "fetched_at" in cached and set(out).issubset(set(cached["items"]))


def test_refresh_dedupes_case_insensitive(feeds_tmp, monkeypatch):
    monkeypatch.setattr(feeds, "signal_headlines", lambda *a, **k: ["Same Line", "same line"])
    monkeypatch.setattr(feeds, "weather_today", lambda *a, **k: ["unique"])
    out = feeds._refresh(max_items=5)
    # "Same Line" / "same line" collapse to one entry (first spelling kept)
    lowered = [s.lower() for s in out]
    assert lowered.count("same line") == 1


def test_refresh_caps_to_max_items(feeds_tmp, monkeypatch):
    monkeypatch.setattr(feeds, "signal_headlines",
                        lambda *a, **k: [f"h{i}" for i in range(20)])
    out = feeds._refresh(max_items=4)
    assert len(out) == 4


def test_refresh_falls_back_to_last_good_cache(feeds_tmp):
    # All live sources empty (the fixture default) BUT a prior cache exists ->
    # _refresh serves the last-known-good cache rather than the STUB.
    _write_cache(feeds_tmp, ["last good one", "last good two"],
                 fetched_at=datetime.datetime.now())
    out = feeds._refresh(max_items=5)
    assert out == ["last good one", "last good two"]


def test_refresh_falls_back_to_stub_on_cold_box(feeds_tmp):
    # No live material AND no cache file -> the hardcoded STUB.
    out = feeds._refresh(max_items=5)
    assert out == feeds.STUB[:5]
    assert all(isinstance(s, str) for s in out)


def test_refresh_survives_a_throwing_fetcher(feeds_tmp, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("source exploded")

    monkeypatch.setattr(feeds, "signal_headlines", _boom)
    monkeypatch.setattr(feeds, "weather_today", lambda *a, **k: ["survivor line"])
    out = feeds._refresh(max_items=5)
    assert "survivor line" in out, "a throwing fetcher sank the whole feed"


# --------------------------------------------------------------------------- #
# signal_headlines / weather_today parse correctly off a stubbed network
# --------------------------------------------------------------------------- #
def test_signal_headlines_parses_story_headings(feeds_paths, monkeypatch):
    feed_json = {"items": [{"title": "Week 06", "url": "https://x/newsletter/week-06",
                            "summary": "the dek"}]}
    md = ("# THE SIGNAL\n\n"
          "### 68 · First story headline.\n"
          "body body\n"
          "### 69 · Second story headline.\n"
          "#### a receipt sub-step (should NOT match)\n")
    monkeypatch.setattr(feeds, "_get_json", lambda *a, **k: feed_json)
    monkeypatch.setattr(feeds, "_get_text", lambda *a, **k: md)
    out = feeds.signal_headlines(max_lines=5)
    assert "First story headline." in out
    assert "Second story headline." in out
    assert not any("receipt" in line for line in out), "level-4 sub-step leaked in as a story"


def test_signal_headlines_falls_back_to_title_dek(feeds_paths, monkeypatch):
    feed_json = {"items": [{"title": "Week 07", "url": "https://x/newsletter/week-07",
                            "summary": "the one-paragraph dek"}]}
    monkeypatch.setattr(feeds, "_get_json", lambda *a, **k: feed_json)

    def _md_fails(*a, **k):
        raise RuntimeError("md 404")

    monkeypatch.setattr(feeds, "_get_text", _md_fails)
    out = feeds.signal_headlines(max_lines=5)
    assert any("Week 07" in line for line in out), "lost the issue-title fallback"


def test_signal_headlines_empty_on_total_failure(feeds_paths, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("feed down")

    monkeypatch.setattr(feeds, "_get_json", _boom)
    assert feeds.signal_headlines() == []


def test_weather_today_formats_phrase_and_temp(feeds_paths, monkeypatch):
    monkeypatch.setattr(feeds, "_get_json",
                        lambda *a, **k: {"current": {"temperature_2m": 61.4, "weather_code": 3}})
    out = feeds.weather_today()
    assert len(out) == 1
    assert "overcast" in out[0] and "61" in out[0]  # code 3 -> "overcast", temp rounded


def test_weather_today_empty_on_failure(feeds_paths, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("weather down")

    monkeypatch.setattr(feeds, "_get_json", _boom)
    assert feeds.weather_today() == []


# --------------------------------------------------------------------------- #
# signals.gather(): the shape the director consumes
# --------------------------------------------------------------------------- #
def test_gather_shape(tmp_path, monkeypatch):
    """gather() returns {now, part_of_day, mood, material} with material a list[str].
    Repoint signals.DATA at a tmp dir and stub _feed so no network / real file is hit."""
    import signals
    sdata = tmp_path / "sdata"
    sdata.mkdir()
    monkeypatch.setattr(signals, "DATA", sdata)
    monkeypatch.setattr(signals, "_feed", lambda: ["stub material line"])

    result = signals.gather()
    assert set(result.keys()) == {"now", "part_of_day", "mood", "material"}
    assert isinstance(result["now"], str) and result["now"]
    assert isinstance(result["part_of_day"], str) and result["part_of_day"]
    assert isinstance(result["mood"], str) and result["mood"] in signals.MOODS
    assert isinstance(result["material"], list)
    assert all(isinstance(m, str) for m in result["material"])
    assert result["material"] == ["stub material line"]


def test_gather_material_falls_through_feeds(tmp_path, monkeypatch):
    """gather()'s real _feed() pulls feeds.get_material; with feeds stubbed to a
    fresh cache, gather()'s material is that cache -- end-to-end signals->feeds, offline."""
    import signals
    sdata = tmp_path / "sdata"
    sdata.mkdir()
    monkeypatch.setattr(signals, "DATA", sdata)
    # point feeds at a fresh cache so _feed() -> get_material() takes the no-network path
    fdata = tmp_path / "fdata"
    fdata.mkdir()
    fcache = fdata / "feed.json"
    monkeypatch.setattr(feeds, "DATA", fdata)
    monkeypatch.setattr(feeds, "CACHE", fcache)
    _write_cache(fcache, ["material via feeds"], fetched_at=datetime.datetime.now())

    result = signals.gather()
    assert "material via feeds" in result["material"]


def test_part_of_day_buckets():
    import signals
    assert signals.part_of_day(3) == "the small hours"
    assert signals.part_of_day(9) == "morning"
    assert signals.part_of_day(14) == "afternoon"
    assert signals.part_of_day(19) == "evening"
    assert signals.part_of_day(23) == "night"
