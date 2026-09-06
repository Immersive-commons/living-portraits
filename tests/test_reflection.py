"""test_reflection.py -- nightly reflection, tested against the REAL production journals.

Two halves:

  * REAL-DATA half -- runs over `data/_realdata/{phineas,maxx}.jsonl`, the 66-day
    production snapshot (11,601 and 13,804 entries). These tests copy the journal into
    tmp_path first and assert on facts computed INDEPENDENTLY from the raw file (a plain
    Counter, not journal_score), so a bug in the scorer cannot make them agree with
    themselves. `pytest.skip` with an explicit reason if the snapshot is absent.
  * UNIT half -- the gates, the parser, and every degrade path, on synthetic journals.

The LLM call is the ONLY thing mocked, ever: `maybe_reflect(complete_json=...)`. Nothing
that feeds the prompt is a stub.

NOTHING here writes to data/_realdata/ or to any real journal. The one test that appends
asserts the source file is byte-identical afterwards.
"""
from __future__ import annotations

import collections
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from director import reflect
from runtime import journal_score as js

REALDATA = ROOT / "data" / "_realdata"
BEDTIME = ROOT / "prompts" / "bedtime_routine.json"
DAY = 86400.0

# A fixed wall-clock "now": 01:00 local, the night after the snapshot's last entry
# (2026-08-10 14:40). Deep inside phineas's sleep window (21:00-05:00) and maxx's
# (00:00-05:00), and its night_id is the evening before.
NOW = datetime(2026, 8, 11, 1, 0).timestamp()


# --------------------------------------------------------------------------- helpers
def _emit(text):
    """print() that survives a cp1252 console (the journals are full of em-dashes)."""
    try:
        print(text)
    except Exception:
        print(str(text).encode("ascii", "replace").decode("ascii"))


def _require_realdata(name):
    src = REALDATA / (name + ".jsonl")
    if not src.exists():
        pytest.skip("real production journal %s is absent -- this suite tests against the "
                    "66-day snapshot, not stubs (see tests/ brief)" % src)
    return src


def _copy_real(name, tmp_path):
    """The real journal, copied into a throwaway journal dir. Returns (dir, copy, src)."""
    src = _require_realdata(name)
    jdir = tmp_path / "journal"
    jdir.mkdir(parents=True, exist_ok=True)
    dst = jdir / (name + ".jsonl")
    shutil.copyfile(src, dst)
    return jdir, dst, src


def _raw_rows(path):
    """The journal parsed WITHOUT journal_score, so assertions are independent of it."""
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def _bedtime_spec():
    if not BEDTIME.exists():
        pytest.skip("prompts/bedtime_routine.json absent -- circadian window unknown")
    return json.loads(BEDTIME.read_text(encoding="utf-8"))


def _fake_llm(reflections, mood="hollowed out", calls=None):
    def _call(system, user, **kw):
        if calls is not None:
            calls.append({"system": system, "user": user, "kw": kw})
        return {"reflections": list(reflections), "mood": mood}
    return _call


def _boom(*a, **kw):
    raise RuntimeError("gateway unreachable")


def _entry(ts, pose, goal, mood="flat", reason="x", band=None):
    e = {"ts": ts, "pose": pose, "goal": goal, "mood": mood, "reason": reason}
    if band:
        e["band"] = band
    return e


def _write(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


# =========================================================================== REAL DATA
def test_real_prompt_carries_the_real_dominant_want_and_its_real_count(tmp_path, capsys):
    """The prompt is built from the REAL journal and must carry REAL facts from it.

    The dominant want and its count are recomputed here with a plain Counter over the raw
    file; the prompt has to contain that exact number. Also prints the whole prompt."""
    jdir, dst, _ = _copy_real("phineas", tmp_path)
    rows = _raw_rows(dst)
    counts = collections.Counter(r.get("goal") for r in rows if r.get("goal"))
    top_goal, top_n = counts.most_common(1)[0]
    distinct = len(counts)

    entries = reflect.load_entries("phineas", jdir)
    assert len(entries) == len(rows) > 10000            # the real snapshot, not a stub

    d = reflect.due("phineas", entries=entries, now=NOW, spec=_bedtime_spec())
    system, user = reflect.build_prompt("phineas", entries, day=d["day"], shift=d["shift"],
                                        now=NOW)

    # the real aggregate: the exact want, the exact count, the exact vocabulary size
    assert top_goal == "phineas:jealous_glare"          # sanity on the snapshot itself
    assert top_goal in user
    assert str(top_n) in user
    assert "%d different wants" % distinct in user
    assert str(len(rows)) in user                       # "N remembered moments"

    # real retrieved monologue: at least one line is verbatim from the real file
    reasons = [(r.get("reason") or "").strip() for r in rows if (r.get("reason") or "").strip()]
    assert any(r in user for r in reasons), "no real journal line reached the prompt"

    # real identity, and the reflection task
    assert "Master Phineas Quill" in system
    assert "faded tragedian" in system
    assert "middle of the night" in system

    with capsys.disabled():
        _emit("\n" + "=" * 78)
        _emit("REAL REFLECTION PROMPT -- phineas, %d journal entries, %d lived today"
              % (len(entries), len(d["day"])))
        _emit("due=%s  why=%s  night=%s" % (d["due"], d["why"], d["night"]))
        _emit("=" * 78 + "\n--- SYSTEM ---")
        _emit(system)
        _emit("\n--- USER ---")
        _emit(user)
        _emit("=" * 78 + "\n")


def test_real_shift_detection_fires_on_real_signal(tmp_path):
    """The GAM-style trigger has to find something real, not fire on noise.

    maxx really did want `analog_brew` fifteen ticks in a row on the snapshot's last day
    and then stop; the run length is recomputed here from the raw file."""
    jdir, dst, _ = _copy_real("maxx", tmp_path)
    entries = reflect.load_entries("maxx", jdir)
    day = reflect.day_window(entries, now=NOW)
    assert len(day) > 50                                # a real day, not a handful

    shift = reflect.detect_shift(day, entries)
    assert shift["shift"] is True
    assert "broken_run" in shift["labels"]

    # independent recount of the longest ENDED run of identical wants in that window
    rows = [r for r in _raw_rows(dst) if NOW - DAY <= r["ts"] <= NOW]
    runs = []
    for g, grp in __import__("itertools").groupby(r.get("goal") for r in rows):
        runs.append((g, len(list(grp))))
    ended = [(g, n) for i, (g, n) in enumerate(runs) if n >= reflect.RUN_MIN and i < len(runs) - 1]
    goal, n = max(ended, key=lambda t: t[1])
    assert ("You wanted %s %d times in a row today, and then you stopped." % (goal, n)
            in shift["lines"])


def test_real_end_to_end_writes_reflections_and_is_idempotent_across_restart(tmp_path):
    """One pass writes; every later pass in the same night is a no-op, INCLUDING one that
    re-reads the journal from disk with no memory of the first (the heartbeat restarts)."""
    jdir, dst, src = _copy_real("phineas", tmp_path)
    before_bytes = src.read_bytes()
    n_before = len(_raw_rows(dst))
    spec = _bedtime_spec()
    calls = []
    llm = _fake_llm(["I have been performing at that idiot for two months and he has never "
                     "once looked over.",
                     "The corridor was never my audience; it was my mirror."],
                    mood="clear-eyed spite", calls=calls)

    r1 = reflect.maybe_reflect("phineas", now=NOW, spec=spec, journal_dir=jdir,
                               complete_json=llm)
    assert r1["ok"] is True and r1["wrote"] == 2, r1
    assert r1["night"] == "2026-08-10"
    assert len(calls) == 1

    rows = _raw_rows(dst)
    assert len(rows) == n_before + 2
    tail = rows[-2:]
    for e in tail:
        assert e["kind"] == "reflection"
        assert e["night"] == "2026-08-10"
        assert "goal" not in e                          # the load-bearing omission
        assert e["reason"].startswith("Looking back: ")
        assert e["pose"] == "phineas:sleep"             # circadian.sleep_node
        # `over` records how many lived moments the conclusion was drawn from
        assert e["over"] == len(reflect.day_window(_raw_rows(dst), now=NOW)) > 50
    assert "performing at that idiot" in tail[0]["reason"]

    # same process, called again
    r2 = reflect.maybe_reflect("phineas", now=NOW, spec=spec, journal_dir=jdir,
                               complete_json=llm)
    assert r2["ok"] is False and r2["wrote"] == 0
    assert "already reflected" in r2["why"]
    assert len(calls) == 1                              # and it never even asked the model

    # a RESTART: fresh load from disk, four hours later, still the same night
    r3 = reflect.maybe_reflect("phineas", now=NOW + 4 * 3600, spec=spec, journal_dir=jdir,
                               complete_json=llm)
    assert r3["wrote"] == 0 and "already reflected" in r3["why"]
    assert len(_raw_rows(dst)) == n_before + 2
    assert len(calls) == 1

    # the NEXT night is a different night, and reflects again
    r4 = reflect.maybe_reflect("phineas", now=NOW + DAY, spec=spec, journal_dir=jdir,
                               complete_json=llm)
    assert r4["night"] == "2026-08-11"

    assert src.read_bytes() == before_bytes             # the irreplaceable original: untouched


def test_real_reflection_reenters_retrieval_and_does_not_pollute_the_aggregate(tmp_path):
    """Park's whole point: a reflection is itself retrievable. And ours must not be
    mistaken for a WANT, or the life-shape line starts counting reflections as desires."""
    jdir, dst, _ = _copy_real("phineas", tmp_path)
    before = reflect.load_entries("phineas", jdir)
    agg_before = js.aggregate_line(before, now=NOW)
    counts_before = js.goal_counts(before)

    reflect.maybe_reflect("phineas", now=NOW, spec=_bedtime_spec(), journal_dir=jdir,
                          complete_json=_fake_llm(["Two thousand glares and not one of them "
                                                   "was ever aimed at HER."]))
    after = reflect.load_entries("phineas", jdir)
    assert len(after) == len(before) + 1

    # 1. selectable by the SAME scorer the waking prompt uses
    picked = js.select(after, now=NOW + 60)
    assert any(reflect.is_reflection(e) for e in picked), \
        "a fresh maximum-importance reflection was not retrieved"

    # 2. renders sensibly in the prompt
    line = [l for l in js.render(picked, now=NOW + 60) if "Two thousand glares" in l]
    assert line and line[0].startswith("- (")
    assert "at phineas:sleep" in line[0] and "Looking back:" in line[0]

    # 3. scores at the top: no goal -> maximum importance
    counts_after = js.goal_counts(after)
    assert counts_after == counts_before                # not counted as a want
    total = sum(counts_after.values())
    r = next(e for e in after if reflect.is_reflection(e))
    assert js.importance(r, counts_after, total) == 1.0

    # 4. the life-shape line is unchanged except for the moment count
    agg_after = js.aggregate_line(after, now=NOW)
    assert agg_after.split("moments", 1)[1] == agg_before.split("moments", 1)[1]

    # 5. it does not invent a pose the character has never stood in
    assert r["pose"] in js.visited_poses(before)


def test_real_llm_failure_writes_absolutely_nothing(tmp_path):
    jdir, dst, _ = _copy_real("maxx", tmp_path)
    before = dst.read_bytes()
    r = reflect.maybe_reflect("maxx", now=NOW, spec=_bedtime_spec(), journal_dir=jdir,
                              complete_json=_boom)
    assert r["ok"] is False and r["wrote"] == 0
    assert "error" in r["why"] and "gateway" in r["why"]
    assert dst.read_bytes() == before


def test_real_dry_run_produces_the_entries_but_writes_nothing(tmp_path):
    jdir, dst, _ = _copy_real("maxx", tmp_path)
    before = dst.read_bytes()
    r = reflect.maybe_reflect("maxx", now=NOW, spec=_bedtime_spec(), journal_dir=jdir,
                              dry_run=True, complete_json=_fake_llm(["The brew was the point."]))
    assert r["ok"] is True and r["wrote"] == 0
    assert r["entries"] and r["entries"][0]["kind"] == "reflection"
    assert dst.read_bytes() == before


def test_real_journal_directory_is_never_the_write_target(tmp_path):
    """Belt and braces: the module's default journal dir is the live one, and every test
    above passes an explicit tmp dir. Assert the two are not the same path."""
    _require_realdata("phineas")
    assert REALDATA.resolve() != Path(reflect.JOURNAL_DIR).resolve()


# =========================================================================== UNIT
def test_night_id_collapses_a_night_that_spans_midnight():
    ev = datetime(2026, 8, 10, 23, 40).timestamp()
    am = datetime(2026, 8, 11, 1, 20).timestamp()
    late = datetime(2026, 8, 11, 4, 59).timestamp()
    assert reflect.night_id(ev) == reflect.night_id(am) == reflect.night_id(late) == "2026-08-10"
    assert reflect.night_id(datetime(2026, 8, 11, 22, 0).timestamp()) == "2026-08-11"
    # midday belongs forward, so a forced daytime pass never collides with last night
    assert reflect.night_id(datetime(2026, 8, 11, 13, 0).timestamp()) == "2026-08-11"


def test_already_reflected_is_read_from_the_journal_not_from_memory(tmp_path):
    jdir = tmp_path / "j"
    path = jdir / "x.jsonl"
    day = [_entry(NOW - 3600 * i, "x:a", "x:b") for i in range(20)]
    _write(path, day)
    assert reflect.already_reflected(reflect.load_entries("x", jdir), "2026-08-10") is False
    reflect.append("x", reflect.make_entries(["Looking back: so."], mood="m",
                                             nid="2026-08-10", pose="x:sleep",
                                             trigger=[], over=20, now=NOW), jdir)
    assert reflect.already_reflected(reflect.load_entries("x", jdir), "2026-08-10") is True
    assert reflect.already_reflected(reflect.load_entries("x", jdir), "2026-08-11") is False


def test_due_gates_in_order(tmp_path):
    spec = {"characters": {"x": {"circadian": {"bedtime_hour": 21, "wake_hour": 5}}}}
    old = [_entry(NOW - 30 * DAY, "x:a", "x:g%d" % (i % 7)) for i in range(5)]  # prior life
    day = [_entry(NOW - 60 * i, "x:a", "x:g%d" % (i % 7)) for i in range(40)][::-1]

    # daytime -> not due, whatever the material says
    d = reflect.due("x", entries=old + day, now=NOW, hour=14, spec=spec)
    assert d["due"] is False and "not night" in d["why"]

    # too little material -> not due
    d = reflect.due("x", entries=old + day[:3], now=NOW, hour=1, spec=spec)
    assert d["due"] is False and "lived moments today" in d["why"]

    # night + material + nothing shifted + never reflected -> due on the floor, not a shift
    d = reflect.due("x", entries=old + day, now=NOW, hour=1, spec=spec)
    assert d["shift"]["shift"] is False
    assert d["due"] is True and d["why"] == "first reflection ever"

    # force bypasses the clock but never the idempotency gate
    d = reflect.due("x", entries=day[:1], now=NOW, hour=14, spec=spec, force=True)
    assert d["due"] is True and d["why"] == "forced"


def test_a_flat_night_is_skipped_but_the_floor_still_fires(tmp_path):
    """GAM says gate on the shift; a character that only reflects on eventful days has no
    record of the flat ones, so MAX_SILENT_NIGHTS forces one through."""
    spec = {"characters": {"x": {"circadian": {"bedtime_hour": 21, "wake_hour": 5}}}}
    # a day with no broken run, no bands, and no first-time pose
    day = [_entry(NOW - 300 * i, "x:a", "x:g%d" % (i % 3)) for i in range(40)][::-1]
    old = [_entry(NOW - 30 * DAY, "x:a", "x:g%d" % (i % 3)) for i in range(5)]

    recent = reflect.make_entries(["Looking back: nothing."], mood="m", nid="2026-08-09",
                                  pose="x:a", trigger=[], over=10, now=NOW - DAY)
    d = reflect.due("x", entries=old + recent + day, now=NOW, hour=1, spec=spec)
    assert d["shift"]["shift"] is False
    assert d["due"] is False and "nothing shifted" in d["why"]

    stale = reflect.make_entries(["Looking back: nothing."], mood="m", nid="2026-08-06",
                                 pose="x:a", trigger=[], over=10,
                                 now=NOW - reflect.MAX_SILENT_NIGHTS * DAY - 60)
    d = reflect.due("x", entries=old + stale + day, now=NOW, hour=1, spec=spec)
    assert d["due"] is True and "nights without reflecting" in d["why"]


def test_a_run_still_in_progress_is_not_a_break():
    day = [_entry(NOW - 100 * i, "x:a", "x:g") for i in range(10)]          # all the same
    assert reflect._broken_run(day) is None
    day = ([_entry(NOW - 1000, "x:a", "x:g")] * 6) + [_entry(NOW, "x:a", "x:other")]
    assert reflect._broken_run(day) == ("x:g", 6)


def test_band_shift_needs_enough_banded_entries():
    early = [_entry(NOW - 3600, "x:a", "x:g", band="fixated") for _ in range(5)]
    late = [_entry(NOW - 60, "x:a", "x:g", band="restless") for _ in range(5)]
    assert reflect._band_shift(early + late) == ("fixated", "restless")
    # the historical journal mostly predates the `band` field: signal drops out, no noise
    assert reflect._band_shift([_entry(NOW, "x:a", "x:g") for _ in range(20)]) is None
    assert reflect._band_shift(early + early) is None                       # no change


def test_parse_reply_is_tolerant_but_refuses_junk():
    t, m = reflect.parse_reply({"reflections": ["one", "two", "three"], "mood": "sour"})
    assert t == ["Looking back: one", "Looking back: two"] and m == "sour"   # capped at 2
    assert reflect.parse_reply(["solo"])[0] == ["Looking back: solo"]
    assert reflect.parse_reply({"reflection": "solo"})[0] == ["Looking back: solo"]
    assert reflect.parse_reply({"reflections": [{"text": "wrapped"}]})[0] == \
        ["Looking back: wrapped"]
    assert reflect.parse_reply({"reflections": ["dupe", "dupe"]})[0] == ["Looking back: dupe"]
    assert reflect.parse_reply({"reflections": ["Looking back: already"]})[0] == \
        ["Looking back: already"]
    for junk in (None, "", 7, {}, {"reflections": []}, {"reflections": "   "}, {"mood": "x"}):
        assert reflect.parse_reply(junk)[0] == []
    long = reflect.parse_reply({"reflections": ["z" * 5000]})[0][0]
    assert len(long) <= reflect.MAX_REFLECTION_CHARS + len(reflect.REFLECTION_LEAD)


def test_a_model_that_says_nothing_usable_writes_nothing(tmp_path):
    jdir = tmp_path / "j"
    _write(jdir / "x.jsonl", [_entry(NOW - 60 * i, "x:a", "x:g%d" % i) for i in range(30)])
    before = (jdir / "x.jsonl").read_bytes()
    r = reflect.maybe_reflect("x", now=NOW, hour=1, journal_dir=jdir, force=True,
                              complete_json=lambda s, u, **k: {"mood": "blank"})
    assert r["ok"] is False and r["wrote"] == 0
    assert r["why"] == "model returned no usable reflection"
    assert (jdir / "x.jsonl").read_bytes() == before


def test_missing_and_malformed_journals_degrade_to_a_skip(tmp_path):
    jdir = tmp_path / "j"
    jdir.mkdir()
    # no journal at all
    r = reflect.maybe_reflect("ghost", now=NOW, hour=1, journal_dir=jdir,
                              complete_json=_fake_llm(["never called"]))
    assert r["ok"] is False and r["wrote"] == 0 and "lived moments" in r["why"]

    # a torn file: half a line, a blank, and an entry with a junk ts
    p = jdir / "x.jsonl"
    good = [json.dumps(_entry(NOW - 60 * i, "x:a", "x:g%d" % i)) for i in range(30)]
    good.insert(4, "{ half writ")
    good.insert(9, "")
    good.insert(14, json.dumps({"ts": "not-a-number", "pose": "x:a", "goal": "x:g"}))
    p.write_text("\n".join(good), encoding="utf-8")
    r = reflect.maybe_reflect("x", now=NOW, hour=1, journal_dir=jdir, force=True,
                              complete_json=_fake_llm(["survived"]))
    assert r["ok"] is True and r["wrote"] == 1

    # an entry that is a bare string, not an object, upstream of everything
    assert reflect.is_reflection("not a dict") is False
    assert reflect.day_window([{"ts": None}, {"no": "ts"}], now=NOW) == []


def test_an_unwritable_journal_directory_never_raises(tmp_path, monkeypatch):
    jdir = tmp_path / "j"
    _write(jdir / "x.jsonl", [_entry(NOW - 60 * i, "x:a", "x:g%d" % i) for i in range(30)])
    monkeypatch.setattr(reflect, "append",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    r = reflect.maybe_reflect("x", now=NOW, hour=1, journal_dir=jdir, force=True,
                              complete_json=_fake_llm(["nope"]))
    assert r["ok"] is False and "error" in r["why"] and r["wrote"] == 0


def test_append_writes_all_lines_or_none(tmp_path):
    p = tmp_path / "j" / "x.jsonl"
    _write(p, [_entry(NOW, "x:a", "x:g")])
    before = p.read_bytes()

    class _Unserialisable:
        pass

    bad = reflect.make_entries(["ok"], mood="m", nid="n", pose="x:a", trigger=[], over=1,
                               now=NOW)
    bad.append({"ts": 1, "kind": "reflection", "boom": _Unserialisable()})
    with pytest.raises(TypeError):
        reflect.append("x", bad, p.parent)
    assert p.read_bytes() == before                      # serialisation fails BEFORE the open
    assert reflect.append("x", [], p.parent) == 0


def test_a_reflection_entry_never_carries_a_goal():
    made = reflect.make_entries(["a", "b"], mood="wry", nid="2026-08-10", pose="x:sleep",
                                trigger=["broken_run"], over=42, now=NOW)
    assert len(made) == 2
    for e in made:
        assert set(e) == {"ts", "kind", "night", "pose", "mood", "reason", "trigger", "over"}
        assert e["kind"] == "reflection" and e["trigger"] == ["broken_run"] and e["over"] == 42
    assert js.goal_counts(made) == collections.Counter()


def test_the_caller_can_supply_the_richer_identity_block(tmp_path):
    """heartbeat holds BASE_SYSTEM + the identity block already; passing it in must win
    over this module's standalone fallback (and avoids an import cycle)."""
    entries = [_entry(NOW - 60 * i, "x:a", "x:g%d" % i) for i in range(30)]
    sysmsg, _ = reflect.build_prompt("x", entries, now=NOW, system="I AM THE WALL.")
    assert sysmsg.startswith("I AM THE WALL.")
    assert "middle of the night" in sysmsg
