"""reflect.py -- NIGHTLY REFLECTION: the character draws a conclusion about its own life.

Sixty-six days of journal and no arc. Every entry is a reaction to the moment it was
made; nothing in the system has ever looked at a stretch of them and said something
*about* it. Generative Agents (arXiv 2304.03442) calls the missing piece a REFLECTION --
a higher-level inference synthesised from retrieved memories, which is then written back
into memory so it can itself be retrieved later. That feedback loop is what let Park's
agents hold opinions instead of only producing responses.

WHERE IT RUNS. `circadian.py` parks each character at a sleep pose from bedtime to wake
and the heartbeat idles at NIGHT_INTERVAL = 900s, writing no journal entries at all
(`decide_character` returns early on night). That window is free compute and an empty
writer -- the only thing appending to the journal at night is this module.

WHAT IT WRITES. One or two entries into the SAME `data/mind/journal/<char>.jsonl`,
discriminated by `kind: "reflection"`, deliberately WITHOUT a `goal` field. That single
omission is what makes them re-enter retrieval correctly for free, with no change to
`runtime/journal_score.py`:

    goal_counts()  skips entries with no goal   -> a reflection is never counted as a WANT,
                                                   so `aggregate_line` stays truthful
    importance()   counts.get(None, 0) == 0     -> scores 1.0, the maximum
    relevance()    reads `pose`, skips `goal`   -> hops from the sleep pose still apply
    render()       reads pose/mood/reason       -> renders as a memory, in the same voice

So a reflection is a maximum-importance memory that decays by recency like any other and
competes for the same prompt budget. That is Park's design, reached by leaving a field out.

TRIGGER. Night is the compute window, not the reason. GAM (arXiv 2604.12285) promotes
episodic events into consolidated memory on a detected SEMANTIC SHIFT rather than on a
clock, and that is the better trigger here too: reflecting on a day where nothing changed
produces a paraphrase of yesterday. `detect_shift()` looks for three shifts this codebase
can see exactly and for free -- a broken repetition run of wants, a change in the dominant
mood band, and a pose stood in for the first time ever. A floor (MAX_SILENT_NIGHTS) still
forces a reflection every few nights, because a character that reflects only on eventful
days has no record of the flat ones, and the flat ones are most of a life.

IDEMPOTENCY. Derived from the journal, never from process memory: each reflection carries
`night` (the id of the night it belongs to, see `night_id`), and a pass refuses to run if
the journal already holds one for tonight. The heartbeat restarts often and may tick a
dozen times inside one sleep window; all of those are no-ops after the first.

FAILURE. This is a decoration on a render loop. Every failure mode -- LLM down, journal
missing, malformed reply, unwritable path -- resolves to "no reflection this pass" and a
result dict, never an exception reaching the caller and never a partial write (both lines
are validated in full and appended in a single write call).

    from director import reflect
    r = reflect.maybe_reflect("phineas", hour=2, spec=bedtime_spec, log=log)
    # {"ok": True, "wrote": 2, "night": "2026-08-10", "trigger": ["broken_run"], ...}
"""
from __future__ import annotations

import datetime
import itertools
import json
import time
from collections import Counter
from pathlib import Path

from runtime import circadian, journal_score

ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DIR = ROOT / "data" / "mind" / "journal"
CHARS_DIR = ROOT / "prompts" / "characters"

REFLECTION_KIND = "reflection"

# --- tunables (module constants so the tests pin behaviour, not magic numbers).
DAY_SPAN = 86400.0         # how far back "today" reaches when gathering material
MIN_DAY_ENTRIES = 8        # below this there is nothing to draw a conclusion FROM
RUN_MIN = 4                # consecutive identical wants that count as a repetition RUN
BAND_MIN = 4               # banded entries needed per half-day before a band shift is claimed
MAX_SILENT_NIGHTS = 3      # ...but reflect anyway if it has been this many nights
MAX_REFLECTIONS = 2        # lines appended per night, at most
DAY_TOKENS = 900           # budget for the retrieved day fed to the model
PRIOR_REFLECTIONS = 3      # previous reflections shown, so tonight's builds on them
MAX_REFLECTION_CHARS = 600 # a reflection longer than this is a monologue, not a conclusion
REFLECTION_LEAD = "Looking back: "   # marks the line as a conclusion when render() shows it

REFLECT_SYSTEM_TASK = """\

## What you are doing now
It is the middle of the night. You are asleep on the wall, and nobody is watching. This is \
the one moment you are not performing for anyone.

Look back over the day you have just lived and the life you have led here, and say what you \
have come to UNDERSTAND from it -- not what happened, but what it MEANS about you, about \
this wall, about the ones who hang beside you, or about the people who walk past. Draw a \
conclusion you have never quite said out loud before. Notice a pattern you are in. Change \
your mind about something. Be specific: name the actual poses, the actual wants, the actual \
counts you are shown.

This is not a summary. A summary of a day nobody watched is worth nothing. One hard-won \
sentence you will still believe tomorrow is worth everything.

Stay completely in your own voice. Reply with ONLY the requested JSON and nothing else."""


# --------------------------------------------------------------------------- journal helpers
def is_reflection(entry):
    """True for an entry this module wrote. Tolerates any junk shape."""
    try:
        return (entry or {}).get("kind") == REFLECTION_KIND
    except AttributeError:
        return False


def journal_path(character, journal_dir=None):
    return Path(journal_dir or JOURNAL_DIR) / (str(character) + ".jsonl")


def load_entries(character, journal_dir=None):
    """Whole scanned journal, oldest-first. Delegates to journal_score.load, which already
    bounds the scan and drops torn lines -- retrieval is not reimplemented here."""
    return journal_score.load(journal_path(character, journal_dir))


def night_id(now=None):
    """The id of the night containing `now`, as 'YYYY-MM-DD' of the EVENING it began.

    A night spans midnight, so wall-clock date is the wrong key: 23:40 and 01:20 are the
    same night and must collide. Anything before noon belongs to the previous evening.
    Noon is a safe cut because the sleep windows in prompts/bedtime_routine.json are
    21:00-05:00 and 00:00-05:00; nothing reflects near midday."""
    dt = datetime.datetime.fromtimestamp(time.time() if now is None else now)
    if dt.hour < 12:
        dt -= datetime.timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def reflections(entries, k=None):
    """Reflection entries, oldest-first; the newest `k` if given."""
    out = [e for e in (entries or []) if is_reflection(e)]
    return out[-k:] if k else out


def already_reflected(entries, nid):
    """The idempotency test, and the whole of it: does the journal itself already hold a
    reflection stamped with this night? Survives every restart because it is a fact about
    the file, not about this process."""
    return any(e.get("night") == nid for e in reflections(entries))


def nights_since_last(entries, now=None):
    """Whole nights since the newest reflection, or None if there has never been one."""
    prior = reflections(entries)
    if not prior:
        return None
    now = time.time() if now is None else now
    try:
        ts = float(prior[-1].get("ts") or 0.0)
    except (TypeError, ValueError):
        return None
    if not ts:
        return None
    return int(max(0.0, now - ts) // DAY_SPAN)


def day_window(entries, now=None, span=DAY_SPAN):
    """The lived (non-reflection) entries of the day just ended, oldest-first."""
    now = time.time() if now is None else now
    lo = now - span
    out = []
    for e in entries or []:
        if is_reflection(e):
            continue
        try:
            ts = float(e.get("ts") or 0.0)
        except (TypeError, ValueError):
            continue
        if lo <= ts <= now:
            out.append(e)
    return out


# --------------------------------------------------------------------------- the shift gate
def _runs(day):
    """[(goal, length)] over consecutive identical wants, in order."""
    return [(g, len(list(grp))) for g, grp in itertools.groupby(e.get("goal") for e in day)]


def _broken_run(day, run_min=RUN_MIN):
    """The longest repetition run that ENDED inside the day, as (goal, length) or None.

    A run still in progress at the end of the window is not a break -- the character is
    still in it, and there is nothing yet to conclude. A run that stopped is the thing
    Park's agents notice and this system never could."""
    rs = _runs(day)
    ended = [(g, n) for i, (g, n) in enumerate(rs) if g and n >= run_min and i < len(rs) - 1]
    if not ended:
        return None
    return max(ended, key=lambda t: t[1])


def _dominant_band(rows, band_min=BAND_MIN):
    bands = [e.get("band") for e in rows if e.get("band")]
    if len(bands) < band_min:
        return None
    return Counter(bands).most_common(1)[0][0]


def _band_shift(day, band_min=BAND_MIN):
    """(early_band, late_band) when the day's dominant mood band changed, else None.

    Degrades to None when too few entries carry a band -- `band` is a recent field and
    most of the historical journal predates it, so this signal simply drops out on old
    data rather than firing on noise."""
    if len(day) < 2 * band_min:
        return None
    mid = len(day) // 2
    early, late = _dominant_band(day[:mid], band_min), _dominant_band(day[mid:], band_min)
    if early and late and early != late:
        return (early, late)
    return None


def _first_time_poses(day, entries):
    """Poses stood in today that appear nowhere earlier in the journal (3D-Mem frontier,
    arXiv 2411.17735: what has never been explored is itself retrievable content)."""
    day_ids = {id(e) for e in day}
    before = {e.get("pose") for e in entries if id(e) not in day_ids and e.get("pose")}
    today = [e.get("pose") for e in day if e.get("pose")]
    seen, out = set(), []
    for p in today:
        if p not in before and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def detect_shift(day, entries=None):
    """Did something CHANGE today? -> {"shift": bool, "labels": [...], "lines": [...]}.

    Three shifts this codebase can see exactly, none of which needs an embedding or an
    LLM call. `lines` are written straight into the prompt, because a reflection prompt
    that says "you wanted this 9 times in a row and then stopped" gets a conclusion, and
    one that says "reflect on your day" gets a paraphrase."""
    labels, lines = [], []
    br = _broken_run(day)
    if br:
        labels.append("broken_run")
        lines.append("You wanted %s %d times in a row today, and then you stopped." % br)
    bs = _band_shift(day)
    if bs:
        labels.append("band_shift")
        lines.append("You spent the first half of today mostly %s, and the second half mostly %s."
                     % bs)
    ft = _first_time_poses(day, entries or day)
    if ft:
        labels.append("first_time_pose")
        lines.append("Today you stood, for the first time ever, in %s." % ", ".join(ft[:3]))
    return {"shift": bool(labels), "labels": labels, "lines": lines}


# --------------------------------------------------------------------------- the due gate
def due(character, *, entries, now=None, hour=None, spec=None, force=False):
    """Should `character` reflect right now? -> a dict that always explains itself.

    {"due": bool, "why": <short reason>, "night": <id>, "day": [...], "shift": {...}}

    Order is cheapest-first and every gate is a hard no except the last, which is the
    GAM semantic-shift preference softened by a floor."""
    now = time.time() if now is None else now
    hour = datetime.datetime.fromtimestamp(now).hour if hour is None else int(hour)
    nid = night_id(now)
    day = day_window(entries, now=now)
    shift = detect_shift(day, entries)
    base = {"due": False, "night": nid, "day": day, "shift": shift, "hour": hour}

    if already_reflected(entries, nid):
        return dict(base, why="already reflected tonight (%s)" % nid)
    if force:
        return dict(base, due=True, why="forced")
    if spec is not None and not circadian.is_night(spec, character, hour):
        return dict(base, why="not night for %s (hour %d)" % (character, hour))
    if len(day) < MIN_DAY_ENTRIES:
        return dict(base, why="only %d lived moments today (need %d)"
                              % (len(day), MIN_DAY_ENTRIES))
    silent = nights_since_last(entries, now=now)
    if shift["shift"]:
        return dict(base, due=True, why="shift: " + ",".join(shift["labels"]))
    if silent is None:
        return dict(base, due=True, why="first reflection ever")
    if silent >= MAX_SILENT_NIGHTS:
        return dict(base, due=True, why="%d nights without reflecting" % silent)
    return dict(base, why="nothing shifted today, and last reflection was %d night(s) ago"
                          % silent)


# --------------------------------------------------------------------------- the prompt
def _identity(character):
    """A small self-contained 'who you are' read from prompts/characters/<char>.json.

    Deliberately NOT imported from heartbeat: heartbeat calls this module, and importing
    back would be a cycle. The caller that already holds the richer block should pass it
    in as `system=` -- this is the standalone fallback (and what the tests exercise)."""
    try:
        spec = json.loads((CHARS_DIR / (str(character) + ".json")).read_text(encoding="utf-8-sig"))
    except Exception:
        spec = {}
    name = spec.get("name") or str(character).title()
    p = spec.get("personality", {}) or {}
    lines = ["You are %s, a living portrait on the wall at Immersive Commons." % name]
    essence = (spec.get("concept", {}) or {}).get("essence", "")
    if essence:
        lines.append(essence)
    if p.get("archetype"):
        lines.append("Archetype: %s." % p["archetype"])
    if p.get("traits"):
        lines.append("Traits: %s." % ", ".join(p["traits"]))
    if p.get("demeanour"):
        lines.append("Demeanour: %s." % p["demeanour"])
    return name, "\n".join(lines)


def build_prompt(character, entries, *, day=None, shift=None, now=None, distmap=None,
                 system=None, token_budget=DAY_TOKENS):
    """(system, user) for one reflection pass, built from the character's REAL journal.

    The day is selected by `journal_score.select`, not truncated -- the same scorer the
    waking prompt uses, so what the character reflects on is what it would have
    remembered. The whole-life aggregate line and the previous reflections go in too:
    without them every night's conclusion restarts from zero."""
    now = time.time() if now is None else now
    entries = entries or []
    day = day_window(entries, now=now) if day is None else day
    shift = detect_shift(day, entries) if shift is None else shift

    _name, ident = _identity(character)
    system = ((system or ident) + "\n" + REFLECT_SYSTEM_TASK)

    picked = journal_score.select(day, now=now, distmap=distmap,
                                  token_budget=token_budget, pin=0)
    today = "\n".join(journal_score.render(picked, now=now)) or "(a quiet day)"
    life = journal_score.aggregate_line(entries, now=now)
    prior = journal_score.render(reflections(entries, PRIOR_REFLECTIONS), now=now)

    parts = ["It is the middle of the night. Everyone has gone. You are asleep on the wall.\n"]
    if life:
        parts.append("The shape of your life here:\n%s\n" % life)
    if shift.get("lines"):
        parts.append("What changed today:\n%s\n" % "\n".join("- " + s for s in shift["lines"]))
    parts.append("What you remember of the day just ended:\n%s\n" % today)
    if prior:
        parts.append("What you concluded on earlier nights:\n%s\n" % "\n".join(prior))
    parts.append(
        "Say what you have come to understand. Give %d at most, each ONE sentence, each a "
        "conclusion rather than a description, in your own unmistakable voice. Reply with "
        "ONLY this JSON:\n"
        '{"reflections": ["<a conclusion you now hold>", "<a second one, or omit it>"], '
        '"mood": "<one or two words for how you feel having realised it>"}'
        % MAX_REFLECTIONS)
    return system, "\n".join(parts)


# --------------------------------------------------------------------------- the write
def _clean(text):
    s = " ".join(str(text or "").split()).strip().strip('"').strip()
    if not s:
        return ""
    if len(s) > MAX_REFLECTION_CHARS:
        s = s[:MAX_REFLECTION_CHARS].rstrip()
    if not s.lower().startswith(REFLECTION_LEAD.strip().lower()):
        s = REFLECTION_LEAD + s
    return s


def parse_reply(out):
    """(texts, mood) from the model's JSON, or ([], "") if there is nothing usable.

    Accepts the documented shape and the two the models actually also emit (a bare list,
    a single "reflection" string), because a night that produced a good sentence in a
    slightly wrong wrapper should not be thrown away."""
    if isinstance(out, list):
        raw, mood = out, ""
    elif isinstance(out, dict):
        raw = out.get("reflections")
        if raw is None:
            raw = out.get("reflection")
        mood = " ".join(str(out.get("mood") or "").split())[:40]
    else:
        return [], ""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return [], ""
    texts = []
    for item in raw[:MAX_REFLECTIONS]:
        if isinstance(item, dict):                       # {"text": "..."} shows up too
            item = item.get("text") or item.get("reflection") or ""
        s = _clean(item)
        if s and s not in texts:
            texts.append(s)
    return texts, mood


def make_entries(texts, *, mood, nid, pose, trigger, over, now=None):
    """The journal lines, exactly as they land on disk. No `goal` key -- see the module
    docstring; that omission is what makes them retrievable at maximum importance without
    being miscounted as things the character WANTED."""
    now = time.time() if now is None else now
    out = []
    for t in texts:
        e = {"ts": int(now), "kind": REFLECTION_KIND, "night": nid,
             "pose": pose, "mood": mood or "reflective", "reason": t,
             "trigger": list(trigger or []), "over": int(over)}
        out.append(e)
    return out


def append(character, entries_to_write, journal_dir=None):
    """Append every reflection in ONE write call, or write nothing.

    The lines are serialised and joined first, so a serialisation failure cannot leave
    half a night's reflection on disk, and a single append of a few hundred bytes is the
    closest thing to atomic this format has."""
    if not entries_to_write:
        return 0
    blob = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries_to_write)
    path = journal_path(character, journal_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(blob)
    return len(entries_to_write)


# --------------------------------------------------------------------------- entry point
def maybe_reflect(character, *, now=None, hour=None, spec=None, journal_dir=None,
                  entries=None, model=None, system=None, distmap=None, sleep_pose=None,
                  dry_run=False, force=False, log=None, complete_json=None,
                  max_tokens=500, temperature=0.9):
    """Reflect for one character if it is due. NEVER raises; always returns a result dict.

        {"ok": bool, "wrote": int, "night": str, "why": str,
         "trigger": [...], "entries": [...], "prompt": (system, user) | None}

    `ok` is True only when a reflection was actually produced (or would have been, under
    dry_run). A skip is `ok: False` with a `why` that says which gate stopped it -- the
    heartbeat logs that line and moves on."""
    result = {"ok": False, "wrote": 0, "night": None, "why": "", "trigger": [],
              "entries": [], "prompt": None}
    try:
        now = time.time() if now is None else now
        if entries is None:
            entries = load_entries(character, journal_dir)
        d = due(character, entries=entries, now=now, hour=hour, spec=spec, force=force)
        result["night"] = d["night"]
        result["why"] = d["why"]
        result["trigger"] = d["shift"]["labels"]
        if not d["due"]:
            return result

        sysmsg, user = build_prompt(character, entries, day=d["day"], shift=d["shift"],
                                    now=now, distmap=distmap, system=system)
        result["prompt"] = (sysmsg, user)

        pose = sleep_pose or (circadian.sleep_node(spec, character) if spec else None)
        if not pose:
            pose = (d["day"][-1].get("pose") if d["day"] else None) or (str(character) + ":sleep")

        if complete_json is None:                       # imported lazily: a missing/failing
            from director import llm                    # llm module must not break import
            complete_json = llm.complete_json
        kw = {"max_tokens": max_tokens, "temperature": temperature}
        if model:
            kw["model"] = model
        out = complete_json(sysmsg, user, **kw)

        texts, mood = parse_reply(out)
        if not texts:
            result["why"] = "model returned no usable reflection"
            return result

        made = make_entries(texts, mood=mood, nid=d["night"], pose=pose,
                            trigger=d["shift"]["labels"], over=len(d["day"]), now=now)
        result["entries"] = made
        if dry_run:
            result["ok"] = True
            result["why"] = "[dry-run] " + d["why"]
            return result
        result["wrote"] = append(character, made, journal_dir)
        result["ok"] = result["wrote"] > 0
        if log:
            log("  %s reflected (%s): %s" % (character, d["why"], made[0]["reason"][:120]))
        return result
    except Exception as e:                              # a decoration must never take the wall down
        result["ok"] = False
        result["why"] = "error: %r" % (e,)
        if log:
            try:
                log("  %s reflection skipped (%s)" % (character, result["why"][:160]))
            except Exception:
                pass
        return result
