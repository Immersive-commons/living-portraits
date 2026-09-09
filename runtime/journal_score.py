"""journal_score.py -- SCORED RETRIEVAL over a character's journal.

The heartbeat used to read its own past with `lines[-5:]`. At the measured ~7-minute
tick that is about 35 minutes of remembered life out of 64 days (0.04%), which is why
Phineas could choose `jealous_glare` 2,060 times without ever noticing he had.

This module replaces recency-only truncation with the Generative-Agents retrieval
score -- recency + importance + relevance, all weights 1.0 (arXiv 2304.03442) -- plus
the 2026 delta the pose graph makes available for free: relevance measured in
TRANSITION HOPS, not text similarity. A memory made two steps from where the character
is standing is more relevant than one made across the graph, and the graph already
knows the distance.

Deliberately NOT here (each ruled out with a citation in ROADMAP.md v0.3.0):
embeddings over the journal (a 120-word goal vocabulary -- exact match beats cosine
and costs nothing), a graph DB, PPR, GraphRAG community summaries.

Pure + import-safe (stdlib only). The scan is bounded (`MAX_SCAN`) and the selection is
bounded by a TOKEN budget, so prompt size stays flat as the journal grows forever.

    from runtime import journal_score as js
    entries = js.load(path)                      # newest MAX_SCAN, parsed
    picked  = js.select(entries, now=time.time(), distmap=hops, token_budget=700)
    lines   = js.render(picked, now=time.time()) # chronological, with relative ages
    agg     = js.aggregate_line(entries, now=time.time())
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, deque

# --- tunables (module constants so the tests pin behaviour, not magic numbers).
RECENCY_DECAY = 0.995      # per HOUR; the Generative Agents constant (0.995^h)
MAX_SCAN = 40000           # newest N journal lines parsed per tick (bounds the scan)
TOKEN_BUDGET = 700         # approx tokens of retrieved monologue fed to the model
CHARS_PER_TOKEN = 4        # crude but stable; we only need a bound, not a tokenizer
MIN_PICKED = 3             # always return at least this many if the journal has them
PIN_IMPORTANT = 2          # slots reserved for DISTANT memories that still stand out (below)
PIN_MIN_AGE = 86400.0      # ...and "distant" starts here, so a pin is never a recent line

W_RECENCY = 1.0            # Park et al. set all three alphas to 1.0 in the released code
W_IMPORTANCE = 1.0
W_RELEVANCE = 1.0


def load(path, max_scan=MAX_SCAN):
    """Newest `max_scan` journal lines, parsed, oldest-first. Unreadable file -> [].

    Streams with a bounded deque so a journal that grows to millions of lines costs
    constant memory. Malformed lines are skipped, never raised: this feeds a render
    loop's brain, and a half-written line must not take the panels down."""
    out = deque(maxlen=max_scan)
    try:
        with open(str(path), encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    out.append(ln)
    except Exception:
        return []
    entries = []
    for ln in out:
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if isinstance(d, dict):
            entries.append(d)
    return entries


def _ts(entry):
    try:
        return float(entry.get("ts") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def recency(entry, now):
    """0..1, exponential decay per hour since the memory was made."""
    age_h = max(0.0, (now - _ts(entry)) / 3600.0)
    return RECENCY_DECAY ** age_h


def goal_counts(entries):
    """Counter over what this character has WANTED, across the whole scanned history."""
    return Counter(e.get("goal") for e in entries if e.get("goal"))


def importance(entry, counts, total):
    """0..1 surprisal of this memory's goal: -log2 p(goal), normalised by the most
    surprising thing possible (a goal seen exactly once). A want chosen 2,060 times out
    of 11,227 scores ~0.18; a want chosen once scores 1.0. No LLM call, no config."""
    if not total:
        return 0.0
    c = counts.get(entry.get("goal"), 0)
    if c <= 0:
        return 1.0
    p = c / float(total)
    denom = math.log2(total) or 1.0
    return min(1.0, (-math.log2(p)) / denom)


def relevance(entry, distmap):
    """0..1 by TRANSITION HOPS from where the character stands now to where the memory
    happened (`pose`) or to what it wanted (`goal`). Happened-right-here -> 1.0;
    unreachable or unknown -> 0.0. Empty distmap -> 0.0 for every entry, i.e. the term
    simply drops out and the score degrades to recency+importance."""
    if not distmap:
        return 0.0
    best = 0.0
    for key in ("pose", "goal"):
        node = entry.get(key)
        if node is None:
            continue
        d = distmap.get(node)
        if d is not None:
            best = max(best, 1.0 / (1.0 + float(d)))
    return best


def score_entries(entries, *, now=None, distmap=None, counts=None):
    """[(score, index, entry)] highest first. `index` breaks ties deterministically
    toward the more recent memory."""
    now = time.time() if now is None else now
    counts = goal_counts(entries) if counts is None else counts
    total = sum(counts.values())
    scored = []
    for i, e in enumerate(entries):
        s = (W_RECENCY * recency(e, now)
             + W_IMPORTANCE * importance(e, counts, total)
             + W_RELEVANCE * relevance(e, distmap))
        scored.append((s, i, e))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return scored


def _line_len(entry):
    return len(str(entry.get("reason") or "")) + len(str(entry.get("mood") or "")) + 48


def pinned(entries, *, now=None, distmap=None, counts=None, k=PIN_IMPORTANT,
           min_age=PIN_MIN_AGE):
    """The DISTANT memories that still stand out: ranked by importance + relevance with
    the recency term switched off, among entries older than `min_age`.

    Without this, exponential decay makes the whole design short-sighted: measured on
    hil's real 66-day journals, pure recency+importance+relevance retrieved nothing older
    than 9 days, so a character with two months of life could still only reach back a
    week. Two reserved slots are what let it say "the night I finally took a bow" -- the
    long-term half of remembering, at the cost of two lines of prompt."""
    now = time.time() if now is None else now
    counts = goal_counts(entries) if counts is None else counts
    total = sum(counts.values())
    old = [(i, e) for i, e in enumerate(entries) if now - _ts(e) >= min_age]
    if not old:
        return []
    ranked = sorted(
        old,
        key=lambda t: (-(importance(t[1], counts, total) + relevance(t[1], distmap)), -t[0]))
    return ranked[:max(0, int(k))]


def select(entries, *, now=None, distmap=None, token_budget=TOKEN_BUDGET,
           min_picked=MIN_PICKED, pin=PIN_IMPORTANT):
    """The memories worth showing the model, returned CHRONOLOGICALLY.

    A few reserved long-term pins, then top-scored under a token budget, then re-sorted
    oldest->newest so the monologue still reads as a life rather than a ranked list."""
    if not entries:
        return []
    now = time.time() if now is None else now
    counts = goal_counts(entries)
    budget_chars = max(0, int(token_budget) * CHARS_PER_TOKEN)
    picked, spent, seen = [], 0, set()

    for idx, e in pinned(entries, now=now, distmap=distmap, counts=counts, k=pin):
        picked.append((idx, e))
        seen.add(idx)
        spent += _line_len(e)

    for _, idx, e in score_entries(entries, now=now, distmap=distmap, counts=counts):
        if idx in seen:
            continue
        cost = _line_len(e)
        if picked and spent + cost > budget_chars and len(picked) >= min_picked:
            break
        picked.append((idx, e))
        seen.add(idx)
        spent += cost
    picked.sort(key=lambda t: t[0])
    return [e for _, e in picked]


def _ago(now, ts):
    if not ts:
        return "some time ago"
    secs = max(0.0, now - ts)
    if secs < 3600:
        return "%dm ago" % int(secs // 60)
    if secs < 86400:
        return "%dh ago" % int(secs // 3600)
    return "%dd ago" % int(secs // 86400)


def render(entries, now=None):
    """Prompt lines, one per memory, carrying WHEN it happened. The relative age is the
    point: without it the model reads its whole remembered life as though it were now."""
    now = time.time() if now is None else now
    out = []
    for e in entries:
        out.append("- (%s, at %s, feeling %s) %s" % (
            _ago(now, _ts(e)), e.get("pose", "?"), e.get("mood", "?"),
            (e.get("reason") or "").strip()))
    return out


def summary(entries, now=None):
    """Aggregate facts about a life: {n, days, top_goal, top_count, top_share, distinct}."""
    now = time.time() if now is None else now
    if not entries:
        return {}
    counts = goal_counts(entries)
    total = sum(counts.values())
    stamps = [t for t in (_ts(e) for e in entries) if t]
    days = ((now - min(stamps)) / 86400.0) if stamps else 0.0
    top_goal, top_count = (counts.most_common(1) or [(None, 0)])[0]
    return {"n": len(entries), "days": days, "top_goal": top_goal,
            "top_count": top_count,
            "top_share": (top_count / float(total)) if total else 0.0,
            "distinct": len(counts)}


def aggregate_line(entries, now=None):
    """One line of self-knowledge no window of individual memories can carry -- the shape
    of a life rather than a sample of it. This is the line that lets a character notice a
    rut it has been in for two months. Empty journal -> ""."""
    s = summary(entries, now=now)
    if not s or not s.get("top_goal"):
        return ""
    return ("Looking back over %d remembered moments across %.0f days: you have wanted "
            "%s %d times (%.0f%% of everything you have ever chosen), out of %d different "
            "wants." % (s["n"], s["days"], s["top_goal"], s["top_count"],
                        100.0 * s["top_share"], s["distinct"]))


def visited_poses(entries):
    """Every pose this character has actually STOOD IN, per its own journal. The
    complement of this set inside `reachable_poses` is the frontier."""
    out = set()
    for e in entries:
        p = e.get("pose")
        if p:
            out.add(p)
    return out
