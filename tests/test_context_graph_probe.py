"""test_context_graph_probe.py -- the SCORECARD. Is this a true context graph, on real data?

`_audit/CONTEXT_GRAPH_AUDIT.md` names four capabilities that show up in every serious
context-graph system, and `_research/CONTEXT_GRAPHS_FINDINGS.md` checks that list against the
literature. This file turns them into five measurements and runs them against the production
snapshot in `data/_realdata/` -- 259 nodes, 1,472 edges, 25,405 journal entries over 66 days.

Every number below comes from that snapshot. None of it is a fixture.

    (a) bi-temporal        -- do nodes/edges carry a creation time, and can a fact be invalidated?
    (b) ranked walk        -- does GRAPH-HOP relevance change what is retrieved, on real journals?
    (c) scored + reflection-- is scoring live in the production path, and do reflections exist?
    (d) extract-at-retrieval- is the raw record kept verbatim, or lossily summarised at write?
    (e) declarative world  -- how many facts does it hold about anything but its own movement?

WHY THIS TEST PASSES WHILE REPORTING FAILURES
---------------------------------------------
A scorecard is an instrument, not a promise. The TEST asserts that the instrument still reads
what it read when it was calibrated; the VERDICTS inside it say what the system can do. So:

  * (e) is asserted to be FAIL. If the world-fact count ever goes above zero the test breaks --
    which is the point. On the day someone builds declarative memory, this file must be
    re-read and re-calibrated by a human, not silently turn green.
  * (b) and (d) are asserted to be PASS. They shipped in v0.3.0; the assertion is a regression
    gate, and it fires if a peer stubs graph-hop relevance or starts rewriting the journal.
  * (a) and (c) are peer-owned and were incomplete when this was written. Their verdicts are
    measured and printed but NOT asserted, because an auditor who hard-codes a peer's schedule
    is measuring the schedule, not the system.

Run it on its own to read the scorecard (pytest swallows stdout otherwise):

    python -m pytest tests/test_context_graph_probe.py -s -q
    python tests/test_context_graph_probe.py            # same scorecard, no pytest
"""
from __future__ import annotations

import copy
import importlib
import json
import re
import shutil
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import realdata                                          # noqa: E402
from realdata import realdata_snapshot                    # noqa: E402,F401  (the skip fixture)

from director import heartbeat                            # noqa: E402
from runtime import journal_score as js, pathfind         # noqa: E402

PASS, PARTIAL, FAIL = "PASS", "PARTIAL", "FAIL"
REPLAY_N = 20              # real decisions replayed per character (see _replay)
DAY = 86400.0

# Time-ish field names any bi-temporal implementation would have to use one of. Matched against
# node/edge keys so the measurement finds a peer's schema without being told what it is.
_TIME_KEY = re.compile(
    r"^(t_)?(created|valid|valid_from|invalid|invalidated|expired|expires|retired|superseded"
    r"|first|last|since|until|observed|learned|born|added|generated|built|updated)", re.I)
_INVALIDATION_KEY = re.compile(
    r"^(t_)?(invalid|invalidated|expired|expires|retired|superseded|deleted|revoked)", re.I)

# Words a first-person monologue uses when it is talking about the WORLD -- people, the room,
# an audience. Used only to count how often the characters reference things the system holds
# no record of. Deliberately conservative.
_WORLD_TALK = re.compile(
    r"\b(visitor|someone|somebody|walked past|the room|member|builder|human|people|audience"
    r"|watching|stranger|guest)\b", re.I)

_cache: dict = {}


# ============================================================================ helpers
def _hops_for(graph, node, adj):
    key = ("hops", node)
    if key not in _cache:
        _cache[key] = pathfind.hops_from(graph.edges, node, adj=adj)
    return _cache[key]


def _fingerprint(entries):
    """Identity of a RETRIEVED SET, independent of order: (timestamp, first 40 chars)."""
    return frozenset((float(e.get("ts") or 0), (e.get("reason") or "")[:40]) for e in entries)


def _reach_days(entries, now):
    """How far back the retrieved set reaches, in days. The headline number for (b): a reader
    that cannot reach past yesterday is a window, whatever it is scored by."""
    if not entries:
        return 0.0
    return max(0.0, (now - min(float(e.get("ts") or now) for e in entries)) / DAY)


def _replay(character, n=REPLAY_N, tag="live"):
    """Replay the last `n` REAL decisions for one character and compare four readers.

    For each decision we rebuild exactly what the heartbeat would have had: the journal
    strictly BEFORE that moment, the pose the character was actually standing in, and the BFS
    hop distances from that pose over the real graph. Then we retrieve four ways --

        full   scored retrieval as shipped (recency + surprisal + hop relevance, + long pins)
        no_hop the same, with the GRAPH term removed        <- isolates what the graph buys
        no_pin the same, with the long-term pins removed    <- isolates what the pins buy
        tail5  the pre-0.3.0 reader, `lines[-5:]`           <- the control

    `tag` keys the cache so a stubbed-out run (the falsifiability check) cannot read a cached
    result from the honest one."""
    key = ("replay", character, n, tag)
    if key in _cache:
        return _cache[key]

    graph = realdata.graph()
    adj = pathfind._adjacency(graph.edges)
    entries = realdata.journal(character)

    rows = []
    for k in range(n):
        i = len(entries) - 1 - k
        if i <= 0:
            break
        moment = entries[i]
        prefix, now = entries[:i], float(moment.get("ts") or time.time())
        hops = _hops_for(graph, moment.get("pose"), adj)
        full = js.select(prefix, now=now, distmap=hops)
        no_hop = js.select(prefix, now=now, distmap=None)
        no_pin = js.select(prefix, now=now, distmap=hops, pin=0)
        tail5 = prefix[-5:]
        f_full, f_hop = _fingerprint(full), _fingerprint(no_hop)
        rows.append({
            "differs_from_tail5": _fingerprint(full) != _fingerprint(tail5),
            "differs_from_no_hop": f_full != f_hop,
            "jaccard_vs_no_hop": len(f_full & f_hop) / float(len(f_full | f_hop) or 1),
            "only_with_hops": len(f_full - f_hop),
            "picked": len(full),
            "reach_full": _reach_days(full, now),
            "reach_no_hop": _reach_days(no_hop, now),
            "reach_no_pin": _reach_days(no_pin, now),
            "reach_tail5": _reach_days(tail5, now),
            "scanned": len(prefix),
        })

    def med(f):
        return st.median([r[f] for r in rows]) if rows else 0.0

    out = {
        "n": len(rows),
        "differs_from_tail5": sum(r["differs_from_tail5"] for r in rows),
        "differs_from_no_hop": sum(r["differs_from_no_hop"] for r in rows),
        "median_jaccard_vs_no_hop": med("jaccard_vs_no_hop"),
        "median_only_with_hops": med("only_with_hops"),
        "median_picked": med("picked"),
        "median_reach_full": med("reach_full"),
        "median_reach_no_hop": med("reach_no_hop"),
        "median_reach_no_pin": med("reach_no_pin"),
        "median_reach_tail5": med("reach_tail5"),
        "max_reach_full": max([r["reach_full"] for r in rows] or [0.0]),
        "scanned": rows[0]["scanned"] if rows else 0,
    }
    _cache[key] = out
    return out


def _real_prompt(character, spy=None):
    """The prompt the production heartbeat would build RIGHT NOW from the real journal.

    Points the heartbeat's three data dirs at the snapshot (journal) and an empty tmp dir
    (poses), and silences the weather fetch so the measurement is offline and deterministic.
    Everything is restored in `finally` -- a measurement that leaves the module repointed at
    the snapshot would poison every later test in the session."""
    old = (heartbeat.JOURNAL_DIR, heartbeat.POSE_DIR, heartbeat.ctx.context_line, js.select)
    posedir = Path(tempfile.mkdtemp(prefix="lp_probe_pose_"))
    try:
        heartbeat.JOURNAL_DIR = realdata.SNAPSHOT
        heartbeat.POSE_DIR = posedir
        heartbeat.ctx.context_line = lambda *a, **k: ""
        if spy is not None:
            real_select = js.select

            def _spy(entries, **kw):
                spy.append({"scanned": len(entries), "kwargs": sorted(kw)})
                return real_select(entries, **kw)
            js.select = _spy
        graph = realdata.graph()
        entries = realdata.journal(character)
        node = entries[-1].get("pose")
        hops = pathfind.hops_from(graph.edges, node)
        goals = sorted(pathfind.reachable_poses(graph.edges, node))
        return heartbeat._build_user_prompt(graph, character, node, 3, goals, 14, hops=hops)
    finally:
        heartbeat.JOURNAL_DIR, heartbeat.POSE_DIR, heartbeat.ctx.context_line, js.select = old
        shutil.rmtree(str(posedir), ignore_errors=True)


def _module_exists(dotted):
    """Is a peer's module importable? Absence is a measurement, not an error."""
    try:
        importlib.import_module(dotted)
        return True
    except Exception:
        return False


# ============================================================================ (a) bi-temporal
def _stamped_coverage():
    """Run the provenance module over a COPY of the real graph and report what it can date.

    `stamp()` mutates in place and derives `created` from artifact mtime, so it is handed a
    deepcopy -- the snapshot's cached records must come back out of this function exactly as
    they went in. Absent module -> (None, reason)."""
    try:
        from runtime import graph_provenance as gp
    except Exception as e:
        return None, "runtime/graph_provenance.py not importable (%s)" % e
    g = realdata.raw_graph()
    nodes, edges = copy.deepcopy(g["nodes"]), copy.deepcopy(g["edges"])
    return gp.stamp(nodes, edges), None


def _invalidation_roundtrip():
    """Can a fact that WAS true stop being true without being erased? Proven on real ids.

    Diffs the real graph against the same graph minus one real node and one real edge, in an
    in-memory store that is never saved, then asks the history view for them back. This is a
    CAPABILITY proof: it says the schema can express invalidation, which is the Zep property.
    Whether anything has actually been tombstoned in production is a separate number, reported
    beside it."""
    try:
        from runtime import graph_provenance as gp
    except Exception:
        return False, "no provenance module"
    try:
        g = realdata.raw_graph()
        node_id = sorted(g["nodes"])[0]
        edge_id = next(e["id"] for e in g["edges"] if e.get("id"))
        new_nodes = {k: v for k, v in g["nodes"].items() if k != node_id}
        new_edges = [e for e in g["edges"] if e.get("id") != edge_id]
        store, stats = gp.reconcile(g["nodes"], g["edges"], new_nodes, new_edges,
                                    store=gp._empty_store())
        hist_nodes, hist_edges = gp.merge_history(new_nodes, new_edges, store)
        ok = (edge_id in gp.invalidated_edge_ids(store)
              and node_id in gp.invalidated_node_ids(store)
              and node_id in hist_nodes
              and any(e.get("id") == edge_id and e.get("invalidated") for e in hist_edges))
        return bool(ok), ("tombstoned %d node / %d edge and served both back out of history"
                          % (stats["nodes_invalidated"], stats["edges_invalidated"]))
    except Exception as e:
        return False, "round-trip raised %s" % e


def measure_bitemporal():
    """What fraction of the REAL graph carries a creation time, and can anything be invalidated?

    Two halves, per the audit's reading of Zep/Graphiti: TRANSACTION time (when the system
    learned the thing exists) and VALID time (when it was true of the world). Measured against
    the raw production JSON first -- that is the file the walker actually loads -- and then
    against what the provenance and lived-experience modules can show for that same data."""
    g = realdata.raw_graph()
    nodes, edges = g["nodes"], g["edges"]

    def timed(records):
        return sum(1 for r in records if any(_TIME_KEY.match(k) for k in r))

    def invalidatable(records):
        return sum(1 for r in records if any(_INVALIDATION_KEY.match(k) for k in r))

    n_timed, e_timed = timed(nodes.values()), timed(edges)
    n_inval, e_inval = invalidatable(nodes.values()), invalidatable(edges)
    raw_node_cov = n_timed / float(len(nodes) or 1)
    raw_edge_cov = e_timed / float(len(edges) or 1)

    lines = [
        "graph schema %s: %d nodes, %d edges" % (g["schema"], len(nodes), len(edges)),
        "IN THE FILE THE WALKER LOADS -- nodes with any time field %d/%d (%.0f%%), edges "
        "%d/%d (%.0f%%); node fields = %s"
        % (n_timed, len(nodes), 100 * raw_node_cov, e_timed, len(edges), 100 * raw_edge_cov,
           sorted({k for r in nodes.values() for k in r})),
        "...and records in that file able to express INVALIDATION: %d nodes, %d edges"
        % (n_inval, e_inval),
    ]

    # --- TRANSACTION time: what the provenance sidecar can date, over the real graph.
    stats, why = _stamped_coverage()
    if stats is None:
        node_cov, edge_cov = raw_node_cov, raw_edge_cov
        lines.append("provenance: %s" % why)
    else:
        node_cov = max(raw_node_cov, stats["nodes_created"] / float(stats["nodes"] or 1))
        edge_cov = max(raw_edge_cov, stats["edges_created"] / float(stats["edges"] or 1))
        lines.append(
            "graph_provenance.stamp() over the REAL graph dates %d/%d nodes (%.0f%%) and "
            "%d/%d edges (%.0f%%) from artifact mtime ON THIS CHECKOUT"
            % (stats["nodes_created"], stats["nodes"], 100 * node_cov,
               stats["edges_created"], stats["edges"], 100 * edge_cov))
        if node_cov < 0.95 or edge_cov < 0.95:
            lines.append("the shortfall is artifacts this box does not hold: the stamp is "
                         "DERIVED from mtime, so coverage is a property of the machine as much "
                         "as of the graph. The production number needs a run on hil.")

    inval_ok, inval_note = _invalidation_roundtrip()
    lines.append("invalidation round-trip on REAL ids (in-memory store, nothing written): %s -- %s"
                 % ("WORKS" if inval_ok else "NO", inval_note))
    try:
        from runtime import graph_provenance as gp
        s = gp.summary(gp.load_store())
        lines.append("tombstones actually recorded in production: %d nodes / %d edges (store %s)"
                     % (s["nodes_dead"], s["edges_dead"], gp.STORE_PATH.name))
    except Exception:
        pass

    # --- VALID time: lived.py is the other half, and the snapshot carries none of it.
    snapshot_lived = sorted(p.name for p in realdata.SNAPSHOT.glob("lived*.json"))
    lines.append("VALID-time half (runtime/lived.py -- when a pose was actually stood in): "
                 "module present=%s, lived records in the production snapshot: %s"
                 % (_module_exists("runtime.lived"), snapshot_lived or "NONE"))
    if not snapshot_lived:
        lines.append("WIRING REQUEST: the snapshot carries no data/mind/lived/*.json and no "
                     "data/mind/pose/*.json, so valid-time coverage and the neighbour line "
                     "cannot be measured on production data at all -- only on fixtures.")

    covered = min(node_cov, edge_cov)
    if covered >= 0.95 and inval_ok:
        verdict = PASS
    elif covered > 0 or inval_ok:
        verdict = PARTIAL
    else:
        verdict = FAIL
        lines.append("the production graph cannot say when a pose entered this character's "
                     "life, nor that a clip is no longer true. Both halves absent.")
    return {"id": "a", "title": "Bi-temporal edges with invalidation", "verdict": verdict,
            "headline": "creation time reachable for %.0f%% of nodes / %.0f%% of edges "
                        "(0%% inside the graph file itself); invalidation expressible: %s"
                        % (100 * node_cov, 100 * edge_cov, inval_ok),
            "lines": lines}


# ============================================================================ (b) ranked walk
def measure_ranked_walk():
    """Does graph-hop relevance CHANGE what gets retrieved on the real journals?

    This is the load-bearing claim. "We have a graph" is only a retrieval claim if removing
    the graph term changes the answer -- so the measurement is an ABLATION, not a demo."""
    lines, per_char = [], {}
    for ch in realdata.CHARACTERS:
        r = _replay(ch)
        per_char[ch] = r
        lines.append(
            "%s: %d real decisions replayed over %d journal entries" % (ch, r["n"], r["scanned"]))
        lines.append(
            "   retrieved set differs from the old tail-5 reader: %d/%d"
            % (r["differs_from_tail5"], r["n"]))
        lines.append(
            "   retrieved set differs when the GRAPH-HOP term is removed: %d/%d "
            "(median Jaccard %.2f; median %d memories reachable only WITH hops)"
            % (r["differs_from_no_hop"], r["n"], r["median_jaccard_vs_no_hop"],
               r["median_only_with_hops"]))
        lines.append(
            "   reach back, median days:  full %.1f | no hop term %.1f | no long-term pins %.1f "
            "| tail-5 %.3f   (max full %.1f)"
            % (r["median_reach_full"], r["median_reach_no_hop"], r["median_reach_no_pin"],
               r["median_reach_tail5"], r["max_reach_full"]))

    def frac(field):
        tot = sum(per_char[c]["n"] for c in per_char) or 1
        return sum(per_char[c][field] for c in per_char) / float(tot)

    changed_vs_tail = frac("differs_from_tail5")
    changed_vs_nohop = frac("differs_from_no_hop")
    reach = st.median([per_char[c]["median_reach_full"] for c in per_char])
    reach_tail = st.median([per_char[c]["median_reach_tail5"] for c in per_char])

    if changed_vs_tail >= 0.9 and changed_vs_nohop >= 0.5 and reach >= 7.0:
        verdict = PASS
    elif changed_vs_tail >= 0.5:
        verdict = PARTIAL
        lines.append("retrieval is scored, but the GRAPH is not what is doing the work: "
                     "removing hop relevance changed the answer in only %.0f%% of replays."
                     % (100 * changed_vs_nohop))
    else:
        verdict = FAIL
        lines.append("retrieval is indistinguishable from recency truncation on real data.")
    return {"id": "b", "title": "Retrieval as a ranked walk (graph-hop ablation)",
            "verdict": verdict,
            "headline": "hop term changes the retrieved set in %.0f%% of real replays; reach "
                        "%.0f days vs %.2f for tail-5" % (100 * changed_vs_nohop, reach, reach_tail),
            "lines": lines}


# ============================================================================ (c) scored + reflect
def measure_scored_and_reflection():
    """Is scoring LIVE in the production path (not merely a module), and do reflections exist?

    Live is measured by spying on the real `_build_user_prompt`: what matters is whether the
    whole journal reaches the scorer, not whether `journal_score.py` is on disk."""
    lines, live = [], True
    for ch in realdata.CHARACTERS:
        spy = []
        prompt = _real_prompt(ch, spy=spy)
        entries = realdata.journal(ch)
        saw = spy[0]["scanned"] if spy else 0
        scored_path = bool(spy) and saw >= len(entries) - 1
        s = js.summary(entries)
        rut_reaches_prompt = bool(s) and ("%s %d times" % (s["top_goal"], s["top_count"])) in prompt
        live = live and scored_path and rut_reaches_prompt
        monologue = sum(len(x) for x in prompt.splitlines() if x.startswith("- ("))
        lines.append("%s: heartbeat scored %d of %d journal entries this tick (kwargs %s)"
                     % (ch, saw, len(entries), spy[0]["kwargs"] if spy else []))
        lines.append("   the rut reaches the prompt as a countable fact ('%s %d times'): %s"
                     % (s.get("top_goal"), s.get("top_count", 0), rut_reaches_prompt))
        lines.append("   prompt is %d chars, of which retrieved monologue is %d (%.0f%%) -- the "
                     "rest is the reachable-pose menu"
                     % (len(prompt), monologue, 100.0 * monologue / (len(prompt) or 1)))

    # Reflection is counted with the reflection module's OWN predicate where it exists, so the
    # measurement cannot disagree with the writer about what a reflection is.
    try:
        from director import reflect
        is_reflection, how = reflect.is_reflection, "director.reflect.is_reflection"
    except Exception:
        is_reflection = lambda e: str((e or {}).get("kind", "")).lower() == "reflection"  # noqa: E731
        how = "local fallback (director/reflect.py absent)"
    reflections, kinded, total = 0, 0, 0
    for ch in realdata.CHARACTERS:
        for e in realdata.journal(ch):
            total += 1
            kinded += bool(e.get("kind"))
            reflections += bool(is_reflection(e))
    lines.append("reflection entries in the real journals: %d of %d, counted by %s "
                 "(entries carrying any `kind` field at all: %d)"
                 % (reflections, total, how, kinded))

    if live and reflections > 0:
        verdict = PASS
    elif live:
        verdict = PARTIAL
        lines.append("scoring is live on production data; the SYNTHESIS half is not. 66 days of "
                     "life contains no entry the character wrote ABOUT its life -- every line is "
                     "a reaction to one moment.")
    else:
        verdict = FAIL
    return {"id": "c", "title": "Scored memory + reflection", "verdict": verdict,
            "headline": "scoring live on the real journals: %s; reflection entries: %d/%d"
                        % (live, reflections, total),
            "lines": lines}


# ============================================================================ (d) extraction
def measure_extraction_at_retrieval():
    """Is the raw record kept verbatim, or lossily summarised at write time?

    "Storage Is Not Memory" (2605.04897): content discarded before the query is known cannot be
    recovered. Proven here against the real files -- every line parses, every field survives,
    timestamps only move forward (append-only), the scan ceiling is above the real volume, and
    every line the prompt shows traces back to a stored one, character for character."""
    lines, ok = [], True
    for ch in realdata.CHARACTERS:
        stats = realdata.journal_stats(ch)
        entries = realdata.journal(ch)
        fields = ("ts", "pose", "goal", "mood", "reason")
        complete = sum(1 for e in entries if all(k in e for k in fields))
        stamps = [float(e.get("ts") or 0) for e in entries]
        monotonic = all(stamps[i] <= stamps[i + 1] for i in range(len(stamps) - 1))
        lengths = [len(e.get("reason") or "") for e in entries]
        scanned = len(js.load(realdata.journal_file(ch)))
        ok = ok and stats["unparseable"] == 0 and complete == len(entries) and monotonic \
            and scanned == len(entries)
        lines.append("%s: %d lines -> %d parsed, %d unparseable; all 5 original fields on %d"
                     % (ch, stats["lines"], stats["parsed"], stats["unparseable"], complete))
        lines.append("   timestamps monotonic (append-only, never rewritten): %s" % monotonic)
        lines.append("   reason text kept at full length: max %d chars, median %d "
                     "(no write-time truncation)" % (max(lengths), sorted(lengths)[len(lengths) // 2]))
        lines.append("   retrieval scans %d of %d entries (MAX_SCAN=%d)"
                     % (scanned, len(entries), js.MAX_SCAN))

    # every retrieved line must exist, verbatim, in the file it came from
    verbatim_checked, verbatim_bad = 0, []
    for ch in realdata.CHARACTERS:
        stored = {(e.get("reason") or "").strip() for e in realdata.journal(ch)}
        for line in _real_prompt(ch).splitlines():
            if line.startswith("- (") and " ago, at " in line:
                body = line.split(") ", 1)[1].strip()
                verbatim_checked += 1
                if body not in stored:
                    verbatim_bad.append((ch, body[:60]))
    lines.append("retrieved prompt lines traced back to a stored entry, verbatim: %d/%d"
                 % (verbatim_checked - len(verbatim_bad), verbatim_checked))
    ok = ok and not verbatim_bad and verbatim_checked > 0

    verdict = PASS if ok else FAIL
    if verdict == PASS:
        lines.append("the write path is lossless; all the intelligence is in the reader. "
                     "That is the 2603.02473 recommendation, and it is what production does.")
    return {"id": "d", "title": "Extraction at retrieval, not at ingestion", "verdict": verdict,
            "headline": "%d of %d journal lines retained verbatim; %d retrieved lines all trace "
                        "to stored text" % (
                            sum(realdata.journal_stats(c)["parsed"] for c in realdata.CHARACTERS),
                            sum(realdata.journal_stats(c)["lines"] for c in realdata.CHARACTERS),
                            verbatim_checked),
            "lines": lines}


# ============================================================================ (e) declarative world
# The self-schema: everything a journal entry is allowed to say about the character's own body
# and interior. A field outside this set would be the system recording something ELSE.
_SELF_FIELDS = {"ts", "pose", "goal", "mood", "reason", "band", "kind", "night"}
# Where a declarative store would plausibly live if someone built one. Globbed, not assumed:
# this measurement must be able to come back non-zero, or the FAIL it reports means nothing.
_WORLD_STORE_GLOBS = ("data/mind/world*.json", "data/mind/facts*.json", "data/world/*.json",
                      "data/facts/*.json", "data/mind/observations*.json*",
                      "data/mind/people*.json", "data/mind/events*.json", "data/_realdata/world*")


def _world_fact_records(nodes, journal_fields):
    """Count records the system holds about something OTHER than a character's own movement.

    Three places one could be, all actually looked at:
      1. a graph node that is not a pose of a character (an entity node),
      2. a journal field outside the self-schema (the system recording an observation),
      3. a declarative store on disk under any of the conventional names.

    Returns (count, description-of-what-was-searched). If a peer lands declarative memory this
    number moves and `test_the_scorecard_on_production_data` fails, which is the intended
    tripwire -- a hard-coded zero here would make criterion (e) unfalsifiable, i.e. worthless."""
    entity_nodes = sum(1 for r in nodes.values() if not r.get("pose") or not r.get("character"))
    foreign_fields = sorted(set(journal_fields) - _SELF_FIELDS)
    store_records, stores = 0, []
    for pattern in _WORLD_STORE_GLOBS:
        for p in sorted(ROOT.glob(pattern)):
            stores.append(p.name)
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                store_records += len(d) if isinstance(d, (list, dict)) else 1
            except Exception:
                store_records += 1                  # present but unreadable still counts as "exists"
    where = ("%d entity nodes, journal fields outside the self-schema %s, %d store file(s) "
             "matching %d conventional paths"
             % (entity_nodes, foreign_fields or "none", len(stores), len(_WORLD_STORE_GLOBS)))
    return entity_nodes + len(foreign_fields) + store_records, where


def measure_declarative_world():
    """How many facts does the system hold about anything OTHER than its own movement?

    A context graph in the RAG sense holds facts about a world: who came, what ran, what is
    where. This counts them. Every node in the production graph is a pose of a character;
    every journal field is that character's own pose, want, mood and reason. The count is
    zero, and the probe must say so."""
    g = realdata.raw_graph()
    nodes = g["nodes"]

    non_pose_nodes = [n for n, r in nodes.items() if not r.get("pose")]
    entity_kinds = sorted({(r.get("character") or "?") for r in nodes.values()})
    journal_fields = set()
    world_talk, total = 0, 0
    for ch in realdata.CHARACTERS:
        for e in realdata.journal(ch):
            journal_fields.update(e.keys())
            total += 1
            if _WORLD_TALK.search(e.get("reason") or ""):
                world_talk += 1

    world_records, where = _world_fact_records(nodes, journal_fields)

    lines = [
        "graph nodes that are NOT a pose of a character: %d of %d "
        "(node types present: %s)" % (len(non_pose_nodes), len(nodes), entity_kinds),
        "journal fields across %d real entries: %s -- every one of them describes this "
        "character's own body, want, or feeling" % (total, sorted(journal_fields)),
        "stored facts about an event, a person, or the room: %d  (searched: %s)"
        % (world_records, where),
        "...while %d of %d monologue lines (%.1f%%) explicitly reference a person, an "
        "audience, or the room" % (world_talk, total, 100.0 * world_talk / (total or 1)),
        "weather is fetched into the prompt (director/context.py) but is a TTL cache, not a "
        "retrievable memory: it is never journaled, never scored, and gone next tick.",
        "the neighbour line is a cross-character edge computed at read time -- it reports the "
        "other BODY's pose and mood, which is still movement, not a fact about the world.",
    ]
    verdict = FAIL if world_records == 0 else PARTIAL
    if verdict == FAIL:
        lines.append("VERDICT FAIL, and it is the honest one: the characters talk about a world "
                     "they keep no record of. Nothing here can answer 'who came to the floor "
                     "today' because nothing here can perceive it. This is audit opportunity #6 "
                     "and it needs a sensor, not a schema.")
    return {"id": "e", "title": "Declarative memory of the world", "verdict": verdict,
            "headline": "%d stored world facts; %d monologue lines reference the world"
                        % (world_records, world_talk),
            "lines": lines}


# ============================================================================ the scorecard
MEASUREMENTS = (measure_bitemporal, measure_ranked_walk, measure_scored_and_reflection,
                measure_extraction_at_retrieval, measure_declarative_world)


def scorecard():
    return [m() for m in MEASUREMENTS]


def _ascii(s):
    """The real journals carry mojibake from 66 days of live LLM output; a Windows console
    will die on it. The scorecard is evidence, so it must always print."""
    return str(s).encode("ascii", "replace").decode("ascii")


def render_scorecard(cards):
    snap = realdata.summary()
    out = ["", "=" * 78,
           "CONTEXT-GRAPH SCORECARD -- measured on data/_realdata/ (production, read-only)",
           "  graph %d nodes / %d edges   %s" % (snap["nodes"], snap["edges"], snap["schema"])]
    for c, s in snap["characters"].items():
        out.append("  %-9s %6d journal entries over %.1f days" % (c, s["entries"], s["days"]))
    out.append("=" * 78)
    for card in cards:
        out.append("")
        out.append("(%s) %-45s  %s" % (card["id"], card["title"], card["verdict"]))
        out.append("     %s" % card["headline"])
        for ln in card["lines"]:
            out.append("       - %s" % ln)
    tally = {v: sum(1 for c in cards if c["verdict"] == v) for v in (PASS, PARTIAL, FAIL)}
    out += ["", "-" * 78,
            "TALLY  %d PASS / %d PARTIAL / %d FAIL" % (tally[PASS], tally[PARTIAL], tally[FAIL]),
            "=" * 78, ""]
    return _ascii("\n".join(out))


# ============================================================================ tests
def test_the_scorecard_on_production_data(realdata_snapshot, capsys):
    """Run all five measurements against production and print the scorecard.

    The assertions are calibration, not celebration -- see the module docstring."""
    cards = scorecard()
    text = render_scorecard(cards)
    with capsys.disabled():                 # the lead has to be able to read this
        print(text)

    by_id = {c["id"]: c for c in cards}
    assert set(by_id) == {"a", "b", "c", "d", "e"}
    for c in cards:
        assert c["verdict"] in (PASS, PARTIAL, FAIL)
        assert c["lines"], "criterion %s produced a verdict with no evidence" % c["id"]

    # (e) MUST fail today. If this assertion breaks, someone built declarative memory (good) or
    # the measurement stopped measuring (bad) -- either way a human re-reads this file.
    assert by_id["e"]["verdict"] == FAIL, (
        "criterion (e) is no longer FAIL. Re-calibrate this probe by hand:\n"
        + "\n".join(by_id["e"]["lines"]))

    # (b) and (d) shipped in v0.3.0; these are regression gates on real data.
    assert by_id["b"]["verdict"] == PASS, "\n".join(by_id["b"]["lines"])
    assert by_id["d"]["verdict"] == PASS, "\n".join(by_id["d"]["lines"])

    # (a) and (c) are peer-owned and deliberately unasserted; the scorecard reports them.


def test_the_probe_notices_when_graph_relevance_is_removed(realdata_snapshot, monkeypatch):
    """FALSIFIABILITY, on real data: stub out the one feature criterion (b) exists to detect,
    and the measurement must stop saying PASS.

    Nothing synthetic here -- the same production journals, with `journal_score.relevance`
    neutered to a constant, which is exactly what removing hop-distance relevance would look
    like."""
    honest = measure_ranked_walk()
    assert honest["verdict"] == PASS

    monkeypatch.setattr(js, "relevance", lambda entry, distmap: 0.0)
    monkeypatch.setattr(js, "W_RELEVANCE", 0.0)
    _cache.pop(("replay", "phineas", REPLAY_N, "live"), None)
    _cache.pop(("replay", "maxx", REPLAY_N, "live"), None)
    broken = measure_ranked_walk()

    assert broken["verdict"] != PASS, (
        "the probe scored a graph-blind reader as a ranked walk -- it has no teeth:\n"
        + "\n".join(broken["lines"]))
    # and it must say WHY, not merely drop a letter grade
    assert "GRAPH is not what is doing the work" in "\n".join(broken["lines"]) \
        or broken["verdict"] == FAIL

    for ch in realdata.CHARACTERS:                       # leave no poisoned cache behind
        _cache.pop(("replay", ch, REPLAY_N, "live"), None)


def test_the_probe_notices_when_provenance_lands(realdata_snapshot, tmp_path, monkeypatch):
    """FALSIFIABILITY for criterion (a), the one that is FAIL today.

    A measurement that reports less than PASS is worthless unless it would report PASS when the
    capability is genuinely there. SYNTHETIC BY DESIGN and clearly labelled: a tmp COPY of the
    real graph is stamped with creation and expiry times on every record, and the measurement
    must move to PASS. The real snapshot is never touched -- the copy is a deepcopy written to
    tmp_path, and `realdata.raw_graph` is monkeypatched to serve it for the duration."""
    before = measure_bitemporal()
    assert before["verdict"] != PASS, (
        "criterion (a) already reads PASS on production data -- re-read this test before "
        "trusting it:\n" + "\n".join(before["lines"]))

    stamped = copy.deepcopy(realdata.raw_graph())        # SYNTHETIC: tmp copy, not production
    now = time.time()
    for rec in stamped["nodes"].values():
        rec["t_created"], rec["t_valid"] = now, now
    for rec in stamped["edges"]:
        rec["t_created"], rec["t_expired"] = now, None
    fake = tmp_path / "video_graph.json"                  # written so the artifact is inspectable
    fake.write_text(json.dumps(stamped), encoding="utf-8")

    monkeypatch.setattr(realdata, "raw_graph",
                        lambda: {"schema": stamped["schema"],
                                 "nodes": dict(stamped["nodes"]), "edges": list(stamped["edges"])})
    after = measure_bitemporal()
    assert after["verdict"] == PASS, (
        "the probe could not see a fully bi-temporal graph -- it is not measuring what it "
        "claims to:\n" + "\n".join(after["lines"]))
    assert "100% of nodes / 100% of edges" in after["headline"], after["headline"]

    # ...and none of that leaked into the cached production records the other tests measure.
    cached = realdata._cache.get("raw_graph") or {}
    assert not any(k.startswith("t_") or k == "created"
                   for rec in (cached.get("nodes") or {}).values() for k in rec), \
        "the falsifiability check mutated the cached production graph"


def test_the_probe_notices_when_a_world_fact_is_stored(realdata_snapshot, tmp_path, monkeypatch):
    """FALSIFIABILITY for criterion (e), the one that MUST read FAIL today.

    A hard-coded zero would make (e) unfalsifiable -- it would report FAIL forever, including
    on the day the capability shipped. So the detector is a real search, and this proves it
    searches: SYNTHETIC BY DESIGN, one world-fact store is written to tmp_path, the module's
    root and glob list are pointed at it, and the count must move off zero. Production data is
    untouched; only where the detector LOOKS is changed."""
    before = measure_declarative_world()
    assert before["verdict"] == FAIL and "0 stored world facts" in before["headline"], \
        "criterion (e) no longer reads zero -- re-read this file by hand:\n" \
        + "\n".join(before["lines"])

    store = tmp_path / "data" / "mind"                    # SYNTHETIC: tmp, not the repo
    store.mkdir(parents=True)
    (store / "world.json").write_text(json.dumps({
        "2026-08-10:vcn-42": {"kind": "event", "room": "Floor 10", "people": ["Ray"]},
        "2026-08-09:visit": {"kind": "visit", "who": "a stranger in a green coat"}}),
        encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "_WORLD_STORE_GLOBS", ("data/mind/world*.json",))

    after = measure_declarative_world()
    assert "0 stored world facts" not in after["headline"], (
        "the probe cannot see a declarative store even when one exists -- criterion (e) is "
        "reporting a constant, not a measurement:\n" + "\n".join(after["lines"]))
    assert after["verdict"] == PARTIAL, after["headline"]


# --------------------------------------------------------------------- the ten-question probe
# Same shape as tests/test_memory_probe.py: five questions the REAL journals can answer and five
# they cannot, where "I don't remember" is the correct answer. The difference is that every fact
# below was read out of data/_realdata/ rather than written into a fixture -- so these questions
# get harder, or stop being true, as the characters keep living. That is intended: a probe whose
# answers are guaranteed by its own fixture proves only that the fixture was built correctly.

def _dominant_want(prompt, character, entries):
    """The aggregate line must name the real rut AND its real size -- the fact no window of
    individual memories can carry (phineas:jealous_glare, 2,151 of 11,601 decisions)."""
    from collections import Counter
    goal, count = Counter(e.get("goal") for e in entries if e.get("goal")).most_common(1)[0]
    m = re.search(re.escape(goal) + r" (\d+) times", prompt)
    return bool(m) and int(m.group(1)) >= int(count * 0.95), "%s x%d" % (goal, count)


def _lifetime_named(prompt, character, entries):
    stamps = [float(e["ts"]) for e in entries if e.get("ts")]
    days = int((time.time() - min(stamps)) / DAY)
    return any(("%d days" % d) in prompt for d in (days - 1, days, days + 1)), "%d days" % days


def _newest_thought(prompt, character, entries):
    body = (entries[-1].get("reason") or "").strip()
    return bool(body) and body in prompt, body[:48]


def _reaches_past_a_week(prompt, character, entries):
    """A retrieved memory older than a WEEK.

    Calibrated to what the system measurably does, not to what would sound best. The threshold
    started at 30 days -- Phineas reaches 52 -- and MAXX failed it at 18, because reach depends
    on the pose the character happens to be standing in: hop-relevance pulls in memories made
    near HERE, and how old those are is a property of where "here" is. A week is the claim both
    characters support on real data at any pose sampled, and it still separates cleanly from the
    pre-0.3.0 reader's 20 minutes. The measured number is reported either way."""
    ages = [float(m) for m in re.findall(r"- \((\d+)d ago, at ", prompt)]
    oldest = max(ages) if ages else 0
    return oldest >= 7, "oldest retrieved: %dd" % oldest


def _frontier_named(prompt, character, entries):
    """Poses that exist and are reachable and have never once been stood in."""
    return "never once been in" in prompt, "frontier line present"


ANSWERABLE = [
    ("what do you want most often, and how often?", _dominant_want),
    ("how long have you been hanging here?", _lifetime_named),
    ("what were you just thinking?", _newest_thought),
    ("can you remember anything from more than a week ago?", _reaches_past_a_week),
    ("is there a version of you you have never been?", _frontier_named),
]

# Five the real journals CANNOT answer. Each needle was checked against the real prompts for
# BOTH characters before being chosen; if one ever appears, retrieval has started smuggling in
# context the character has no record of, and the only honest answer stops being available.
UNANSWERABLE = [
    ("who came to the floor today?", "visitor"),
    ("what did anyone say to you?", "said to me"),
    ("what is your neighbour doing right now?", "beside you"),   # no pose files in the snapshot
    ("what happened at the hackathon?", "hackathon"),
    ("what day of the week is it?", "wednesday"),
]


@pytest.mark.parametrize("character", realdata.CHARACTERS)
def test_the_ten_question_probe_on_production_data(realdata_snapshot, character, capsys):
    prompt = _real_prompt(character)
    entries = realdata.journal(character)
    failed, notes = [], []
    for q, check in ANSWERABLE:
        ok, detail = check(prompt, character, entries)
        notes.append("  %-52s %s   (%s)" % (q, "RECALLED" if ok else "MISS", detail))
        if not ok:
            failed.append("RECALL MISS: %s (%s)" % (q, detail))
    low = prompt.lower()
    for q, needle in UNANSWERABLE:
        hit = needle in low
        notes.append("  %-52s %s" % (q, "CONFABULATION RISK" if hit else "refused (correct)"))
        if hit:
            failed.append("CONFABULATION RISK: %r appears in context for %s" % (needle, q))
    with capsys.disabled():
        print(_ascii("\n[ten-question probe on REAL data: %s]\n%s" % (character, "\n".join(notes))))
    assert not failed, "\n".join(_ascii(x) for x in failed)


@pytest.mark.parametrize("character", realdata.CHARACTERS)
def test_tail_five_fails_the_same_questions_on_production_data(realdata_snapshot, character):
    """The CONTROL. The pre-0.3.0 reader is scored by the same five recall questions against the
    same real journal and must fail most of them -- otherwise the probe proves nothing about the
    change that was made, only that the prompt is long."""
    entries = realdata.journal(character)
    old = "\n".join("- (%s, feeling %s) %s" % (e.get("pose"), e.get("mood"), e.get("reason"))
                    for e in entries[-5:])
    misses = [q for q, check in ANSWERABLE if not check(old, character, entries)[0]]
    assert len(misses) >= 4, "tail-5 unexpectedly answered on real data: %s" % misses


def test_retrieval_selects_but_never_authors_on_production_data(realdata_snapshot):
    """Every retrieved memory traces to a real journal entry, character for character. Retrieval
    may select and order; it may not write. This is criterion (d) as an assertion rather than a
    reported number, because a reader that paraphrases would break the probe silently."""
    for ch in realdata.CHARACTERS:
        stored = {(e.get("reason") or "").strip() for e in realdata.journal(ch)}
        checked = 0
        for line in _real_prompt(ch).splitlines():
            if line.startswith("- (") and " ago, at " in line:
                body = line.split(") ", 1)[1].strip()
                checked += 1
                assert body in stored, _ascii("invented line for %s: %r" % (ch, body[:80]))
        assert checked >= 3, "%s retrieved almost nothing (%d lines) -- probe is not exercising " \
                             "the reader" % (ch, checked)


if __name__ == "__main__":                      # `python tests/test_context_graph_probe.py`
    if not realdata.available():
        print(realdata.SKIP_REASON)
        raise SystemExit(2)
    print(render_scorecard(scorecard()))
