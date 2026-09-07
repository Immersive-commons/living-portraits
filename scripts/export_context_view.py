"""export_context_view.py -- one JSON the browser can hold the whole context graph in.

The viewer (`graph_viewer.html`) was written in May against `video_graph.json`, which is an
ASSET MANIFEST: poses a generator made, clips it rendered. Everything the 0.3.x/0.4.x work
added lives in OTHER files on purpose -- `video_graph.build()` regenerates the graph roughly
every twenty minutes and would erase anything the runtime wrote into it
(`runtime/lived.py`, `runtime/graph_provenance.py`). So the context half of the graph is
spread across six single-writer files plus two derivations that only exist in Python.

This joins them, once, into `data/graph/context_view.json`:

    video_graph.json          the map        nodes + edges (live view, never history)
    graph_provenance.stamp    TRANSACTION t  when the system learned each thing exists
    lived/<char>.json         VALID t        when a character actually stood there, how long
    edge_style.index          typing         manner + valence derived from each clip's prose
    journal/<char>.jsonl      memory         what was wanted, said, and reflected on
    journal_score.select      retrieval      what THIS character is remembering right now
    pose/<char>.json          the body       where each panel is standing this second
    intent.json               the want       the goal the brain last set, with its mood band
    policy.weigh              the choice     the live weight on every exit from where it stands

READ-ONLY over production. It never calls `video_graph.build()`, never writes back into
video_graph.json, and never touches the journals -- a viewer that can perturb the thing it
is viewing is not an instrument. `gp.stamp()` mutates the graph dicts we loaded, which is
why they are loaded here and thrown away rather than saved.

Fail-soft per section: a missing journal or an unreadable lived file costs that section and
leaves the rest, because the point of the artifact is to show what IS there. Every section
that could be silently thin instead reports its own coverage -- 464 untyped edges are
reported untyped rather than quietly called neutral.

    python scripts/export_context_view.py            # -> data/graph/context_view.json
    python scripts/export_context_view.py --stdout   # print, write nothing
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime import circadian, edge_style, journal_score, pathfind, policy  # noqa: E402
from runtime import graph_provenance as gp  # noqa: E402
from runtime import lived as lived_mod  # noqa: E402

SCHEMA = "living-portrait.context-view/v1"
OUT = ROOT / "data" / "graph" / "context_view.json"
GRAPH = ROOT / "data" / "clips" / "video_graph.json"
MIND = ROOT / "data" / "mind"
BEDTIME = ROOT / "prompts" / "bedtime_routine.json"

TOP_WANTS = 14          # goal histogram rows kept per character
REFLECTIONS = 6         # nightly reflections kept per character
DECISION_ROWS = 14      # exits shown on the decision surface
SAID_CHARS = 260        # per-node "last thing said here", truncated


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


# --------------------------------------------------------------------------- codemap
# WHAT PRODUCES WHAT. Hand-authored (a machine cannot say what a module is *for*), but every
# row is checked against disk below: a path that has moved shows up as `exists: false`
# instead of quietly describing a file that is not there.
CODEMAP = [
    # layer, path, what it contributes to this artifact, which lens/panel shows it
    ("map", "runtime/video_graph.py",
     "Builds the graph from NODE_SPECS/EDGE_SPECS and merges the bedtime + autogen additions. "
     "Regenerates video_graph.json wholesale, which is why nothing the runtime learns can live in it.",
     "Structure lens"),
    ("map", "runtime/clip_graph.py",
     "The older per-(from,to) clip index the video graph superseded. Still on disk, not on the walk.",
     "-"),
    ("map", "runtime/pathfind.py",
     "One BFS per tick: hop distance from where a character stands. Serves the goal gradient, "
     "memory relevance, and the frontier.",
     "Frontier lens / goal path"),
    ("time", "runtime/graph_provenance.py",
     "TRANSACTION time. `created` derived from each artifact's mtime so it survives a rebuild; "
     "invalidation is a diff against the last graph, so a vanished clip is tombstoned with its record.",
     "Provenance lens + scrubber"),
    ("time", "runtime/lived.py",
     "VALID time. Sole writer per character: visits, first/last stood, dwell, mood bands brought, "
     "and which clips have ACTUALLY rolled as opposed to merely existing.",
     "Lived lens"),
    ("time", "scripts/backfill_lived.py",
     "Recovered 76 days of traversal from the walker's own pick log. Plays come back at LABEL level "
     "only; timestamps do not come back at all, because the log carries no date.",
     "Honesty tab"),
    ("typing", "runtime/edge_style.py",
     "Manner + valence derived from each clip's own motion prompt. Also proves which edges carry "
     "their twin's prose and inverts those.",
     "Manner lens"),
    ("choice", "runtime/policy.py",
     "The one softmax the body actually walks: novelty, anti-reverse (forgiven with dwell), "
     "soft goal pull, mood band, escape velocity, manner preference last.",
     "Decision lens + now strip"),
    ("choice", "runtime/circadian.py",
     "Night is non-negotiable and owns the body: the bedtime chain, the sleep dwell, and the mask "
     "that keeps the daytime walk out of the bedroom.",
     "Structure lens (bedtime)"),
    ("choice", "runtime/mind.py",
     "Reads intent.json and hands the walker a goal, with a TTL so a dead brain never freezes a panel.",
     "Now strip"),
    ("memory", "runtime/journal_score.py",
     "Scored retrieval over the whole journal: recency decay + importance as surprisal of the want "
     "+ relevance in transition hops, bounded by a token budget, with slots reserved for distant memories.",
     "Memory tab"),
    ("memory", "director/reflect.py",
     "Nightly reflection inside the circadian dwell. Lands in the same JSONL with no `goal` key, so it "
     "never pollutes the want histogram and retrieval picks it up for free.",
     "Memory tab (reflections)"),
    ("memory", "director/heartbeat.py",
     "The brain loop: senses pose + neighbours + world, retrieves memory, asks the model for a goal "
     "and a mood, journals it, and proposes new poses.",
     "Now strip / Memory tab"),
    ("memory", "director/llm.py",
     "z.ai gateway with a local qwen3 fallback and a circuit breaker; balanced-object JSON extraction.",
     "-"),
    ("world", "director/context.py",
     "The world seam: weather, hour, daypart -- as conditions, cached, never journaled. The declarative "
     "gap the audit still scores as a FAIL.",
     "Now strip / Honesty tab"),
    ("body", "_preview_graph.py",
     "The walker. Renders both panels at ~10fps, writes pose/<char>.json and the lived record, and is "
     "the only process that decides what plays next.",
     "Everything under Now"),
    ("body", "runtime/panels.py",
     "Panel geometry and the borderless window pinned to the physical LED panels.",
     "-"),
    ("growth", "pipeline/autogen.py",
     "Turns an approved proposal into a pose: still, transition, real reverse, idles -- inside the "
     "daily budget rail, merged walk-safe.",
     "Provenance lens (new nodes)"),
    ("growth", "director/mj_safe.py",
     "The prompt linter that keeps a generator from tripping a moderation block.",
     "-"),
    ("proof", "health/checks.py",
     "Eight deterministic detectors, one per incident a human had to notice. Off-by-choice is a PASS.",
     "Honesty tab"),
    ("proof", "scripts/e2e_viewer.py",
     "Drives graph_viewer.html in a real browser across four viewports -- every lens, tab, "
     "control and the scrubber -- and reports the console errors a 200 hides. The viewer is "
     "the only surface pytest cannot reach.",
     "Everything in this artifact"),
    ("proof", "tests/test_context_graph_probe.py",
     "Replays real decisions against the retrieval layer and asserts it reaches past `tail -5`.",
     "Memory tab"),
    ("view", "scripts/export_context_view.py",
     "This file: joins the six single-writer stores plus two Python-only derivations into one JSON.",
     "-"),
    ("view", "graph_viewer.html",
     "The artifact. Six lenses over one graph, a scrubber over transaction time, and the live now strip.",
     "-"),
]


def codemap():
    rows = []
    for layer, rel, role, shows in CODEMAP:
        p = ROOT / rel
        row = {"layer": layer, "path": rel, "role": role, "shows": shows, "exists": p.exists()}
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                row["lines"] = len(text.splitlines())
                row["mtime"] = p.stat().st_mtime
                if text.lstrip().startswith(('"""', "'''")):
                    body = text.lstrip()[3:]
                    first = body.split("\n", 1)[0].strip() or body.split("\n")[1].strip()
                    row["summary"] = first[:180]
            except Exception:
                pass
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- sections
def graph_sections(now):
    """The map, stamped with transaction time and typed by manner. Returns (nodes, edges, meta)."""
    g = _read_json(GRAPH, {}) or {}
    nodes = g.get("nodes") or {}
    edges = g.get("edges") or []
    stamped = gp.stamp(nodes, edges, root=ROOT)      # in memory only -- never written back
    styles = edge_style.index(edges)
    coverage = edge_style.coverage(edges)
    store = gp.load_store()

    out_edges = []
    for e in edges:
        st = styles.get(e.get("id"))
        rec = {k: e.get(k) for k in
               ("id", "character", "kind", "label", "variant", "from", "to",
                "gif", "motion_prompt", "loop", "created")}
        if st is not None:
            rec["manner"] = st.manner
            rec["valence"] = round(st.valence, 3)
            rec["style_score"] = round(st.score, 2)
        out_edges.append(rec)

    meta = {
        "graph_schema": g.get("schema"),
        "stamped": stamped,
        "style_coverage": coverage,
        "provenance": gp.summary(store, now=now),
        "tombstones": {
            "edges": [{"id": k, "invalidated": v.get("invalidated")}
                      for k, v in list((store.get("edges") or {}).items())[:200]],
            "nodes": [{"id": k, "invalidated": v.get("invalidated")}
                      for k, v in list((store.get("nodes") or {}).items())[:200]],
        },
    }
    return nodes, out_edges, meta


def merge_lived(nodes, edges, characters):
    """VALID time onto the map: what has been stood in and what has actually rolled."""
    books, stats = {}, {}
    for c in characters:
        try:
            books[c] = lived_mod.load(c)
        except Exception:
            continue
    for nid, n in nodes.items():
        b = books.get(n.get("character"))
        rec = (b.nodes.get(nid) if b else None) or {}
        n["visits"] = int(rec.get("visits", 0))
        n["dwell"] = int(rec.get("dwell", 0))
        for k in ("first", "last", "backfilled"):
            if rec.get(k) is not None:
                n[k] = rec[k]
        if rec.get("moods"):
            n["moods"] = rec["moods"]
    for e in edges:
        b = books.get(e.get("character"))
        rec = (b.edges.get(e.get("id")) if b else None) or {}
        e["plays"] = int(rec.get("plays", 0))
        for k in ("first", "last"):
            if rec.get(k) is not None:
                e["play_" + k] = rec[k]
    for c, b in books.items():
        nvis = sum(1 for k, v in b.nodes.items() if v.get("visits"))
        stats[c] = {
            "nodes_recorded": len(b.nodes),
            "nodes_visited": nvis,
            "visits": sum(int(v.get("visits", 0)) for v in b.nodes.values()),
            "dwell_units": sum(int(v.get("dwell", 0)) for v in b.nodes.values()),
            "clips_played": sum(1 for v in b.edges.values() if v.get("plays")),
            "plays": sum(int(v.get("plays", 0)) for v in b.edges.values()),
            "backfill": (b._extra or {}).get("backfill"),
        }
    return stats, books


def memory_section(character, node, hops, now):
    """What this character has said, wanted, and understood -- and what it is remembering NOW."""
    path = MIND / "journal" / (character + ".jsonl")
    try:
        entries = journal_score.load(path)
    except Exception:
        return {"available": False}
    if not entries:
        return {"available": False}

    counts = journal_score.goal_counts(entries)
    total = sum(counts.values()) or 1
    wants = [{"goal": g, "n": n, "share": round(n / total, 4)}
             for g, n in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_WANTS]]

    picked = []
    try:
        for e in journal_score.select(entries, now=now, distmap=hops or {}):
            ts = journal_score._ts(e) or 0
            picked.append({
                "ts": ts,
                "age_days": round((now - ts) / 86400.0, 2) if ts else None,
                "pose": e.get("pose"),
                "goal": e.get("goal"),
                "mood": e.get("mood"),
                "band": e.get("band"),
                "kind": e.get("kind"),
                "hops": (hops or {}).get(e.get("pose")),
                "reason": (e.get("reason") or "")[:600],
            })
    except Exception:
        pass

    reflections = [e for e in entries if e.get("kind") == "reflection"][-REFLECTIONS:]
    first_ts = journal_score._ts(entries[0]) or 0
    last_ts = journal_score._ts(entries[-1]) or 0
    ages = [p["age_days"] for p in picked if p.get("age_days") is not None]
    return {
        "available": True,
        "entries": len(entries),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "days": round((last_ts - first_ts) / 86400.0, 1) if first_ts and last_ts else None,
        "wants": wants,
        "aggregate_line": journal_score.aggregate_line(entries, now=now),
        "retrieved": picked,
        "reach_days": {"min": min(ages), "max": max(ages),
                       "median": sorted(ages)[len(ages) // 2]} if ages else None,
        "reflections": [{"ts": journal_score._ts(e), "night": e.get("night"), "pose": e.get("pose"),
                         "mood": e.get("mood"), "trigger": e.get("trigger"), "over": e.get("over"),
                         "reason": e.get("reason")} for e in reflections],
        "retrieval_params": {
            "token_budget": journal_score.TOKEN_BUDGET,
            "recency_decay_per_hour": journal_score.RECENCY_DECAY,
            "weights": {"recency": journal_score.W_RECENCY,
                        "importance": journal_score.W_IMPORTANCE,
                        "relevance": journal_score.W_RELEVANCE},
            "pins": journal_score.PIN_IMPORTANT,
            "pin_min_age_days": round(journal_score.PIN_MIN_AGE / 86400.0, 2),
            "max_scan": journal_score.MAX_SCAN,
            "control": "tail -5 (the reader this replaced)",
        },
    }


def node_voice(nodes, characters):
    """The last thing each character said while standing in each pose. One line, truncated.

    A pose is not just a still and a set of exits: it is where things were said. This is the
    cheapest honest version of that -- a count and the most recent line, not the whole log.
    """
    for c in characters:
        path = MIND / "journal" / (c + ".jsonl")
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                n = nodes.get(e.get("pose"))
                if n is None:
                    continue
                n["said_n"] = int(n.get("said_n", 0)) + 1
                if e.get("reason"):
                    n["said"] = e["reason"][:SAID_CHARS]
                    n["said_ts"] = e.get("ts")
                    n["said_mood"] = e.get("mood")
        except Exception:
            continue


def now_section(character, nodes, edges, spec, books, now):
    """Where the body is, what the brain wants, and the live weight on every exit.

    The decision surface is computed with the same call the walker makes
    (`_preview_graph._pick_policy`), so it is the real weighting and not a description of one.
    """
    pose = _read_json(MIND / "pose" / (character + ".json"), {}) or {}
    intent = _read_json(MIND / "intent.json", {}) or {}
    want = ((intent.get("characters") or {}).get(character)) or {}
    node = pose.get("node")
    hour = datetime.fromtimestamp(now).hour
    night = bool(circadian.is_night(spec, character, hour)) if spec else False

    mine = [e for e in edges if e.get("character") == character]
    masked = circadian.bedtime_labels(spec, character) if spec else set()
    walkable = mine if night else [e for e in mine if e.get("label") not in masked]
    hops = pathfind.hops_from(walkable, node) if node else {}

    out = {
        "node": node,
        "dwell": pose.get("dwell"),
        "last_label": pose.get("last_label"),
        "pose_updated": pose.get("updated"),
        "goal": want.get("goal"),
        "mood": want.get("mood"),
        "band": want.get("band"),
        "route": want.get("route", "wander"),
        "goal_set_at": want.get("set_at"),
        "hour": hour,
        "night": night,
        "sleep_node": circadian.sleep_node(spec, character) if spec else None,
        "hops": hops,
    }

    if node and out["goal"]:
        try:
            out["path"] = pathfind.shortest_path(walkable, node, out["goal"])
        except Exception:
            out["path"] = None

    book = books.get(character)
    visited = {k for k, v in (book.nodes if book else {}).items() if v.get("visits")}
    if node:
        try:
            reach = pathfind.reachable_poses(walkable, node)
            frontier = sorted(reach - visited - (circadian.bedtime_poses(spec, character) if spec else set()),
                              key=lambda g: (hops.get(g, 10 ** 6), g))
            out["reachable"] = len(reach)
            out["frontier"] = [{"node": g, "hops": hops.get(g)} for g in frontier]
        except Exception:
            pass

    # the live weighting, mirroring _preview_graph._pick_policy's arguments
    if node:
        try:
            styles = edge_style.index(mine)
            outs = [e for e in walkable if e.get("from") == node]
            dwell = 0 if out["sleep_node"] == node else int(pose.get("dwell") or 0)
            weights = policy.weigh(
                node, outs, walkable, goal=out["goal"], mood=out["mood"], band=out["band"],
                route=out["route"], exclude=masked, dwell=dwell, styles=styles,
                # `choose` derives this from the last label; the pose file carries the label,
                # so the anti-reverse guard is reconstructable. The novelty terms are NOT --
                # recent_clips/recent_nodes live only in the walker's memory.
                reverse_of=policy._reverse_of_last(pose.get("last_label")),
                context_energy=(_read_json(MIND / "context_signal.json", {}) or {})
                .get("signal", {}).get("energy"))
            tot = sum(w for _, w in weights) or 1.0
            rows = sorted(({"id": e.get("id"), "label": e.get("label"), "kind": e.get("kind"),
                            "to": e.get("to"), "weight": round(w, 4), "share": round(w / tot, 4)}
                           for e, w in weights), key=lambda r: -r["weight"])
            out["decision"] = {"exits": len(weights), "rows": rows[:DECISION_ROWS],
                               # AT NIGHT THESE WEIGHTS ARE NOT CONSULTED. `_pick_policy` asks
                               # circadian first and returns on a "force", so the bedtime chain and
                               # the sleep dwell own the body outright. Showing a softmax over a
                               # sleeping character's exits would be a picture of a decision nothing
                               # is making.
                               "consulted": not night,
                               "owner": "circadian (bedtime chain / sleep dwell)" if night else "policy.weigh"}
        except Exception as exc:
            out["decision"] = {"error": repr(exc)}
    return out, hops


def build(now=None):
    now = time.time() if now is None else now
    nodes, edges, meta = graph_sections(now)
    characters = sorted({v.get("character") for v in nodes.values() if v.get("character")})
    lived_stats, books = merge_lived(nodes, edges, characters)
    node_voice(nodes, characters)
    spec = _read_json(BEDTIME, {}) or {}

    now_by_char, memory_by_char = {}, {}
    for c in characters:
        block, hops = now_section(c, nodes, edges, spec, books, now)
        now_by_char[c] = block
        memory_by_char[c] = memory_section(c, block.get("node"), hops, now)

    live_chars = [c for c in characters if (now_by_char.get(c) or {}).get("node")]
    view = {
        "schema": SCHEMA,
        "generated_at": now,
        "generated_iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "root": str(ROOT),
        "characters": characters,
        "live_characters": live_chars,
        "nodes": nodes,
        "edges": edges,
        "meta": meta,
        "lived": lived_stats,
        "now": now_by_char,
        "memory": memory_by_char,
        "world": {
            "context": _read_json(MIND / "context.json", {}),
            "signal": _read_json(MIND / "context_signal.json", {}),
        },
        "budgets": {
            "gen": _read_json(MIND / "gen_budget.json", {}),
            "clips": _read_json(MIND / "clip_budget.json", {}),
        },
        "policy": {
            "REVERSE_PENALTY": policy.REVERSE_PENALTY, "LAST_CLIP_PENALTY": policy.LAST_CLIP_PENALTY,
            "GOAL_ARRIVE_BOOST": policy.GOAL_ARRIVE_BOOST, "AWAY_PENALTY": policy.AWAY_PENALTY,
            "GOAL_HOLD_BOOST": policy.GOAL_HOLD_BOOST, "ROUTE_PULL": policy.ROUTE_PULL,
            "ESCAPE_AFTER": policy.ESCAPE_AFTER, "ESCAPE_PER_UNIT": policy.ESCAPE_PER_UNIT,
            "ESCAPE_MAX": policy.ESCAPE_MAX, "REVERSE_FORGIVE": policy.REVERSE_FORGIVE,
            "MOOD_BIAS": policy.MOOD_BIAS,
        },
        "codemap": codemap(),
        "caveats": [
            "`created` is DERIVED FROM ARTIFACT MTIME, not a ledger. It survives a graph rebuild "
            "and is checkable against the filesystem, but copying the tree without -p resets it and "
            "a regenerated still moves it forward. Where there is no artifact there is no field.",
            "The 76-day backfill recovered visits and dwell exactly, clip plays at LABEL level only "
            "(83 labels have 2-4 rendered variants sharing a name), and timestamps NOT AT ALL -- the "
            "walker's log carries an hour and never a date. Backfilled poses carry no first/last.",
            "Manner and valence are DERIVED from each clip's own prose. Edges the lexicon cannot read "
            "are reported untyped, never bucketed as neutral.",
            "Cross-character links are RECONSTRUCTED AT READ TIME from what each character can see of "
            "the other. Production stores zero cross-character edges, and neither does Generative Agents.",
            "The world seam is conditions, not facts: weather and hour, cached and gone next tick, never "
            "journaled or scored. Declarative memory about events, people and rooms is still the open gap.",
            "The decision surface is the real `policy.weigh` call the walker makes, evaluated at export "
            "time. It is a snapshot of a choice made ten times a second, not a recording of one -- and "
            "it is missing one term: novelty counts recent clips and poses, which live only in the "
            "walker's memory and are not on disk. Anti-reverse, goal pull, mood, escape velocity and "
            "manner are all reconstructed exactly.",
        ],
    }
    return view


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--stdout", action="store_true", help="print the view, write nothing")
    args = ap.parse_args()

    t0 = time.time()
    view = build()
    blob = json.dumps(view, ensure_ascii=False, separators=(",", ":"))
    if args.stdout:
        print(blob)
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(blob, encoding="utf-8")
    os.replace(tmp, out)          # atomic: the viewer never reads a half-written file
    print("wrote %s  %.1f KB  %d nodes  %d edges  %.1fs"
          % (out, len(blob) / 1024.0, len(view["nodes"]), len(view["edges"]), time.time() - t0))
    for c in view["live_characters"]:
        n = view["now"][c]
        m = view["memory"].get(c) or {}
        print("  %-10s at %-26s goal=%-26s retrieved=%s of %s entries"
              % (c, n.get("node"), n.get("goal"), len(m.get("retrieved") or []), m.get("entries")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
