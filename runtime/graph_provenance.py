"""graph_provenance.py -- BI-TEMPORAL PROVENANCE for the video graph.

The audit's first structural finding (`_audit/CONTEXT_GRAPH_AUDIT.md` §1.1) is that the
graph has no time dimension at all: "There is no timestamp, no valence, no importance,
no type." A node cannot say when it entered the system, and an edge whose clip is deleted
does not become history -- it stops existing, because `build()` simply skips it and
overwrites video_graph.json. The graph forgets, silently, on a ~20-minute cadence.

Zep/Graphiti (arXiv 2501.13956) keeps FOUR timestamps per fact:

    t_created / t_expired   TRANSACTION time -- when the SYSTEM learned / retired it
    t_valid   / t_invalid   VALID time       -- when it was true in the WORLD

This module is the TRANSACTION half, and it is the half the graph could never express:

    created      when the system learned this node/edge exists  (= t_created)
    invalidated  when the system noticed it was gone            (= t_expired)

The VALID half already exists next door in `runtime/lived.py` -- when a character actually
stood in a pose and how long. Between the two files the graph is bi-temporal.

THREE DECISIONS WORTH DEFENDING
-------------------------------
1. `created` is DERIVED FROM ARTIFACT MTIME, not stored. A node's still under data/gen/,
   an edge's gif under data/clips/_proto/. `build()` regenerates video_graph.json from
   NODE_SPECS/EDGE_SPECS and overwrites it, and Phase-2 autogen rebuilds roughly every 20
   minutes, so any timestamp written INTO that file is destroyed by the next pose the
   characters grow. A derived stamp survives every rebuild for free and is checkable
   against the filesystem. Its honest limit: mtime is EVIDENCE, not a ledger. Copying the
   repo to `hil` without `-p` resets it, and a regenerated still moves it forward. It is
   the best available answer to "when did this enter the system", and when there is no
   artifact there is NO FIELD -- never an invented timestamp.

2. INVALIDATION IS A DIFF, not a flag. An edge does not vanish in one place -- its gif can
   be deleted, its spec removed, a variant dropped, a bedtime character can regress to
   partially-generated. All of those show up as one thing: an edge id that was in the last
   committed graph and is not in the new one. So `reconcile()` diffs the PREVIOUS live
   graph against the one just built and tombstones what disappeared. That catches every
   disappearance, including ones nobody has thought of yet.

3. THE TOMBSTONE LIVES HERE, NOT IN video_graph.json. Two reasons. The file is overwritten
   on every rebuild, so state put there is state lost. And `_preview_graph.py` -- the live
   player -- reads that JSON RAW (`graph.get("edges", [])`), so a dead edge written back
   into it is a dead clip handed straight to the walker, which would try to open a gif that
   is not on disk. Keeping video_graph.json a pure LIVE view makes "the walker never plays
   an invalidated clip" true BY CONSTRUCTION rather than by a filter somebody has to
   remember to write. History is reassembled on request by `merge_history()`.

Pure stdlib, import-safe, and fail-soft throughout: every entry point returns an empty /
unchanged result rather than raising. A statistics file is never worth a dark panel.

    from runtime import graph_provenance as gp
    gp.stamp(g.nodes, g.edges)                   # adds `created` where an artifact exists
    store = gp.load_store()
    gp.reconcile(prev_nodes, prev_edges, g.nodes, g.edges, store=store)
    nodes, edges = gp.merge_history(g.nodes, g.edges, store)   # live + the dead
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE_PATH = ROOT / "data" / "graph" / "provenance.json"
SCHEMA = "living-portrait.graph-provenance/v1"

# --- tunables (module constants so the tests pin behaviour, not magic numbers).
MAX_TOMBSTONES = 5000     # per kind; oldest death evicted first, so the store cannot grow forever
SCANDIR_MIN_REFS = 8      # a dir referenced this often is listed once instead of stat'ed per file

# Measured over the real 259-node / 1472-edge production snapshot (data/_realdata/):
# 1731 individual os.stat calls = 39 ms; two os.scandir passes = 1.7 ms. Hence the
# threshold above -- the graph's artifacts live in exactly two directories, and listing
# each one once is 23x cheaper than asking about every file in it by name.

CREATED = "created"
INVALIDATED = "invalidated"


# --------------------------------------------------------------------------- mtimes
def _split(rel):
    """('data/clips/_proto', 'x.gif') from either separator. Graph paths are posix."""
    rel = str(rel).replace("\\", "/")
    i = rel.rfind("/")
    return (rel[:i], rel[i + 1:]) if i >= 0 else ("", rel)


def mtimes(rel_paths, root=None):
    """{project-relative path -> mtime} for the ones that EXIST. Missing files are simply
    absent from the result -- the caller must not stamp what it cannot see.

    Groups by directory first: a directory referenced more than SCANDIR_MIN_REFS times is
    listed once (DirEntry carries mtime, so no second syscall per file), everything else
    is stat'ed directly. A directory that has gone away degrades to per-file stats that
    also fail, which is the right answer and costs nothing extra."""
    root = ROOT if root is None else Path(root)
    by_dir = {}
    for r in rel_paths or ():
        if not r:
            continue
        d, name = _split(r)
        by_dir.setdefault(d, {}).setdefault(name, []).append(str(r))
    out = {}
    for d, names in by_dir.items():
        base = (root / d) if d else root
        if len(names) >= SCANDIR_MIN_REFS:
            try:
                with os.scandir(str(base)) as it:
                    for de in it:
                        originals = names.get(de.name)
                        if not originals:
                            continue
                        try:
                            m = de.stat().st_mtime
                        except OSError:
                            continue
                        for o in originals:
                            out[o] = m
                continue
            except OSError:
                pass          # dir unreadable -> fall through to per-file stat
        for name, originals in names.items():
            try:
                m = os.stat(str(base / name)).st_mtime
            except OSError:
                continue
            for o in originals:
                out[o] = m
    return out


def stamp(nodes, edges, root=None):
    """Add `created` to every node/edge whose artifact is on disk; REMOVE it from every one
    whose artifact is not (so a stale stamp can never outlive the file that justified it).

    Mutates in place, returns {nodes, nodes_created, edges, edges_created}. Node artifact
    is its still `image`, edge artifact is its `gif`. Fail-soft: any error leaves the graph
    exactly as it was and reports zero coverage."""
    nodes = nodes if isinstance(nodes, dict) else {}
    edges = edges if isinstance(edges, list) else []
    stats = {"nodes": len(nodes), "nodes_created": 0,
             "edges": len(edges), "edges_created": 0}
    try:
        refs = [v.get("image") for v in nodes.values() if isinstance(v, dict)]
        refs += [e.get("gif") for e in edges if isinstance(e, dict)]
        table = mtimes(refs, root=root)
        for v in nodes.values():
            if not isinstance(v, dict):
                continue
            m = table.get(v.get("image"))
            if m is None:
                v.pop(CREATED, None)
            else:
                v[CREATED] = m
                stats["nodes_created"] += 1
        for e in edges:
            if not isinstance(e, dict):
                continue
            m = table.get(e.get("gif"))
            if m is None:
                e.pop(CREATED, None)
            else:
                e[CREATED] = m
                stats["edges_created"] += 1
    except Exception:
        pass
    return stats


# --------------------------------------------------------------------------- the store
def _empty_store():
    return {"schema": SCHEMA, "updated": None, "nodes": {}, "edges": {}}


def load_store(path=None):
    """The tombstone store, normalised. Missing / corrupt / half-written -> empty store,
    never an exception: history is a nice-to-have and a graph load is not."""
    store = _empty_store()
    try:
        d = json.loads(Path(STORE_PATH if path is None else path).read_text(encoding="utf-8"))
    except Exception:
        return store
    if not isinstance(d, dict):
        return store
    for kind in ("nodes", "edges"):
        got = d.get(kind)
        if isinstance(got, dict):
            store[kind] = {k: v for k, v in got.items() if isinstance(v, dict)}
    store["updated"] = d.get("updated")
    return store


def save_store(store, path=None, now=None):
    """Atomic tmp+replace, exactly as video_graph.save() and lived.flush() do -- a build
    running while somebody reads history must never expose a half-written file.
    Returns True if it wrote."""
    p = Path(STORE_PATH if path is None else path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "schema": SCHEMA,
            "updated": time.time() if now is None else now,
            "nodes": (store or {}).get("nodes") or {},
            "edges": (store or {}).get("edges") or {}}, indent=1), encoding="utf-8")
        os.replace(str(tmp), str(p))
        return True
    except Exception:
        return False


def _prune(tombs, limit):
    """Keep the `limit` most recent deaths. Bounded like lived.MAX_MOODS: a store that
    grows without limit is a slow leak in a process that runs forever."""
    if limit is None or len(tombs) <= limit:
        return tombs
    ranked = sorted(tombs.items(), key=lambda kv: -(kv[1].get(INVALIDATED) or 0.0))
    return dict(ranked[:limit])


def _reconcile_one(prev_by_id, new_ids, tombs, now, limit):
    """One kind (nodes or edges). Returns (tombs, n_invalidated, n_revived)."""
    invalidated = revived = 0
    for tid in list(tombs):
        if tid in new_ids:                     # it came back -> it is live again
            tombs.pop(tid, None)
            revived += 1
    for pid, record in prev_by_id.items():
        if pid in new_ids or pid in tombs:     # still live, or already dead and dated
            continue
        tombs[pid] = {INVALIDATED: now, "record": record}
        invalidated += 1
    return _prune(tombs, limit), invalidated, revived


def reconcile(prev_nodes, prev_edges, new_nodes, new_edges, *,
              store=None, now=None, max_tombstones=MAX_TOMBSTONES):
    """Diff the last committed graph against the one just built and update the store.

    Anything that was there and is not now gets a tombstone carrying its FULL previous
    record (so history can reconstruct the edge, not merely name it) plus the moment the
    system noticed. A tombstone's date is set ONCE -- a rebuild that finds it still gone
    does not re-date the death. An id that reappears is dropped from the store: it is live
    again, and this is a tombstone list, not a churn ledger.

    Returns (store, stats). Fail-soft: on any error the store comes back untouched."""
    store = _empty_store() if store is None else store
    now = time.time() if now is None else now
    stats = {"nodes_invalidated": 0, "nodes_revived": 0,
             "edges_invalidated": 0, "edges_revived": 0,
             "nodes_dead": 0, "edges_dead": 0}
    try:
        prev_n = {k: v for k, v in (prev_nodes or {}).items() if isinstance(v, dict)}
        new_n = set((new_nodes or {}).keys())
        store["nodes"], stats["nodes_invalidated"], stats["nodes_revived"] = _reconcile_one(
            prev_n, new_n, store.get("nodes") or {}, now, max_tombstones)

        prev_e = {e["id"]: e for e in (prev_edges or [])
                  if isinstance(e, dict) and e.get("id")}
        new_e = {e.get("id") for e in (new_edges or []) if isinstance(e, dict)}
        store["edges"], stats["edges_invalidated"], stats["edges_revived"] = _reconcile_one(
            prev_e, new_e, store.get("edges") or {}, now, max_tombstones)

        stats["nodes_dead"] = len(store["nodes"])
        stats["edges_dead"] = len(store["edges"])
    except Exception as e:
        # Fail-open stays: a broken reconciliation must not stop a build. But the
        # counters above are all still ZERO here, and a caller reading them cannot
        # tell "nothing changed" from "this crashed". Almost nothing in that body
        # SHOULD be able to raise -- it is dict work over data validated upstream --
        # which is exactly why swallowing it hides a real bug instead of absorbing
        # an expected one. AGENTS.md rule 4: absence with a reason.
        stats["error"] = repr(e)
    return store, stats


# --------------------------------------------------------------------------- history view
def merge_history(nodes, edges, store, on_error=None):
    """(nodes, edges) = the LIVE graph plus everything the store says used to be there,
    each dead entry carrying `invalidated`. Returns NEW containers; the live ones handed in
    are never mutated, so a caller holding the live view keeps holding the live view.

    This is what answers "what could this character do last month" -- and the reason the
    dead records are stored whole rather than as bare ids.

    `on_error(repr)` fires if the merge fails. It exists because the fail-open return
    here is a WRONG answer rather than a degraded one: handing back the live graph says
    "there is no history" to a question about history, and the caller then records that
    it HAS history. video_graph.load() used to set `g.history = True` on exactly that
    path. Callers that do not care may omit it and get the old behaviour.
    """
    out_nodes = dict(nodes or {})
    out_edges = list(edges or [])
    try:
        live_edge_ids = {e.get("id") for e in out_edges if isinstance(e, dict)}
        for nid, rec in sorted(((store or {}).get("nodes") or {}).items()):
            if nid in out_nodes:
                continue
            d = dict(rec.get("record") or {})
            d[INVALIDATED] = rec.get(INVALIDATED)
            out_nodes[nid] = d
        for eid, rec in sorted(((store or {}).get("edges") or {}).items()):
            if eid in live_edge_ids:
                continue
            d = dict(rec.get("record") or {})
            d.setdefault("id", eid)
            d[INVALIDATED] = rec.get(INVALIDATED)
            out_edges.append(d)
    except Exception as e:
        if on_error is not None:
            on_error(repr(e))
        return dict(nodes or {}), list(edges or [])
    return out_nodes, out_edges


def invalidated_edge_ids(store):
    """The clips that exist in history and MUST NOT be played."""
    return set(((store or {}).get("edges") or {}).keys())


def invalidated_node_ids(store):
    return set(((store or {}).get("nodes") or {}).keys())


def summary(store, now=None):
    """{nodes_dead, edges_dead, oldest_death_age_days, newest_death_age_days} -- one line's
    worth of "how much has this graph forgotten". Empty store -> zeroes, no None arithmetic."""
    now = time.time() if now is None else now
    deaths = [r.get(INVALIDATED) for kind in ("nodes", "edges")
              for r in (((store or {}).get(kind) or {}).values())
              if isinstance(r.get(INVALIDATED), (int, float))]
    return {"nodes_dead": len((store or {}).get("nodes") or {}),
            "edges_dead": len((store or {}).get("edges") or {}),
            "oldest_death_age_days": ((now - min(deaths)) / 86400.0) if deaths else 0.0,
            "newest_death_age_days": ((now - max(deaths)) / 86400.0) if deaths else 0.0}
