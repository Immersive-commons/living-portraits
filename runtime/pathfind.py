"""pathfind.py -- shortest-path traversal over the video graph's TRANSITION edges.

The walker moves between poses by playing TRANSITION clips (kind="transition",
from-pose -> to-pose). To send a character from where it is now to a goal pose
chosen by the mind (director/heartbeat.py), we need the next transition to play
on a shortest path. That's a plain BFS over the transition sub-graph.

This is the read-side twin of VideoGraph._reachable_from (video_graph.py): same
walk, but it remembers the FIRST edge taken so the walker can play exactly one
hop, land, and ask again -- so the character visibly walks there step by step.

Pure + import-safe (stdlib only). Operates on the flat edge list the walker
already holds (GraphCycler.self.all), so it needs no VideoGraph object.

    from runtime import pathfind
    edge = pathfind.next_step(all_edges, "phineas:anchor", "phineas:glower")
    edge["label"]  # -> "a2g" : play this, land on glower, then ask again
"""
from __future__ import annotations

from collections import deque


def _adjacency(all_edges):
    """node -> [transition edges leaving it]. Idle self-loops are ignored: they do
    not move the character, so they are never part of a path between two poses."""
    adj = {}
    for e in all_edges:
        if e.get("kind") == "transition":
            adj.setdefault(e.get("from"), []).append(e)
    return adj


def next_step(all_edges, start, goal, adj=None):
    """The FIRST transition edge on a shortest path start -> goal, or None if we're
    already there / there is no path. BFS guarantees the fewest hops; ties broken by
    edge declaration order (deterministic given the edge list)."""
    if start is None or goal is None or start == goal:
        return None
    adj = adj if adj is not None else _adjacency(all_edges)
    seen = {start}
    # queue holds (node_reached, first_edge_used_to_leave_start)
    q = deque()
    for e in adj.get(start, []):
        to = e.get("to")
        if to == goal:
            return e                       # direct neighbour
        if to not in seen:
            seen.add(to)
            q.append((to, e))
    while q:
        node, first = q.popleft()
        for e in adj.get(node, []):
            to = e.get("to")
            if to == goal:
                return first
            if to not in seen:
                seen.add(to)
                q.append((to, first))
    return None


def shortest_path(all_edges, start, goal, adj=None):
    """The full node sequence [start, ..., goal] on a shortest path, or [] if
    unreachable. For logging / the heartbeat's reasoning, not the hot loop."""
    if start is None or goal is None:
        return []
    if start == goal:
        return [start]
    adj = adj if adj is not None else _adjacency(all_edges)
    seen = {start}
    q = deque([(start, [start])])
    while q:
        node, path = q.popleft()
        for e in adj.get(node, []):
            to = e.get("to")
            if to == goal:
                return path + [to]
            if to not in seen:
                seen.add(to)
                q.append((to, path + [to]))
    return []


def reachable_poses(all_edges, start, adj=None):
    """The set of poses reachable from `start` via transitions (excluding `start`).
    The heartbeat constrains the LLM's goal choice to this set so it can never pick
    a pose the body cannot actually walk to."""
    if start is None:
        return set()
    adj = adj if adj is not None else _adjacency(all_edges)
    seen, stack = {start}, [start]
    while stack:
        for e in adj.get(stack.pop(), []):
            to = e.get("to")
            if to not in seen:
                seen.add(to)
                stack.append(to)
    seen.discard(start)
    return seen
