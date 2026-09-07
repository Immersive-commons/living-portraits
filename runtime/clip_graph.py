"""
clip_graph.py -- the clip library as a directed graph of poses and transitions.

A living portrait performs by walking a graph:

    nodes = poses  (the 7 stage-manager ACTIONS)
    edges = clips  (a baked transition / held-loop video between two poses)

The stage-manager writes a beat whose `action` names a *target pose*. The
player asks this graph for the shortest sequence of clips that gets the rig
from where it currently stands to that target, then hands that sequence to the
ClipPlayer. A self-loop edge (from_node == to_node) is the held "idle on this
pose" loop the rig plays while waiting for the next beat.

Clips are produced offline by the generative pipeline (img2vid + RIFE) and only
admitted to the library after the verify gate (SSIM continuity / loopable /
ID / register). Until a real clip exists for an edge, the edge may carry
`asset=None` -- a SYNTHETIC placeholder the player can still render, and
`missing_edges()` surfaces exactly which transitions the director must still
ask the pipeline to generate.

The library persists as a JSON manifest under data/clips/manifest.json. That
directory is the KV-index: each clip's metadata is one row; the actual mp4 /
png-sequence sits beside it. (`data/` is gitignored runtime state by design.)

No pygame, no opencv here -- this module is pure stdlib so it imports and tests
anywhere. Frame decode + blit live in clip_player.py.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Iterable

# The pose vocabulary. MUST stay in lock-step with
# director/stage_manager.py:ACTIONS -- those are the only `action` values a beat
# can carry, and every one of them must be a node here or the player can never
# reach it. Kept as a module constant (not imported from director/) so this
# runtime package has no dependency on the director package; the self-test
# asserts the two lists agree in spirit.
POSES: tuple[str, ...] = (
    "idle",
    "address_house",
    "lean_in",
    "glance_aside",
    "ponder",
    "leaving",
    "arriving",
)

# Where the library lives. data/ is gitignored runtime state.
ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = ROOT / "data" / "clips"
MANIFEST_PATH = CLIPS_DIR / "manifest.json"


@dataclass
class Clip:
    """One edge of the graph: a baked clip carrying the rig from one pose to another.

    from_node / to_node : pose names (must be in POSES).
    asset    : path to an mp4 or a png-sequence directory, relative to CLIPS_DIR,
               or None when no real clip exists yet (player renders SYNTHETIC).
    frames   : number of frames in the clip (used for timing / seam math). For a
               synthetic placeholder this is the desired length to render.
    fps      : playback rate.
    loopable : True if the last frame flows back into the first without a visible
               seam (verify gate asserts this). Held-pose self-loops must be
               loopable; one-shot transitions generally are not.
    tags     : free-form labels (e.g. "transition", "hold", "synthetic",
               "verified", a mood tag). Used by the director / ranker later.

    Cost for shortest-path defaults to frame count (shorter visual path wins);
    a missing (asset=None) edge is traversable but carries a penalty so the
    planner prefers real footage when a real route exists.
    """

    from_node: str
    to_node: str
    asset: str | None = None
    frames: int = 24
    fps: int = 24
    loopable: bool = False
    tags: list[str] = field(default_factory=list)

    # ---- derived ----------------------------------------------------------
    @property
    def is_synthetic(self) -> bool:
        return self.asset is None

    @property
    def is_hold(self) -> bool:
        """A held-pose self-loop (idle-on-pose), not a transition."""
        return self.from_node == self.to_node

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def cost(self) -> float:
        """Edge weight for shortest-path. Frames + a penalty for placeholders."""
        penalty = 1000 if self.is_synthetic else 0
        return float(self.frames) + penalty

    def __post_init__(self) -> None:
        if self.from_node not in POSES:
            raise ValueError(f"unknown from_node pose: {self.from_node!r} (expected one of {POSES})")
        if self.to_node not in POSES:
            raise ValueError(f"unknown to_node pose: {self.to_node!r} (expected one of {POSES})")
        if self.frames <= 0:
            raise ValueError(f"frames must be positive, got {self.frames}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    # ---- serialization (KV-index rows) ------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Clip:
        # Tolerate extra keys (forward-compat with whatever the verify gate stamps on).
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def key(self) -> str:
        """Stable identity of this edge: from->to. One clip per ordered pair."""
        return f"{self.from_node}->{self.to_node}"


class ClipGraph:
    """Directed graph: pose nodes, clip edges, shortest-path traversal + a JSON KV-index.

    One clip per ordered (from, to) pair -- add_clip replaces any existing edge
    for that pair (last write wins, so a verified clip cleanly supersedes a
    synthetic placeholder). Nodes are the fixed POSES; edges accrue as the
    pipeline bakes (or stubs) transitions.
    """

    def __init__(self, nodes: Iterable[str] = POSES) -> None:
        self.nodes: list[str] = list(nodes)
        # adjacency: from_node -> { to_node -> Clip }
        self._edges: dict[str, dict[str, Clip]] = {n: {} for n in self.nodes}

    # ---- mutation ---------------------------------------------------------
    def add_clip(self, clip: Clip) -> ClipGraph:
        """Add (or replace) the edge for clip's (from_node, to_node) pair."""
        if clip.from_node not in self._edges:
            self._edges[clip.from_node] = {}
            self.nodes.append(clip.from_node)
        if clip.to_node not in self._edges:
            self._edges[clip.to_node] = {}
            self.nodes.append(clip.to_node)
        self._edges[clip.from_node][clip.to_node] = clip
        return self

    def add(self, from_node: str, to_node: str, **kw) -> Clip:
        """Convenience: build a Clip and add it. Returns the clip."""
        clip = Clip(from_node=from_node, to_node=to_node, **kw)
        self.add_clip(clip)
        return clip

    def get_clip(self, from_node: str, to_node: str) -> Clip | None:
        return self._edges.get(from_node, {}).get(to_node)

    def edges(self) -> list[Clip]:
        return [c for outs in self._edges.values() for c in outs.values()]

    def neighbors(self, node: str) -> dict[str, Clip]:
        return self._edges.get(node, {})

    # ---- traversal --------------------------------------------------------
    def path(self, from_node: str, to_node: str) -> list[Clip] | None:
        """Shortest sequence of clips from `from_node` to `to_node`.

        Dijkstra over clip.cost (frames, with a heavy penalty on synthetic
        placeholder edges so a real-footage route is preferred whenever one
        exists). Returns the list of clips to play in order, [] if already at
        the target, or None if the target is unreachable with current edges.
        """
        if from_node not in self._edges or to_node not in self._edges:
            return None
        if from_node == to_node:
            return []

        # Dijkstra. Tiny graph (7 nodes), so a simple O(V^2) scan is plenty and
        # avoids pulling in heapq subtleties around the comparator.
        dist: dict[str, float] = {n: float("inf") for n in self._edges}
        prev: dict[str, tuple[str, Clip] | None] = dict.fromkeys(self._edges)
        dist[from_node] = 0.0
        unvisited = set(self._edges)

        while unvisited:
            # node with smallest tentative distance
            cur = min(unvisited, key=lambda n: dist[n])
            if dist[cur] == float("inf"):
                break  # remaining nodes are unreachable
            if cur == to_node:
                break
            unvisited.discard(cur)
            for nxt, clip in self._edges[cur].items():
                alt = dist[cur] + clip.cost
                if alt < dist[nxt]:
                    dist[nxt] = alt
                    prev[nxt] = (cur, clip)

        if dist[to_node] == float("inf"):
            return None  # unreachable; director should request generation

        # walk prev[] back to from_node, collecting clips
        seq: list[Clip] = []
        node = to_node
        while node != from_node:
            step = prev[node]
            if step is None:
                return None  # defensive: broken chain
            parent, clip = step
            seq.append(clip)
            node = parent
        seq.reverse()
        return seq

    def hold_clip(self, pose: str) -> Clip | None:
        """The held self-loop for a pose (idle-on-pose). May be synthetic."""
        return self.get_clip(pose, pose)

    # ---- gaps the director must fill --------------------------------------
    def missing_edges(self, required: Iterable[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
        """(from, to) pairs the director should ask the pipeline to generate.

        With no argument: every edge currently carrying a synthetic placeholder
        (asset=None) -- i.e. clips that "exist" only as stubs and still need
        real footage baked + verified.

        With `required`: the subset of those requested pairs that are either
        absent entirely OR present-but-synthetic. This lets the director pass
        the transitions a beat actually needs and learn exactly which ones it
        cannot yet perform with real art.
        """
        if required is None:
            return sorted(
                (c.from_node, c.to_node) for c in self.edges() if c.is_synthetic
            )
        gaps: list[tuple[str, str]] = []
        for fn, tn in required:
            clip = self.get_clip(fn, tn)
            if clip is None or clip.is_synthetic:
                gaps.append((fn, tn))
        return gaps

    def is_fully_baked(self) -> bool:
        """True when no edge is a synthetic placeholder."""
        return not self.missing_edges()

    # ---- KV-index: load / save a JSON manifest ----------------------------
    def save(self, path: Path | str = MANIFEST_PATH) -> Path:
        """Persist the clip rows to a JSON manifest (atomic write)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "nodes": list(self.nodes),
            "clips": [c.to_dict() for c in self.edges()],
        }
        # atomic: write tmp in the same dir, then os.replace -- a reader (the
        # player) never sees a half-written manifest. Mirrors the director's
        # stage_state.json discipline.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    @classmethod
    def load(cls, path: Path | str = MANIFEST_PATH) -> ClipGraph:
        """Load a ClipGraph from a JSON manifest. Empty graph if absent."""
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        graph = cls(nodes=data.get("nodes", POSES))
        for row in data.get("clips", []):
            graph.add_clip(Clip.from_dict(row))
        return graph

    # ---- introspection ----------------------------------------------------
    def summary(self) -> str:
        total = len(self.edges())
        synth = len(self.missing_edges())
        return (
            f"ClipGraph: {len(self.nodes)} poses, {total} clips "
            f"({total - synth} real, {synth} synthetic placeholder)"
        )


def default_graph() -> ClipGraph:
    """A fully-connected synthetic graph: every pose -> every pose + self-loops.

    This is the bootstrap library the player can run against on day one, before
    any real footage is baked. Every edge is a synthetic placeholder, so
    missing_edges() returns the full generation worklist for the pipeline.
    Self-loops are marked loopable (a held pose must loop seamlessly); cross-pose
    transitions are not.
    """
    g = ClipGraph()
    for a in POSES:
        for b in POSES:
            g.add(
                a,
                b,
                asset=None,
                frames=18 if a == b else 12,
                fps=24,
                loopable=(a == b),
                tags=["synthetic", "hold" if a == b else "transition"],
            )
    return g


if __name__ == "__main__":
    # Tiny smoke of the graph alone (the full headless player self-test lives in
    # clip_player.py). Build idle <-> address_house <-> leaving and route across.
    g = ClipGraph()
    g.add("idle", "address_house", frames=10, tags=["transition"])
    g.add("address_house", "idle", frames=10, tags=["transition"])
    g.add("address_house", "leaving", frames=14, tags=["transition"])
    g.add("idle", "idle", frames=18, loopable=True, tags=["hold"])

    p = g.path("idle", "leaving")
    assert p is not None, "idle->leaving should be reachable via address_house"
    assert [c.key() for c in p] == ["idle->address_house", "address_house->leaving"], [c.key() for c in p]
    assert g.path("idle", "idle") == [], "same-node path is empty"
    assert g.path("leaving", "idle") is None, "no edge out of leaving -> unreachable"
    # these clips left asset=None, so they are synthetic placeholders ->
    # every one of them is a missing edge the pipeline still owes us.
    assert len(g.missing_edges()) == len(g.edges()), "all placeholder edges are missing"
    # a clip WITH a (bogus) asset is NOT 'missing' -- it claims real footage that
    # merely fails to decode. That's a different failure mode from never-generated.
    g.add("idle", "ponder", asset="some_clip.mp4", frames=10, tags=["transition"])
    assert ("idle", "ponder") not in g.missing_edges(), "asset-set edge must not count as missing"
    # required-subset form: ask only about the pairs a beat needs
    assert g.missing_edges([("idle", "address_house"), ("idle", "ponder")]) == [("idle", "address_house")]

    # round-trip the manifest through a temp file
    import tempfile as _tf

    tmpdir = Path(_tf.mkdtemp())
    mpath = tmpdir / "manifest.json"
    g.save(mpath)
    g2 = ClipGraph.load(mpath)
    assert {c.key() for c in g2.edges()} == {c.key() for c in g.edges()}, "manifest round-trip lost edges"

    # default fully-connected synthetic graph: 7*7 = 49 edges, all synthetic
    dg = default_graph()
    assert len(dg.edges()) == len(POSES) * len(POSES), len(dg.edges())
    assert len(dg.missing_edges()) == len(POSES) * len(POSES), "every default edge is a placeholder"
    assert dg.path("leaving", "arriving") is not None, "default graph is fully connected"

    print("clip_graph OK:", g.summary(), "|", dg.summary())
