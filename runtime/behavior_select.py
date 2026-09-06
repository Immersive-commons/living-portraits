"""
behavior_select.py -- personality-driven idle-behaviour picker for the player.

A living portrait idles by playing one behaviour loop per cycle (Phineas
*declaims*, then *wounded_sulk*s, then a resentful *side_eye_rival*...). WHICH
behaviour plays each cycle is the dynamic part of the "literal personality"
model (prompts/LIVING_PORTRAITS_PROMPT_SPEC.md sec.dynamic): the character JSON
gives each behaviour a base `weight` under motion.idle_behaviors, and the
director's current mood (data/mood.json) bends those weights via the character's
dynamic.mood_reweights. This module is the consume path for that contract.

What it does, in one call::

    chosen_clip, behavior_id = pick_behavior("phineas", graph_or_clips)

  1. load_character("phineas")          -> the JSON spec (cached by mtime)
  2. current_mood()                     -> data/mood.json's mood (safe default)
  3. match the character's idle_behaviors to the behaviour-tagged LOOP clips
     actually available (a ClipGraph or a plain clip iterable -- whichever the
     player hands us), keeping only behaviours that have a baked/stubbed loop
  4. weight each candidate by its base weight * mood_reweight[mood], normalise,
     and draw one with `rng` (default: the module's own random.Random)
  5. return (clip, behavior_id), or None when no behaviour loop exists yet

Design constraints (this module is imported by the LIVE player loop):

  * IMPORT-SAFE with NEITHER pygame NOR torch NOR numpy installed, and with no
    network and no GPU. The only hard dep is the stdlib. `Clip`/`ClipGraph`
    types are duck-typed (we read `.tags`, `.from_node`, `.to_node`,
    `.loopable`, `.neighbors()`/`.edges()`); we do NOT import clip_graph at
    module top level, so a missing/!broken runtime never stops the player from
    importing this. Optional imports are guarded.
  * SIDE-EFFECT-FREE on import: no file reads, no clock, no RNG seeding at
    import time. Everything happens inside the functions.
  * GRACEFUL: every "not ready yet" path returns a safe default (None / the
    base weights / the default mood) rather than raising, because the show is
    running and a half-built clip library must not crash a render frame.

Behaviour-loop contract (how a behaviour clip is recognised)
------------------------------------------------------------
Per phineas.json.dynamic.render_pseudocode, the bake registers each idle
behaviour as a self-loop edge on the anchor pose, tagged with the behaviour id::

    clip_graph.add_edge(anchor_node -> anchor_node, clip, tags=[behavior.id])

So a clip is "the loop for behaviour B of character C" when it is a held loop
(from_node == to_node, i.e. an anchor self-loop) AND its tags identify both the
behaviour id B and -- when present -- the character C. We match leniently: a
behaviour id may appear as a bare tag ("declaim"), a namespaced tag
("behavior:declaim" / "behaviour:declaim"), and the character as "character:<slug>"
(the convention pipeline/orchestrate.py already stamps). A clip with no
character tag is accepted for any character (single-character graphs); a clip
whose character tag names a DIFFERENT slug is rejected. This tolerates the
clip-graph invariant of one-clip-per-ordered-pair: behaviour loops that cannot
all be `idle->idle` may live as distinct anchor self-loops or be handed in as a
plain clip list -- either way we key on tags, not on the (from,to) slot.

Run `python runtime/behavior_select.py --selftest` to exercise the weighting +
mood-reweight math on synthetic data (no files, no clip graph, no deps).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

# Repo root is the parent of runtime/. Used to locate the character JSON and
# data/mood.json. Resolved lazily-safe (no I/O here, just a path object).
_ROOT = Path(__file__).resolve().parent.parent
_CHAR_DIR = _ROOT / "prompts" / "characters"
_MOOD_PATH = _ROOT / "data" / "mood.json"

# A neutral mood that means "no reweight" -- used when data/mood.json is missing,
# malformed, or names a mood the character has no reweight rule for. It is NOT a
# magic string the character JSON must declare; it simply won't match any
# mood_reweights key, so every behaviour keeps its base weight.
DEFAULT_MOOD = "neutral"

# Module-private RNG so repeated picks vary without the caller threading a
# Random through every call, yet a caller CAN pass its own rng for reproducible
# tests / seeded shows. Created once; never reseeded on import.
_RNG = random.Random()


# --------------------------------------------------------------------------- #
# character spec loading (cached by mtime)
# --------------------------------------------------------------------------- #
# slug -> (mtime_ns, parsed_dict). Re-read only when the file's mtime changes,
# so the live player can poll pick_behavior() every cycle without re-parsing the
# JSON each frame, yet still pick up an edited character spec without a restart.
_CHAR_CACHE: dict[str, tuple[int, dict]] = {}


def character_path(slug: str) -> Path:
    """Path to a character's JSON spec (prompts/characters/<slug>.json)."""
    return _CHAR_DIR / f"{slug}.json"


def load_character(slug: str) -> dict | None:
    """Load + cache a character JSON by slug, refreshing on mtime change.

    Returns the parsed dict, or None if the file is missing or unparseable
    (the caller then has no behaviours to pick from and pick_behavior returns
    None). Never raises -- a broken spec must not crash the player.
    """
    path = character_path(slug)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        # missing / unreadable -> drop any stale cache entry, signal "no spec"
        _CHAR_CACHE.pop(slug, None)
        return None

    cached = _CHAR_CACHE.get(slug)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _CHAR_CACHE.pop(slug, None)
        return None
    if not isinstance(data, dict):
        _CHAR_CACHE.pop(slug, None)
        return None

    _CHAR_CACHE[slug] = (mtime, data)
    return data


def idle_behaviors(spec: Mapping[str, Any]) -> list[dict]:
    """The character's motion.idle_behaviors, as a list of well-formed dicts.

    Each kept entry has a non-empty string `id` and a numeric base `weight`
    (defaulting to 0.0 when absent / non-numeric). Malformed entries are
    dropped rather than raising. Returns [] when the block is missing.
    """
    motion = spec.get("motion") if isinstance(spec, Mapping) else None
    raw = motion.get("idle_behaviors") if isinstance(motion, Mapping) else None
    out: list[dict] = []
    if not isinstance(raw, Sequence):
        return out
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        bid = item.get("id")
        if not isinstance(bid, str) or not bid:
            continue
        out.append({"id": bid, "weight": _as_float(item.get("weight"), 0.0)})
    return out


def mood_reweights(spec: Mapping[str, Any], mood: str) -> dict[str, float]:
    """The multiplicative reweights for `mood` from dynamic.mood_reweights.

    Shape in the JSON::

        "mood_reweights": { "<mood>": { "<behavior_id>": <multiplier>, ... }, ... }

    Returns {behavior_id: multiplier} for the given mood, or {} when the mood is
    unknown / the block is missing / malformed. A behaviour not named keeps its
    base weight (implicit multiplier 1.0).
    """
    dyn = spec.get("dynamic") if isinstance(spec, Mapping) else None
    table = dyn.get("mood_reweights") if isinstance(dyn, Mapping) else None
    if not isinstance(table, Mapping):
        return {}
    per_mood = table.get(mood)
    if not isinstance(per_mood, Mapping):
        return {}
    out: dict[str, float] = {}
    for bid, mult in per_mood.items():
        if isinstance(bid, str) and bid:
            out[bid] = _as_float(mult, 1.0)
    return out


# --------------------------------------------------------------------------- #
# mood (data/mood.json, written by the director)
# --------------------------------------------------------------------------- #
def current_mood(path: Path | str | None = None) -> str:
    """Read the director's current mood from data/mood.json.

    The director writes ``{"mood": "<word>"}``. Returns that word, or
    DEFAULT_MOOD if the file is missing, unreadable, not JSON, not an object,
    or carries no usable string mood. Never raises.
    """
    p = Path(path) if path is not None else _MOOD_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_MOOD
    if not isinstance(data, Mapping):
        return DEFAULT_MOOD
    mood = data.get("mood")
    if isinstance(mood, str) and mood.strip():
        return mood.strip()
    return DEFAULT_MOOD


# --------------------------------------------------------------------------- #
# clip discovery: find the behaviour-tagged loop clips available right now
# --------------------------------------------------------------------------- #
def _as_float(v: Any, default: float) -> float:
    """bool/int/float -> float (bool excluded -- True must not read as 1.0 weight)."""
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    return default


def _clip_tags(clip: Any) -> list[str]:
    """A clip's tags as a list of strings (empty list if absent/odd-typed)."""
    tags = getattr(clip, "tags", None)
    if not isinstance(tags, (list, tuple)):
        return []
    return [t for t in tags if isinstance(t, str)]


def _is_loop_clip(clip: Any) -> bool:
    """True for a held loop (anchor self-loop): from_node == to_node.

    Falls back to the `loopable` flag when from/to aren't both present, so a
    clip handed in without explicit node names but marked loopable still counts.
    A behaviour loop is by contract a self-loop on the anchor pose, but we accept
    either signal to stay tolerant of how the caller materialises the clip.
    """
    frm = getattr(clip, "from_node", None)
    to = getattr(clip, "to_node", None)
    if frm is not None and to is not None:
        return frm == to
    return bool(getattr(clip, "loopable", False))


def _clip_character(tags: Sequence[str]) -> str | None:
    """The slug a clip is tagged for ("character:<slug>"), or None if untagged."""
    for t in tags:
        if t.startswith("character:"):
            return t.split(":", 1)[1] or None
    return None


def _clip_matches_behavior(tags: Sequence[str], behavior_id: str) -> bool:
    """True when a clip's tags identify it as the loop for `behavior_id`.

    Accepts the bare id ("declaim"), or a namespaced form
    ("behavior:declaim" / "behaviour:declaim"). Matching is exact on the id
    component (no substring), so "declaim" never matches "declaim_long".
    """
    for t in tags:
        if t == behavior_id:
            return True
        if ":" in t:
            ns, _, val = t.partition(":")
            if ns in ("behavior", "behaviour") and val == behavior_id:
                return True
    return False


def _iter_clips(source: Any) -> list[Any]:
    """Normalise the player's `available_clips_or_graph` arg into a clip list.

    Accepts:
      * a ClipGraph-like object (has .edges() -> iterable of clips). We take all
        edges and let the loop/tag filters keep only behaviour self-loops.
      * a single clip-like object (has .tags). Wrapped in a one-element list.
      * any other iterable of clips.
      * None -> [].
    Never raises; an object we can't interpret yields [].
    """
    if source is None:
        return []
    # ClipGraph-like: prefer .edges() (every clip, both directions).
    edges = getattr(source, "edges", None)
    if callable(edges):
        try:
            return list(edges())
        except Exception:
            return []
    # A single clip handed in directly (duck-typed: it has tags).
    if hasattr(source, "tags") and not isinstance(source, (list, tuple, set)):
        return [source]
    # A plain iterable of clips.
    if isinstance(source, Iterable):
        try:
            return list(source)
        except Exception:
            return []
    return []


def available_behavior_clips(
    slug: str,
    source: Any,
    behavior_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Map behaviour-id -> loop clip, for the clips currently in `source`.

    Keeps only clips that (a) are held loops, (b) belong to `slug` (untagged
    clips are accepted as belonging to everyone; a clip tagged for a different
    character is rejected), and (c) match one of `behavior_ids` by tag. When
    `behavior_ids` is None, every behaviour id found on a matching loop clip is
    admitted. If two clips claim the same behaviour id, the first wins (stable
    over edges() order); the caller can dedupe upstream if it cares.
    """
    wanted = set(behavior_ids) if behavior_ids is not None else None
    found: dict[str, Any] = {}
    for clip in _iter_clips(source):
        if not _is_loop_clip(clip):
            continue
        tags = _clip_tags(clip)
        owner = _clip_character(tags)
        if owner is not None and owner != slug:
            continue
        if wanted is not None:
            for bid in wanted:
                if bid not in found and _clip_matches_behavior(tags, bid):
                    found[bid] = clip
        else:
            # admit every behaviour id this clip declares (bare or namespaced)
            for bid in _behavior_ids_in_tags(tags):
                found.setdefault(bid, clip)
    return found


def _behavior_ids_in_tags(tags: Sequence[str]) -> list[str]:
    """Behaviour ids a clip declares: namespaced ones, else every bare non-reserved tag.

    Namespaced ("behavior:X"/"behaviour:X") are taken verbatim. If none are
    namespaced, bare tags that aren't structural/provenance markers
    (hold/synthetic/transition/verified/loop + "ns:..." forms) are treated as
    behaviour ids -- the lenient path for clips tagged simply [behavior.id].
    """
    reserved = {"hold", "synthetic", "transition", "verified", "loop", "loopable"}
    namespaced = []
    bare = []
    for t in tags:
        if ":" in t:
            ns, _, val = t.partition(":")
            if ns in ("behavior", "behaviour") and val:
                namespaced.append(val)
        elif t and t not in reserved:
            bare.append(t)
    return namespaced if namespaced else bare


# --------------------------------------------------------------------------- #
# the weighting math (pure -- the heart of the selftest)
# --------------------------------------------------------------------------- #
def effective_weights(
    behaviors: Sequence[Mapping[str, Any]],
    reweights: Mapping[str, float],
) -> dict[str, float]:
    """base_weight * mood_multiplier per behaviour, clamped to >= 0.

    `behaviors` is the idle_behaviors() shape ({id, weight}); `reweights` is
    mood_reweights() for the active mood ({id: multiplier}, default 1.0). A
    negative product is clamped to 0 (a behaviour can be suppressed, never
    negative-probability). Returns {id: weight}; ids with weight 0 are kept so
    callers can see the full table, but draw_weighted() ignores them.
    """
    out: dict[str, float] = {}
    for b in behaviors:
        bid = b.get("id")
        if not isinstance(bid, str) or not bid:
            continue
        base = _as_float(b.get("weight"), 0.0)
        mult = reweights.get(bid, 1.0)
        out[bid] = max(0.0, base * mult)
    return out


def draw_weighted(
    weights: Mapping[str, float],
    rng: random.Random | None = None,
) -> str | None:
    """Draw one key from {key: weight} with probability proportional to weight.

    Returns None when there are no keys or every weight is <= 0 (nothing to
    pick). Uses the passed rng, else the module RNG. Pure: does not touch files
    or the clock.
    """
    r = rng if rng is not None else _RNG
    items = [(k, w) for k, w in weights.items() if w > 0]
    total = sum(w for _, w in items)
    if not items or total <= 0:
        return None
    x = r.random() * total
    upto = 0.0
    for k, w in items:
        upto += w
        if x < upto:
            return k
    return items[-1][0]  # float-rounding guard: hand back the last key


def distribution(
    behaviors: Sequence[Mapping[str, Any]],
    reweights: Mapping[str, float],
) -> dict[str, float]:
    """Normalised pick probabilities {id: p}, p summing to 1 (or {} if all-zero).

    The analytic distribution draw_weighted samples from -- handy for the
    selftest and for logging "what mood is doing to the body language" without
    running thousands of draws.
    """
    w = effective_weights(behaviors, reweights)
    total = sum(v for v in w.values() if v > 0)
    if total <= 0:
        return {}
    return {k: (v / total if v > 0 else 0.0) for k, v in w.items()}


# --------------------------------------------------------------------------- #
# the public entry point the player calls each cycle
# --------------------------------------------------------------------------- #
def pick_behavior(
    slug: str,
    available_clips_or_graph: Any,
    mood: str | None = None,
    rng: random.Random | None = None,
) -> tuple[Any, str] | None:
    """Choose the next idle behaviour loop for character `slug`.

    Reads the character spec (motion.idle_behaviors + dynamic.mood_reweights),
    resolves the active mood (the `mood` arg, else data/mood.json, else
    DEFAULT_MOOD), restricts to behaviours that actually have a loop clip in
    `available_clips_or_graph`, computes base*mood weights, and draws one.

    Returns (chosen_clip, behavior_id), or None when the character can't perform
    any behaviour yet -- no spec, no idle_behaviors, or none of them have a baked
    loop clip available. None is the player's signal to keep holding its current
    loop (do NOT switch this cycle), so a not-yet-baked library degrades to a
    static hold rather than an error.

    `available_clips_or_graph` may be a ClipGraph, a single clip, or an iterable
    of clips. `rng` lets a test/seeded show make the draw reproducible.
    """
    spec = load_character(slug)
    if spec is None:
        return None
    behaviors = idle_behaviors(spec)
    if not behaviors:
        return None

    # restrict to behaviours we can actually play right now
    clips_by_id = available_behavior_clips(slug, available_clips_or_graph,
                                            [b["id"] for b in behaviors])
    if not clips_by_id:
        return None
    playable = [b for b in behaviors if b["id"] in clips_by_id]
    if not playable:
        return None

    use_mood = mood if mood is not None else current_mood()
    reweights = mood_reweights(spec, use_mood)
    weights = effective_weights(playable, reweights)
    chosen_id = draw_weighted(weights, rng=rng)
    if chosen_id is None:
        # every playable behaviour got weight 0 under this mood (e.g. all bases
        # 0). Fall back to a uniform draw over what IS playable so the portrait
        # still moves rather than freezing -- a degenerate-mood safety net.
        ids = [b["id"] for b in playable]
        chosen_id = (rng or _RNG).choice(ids) if ids else None
        if chosen_id is None:
            return None
    return clips_by_id[chosen_id], chosen_id


# --------------------------------------------------------------------------- #
# self-test: prove the weighting + mood-reweight math on SYNTHETIC data
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Exercise weight/reweight/draw math on synthetic data -- no files, no deps.

    Builds a Phineas-shaped behaviour set, then shows the resulting pick
    distribution under two moods (neutral vs a mood that boosts one behaviour),
    and asserts the math is correct (reweight raises the boosted behaviour's
    share; suppression and degenerate moods behave; the empirical draw over many
    trials tracks the analytic distribution).
    """
    # Phineas's real base weights (from phineas.json.motion.idle_behaviors).
    behaviors = [
        {"id": "declaim", "weight": 0.40},
        {"id": "wounded_sulk", "weight": 0.30},
        {"id": "side_eye_rival", "weight": 0.20},
        {"id": "appraise_viewer", "weight": 0.10},
    ]
    # Two moods: neutral (no reweight) and conspiratorial (side_eye x2.5),
    # mirroring phineas.json.dynamic.mood_reweights.
    moods = {
        "neutral": {},
        "conspiratorial": {"side_eye_rival": 2.5},
    }

    print("behavior_select selftest -- pick distribution by mood")
    print("(synthetic Phineas weights; no files / clip-graph / pygame / torch)\n")

    dists: dict[str, dict[str, float]] = {}
    for mood, rw in moods.items():
        dist = distribution(behaviors, rw)
        dists[mood] = dist
        print(f"  mood = {mood}")
        for bid in (b["id"] for b in behaviors):
            p = dist.get(bid, 0.0)
            bar = "#" * int(round(p * 40))
            print(f"    {bid:<16} {p:6.1%}  {bar}")
        print()

    # --- assertions on the analytic distribution -------------------------- #
    # neutral: probabilities equal the normalised base weights (sum 1.0).
    dn = dists["neutral"]
    assert abs(sum(dn.values()) - 1.0) < 1e-9, dn
    assert abs(dn["declaim"] - 0.40) < 1e-9, dn
    assert abs(dn["side_eye_rival"] - 0.20) < 1e-9, dn

    # conspiratorial: side_eye's share must RISE vs neutral; the others fall;
    # total still 1.0. side_eye weight 0.2*2.5=0.5; total 0.4+0.3+0.5+0.1=1.3.
    dc = dists["conspiratorial"]
    assert abs(sum(dc.values()) - 1.0) < 1e-9, dc
    assert dc["side_eye_rival"] > dn["side_eye_rival"], (dc, dn)
    assert dc["declaim"] < dn["declaim"], (dc, dn)
    assert abs(dc["side_eye_rival"] - (0.5 / 1.3)) < 1e-9, dc

    # suppression: a 0.0 multiplier removes a behaviour entirely.
    ds = distribution(behaviors, {"declaim": 0.0})
    assert ds.get("declaim", 0.0) == 0.0, ds
    assert abs(sum(ds.values()) - 1.0) < 1e-9, ds

    # all-zero bases -> empty analytic distribution (draw_weighted returns None,
    # and pick_behavior's uniform fallback covers the live path).
    zero = distribution([{"id": "x", "weight": 0.0}], {})
    assert zero == {}, zero

    # --- empirical draw tracks analytic (seeded, deterministic) ----------- #
    rng = random.Random(1234)
    N = 40000
    counts = {b["id"]: 0 for b in behaviors}
    w = effective_weights(behaviors, moods["conspiratorial"])
    for _ in range(N):
        counts[draw_weighted(w, rng=rng)] += 1
    print("  empirical draw (conspiratorial, N=%d, seeded):" % N)
    for bid in counts:  # noqa: PLC0206  -- the key is the subject here; values are read from a second map below
        emp = counts[bid] / N
        ana = dc[bid]
        print(f"    {bid:<16} empirical {emp:6.1%}  analytic {ana:6.1%}")
        assert abs(emp - ana) < 0.01, (bid, emp, ana)

    # --- the (clip, id) shape via a synthetic clip list ------------------- #
    # Stand-in clips: duck-typed objects with .tags + equal from/to (loop).
    class _C:
        def __init__(self, bid, slug="phineas"):
            self.from_node = "idle"
            self.to_node = "idle"
            self.loopable = True
            self.tags = ["hold", "synthetic", f"character:{slug}", bid]

    clips = [_C(b["id"]) for b in behaviors]
    by_id = available_behavior_clips("phineas", clips, [b["id"] for b in behaviors])
    assert set(by_id) == {b["id"] for b in behaviors}, by_id
    # a clip tagged for another character must be rejected
    assert available_behavior_clips("phineas", [_C("declaim", slug="seraphina")],
                                    ["declaim"]) == {}
    # bare-tag discovery (no explicit behavior_ids) finds the same four
    auto = available_behavior_clips("phineas", clips, None)
    assert set(auto) == {b["id"] for b in behaviors}, auto

    print("\nbehavior_select OK -- weighting, mood reweight, suppression, "
          "empirical match, and clip matching all verified.")


def _print_mood_distribution(slug: str) -> None:
    """Helper for `--show <slug>`: print a real character's distribution per its
    own moods, read from the actual JSON (proves the file parses + the math)."""
    spec = load_character(slug)
    if spec is None:
        print(f"no character spec for {slug!r} at {character_path(slug)}")
        return
    behaviors = idle_behaviors(spec)
    dyn = spec.get("dynamic", {}) if isinstance(spec, dict) else {}
    table = dyn.get("mood_reweights", {}) if isinstance(dyn, dict) else {}
    moods = ["neutral", *table.keys()]
    print(f"{slug}: idle behaviours = {[b['id'] for b in behaviors]}")
    for mood in moods:
        rw = mood_reweights(spec, mood)
        dist = distribution(behaviors, rw)
        head = " ".join(f"{k}={v:.0%}" for k, v in dist.items())
        print(f"  mood={mood:<15} {head}")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if "--selftest" in args:
        _selftest()
    elif "--show" in args:
        i = args.index("--show")
        slug = args[i + 1] if i + 1 < len(args) else "phineas"
        _print_mood_distribution(slug)
    else:
        print("usage: python runtime/behavior_select.py --selftest | --show <slug>")
