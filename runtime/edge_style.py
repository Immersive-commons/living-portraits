"""edge_style.py -- MANNER and VALENCE for a graph edge, derived from its own prose.

The audit's finding is blunt: the edge type system is ONE BIT. `kind` is `idle` or
`transition` and both mean "a playable video clip" (`_audit/CONTEXT_GRAPH_AUDIT.md (internal)`
S1.1, S3). The `tags` field, the one place edge semantics could live, holds
`[kind, label, kind]` and no runtime module reads it. Every claim about typed edges
carrying relational semantics is currently describing somebody else's system.

The type information was never missing. It was sitting in `motion_prompt`, the text
that was actually sent to the video generator, unread by anything at runtime:

    "the tragedian unbends from his glare and settles back square into the seated rest"
    "Body snaps forward from the relaxed slump, hand jerking up to claw at the air"

Those two clips are both `kind: transition`. One is a `drift`, one is a `surge`. This
module reads that prose and returns the difference, so `policy.weigh` can prefer a
manner the current mood would actually choose.

WHAT IT DERIVES
    manner   -- one of MANNERS (10), or None when the prose carries no movement verb
                this lexicon knows. None is a real answer, not a bucket: an edge with
                no manner gets multiplier 1.0 and behaves exactly as it does today.
    valence  -- roughly -1..+1. The manner's own base tone, adjusted by any affect
                words in the prose ("resentful", "triumphant"). Affect words are rare
                in the real corpus (~20 edges each at best), so the manner carries most
                of it and the affect lexicon only nudges.

THE REVERSE-PROSE TRAP (why `index()` exists and `derive()` alone is not enough)
    `video_graph.py:423,480,501` builds a reverse edge's prompt as `"(reverse) " + motion`
    where `motion` is the FORWARD clip's text. So on 316 of the 1,472 real edges the prose
    narrates the OPPOSITE direction of travel, and a naive read types a rise as a collapse.
    The marker alone is not the discriminator -- ~36 hand-authored `(reverse)` entries in
    `video_graph.py:337-365` do describe their own direction. `index()` settles it from the
    data instead of the marker: a marked edge is inverted only when an UNMARKED edge going
    the other way carries the identical text. Directional manners flip (collapse<->rise,
    reach<->recoil); the rest are their own inverse (a bow run backwards is still a
    flourish, a glitch backwards is still a dissolve).

DELIBERATELY NOT HERE
    No LLM at runtime -- `policy.weigh` is called by the 10fps render loop. No embeddings
    over a 600-word motion vocabulary. No write-back into the graph: this is derived at
    READ time from a field the pipeline already wrote, so nothing has to be regenerated
    and no snapshot has to be migrated.

Pure + import-safe (stdlib only), same contract as `journal_score.py`.

    from runtime import edge_style
    styles = edge_style.index(graph.edges)          # once, at load: {edge_id: Style}
    edge_style.histogram(styles)                    # Counter, for the honesty report
    w = edge_style.affinity(styles.get(eid), "weary")   # 1.0 when either is missing
"""
from __future__ import annotations

import re
from collections import Counter, namedtuple

# --- the closed vocabulary. Chosen by reading the 704 real transition prompts in
# data/_realdata/video_graph.json, not by imagining what a motion prompt looks like.
# Ordered by descending specificity: ties in the cue score break toward the earlier
# entry, so a prompt that is both a lunge and a slow drift reads as the lunge.
MANNERS = ("surge", "collapse", "rise", "recoil", "reach", "dissolve",
           "flourish", "fidget", "still", "drift")

# Directional manners invert when a clip is played the other way; the rest are their
# own inverse. Used ONLY by index(), on edges proven to carry their twin's prose.
INVERSE_MANNER = {"collapse": "rise", "rise": "collapse",
                  "reach": "recoil", "recoil": "reach"}

# Base tone of each manner, before any affect words in the prose. A collapse reads
# negative whatever the costume; a rise reads positive.
MANNER_VALENCE = {
    "surge": -0.15, "collapse": -0.60, "rise": 0.50, "recoil": -0.50,
    "reach": 0.35, "dissolve": -0.10, "flourish": 0.40, "fidget": -0.05,
    "still": 0.0, "drift": 0.10,
}

STRONG = 2.0   # an unambiguous manner verb ("slumps", "lunges", "dissolves")
WEAK = 1.0     # a supporting word that only counts alongside others ("slowly", "away")
MIN_SCORE = 2.0   # below this the prose has not named a manner -> None, not a bucket


def _cues(strong, weak):
    d = dict.fromkeys(strong.split(), STRONG)
    for w in weak.split():
        d.setdefault(w, WEAK)
    return d


# Every cue below appears in the real corpus; the frequency check that built this is
# reproducible from data/_realdata/video_graph.json.
CUES = {
    "surge": _cues(
        "snap snaps snapping snapped violent violently lunge lunges lunging jerk jerks"
        " jerking whip whips whipping slam slams explode explodes explosive spasm"
        " jackknife thrust thrusts hurls flings flung stab lash lashes rips slashes"
        " punches strikes surge surges burst bursts bursting storms charges",
        "sudden suddenly sharp sharply rapid rapidly abrupt abruptly hard force forceful"
        " electric fierce furious rage fury violent aggressively"),
    "collapse": _cues(
        "slump slumps slumped slumping collapse collapses collapsing crumple crumples"
        " crumpling sag sags sagging wilt wilts wilting buckle buckles droop droops"
        " slouch crumbles crumbling caves",
        "sink sinks sinking lowers lowering fold folds folding heavy downward weight"
        " defeated slack limp"),
    "rise": _cues(
        "rise rises rising straighten straightens straightening unbend unbends unfurl"
        " unfurls ascend ascends soars levitates towering",
        "lift lifts lifting upright stands standing upward climbs lengthens elongates"
        " vertebra"),
    "recoil": _cues(
        "recoil recoils recoiling flinch flinches flinching shrink shrinks shrinking"
        " retreat retreats withdraw withdraws cringe cringes shudder shudders shuddering"
        " shies wince winces jerking",
        "away backward retreats tense tenses clutch clutches inward guarded hunches"
        " curls shields resentful"),
    "reach": _cues(
        "reach reaches reaching extend extends extending offer offers offering cups"
        " cupping beckons outstretched outward",
        "gather gathers gathering spread spreads spreading wide toward palms presents"
        " forward open opens"),
    "dissolve": _cues(
        "dissolve dissolves dissolving fragment fragments fragmenting pixel pixels"
        " pixelated glitch glitches glitching glitched datamosh shards scatter scatters"
        " fracture fractures fractured reassemble reassembles dematerialize shimmer",
        "particles static corrupted distort distorts flicker flickers ripple ripples"
        " cascade streams unravel"),
    "flourish": _cues(
        "flourish flourishes theatrical theatrically grand grandly bow bows bowing"
        " proclaim declaims ovation sweeping sweeps sweep majestic operatic swaggers"
        " swagger",
        "dramatic dramatically gesture gestures arc deliberate deliberately audience"
        " stage applause chin grandiose"),
    "fidget": _cues(
        "tap taps tapping flick flicks flicking twitch twitches twitching tremble"
        " trembles trembling fidget scratches shivers quiver",
        "adjusts rubs picks micro repeated small once jitters restless"),
    "still": _cues(
        "freeze freezes freezing frozen motionless unmoving stillness rigid statue"
        " transfixed petrified",
        "still holds held stays remains fixed locked pause pauses stares"),
    "drift": _cues(
        "drift drifts drifting glide glides gliding saunter saunters languid float"
        " floats floating sway sways swaying trance meander ambles strolls shuffles"
        " pads",
        "slow slowly gently gentle soft softly ease eases easing settle settles"
        " settling calm smooth unhurried quiet lazily"),
}

# Affect words that survive a frequency check against the real corpus. Sparse on purpose:
# they nudge a manner's base tone, they never carry the valence on their own.
AFFECT = {
    -0.5: ("jealous jealousy rage furious fury resentful bitter grief mournful sorrow"
           " despair anguish contempt disdain scorn snarl sneer venom spite cruel"),
    -0.3: ("weary tired exhausted defeated sulky sulk sullen glare glower grim bleak"
           " hollow melancholy tragic tragedy wounded broken desperate lonely shame"
           " dread panic"),
    0.3: ("gentle tender warm warmth affection contented content serene peaceful calm"
          " grace graceful hopeful reverent wonder awe smile smiles grin"),
    0.5: ("triumphant triumph joy joyful delight delighted proud pride rapture ecstatic"
          " radiant bliss victory cocky playful"),
}
AFFECT_WEIGHT = 0.35    # per hit
AFFECT_CAP = 0.6        # total affect adjustment, either direction

# --- how a mood band prefers a manner. Multipliers only: >1 prefers, <1 discourages,
# and STYLE_FLOOR keeps the smallest possible product well above zero so styling can
# never strand a walk or veto an edge. Band names are policy.MOOD_BIAS's 8 keys plus
# the two free-text energy buckets policy falls back to.
STYLE_AFFINITY = {
    "restless": {"surge": 1.50, "fidget": 1.30, "dissolve": 1.20, "recoil": 1.05,
                 "drift": 0.70, "still": 0.60, "collapse": 0.80},
    "curious":  {"reach": 1.50, "fidget": 1.20, "drift": 1.15, "rise": 1.10,
                 "surge": 0.85, "collapse": 0.80},
    "playful":  {"flourish": 1.40, "surge": 1.20, "fidget": 1.20, "rise": 1.15,
                 "collapse": 0.70, "recoil": 0.75},
    "weary":    {"collapse": 1.50, "drift": 1.35, "still": 1.20, "rise": 0.80,
                 "surge": 0.60, "flourish": 0.75},
    "calm":     {"drift": 1.45, "still": 1.25, "reach": 1.10, "fidget": 0.85,
                 "dissolve": 0.80, "surge": 0.60},
    "serene":   {"drift": 1.45, "rise": 1.20, "still": 1.20, "fidget": 0.80,
                 "recoil": 0.70, "surge": 0.60},
    "fixated":  {"still": 1.50, "recoil": 1.20, "surge": 1.10, "flourish": 0.80,
                 "drift": 0.75, "dissolve": 0.85},
    "wistful":  {"drift": 1.30, "collapse": 1.25, "reach": 1.15, "still": 1.10,
                 "flourish": 0.85, "surge": 0.70},
    # policy's free-text energy buckets, for a mood that names no band
    "high":     {"surge": 1.45, "fidget": 1.25, "dissolve": 1.15, "rise": 1.10,
                 "drift": 0.70, "still": 0.65},
    "low":      {"drift": 1.40, "collapse": 1.30, "still": 1.20, "reach": 1.05,
                 "surge": 0.62, "flourish": 0.80},
}

# Where each band sits on the valence axis. An edge whose tone is near the band's own
# gets a small lift; one at the far end gets a small cut. Bounded by VALENCE_PULL, so
# this can only ever lean the choice the manner term already made.
BAND_VALENCE = {
    "restless": -0.1, "curious": 0.3, "playful": 0.5, "weary": -0.4,
    "calm": 0.3, "serene": 0.4, "fixated": -0.3, "wistful": -0.3,
    "high": 0.0, "low": -0.2,
}
VALENCE_PULL = 0.18     # |mult - 1| ceiling from the valence term alone
STYLE_FLOOR = 0.35      # a styled edge is never worth less than this share of its weight
STYLE_CEIL = 1.90       # ...nor more than this. Both bound the whole product.

_WORD = re.compile(r"[a-z]+")
_REVERSE_MARK = re.compile(r"^\(\s*reverse\b[^)]*\)\s*", re.I)

Style = namedtuple("Style", "manner valence score")
NEUTRAL = Style(None, 0.0, 0.0)


def strip_reverse(text):
    """(prose without the '(reverse ...)' marker, was_marked). The marker is a pipeline
    annotation, never part of the motion description, so it must not reach the cues."""
    s = str(text or "").strip()
    m = _REVERSE_MARK.match(s)
    return (s[m.end():].strip(), True) if m else (s, False)


def _scores(text):
    counts = Counter(_WORD.findall(str(text or "").lower()))
    out = {}
    for manner, cues in CUES.items():
        total = 0.0
        for word, n in counts.items():
            w = cues.get(word)
            if w:
                total += w * n
        out[manner] = total
    return out


def _affect(text):
    counts = Counter(_WORD.findall(str(text or "").lower()))
    adj = 0.0
    for weight, words in AFFECT.items():
        for w in words.split():
            if w in counts:
                adj += weight * AFFECT_WEIGHT * counts[w]
    return max(-AFFECT_CAP, min(AFFECT_CAP, adj))


def derive(motion_prompt):
    """Style for ONE piece of motion prose, read literally as written.

    No graph context, so no reverse-prose correction -- see `index()` for that. Prose
    that names no manner this lexicon knows returns NEUTRAL, whose manner is None and
    whose affinity is exactly 1.0: an honest "I cannot type this edge"."""
    text, _ = strip_reverse(motion_prompt)
    if not text:
        return NEUTRAL
    scores = _scores(text)
    best = max(scores.values()) if scores else 0.0
    if best < MIN_SCORE:
        return NEUTRAL
    manner = next(m for m in MANNERS if scores[m] == best)
    valence = MANNER_VALENCE.get(manner, 0.0) + _affect(text)
    return Style(manner, max(-1.0, min(1.0, valence)), best)


def invert(style):
    """The same clip seen playing the other way: directional manners flip, the rest are
    their own inverse, and the valence follows the manner it now has."""
    if not style or not style.manner:
        return NEUTRAL
    manner = INVERSE_MANNER.get(style.manner, style.manner)
    if manner == style.manner:
        return style
    shift = MANNER_VALENCE.get(manner, 0.0) - MANNER_VALENCE.get(style.manner, 0.0)
    return Style(manner, max(-1.0, min(1.0, style.valence + shift)), style.score)


def for_edge(edge):
    """Style from one edge in isolation. Convenience over `derive`; carries the same
    caveat -- a reverse edge whose prose was copied from its twin reads backwards here.
    Prefer `index()` on a whole graph."""
    return derive((edge or {}).get("motion_prompt"))


def _copied_prose_ids(edges):
    """Edge ids whose prose is PROVABLY their twin's, so their manner must be inverted.

    Proof, not heuristic: the edge carries the `(reverse)` marker AND an unmarked edge
    running the opposite way carries byte-identical prose -- which is exactly what
    `video_graph.py` writes when it stamps `"(reverse) " + motion` onto the return leg.
    A hand-authored `(reverse)` line with prose of its own has no such twin and is left
    alone."""
    plain, marked = {}, []
    for e in edges:
        text, was_marked = strip_reverse(e.get("motion_prompt"))
        key = text.lower()
        if was_marked:
            marked.append((e, key))
        elif key:
            plain.setdefault(key, []).append(e)
    out = set()
    for e, key in marked:
        for twin in plain.get(key, ()):
            if twin.get("from") == e.get("to") and twin.get("to") == e.get("from"):
                out.add(e.get("id"))
                break
    return out


def index(edges):
    """{edge_id: Style} over a whole graph, reverse-prose corrected. THE entry point.

    Only typed edges appear, so a plain `.get(id)` returns None for everything this
    lexicon could not read, and `affinity(None, band) == 1.0`. Cheap enough to build at
    graph load (two linear passes, no I/O) and stable, so the render loop builds it once
    and never again."""
    edges = list(edges or ())
    flip = _copied_prose_ids(edges)
    out = {}
    for e in edges:
        style = derive(e.get("motion_prompt"))
        if not style.manner:
            continue
        eid = e.get("id")
        if eid is None:
            continue
        out[eid] = invert(style) if eid in flip else style
    return out


def histogram(styles):
    """Counter over manners, for the coverage report. Accepts an `index()` result, an
    iterable of Style, or raw edges."""
    if isinstance(styles, dict):
        values = styles.values()
    else:
        values = [s if isinstance(s, Style) else for_edge(s) for s in (styles or ())]
    return Counter(s.manner for s in values if getattr(s, "manner", None))


def affinity(style, band, strength=1.0):
    """How much this band WANTS this manner, as a bounded multiplier on the edge weight.

    1.0 -- exactly no opinion -- whenever the edge has no manner, the band is unknown, or
    strength is 0. It is a preference and never a veto: the result is clamped into
    [STYLE_FLOOR, STYLE_CEIL], both strictly positive, so a styled edge always keeps a
    real share of its weight and a character in a foul mood can still take the only exit
    out of a leaf pose."""
    manner = getattr(style, "manner", None)
    if not manner or not band or not strength:
        return 1.0
    table = STYLE_AFFINITY.get(str(band).strip().lower())
    if table is None:
        return 1.0
    mult = table.get(manner, 1.0)
    target = BAND_VALENCE.get(str(band).strip().lower())
    if target is not None:
        valence = getattr(style, "valence", 0.0) or 0.0
        mult *= 1.0 + VALENCE_PULL * (1.0 - abs(valence - target))
    s = max(0.0, min(1.0, float(strength)))
    mult = 1.0 + (mult - 1.0) * s       # strength interpolates toward "no opinion"
    return max(STYLE_FLOOR, min(STYLE_CEIL, mult))


def coverage(edges):
    """{n, typed, untyped, share, by_kind, manners} -- the honesty report, computed from
    whatever graph you hand it rather than quoted from a README."""
    edges = list(edges or ())
    styles = index(edges)
    by_kind = {}
    for e in edges:
        kind = e.get("kind") or "?"
        seen, typed = by_kind.setdefault(kind, [0, 0])
        by_kind[kind] = [seen + 1, typed + (1 if e.get("id") in styles else 0)]
    return {
        "n": len(edges),
        "typed": len(styles),
        "untyped": len(edges) - len(styles),
        "share": (len(styles) / float(len(edges))) if edges else 0.0,
        "by_kind": {k: {"n": v[0], "typed": v[1],
                        "share": (v[1] / float(v[0])) if v[0] else 0.0}
                    for k, v in by_kind.items()},
        "manners": dict(histogram(styles)),
    }
