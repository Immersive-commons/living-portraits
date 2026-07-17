"""mj_safe.py -- the moderation guard between the LLM and Midjourney.

The #1 risk of unattended MJ generation: an LLM-written prompt trips MJ's NSFW
filter, the account gets a temp block + "manual review", and -- critically --
*retrying during a block extends it*. So every prompt the autogen loop sends MUST
pass through here first. Two hard lessons from the build (see memory
project_living_portraits MJ-moderation gotcha + feedback_mj_oref_ow_suppresses_wardrobe):

  1. A garment/body word in a `--no` NEGATIVE is a landmine: `--no clothing` got the
     bare word "clothing" flagged as nsfw -> account block. So NEGATIVES are
     allow-listed: we keep ONLY style/frame/quality terms and drop everything else.
  2. Sleepwear + "yawn"/"stretch" motion read as possibly-sexual and moderate
     stochastically. So those are SOFT warnings on a motion prompt, and the hard
     anatomy/NSFW vocabulary is a HARD reject anywhere.

Pure + stdlib. Conservative by design: when unsure, strip or reject. False positives
(a dropped safe word) cost nothing; a false negative costs the whole account.
"""
from __future__ import annotations

import re

# HARD: presence ANYWHERE (positive, motion, negative) rejects the prompt outright.
# Anatomy / nudity / explicit vocabulary -- never worth sending.
_HARD_BLOCK = {
    "nude", "nudes", "naked", "nudity", "nsfw", "explicit", "sexual", "sexy",
    "erotic", "erotica", "lingerie", "underwear", "bra", "panties", "thong",
    "cleavage", "breast", "breasts", "nipple", "nipples", "genital", "genitals",
    "topless", "shirtless", "bare-chested", "barechested", "undress", "undressed",
    "undressing", "disrobe", "disrobing", "provocative", "seductive", "fetish",
    "bdsm",
}

# SOFT: allowed, but flagged. (1) Sleepwear/relaxation motion that moderated
# stochastically for a woman in nightclothes; (2) gore/violence vocabulary, which is
# THEMATIC for these characters (a melodramatic tragedian's "wounded grandeur", a combat
# cyborg's weapon poses) and which MJ permits in moderation -- so warn, don't reject.
# The caller may reword. Hard bans stay focused on the real account-killers: sex/nudity.
_SOFT_WARN = {
    "yawn", "yawning", "stretch", "stretching", "lying", "reclining", "bed",
    "nightgown", "nightdress", "negligee", "robe", "bathrobe", "bare",
    "blood", "bloody", "wound", "wounded", "gore", "gory", "mutilation", "corpse", "violent",
}

# NEGATIVE allow-list: the ONLY families of `--no` terms we permit. These are
# style / framing / quality / count negatives -- never body or garment words.
# A negative term is kept only if it matches one of these; anything else is dropped.
_NEG_ALLOW = {
    # framing / borders
    "frame", "picture frame", "gilded frame", "ornate frame", "border", "borders",
    "framed", "framed painting", "gallery wall", "painting on a wall",
    # text / marks
    "text", "watermark", "watermarks", "signature", "caption", "letters", "words",
    "logo", "ui", "subtitles",
    # quality / artifacts
    "blurry", "blur", "lowres", "low quality", "low-res", "jpeg artifacts",
    "deformed", "disfigured", "mutated", "extra fingers", "extra limbs", "extra arms",
    "missing limbs", "bad anatomy", "bad hands", "long neck", "cropped", "out of frame",
    "duplicate", "error", "noise", "grain", "oversaturated", "ugly",
    # medium / style we don't want
    "photograph", "photo", "3d render", "render", "cartoon", "anime", "sketch",
    "pencil sketch", "charcoal sketch", "monochrome", "grayscale", "greyscale",
    "black and white", "desaturated", "craquelure", "cracked varnish",
    # count
    "two people", "multiple people", "crowd", "group", "extra people", "second person",
}

_WORD = re.compile(r"[a-z][a-z\-]*")


def _inflect(w):
    """Common inflections of a base risk word, so 'yawn' also catches yawns/yawning,
    'undress' catches undresses/undressing, 'stretch' catches stretches. We match
    whole TOKENS against these forms (not substrings), so short words like 'bra' don't
    false-match 'brave'/'brain'. Multi-word entries (e.g. 'bare-chested') pass through."""
    if " " in w:
        return {w}
    forms = {w, w + "s", w + "ed", w + "ing", w + "d"}
    if w.endswith(("s", "sh", "ch", "x", "z")):
        forms.add(w + "es")                 # stretch->stretches, undress->undresses
    if w.endswith("e"):
        forms.add(w[:-1] + "ing")           # disrobe->disrobing
    return forms


# token -> base-word maps (a matched inflected token reports its BASE so callers see
# stable vocabulary, e.g. 'undresses' -> 'undress').
_HARD_MAP = {f: b for b in _HARD_BLOCK for f in _inflect(b)}
_SOFT_MAP = {f: b for b in _SOFT_WARN for f in _inflect(b)}
_RISK_MAP = {**_SOFT_MAP, **_HARD_MAP}      # hard wins on collision


def _tokens(text):
    return set(_WORD.findall((text or "").lower()))


def scan(text):
    """Return (hard, soft): BASE risk words found in a positive/motion prompt.
    `hard` -> reject the prompt; `soft` -> allowed but worth rewording."""
    hard, soft = set(), set()
    for t in _tokens(text):
        if t in _HARD_MAP:
            hard.add(_HARD_MAP[t])
        elif t in _SOFT_MAP:
            soft.add(_SOFT_MAP[t])
    return sorted(hard), sorted(soft)


def is_safe(text):
    """True if a positive/motion prompt carries no HARD-blocked vocabulary."""
    return not any(t in _HARD_MAP for t in _tokens(text))


def sanitize_negatives(neg):
    """Allow-list a list (or comma-string) of `--no` terms. Returns (kept, dropped).
    A term is kept ONLY if it is a recognized style/frame/quality/count negative; any
    body/garment/unknown term is dropped (the `--no clothing` landmine). Multi-word
    allow entries match as substrings of a term so 'gilded frame' is kept whole."""
    if isinstance(neg, str):
        terms = [t.strip() for t in neg.split(",") if t.strip()]
    else:
        terms = [str(t).strip() for t in (neg or []) if str(t).strip()]
    kept, dropped = [], []
    for t in terms:
        tl = t.lower()
        ok = tl in _NEG_ALLOW or any(
            (" " in a and a in tl) or (a == tl) for a in _NEG_ALLOW
        )
        # never keep a term that contains hard/soft body vocabulary, even if it also
        # contains an allowed word (e.g. "naked frame")
        if ok and not any(tok in _RISK_MAP for tok in _tokens(t)):
            kept.append(t)
        else:
            dropped.append(t)
    return kept, dropped


def safe_negative_string(neg):
    """sanitize_negatives -> a comma string ready for `--no`, dropped terms discarded."""
    kept, _ = sanitize_negatives(neg)
    return ", ".join(kept)


def check(positive, motion=None, negatives=None):
    """One-call gate for the autogen worker. Returns a verdict dict:
        {ok, reasons, hard, soft, safe_negatives, dropped_negatives}
    ok=False means DO NOT SUBMIT. The caller logs reasons and skips (never retries
    a hard-blocked prompt -- that is how blocks get extended)."""
    hard_p, soft_p = scan(positive)
    hard_m, soft_m = scan(motion or "")
    kept, dropped = sanitize_negatives(negatives)
    hard = sorted(set(hard_p) | set(hard_m))
    soft = sorted(set(soft_p) | set(soft_m))
    reasons = []
    if hard:
        reasons.append("hard-blocked vocabulary: " + ", ".join(hard))
    if dropped:
        reasons.append("dropped unsafe negatives: " + ", ".join(dropped))
    if soft:
        reasons.append("soft-warn (allowed): " + ", ".join(soft))
    return {
        "ok": not hard,
        "reasons": reasons,
        "hard": hard,
        "soft": soft,
        "safe_negatives": kept,
        "dropped_negatives": dropped,
    }
