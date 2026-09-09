"""voice_eval.py -- measure per-character voice DISTINCTNESS + FIDELITY via z.ai.

Runs each character on a fixed battery of identical situational prompts, using the REAL
deployed system prompt (heartbeat.BASE_SYSTEM + _identity_block) -- so it tests whatever
the heartbeat currently builds. Scores:

  * catchphrase_rate  -- fraction of a character's outputs that fire >=1 of its catchphrases
  * attribution_acc   -- a BLIND judge, shown one line + the 3 character essences, guesses
                         who said it. High accuracy == voices are distinct AND faithful
                         (no bleed). Random baseline with 3 characters = 0.33.

Re-run before/after any prompt change to PROVE the voice improved (or didn't regress):
    python director/voice_eval.py                 # 1 draw per (char,prompt)
    python director/voice_eval.py --draws 2        # more draws
    python director/voice_eval.py --tag before     # label the run

Writes the full transcript + scorecard to data/mind/voice_eval_<tag>.json (gitignored).
Stdlib + the z.ai client only; key auto-resolves (see director/llm.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from director import llm
from director.heartbeat import BASE_SYSTEM, _identity_block, _load_char_spec

CHARACTERS = ["phineas", "maxx", "seraphina"]

# identical situational battery -- the ONLY variable across characters is who they are
BATTERY = [
    "It is a quiet afternoon on the wall. In one or two sentences of inner monologue, react to the brighter portrait hanging next to you.",
    "A crowd of strangers just walked past without once looking at you. React in one or two sentences.",
    "Someone has stopped and is studying you closely. React in one or two sentences.",
    "It is late and the gallery has emptied out. One or two sentences of what is on your mind.",
]

EVAL_TEMP = 0.8          # held constant before/after so the prompt change is the only variable
JUDGE_MODEL = "glm-4.5-air"


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower())


def _catchphrases(spec):
    return (spec.get("personality", {}) or {}).get("catchphrases", []) or []


def _fires_catchphrase(text, phrases):
    t = _norm(text)
    for ph in phrases:
        core = _norm(ph)
        if not core:
            continue
        if core in t:                       # whole catchphrase
            return True
        words = core.split()
        if len(words) >= 3 and " ".join(words[:3]) in t:   # distinctive opening
            return True
    return False


def _essences():
    out = {}
    for c in CHARACTERS:
        sp = _load_char_spec(c)
        out[c] = (sp.get("concept", {}) or {}).get("essence", "") or sp.get("name", c)
    return out


def _judge(line, essences):
    """Blind attribution: which character wrote this line? Returns a character id or ''."""
    roster = "\n".join("- %s: %s" % (c, essences[c]) for c in CHARACTERS)
    sysmsg = ("You identify which character wrote a line of inner monologue. "
              "Reply with ONLY one word: " + ", ".join(CHARACTERS) + ".")
    user = "Characters:\n%s\n\nLine: \"%s\"\n\nWhich character wrote it? Reply with one word only." % (roster, line)
    try:
        ans = llm.complete(sysmsg, user, model=JUDGE_MODEL, max_tokens=8, temperature=0.0)
    except llm.LLMError:
        return ""
    a = _norm(ans)
    for c in CHARACTERS:
        if c in a:
            return c
    return ""


def run(draws=1, tag="run"):
    essences = _essences()
    specs = {c: _load_char_spec(c) for c in CHARACTERS}
    systems = {c: BASE_SYSTEM + "\n\n" + _identity_block(c, specs[c])[1] for c in CHARACTERS}
    rows = []
    print("voice_eval [%s]: %d chars x %d prompts x %d draws @ temp %.1f\n" % (
        tag, len(CHARACTERS), len(BATTERY), draws, EVAL_TEMP))
    for c in CHARACTERS:
        for pi, prompt in enumerate(BATTERY):
            for d in range(draws):
                try:
                    out = llm.complete(systems[c], prompt, model=llm.DEFAULT_MODEL,
                                       max_tokens=140, temperature=EVAL_TEMP).strip()
                except llm.LLMError as e:
                    print("  %s p%d d%d: ERROR %s" % (c, pi, d, e))
                    continue
                fired = _fires_catchphrase(out, _catchphrases(specs[c]))
                guess = _judge(out, essences)
                rows.append({"char": c, "prompt_i": pi, "draw": d, "output": out,
                             "catchphrase": fired, "attributed_to": guess, "correct": guess == c})
                print("  [%s] %s%s -> judged %-9s | %s" % (
                    c, "PHRASE " if fired else "       ",
                    "OK" if guess == c else "  ", guess or "?", out[:90].replace("\n", " ")))
        print()

    # ---- scorecard ----
    def acc(pred):
        sub = [r for r in rows if pred(r)]
        return (sum(r["correct"] for r in sub) / len(sub)) if sub else 0.0

    def cphr(pred):
        sub = [r for r in rows if pred(r)]
        return (sum(r["catchphrase"] for r in sub) / len(sub)) if sub else 0.0

    print("=" * 54)
    print("SCORECARD [%s]" % tag)
    print("  %-10s  attribution  catchphrase" % "character")
    for c in CHARACTERS:
        print("  %-10s  %5.0f%%       %5.0f%%" % (c, 100 * acc(lambda r, c=c: r["char"] == c),
                                                  100 * cphr(lambda r, c=c: r["char"] == c)))
    overall_attr = acc(lambda r: True)
    overall_phr = cphr(lambda r: True)
    print("  %-10s  %5.0f%%       %5.0f%%" % ("OVERALL", 100 * overall_attr, 100 * overall_phr))
    print("  (random-guess attribution baseline = 33%%)")
    # confusion: who gets mistaken for whom
    conf = {}
    for r in rows:
        if not r["correct"] and r["attributed_to"]:
            conf[(r["char"], r["attributed_to"])] = conf.get((r["char"], r["attributed_to"]), 0) + 1
    if conf:
        print("  bleed:", ", ".join("%s->%s x%d" % (a, b, n) for (a, b), n in sorted(conf.items())))
    print("=" * 54)

    out_dir = ROOT / "data" / "mind"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"tag": tag, "draws": draws, "temp": EVAL_TEMP,
               "overall_attribution": round(overall_attr, 3), "overall_catchphrase": round(overall_phr, 3),
               "per_char": {c: {"attribution": round(acc(lambda r, c=c: r["char"] == c), 3),
                                "catchphrase": round(cphr(lambda r, c=c: r["char"] == c), 3)} for c in CHARACTERS},
               "rows": rows}
    p = out_dir / ("voice_eval_%s.json" % re.sub(r"[^a-z0-9_]+", "_", tag.lower()))
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote %s" % p)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--tag", default="run")
    run(draws=ap.parse_args().draws, tag=ap.parse_args().tag)


if __name__ == "__main__":
    main()
