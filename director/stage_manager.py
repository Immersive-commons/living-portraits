"""
stage_manager.py -- the playwright/director.

Reads the prompt library + current signals, asks qwen3 (local Ollama on SC2)
for the next short beat of the two-portrait play, and writes data/stage_state.json
(atomically) for the player to perform.

    python director/stage_manager.py            # one beat
    python director/stage_manager.py --loop 45   # a beat every 45s (the live show)
"""
import datetime
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from director import signals  # director/ is on sys.path when run as a script

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
DATA = ROOT / "data"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3:8b"
ACTIONS = ["idle", "address_house", "lean_in", "glance_aside", "ponder", "leaving", "arriving"]


def read(path):
    return path.read_text(encoding="utf-8-sig")  # tolerate BOM


def build_system():
    directives = read(PROMPTS / "_stage-directives.md")
    a = read(PROMPTS / "characters" / "phineas.md")
    b = read(PROMPTS / "characters" / "seraphina.md")
    return (
        directives
        + "\n\nCAST:\n[PANEL A]\n" + a + "\n[PANEL B]\n" + b
        + "\n\nYou are the STAGE MANAGER. Compose the next short beat of the ongoing"
        + " two-portrait play. Both characters break the fourth wall. Each aside is"
        + " ONE or TWO sentences, strictly in that character's voice.\n"
        + "action MUST be exactly one of: " + ", ".join(ACTIONS) + ".\n"
        + 'Return ONLY JSON, no prose:\n'
        + '{"scene":"<mood/scene tag>",'
        + '"A":{"action":"<action>","aside":"<Phineas line>"},'
        + '"B":{"action":"<action>","aside":"<Seraphina line>"},'
        + '"bridge":"<short note or null>"}\n/no_think'
    )


def build_user(sig):
    return (
        "It is " + sig["now"] + " (" + sig["part_of_day"] + "). "
        + "The house mood is " + sig["mood"] + ".\n"
        + "Today's material the actors may riff on:\n- "
        + "\n- ".join(sig["material"])
    )


def call_ollama(system, user):
    body = json.dumps({
        "model": MODEL,
        "stream": False,
        "format": "json",
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.85},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]["content"]


def parse_beat(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(match.group(0) if match else text)


def clamp(action):
    return action if action in ACTIONS else "idle"


# --- cross-frame walk cadence ------------------------------------------------ #
# Every Nth beat, instead of two independent portraits, one character walks out of
# its panel and into the other (player.py + runtime/crossframe.py perform it from
# a top-level `move`). Kept rare so it reads as a deliberate beat, not a scene cut.
MOVE_EVERY = 3            # emit a move on beats 3, 6, 9, ... (beat 0 establishes the cast)
_BEAT_COUNT = 0           # run_once() calls so far this process
_MOVE_TOGGLE = 0          # alternates the direction each time a move is emitted

# Who lives where at rest, and who walks for each direction. With two panels and
# two characters, an A->B walk carries Panel A's resident (Phineas) into B, and a
# B->A walk carries Panel B's resident (Seraphina) into A.
PANEL_CHAR = {"A": "Phineas", "B": "Seraphina"}


def maybe_move(state):
    """Occasionally fold a cross-frame `move` into a freshly-built beat (in place).

    Additive + optional: when it fires it adds a top-level ``move``
    ({"char","from","to"}) and, per crossframe's contract, makes BOTH panel chars
    name the mover for the duration (so the figure layers + bg plates resolve to
    the mover's assets and the post-walk static beat is consistent) while keeping
    each panel's generated action/aside. The action tags are nudged to
    leaving/arriving so the text reads with the motion. The beat schema is
    otherwise unchanged; most beats get no move and pass through untouched.

    Direction alternates A->B, B->A, A->B, ... across successive moves. Returns the
    (possibly mutated) state for convenience.
    """
    global _BEAT_COUNT, _MOVE_TOGGLE
    _BEAT_COUNT += 1
    if _BEAT_COUNT % MOVE_EVERY != 0:
        return state  # an ordinary two-portrait beat -- no move

    frm, to = ("A", "B") if _MOVE_TOGGLE % 2 == 0 else ("B", "A")
    _MOVE_TOGGLE += 1
    mover = PANEL_CHAR[frm]  # the resident of the from-panel is the one who walks

    panels = state.get("panels", {})
    # Both panels name the mover through the walk (crossframe note); preserve the
    # generated asides, override only char + the action verb to match the motion.
    if frm in panels and to in panels:
        panels[frm]["char"] = mover
        panels[to]["char"] = mover
        panels[frm]["action"] = "leaving"
        panels[to]["action"] = "arriving"
    state["move"] = {"char": mover, "from": frm, "to": to}
    return state


def write_state(state):
    DATA.mkdir(exist_ok=True)
    tmp = DATA / "stage_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(DATA / "stage_state.json")  # atomic: player never reads a half-written file


def run_once():
    sig = signals.gather()
    beat = parse_beat(call_ollama(build_system(), build_user(sig)))
    a = beat.get("A", {})
    b = beat.get("B", {})
    state = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "scene": beat.get("scene", ""),
        "panels": {
            "A": {"char": "Phineas", "action": clamp(a.get("action", "idle")), "aside": (a.get("aside") or "").strip()},
            "B": {"char": "Seraphina", "action": clamp(b.get("action", "idle")), "aside": (b.get("aside") or "").strip()},
        },
        "bridge": beat.get("bridge"),
    }
    maybe_move(state)  # occasionally adds a top-level `move` (cross-frame walk)
    write_state(state)
    print(json.dumps(state, indent=2, ensure_ascii=True), flush=True)


def main():
    # Self-log so we can run under pythonw.exe (no console window).
    sys.stdout = sys.stderr = open(ROOT / "director.log", "a", buffering=1, encoding="utf-8", errors="replace")
    loop = 0
    argv = sys.argv[1:]
    if "--loop" in argv:
        idx = argv.index("--loop")
        loop = int(argv[idx + 1]) if idx + 1 < len(argv) else 45
    if loop:
        print("stage-manager loop: a beat every", loop, "s", flush=True)
        while True:
            try:
                run_once()
            except Exception:
                import traceback
                traceback.print_exc()
            time.sleep(loop)
    else:
        run_once()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
