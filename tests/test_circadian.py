"""Headless test of the circadian bedtime sequencer against the REAL routine spec.

Builds the bedtime edge list the way video_graph does (from prompts/bedtime_routine.json),
adds a couple of daytime anchor idles, then simulates the walker's pick loop (the same
advance-on-transition / dwell-on-idle logic as _preview_graph.GraphCycler) and asserts:

  * NIGHT: from the awake hub the walk reaches the sleep pose and then STAYS (only sleep
    idles, never an exit) -- "they go to sleep and stay there".
  * NIGHT pacing: it dwells (plays idles) at the empty + prep beats before advancing.
  * DAY from sleep: it walks the chain in REVERSE back to the hub (the morning wake-up)
    and then never touches a bedtime edge again.

Pure logic, no pygame/network. Run: python projects/living-portraits/tests/test_circadian.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from runtime import circadian

SPEC = json.loads((ROOT / "prompts" / "bedtime_routine.json").read_text(encoding="utf-8"))


def build_edges(char):
    """Mirror video_graph._bedtime_additions edge shape + add 2 daytime hub idles."""
    cs = SPEC["characters"][char]
    hub = cs["hub_pose"]
    edges = [
        {"character": char, "kind": "idle", "label": "day_breathe", "id": f"{char}/day_breathe/v0",
         "from": f"{char}:{hub}", "to": f"{char}:{hub}"},
        {"character": char, "kind": "idle", "label": "day_look", "id": f"{char}/day_look/v0",
         "from": f"{char}:{hub}", "to": f"{char}:{hub}"},
    ]
    for beat in cs["routine"]:
        if beat["kind"] == "transition":
            for lbl, fr, to in ((beat["label"], beat["from"], beat["to"]),
                                (beat["reverse_label"], beat["to"], beat["from"])):
                edges.append({"character": char, "kind": "transition", "label": lbl,
                              "id": f"{char}/{lbl}/v0", "from": f"{char}:{fr}", "to": f"{char}:{to}"})
        else:
            for idle in beat["idles"]:
                edges.append({"character": char, "kind": "idle", "label": idle["id"],
                              "id": f"{char}/{idle['id']}/v0",
                              "from": f"{char}:{beat['at']}", "to": f"{char}:{beat['at']}"})
    return edges


def simulate(char, hour, start_pose, steps=120, dwell_cap=200):
    edges = build_edges(char)
    node = f"{char}:{start_pose}"
    pose_dwell, last_id = 0, 0
    rng = random.Random(0)
    path = []
    for _ in range(steps):
        out = [e for e in edges if e["from"] == node]
        dec = circadian.decide(SPEC, char, node, out, pose_dwell, last_id, hour, rng)
        if dec["action"] == "force":
            cur = dec["edge"]
        elif dec["action"] == "normal":
            cands = [e for e in out if e["label"] not in dec.get("exclude", set())] or out
            idles = [e for e in cands if e["kind"] == "idle" and e["id"] != last_id]
            trans = [e for e in cands if e["kind"] == "transition"]
            cur = (idles or trans or cands)[0]
        else:
            cur = out[0]
        last_id = cur["id"]
        path.append((node, cur["label"], cur["kind"]))
        if cur["kind"] == "transition":
            node = cur["to"]
            pose_dwell = 0
        else:
            pose_dwell = min(pose_dwell + 1, dwell_cap)
    return path, node


def test_phineas_night_reaches_sleep_and_stays():
    path, end = simulate("phineas", hour=23, start_pose="anchor")
    assert end == "phineas:sleep", f"did not reach sleep, ended at {end}"
    # once asleep, the tail must be ONLY sleep-pose idles (never an exit)
    sleep_idx = next(i for i, (n, _l, _k) in enumerate(path) if n == "phineas:sleep")
    tail = path[sleep_idx:]
    assert all(n == "phineas:sleep" and k == "idle" for (n, _l, k) in tail), \
        f"left the sleep pose: {[(n,l) for n,l,k in tail if n!='phineas:sleep']}"
    labels = {l for _n, l, _k in tail}
    assert labels <= {"sleep_breathe", "sleep_turn", "sleep_mutter"}, f"non-sleep idle at sleep: {labels}"
    print("  night: anchor ->", " -> ".join(l for _n, l, _k in path[:sleep_idx + 1]))


def test_phineas_night_dwells_before_advancing():
    path, _ = simulate("phineas", hour=23, start_pose="anchor")
    # must PLAY the wait/prep idles (not jump straight through transitions)
    played = {l for _n, l, k in path if k == "idle"}
    assert "nightcap_toast" in played, "skipped the nightcap toast"
    assert {"empty_flicker", "empty_moth"} & played, "skipped the empty-study wait"
    assert any(l.startswith("prep_") for l in played), "skipped bed prep"


def test_phineas_day_from_sleep_wakes_back_to_hub():
    path, end = simulate("phineas", hour=10, start_pose="sleep", steps=60)
    assert end == "phineas:anchor", f"did not wake back to anchor, ended at {end}"
    # reverse chain must be used to climb out
    used = [l for _n, l, k in path if k == "transition"]
    assert used[:3] == ["sl2n", "n2e", "e2a"], f"unexpected wake chain: {used[:3]}"
    # after waking, never touch a bedtime edge again
    bt = circadian.bedtime_labels(SPEC, "phineas")
    after = path[next(i for i, (n, _l, _k) in enumerate(path) if n == "phineas:anchor"):]
    assert all(l not in bt for _n, l, _k in after), \
        f"used a bedtime edge by day: {[l for _n,l,_k in after if l in bt]}"
    print("  day-from-sleep wake chain:", " -> ".join(used[:3]), "-> (daytime idle)")


def test_in_window_wraps_midnight():
    assert circadian.in_window(23, 21, 5) and circadian.in_window(2, 21, 5)
    assert not circadian.in_window(12, 21, 5) and not circadian.in_window(5, 21, 5)
    assert circadian.in_window(2, 0, 5) and not circadian.in_window(23, 0, 5)   # maxx: 0->5


if __name__ == "__main__":
    test_in_window_wraps_midnight()
    test_phineas_night_reaches_sleep_and_stays()
    test_phineas_night_dwells_before_advancing()
    test_phineas_day_from_sleep_wakes_back_to_hub()
    print("OK -- circadian sequencer: night->sleep+dwell, stays asleep, day->wake-chain->hub")
