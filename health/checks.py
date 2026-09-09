#!/usr/bin/env python3
"""health/checks.py -- the deterministic detectors for a running Living Portraits install.

Every check here exists because something went wrong and a HUMAN had to notice. That is the
whole ratchet: an incident nobody had a check for becomes a check, so it can only cost you
once. Written 2026-08-10, when one afternoon's audit found that production had been running
for months without one of its two character specs, that a character had been looping three
clips for an hour, and that one of its poses had been unreachable for its entire 76-day life.

CONTRACT (this is what makes them detectors and not opinions):
  * Each prints exactly one verdict line "CHECK <id>: PASS|FAIL", plus evidence lines.
  * DETERMINISTIC -- the same world gives the same verdict twice in a row. Enforced by
    `python main.py verify probe health/oracle.yaml`, which runs everything twice and calls a
    flapping check invalid. So thresholds are coarse and counts are read over fixed windows.
  * A check NEVER repairs anything. Detection and repair are separate so that a repair can
    never quietly redefine health -- scripts/maintain.py owns repair, under policy.
  * Unreachable host is a FAIL with its own reason, not a crash and not a pass. A checker
    that cannot see the thing it checks must never report green.

    python health/checks.py panels_alive
    python health/checks.py --all
"""
from __future__ import annotations

import json
import subprocess
import sys

HOST = "hil"
REMOTE = "C:/living-portraits"
PY = REMOTE + "/.venv/Scripts/python.exe"

STUCK_PICKS = 40        # consecutive picks at ONE pose before we call it stuck. Phineas sat
                        # at commanding_aether for 51 and it read as broken on the wall.
LOG_STALE_S = 300       # the walker prints a line every few seconds; 5 min of silence is down
ERR_WINDOW = 400        # heartbeat lines to scan
ERR_MAX = 40            # >10% of recent ticks erroring is a real degradation, not noise


def _remote_json(snippet):
    """Run a python snippet on the host, get JSON back. Any failure -> (None, reason).

    The snippet is base64'd because it travels through local shell -> ssh -> the host's
    cmd.exe -> python, and newlines and quotes do not survive that intact. The first draft
    passed the source verbatim and every multi-line check silently returned nothing --
    which the checks correctly reported as "cannot see the host", but for the wrong reason.
    Same lesson as the repo's Invoke-RemotePS helper, which exists for exactly this."""
    import base64
    blob = base64.b64encode(snippet.encode("utf-8")).decode("ascii")
    cmd = "%s -c \"import base64;exec(base64.b64decode('%s').decode('utf-8'))\"" % (PY, blob)
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", HOST, cmd],
                           capture_output=True, text=True, timeout=120, check=False)
    except Exception as e:
        return None, "host unreachable: %r" % (e,)
    if r.returncode != 0:
        return None, "remote failed rc=%s %s" % (r.returncode, (r.stderr or "").strip()[:200])
    for line in reversed((r.stdout or "").strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line), None
            except Exception:
                continue
    return None, "no JSON in remote output: %s" % (r.stdout or "")[:200]


def _verdict(cid, ok, evidence):
    for e in evidence:
        print("    %s" % e)
    print("CHECK %s: %s" % (cid, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- detectors
def _task_enabled(task):
    """Is a scheduled task ENABLED? A Disabled lp-* task is this system's deliberate OFF
    switch -- stop_portraits.ps1 disables lp-preview + lp-mind, and the watchdog skips a
    Disabled task on purpose so a deliberate stop stays stopped. Returns True/False, or
    None when we cannot tell (never guess ON)."""
    r = _remote_json(
        "import json,subprocess;"
        "o=subprocess.run(['powershell','-NoProfile','-Command',"
        "'(Get-ScheduledTask -TaskName %s).Settings.Enabled'],capture_output=True,text=True);"
        "print(json.dumps({'enabled': 'True' in (o.stdout or '')}))" % task)[0]
    return None if r is None else bool(r.get("enabled"))


def panels_alive():
    """The panels are painting, OR they are deliberately switched off.

    Process liveness is not the question -- the player has clean-exited on a fullscreen game
    before, and a live process printing nothing is a dark wall. But neither is a dark wall
    automatically a fault: hil is also a gaming box, `stop_portraits.ps1` exists precisely so
    a human can turn the show off, and it works by DISABLING the task. Reporting that as a
    failure would page someone every two hours about a decision they made on purpose --
    which is how a monitor teaches you to ignore it. Measured 2026-08-10 19:55: the Stop
    button was pressed (LastTaskResult 267014, user-terminated) and this check cried wolf all
    night. Off-by-choice is now a PASS that says so."""
    enabled = _task_enabled("lp-preview")
    if enabled is False:
        return _verdict("panels_alive", True,
                        ["lp-preview is DISABLED -- the show is off deliberately "
                         "(stop_portraits.ps1 / the Stop shortcut), not broken.",
                         "start it with start_portraits.ps1 on the console, or "
                         "Enable-ScheduledTask lp-preview,lp-mind"])
    d, err = _remote_json(
        "import os,time,json;"
        "p=r'%s/_preview.log';"
        "print(json.dumps({'age': time.time()-os.path.getmtime(p) if os.path.exists(p) else -1}))"
        % REMOTE)
    if d is None:
        return _verdict("panels_alive", False, ["could not reach the host: %s" % err])
    age = d["age"]
    if age < 0:
        return _verdict("panels_alive", False, ["_preview.log does not exist"])
    note = [] if enabled else ["(could not read lp-preview's enabled state -- assuming ON)"]
    return _verdict("panels_alive", age <= LOG_STALE_S,
                    ["walker last printed %.0fs ago (limit %ds)" % (age, LOG_STALE_S), *note])


def walker_moving():
    """Nobody is stuck in a cul-de-sac. A character can legitimately dwell, but a long
    unbroken run at ONE pose is the shape of the bug that made Phineas look broken: a pose
    with one exit, an unreachable goal, and a mood that suppresses transitions."""
    d, err = _remote_json(
        "import re,json,collections;"
        "runs=collections.defaultdict(int);last={};"
        "f=open(r'%s/_preview.log',encoding='utf-8',errors='replace');"
        "lines=f.readlines()[-3000:];f.close();"
        "[ (lambda ch,nd: (runs.__setitem__(ch, runs[ch]+1 if last.get(ch)==nd else 1), last.__setitem__(ch,nd)))"
        "  (m.group(1), m.group(2)) for m in (re.match(r'\\[(\\w+)\\] \\S+ @ (\\S+)', l) for l in lines) if m ];"
        "print(json.dumps({'runs':dict(runs),'at':last}))" % REMOTE)
    if d is None:
        return _verdict("walker_moving", False, ["could not reach the host: %s" % err])
    bad = {c: n for c, n in d["runs"].items() if n >= STUCK_PICKS}
    ev = ["%s: %d consecutive picks at %s" % (c, n, d["at"].get(c, "?"))
          for c, n in sorted(d["runs"].items())]
    return _verdict("walker_moving", not bad, ev)


def goal_reachable():
    """Every goal the brain is holding is reachable by the body AT THIS HOUR. The daytime
    walk masks the bedtime chain, so a goal routed through it can never arrive -- that pinned
    Phineas for an hour and left one pose unvisited for its whole 76-day life. This is the
    regression guard for the fix: the menu and the walk must read the same graph."""
    # NOTE: built with .replace, never .format/% -- the snippet is full of dict and set
    # literals, and brace-escaping them by hand is how the first draft of this file crashed.
    d, err = _remote_json((
        "import json,sys,os,datetime;sys.path.insert(0,r'__ROOT__');os.chdir(r'__ROOT__');"
        "from runtime import video_graph,pathfind,circadian;"
        "g=video_graph.VideoGraph.load();"
        "spec=json.load(open('prompts/bedtime_routine.json',encoding='utf-8'));"
        "cs=json.load(open('data/mind/intent.json',encoding='utf-8')).get('characters',{});"
        "h=datetime.datetime.now().hour;out={};"
        "\n"
        "for c,v in cs.items():\n"
        "    goal=v.get('goal')\n"
        "    pose=json.load(open('data/mind/pose/'+c+'.json',encoding='utf-8')).get('node')\n"
        "    ed=g.edges if circadian.is_night(spec,c,h) else [e for e in g.edges if e.get('label') not in circadian.bedtime_labels(spec,c)]\n"
        "    out[c]={'goal':goal,'at':pose,'ok':bool(goal) and (goal==pose or goal in pathfind.reachable_poses(ed,pose))}\n"
        "print(json.dumps(out))").replace("__ROOT__", REMOTE))
    if d is None:
        return _verdict("goal_reachable", False, ["could not reach the host: %s" % err])
    bad = [c for c, v in d.items() if not v.get("ok")]
    ev = ["%s at %s wants %s -> %s" % (c, (v.get("at") or "?").split(":")[-1],
                                       (v.get("goal") or "(none)").split(":")[-1],
                                       "reachable" if v.get("ok") else "UNREACHABLE at this hour")
          for c, v in sorted(d.items())]
    return _verdict("goal_reachable", not bad, ev)


def lived_integrity():
    """The lived record keeps what it did not write, and never goes backwards. The walker's
    flush silently erased the 76-day backfill's provenance once; counts running backwards
    would mean a record was replaced by a fresh one and a life was lost."""
    d, err = _remote_json((
        "import json,glob,os;out={};"
        "\n"
        "for p in glob.glob(r'__ROOT__/data/mind/lived/*.json'):\n"
        "    try:\n"
        "        d=json.load(open(p,encoding='utf-8'))\n"
        "    except Exception:\n"
        "        out[os.path.basename(p)]={'readable':False}\n"
        "        continue\n"
        "    n=d.get('nodes') or {}\n"
        "    out[os.path.basename(p)]={'readable':True,'backfill':'backfill' in d,"
        "'poses':len(n),'visits':sum(int(v.get('visits',0)) for v in n.values())}\n"
        "print(json.dumps(out))").replace("__ROOT__", REMOTE))
    if d is None:
        return _verdict("lived_integrity", False, ["could not reach the host: %s" % err])
    if not d:
        return _verdict("lived_integrity", False, ["no lived records on the host at all"])
    bad = [f for f, v in d.items() if not v.get("readable") or not v.get("backfill")]
    ev = ["%s: %s, %s poses, %s arrivals, provenance=%s"
          % (f, "readable" if v.get("readable") else "UNREADABLE", v.get("poses"),
             v.get("visits"), v.get("backfill")) for f, v in sorted(d.items())]
    return _verdict("lived_integrity", not bad, ev)


def deploy_drift():
    """Production matches the repo, and nobody hand-edited the box. maxx.json was missing
    from production for months and nothing could see it."""
    try:
        r = subprocess.run([sys.executable, "scripts/deploy_hil.py", "--status"],
                           capture_output=True, text=True, timeout=180,
                           cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent), check=False)
    except Exception as e:
        return _verdict("deploy_drift", False, ["could not run the deploy status: %r" % e])
    out = r.stdout or ""
    dirty = "production matches its last deploy" not in out
    ev = [l.strip() for l in out.split("\n") if l.strip() and not l.startswith("--")][:6]
    return _verdict("deploy_drift", not dirty, ev or ["no output from deploy status"])


def error_rate():
    """The brain is not quietly failing. 165 gateway 401s and 62 unparseable replies had
    accumulated unnoticed, each one a tick where a character stood still."""
    d, err = _remote_json(
        "import json;"
        "ls=open(r'%s/_heartbeat.log',encoding='utf-8',errors='replace').readlines()[-%d:];"
        "print(json.dumps({'errors':sum(1 for l in ls if 'LLM error' in l or 'Traceback' in l),"
        "'lines':len(ls)}))" % (REMOTE, ERR_WINDOW))
    if d is None:
        return _verdict("error_rate", False, ["could not reach the host: %s" % err])
    return _verdict("error_rate", d["errors"] <= ERR_MAX,
                    ["%d errors in the last %d heartbeat lines (limit %d)"
                     % (d["errors"], d["lines"], ERR_MAX)])


def world_context():
    """The one real-world input these portraits have is actually arriving.

    `director/context.py` is fail-soft BY CONTRACT -- "a dead source just omits the clause;
    it can NEVER break a heartbeat tick." That is correct design and it is also why nobody
    noticed the IC context token going 401 on 2026-07-14: for 27 days the characters were
    told nothing about the weather, the hour outside, or the building they hang in, and
    every tick looked perfectly healthy. Fail-soft without a detector is just silence."""
    d, err = _remote_json((
        "import json,os,sys,time;sys.path.insert(0,r'__ROOT__');os.chdir(r'__ROOT__')\n"
        "from director import context as c\n"
        "out={'token':bool(c.resolve_context_token()),'phrase':'','age':-1}\n"
        "try:\n"
        "    p='data/mind/context.json'\n"
        "    if os.path.exists(p):\n"
        "        out['age']=time.time()-os.path.getmtime(p)\n"
        "except Exception: pass\n"
        "try:\n"
        "    out['phrase']=c.context_line() or ''\n"
        "except Exception as e:\n"
        "    out['phrase']=''\n"
        "print(json.dumps(out))").replace("__ROOT__", REMOTE))
    if d is None:
        return _verdict("world_context", False, ["could not reach the host: %s" % err])
    ev = ["token present: %s" % d["token"],
          "cache age: %s" % ("%.1f days" % (d["age"] / 86400.0) if d["age"] >= 0 else "no cache"),
          "phrase the characters would be told: %r" % (d["phrase"] or "(nothing)")]
    return _verdict("world_context", bool(d["phrase"]), ev)


def reflection_fired():
    """A reflection was written last night. Reflection runs once per character per night in
    a window nobody watches, so if it silently stopped -- a dead brain, a changed sleep
    window, an idempotency key that never clears -- the only symptom is a character that
    slowly stops having opinions. Checked over 48h so one skipped night does not page."""
    # Same off-switch rule as panels_alive: no brain, no reflection, and that is a choice
    # rather than a fault. A character cannot think about its day while it is switched off.
    if _task_enabled("lp-mind") is False:
        return _verdict("reflection_fired", True,
                        ["lp-mind is DISABLED -- the brain is off deliberately, so no "
                         "reflection is expected or owed."])
    d, err = _remote_json((
        "import json,glob,os,time\n"
        "cut=time.time()-172800;out={}\n"
        "for p in glob.glob(r'__ROOT__/data/mind/journal/*.jsonl'):\n"
        "    n=0\n"
        "    try:\n"
        "        for line in open(p,encoding='utf-8',errors='replace'):\n"
        "            if '\"kind\": \"reflection\"' in line or '\"kind\":\"reflection\"' in line:\n"
        "                try:\n"
        "                    e=json.loads(line)\n"
        "                except Exception:\n"
        "                    continue\n"
        "                if (e.get('ts') or 0) >= cut: n+=1\n"
        "    except Exception: pass\n"
        "    out[os.path.basename(p).split('.')[0]]=n\n"
        "print(json.dumps(out))").replace("__ROOT__", REMOTE))
    if d is None:
        return _verdict("reflection_fired", False, ["could not reach the host: %s" % err])
    ev = ["%s: %d reflection(s) in the last 48h" % (c, n) for c, n in sorted(d.items())]
    # a character with no bedtime (seraphina is off-panel) legitimately never reflects, so
    # the bar is "at least one character did", not "every one did".
    return _verdict("reflection_fired", any(n > 0 for n in d.values()), ev)


CHECKS = {
    "panels_alive": panels_alive,
    "walker_moving": walker_moving,
    "goal_reachable": goal_reachable,
    "lived_integrity": lived_integrity,
    "deploy_drift": deploy_drift,
    "error_rate": error_rate,
    "world_context": world_context,
    "reflection_fired": reflection_fired,
}


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("usage: checks.py {%s|--all}" % "|".join(CHECKS))
        return 2
    if args[0] == "--all":
        rc = 0
        for _cid, fn in CHECKS.items():
            rc |= fn()
        return rc
    fn = CHECKS.get(args[0])
    if not fn:
        print("unknown check %r" % args[0])
        return 2
    return fn()


if __name__ == "__main__":
    sys.exit(main())
