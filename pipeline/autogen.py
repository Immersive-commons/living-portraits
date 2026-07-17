"""autogen.py -- the portraits GROW THEIR OWN POSES (Phase 2, HUMAN-GATED).

Flow:
  1. the heartbeat (director/heartbeat.py, propose-mode) writes PROPOSALS to
     data/mind/proposals.json -- a new pose the character wishes it had: a still
     /imagine prompt + a transition motion + a couple of idle motions. status=pending.
  2. a human reviews + approves:  python pipeline/autogen.py review / approve <id>
  3. this worker generates the APPROVED ones via Midjourney:
        upload hub still -> --oref identity lock
        imagine the new still  -> download to data/gen/<char>_<label>.png
        video hub->new (transition)   + its FREE reverse (gif flipped)
        video new (idle loops, --end loop)
        mp4 -> gif into data/clips/_proto/<char>_<edgelabel>_v0.gif
     then records the pose in data/mind/autogen_poses.json (which video_graph.build()
     merges atomically) and runs build. The walker hot-reloads and can now walk there.

Guardrails (every one matters for an unattended account):
  * EVERY prompt passes director.mj_safe.check first; a hard-block -> mark failed, NEVER
    submit, NEVER retry (retrying into a moderation block EXTENDS it).
  * a daily per-character budget caps spend (data/mind/gen_budget.json).
  * single-flight: one generate run at a time (a lock file).
  * on any MJ error whose text smells like moderation/block, STOP the whole run and
    record a cooldown -- do not march into more submits.

The Midjourney client + imageio are imported LAZILY inside the generate path, so this
module imports + the queue/budget/review CLI run anywhere (the dev box has no vendored
midjourney/). Atomic JSON writes throughout.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from director import mj_safe
from runtime import video_graph

MIND = ROOT / "data" / "mind"
PROPOSALS = MIND / "proposals.json"
AUTOGEN = MIND / "autogen_poses.json"
BUDGET = MIND / "gen_budget.json"
LOCK = MIND / "autogen.lock"
LOCK_MAX_HOLD_SECS = 3 * 3600     # a real run can't hold the lock this long (5 relax jobs x ~30min
                                  # worst case ~2.5h). Beyond -> stale; steal it. Closes the 2026-06-19
                                  # bug where a crashed gen left the lock and blocked generation for 3h.
GEN_EVENTS = MIND / "gen_events.jsonl"   # append-only structured per-generation telemetry


def _telemetry(event, **fields):
    """Append one structured telemetry record (gen outcome / lock steal). Best-effort --
    NEVER raises into the generation loop. Read by `autogen.py status` + observability."""
    try:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()), "event": event}
        rec.update(fields)
        GEN_EVENTS.parent.mkdir(parents=True, exist_ok=True)
        with GEN_EVENTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
COOLDOWN = MIND / "autogen_cooldown.json"
GEN_DIR = ROOT / "data" / "gen"
PROTO = ROOT / "data" / "clips" / "_proto"

DAILY_CAP = 9999              # effectively UNLIMITED (relax is unlimited on this plan; the slow relax
                              # queue + the single-flight lock self-rate-limit). Kept as a high runaway
                              # backstop only -- not a real per-character ceiling (Ray: no limits 2026-06-18).
                              # was 15. Original note: new poses per character per day. Not a hard ceiling --
                              # a sanity rail so a runaway loop can't burn the MJ plan overnight.
                              # Each pose now = 1 imagine + 4 videos (fwd + REAL reverse + 2 idles).
OREF_WEIGHT = 160             # --ow for the identity lock (keeps the face, frees wardrobe)
# --oref (omni-reference) requires MJ v7. The account default moved to v8.1, which
# rejects --oref ("`--oref` is not compatible with `--version 8.1`") -> every still
# submit failed (166 dead proposals, 2026-06-17). Pin v7 so --oref works again.
IMAGINE_VERSION = "7"
POLL_TIMEOUT = 600            # seconds to wait for one FAST MJ job
RELAX_TIMEOUT = 1800          # a RELAX still can sit in the slow queue -> give it up to 30 min
POLL_EVERY = 6
IMAGINE_MODE = "relax"        # stills: RELAX (unlimited on this plan, no Fast burn). The old "relax"
                              # 400'd "Malformed" because MJ's wire value is "relaxed"; midjourney.api
                              # now normalizes relax->relaxed (captured from MJ web UI 2026-06-18).
VIDEO_MODE = "relax"          # videos: RELAX (unlimited); same relax->relaxed normalization.
VIDEO_TIMEOUT = RELAX_TIMEOUT if VIDEO_MODE == "relax" else POLL_TIMEOUT   # relax VIDEOS also sit in
                              # the slow queue -> give them the long wait too. 2026-06-19: the default
                              # 600s video wait timed out ~1/3 of relax poses even after the still landed.
_MOD_MARKERS = ("moderation", "banned", "nsfw", "flagged", "blocked", "manual review",
                "content policy", "not allowed")


# ----------------------------------------------------------------- atomic json
def _load(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------- proposals
def load_proposals():
    return _load(PROPOSALS, {"proposals": []})


def add_proposal(p):
    """Append a proposal (status=pending). Returns its id. Dedupes on (character,label)
    for any non-terminal existing proposal so the heartbeat can't spam the same idea."""
    q = load_proposals()
    for ex in q["proposals"]:
        if ex["character"] == p["character"] and ex["label"] == p["label"] \
                and ex["status"] in ("pending", "approved", "generating", "done"):
            return ex["id"]
    pid = "%s_%s_%d" % (p["character"], p["label"], int(p.get("proposed_at", time.time())))
    p = {**p, "id": pid, "status": "pending"}
    q["proposals"].append(p)
    _save(PROPOSALS, q)
    return pid


def set_status(pid, status, **extra):
    q = load_proposals()
    for p in q["proposals"]:
        if p["id"] == pid:
            p["status"] = status
            p.update(extra)
    _save(PROPOSALS, q)


def _by_status(status):
    return [p for p in load_proposals()["proposals"] if p["status"] == status]


# ----------------------------------------------------------------- budget + cooldown
def _today():
    # date is passed in as part of the budget record's own stamp; we derive from
    # localtime here (autogen is not resume-journaled, so time.* is fine).
    return time.strftime("%Y-%m-%d", time.localtime())


def budget_left(character):
    b = _load(BUDGET, {})
    rec = b.get(character, {})
    used = rec.get("count", 0) if rec.get("date") == _today() else 0
    return max(0, DAILY_CAP - used)


def _spend(character):
    b = _load(BUDGET, {})
    rec = b.get(character, {})
    used = rec.get("count", 0) if rec.get("date") == _today() else 0
    b[character] = {"date": _today(), "count": used + 1}
    _save(BUDGET, b)


def in_cooldown():
    c = _load(COOLDOWN, {})
    until = c.get("until", 0)
    return until > time.time(), until


def _set_cooldown(seconds, reason):
    _save(COOLDOWN, {"until": time.time() + seconds, "reason": reason, "set_at": time.time()})


# ----------------------------------------------------------------- mp4 -> gif
def _mp4_to_gif(mp4_path, gif_path, size=256):
    """Read an mp4 (imageio+ffmpeg) -> square gif (PIL). Lazy heavy imports."""
    import imageio.v3 as iio
    from PIL import Image
    imgs = []
    for fr in iio.imiter(mp4_path):            # v3 video iterator -> all frames
        im = Image.fromarray(fr).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size), Image.LANCZOS)
        imgs.append(im)
    if not imgs:
        raise RuntimeError("no frames decoded from %s" % mp4_path)
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], loop=0, duration=80, disposal=2)
    return len(imgs)


def _reverse_gif(src_gif, dst_gif):
    """The free reverse edge: the forward clip played backwards = pixel-exact return."""
    from PIL import Image, ImageSequence
    im = Image.open(src_gif)
    frames = [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)]
    frames.reverse()
    frames[0].save(dst_gif, save_all=True, append_images=frames[1:], loop=0, duration=80, disposal=2)


# ----------------------------------------------------------------- MJ orchestration
def _client():
    from midjourney.api import MidjourneyAPIClient   # lazy: only present on hil
    return MidjourneyAPIClient()


def _job_id(resp):
    succ = (resp or {}).get("success") or []
    if not succ:
        raise RuntimeError("submit had no success job: %s" % str(resp)[:200])
    return succ[0]["job_id"]


def _wait(client, job_id, timeout=POLL_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.job_status([job_id]) or []
        rec = st[0] if st else {}
        status = rec.get("current_status") or rec.get("status")
        if status == "completed":
            return rec
        if status in ("failed", "moderated", "rejected"):
            raise RuntimeError("job %s -> %s" % (job_id, status))
        time.sleep(POLL_EVERY)
    raise TimeoutError("job %s not complete in %ss" % (job_id, timeout))


def _edge_labels(hub, label):
    return ("%s_to_%s" % (hub, label), "%s_to_%s" % (label, hub))


def plan(p):
    """The lint + label plan for a proposal (no network). Returns (verdict, labels)."""
    hub = p.get("hub", "anchor")
    fwd, rev = _edge_labels(hub, label=p["label"])
    idle_motions = [i.get("motion", "") for i in p.get("idles", [])]
    # extra sibling-link motions are linted in the SAME pass -- a blocked link motion must
    # fail the whole proposal, never sneak an unsafe submit through the optional second path.
    extra_motions = []
    for el in p.get("extra_links", []) or []:
        extra_motions += [el.get("motion", ""), el.get("reverse_motion", "")]
    verdict = mj_safe.check(p["still_prompt"],
                            motion=" ".join([p.get("transition_motion", ""), p.get("reverse_motion", "")]
                                            + idle_motions + extra_motions),
                            negatives=p.get("negatives"))
    return verdict, {"hub": hub, "fwd": fwd, "rev": rev}


def generate_one(p, *, dry_run=False, log=print):
    """Generate one approved proposal end-to-end. Returns True on success."""
    char, label = p["character"], p["label"]
    _t0 = time.time()
    verdict, lab = plan(p)
    if not verdict["ok"]:
        log("  REJECT %s:%s -- %s" % (char, label, "; ".join(verdict["reasons"])))
        if not dry_run:
            set_status(p["id"], "failed", fail_reason="; ".join(verdict["reasons"]))
            _telemetry("gen", char=char, label=label, outcome="rejected",
                       reason="; ".join(verdict["reasons"])[:200])
        return False
    hub, fwd, rev = lab["hub"], lab["fwd"], lab["rev"]
    g = video_graph.VideoGraph.load()
    hub_img = (g.nodes.get("%s:%s" % (char, hub)) or {}).get("image")
    if not hub_img or not (ROOT / hub_img).exists():
        log("  SKIP %s:%s -- hub still %s missing" % (char, label, hub_img))
        if not dry_run:
            set_status(p["id"], "failed", fail_reason="hub still missing")
            _telemetry("gen", char=char, label=label, outcome="skipped", reason="hub still missing")
        return False

    idles = p.get("idles", [])
    safe_neg = ", ".join(verdict["safe_negatives"])
    still_png = GEN_DIR / ("%s_%s.png" % (char, label))
    fwd_gif = PROTO / ("%s_%s_v0.gif" % (char, fwd))
    rev_gif = PROTO / ("%s_%s_v0.gif" % (char, rev))

    def _have(path):
        return Path(path).exists() and Path(path).stat().st_size > 0

    # extra sibling links (mesh the star -> web): each is a real clip-pair sibling<->new.
    # OPTIONAL + fail-safe -- a missing sibling still or a failed link never blocks the core pose.
    extra_links = p.get("extra_links", []) or []
    need_extra = [el for el in extra_links
                  if any(not _have(PROTO / ("%s_%s_v0.gif" % (char, l)))
                         for l in _edge_labels(el.get("sibling", ""), label))]

    # RESUME: only do the work whose artifact isn't already on disk from a prior run.
    need_still = not _have(still_png)
    need_fwd = not _have(fwd_gif)
    need_rev = not _have(rev_gif)
    need_idles = [i for i in idles if not _have(PROTO / ("%s_%s_v0.gif" % (char, i["id"])))]
    n_video = (1 if need_fwd else 0) + (1 if need_rev else 0) + len(need_idles) + 2 * len(need_extra)

    log("  PLAN %s:%s hub=%s need[still=%s fwd=%s rev=%s idles=%s extra=%s] imagine=%s video=%s neg=[%s]%s" % (
        char, label, hub, need_still, need_fwd, need_rev, [i["id"] for i in need_idles],
        [el.get("sibling") for el in need_extra],
        IMAGINE_MODE, VIDEO_MODE, safe_neg,
        "  (soft: %s)" % ", ".join(verdict["soft"]) if verdict["soft"] else ""))
    if dry_run:
        log("  [dry-run] would run %d imagine (%s) + %d video job(s) (%s)" % (
            1 if need_still else 0, IMAGINE_MODE, n_video, VIDEO_MODE))
        return True

    cooled, until = in_cooldown()
    if cooled:
        log("  COOLDOWN active until %s -- not submitting" % time.ctime(until))
        return False

    client = _client()
    set_status(p["id"], "generating")
    try:
        from midjourney.api import cdn_variant_url
        # hub still upload -- serves as the imagine identity oref AND the transition start/end frame
        hub_url = None
        if need_still or need_fwd or need_rev:
            hub_url = client.upload_image(str(ROOT / hub_img))["shortUrl"]
        # 1. the new still (FAST imagine). Reuse if already on disk.
        if need_still:
            prompt = p["still_prompt"].strip() + " --oref %s --ow %d --v %s" % (hub_url, OREF_WEIGHT, IMAGINE_VERSION)
            if safe_neg:
                prompt += " --no " + safe_neg
            grid = _job_id(client.submit_imagine(prompt, mode=IMAGINE_MODE))
            _wait(client, grid, timeout=RELAX_TIMEOUT)
            _download(client, cdn_variant_url(grid, 0), still_png)
            log("  still imagined -> %s" % still_png.name)
        else:
            log("  resume: reuse still %s" % still_png.name)
        # the new-pose still as a frame URL for any video (fwd end / rev start / idle loop / extra link)
        new_url = client.upload_image(str(still_png))["shortUrl"] if (
            need_fwd or need_rev or need_idles or need_extra) else None
        # 2. forward transition hub -> new (FAST)
        if need_fwd:
            tj = _job_id(client.submit_video(image_url=hub_url, end_image=new_url,
                                             motion_prompt=p.get("transition_motion", ""),
                                             loop=False, mode=VIDEO_MODE))
            _wait(client, tj, timeout=VIDEO_TIMEOUT)
            _video_to_gif(client, tj, fwd_gif)
            log("  transition %s generated" % fwd)
        else:
            log("  resume: reuse transition %s" % fwd)
        # 3. REVERSE transition new -> hub -- its OWN generated clip. A flipped forward clip runs
        #    motion/cloth/hair BACKWARDS and reads wrong, so render new->hub for real (start=new
        #    still, end=hub still, with the brain's reverse_motion or a sensible default).
        if need_rev:
            rmotion = (p.get("reverse_motion") or "").strip() or (
                "the figure moves from the %s pose back to its %s pose and settles naturally" % (label, hub))
            rj = _job_id(client.submit_video(image_url=new_url, end_image=hub_url,
                                             motion_prompt=rmotion, loop=False, mode=VIDEO_MODE))
            _wait(client, rj, timeout=VIDEO_TIMEOUT)
            _video_to_gif(client, rj, rev_gif)
            log("  reverse %s generated (real new->hub clip)" % rev)
        else:
            log("  resume: reuse reverse %s" % rev)
        # 4. idle loops on the new pose -- FAST
        for idle in need_idles:
            ij = _job_id(client.submit_video(image_url=new_url, loop=True,
                                             motion_prompt=idle.get("motion", ""), mode=VIDEO_MODE))
            _wait(client, ij, timeout=VIDEO_TIMEOUT)
            _video_to_gif(client, ij, PROTO / ("%s_%s_v0.gif" % (char, idle["id"])))
            log("  idle %s generated" % idle["id"])
        # 5. EXTRA sibling links -- the structural fix: connect the new pose to a nearby pose
        #    too, so leaves interlink and the hub-and-spoke STAR meshes into a WEB. Each is a
        #    real clip-pair (sibling->new, new->sibling). FAIL-SAFE: a missing sibling still or a
        #    failed clip is non-fatal -- we record the core pose without that link rather than
        #    losing the whole pose (the graph merge only emits link edges whose gifs landed).
        recorded_links = _generate_extra_links(client, g, char, label, new_url, extra_links, log)
        if extra_links:
            log("  web-links: %d/%d sibling-pair(s) landed (star->web)" % (len(recorded_links), len(extra_links)))
        # 6. record the ready pose + rebuild the graph (fresh process -- see _rebuild_graph)
        _record_pose(p, hub, fwd, rev, recorded_links)
        _rebuild_graph(log)
        _spend(char)
        set_status(p["id"], "done", node_image="data/gen/%s_%s.png" % (char, label))
        _telemetry("gen", char=char, label=label, outcome="done", mode=IMAGINE_MODE,
                   videos=n_video, extra_links=len(recorded_links), dur_s=round(time.time() - _t0))
        log("  DONE %s:%s -- generated + merged in %ds" % (char, label, round(time.time() - _t0)))
        return True
    except Exception as e:
        msg = str(e)
        log("  FAIL %s:%s -- %r" % (char, label, e))
        if any(m in msg.lower() for m in _MOD_MARKERS):
            _set_cooldown(6 * 3600, msg[:200])   # moderation smell -> 6h cooldown, do not retry
            log("  -> moderation smell: 6h cooldown set; NOT retrying")
        set_status(p["id"], "failed", fail_reason=msg[:300])
        _telemetry("gen", char=char, label=label, outcome="failed", mode=IMAGINE_MODE,
                   dur_s=round(time.time() - _t0), reason=msg[:200])
        return False


def _download(client, url, out_path):
    r = client._session.get(url, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError("download %s -> %s" % (url, r.status_code))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_bytes(r.content)
    os.replace(tmp, out_path)


def _video_to_gif(client, video_job_id, gif_path):
    tmp_mp4 = MIND / ("_dl_%s.mp4" % video_job_id)
    client.download_video(video_job_id, variant=0, out_path=str(tmp_mp4))
    try:
        _mp4_to_gif(str(tmp_mp4), str(gif_path))
    finally:
        try:
            tmp_mp4.unlink()
        except Exception:
            pass


def _rebuild_graph(log=print):
    """Rebuild the video graph in a FRESH process so it sees the pose just recorded this
    run. (In-process video_graph.build() derives from NODE_SPECS/EDGE_SPECS frozen at IMPORT
    time -- before this pose existed -- so it would silently omit it; matches how the bedtime
    poses build, via a fresh CLI process.) Raises on a walk-safety failure."""
    r = subprocess.run([sys.executable, str(ROOT / "runtime" / "video_graph.py"), "build"],
                       cwd=str(ROOT), capture_output=True, text=True)
    out = (r.stdout or "").strip().splitlines()
    log("  graph build: " + (out[0] if out else "(no output)"))
    if r.returncode != 0:
        raise RuntimeError("video_graph build failed: " + ((r.stderr or r.stdout or "")[:300]))


def _generate_extra_links(client, g, char, label, new_url, extra_links, log):
    """Generate the OPTIONAL sibling<->new clip-pairs (the star->web mesh). Returns the list of
    links whose clips actually landed, shaped for _record_pose. Each link failure is swallowed
    (logged) -- the core pose is already worth keeping, and the merge only emits link edges whose
    gifs exist. The link motions were already linted by plan() in the SAME mj_safe pass."""
    recorded = []
    for el in extra_links or []:
        sib = el.get("sibling")
        sib_img = (g.nodes.get("%s:%s" % (char, sib)) or {}).get("image")
        if not sib or not sib_img or not (ROOT / sib_img).exists():
            log("  SKIP extra link %s<->%s -- sibling still missing" % (sib, label))
            continue
        lfwd, lrev = _edge_labels(sib, label)
        lfwd_gif = PROTO / ("%s_%s_v0.gif" % (char, lfwd))
        lrev_gif = PROTO / ("%s_%s_v0.gif" % (char, lrev))
        need_fwd = not (lfwd_gif.exists() and lfwd_gif.stat().st_size > 0)
        need_rev = not (lrev_gif.exists() and lrev_gif.stat().st_size > 0)
        try:
            sib_url = client.upload_image(str(ROOT / sib_img))["shortUrl"] if (need_fwd or need_rev) else None
            if need_fwd:
                j = _job_id(client.submit_video(image_url=sib_url, end_image=new_url,
                                                motion_prompt=el.get("motion", ""), loop=False, mode=VIDEO_MODE))
                _wait(client, j, timeout=VIDEO_TIMEOUT)
                _video_to_gif(client, j, lfwd_gif)
                log("  extra link %s generated" % lfwd)
            if need_rev:
                rmot = (el.get("reverse_motion") or "").strip() or (
                    "the figure moves from the %s pose to its %s pose and settles naturally" % (label, sib))
                j = _job_id(client.submit_video(image_url=new_url, end_image=sib_url,
                                                motion_prompt=rmot, loop=False, mode=VIDEO_MODE))
                _wait(client, j, timeout=VIDEO_TIMEOUT)
                _video_to_gif(client, j, lrev_gif)
                log("  extra link %s generated" % lrev)
            recorded.append({"sibling": sib, "label": lfwd, "reverse_label": lrev,
                             "motion": el.get("motion", "")})
        except Exception as le:
            log("  WARN extra link %s<->%s failed (%r) -- recording core pose without it" % (sib, label, le))
    return recorded


def _record_pose(p, hub, fwd, rev, extra_links=None):
    char, label = p["character"], p["label"]
    ag = _load(AUTOGEN, {"characters": {}})
    ag.setdefault("characters", {}).setdefault(char, {}).setdefault("poses", {})[label] = {
        "node_image": "data/gen/%s_%s.png" % (char, label),
        "still_prompt": p["still_prompt"],
        "hub": hub,
        "transition": {"label": fwd, "reverse_label": rev, "motion": p.get("transition_motion", "")},
        "idles": [{"id": i["id"], "motion": i.get("motion", "")} for i in p.get("idles", [])],
        "extra_links": extra_links or [],
    }
    _save(AUTOGEN, ag)


# ----------------------------------------------------------------- run + CLI
def _pid_alive(pid):
    """Best-effort liveness of a PID (Windows via ctypes OpenProcess; POSIX via signal 0).
    On uncertainty returns True so we NEVER steal a lock we can't prove is dead."""
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return bool(ok) and code.value == 259         # STILL_ACTIVE
        except Exception:
            return True
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _lock_holder():
    """(pid, age_secs) of the current lock holder, or (None, None) if there is no lock."""
    try:
        age = time.time() - LOCK.stat().st_mtime
    except OSError:
        return None, None
    try:
        pid = int((LOCK.read_text(encoding="utf-8") or "0").strip() or 0)
    except Exception:
        pid = 0
    return pid, age


def _acquire_lock(log=lambda m: None):
    MIND.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        pid, age = _lock_holder()
        dead = bool(pid) and not _pid_alive(pid)
        too_old = age is not None and age > LOCK_MAX_HOLD_SECS
        if dead or too_old:
            why = ("dead pid %s" % pid) if dead else ("held %dmin > max" % ((age or 0) / 60))
            log("  STEALING stale autogen.lock (%s)" % why)
            _telemetry("lock_steal", stolen_pid=pid, age_min=round((age or 0) / 60, 1), why=why)
            try:
                LOCK.unlink()
                fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return True
            except Exception:
                return False
        return False


def _release_lock():
    try:
        LOCK.unlink()
    except Exception:
        pass


def run_generate(dry_run=False, limit=None, log=print, auto=False):
    # auto = AUTONOMOUS Phase 2: also generate `pending` proposals without a human approval
    # step. The safety net for unattended runs is the linter (mj_safe), the daily budget
    # (DAILY_CAP), the moderation cooldown, and the single-flight lock -- NOT human review.
    statuses = ("approved", "pending") if auto else ("approved",)
    props = load_proposals()["proposals"]
    queue = [p for p in props if p["status"] in statuses]
    orphans = [p for p in props if p.get("status") == "generating"]
    if not queue and not orphans:
        log("no %s proposals." % " / ".join(statuses))
        return
    if not dry_run and not _acquire_lock(log):
        log("another autogen run holds the lock (%s) -- skipping." % LOCK)
        return
    try:
        if not dry_run and orphans:
            # We hold the lock now -> no other gen is running, so any proposal still in
            # 'generating' was orphaned by a crashed run. Reset it so it retries -- closes the
            # stuck-generating leak (2026-06-19 telemetry caught signal_jam + supine_perish).
            for op in orphans:
                set_status(op["id"], "approved")
                log("  recovered orphaned 'generating' -> approved: %s:%s" % (op["character"], op["label"]))
                _telemetry("orphan_recover", char=op["character"], label=op["label"])
            queue = [p for p in load_proposals()["proposals"] if p["status"] in statuses]
        done = 0
        for p in queue:
            if limit and done >= limit:
                break
            if budget_left(p["character"]) <= 0:
                log("  budget exhausted for %s today (cap %d) -- skipping %s" % (
                    p["character"], DAILY_CAP, p["label"]))
                continue
            if generate_one(p, dry_run=dry_run, log=log):
                done += 1
        log("generate run complete: %d generated." % done)
    finally:
        if not dry_run:
            _release_lock()


def _fmt(p):
    return "  [%s] %s:%s  (%s)  -- %s" % (p["status"], p["character"], p["label"],
                                          p.get("reason", "")[:60], p["id"])


def _status_report():
    """One-screen health view of the autogen pose-growth pipeline: scheduled tasks, the
    generation telemetry (last success / last attempt / 24h outcomes / top failures), the
    queue, the graph, the lock state, budget + cooldown -- with STUCK/STALE alerts."""
    import collections
    now = time.time()
    out = []

    # scheduled tasks (Windows; best-effort -- silent off-box)
    tlines = []
    for t in ("lp-gen", "lp-mind", "lp-preview", "lp-watchdog-preview", "OllamaServe"):
        try:
            r = subprocess.run(["schtasks", "/query", "/TN", t, "/FO", "LIST"],
                               capture_output=True, text=True, timeout=10)
            st = next((ln.split(":", 1)[1].strip() for ln in r.stdout.splitlines()
                       if ln.strip().lower().startswith("status")), "?")
            if r.returncode == 0:
                tlines.append("  %-20s %s" % (t, st))
        except Exception:
            pass
    if tlines:
        out.append("TASKS:"); out.extend(tlines)

    # generation telemetry
    events = []
    try:
        for ln in GEN_EVENTS.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(ln))
            except Exception:
                pass
    except Exception:
        pass

    def _age(ev):
        try:
            return now - time.mktime(time.strptime(ev["ts"], "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            return None

    gens = [e for e in events if e.get("event") == "gen"]
    dones = [e for e in gens if e.get("outcome") == "done"]
    out.append("GENERATION:")
    if dones:
        ld = dones[-1]; a = _age(ld)
        out.append("  last SUCCESS: %s:%s  %s  (%s)" % (
            ld.get("char"), ld.get("label"), ld["ts"], ("%.1fh ago" % (a / 3600)) if a else "?"))
        if a and a > 6 * 3600:
            out.append("  !! ALERT: no successful gen in %.1fh" % (a / 3600))
    else:
        out.append("  last SUCCESS: (none recorded since telemetry began)")
    if gens:
        la = gens[-1]
        out.append("  last attempt: %s:%s -> %s  %s" % (
            la.get("char"), la.get("label"), la.get("outcome"), la["ts"]))
    recent = [e for e in gens if (_age(e) or 1e9) < 86400]
    if recent:
        oc = collections.Counter(e.get("outcome") for e in recent)
        out.append("  last 24h: " + ", ".join("%s=%d" % (k, v) for k, v in oc.items()))
        fails = collections.Counter((e.get("reason") or "?")[:50]
                                    for e in recent if e.get("outcome") in ("failed", "rejected"))
        for reason, n in fails.most_common(3):
            out.append("    fail x%d: %s" % (n, reason))
    steals = [e for e in events if e.get("event") == "lock_steal"]
    if steals:
        out.append("  lock-steals (lifetime): %d (last %s)" % (len(steals), steals[-1]["ts"]))

    # queue + stuck alert
    props = load_proposals()["proposals"]
    out.append("QUEUE: " + str(dict(collections.Counter(p["status"] for p in props))))
    stuck = [p for p in props if p.get("status") == "generating"]
    if stuck:
        out.append("  !! ALERT: %d proposal(s) STUCK in 'generating': %s" % (
            len(stuck), ", ".join("%s:%s" % (p["character"], p["label"]) for p in stuck)))

    # graph
    try:
        g = video_graph.VideoGraph.load()
        out.append("GRAPH: %d nodes / %d edges" % (len(g.nodes), len(g.edges)))
    except Exception:
        pass

    # lock state
    pid, age = _lock_holder()
    if age is None:
        out.append("LOCK: free")
    elif _pid_alive(pid) and age <= LOCK_MAX_HOLD_SECS:
        out.append("LOCK: HELD (gen in progress) pid=%s age=%.0fmin" % (pid, age / 60))
    else:
        out.append("LOCK: !! STALE (pid=%s dead/old, age=%.0fmin) -- auto-stolen next run" % (pid, age / 60))

    b = _load(BUDGET, {})
    out.append("BUDGET today: %s  (DAILY_CAP=%d)" % (
        {k: v.get("count") for k, v in b.items() if v.get("date") == _today()}, DAILY_CAP))
    cooled, until = in_cooldown()
    out.append("COOLDOWN: " + (time.ctime(until) if cooled else "none"))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Living-portraits autogen (human-gated pose growth)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list")
    sub.add_parser("review")
    a = sub.add_parser("approve"); a.add_argument("id")
    r = sub.add_parser("reject"); r.add_argument("id")
    g = sub.add_parser("generate"); g.add_argument("--dry-run", action="store_true"); g.add_argument("--limit", type=int)
    g.add_argument("--auto", action="store_true", help="AUTONOMOUS: also generate pending proposals (no human approval); rails = linter + daily cap + cooldown + lock")
    sub.add_parser("status")
    pr = sub.add_parser("propose")          # manual proposal (testing)
    pr.add_argument("--character", required=True); pr.add_argument("--label", required=True)
    pr.add_argument("--still", required=True); pr.add_argument("--hub", default="anchor")
    pr.add_argument("--transition", default=""); pr.add_argument("--idle", action="append", default=[])
    args = ap.parse_args()

    if args.cmd in ("list", "review"):
        for p in load_proposals()["proposals"]:
            print(_fmt(p))
    elif args.cmd == "approve":
        set_status(args.id, "approved"); print("approved", args.id)
    elif args.cmd == "reject":
        set_status(args.id, "rejected"); print("rejected", args.id)
    elif args.cmd == "generate":
        logf = open(ROOT / "_autogen.log", "a", buffering=1, encoding="utf-8", errors="replace")

        def _log(m):
            line = "%s %s" % (time.strftime("%H:%M:%S"), m)
            print(line, file=logf, flush=True)
            print(line, flush=True)

        _log("=== generate run start (dry_run=%s limit=%s auto=%s) ===" % (args.dry_run, args.limit, args.auto))
        run_generate(dry_run=args.dry_run, limit=args.limit, log=_log, auto=args.auto)
        _log("=== generate run end ===")
    elif args.cmd == "status":
        print(_status_report())
    elif args.cmd == "propose":
        idles = [{"id": "%s_%d" % (args.label, i), "motion": m} for i, m in enumerate(args.idle)]
        pid = add_proposal({"character": args.character, "label": args.label, "still_prompt": args.still,
                            "hub": args.hub, "transition_motion": args.transition, "idles": idles,
                            "reason": "manual", "proposed_at": time.time()})
        print("proposed", pid)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
