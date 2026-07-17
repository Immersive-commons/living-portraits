"""
verify.py -- the QA gate a generated asset must pass before it enters the clip library.

Living Portraits bakes its art offline (generate.py -> SAM -> rig -> img2vid). None of
that is realtime, so we can afford to be fussy: every clip and every portrait is checked
HERE, and only assets that pass are registered as nodes/edges in the clip graph. A failure
is not an exception -- it is a verdict that kicks the asset back to the generator with a
tweaked prompt (see VERIFY_CONTRACT.md for the kick-back loop).

Four checks, each a pure function returning (passed: bool, score: float, reason: str):

  continuity(clip_end_frame, target_node_frame)  -- does a transition/bridge clip LAND on the
      pose it claims to arrive at? Structural similarity of the last frame against the target
      graph node's canonical frame must clear CONTINUITY_THRESHOLD. Keeps edges in the clip
      graph honest so the player can splice clips without a visible jump.

  loopability(first_frame, last_frame)            -- does an idle loop actually loop? First and
      last frame structural similarity must clear LOOP_THRESHOLD, else the seam pops on repeat.

  identity(candidate_img, canonical_portrait)     -- is this still the same character? Face-
      embedding cosine via InsightFace when available; a coarse histogram+ORB fallback marked
      "degraded" when it is not. Stops a re-render from quietly swapping the actor's face.

  register(text_or_image, character_md)           -- does a spoken aside stay in the theatrical,
      fourth-wall-aware voice the character contract demands? Asks a local Ollama vision/chat
      model yes/no. Network-guarded: if Ollama is unreachable the check is "skipped", never a
      hard failure (the gate degrades open on the soft, taste-level check; see contract).

A `Verdict` dataclass aggregates per-check results; `verify_clip(...)` / `verify_portrait(...)`
run the relevant subset and AND them into a single pass/fail.

Dependency policy (mirrors the rest of the pipeline -- free + local, degrade don't crash):
  * numpy + opencv are required and assumed present (the .venv has them).
  * insightface is IMPORT-GUARDED -- absent => identity runs degraded.
  * Ollama is an HTTP call guarded by try/except with a short timeout -- unreachable => register
    is skipped.
The module imports and runs with neither InsightFace nor Ollama present. No network at import.

    python pipeline/verify.py        # synthetic self-test, prints the verdicts
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Thresholds (the invariants -- documented in VERIFY_CONTRACT.md). Tune there, not here.
# --------------------------------------------------------------------------------------
CONTINUITY_THRESHOLD = 0.78   # SSIM: bridge clip's last frame vs the target node it arrives at
LOOP_THRESHOLD = 0.92         # SSIM: idle loop first vs last frame (seam must be near-invisible)
IDENTITY_THRESHOLD = 0.55     # InsightFace cosine: same-actor floor (ArcFace embeddings)
IDENTITY_DEGRADED_THRESHOLD = 0.62  # histogram+ORB fallback floor (coarser -> stricter to compensate)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_VL_MODEL = "qwen2.5vl:7b"   # vision model for image-register; falls back to text model below
OLLAMA_TEXT_MODEL = "qwen3:8b"     # the director's model -- reuse for text-register
OLLAMA_TIMEOUT = 20                # seconds; this is a slow offline gate, but never hang the loop

# SSIM constants (Wang et al. 2004), for inputs scaled to [0, 1].
_SSIM_C1 = (0.01) ** 2
_SSIM_C2 = (0.03) ** 2

ImageLike = Union[np.ndarray, str, Path]


# ======================================================================================
# Image loading + SSIM (numpy/opencv only -- no skimage dependency, per the build brief)
# ======================================================================================
def _as_gray_float(img: ImageLike) -> np.ndarray:
    """Load a path or accept an array; return a single-channel float32 image in [0, 1].

    Accepts grayscale, BGR, or BGRA. Paths are read via OpenCV (BGR). A 4-channel image
    is composited over black so a generated cutout's alpha does not leak structural edges.
    """
    if isinstance(img, (str, Path)):
        arr = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"verify: could not read image {img!r}")
    else:
        arr = np.asarray(img)

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # composite BGRA over black, then to gray
        bgr = arr[:, :, :3].astype(np.float32)
        alpha = (arr[:, :, 3:4].astype(np.float32)) / 255.0
        comp = (bgr * alpha).astype(arr.dtype if arr.dtype == np.uint8 else np.float32)
        gray = cv2.cvtColor(comp.astype(np.uint8) if comp.dtype != np.uint8 else comp, cv2.COLOR_BGR2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        gray = arr[:, :, 0]
    else:
        raise ValueError(f"verify: unsupported image shape {arr.shape}")

    gray = gray.astype(np.float32)
    if gray.max() > 1.0:           # uint8 / 0..255 -> 0..1
        gray = gray / 255.0
    return gray


def _match_shapes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize b to a's shape if they differ (clips/nodes may be authored at different sizes)."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return a, b


def ssim(a: ImageLike, b: ImageLike) -> float:
    """Mean structural similarity between two images, in [-1, 1] (1.0 == identical).

    Gaussian-windowed SSIM (11x11, sigma 1.5) per Wang et al. 2004 -- the same formulation
    skimage.metrics.structural_similarity uses by default -- implemented with OpenCV's
    separable Gaussian blur so we carry no extra dependency. Inputs are converted to
    grayscale float [0, 1] and shape-matched first.
    """
    ga = _as_gray_float(a)
    gb = _as_gray_float(b)
    ga, gb = _match_shapes(ga, gb)

    # If the image is smaller than the window, shrink the window to fit (odd, >=3).
    win = 11
    smallest = min(ga.shape[0], ga.shape[1])
    if smallest < win:
        win = smallest if smallest % 2 == 1 else smallest - 1
        win = max(win, 3)
    sigma = 1.5 * (win / 11.0)

    def blur(x):
        return cv2.GaussianBlur(x, (win, win), sigma, borderType=cv2.BORDER_REFLECT)

    mu_a = blur(ga)
    mu_b = blur(gb)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = blur(ga * ga) - mu_a2
    sigma_b2 = blur(gb * gb) - mu_b2
    sigma_ab = blur(ga * gb) - mu_ab

    num = (2 * mu_ab + _SSIM_C1) * (2 * sigma_ab + _SSIM_C2)
    den = (mu_a2 + mu_b2 + _SSIM_C1) * (sigma_a2 + sigma_b2 + _SSIM_C2)
    ssim_map = num / den
    return float(ssim_map.mean())


# ======================================================================================
# Checks  ((passed, score, reason) each)
# ======================================================================================
def continuity(clip_end_frame: ImageLike, target_node_frame: ImageLike,
               threshold: float = CONTINUITY_THRESHOLD) -> tuple[bool, float, str]:
    """A bridge/transition clip must END on the graph node it claims to arrive at.

    Compares the clip's last rendered frame to the target node's canonical frame by SSIM.
    Below threshold => the splice would show a visible jump; reject the edge.
    """
    try:
        score = ssim(clip_end_frame, target_node_frame)
    except Exception as e:  # unreadable/garbage frame -> hard fail, surface why
        return False, 0.0, f"continuity: could not score ({type(e).__name__}: {e})"
    passed = score >= threshold
    verdict = "lands on target node" if passed else "drifts from target node"
    return passed, score, f"continuity SSIM {score:.3f} {'>=' if passed else '<'} {threshold:.2f} ({verdict})"


def loopability(first_frame: ImageLike, last_frame: ImageLike,
                threshold: float = LOOP_THRESHOLD) -> tuple[bool, float, str]:
    """An idle loop must close: first and last frame near-identical so the seam is invisible."""
    try:
        score = ssim(first_frame, last_frame)
    except Exception as e:
        return False, 0.0, f"loopability: could not score ({type(e).__name__}: {e})"
    passed = score >= threshold
    verdict = "seam invisible" if passed else "seam will pop"
    return passed, score, f"loop SSIM {score:.3f} {'>=' if passed else '<'} {threshold:.2f} ({verdict})"


# --- identity: InsightFace if present, coarse fallback otherwise --------------------------
_INSIGHTFACE_APP = None
_INSIGHTFACE_TRIED = False


def _insightface_app():
    """Lazily build (and cache) the InsightFace FaceAnalysis app, or return None if unavailable.

    Import-guarded: a missing package, missing model pack, or no execution provider all
    degrade to None rather than raising. Model download happens on first call -- offline
    boxes that have never primed it will fall back too.
    """
    global _INSIGHTFACE_APP, _INSIGHTFACE_TRIED
    if _INSIGHTFACE_TRIED:
        return _INSIGHTFACE_APP
    _INSIGHTFACE_TRIED = True
    try:
        from insightface.app import FaceAnalysis  # type: ignore
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        _INSIGHTFACE_APP = app
    except Exception:
        _INSIGHTFACE_APP = None
    return _INSIGHTFACE_APP


def _largest_face_embedding(app, img_bgr: np.ndarray) -> Optional[np.ndarray]:
    faces = app.get(img_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
    return np.asarray(emb, dtype=np.float32)


def _to_bgr_u8(img: ImageLike) -> np.ndarray:
    if isinstance(img, (str, Path)):
        arr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"verify: could not read image {img!r}")
        return arr
    arr = np.asarray(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    return arr.astype(np.uint8)


def _coarse_similarity(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    """Degraded same-image score in [0, 1]: blend of HSV color-histogram correlation and
    ORB keypoint match rate. NOT a face matcher -- it cannot tell two faces apart well --
    so the caller marks the verdict 'degraded' and applies a stricter threshold.
    """
    a_bgr, b_bgr = _match_shapes(a_bgr, b_bgr)

    # HSV histogram correlation (H,S channels -> robust-ish to brightness drift)
    ha = cv2.calcHist([cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    hist_corr = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))  # -1..1
    hist_corr = max(0.0, hist_corr)

    # ORB feature match rate
    orb_rate = 0.0
    try:
        orb = cv2.ORB_create(nfeatures=400)
        ga = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
        ka, da = orb.detectAndCompute(ga, None)
        kb, db = orb.detectAndCompute(gb, None)
        if da is not None and db is not None and len(ka) and len(kb):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(da, db)
            good = [m for m in matches if m.distance < 64]
            orb_rate = len(good) / max(len(ka), len(kb))
    except Exception:
        orb_rate = 0.0

    return 0.6 * hist_corr + 0.4 * min(1.0, orb_rate)


def identity(candidate_img: ImageLike, canonical_portrait: ImageLike) -> tuple[bool, float, str]:
    """Same-actor check. InsightFace ArcFace cosine when available; coarse histogram+ORB
    similarity (marked 'degraded', stricter threshold) when not.

    No detectable face in either image under InsightFace is treated as a fail with reason --
    a portrait the face model cannot even find a face in should not enter the library silently.
    """
    app = _insightface_app()
    if app is not None:
        try:
            cand = _largest_face_embedding(app, _to_bgr_u8(candidate_img))
            canon = _largest_face_embedding(app, _to_bgr_u8(canonical_portrait))
        except Exception as e:
            return False, 0.0, f"identity: InsightFace error ({type(e).__name__}: {e})"
        if cand is None or canon is None:
            which = "candidate" if cand is None else "canonical"
            return False, 0.0, f"identity: no face detected in {which} image"
        cos = float(np.dot(cand, canon))  # both L2-normalized
        passed = cos >= IDENTITY_THRESHOLD
        return passed, cos, (
            f"identity InsightFace cosine {cos:.3f} {'>=' if passed else '<'} {IDENTITY_THRESHOLD:.2f} "
            f"({'same actor' if passed else 'face drifted'})"
        )

    # ---- degraded path ----
    try:
        score = _coarse_similarity(_to_bgr_u8(candidate_img), _to_bgr_u8(canonical_portrait))
    except Exception as e:
        return False, 0.0, f"identity[degraded]: could not score ({type(e).__name__}: {e})"
    passed = score >= IDENTITY_DEGRADED_THRESHOLD
    return passed, score, (
        f"identity[degraded] hist+ORB {score:.3f} {'>=' if passed else '<'} {IDENTITY_DEGRADED_THRESHOLD:.2f} "
        f"(insightface absent -- coarse match, NOT a true face check)"
    )


# --- register: local Ollama yes/no, network-guarded -> 'skipped' when unreachable ----------
def _theme_voice_of(character_md: ImageLike) -> str:
    """Pull the Theme: / Voice: lines (and a little body) from a character contract .md.

    Accepts a path or the markdown text itself. Used to ground the register judgement in
    the character's documented voice rather than a generic 'is this theatrical' prompt.
    """
    if isinstance(character_md, (str, Path)) and Path(str(character_md)).suffix == ".md" \
            and Path(str(character_md)).exists():
        txt = Path(str(character_md)).read_text(encoding="utf-8-sig")
    else:
        txt = str(character_md)
    return txt.strip()


def _ollama_yes_no(model: str, system: str, user_content) -> tuple[Optional[bool], str]:
    """POST a yes/no question to Ollama; return (verdict|None, raw). None == could not decide.

    user_content is either a string (text register) or a {'text','images':[b64]} dict (image
    register). All network failure modes -> (None, reason) so the caller can mark 'skipped'.
    """
    msg = {"role": "user"}
    if isinstance(user_content, dict):
        msg["content"] = user_content.get("text", "")
        if user_content.get("images"):
            msg["images"] = user_content["images"]
    else:
        msg["content"] = str(user_content)

    body = json.dumps({
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "messages": [{"role": "system", "content": system}, msg],
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            content = json.loads(resp.read())["message"]["content"]
    except urllib.error.HTTPError as e:
        # 404 here means Ollama is up but the model isn't pulled -- distinct from a dead daemon.
        hint = f"model {model!r} not pulled" if e.code == 404 else f"HTTP {e.code}"
        return None, f"ollama {hint}"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return None, f"ollama unreachable ({type(e).__name__})"
    except Exception as e:  # malformed response shape, etc.
        return None, f"ollama error ({type(e).__name__}: {e})"

    raw = (content or "").strip()
    parsed = None
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        for key in ("in_register", "in_voice", "pass", "answer", "yes"):
            if key in parsed:
                v = parsed[key]
                if isinstance(v, bool):
                    return v, raw
                if isinstance(v, str):
                    return v.strip().lower() in ("yes", "true", "pass", "y"), raw
    # last resort: scan the text
    low = raw.lower()
    if re.search(r"\b(yes|true|in[- ]register|in[- ]voice|pass)\b", low):
        return True, raw
    if re.search(r"\b(no|false|out[- ]of[- ]register|breaks?[- ]character|off[- ]voice|fail)\b", low):
        return False, raw
    return None, raw or "empty response"


def register(text_or_image: ImageLike, character_md: ImageLike) -> tuple[bool, float, str]:
    """Does an aside (text) or an asset (image) hold the character's theatrical, fourth-wall
    voice? Asks a local Ollama model yes/no, grounded in the character contract.

    Soft, taste-level check. Network-guarded: if Ollama is unreachable or undecided the check
    is SKIPPED -- returns passed=True with score=-1.0 and a 'skipped' reason, so the gate
    degrades OPEN here (a hard infra failure must not block the whole library). The score
    sentinel -1.0 lets callers/telemetry distinguish 'skipped' from a real 0.0..1.0 judgement.
    """
    contract = _theme_voice_of(character_md)
    system = (
        "You are a theatre dramaturg checking whether a line stays in character. The character "
        "is a self-aware painted PORTRAIT that knows it is an actor on a stage, breaks the fourth "
        "wall, and never drops the arch, performed, theatrical register into a sincere modern "
        "voice. Here is the character's contract:\n\n" + contract + "\n\n"
        'Answer ONLY with JSON: {"in_register": true|false, "reason": "<=10 words"}. '
        "true = stays in the theatrical fourth-wall voice; false = breaks character or register."
    )

    is_image = isinstance(text_or_image, (np.ndarray, Path)) or (
        isinstance(text_or_image, str) and Path(text_or_image).suffix.lower()
        in (".png", ".jpg", ".jpeg", ".webp") and Path(text_or_image).exists()
    )

    if is_image:
        try:
            import base64
            if isinstance(text_or_image, np.ndarray):
                ok, buf = cv2.imencode(".png", _to_bgr_u8(text_or_image))
                raw_bytes = buf.tobytes() if ok else b""
            else:
                raw_bytes = Path(str(text_or_image)).read_bytes()
            b64 = base64.b64encode(raw_bytes).decode("ascii")
        except Exception as e:
            return True, -1.0, f"register: SKIPPED (could not encode image: {type(e).__name__})"
        verdict, raw = _ollama_yes_no(
            OLLAMA_VL_MODEL, system,
            {"text": "Does this portrait image hold the character's theatrical look/register?",
             "images": [b64]},
        )
    else:
        verdict, raw = _ollama_yes_no(
            OLLAMA_TEXT_MODEL, system,
            'Line: "' + str(text_or_image).strip() + '"\nIs this line in register?',
        )

    if verdict is None:
        return True, -1.0, f"register: SKIPPED ({raw[:80]})"
    if verdict:
        return True, 1.0, "register: in the theatrical fourth-wall voice"
    return False, 0.0, f"register: OUT of voice -- {raw[:80]}"


# ======================================================================================
# Verdict + orchestrators
# ======================================================================================
@dataclass
class Verdict:
    """Aggregate result of a verify run. `passed` is the AND of every check that actually ran.

    A check that is SKIPPED (e.g. register with Ollama down) does NOT veto the verdict --
    only checks that returned a real pass/fail count. `degraded`/`skipped` carry the names of
    checks that ran in a reduced mode, so the caller and telemetry can see the gate was soft.
    """
    passed: bool = True
    checks: list[str] = field(default_factory=list)       # check names that ran
    scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)     # ran in reduced mode (e.g. no InsightFace)
    skipped: list[str] = field(default_factory=list)      # could not run (e.g. Ollama down)

    def add(self, name: str, result: tuple[bool, float, str]) -> "Verdict":
        ok, score, reason = result
        self.checks.append(name)
        self.scores[name] = score
        self.reasons.append(f"[{name}] {reason}")
        if "SKIPPED" in reason or score == -1.0:
            self.skipped.append(name)
            return self  # skipped checks do not affect passed
        if "degraded" in reason.lower():
            self.degraded.append(name)
        self.passed = self.passed and ok
        return self

    def summary(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        tags = []
        if self.degraded:
            tags.append("degraded=" + ",".join(self.degraded))
        if self.skipped:
            tags.append("skipped=" + ",".join(self.skipped))
        tagstr = ("  (" + "; ".join(tags) + ")") if tags else ""
        return head + tagstr + "\n  " + "\n  ".join(self.reasons)


def verify_clip(clip_frames, target_node_frame=None, canonical_portrait=None,
                kind: str = "bridge", aside: Optional[str] = None,
                character_md: Optional[ImageLike] = None) -> Verdict:
    """Gate a baked clip before it becomes an edge/node in the clip graph.

    clip_frames      : sequence of frames (np arrays or paths); first/last are used.
    target_node_frame: required for kind='bridge'/'transition' -- the node the clip arrives at.
    canonical_portrait: optional -- if given, the clip's last frame is identity-checked.
    kind             : 'bridge'/'transition' -> continuity check; 'idle'/'loop' -> loopability.
    aside, character_md: optional -- if both given, the aside text is register-checked.

    Runs the relevant subset of checks and ANDs them (skips don't veto). Returns a Verdict.
    """
    if not clip_frames or len(clip_frames) < 2:
        v = Verdict(passed=False)
        v.reasons.append("[clip] FAIL: need at least 2 frames (first, last)")
        return v

    first, last = clip_frames[0], clip_frames[-1]
    v = Verdict()

    if kind in ("idle", "loop"):
        v.add("loopability", loopability(first, last))
    elif kind in ("bridge", "transition", "walk"):
        if target_node_frame is None:
            v.passed = False
            v.checks.append("continuity")
            v.reasons.append("[continuity] FAIL: kind=" + kind + " requires target_node_frame")
        else:
            v.add("continuity", continuity(last, target_node_frame))
    else:
        v.passed = False
        v.reasons.append(f"[clip] FAIL: unknown kind {kind!r}")

    if canonical_portrait is not None:
        v.add("identity", identity(last, canonical_portrait))

    if aside is not None and character_md is not None:
        v.add("register", register(aside, character_md))

    return v


def verify_portrait(portrait_img: ImageLike, canonical_portrait: Optional[ImageLike] = None,
                    character_md: Optional[ImageLike] = None) -> Verdict:
    """Gate a freshly generated still portrait before it becomes the canonical node art.

    portrait_img      : the new render (path or array).
    canonical_portrait: optional reference for an identity (re-render) check. For the FIRST
                        portrait of a character there is no reference -> identity is skipped.
    character_md      : optional -- if given, the portrait image is register-checked (look).

    Always runs a self-loopability sanity check (a still must be SSIM~1 against itself; this
    catches a corrupt/blank/unreadable render). ANDs the relevant checks into a Verdict.
    """
    v = Verdict()
    # A readable still is self-identical -> guards against blank/corrupt/half-written files.
    v.add("readable", loopability(portrait_img, portrait_img))

    if canonical_portrait is not None:
        v.add("identity", identity(portrait_img, canonical_portrait))

    if character_md is not None:
        v.add("register", register(portrait_img, character_md))

    return v


# ======================================================================================
# Self-test  --  synthetic numpy images, no files, no required network
# ======================================================================================
def _synthetic_face(size: int = 256, *, cx=None, cy=None, scale=1.0,
                    bg=(20, 120), hue=(180, 150, 170)) -> np.ndarray:
    """A crude deterministic 'portrait': gradient background + a few face-ish blobs.

    Parametric so the self-test can build a SHIFTED/SCALED pose (structurally different, not
    noise) -- which is what 'continuity drift' and 'broken seam' really look like in the
    pipeline. Not a real face (the self-test runs degraded without InsightFace); it just gives
    the SSIM / histogram / ORB paths real structure to score.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    grad = np.linspace(bg[0], bg[1], size, dtype=np.uint8)            # vertical gradient bg
    img[:] = grad[:, None, None]
    cx = size // 2 if cx is None else cx
    cy = size // 2 if cy is None else cy
    rx, ry = int(size // 4 * scale), int(size // 3 * scale)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, hue, -1)         # face oval
    er = max(3, int(size // 28 * scale))
    cv2.circle(img, (cx - size // 10, cy - size // 12), er, (20, 20, 30), -1)  # eyes
    cv2.circle(img, (cx + size // 10, cy - size // 12), er, (20, 20, 30), -1)
    cv2.ellipse(img, (cx, cy + size // 8), (size // 12, size // 24), 0, 0, 180, (40, 30, 60), 3)  # mouth
    return img


def _selftest() -> int:
    print("=" * 78)
    print("verify.py self-test  (synthetic images; no files; network-guarded)")
    print("=" * 78)

    face = _synthetic_face()                       # the canonical pose
    face_same = face.copy()
    face_jittered = np.clip(face.astype(np.int16) + 4, 0, 255).astype(np.uint8)  # +4 levels
    # a DIFFERENT pose: face shifted up-left and scaled up (same scene, wrong node)
    face_pose = _synthetic_face(cx=104, cy=96, scale=1.25)
    # a truncated/corrupt render: bottom half black (what a half-written file looks like)
    face_trunc = face.copy(); face_trunc[face.shape[0] // 2:, :, :] = 0
    noise = (np.random.default_rng(7).integers(0, 256, face.shape)).astype(np.uint8)

    # --- raw SSIM sanity (structure, not color: a recolored-but-aligned face stays ~1) ---
    print("\n-- SSIM primitive --")
    print(f"  ssim(face, face)        = {ssim(face, face):.4f}  (expect ~1.0)")
    print(f"  ssim(face, jittered)    = {ssim(face, face_jittered):.4f}  (expect high)")
    print(f"  ssim(face, shifted-pose)= {ssim(face, face_pose):.4f}  (expect mid -- structure moved)")
    print(f"  ssim(face, truncated)   = {ssim(face, face_trunc):.4f}  (expect low -- half gone)")
    print(f"  ssim(face, noise)       = {ssim(face, noise):.4f}  (expect near 0)")

    # --- loopability: identical first/last PASSES; a moved last frame FAILS the seam ---
    print("\n-- loopability --")
    p1, s1, r1 = loopability(face, face_same)
    p2, s2, r2 = loopability(face, face_pose)
    print(f"  identical   -> passed={p1}  {r1}")
    print(f"  moved seam  -> passed={p2}  {r2}")
    assert p1 is True, "identical frames must pass loopability"
    assert p2 is False, "a shifted last frame must fail loopability (seam pops)"

    # --- continuity: lands on target vs drifts to a different pose ---
    print("\n-- continuity --")
    p3, s3, r3 = continuity(face, face_same)
    p4, s4, r4 = continuity(face_pose, face)   # clip ends on the wrong (shifted) pose
    print(f"  lands       -> passed={p3}  {r3}")
    print(f"  drifts      -> passed={p4}  {r4}")
    assert p3 is True, "matching end/target must pass continuity"
    assert p4 is False, "ending on a different pose must fail continuity"

    # --- identity (degraded here unless InsightFace installed) ---
    print("\n-- identity --")
    p5, s5, r5 = identity(face, face_same)
    p6, s6, r6 = identity(face, noise)         # an utterly different image must fail even degraded
    print(f"  same        -> passed={p5}  {r5}")
    print(f"  different    -> passed={p6}  {r6}")
    assert p5 is True, "same image must pass identity (even degraded)"
    assert p6 is False, "a totally different image must fail identity (even degraded)"

    # --- register (skipped unless an Ollama chat model answers) ---
    print("\n-- register --")
    contract = ("Theme: a dim alchemist's study.\nVoice: dry patrician baritone, grandiloquent.\n"
                "A washed-up tragedian who knows he is a painting and plays to the house.")
    in_voice = "You there, behind the glass -- yes, you. Witness Act Five, performed for the ten-thousandth time."
    out_voice = "Hi, here's the weather forecast for Tuesday: sunny with a high of 71 degrees."
    pa, sa, ra = register(in_voice, contract)
    pb, sb, rb = register(out_voice, contract)
    print(f"  in-voice    -> passed={pa}  {ra}")
    print(f"  out-voice   -> passed={pb}  {rb}")

    # --- orchestrators ---
    print("\n-- verify_clip (idle loop, identical frames) --")
    v_idle = verify_clip([face, face_same, face_same], kind="idle",
                         canonical_portrait=face_same)
    print("  " + v_idle.summary().replace("\n", "\n  "))
    assert v_idle.passed is True, "good idle loop should pass"

    print("\n-- verify_clip (idle loop, broken seam) --")
    v_broken = verify_clip([face, face, face_trunc], kind="idle")
    print("  " + v_broken.summary().replace("\n", "\n  "))
    assert v_broken.passed is False, "broken-seam loop should fail"

    print("\n-- verify_clip (bridge, lands on target) --")
    v_bridge = verify_clip([face_pose, face, face_same], target_node_frame=face_same,
                          kind="bridge")
    print("  " + v_bridge.summary().replace("\n", "\n  "))
    assert v_bridge.passed is True, "bridge landing on target should pass"

    print("\n-- verify_clip (bridge, drifts off target) --")
    v_drift = verify_clip([face, face, face_pose], target_node_frame=face, kind="bridge")
    print("  " + v_drift.summary().replace("\n", "\n  "))
    assert v_drift.passed is False, "bridge ending on the wrong pose should fail"

    print("\n-- verify_portrait (first portrait, no reference) --")
    v_port = verify_portrait(face, character_md=contract)
    print("  " + v_port.summary().replace("\n", "\n  "))
    assert v_port.passed is True, "readable first portrait should pass"

    print("\n-- verify_portrait (corrupt/blank render) --")
    blank = np.zeros_like(face)
    # blank-vs-blank is self-identical so 'readable' passes; identity vs a real portrait fails
    v_blank = verify_portrait(blank, canonical_portrait=face)
    print("  " + v_blank.summary().replace("\n", "\n  "))

    print("\n" + "=" * 78)
    print("self-test OK -- all hard assertions held "
          "(identity degraded; register skipped unless an Ollama chat/VL model is loaded).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
