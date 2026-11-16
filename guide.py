"""Iris guidance: Guide (what to say), Speaker (macOS say), Detector (MediaPipe), describe (Gemini/local).

Prototype. Not a mobility aid replacement: keep using your cane or guide dog.
"""
import base64
import json
import os
import queue
import subprocess
import threading
import time
import urllib.request

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "efficientdet_lite0.tflite")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/"
             "efficientdet_lite0.tflite")
MODEL_SHA256 = "40338edf5ec70d43e318b0a716a84d4564cd1802759a7a07170c7e43796dbf58"


def ensure_detector_model(path=MODEL_PATH):
    """Download the COCO detector once (13.8 MB, not in git); refuse a file whose sha256 does not match."""
    import hashlib
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest} != {MODEL_SHA256}; delete it and retry")
    return path
RANK = {"clear": 0, "caution": 1, "stop": 2}
POS = {"left": "on your left", "center": "ahead", "right": "on your right"}
ZONES = ("left", "center", "right")


def zone_of(box, width):
    """Zone by box centre x thirds."""
    x, _, w, _ = box
    c = (x + w / 2.0) / max(width, 1)
    return "left" if c < 1 / 3 else ("center" if c < 2 / 3 else "right")


def _label_in(detections, zone):
    """Label of the highest-score detection in a zone, else None."""
    best = [d for d in detections or [] if d.get("zone") == zone]
    if not best:
        return None
    return max(best, key=lambda d: d.get("score", 0))["label"]


def _pretty(label):
    return label.replace("_", " ") if label else None


def _situation(obs, detections):
    """Pure: (level, text) for one frame."""
    zones = obs["zones"]
    center = zones["center"]["level"]
    level = "clear"
    for z in ZONES:
        lv = zones[z]["level"]
        if lv == "stop" and z != "center":
            lv = "caution"  # only the walking line can demand a full stop
        if RANK[lv] > RANK[level]:
            level = lv
    if level == "clear":
        return "clear", "Path clear ahead."
    if level == "stop":
        name = _pretty(_label_in(detections, "center")) or "Obstacle"
        return "stop", "Stop. %s ahead." % name.capitalize()
    # caution: worst non-clear zone (center preferred)
    worst = max(ZONES, key=lambda z: (RANK[zones[z]["level"]], zones[z]["near"], z == "center"))
    name = _pretty(_label_in(detections, worst)) or "Obstacle"
    text = "%s %s." % (name.capitalize(), POS[worst])
    clear = obs.get("clear")
    if clear and clear != worst:
        text += " Path clear ahead." if clear == "center" else " Path clear on the %s." % clear
    elif center == "clear" and worst != "center":
        pass
    return "caution", text


class Guide:
    def __init__(self, cooldown_s=2.5, hysteresis=2, clear_every_s=10.0, warmup_s=2.0):
        self.cooldown_s = cooldown_s
        self.warmup_s, self._t_first, self._last_stop = warmup_s, None, -1e9   # depth range settles in the first seconds
        self.hysteresis = hysteresis
        self.clear_every_s = clear_every_s
        self.level = "clear"
        self._cand = None  # (level, count)
        self._last_text = {}
        self._last_any = -1e9
        self._last_clear = -1e9
        self._said_text = None

    def update(self, obs, detections, t):
        if self._t_first is None:
            self._t_first = t
        if t - self._t_first < self.warmup_s:
            return []
        level, text = _situation(obs, detections)
        # hysteresis on level changes
        if level != self.level:
            n = self._cand[1] + 1 if self._cand and self._cand[0] == level else 1
            self._cand = (level, n)
            if n < self.hysteresis:
                return []
            self.level = level
            self._cand = None
        else:
            self._cand = None
        if self.level == "clear":
            # speak the all-clear rarely: only right after a warning, and not often
            if self._said_text not in (None, text) and t - self._last_clear >= self.clear_every_s:
                self._last_clear = t
                return self._emit("clear", text, t)
            return []
        if level != self.level:
            return []
        return self._emit(self.level, text, t)

    def _emit(self, level, text, t):
        if t - self._last_text.get(text, -1e9) < self.cooldown_s:
            return []
        # one 'Stop' per cooldown even when the object name flips (it chattered 'Clock ahead' / 'Obstacle ahead' live)
        if level == "stop" and t - self._last_stop < self.cooldown_s:
            return []
        if level == "stop":
            self._last_stop = t
        if level != "stop" and text != self._said_text and t - self._last_any < self.cooldown_s * 0.5:
            return []
        if level != "stop" and text == self._said_text and t - self._last_text[text] < self.cooldown_s * 2:
            return []
        self._last_text[text] = t
        self._last_any = t
        self._said_text = text
        return [{"t": t, "level": level, "text": text}]


class Speaker:
    """Non-blocking macOS `say`. Latest line wins; urgent kills current speech. dry_run: no audio."""

    def __init__(self, rate=210, dry_run=False):
        self.rate = rate
        self.dry_run = dry_run
        self.muted = False
        self.spoken = []  # log of lines actually started (for tests)
        self._proc = None
        self._pending = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def say(self, text, urgent=False):
        if self.muted or not text:
            return
        with self._lock:
            self._pending = text  # latest wins: replaces any stale queued line
            if urgent:
                self._kill()
        self._wake.set()

    def set_mute(self, flag):
        self.muted = bool(flag)
        if self.muted:
            with self._lock:
                self._pending = None
                self._kill()

    def close(self):
        self._stop = True
        self._wake.set()
        with self._lock:
            self._kill()

    def _kill(self):
        p = self._proc
        if p and p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass

    def _busy(self):
        return self._proc is not None and self._proc.poll() is None

    def _loop(self):
        while not self._stop:
            self._wake.wait(0.05)
            with self._lock:
                if self._pending is None or self._busy():
                    if self._pending is None:
                        self._wake.clear()
                    continue
                text, self._pending = self._pending, None
                self.spoken.append(text)
                if not self.dry_run:
                    try:
                        self._proc = subprocess.Popen(["say", "-r", str(self.rate), text])
                    except OSError:
                        self._proc = None


class Detector:
    """MediaPipe ObjectDetector (efficientdet_lite0), rate-limited to one run per `interval` s."""

    def __init__(self, interval=0.3, min_score=0.4, model_path=MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self._mp = mp
        opts = vision.ObjectDetectorOptions(
            base_options=mpp.BaseOptions(model_asset_path=os.path.abspath(ensure_detector_model(model_path) if model_path == MODEL_PATH else model_path)),
            score_threshold=min_score, max_results=8)
        self._det = vision.ObjectDetector.create_from_options(opts)
        self.interval = interval
        self._last_t = -1e9
        self._last = []

    def detect(self, frame_bgr, now=None):
        now = time.time() if now is None else now
        if now - self._last_t < self.interval:
            return self._last
        self._last_t = now
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        out = []
        for d in self._det.detect(img).detections:
            c = d.categories[0]
            if c.score < 0.4:
                continue
            b = d.bounding_box
            box = [int(b.origin_x), int(b.origin_y), int(b.width), int(b.height)]
            out.append({"label": c.category_name, "score": float(c.score), "box": box,
                        "zone": zone_of(box, w)})
        self._last = out
        return out


def _local_sentence(detections, obs):
    parts = []
    if obs:
        for z in ZONES:
            lv = obs["zones"][z]["level"]
            if lv != "clear":
                name = _pretty(_label_in(detections, z)) or "obstacle"
                parts.append("%s %s%s" % (name, POS[z], " very close" if lv == "stop" else ""))
    if parts:
        s = "Careful: " + ", ".join(parts) + "."
        if obs.get("clear"):
            s += " Clearest way is %s." % obs["clear"]
        return s
    names = sorted({_pretty(d["label"]) for d in detections or []})
    if names:
        return "Path looks clear. I can see " + ", ".join(names) + "."
    return "Path looks clear ahead."


def describe(frame, detections=None, obs=None, model="gemini-2.0-flash"):
    """One short sentence for a blind pedestrian. Gemini if GOOGLE_API_KEY set, else local."""
    key = os.environ.get("GOOGLE_API_KEY")
    if key and frame is not None:
        try:
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                hint = _local_sentence(detections, obs) if (detections or obs) else ""
                prompt = ("Describe this scene in one short sentence for a blind pedestrian walking forward. "
                          "Mention hazards and obstacles first, then anything useful. "
                          + ("Sensor hints: " + hint if hint else ""))
                body = {"contents": [{"parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg",
                                     "data": base64.b64encode(jpg.tobytes()).decode()}}]}]}
                url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
                req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                             headers={"Content-Type": "application/json",
                                                      "x-goog-api-key": key})
                with urllib.request.urlopen(req, timeout=8) as r:
                    data = json.load(r)
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text:
                    return text
        except Exception:
            pass  # never log: errors may echo request details
    return _local_sentence(detections, obs)
