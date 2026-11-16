"""Iris depth: monocular relative depth (Depth Anything V2 Small, ONNX) + obstacle zones.

Depth is RELATIVE (no metric scale): larger = nearer, normalized per frame.
"""
import os
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
HF = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/"
DEFAULT_MODEL = "model_quantized.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _download(name):
    path = os.path.join(MODEL_DIR, name)
    if not (os.path.exists(path) and os.path.getsize(path) > 1_000_000):
        os.makedirs(MODEL_DIR, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(HF + name, tmp)
        if os.path.getsize(tmp) < 1_000_000:
            raise RuntimeError("model download too small: " + name)
        os.replace(tmp, path)
    return path


class DepthEstimator:
    def __init__(self, model_path=None, short_side=256, providers=None):
        import onnxruntime as ort
        self.model_path = model_path or _download(DEFAULT_MODEL)
        self.short_side = short_side
        # Temporal normalisation: per-frame stretching made a far, empty scene read as 'near'. The range adapts slowly,
        # but grows fast when something nearer than anything recent appears.
        self.temporal = True
        self._lo = self._hi = None
        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ("CPUExecutionProvider",) if p in avail]  # CPU only: CoreML measured slower on int8 (134 vs 43 ms)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            self.sess = ort.InferenceSession(self.model_path, so, providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(self.model_path, so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.fp16 = "float16" in inp.type
        self.provider = self.sess.get_providers()[0]

    def _size(self, h, w):
        s = self.short_side / min(h, w)
        r = lambda v: max(14, int(round(v * s / 14.0)) * 14)
        return r(h), r(w)

    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        th, tw = self._size(h, w)
        img = cv2.cvtColor(cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        if self.fp16:
            x = x.astype(np.float16)
        out = self.sess.run(None, {self.in_name: np.ascontiguousarray(x)})[0]
        d = np.squeeze(out).astype(np.float32)  # inverse-depth style: larger = nearer
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        lo, hi = (float(v) for v in np.percentile(d, [2, 98]))
        if self.temporal:
            if self._lo is None:
                self._lo, self._hi = lo, hi
            else:
                self._lo += (0.5 if lo < self._lo else 0.03) * (lo - self._lo)
                self._hi += (0.5 if hi > self._hi else 0.03) * (hi - self._hi)
            lo, hi = self._lo, self._hi
        return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def _level(near, caution, stop):
    return "stop" if near >= stop else "caution" if near >= caution else "clear"


def obstacles(depth, band=(1 / 3, 1.0), caution=0.55, stop=0.75):
    """Zones from the walking band (rows band[0]..band[1] of height), thirds left/center/right."""
    h, w = depth.shape[:2]
    band_d = depth[int(h * band[0]):int(h * band[1])]
    zones = {}
    for i, name in enumerate(("left", "center", "right")):
        z = band_d[:, int(w * i / 3):int(w * (i + 1) / 3)]
        near = float(np.percentile(z, 95)) if z.size else 0.0
        zones[name] = {"near": near, "level": _level(near, caution, stop)}
    clear = None
    if any(z["level"] == "clear" for z in zones.values()):
        # least-near zone; center wins ties (within 1e-6)
        order = ("center", "left", "right")
        clear = min(order, key=lambda n: (round(zones[n]["near"], 6), order.index(n)))
    return {"zones": zones, "clear": clear, "nearest": max(z["near"] for z in zones.values())}
