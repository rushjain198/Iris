"""Iris: camera -> monocular depth -> obstacle zones -> guidance (speech) + web dashboard on 127.0.0.1:8810."""
import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from camera import open_source

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
LURCH, STILL = 28.0, 2.5   # mean |frame diff| on 80x60 gray: a lurch vs a still view (tune live)
PORT = 8810
SAFETY = "Prototype. Not a mobility aid replacement: keep using your cane or guide dog."
LEVEL_BGR = {"clear": (80, 200, 60), "caution": (0, 190, 255), "stop": (60, 60, 240)}
ZONES = ("left", "center", "right")


# ---------- stubs / fallbacks (used with --stub or when depth.py / guide.py are missing) ----------
class StubDepth:
    def infer(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = np.percentile(g, 2), np.percentile(g, 98)
        return np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def stub_obstacles(depth):
    h, w = depth.shape
    band = depth[h // 3:]
    zones = {}
    for i, name in enumerate(ZONES):
        near = float(np.percentile(band[:, i * w // 3:(i + 1) * w // 3], 95))
        zones[name] = {"near": near, "level": "stop" if near > .85 else "caution" if near > .6 else "clear"}
    clear = min(ZONES, key=lambda z: zones[z]["near"])
    return {"zones": zones, "clear": clear if zones[clear]["level"] != "stop" else None,
            "nearest": max(z["near"] for z in zones.values())}


class FallbackGuide:
    def __init__(self):
        self.last = {}

    def update(self, obs, detections, t):
        ev = []
        for name, z in obs["zones"].items():
            if z["level"] != "clear" and t - self.last.get(name, -99) > 2.5:
                self.last[name] = t
                ev.append({"t": t, "level": z["level"], "text": f"{z['level']} {name}"})
        return ev


class FallbackSpeaker:
    def say(self, text, urgent=False):
        subprocess.Popen(["say", "-r", "210", text])


def fallback_describe(frame, obs=None, detections=None):
    return "Prototype description: " + (f"clear path {obs['clear']}." if obs and obs.get("clear") else "no clear path.")


# ---------- engine ----------
class Engine:
    def __init__(self, source, stub=False, speech=True):
        self.source_name, self.stub, self.speech = source, stub, speech
        self.muted = not speech
        self.cap = open_source(source)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open source {source!r}")
        self.depth = self.obstacles = self.guide = self.speaker = self.detector = None
        self.describe_fn = fallback_describe
        self._load()
        self.lock = threading.Lock()
        self.raw = None                 # latest camera frame (grab thread)
        self.jpeg = None                # latest annotated jpeg
        self.obs = {"zones": {z: {"near": 0, "level": "clear"} for z in ZONES}, "clear": None, "nearest": 0}
        self.dets, self.events, self.fps, self.emergency_t = [], [], 0.0, 0
        # SOS / auto-distress (idea from EchoNet trapped-user detection)
        self.sos = None                 # {"t", "reason"} while active
        self.watch = False              # fall check: off by default (setting the glasses down looks like a fall)
        self.check_t = None             # when "Are you okay?" was asked
        self.motion, self._prev_small, self._lurch_t, self._still_since = 0.0, None, None, None
        self.last_color = ""
        self.frame_ev = threading.Event()
        self.stop = False
        self.last_describe = ""
        self.last_read = ""

    def _load(self):
        if self.stub:
            self.depth, self.obstacles = StubDepth(), stub_obstacles
        else:
            import depth as D
            self.depth, self.obstacles = D.DepthEstimator(), D.obstacles
        try:
            import guide as G
            self.guide = G.Guide()
            self.speaker = G.Speaker()
            self.describe_fn = G.describe
            if not self.stub:
                self.detector = G.Detector()
        except Exception as e:
            if not self.stub:
                print("guide.py unavailable, using fallback:", e)
            self.guide, self.speaker = FallbackGuide(), FallbackSpeaker()

    def start(self):
        threading.Thread(target=self._grab, daemon=True).start()
        threading.Thread(target=self._process, daemon=True).start()

    def _reopen(self):
        """Camera went quiet (cable jiggled, board slept, new /dev/cu.usbmodem name): reopen it."""
        self.cam_status = "reconnecting"
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.cap = open_source(self.source_name)
        except Exception:
            pass                                    # no device yet; try again on the next timeout

    def _grab(self):
        self.cam_status, last_ok = "live", time.time()
        live_src = self.source_name.startswith(("glasses", "webcam"))
        while not self.stop:
            try:
                ok, f = self.cap.read()
            except Exception:
                ok, f = False, None
            if not ok or f is None:
                if live_src and time.time() - last_ok > 2.0:
                    self._reopen()
                    last_ok = time.time()
                time.sleep(0.05)
                continue
            last_ok, self.cam_status = time.time(), "live"
            if getattr(self, 'rotate', 0):
                f = cv2.rotate(f, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotate])
            with self.lock:
                self.raw = f
            self.frame_ev.set()
            if self.source_name.startswith("file") or not self.source_name.startswith(("glasses", "webcam")):
                time.sleep(1 / 25)      # pace files

    def _process(self):
        n, t0 = 0, time.time()
        while not self.stop:
            if not self.frame_ev.wait(1):
                continue
            self.frame_ev.clear()
            with self.lock:
                frame = None if self.raw is None else self.raw.copy()
            if frame is None:
                continue
            t = time.time()
            h, w = frame.shape[:2]
            s = 256 / min(h, w)
            small = cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
            d = self.depth.infer(small)
            d = cv2.resize(d, (w, h))
            obs = self.obstacles(d)
            if self.detector is not None:
                try:
                    self.dets = self.detector.detect(frame, t)
                except Exception:
                    pass
            self._fall_check(frame, t)
            evs = self.guide.update(obs, self.dets, t)
            for ev in evs:
                if not self.muted:
                    self.speaker.say(ev["text"], urgent=ev["level"] == "stop")
                if ev["level"] == "stop":
                    self.emergency_t = t
            vis = self._render(frame, d, obs)
            ok, jp = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            okr, jr = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])   # clean frame for the HUD page
            n += 1
            with self.lock:
                self.obs = obs
                self.events = (self.events + evs)[-30:]
                if ok:
                    self.jpeg = jp.tobytes()
                if okr:
                    self.jpeg_raw, self.frame_size = jr.tobytes(), [w, h]
                if time.time() - t0 > 1:
                    self.fps, n, t0 = n / (time.time() - t0), 0, time.time()

    def _render(self, frame, d, obs):
        heat = cv2.applyColorMap((np.clip(d, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        vis = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)
        h, w = vis.shape[:2]
        for dt in self.dets:
            x, y, bw, bh = [int(v) for v in dt["box"]]
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 255, 255), 2)
            cv2.putText(vis, f"{dt['label']} {dt['score']:.2f}", (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        bar = max(18, h // 14)
        # zone bars now live in the web UI (path radar); drawing them here covered the subtitles
        for i, name in enumerate(ZONES if getattr(self, "draw_bars", False) else ()):
            z = obs["zones"][name]
            x0, x1 = i * w // 3, (i + 1) * w // 3
            cv2.rectangle(vis, (x0 + 2, h - bar), (x1 - 2, h - 2), LEVEL_BGR[z["level"]], -1)
            txt = f"{name.upper()} {z['level'].upper()}" + ("  <- GO" if obs.get("clear") == name else "")
            cv2.putText(vis, txt, (x0 + 10, h - bar // 3), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 2, cv2.LINE_AA)
        return vis

    def state(self):
        with self.lock:
            return {"zones": self.obs["zones"], "clear": self.obs["clear"], "nearest": self.obs.get("nearest", 0),
                    "detections": self.dets, "events": self.events[-30:], "fps": round(self.fps, 1),
                    "source": self.source_name, "muted": self.muted, "stub": self.stub,
                    "emergency": time.time() - self.emergency_t < 6, "describe": self.last_describe, "read": self.last_read, "color": self.last_color,
                    "sos": self.sos, "watch": self.watch, "cam": getattr(self, "cam_status", "live"), "frame": getattr(self, "frame_size", [640, 480]), "checking": self.check_t is not None, "motion": round(self.motion, 1),
                    "safety": SAFETY}

    def describe(self):
        with self.lock:
            frame = None if self.raw is None else self.raw.copy()
            obs = self.obs
        if frame is None:
            return "No camera frame yet."
        try:
            text = self.describe_fn(frame, self.dets, obs)
        except Exception as e:
            text = f"Description failed: {e}"
        self.last_describe = text
        with self.lock:
            self.events = (self.events + [{"t": time.time(), "level": "info", "text": "Describe: " + text}])[-30:]
        if not self.muted:
            self.speaker.say(text)
        return text

    # ---- SOS, fall check, color ----
    def _log(self, level, text):
        with self.lock:
            self.events = (self.events + [{"t": time.time(), "level": level, "text": text}])[-30:]

    def raise_sos(self, reason):
        if self.sos:
            return self.sos
        self.sos = {"t": time.time(), "reason": reason}
        self.check_t = None
        self._log("stop", f"SOS: {reason}")
        self.speaker.say("Calling for help.", urgent=True)        # always spoken, even when muted
        threading.Thread(target=self._alarm, daemon=True).start()
        return self.sos

    def _alarm(self):
        while self.sos and not self.stop and not os.environ.get("IRIS_NO_ALARM"):   # audible alarm until cleared
            subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"], check=False)
            time.sleep(0.6)

    def clear_sos(self):
        was = self.sos or self.check_t
        self.sos, self.check_t = None, None
        if was:
            self._log("clear", "Okay. Alert cleared.")
            self.speaker.say("Okay. Alert cleared.", urgent=True)
        return {"cleared": bool(was)}

    def _fall_check(self, frame, t):
        """Camera-only heuristic: a sudden big view change (lurch) followed by >= 3 s of near-stillness -> ask
        'Are you okay?'; no 'I am okay' within 10 s -> SOS. Off unless watch mode is on."""
        g = cv2.cvtColor(cv2.resize(frame, (80, 60)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev_small is not None:
            self.motion = float(np.mean(np.abs(g - self._prev_small)))
        self._prev_small = g
        if not self.watch or self.sos:
            return
        if self.check_t is not None:
            if t - self.check_t > 10:
                self.raise_sos("no answer after a possible fall")
            return
        if self.motion > LURCH:
            self._lurch_t, self._still_since = t, None
        elif self._lurch_t and t - self._lurch_t < 2.0 and self.motion < STILL:
            self._still_since = self._still_since or t
        elif self.motion >= STILL:
            self._still_since = None
        if self._lurch_t and self._still_since and t - self._still_since >= 3.0:
            self._lurch_t = self._still_since = None
            self.check_t = t
            self._log("caution", "Possible fall. Are you okay? Say: I am okay.")
            self.speaker.say("Are you okay? Say, I am okay.", urgent=True)

    def color(self):
        with self.lock:
            frame = None if self.raw is None else self.raw.copy()
        if frame is None:
            return {"text": "No camera frame yet."}
        import color as C
        c = C.dominant_color(frame)
        text = C.sentence(c)
        self.last_color = text
        self._log("info", "Color: " + text)
        if not self.muted:
            self.speaker.say(text, urgent=True)
        return {"text": text, **c}

    def read_text(self):
        """Read signs/labels in the latest frame (Apple Vision OCR, on-device) and speak them."""
        with self.lock:
            frame = None if self.raw is None else self.raw.copy()
        if frame is None:
            return {"text": "No camera frame yet.", "lines": []}
        try:
            import ocr
            lines = ocr.read_text(frame)
            text = ocr.sentence(lines)
        except Exception as e:
            lines, text = [], f"Text reading failed: {e}"
        self.last_read = text
        with self.lock:
            self.events = (self.events + [{"t": time.time(), "level": "info", "text": text}])[-30:]
        if not self.muted:
            self.speaker.say(text, urgent=True)
        return {"text": text, "lines": [{k: l[k] for k in ("text", "conf", "box")} for l in lines]}


# ---------- http ----------
def make_handler(eng):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(200, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/stream.mjpg":
                return self._stream()
            if p == "/raw.mjpg":
                return self._stream("jpeg_raw")
            if p == "/api/state":
                return self._json(eng.state())
            page = {"/": "index.html", "/caregiver": "caregiver.html", "/hud": "hud.html"}.get(p)
            f = page or p.lstrip("/")
            fp = os.path.normpath(os.path.join(WEB, f))
            if not fp.startswith(WEB) or not os.path.isfile(fp):
                return self._send(404, b"not found", "text/plain")
            ct = {"html": "text/html", "css": "text/css", "js": "application/javascript"}.get(fp.rsplit(".", 1)[-1], "application/octet-stream")
            with open(fp, "rb") as fh:
                body = fh.read()
            self._send(200, body, ct + "; charset=utf-8")

        def do_POST(self):
            if self.path == "/api/describe":
                return self._json({"text": eng.describe()})
            if self.path == "/api/read":
                return self._json(eng.read_text())
            if self.path == "/api/color":
                return self._json(eng.color())
            if self.path == "/api/sos":
                return self._json({"sos": eng.raise_sos("wearer asked for help")})
            if self.path == "/api/okay":
                return self._json(eng.clear_sos())
            if self.path == "/api/watch":
                eng.watch = not eng.watch
                eng._log("info", "Fall check on." if eng.watch else "Fall check off.")
                return self._json({"watch": eng.watch})
            if self.path == "/api/mute":
                eng.muted = not eng.muted
                return self._json({"muted": eng.muted})
            self._send(404, b"not found", "text/plain")

        def _stream(self, attr="jpeg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while not eng.stop:
                    jp = getattr(eng, attr, None)
                    if jp is None or jp is last:
                        time.sleep(0.03)
                        continue
                    last = jp
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jp) + jp + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


def serve(eng, port=PORT):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(eng))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcam:0", help="glasses | webcam:N | file path")
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--stub", action="store_true", help="fake depth from brightness, no detector")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None,
                    help="rotate frames; default 180 for glasses (camera is mounted upside down), else 0")
    a = ap.parse_args()
    eng = Engine(a.source, a.stub, not a.no_speech)
    eng.rotate = a.rotate if a.rotate is not None else (180 if a.source == "glasses" else 0)
    eng.start()
    serve(eng, a.port)
    print(f"Iris on http://127.0.0.1:{a.port}  (caregiver: /caregiver)  source={a.source}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        eng.stop = True


if __name__ == "__main__":
    main()
