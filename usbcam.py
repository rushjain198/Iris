"""Iris glasses camera: read JPEG frames sent by firmware/iris_cam_usb over the USB-C cable. Behaves like cv2.VideoCapture (isOpened/read/
release), so the server can use  --camera serial:/dev/cu.usbmodemXXXX  (or  serial:auto ).

Frame format: 0xA5 0x5A 'J' 'P' | uint32 little-endian length | JPEG bytes.
"""
import glob
import struct
import threading
import time

import cv2
import numpy as np

MAGIC = b"\xa5\x5aJP"
MAX_JPEG = 400_000


def find_port():
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    return ports[0] if ports else None


class SerialCapture:
    def __init__(self, port="auto", framesize=None):
        import serial
        self.port = find_port() if port in ("auto", "", None) else port
        self.ser = None
        self.latest, self.seq, self.frames = None, 0, 0
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self._stop = False
        if not self.port:
            return
        try:
            self.ser = serial.Serial(self.port, 2_000_000, timeout=0.2)
        except Exception:
            self.ser = None
            return
        if framesize is not None:
            self.command(f"F{int(framesize)}")
        threading.Thread(target=self._reader, daemon=True).start()

    def command(self, text):
        if self.ser:
            self.ser.write((text + "\n").encode())

    def isOpened(self):
        return self.ser is not None

    def _reader(self):
        buf = bytearray()
        while not self._stop:
            try:
                chunk = self.ser.read(65536)
            except Exception:
                break
            if not chunk:
                continue
            buf += chunk
            while True:
                i = buf.find(MAGIC)
                if i < 0:
                    del buf[:-3]                     # keep a possible partial magic
                    break
                if len(buf) < i + 8:
                    if i:
                        del buf[:i]
                    break
                (n,) = struct.unpack("<I", bytes(buf[i + 4:i + 8]))
                if n == 0 or n > MAX_JPEG:           # corrupt length: skip this magic
                    del buf[:i + 4]
                    continue
                if len(buf) < i + 8 + n:
                    if i:
                        del buf[:i]
                    break
                jpg = bytes(buf[i + 8:i + 8 + n])
                del buf[:i + 8 + n]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                with self.cond:
                    self.latest, self.seq = img, self.seq + 1
                    self.frames += 1
                    self.cond.notify_all()
        with self.cond:
            self._stop = True
            self.cond.notify_all()

    def read(self, timeout=2.0):
        """Block until a frame newer than the last one returned arrives (like a camera)."""
        with self.cond:
            if self._stop:                      # cable pulled / port gone: report it so the engine reconnects
                return False, None
            seen = getattr(self, "_seen", 0)
            end = time.monotonic() + timeout
            while self.seq == seen and not self._stop:
                left = end - time.monotonic()
                if left <= 0:
                    return False, None
                self.cond.wait(left)
            if self.latest is None or self._stop:
                return False, None
            self._seen = self.seq
            return True, self.latest.copy()

    def release(self):
        self._stop = True
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


if __name__ == "__main__":       # quick fps check:  python usbcam.py [port] [framesize]
    import sys
    cap = SerialCapture(sys.argv[1] if len(sys.argv) > 1 else "auto",
                        int(sys.argv[2]) if len(sys.argv) > 2 else None)
    if not cap.isOpened():
        sys.exit("FAILED: no USB serial port (is the glasses' USB-C cable plugged in?)")
    time.sleep(1.5)
    t0, n0 = time.monotonic(), cap.frames
    time.sleep(6)
    fps = (cap.frames - n0) / (time.monotonic() - t0)
    ok, img = cap.read()
    print(f"port {cap.port} | {fps:.1f} fps | frame {'none' if not ok else f'{img.shape[1]}x{img.shape[0]}'}")
    if ok:
        cv2.imwrite(sys.argv[3] if len(sys.argv) > 3 else "usb_frame.jpg", img)
    cap.release()
