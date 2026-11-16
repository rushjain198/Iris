"""Frame sources: open_source("glasses" | "webcam:N" | path) -> object with read() -> (ok, frame_bgr)."""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # usbcam.py lives next to this file

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


class ImageSource:
    """A still image served forever (for tests / demos)."""

    def __init__(self, path):
        self.img = cv2.imread(path)

    def isOpened(self):
        return self.img is not None

    def read(self):
        return (self.img is not None), (None if self.img is None else self.img.copy())

    def release(self):
        pass


class LoopingVideo:
    """Video file that loops, so demos never end."""

    def __init__(self, path):
        self.path, self.cap = path, cv2.VideoCapture(path)

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return ok, f

    def release(self):
        self.cap.release()


def open_source(spec):
    if spec == "glasses":
        from usbcam import SerialCapture
        return SerialCapture("auto")
    if spec.startswith("webcam"):
        idx = int(spec.split(":")[1]) if ":" in spec else 0
        return cv2.VideoCapture(idx)
    if spec.lower().endswith(IMG_EXT):
        return ImageSource(spec)
    return LoopingVideo(spec)
