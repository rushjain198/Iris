"""Iris color identification: name the dominant color in the centre of view (clothes, fruit, pill bottles).

dominant_color(frame_bgr) -> {"name", "rgb", "share"}: centre 30% crop, pixels clustered in HSV, the largest cluster is
named. Lighting changes colors on a camera, so the answer is "looks <color>"; very dark/bright crops say so.
"""
import cv2
import numpy as np


def _name(h, s, v):
    """OpenCV HSV (h 0-179, s/v 0-255) -> plain color word."""
    if v < 45:
        return "black"
    if s < 40:
        return "white" if v > 200 else "light gray" if v > 140 else "gray" if v > 80 else "dark gray"
    deg = h * 2
    if s < 110 and v < 150 and 10 <= deg < 45:
        return "brown"
    if deg < 12 or deg >= 340:
        return "red" if v > 90 else "dark red"
    if deg < 40:
        return "orange" if v > 140 else "brown"
    if deg < 68:
        return "yellow"
    if deg < 160:
        return "green"
    if deg < 200:
        return "teal"
    if deg < 255:
        return "blue"
    if deg < 290:
        return "purple"
    return "pink"


def dominant_color(frame_bgr, crop=0.3, k=3):
    h, w = frame_bgr.shape[:2]
    ch, cw = int(h * crop), int(w * crop)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    patch = cv2.GaussianBlur(frame_bgr[y0:y0 + ch, x0:x0 + cw], (5, 5), 0)
    px = patch.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    i = int(np.argmax(counts))
    bgr = np.clip(centers[i], 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    return {"name": _name(int(hsv[0]), int(hsv[1]), int(hsv[2])), "rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])],
            "share": round(float(counts[i] / counts.sum()), 2)}


def sentence(c):
    return f"Looks {c['name']}." if c["share"] >= 0.5 else f"Mostly {c['name']}, with other colors."


if __name__ == "__main__":
    for name, bgr in (("red", (30, 30, 210)), ("blue", (200, 80, 20)), ("green", (40, 170, 40)), ("yellow", (30, 220, 230)), ("white", (245, 245, 245)), ("black", (15, 15, 15))):
        img = np.full((480, 640, 3), bgr, np.uint8)
        print(name, "->", dominant_color(img))
