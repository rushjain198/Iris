"""Iris demo: title card for the split-screen video.   python make_title.py OUT.png [walk_speed] [hud_speed]"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1]
ws, hs = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("1.1", "1.4")
W, H = 672, 360
img = Image.new("RGB", (W, H), (5, 9, 14))
d = ImageDraw.Draw(img)


def font(sz, bold=False):
    p = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


CY, MUT, WH = (95, 243, 255), (139, 152, 168), (238, 243, 248)
d.text((36, 28), "IRIS", font=font(64, True), fill=CY)
d.text((40, 100), "sight for the path ahead", font=font(22), fill=MUT)
d.line((40, 140, W - 40, 140), fill=(35, 55, 70), width=2)
d.text((40, 156), "Blindfolded live test", font=font(30, True), fill=WH)


def tri(x, y, kind):   # arrow pointing at the panel it describes
    s = 9
    pts = {"left": [(x + s, y - s), (x - s, y), (x + s, y + s)], "up": [(x - s, y + s), (x, y - s), (x + s, y + s)],
           "right": [(x - s, y - s), (x + s, y), (x - s, y + s)]}[kind]
    d.polygon(pts, fill=CY)


for i, (k, tx) in enumerate((("left", "The real walk: blindfolded, with a cane"), ("up", "What Iris sees and says, live"),
                             ("right", "LiDAR scan of the course"))):
    y = 206 + i * 34
    tri(52, y + 13, k)
    d.text((76, y), tx, font=font(22), fill=WH)
d.text((40, 318), f"Camera glasses + laptop · on-device AI · clips sped up {ws}x / {hs}x", font=font(16), fill=MUT)
img.save(out)
