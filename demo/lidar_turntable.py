"""Iris demo: render a LiDAR scan (STL, Y-up, metres) as a rotating point-cloud turntable.

    python lidar_turntable.py SCAN.stl OUT_DIR            # 45 s at 15 fps -> OUT_DIR/f0000.png ...
    python lidar_turntable.py SCAN.stl OUT_DIR --test     # 8 angles -> OUT_DIR/test.jpg
The roof (anything above ROOF_CUT m) is removed so the walking course shows. Orthographic view, spun about the
scan's centre of mass, one fixed scale that fits every angle.
"""
import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from stl import load
ROOF_CUT, FPS, SECS = 2.2, 15, 45
SCAN, OUT = sys.argv[1], sys.argv[2]
W, H = 640, 360
t = load(SCAN).astype(np.float32)
c = t.mean(1)                                                  # triangle centroids -> point cloud
n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
keep = c[:, 1] < ROOF_CUT                                           # cut the patio roof away: show the course
t, n = t[keep], n[keep]
# denser cloud: centroid + the 3 corners of every triangle (big floor triangles left holes)
c = np.concatenate([t.mean(1), t[:, 0], t[:, 1], t[:, 2]]); n = np.concatenate([n, n, n, n])
hn = np.clip(c[:, 1] / ROOF_CUT, 0, 1)                             # height 0 (floor) .. 1 (roof)
# Iris palette by height: deep blue floor -> cyan -> violet -> white roof
stops = np.array([[0.0, 150, 95, 25], [0.12, 230, 190, 50], [0.45, 255, 240, 110], [0.8, 255, 140, 190], [1.0, 255, 230, 235]], np.float32)  # BGR
base = np.stack([np.interp(hn, stops[:, 0], stops[:, k]) for k in (1, 2, 3)], 1)
# orthographic turntable: spin about the scan's centre of mass, one fixed scale that fits every angle
PIV = np.array([np.median(c[:, 0]), 0.0, np.median(c[:, 2])], np.float32)
P = c - PIV
R95 = float(np.percentile(np.hypot(P[:, 0], P[:, 2]), 99.5))   # ignore stray scan blobs when sizing
YMAX = float(np.percentile(c[:, 1], 99.5))
def frame(ang, elev=np.deg2rad(35)):
    se, ce = np.sin(elev), np.cos(elev)
    s = min(W * 0.97 / (2 * R95), H * 0.94 / (2 * R95 * se + YMAX * ce))     # fits the widest pose
    ca, sa = np.cos(ang), np.sin(ang)
    x = P[:, 0] * ca + P[:, 2] * sa; z = -P[:, 0] * sa + P[:, 2] * ca; y = P[:, 1]
    nx = n[:, 0] * ca + n[:, 2] * sa; nz = -n[:, 0] * sa + n[:, 2] * ca; ny = n[:, 1]
    u = (W / 2 + x * s).astype(np.int32)
    v = (H / 2 + (YMAX * ce / 2) * s - (z * se + y * ce) * s).astype(np.int32)
    depth = z * ce - y * se                                                  # larger = farther from the camera
    view = np.array([0.0, -se, ce], np.float32)                              # camera looks into +z and down
    shade = 0.32 + 0.68 * np.abs(nx * view[0] + ny * view[1] + nz * view[2])
    dn = (depth - depth.min()) / (np.ptp(depth) + 1e-6)
    fog = 1.0 - 0.45 * dn
    col = (base * (shade * fog)[:, None]).clip(0, 255).astype(np.uint8)
    img = np.zeros((H, W, 3), np.uint8); img[:] = (14, 9, 5)
    order = np.argsort(-depth)                                               # far first, near overwrite
    u, v, col = u[order], v[order], col[order]
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        uu, vv = u + du, v + dv; ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        img[vv[ok], uu[ok]] = col[ok]
    return img
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if '--test' in sys.argv:
        ims = [frame(np.deg2rad(a)) for a in range(0, 360, 45)]
        cv2.imwrite(os.path.join(OUT, 'test.jpg'), np.vstack([np.hstack(ims[:4]), np.hstack(ims[4:])]))
    else:
        for i in range(FPS * SECS):
            cv2.imwrite(os.path.join(OUT, f'f{i:04d}.png'), frame(2 * np.pi * i / (FPS * SECS) - np.pi / 2))
        print('frames', FPS * SECS)
