"""Assemble the Iris glasses frame STLs (exported flat for printing) into one positioned model.

    python assemble_glasses.py PARTS_DIR OUT_DIR
PARTS_DIR holds the printable STLs (front.step.stl, Frame_Back_v5.stl.stl, Left/Right Leg, leg covers, ear covers).
Placement comes from geometry fits (outline overlap, hinge-housing width, template matching), not hand offsets.
Writes OUT_DIR/glasses_assembly.stl and OUT_DIR/parts/*.stl.
"""
import os, sys, struct, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from stl import load
IN = os.path.join(sys.argv[1], "")
OUT = sys.argv[2]
os.makedirs(os.path.join(OUT, "parts"), exist_ok=True)
def save_stl(path, parts):
    tris = np.concatenate([t for t, _ in parts]).astype(np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    with open(path, 'wb') as f:
        f.write(b'Iris glasses assembly'.ljust(80, b' ')); f.write(struct.pack('<I', len(tris)))
        rec = np.zeros(len(tris), dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
        rec['n'] = n; rec['v'] = tris.reshape(-1, 9); f.write(rec.tobytes())
def flipwind(t): return t[:, [0, 2, 1]]          # keep normals outward after a mirror
report = []
# ---------- 1. back frame: plane (u,v) = (-y, x), depth Y = z (hinge housings extend backward) ----------
B = load(IN + 'Frame_Back_v5.stl.stl').astype(np.float64)
u, v, w = -B[..., 1], B[..., 0], B[..., 2]
cu, cv = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
Bw = np.stack([u - cu, w, v - cv], -1)                # X, Y(depth, + = back), Z
# up/down from the nose bridge: in the middle column the bridge material should sit in the upper half
mid = np.abs(Bw[..., 0]).max(1) < 6
zsign = 1.0 if Bw[mid][..., 2].mean() > 0 else -1.0
Bw[..., 2] *= zsign
if zsign < 0: Bw = flipwind(Bw)
report.append(f'frame: bridge centre z={Bw[mid][...,2].mean():+.1f} (up), zsign={zsign:+.0f}')
# ---------- 2. front plate: plane (u,v) = (x, y) centred, sits in front (Y<0) ----------
F = load(IN + 'front.step.stl').astype(np.float64)
fu, fv, fw = F[..., 0], F[..., 1], F[..., 2]
Fw = np.stack([fu - (fu.min() + fu.max()) / 2, -(fw - fw.min()) - 0.0, (fv - (fv.min() + fv.max()) / 2) * zsign], -1)
Fw = flipwind(Fw) if zsign > 0 else Fw              # one axis negated (Y) -> mirror
# check the front plate's bridge is up too; if not flip its Z
mid = np.abs(Fw[..., 0]).max(1) < 6
if Fw[mid][..., 2].mean() < 0: Fw[..., 2] *= -1; Fw = flipwind(Fw)
report.append(f'front: bridge centre z={Fw[mid][...,2].mean():+.1f}, Y {Fw[...,1].min():.1f}..{Fw[...,1].max():.1f}')
# ---------- hinge housings: parts of the frame with depth Y > 8 ----------
deep = Bw[Bw[..., 1].min(1) > 8]
hx = deep[..., 0].reshape(-1)
L_h = deep[deep[..., 0].mean(1) > 0].reshape(-1, 3); R_h = deep[deep[..., 0].mean(1) < 0].reshape(-1, 3)
for name, h in (('left(+X)', L_h), ('right(-X)', R_h)):
    report.append(f'housing {name}: X {h[:,0].min():.1f}..{h[:,0].max():.1f}  Y ..{h[:,1].max():.1f}  Z {h[:,2].min():.1f}..{h[:,2].max():.1f}')
# ---------- 3. legs: leg z (0..117, hinge box at top) -> world +Y behind housing; leg y -> Z with the ear end DOWN ----------
def place_leg(fname, hous, side):
    T = load(IN + fname).astype(np.float64)
    lx, ly, lz = T[..., 0], T[..., 1], T[..., 2]
    # first render showed the bend at the hinge: the small connector at z=min is the HINGE, the box at z=max is the ear pod
    Y = (lz - lz.min())                                   # hinge end at Y=0 of the leg
    top = lz.mean(1) < lz.min() + 15; ear = lz.mean(1) > lz.max() - 25
    s = 1.0 if ly[ear].mean() < ly[top].mean() else -1.0   # ear end must be lower than the hinge end
    Z = s * ly
    X = lx
    W = np.stack([X, Y, Z], -1)
    if s * 1.0 * (-1) > 0: pass
    # mirrors: Y = +lz (none), Z = s*ly (one if s<0)
    if (s < 0): W = flipwind(W)
    # align: leg hinge-box centre (X,Z) to housing centre; leg front (Y=0) to housing back face
    hb = W[(W[..., 1].max(1) < 15)].reshape(-1, 3)
    hc = np.array([(hous[:, 0].min() + hous[:, 0].max()) / 2, hous[:, 1].max(), (hous[:, 2].min() + hous[:, 2].max()) / 2])
    off = np.array([hc[0] - (hb[:, 0].min() + hb[:, 0].max()) / 2, hc[1], hc[2] - (hb[:, 2].min() + hb[:, 2].max()) / 2])
    W = W + off
    report.append(f'{fname}: ear-down sign {s:+.0f}; placed X {W[...,0].min():.1f}..{W[...,0].max():.1f} Y {W[...,1].min():.1f}..{W[...,1].max():.1f} Z {W[...,2].min():.1f}..{W[...,2].max():.1f}')
    return W
Lw = place_leg('Left Leg.STL.stl', L_h, +1); Rw = place_leg('Right Leg.STL.stl', R_h, -1)
# ---------- 4. covers: template-match the flat cover outline against the leg's outer side silhouette (Y,Z plane) ----------
RES = 0.25
def sil(pts2, lo, shape):
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, list(((pts2 - lo) / RES).astype(np.int32)), 255); return m
def place_cover(fname, legW, outer_sign, region=None):
    C = load(IN + fname).astype(np.float64)
    leg2 = legW[..., [1, 2]]; lo = leg2.reshape(-1, 2).min(0) - 2; hi = leg2.reshape(-1, 2).max(0) + 2
    shape = (int((hi[1] - lo[1]) / RES) + 1, int((hi[0] - lo[0]) / RES) + 1)
    LS = sil(leg2, lo, shape)
    if region is not None:   # restrict the search to part of the arm (Y range in mm)
        keep = np.zeros_like(LS); c0, c1 = int((region[0] - lo[0]) / RES), int((region[1] - lo[0]) / RES); keep[:, max(0, c0):max(0, c1)] = 255; LS = LS & keep
    best = None
    c2 = C[..., [0, 1]]
    for ang in range(0, 360, 5):
        a = np.deg2rad(ang); Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for mir in (1, -1):
            p = (c2 * [mir, 1]) @ Rm.T
            plo = p.reshape(-1, 2).min(0); tmpl = sil(p, plo, (int(np.ptp(p[..., 1]) / RES) + 1, int(np.ptp(p[..., 0]) / RES) + 1))
            if tmpl.shape[0] >= LS.shape[0] or tmpl.shape[1] >= LS.shape[1]: continue
            r = cv2.matchTemplate(LS, tmpl, cv2.TM_CCORR_NORMED); _, sc, _, loc = cv2.minMaxLoc(r)
            if best is None or sc > best[0]: best = (sc, ang, mir, loc, plo)
    sc, ang, mir, loc, plo = best
    a = np.deg2rad(ang); Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = (c2 * [mir, 1]) @ Rm.T
    p = p - plo + lo + np.array(loc) * RES                 # into world (Y,Z)
    zc = C[..., 2] - C[..., 2].min()
    # sit on the arm surface directly under the cover footprint (not the arm's widest point)
    cy, cz = legW[..., 1].mean(1), legW[..., 2].mean(1)
    under = (cy >= p[..., 0].min()) & (cy <= p[..., 0].max()) & (cz >= p[..., 1].min()) & (cz <= p[..., 1].max())
    src = legW[under] if under.any() else legW
    xface = src[..., 0].max() if outer_sign > 0 else src[..., 0].min()
    X = xface + outer_sign * zc
    W = np.stack([X, p[..., 0], p[..., 1]], -1)
    if (mir < 0) ^ (outer_sign < 0): W = flipwind(W)
    report.append(f'{fname}: match score {sc:.3f} at rot {ang} deg mirror {mir:+d}')
    return W
parts = [(Bw, 'frame'), (Fw, 'front'), (Lw, 'left leg'), (Rw, 'right leg'),
         (place_cover('Left leg Cover.STL.stl', Lw, +1, (35, 125)), 'left leg cover'), (place_cover('Right Leg cover.STL.stl', Rw, -1, (35, 125)), 'right leg cover'),
         (place_cover('Left Leg ear cover.STL.stl', Lw, +1, (105, 170)), 'left ear cover'), (place_cover('Right Leg ear cover.STL.stl', Rw, -1, (105, 170)), 'right ear cover')]
save_stl(os.path.join(OUT, 'glasses_assembly.stl'), parts)
for i, (t, n) in enumerate(parts): save_stl(os.path.join(OUT, 'parts', f'{i}_{n.replace(" ", "_")}.stl'), [(t, n)])
print('\n'.join(report))
