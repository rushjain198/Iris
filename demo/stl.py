import sys, struct, glob, os, numpy as np
def load(p):
    b = open(p, 'rb').read()
    if b[:5] == b'solid' and b'facet' in b[:400]:
        v = [list(map(float, l.split()[1:4])) for l in b.decode('latin1').splitlines() if l.strip().startswith('vertex')]
        return np.array(v, np.float32).reshape(-1, 3, 3)
    n = struct.unpack('<I', b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
    return a['v'].reshape(-1, 3, 3)
if __name__ == '__main__':
    for p in sorted(glob.glob(sys.argv[1] + '/*.stl')):
        t = load(p); v = t.reshape(-1, 3); lo, hi = v.min(0), v.max(0)
        print(f"{os.path.basename(p)[:58]:58s} tris {len(t):6d}  min {np.round(lo,1)}  max {np.round(hi,1)}  size {np.round(hi-lo,1)}")
