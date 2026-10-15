"""Real vs MuJoCo (nominal) vs Blender. ONE free parameter for Blender: a global exposure k chosen so the
scene-camera board median luminance equals the real one. Everything else is an out-of-sample check."""
import sys, numpy as np, cv2, OpenEXR
B, TAG = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "blender"
CAMS = ("agentview", "robot0_eye_in_hand")
def exr(p):
    f = OpenEXR.File(p); ch = f.channels()
    px = ch["RGBA"].pixels if "RGBA" in ch else np.dstack([ch[c].pixels for c in "RGB"])
    return np.asarray(px, np.float64)[..., :3]
def enc(lin):                                   # linear -> sRGB 8 bit
    lin = np.clip(lin, 0, 1)
    return 255 * np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
lum = lambda x: x @ np.array([0.299, 0.587, 0.114])
real = {c: cv2.imread(f"{B}/real_{c}.png")[..., ::-1].astype(float) for c in CAMS}
muj = {c: cv2.imread(f"{B}/mujoco_{c}.png")[..., ::-1].astype(float) for c in CAMS}
lin = {c: exr(f"{B}/{TAG}_{c}.exr") for c in CAMS}
seg = {c: np.load(f"{B}/seg_{c}.npy") for c in CAMS}
er = lambda mk, k=5: cv2.erode(mk.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
def stats(img, c, real_filter=False):
    bm, rm = er(seg[c] == 1), er(seg[c] == 2, 3)
    b = img[bm]; r = img[rm]
    r = r[r[:, 0] > r[:, 1] + 40]          # same red filter on every column (thin parts + AA)
    if real_filter:
        b = b[b.min(1) > 120]
    L = lum(b)
    return dict(board=np.median(b, 0), red=np.median(r, 0) if len(r) > 20 else None,
                spread=np.subtract(*np.percentile(L, [90, 10])), boardL=np.median(L))
R = {c: stats(real[c], c, True) for c in CAMS}
# exposure: bisection on k so that Blender scene board median luminance == real
lo, hi = 1e-3, 1e3
for _ in range(60):
    k = (lo * hi) ** 0.5
    (lo, hi) = (k, hi) if stats(enc(k * lin["agentview"]), "agentview")["boardL"] < R["agentview"]["boardL"] else (lo, k)
print(f"Blender exposure k = {k:.3f} (the only fitted number)")
Bl = {c: stats(enc(k * lin[c]), c) for c in CAMS}
M = {c: stats(muj[c], c) for c in CAMS}
f = lambda v: "   -   " if v is None else "(" + ",".join(f"{x:5.1f}" for x in v) + ")"
for c in CAMS:
    print(f"\n[{c}]")
    for name, s in (("real", R[c]), ("mujoco", M[c]), (TAG, Bl[c])):
        print(f"  {name:8s} board {f(s['board'])}  red {f(s['red'])}  board p90-p10 {s['spread']:5.1f}")
for name, S_ in (("real", R), ("mujoco", M), (TAG, Bl)):
    print(f"wrist/scene board luminance  {name:8s} {S_['robot0_eye_in_hand']['boardL'] / S_['agentview']['boardL']:.3f}")
tiles = []
for c in CAMS:
    tiles.append(np.hstack([real[c], muj[c], enc(k * lin[c])]).astype(np.uint8))
cv2.imwrite(f"{B}/compare_{TAG}.png", np.vstack(tiles)[..., ::-1])
