"""Pick SFM frames from a handheld sweep: sharpest frame per window.

Uniform sampling wastes budget on motion-blurred frames, and blur is the failure
mode COLMAP cannot recover from (analyze_scan.py measured 12% of scan3 below the
blur floor). So split the video into N windows and keep the SHARPEST frame in
each -- same coverage, strictly better frames.
"""
import sys, cv2, numpy as np, os

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
cap = cv2.VideoCapture(src)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
edges = np.linspace(0, total - 1, n + 1).astype(int)
print(f"{src}: {total} frames -> {n} windows of ~{total//n}")
kept = 0
for wi in range(n):
    lo, hi = edges[wi], edges[wi + 1]
    best, best_v, best_i = None, -1.0, -1
    # subsample within the window; decoding every frame of 3667 is wasteful
    for fi in np.linspace(lo, max(hi - 1, lo), min(7, max(hi - lo, 1))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        v = cv2.Laplacian(cv2.resize(g, (640, 360)), cv2.CV_64F).var()
        if v > best_v:
            best, best_v, best_i = fr, v, int(fi)
    if best is not None:
        cv2.imwrite(os.path.join(out, f"f{wi:04d}_{best_i:06d}.jpg"), best,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept += 1
cap.release()
print(f"wrote {kept} frames to {out}")
