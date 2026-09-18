"""Extract sharpest-per-window frames, restricted to KEEP RANGES of the video.

Why ranges: the scene splat must contain NO ROBOT (the sim draws the arm from the
PolaRiS splat, and a baked-in ghost arm is exactly what made the first composites
unusable). The robot is visible only in contiguous stretches of each handheld
sweep, so dropping those stretches removes it from the reconstruction entirely --
no masking, no post-hoc pruning, and the space it occupied simply stays
unobserved (which now renders as the wall colour, not black).

Ranges also drop frames that wandered off the cell (scan5 spends ~13% of its
length on a whiteboard and a monitor in another part of the room): those frames
inflate the scene bounds and steal gaussian budget from the table.

    python extract_frames_ranges.py <video> <outdir> <n_frames> <r0a-r0b,r1a-r1b,...>

Ranges are fractions of the video, e.g. "0.15-0.36,0.51-1.0".
"""
import os
import sys

import cv2
import numpy as np

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ranges = [tuple(float(x) for x in r.split("-")) for r in sys.argv[4].split(",")]
prefix = sys.argv[5] if len(sys.argv) > 5 else "f"

os.makedirs(out, exist_ok=True)
cap = cv2.VideoCapture(src)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# distribute the frame budget across ranges in proportion to their length
lengths = [(b - a) for a, b in ranges]
budget = [max(1, int(round(n * L / sum(lengths)))) for L in lengths]
print(f"{src}: {total} frames, keep {ranges} -> {budget} frames")

kept = 0
for (a, b), k in zip(ranges, budget):
    lo, hi = int(a * (total - 1)), int(b * (total - 1))
    edges = np.linspace(lo, hi, k + 1).astype(int)
    for wi in range(k):
        w0, w1 = edges[wi], edges[wi + 1]
        best, best_v, best_i = None, -1.0, -1
        for fi in np.linspace(w0, max(w1 - 1, w0), min(7, max(w1 - w0, 1))).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            v = cv2.Laplacian(cv2.resize(g, (640, 360)), cv2.CV_64F).var()
            if v > best_v:
                best, best_v, best_i = fr, v, int(fi)
        if best is not None:
            cv2.imwrite(os.path.join(out, f"{prefix}{best_i:06d}.jpg"), best,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept += 1
cap.release()
print(f"wrote {kept} frames to {out}")
