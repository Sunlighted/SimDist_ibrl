"""Decide whether AE/AF is actually LOCKED, which the whole-sweep exposure std cannot.

A locked camera panning a room still changes mean brightness -- that is SCENE
CONTENT, not auto-exposure, so `std < 4` over 80 spread-out frames is a test a
good scan fails. The discriminating signal is between TRULY CONSECUTIVE frames
(1/30 s apart, essentially identical content): locked => |d mean| ~ 0; AE hunting
=> visible steps. Same idea for AF via the Laplacian variance of a fixed window.
"""
import sys, cv2, numpy as np

def check(path, bursts=6, burst_len=25):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    starts = np.linspace(0, max(total - burst_len - 2, 0), bursts).astype(int)
    dm_all, df_all = [], []
    for s in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s))
        means, focs = [], []
        for _ in range(burst_len):
            ok, fr = cap.read()
            if not ok: break
            g = cv2.cvtColor(cv2.resize(fr, (960, 540)), cv2.COLOR_BGR2GRAY)
            means.append(g.mean())
            focs.append(cv2.Laplacian(g, cv2.CV_64F).var())
        if len(means) > 2:
            dm_all += list(np.abs(np.diff(means)))
            f = np.array(focs); df_all += list(np.abs(np.diff(f)) / (f[:-1] + 1e-6))
    cap.release()
    dm, df = np.array(dm_all), np.array(df_all)
    print(f"\n=== {path.split('/')[-1]} ({total} frames; {bursts} bursts x {burst_len}) ===")
    print(f"  consecutive-frame |d brightness|  median={np.median(dm):.3f}  p95={np.percentile(dm,95):.3f}"
          f"   -> AE {'LOCKED' if np.percentile(dm,95) < 1.0 else 'ACTIVE/hunting'}")
    print(f"  consecutive-frame focus |d|/lvl   median={np.median(df):.3f}  p95={np.percentile(df,95):.3f}"
          f"   -> AF {'LOCKED' if np.percentile(df,95) < 0.15 else 'ACTIVE/hunting'}")

for p in sys.argv[1:]:
    check(p)
