"""Quantify whether a handheld sweep is COLMAP-reconstructable.

Four failure modes, all unrecoverable after capture:
  blur          -> variance of Laplacian (low = motion blur, kills feature matching)
  texture       -> ORB keypoints/frame + flat-pixel fraction (white walls give COLMAP nothing)
  exposure      -> drift in mean intensity (auto-exposure breaks 3DGS photometric consistency)
  motion        -> median optical-flow magnitude (too fast = blur + no overlap; ~0 = wasted frames)
"""
import sys, cv2, numpy as np

def analyze(path, n=80):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 2, 0), n).astype(int)
    orb = cv2.ORB_create(nfeatures=4000)
    blur, kps, means, flat, flow = [], [], [], [], []
    prev = None
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(fr, (960, 540)), cv2.COLOR_BGR2GRAY)
        blur.append(cv2.Laplacian(g, cv2.CV_64F).var())
        kps.append(len(orb.detect(g, None)))
        means.append(g.mean())
        # local std over 16x16 tiles: fraction of the frame with no usable texture
        t = g[:536 // 16 * 16, :960 // 16 * 16].reshape(33, 16, 60, 16).std(axis=(1, 3))
        flat.append(float((t < 4.0).mean()))
        if prev is not None:
            f = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 15, 2, 5, 1.2, 0)
            flow.append(float(np.median(np.linalg.norm(f, axis=2))))
        prev = g
    cap.release()
    b, k, m, fl, fw = map(np.array, (blur, kps, means, flat, flow if flow else [0]))
    print(f"\n=== {path.split('/')[-1]}  ({total} frames) ===")
    print(f"  blur  (lapvar)   median={np.median(b):8.1f}   frac<100 (blurry) = {100*(b<100).mean():.0f}%")
    print(f"  texture (ORB kp) median={np.median(k):8.0f}   frac<300 (weak)   = {100*(k<300).mean():.0f}%")
    print(f"  flat-pixel frac  median={np.median(fl):8.2f}   frac>0.5 (mostly featureless) = {100*(fl>0.5).mean():.0f}%")
    print(f"  exposure  mean={m.mean():6.1f}  std={m.std():5.2f}  range={m.max()-m.min():5.1f}"
          f"   {'LOCKED-ish' if m.std()<4 else 'DRIFTING (auto-exposure?)'}")
    print(f"  motion (px/sample) median={np.median(fw):6.2f}")

for p in sys.argv[1:]:
    analyze(p)
