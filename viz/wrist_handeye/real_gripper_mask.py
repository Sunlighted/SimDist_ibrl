"""Segment the real gripper from the 14 wrist frames without thresholds.

The camera is bolted to the flange and so is the gripper => the gripper occupies
the SAME pixels in all 14 frames while everything else moves.  Low temporal
variance + dark  =>  gripper.  No hand-labelling, no scene assumptions.
"""
import json, numpy as np, cv2
R2 = "real2sim/zed_capture/scan_run2"
O = "/scratch/yifanr/temp/claude-3144347/-project-6101830-yifanr-SimDist-ibrl/a1b0d322-aaab-41c7-9a4a-93e3923ef768/scratchpad"
W, H = 1280, 720

g = []
for i in range(14):
    im = cv2.imread(f"{R2}/pose_{i:03d}/zedm_left.png", cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (W, H)).astype(np.float32)
    g.append(im / max(np.median(im), 1.0))          # kill per-frame exposure drift
g = np.stack(g)
std, med = g.std(0), np.median(g, 0)
print(f"temporal std: p5 {np.percentile(std,5):.4f}  p50 {np.percentile(std,50):.4f}  p95 {np.percentile(std,95):.4f}")

cand = ((std < np.percentile(std, 12)) & (med < np.percentile(med, 12))).astype(np.uint8)
cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab, stats, cent = cv2.connectedComponentsWithStats(cand, 8)
# the gripper is the big low-variance dark blob touching the bottom edge
best, bi = -1, -1
for k in range(1, n):
    x, y, w, h, a = stats[k]
    if a < 800: continue
    touches_bottom = (y + h) >= H - 3
    score = a * (2.0 if touches_bottom else 0.4)
    if score > best: best, bi = score, k
mask = (lab == bi).astype(np.uint8)
x, y, w, h, a = stats[bi]
print(f"gripper blob: {a} px ({100*a/(W*H):.2f}% of frame), bbox x{x}..{x+w} y{y}..{y+h}, "
      f"centroid ({cent[bi][0]:.1f},{cent[bi][1]:.1f})")
# tip = topmost point of the blob (finger points away from camera, up in image)
ys, xs = np.nonzero(mask)
tip_y = ys.min(); tip_x = xs[ys == tip_y].mean()
print(f"TIP (topmost point of finger) at ({tip_x:.1f}, {tip_y:.1f}) px @{W}x{H}")
np.save(f"{O}/real_gripper_mask.npy", mask)

vis = cv2.cvtColor((med / med.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
vis[mask > 0] = (0, 200, 255)
cv2.drawMarker(vis, (int(tip_x), int(tip_y)), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
cv2.imwrite(f"{O}/real_gripper_mask.png", vis)
