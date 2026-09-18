"""Robust cell measurement: RANSAC the table plane, then express the table and
the robot base in a TABLE frame (z = table normal = true vertical).

Checks explicitly whether the ChArUco board frame's z is vertical -- if the
calibration clipboard was tilted, every "horizontal" statement made in the
board frame is wrong, including the earlier table-height estimate.
"""
import json, sys
import numpy as np

GS = "/scratch/yifanr/gsbench"
sys.path.insert(0, GS)
from plyio import load_ply

tf = json.load(open(f"{GS}/scene5_world_transform.json"))
s, R, t = float(tf["scale"]), np.array(tf["R"]), np.array(tf["t"])
g = load_ply(f"{GS}/sfm_scan5/scene.ply")
P = s * (g.xyz @ R.T) + t
solid = (g.opacity.reshape(-1) > 0.5) & (g.scale.max(1) * s < 0.05)
P = P[solid]

gr = json.load(open(f"{GS}/ground_transforms.json"))
T_bw_base = np.array(gr["base"]["T_world_base"])
bp = T_bw_base[:3, 3]

# region of interest: the work area around base + ChArUco, ignore walls/floor
board_c = np.array([0.20, 0.30])
roi_c = 0.5 * (bp[:2] + board_c)
d = np.linalg.norm(P[:, :2] - roi_c, axis=1)
cand = P[(d < 1.0) & (np.abs(P[:, 2] - bp[2]) < 0.25)]
print(f"[roi] {len(cand)} points around the work area")

rng = np.random.default_rng(0)
best = None
for _ in range(4000):
    i = rng.choice(len(cand), 3, replace=False)
    a, b, c = cand[i]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        continue
    n = n / nn
    if abs(n[2]) < 0.85:              # want a near-horizontal surface
        continue
    if n[2] < 0:
        n = -n
    dist = np.abs((cand - a) @ n)
    inl = dist < 0.004
    if best is None or inl.sum() > best[0]:
        best = (int(inl.sum()), n, float(a @ n), inl)
n_inl, n, d0, inl = best
# refit on inliers
pts = cand[inl]
c0 = pts.mean(0)
U, S, Vt = np.linalg.svd(pts - c0)
n = Vt[2] * (1 if Vt[2][2] > 0 else -1)
print(f"[plane] {n_inl} inliers, normal {n.round(4)}, "
      f"tilt vs board-frame z: {np.degrees(np.arccos(min(1,abs(n[2])))):.2f} deg")
print(f"[plane] height at the board centre: "
      f"{(c0 @ n + (board_c[0]-c0[0])*0 ):.4f} (plane through {c0.round(3)})")

plane_z_at = lambda xy: c0[2] - (n[0]*(xy[0]-c0[0]) + n[1]*(xy[1]-c0[1])) / n[2]
z_board_c = plane_z_at(board_c)
z_base = plane_z_at(bp[:2])
print(f"[height] table under ChArUco centre: {z_board_c:.4f} "
      f"=> ChArUco sits {(0.8 - z_board_c)*1000:+.1f} mm above the table")
print(f"[height] table under the robot base: {z_base:.4f} "
      f"=> base flange is {(bp[2] - z_base)*1000:+.1f} mm vs the table surface")

# table frame: z = n, x = board-frame x projected into the plane
zt = n
xt = np.array([1.0, 0, 0]) - (np.array([1.0, 0, 0]) @ zt) * zt
xt /= np.linalg.norm(xt)
yt = np.cross(zt, xt)
M = np.stack([xt, yt, zt])                      # world -> table rotation

# footprint: all points within 8 mm of the plane, in table axes
h = (P - c0) @ zt
slab = P[np.abs(h) < 0.008]
loc = (slab - c0) @ M.T
q = lambda a, p: float(np.percentile(a, p))
ext_x, ext_y = q(loc[:, 0], 98) - q(loc[:, 0], 2), q(loc[:, 1], 98) - q(loc[:, 1], 2)
ctr = np.array([0.5*(q(loc[:,0],98)+q(loc[:,0],2)), 0.5*(q(loc[:,1],98)+q(loc[:,1],2))])
print(f"[footprint] {len(slab)} slab points, extent {ext_x:.3f} x {ext_y:.3f} m, "
      f"centre offset in table axes {ctr.round(3)}")

base_loc = (bp - c0) @ M.T - np.array([ctr[0], ctr[1], 0])
bx_t = (T_bw_base[:3, 0] @ M.T)[:2]
base_yaw = float(np.arctan2(bx_t[1], bx_t[0]))
print(f"[base] table frame: x {base_loc[0]:+.3f} y {base_loc[1]:+.3f} "
      f"z {base_loc[2]*1000:+.1f} mm, yaw {np.degrees(base_yaw):+.1f} deg")

json.dump(dict(plane_normal=n.tolist(), plane_point=c0.tolist(),
               tilt_vs_boardframe_deg=float(np.degrees(np.arccos(min(1, abs(n[2]))))),
               charuco_above_table_mm=float((0.8 - z_board_c) * 1000),
               base_above_table_mm=float((bp[2] - z_base) * 1000),
               table_extent=[ext_x, ext_y],
               base_in_table_frame=base_loc.tolist(),
               base_yaw_deg=float(np.degrees(base_yaw)),
               n_plane_inliers=int(n_inl)),
          open(f"{GS}/cell_layout.json", "w"), indent=2)
print("[done] wrote cell_layout.json")
