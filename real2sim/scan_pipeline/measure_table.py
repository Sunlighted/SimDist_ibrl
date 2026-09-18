"""Measure the real work surface (height, footprint, yaw) from the scan5 splat.

Everything in the BOARD frame (ChArUco corner plane at z=0.8 by construction),
so the numbers compose directly with ground_transforms.json's base pose.
Output: cell_layout.json, consumed by the MJCF arena rebuild.
"""
import json, sys
import numpy as np



def load_ply(path):
    """Minimal 3DGS ply reader (standard INRIA field layout, as written by
    train_3dgs.py). Avoids the gaussian_renderer import, which pulls torch and
    trips the login node's thread limit."""
    with open(path, "rb") as f:
        names, count = [], 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(count * 4 * len(names)),
                             dtype=np.float32).reshape(count, len(names))
    col = {n: i for i, n in enumerate(names)}
    get = lambda *k: data[:, [col[x] for x in k]]

    class G:
        pass
    g = G()
    g.xyz = get("x", "y", "z").astype(np.float64)
    g.opacity = 1.0 / (1.0 + np.exp(-get("opacity")))          # stored as logit
    g.scale = np.exp(get("scale_0", "scale_1", "scale_2"))      # stored as log
    return g


GS = "/scratch/yifanr/gsbench"

tf = json.load(open(f"{GS}/scene5_world_transform.json"))
s, R, t = float(tf["scale"]), np.array(tf["R"]), np.array(tf["t"])
g = load_ply(f"{GS}/sfm_scan5/scene.ply")
P = s * (g.xyz @ R.T) + t
op = g.opacity.reshape(-1)
sc = g.scale.max(1) * s
solid = (op > 0.5) & (sc < 0.05)          # drop fog/floaters before measuring
P = P[solid]
print(f"[pts] {len(P)} solid gaussians, z range {P[:,2].min():.3f}..{P[:,2].max():.3f}")

# --- table-top height: densest horizontal slab near the board plane ---------
bins = np.arange(0.60, 1.00, 0.002)
hist, _ = np.histogram(P[:, 2], bins=bins)
# only consider slabs with broad in-plane extent (a table, not a box lid)
cand = []
for i in np.argsort(hist)[::-1][:40]:
    zc = 0.5 * (bins[i] + bins[i + 1])
    sl = P[np.abs(P[:, 2] - zc) < 0.004]
    if len(sl) < 500:
        continue
    span = np.percentile(sl[:, 0], 97) - np.percentile(sl[:, 0], 3)
    span_y = np.percentile(sl[:, 1], 97) - np.percentile(sl[:, 1], 3)
    cand.append((min(span, span_y), zc, len(sl), span, span_y))
cand.sort(reverse=True)
_, z_top, n_sl, sx, sy = cand[0]
print(f"[table] top plane z = {z_top:.4f} ({n_sl} pts, extent {sx:.3f} x {sy:.3f} m)")
print(f"[table] board (ChArUco) plane is {(0.8 - z_top)*1000:+.1f} mm above it")

# --- footprint: oriented bounding box of the slab, yaw in board frame -------
sl = P[np.abs(P[:, 2] - z_top) < 0.006][:, :2]
c = sl.mean(0)
U, S, Vt = np.linalg.svd(sl - c, full_matrices=False)
yaw = float(np.arctan2(Vt[0, 1], Vt[0, 0]))
loc = (sl - c) @ Vt.T                     # into table axes
ext_a = np.percentile(loc[:, 0], 98) - np.percentile(loc[:, 0], 2)
ext_b = np.percentile(loc[:, 1], 98) - np.percentile(loc[:, 1], 2)
ctr_a = 0.5 * (np.percentile(loc[:, 0], 98) + np.percentile(loc[:, 0], 2))
ctr_b = 0.5 * (np.percentile(loc[:, 1], 98) + np.percentile(loc[:, 1], 2))
center = c + np.array([ctr_a, ctr_b]) @ Vt
print(f"[table] footprint {ext_a:.3f} x {ext_b:.3f} m, yaw {np.degrees(yaw):.1f} deg, "
      f"center ({center[0]:.3f}, {center[1]:.3f})")

# --- robot base relative to the table --------------------------------------
gr = json.load(open(f"{GS}/ground_transforms.json"))
T_bw_base = np.array(gr["base"]["T_world_base"])
bp = T_bw_base[:3, 3]
base_local = (bp[:2] - center) @ Vt.T      # table axes, origin at table center
base_z_rel = bp[2] - z_top
# base yaw: the robot's +x axis expressed in table axes
bx = T_bw_base[:3, 0][:2]
base_yaw = float(np.arctan2(*( (bx @ Vt.T)[::-1] )))
print(f"[base] in table frame: x {base_local[0]:+.3f}  y {base_local[1]:+.3f}  "
      f"z {base_z_rel*1000:+.1f} mm vs table top,  yaw {np.degrees(base_yaw):+.1f} deg")

json.dump(dict(table_top_z_boardframe=float(z_top),
               board_plane_above_table_mm=float((0.8 - z_top) * 1000),
               table_extent=[float(ext_a), float(ext_b)],
               table_center_boardframe=[float(center[0]), float(center[1])],
               table_yaw_rad=yaw,
               base_in_table_frame=[float(base_local[0]), float(base_local[1]),
                                    float(base_z_rel)],
               base_yaw_in_table_frame=base_yaw,
               n_slab_points=int(n_sl)),
          open(f"{GS}/cell_layout.json", "w"), indent=2)
print(f"[done] wrote cell_layout.json")
