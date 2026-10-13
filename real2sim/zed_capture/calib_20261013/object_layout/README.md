# Object layout photo (2026-10-13)

Needle and tripod placed on the real table, arm at the home pose. Both cameras, raw HD720 left eye.
Use it to put the sim objects where the real ones are.

- `scene_left.png`, `wrist_left.png`: raw frames (lossless).
- `policy_view_224_agentview_wrist.png`: what the policy would see (undistort → crop → 224;
  agentview fovy 50, wrist fovy 49.014 with the refitted wrist intrinsics).
- `scene_base_grid.jpg`: robot-base-frame grid drawn on the table plane (z = +0.0207 m, 5 cm
  minor / 10 cm major) with the cell yaml's current sim placement ranges (green = needle,
  orange = tripod).
- `layout_estimate.json`: object positions back-projected onto the table, base frame, ~1 cm.

| | real (base frame) | sim range in the cell yaml |
|---|---|---|
| needle centre | x 0.423, y −0.181 (head 0.488/−0.174, tip 0.358/−0.188) | x 0.36–0.51, y −0.14 – −0.04 |
| tripod centre | x ≈ 0.51, y ≈ 0.19 (front feet 0.528/0.238 and 0.460/0.176) | x 0.49–0.55, y 0.23–0.29 |

Both real objects sit ~4 cm further toward −y than the sim ranges; x is inside for both.
Yaw is not converted: the sim's `z_rot` depends on each model's local axes (the real needle
runs roughly along base −x, head → tip).
