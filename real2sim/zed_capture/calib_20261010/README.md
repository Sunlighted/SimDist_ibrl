# 2026-10-10 camera calibration (Threading cell, re-mounted cameras)

Both cameras were re-mounted for the Threading sim2real redo (docs/threading-sim2real-camera-plan.md),
and the real gripper is now a **Robotiq 2F-85** (the sim still has the Panda hand).
Mode for both: **HD720 @ 30 fps**, raw (unrectified) left eye.
Everything from that day is in this folder: `data/` (calibration + a few raw photos), `tools/`,
`figs/` (gate C overlays) and the cell config `threading_real_cell_20261010.yaml`.

Only a few representative raw PNGs are committed (lossless, see ../README.md for why not JPEG).
The full runs (73 wrist views, 48 robot poses) are on the robot workstation in `~/simdist/calib/`.

## Files

| path | what |
|---|---|
| `data/wrist_intrinsics.json` | ZED-M (SN 16478414) hd720 K, D, refitted from 73 views pooled over 3 runs (hand-held board + both robot runs). fx 778.8, cx 648.7, cy 357.7, D [-0.349, 0.171, -0.001, 0.001, -0.045]; bootstrap sd fx 3.5 / cx 2.6 / cy 3.8 px. |
| `data/run_1010_2327/extrinsics.json` | **Current** extrinsics. `T_base_scene` (ZED 2i in `robot0_base`), `T_hand_wrist` (ZED-M in `robot0_right_hand`), OpenCV optical axes. |
| `data/run_1010_2327/sample_011`, `sample_019` | scene + wrist PNG + FR3 joints (`state.json`, fr3_joint1..7). |
| `data/run_1010/extrinsics.json` | First run, **old scene-camera pose** (the ZED 2i was moved afterwards). Its wrist pose used self-refitted intrinsics. |
| `data/run_1010/sample_000`, `sample_016` | scene PNG + joints for that old pose. |

Gate C overlays (sim table edges / arm outline on the preprocessed real photo):
`figs/`, made with `tools/overlay_real_scene.py` (run from the repo root, MUJOCO_GL=egl).
Both scene poses agree with the sim table to roughly 1–2.5 px at 224.

## Numbers

| | value |
|---|---|
| scene reprojection (run_1010_2327) | median 0.25 px, p90 0.50 px |
| scene pose, leave-one-out spread | ≤ 6.6 mm / 0.56° (sim DR: 11 mm / 1°) |
| wrist pose, leave-one-out spread | ≤ 3.9 mm / 0.59° |
| wrist pose, run_1010 vs run_1010_2327 | **7.1 mm / 1.1° — unresolved** |
| board square size (solved) | 42.21 mm (run_1010_2327) vs 41.75 mm (run_1010) |

## Traps found on the way

- **ZED-M factory distortion is wrong** for this stream (factory k1 −0.10, fitted −0.35; factory fx 771.8, fitted 778.8).
  Anything that undistorts wrist frames with the factory D (`real2sim/deploy/preprocess.py`, the policy server)
  feeds the policy a mis-rectified image. Use `data/wrist_intrinsics.json`.
- **Intrinsics need steep views.** A hand-held set with max tilt 25° fitted at 0.3 px rms with fx anywhere in 743–832
  and cy 352–398 depending on the distortion model. `solve_intrinsics.py` now prints the tilt distribution and a
  model-swap check and warns when fx/cy are not pinned down.
- **Coverage.** The Robotiq fingers fill the bottom ~1/3 of the wrist image; board views from the robot alone
  only covered the upper middle (58% of cells), leaving the distortion extrapolated at the policy-crop corners.
- **The board was bumped** during the steepest poses of run_1010_2327 (42 mm / 3.6° jump seen by the static scene
  camera between samples 003 and 007); samples 000–006 are excluded. Check the per-sample scene board pose
  before trusting a run.
- `board_model.json` (scan_pipeline) is not used: its corners scatter up to 2 mm. The solver uses the ideal
  5×7 ChArUco (DICT_5X5_50) and fits the square size.
- The ZED 2i once enumerated on USB 2 (480M) behind a hub and then only offers VGA; it must be on a direct USB 3 port.

## Reproduce

Tools are in `tools/` (copies of the workstation scripts; they expect the `~/simdist` layout):

```
# wrist intrinsics (hand-held board, no robot), then pool with the robot runs
python record_calib.py --intrinsics --out ~/simdist/calib/intr_XXXX
python solve_intrinsics.py ~/simdist/calib/intr_XXXX ~/simdist/calib/run_YYYY [...]
# extrinsics: board taped to the table, joints from joint_tap.py in the ROS container
python record_calib.py --no_auto --out ~/simdist/calib/run_YYYY --wrist_intrinsics .../intrinsics.json
python solve_calib.py ~/simdist/calib/run_YYYY --wrist_intrinsics .../intrinsics.json [--drop sample_000,...]
```

## 2026-10-11: white board re-fit + Robotiq in the sim

**White board.** It had moved since the 09-23 tape measurement. All four board edges were detected in the
undistorted scene photos (sample_011/019, line rms 0.2–0.3 px) and a rectangle was fitted **on the sim
table-top plane** (base z 0.0207) → `table_visual_half (0.3806, 0.3875)`, `table_visual_offset (0.0181, -0.0334)`,
`table_visual_yaw_deg 0.756`. Edge residual in the sim, full 1280×720 frame: **rms 9–26 px / max 33 px before →
rms 0.3–0.6 px / max 1.7 px after** (`figs/*_full_board_before|after.jpg`, `figs/*_board_refit_fovy50.jpg`).

- ⚠️ **`table_visual_offset` had never reached the render** in any cell: the geom compiles with
  `geom_sameframe=1`, so MuJoCo copied the body pose and ignored `geom_pos`. Every old Threading cell rendered
  the board centred on the collision table. Fixed **opt-in** (`table_visual_apply_pose: 1`, set here) so the
  datasets rendered from the old cells still match their configs.
- ⚠️ **The real board top is ~9 mm above the sim table top** (base z ≈ 0.030 ± 0.005 vs 0.0207: wrist-photo
  edges zero at 0.025 / 0.036, the tape's 753 mm square at 0.027–0.036, the ChArUco at 0.0329 minus the
  clipboard). The sim plane was kept, so the scene camera matches to sub-pixel while the wrist camera keeps a
  6–25 px edge misfit. Moving the table top changes the physics/arena and is NOT done here.

**Gripper.** `gripper_types: Robotiq85Gripper` (new construction-time cell field). robosuite bolts the gripper to
`robot0_right_hand`, which is 45.03° about z from the FR3 flange; the real Robotiq is square to the flange
through a coupling ⇒ `gripper_mount_yaw_deg 45.03`, `gripper_mount_z 0.007`, fitted to the two wrist photos:
finger-mask IoU **0.196 → 0.86** (`figs/run_1010_2327_wrist_robotiq_mount.jpg`, left = robosuite mount).
⚠️ Robotiq85 has 6 gripper joints ⇒ `robot0_gripper_qpos` 2 → 6: experts/demos/stage-1 built on the Panda
hand cannot be used with this cell.

**Crop / distortion.** Offline paths (`scripts/convert_real_teleop.py`, `tools/overlay_real_scene.py`) undistort
with newK=K and crop `2·fy·tan(fovy/2)` about the principal point — matches the sim (rounding ≤ 0.45 px @1280).
The refitted wrist D straightens a long board edge from 6.3 px to 0.29 px rms. ⚠️ `real2sim/deploy/preprocess.py`
`Rectifier` (policy server) crops the LARGEST square instead (scene = fovy 67.0° vs the cell's 50°) and uses the
camera_info D (factory, wrong for the ZED-M); it is peg-only today (`SIM_CAMERA`), so Threading deploy needs it fixed.

## 2026-10-11 (later): work-surface height, Robotiq env class, deploy crop fixed

Supersedes the three ⚠️ above.

**Work surface.** New cell field `table_top_dz` (board on top of the real2sim mat; read via
`Sim2RealConfig.work_surface_above_base()`): `+0.0078` ⇒ work surface at base z **0.0285**. The four estimates
of the board top: wrist edges 0.0357 (sample_011) / 0.0213 (sample_019), whose lines cross at 0.028 with a
common −12 px bias; tape vs fitted size ≈ 0.026; ChArUco on its clipboard ≈ 0.030. `dr_base_height 0.0075`
then covers 0.021–0.036, which includes every one of them. The board rectangle was re-fitted on the 0.0285
plane (`table_visual_half 0.3747/0.3815`, `offset 0.0138/-0.0250`). Measured in the sim
(`Threading_RealCellRobotiq`, DR off):
scene edges rms 0.3–0.6 / max 1.7 px @1280 (unchanged); wrist edges −25.3 / +1.0 px → **−12.8 / −11.2 px**.
The two wrist photos now agree, so what remains is a common wrist hand-eye error (~6 mm / 0.9°, inside the
7.1 mm / 1.1° run-to-run gap). It is NOT the board height, and the wrist camera has `randomize: false`.
A fingertip touch-off on the real board (FK at 3–4 points) would replace the height estimate with a measurement
and give the board tilt too.

**`cam_follow_base_dr: 1` (new, opt-in).** Base-frame cameras used to be placed from the base pose BEFORE this
episode's base DR (`body_xpos` not recomputed): the scene camera stayed fixed in the world while the robot moved,
so `dr_base_height` never showed up in the scene image and camera↔robot jittered by up to ~18 mm. With the flag,
the camera rides with the base, which makes the base DR a table-pose DR in both physics and pixels.

**`Threading_RealCellRobotiq`** (env/threading_variants.py): a registered class pinned to this yaml
(`THREADING_ROBOTIQ_CELL_YAML` overrides it). It exists because MimicGen and robomimic build envs from the
env_name, never through our wrapper. It carries:
- `gripper_types` forced to Robotiq85Gripper (a Panda hand is refused);
- the 45.03° / 7 mm mount written into the MJCF;
- the work-surface height;
- an OPEN gripper at reset (robosuite resets the Robotiq half-closed);
- `robot0_gripper_qpos/_qvel` = the 2 driver knuckles (0 open … ~0.52 closed on nothing, 0.8 max). State stays
  37 and prop stays 9, so PROP_DIM-sliced trainers are untouched. ⚠️ The real side must map the Robotiq position
  register onto these angles.

**Deploy crop fixed.** `Rectifier(K, D, w, h, S, fovy_deg, reduce)` undistorts straight into the sim pinhole:
principal point at the crop centre, focal for exactly the sim fovy, sub-pixel. It replaces the old
largest-square crop plus integer rounding. The policy server takes the fovy from the ckpt's `sim2real_cell`
(task constants for peg) and the K/D from the cell camera's new `real_intrinsics` field (this cell's wrist →
`data/wrist_intrinsics.json`), else camera_info. `scripts/convert_real_teleop.py` calls the same class.
`scripts/test_preprocess_crop.py`:
- projected 3D points land on the sim pinhole's pixel within **0.05 px @224** (both cameras, reduce 1 and 2);
- the converter is byte-identical to the server;
- peg's crop is unchanged (705 vs 704 px).

⚠️ The server's ReferenceArm (FK/prop/OSC model) is still PegInsertion-only and now refuses other tasks.
