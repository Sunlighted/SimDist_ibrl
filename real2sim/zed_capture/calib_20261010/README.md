# 2026-10-10 camera calibration (Threading cell, re-mounted cameras)

Both cameras were re-mounted for the Threading sim2real redo (docs/threading-sim2real-camera-plan.md),
and the real gripper is now a **Robotiq 2F-85** (the sim still has the Panda hand).
Mode for both: **HD720 @ 30 fps**, raw (unrectified) left eye.
The cell config built from this calibration is `release/cfgs/sim2real/threading_real_cell_20261010.yaml`.

Only a few representative raw PNGs are committed (lossless, see ../README.md for why not JPEG).
The full runs (73 wrist views, 48 robot poses) are on the robot workstation in `~/simdist/calib/`.

## Files

| path | what |
|---|---|
| `wrist_intrinsics.json` | ZED-M (SN 16478414) hd720 K, D, refitted from 73 views pooled over 3 runs (hand-held board + both robot runs). fx 778.8, cx 648.7, cy 357.7, D [-0.349, 0.171, -0.001, 0.001, -0.045]; bootstrap sd fx 3.5 / cx 2.6 / cy 3.8 px. |
| `run_1010_2327/extrinsics.json` | **Current** extrinsics. `T_base_scene` (ZED 2i in `robot0_base`), `T_hand_wrist` (ZED-M in `robot0_right_hand`), OpenCV optical axes. |
| `run_1010_2327/sample_011`, `sample_019` | scene + wrist PNG + FR3 joints (`state.json`, fr3_joint1..7). |
| `run_1010/extrinsics.json` | First run, **old scene-camera pose** (the ZED 2i was moved afterwards). Its wrist pose used self-refitted intrinsics. |
| `run_1010/sample_000`, `sample_016` | scene PNG + joints for that old pose. |

Gate C overlays (sim table edges / arm outline on the preprocessed real photo):
`figs/sim2real/overlay_scene_20261010/`, made with `scripts/overlay_real_scene.py`.
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
  feeds the policy a mis-rectified image. Use `wrist_intrinsics.json`.
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

Tools are in `real2sim/calib_tools/` (copies of the workstation scripts; they expect the `~/simdist` layout):

```
# wrist intrinsics (hand-held board, no robot), then pool with the robot runs
python record_calib.py --intrinsics --out ~/simdist/calib/intr_XXXX
python solve_intrinsics.py ~/simdist/calib/intr_XXXX ~/simdist/calib/run_YYYY [...]
# extrinsics: board taped to the table, joints from joint_tap.py in the ROS container
python record_calib.py --no_auto --out ~/simdist/calib/run_YYYY --wrist_intrinsics .../intrinsics.json
python solve_calib.py ~/simdist/calib/run_YYYY --wrist_intrinsics .../intrinsics.json [--drop sample_000,...]
```
