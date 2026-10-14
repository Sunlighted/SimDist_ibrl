# 2026-10-13 camera calibration (Threading cell, NEW scene camera)

The external ZED 2i was **replaced**: SN 37597132 → **SN 31011113** (same 2.1 mm model). The wrist
ZED-M (SN 16478414) is unchanged. Mode for both: HD720 @ 30 fps, raw (unrectified) left eye.
Supersedes `../calib_20261010/` for the scene camera; tools and the wrist intrinsics are reused from
there (`../calib_20261010/tools/`, `../calib_20261010/data/wrist_intrinsics.json`).

Cell config: `threading_real_cell_20261013.yaml` (= the 10-10 cell with only `cameras` replaced).

## Files

| path | what |
|---|---|
| `data/run_1013_1757/extrinsics.json` | `T_base_scene` (ZED 2i in `robot0_base`), `T_hand_wrist` (ZED-M in `robot0_right_hand`), OpenCV optical axes. 25 of 27 robot poses (2 had < 10 wrist corners). |
| `data/run_1013_1757/sample_000`, `sample_020` | scene + wrist raw PNG + FR3 joints. |
| `data/SN31011113_factory.conf` | Stereolabs factory calibration of the new camera (downloaded with `ZED_Explorer --dc 31011113`). Its LEFT_CAM_HD / VGA / FHD values are now the `scene` entry of `release/cfgs/sim2real/zed_factory_intrinsics.json`; the old camera is kept there as `scene_SN37597132`. |
| `data/scene_intrinsics_refit.json` | Hand-held refit of the new ZED 2i (27 views, 9 tilted ≥ 35°), kept for reference only. |
| `figs/` | gate C overlays (`tools/overlay_real_scene.py`) and the fovy-50 / 224 / max-square crops of the new view. |

## Numbers

| | value |
|---|---|
| new ZED 2i, factory vs hand-held refit | fx 534.9 vs 540.0, cx 629.4 vs 624.2, cy 346.2 vs 343.7; bootstrap sd of the refit fx 3.6 / cx 4.1 / cy 3.1 px ⇒ not distinguishable; own-PnP 0.36 px (factory) vs 0.30 px (refit) ⇒ **factory used** |
| scene reprojection | median 0.33 px, p90 0.75 px; the board did not move during the run (every view 0.27–0.58 px against the pooled pose) |
| scene pose, leave-one-out | ≤ 1.4 mm / 0.12° |
| wrist pose, leave-one-out | ≤ 1.4 mm / 0.16° |
| board square size (solved) | 42.02 mm |

## ⚠️ The wrist camera mount is drifting

`T_hand_wrist` solved from three sessions with the **same** wrist intrinsics, relative to today's,
in camera axes:

| session | rotation (deg, x y z) | translation (mm) |
|---|---|---|
| run_1010 (10-10) | [-0.39, 0.35, **-1.91**] | [5.5, -8.4, -1.3] |
| run_1010_2327 (10-10 late) | [-0.19, 0.50, **-0.83**] | [0.5, -3.4, -1.6] |
| run_1013_1757 (10-13) | 0 | 0 |

Each session is internally consistent to ~1.5 mm / 0.16°, but the pose moves **monotonically**,
almost entirely as a roll about the optical axis: the camera is very likely turning on its mount
(loosening screw). Tighten / pin the mount, then re-check with ~10 robot poses; until then the
wrist pose is only good to ~1°, and the cell has `randomize: false` for the wrist camera.

## Notes

- The new ZED 2i first enumerated on USB 2 (480M ⇒ VGA only, and the ZED SDK does not list it);
  the fix was the cable / Type-C seating, not the port. Check `.../speed` = 5000.
- The 65.8° crop of the new view shows clutter (balls, bowls) at the lower left that the sim does
  not have; at fovy 50 only the bowl corner enters. Clear it before collecting data.

## 2026-10-14: cell rebuilt on the 10-11 fixes, board re-fitted, layout re-centred

**The cell yaml was regenerated.** Its first version (commit 5e215c5) had been built from the
*committed* 10-10 yaml, which predates the 10-11 fixes, so it still had the Panda hand, the mat
height (0.0207 instead of the 0.0285 work surface), and the inert board offset. It is now
`calib_20261010/threading_real_cell_20261010.yaml` (Robotiq + 45.03°/7 mm mount,
`table_top_dz`, `cam_follow_base_dr`, board pose actually applied) with these changes:

- **Cameras:** the 10-13 `cameras`. The wrist camera's `real_intrinsics` points at
  `../calib_20261010/data/wrist_intrinsics.json`.
- **White board:** checked with the new camera on `object_layout/scene_left.png`. Three edges
  are visible.
  - The 10-11 rectangle, fitted with the *old* camera, already lands within 1.4 / 3.0 / 1.7 px
    rms of them. Two independent cameras agree, so the board has not moved.
  - Centre and yaw were re-fitted in the deployed camera (dy +9.4 mm, dyaw −0.52°). The
    residual is now 0.47–1.24 px rms per side at 1280 px.
- **Object layout:** sim objects rendered in the scene camera and IoU-fitted to the red masks
  of the real ones. Output: `object_layout/layout_fit_sim.json`; figure:
  `figs/layout_fit_policy_views_224.png`.

  | | real (fit) | IoU | new sim range |
  |---|---|---|---|
  | needle | (0.4271, −0.1558), z_rot −100.3° | 0.854 | x[0.352, 0.502] y[−0.206, −0.106] rot[−130, −70] |
  | tripod | (0.5269, 0.1758), z_rot −40.6° | 0.910 (next yaw −130°: 0.809) | x[0.497, 0.557] y[0.146, 0.206] rot[−45.6, −35.6] |

  - The ranges keep their old widths. The real tripod sat 5.4 cm toward −y of the old range.
  - The ~1 cm back-projections in `layout_estimate.json` assumed the mat plane (0.0207). The
    scene ray is oblique, so the height matters: ±7.5 mm of table height moves the needle fit
    by ±1.2 cm in y. `dr_base_height` covers this.
- `env/threading_variants.py`: `Threading_RealCellRobotiq` now defaults to this yaml.

**Open issues**

- **Scene fovy.** At fovy 50, 13 of the 27 corner poses of the needle range touch the right edge
  of the crop; the worst still has 102 px of needle. Fitting the whole range inside needs fovy
  ~62, which costs tripod pixels (1482 → 891). This is a decision for the owner, so the fovy is
  unchanged.
- **Wrist vs scene camera.** The wrist camera puts the needle 12 mm / 8° away from where the
  scene camera does. No common table height reconciles the two views (joint fit: either camera
  can be matched, not both). The board edge in the wrist view agrees to 0.5°, so the wrist
  camera's roll is not the cause. Remaining suspects: wrist translation, or the arm not being
  exactly at `q_last_known` (`joint_tap` had stopped 301 s before the photo). Re-shoot with
  live joints after fixing the mount.

## To check on the real robot (next visit)

Listed by priority. Each item says what to record so the sim side can close it without
another visit.

1. **Fix the wrist mount, then re-solve `T_hand_wrist`.** Tighten or pin the ZED-M mount on the
   Robotiq (see "The wrist camera mount is drifting"). Solve again from ~10 robot poses. Pass:
   leave-one-out ≤ 1.5 mm / 0.2°, and **within 0.3° of the 10-13 roll**. If it still drifts
   between sessions, the mount is not fixed.
2. **Re-shoot the object layout with live joints.** Keep `joint_tap` (or a joint logger)
   running while you take the photos, and save q with the timestamp of each frame. Use the same
   placement and grab both `scene_left` and `wrist_left`. Also take 2–3 more wrist shots from
   different arm poses. This decides whether the 12 mm / 8° wrist-vs-scene needle disagreement
   comes from a stale q or from the wrist translation.
3. **Measure the work-surface height in the base frame.** Touch the TCP (gripper closed, known
   finger-tip offset) to the white board at ≥ 4 spread-out points, and record the robot-reported
   z. Expected: base z 0.0285 (`table_top_dz` 0.0078). This is the largest single lever on the
   layout: in the oblique scene view every 1 mm of height moves the fitted y by ~1.5 mm.
   Touching the board at its corners also gives an independent check of the board pose.
4. **Needle vs the scene crop (only if fovy stays at 50).** The real needle as placed on 10-13
   already touches the right edge of the fovy-50 crop. Either:
   - move the real placement toward the image centre; re-shoot it (item 2) and the sim range
     gets re-centred; or
   - widen the sim fovy to ~62.

   Decide this before collecting any data.
5. **Placement protocol.** Mark the range centres on the board: needle (0.427, −0.156), tripod
   (0.527, 0.176), in base frame. Also mark the extents: needle ±7.5 cm in x, ±5 cm in y,
   yaw ±30°; tripod ±3 cm in x and y, yaw ±5°. Real resets must stay inside these, or update
   the ranges to match what you actually do.
6. **Gripper observation mapping.** Record the Robotiq position reading (0–255 or mm) fully
   open, fully closed on nothing, and closed on the needle. The sim obs uses the 2 driver joints
   (state 37 / prop 9). The deploy side needs the real → sim map, and "open at reset" must match.
7. **Home / reset pose.** Confirm the real reset q equals `q_last_known` in
   `object_layout/layout_estimate.json`. Sim resets around it with joint σ 0.1. Note any
   difference.
8. **Colour / lighting (still open).** Fix the ZED exposure, gain and white balance (auto off),
   and write the values down. Take one scene and one wrist frame with the lights as they will be
   during deployment. Also take one frame with the room lights off, to see the light-strip
   contribution. The sim side then does render → measure → fix (CLAUDE.md iron rule 28). Do not
   tune albedos against auto-exposed frames.
9. **Capture mode / frame rate.** Write down the ZED resolution and fps used by the policy
   runner. HD720 matches the intrinsics in `zed_factory_intrinsics.json`. HD2K is 15 fps only,
   and the policy runs at 20 Hz (iron rule 37).
10. **Housekeeping.**
    - Clear the clutter (balls, bowls) at the lower left of the scene view.
    - Check that both cameras enumerate at USB 3 (`.../speed` = 5000).
    - Make sure the next calibration yaml is built from the *current* cell yaml in this folder,
      not from git HEAD.
