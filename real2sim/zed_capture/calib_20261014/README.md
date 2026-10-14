# 2026-10-14 robot visit (Threading cell, Robotiq): wrist check, needle placement, touch-off, scene bump

Builds on `../calib_20261013/` (same cameras: scene ZED 2i SN 31011113, wrist ZED-M SN 16478414,
HD720 raw left eye). Cell config: `threading_real_cell_20261014.yaml` = the 10-13 cell (git 2d57392)
with **three** measured changes (scene quat, gripper length, needle range); everything else is identical.
`env/threading_variants.py` still defaults `Threading_RealCellRobotiq` to the 10-13 yaml; select this
one with `THREADING_ROBOTIQ_CELL_YAML=<path>` or change the default.

## Conclusions

| item | result | cell change |
|---|---|---|
| wrist hand-eye `T_hand_wrist` | roll vs 10-13 +0.22° (pass); translation ~10 mm but only 7 usable views (LOO 6.2 mm) ⇒ inconclusive | **kept 10-13** (owner's decision) |
| wrist camera on the gripper | gripper silhouette in the wrist image vs 10-10 / 10-10 late / 10-13: ≤0.3° (method resolves 0.45° on a synthetic 0.5° rotation) | — |
| ⇒ the old 1.9° wrist "drift" | was NOT the camera turning on the Robotiq: the whole Robotiq (+camera) turned on the flange. Owner checked: it no longer turns. | — |
| Robotiq length | tape, flange → closed fingertip ≈ **174 mm**; sim had 167.5 mm. Touch-off says the same (+6.6 mm) | `gripper_mount_z` 0.007 → **0.0135** |
| work surface height | 5 fingertip touch-offs (Robotiq closed), with the corrected length: base z **0.0287 ± 0.0012** (0.0271–0.0302) | none (cell 0.0285) |
| table tilt vs base | plane through the 5 touches: **0.60° about y (dz/dx), 0.27° about x (dz/dy)**, residual ≤ 0.4 mm | none (`dr_base_tilt_deg` 1°) |
| scene camera bump | re-plugging cables bumped the ZED 2i: pure rotation from static features, 10-13 frames → now **0.30°** (two frames agree to 0.015°), residual 0.45 px ⇒ translation ≲ 1 mm. Method noise: 0.008° between two undisturbed frames | agentview **quat** → `[0.15341, -0.29297, 0.85476, -0.40002]`, pos kept |
| scene validation | FK fingertips (corrected length) projected with the corrected scene pose land on the real fingertips in all 5 touch photos within 1–2 px @1280 (`figs/touch_fingertips_on_scene.jpg`, green = corrected, red = 10-13) | — |
| needle range vs the fovy-50 scene crop | 10-13 centre (0.427, −0.156): range corners up to **33 px @224 outside** the crop (needle cut at the right edge). Feasible centres need x ≥ 0.51 | needle range re-centred on **(0.55, −0.12, −100°)**, widths unchanged (±75 / ±50 mm / ±30°): every corner ≥ 8 px inside; at home the wrist crop sees the centre pose with 18 px margin |
| real needle as placed | scene-camera fit (sim NeedleObject silhouette, IoU 0.77): **(0.553, −0.108)**, z_rot −101…−108° | — |
| wrist vs scene on the needle | the wrist (3 arm poses, live joints, IoU 0.86–0.91) puts the needle 12 mm / 7° away from the scene. NOT stale joints (live q this time), NOT the needle size (scaling the model drops IoU to 0.5–0.7), NOT the table height (0.0285 is now measured). The scene is validated by the fingertips ⇒ **the wrist hand-eye translation is ~1 cm off**. Today's 7-view wrist solve makes it worse (24 mm) | ⚠️ wrist has `randomize: false`; consider ±1 cm wrist position DR, or re-solve the wrist with ≥ 15 poses |

## Process notes (traps)

- `run_teleop_session.py` defaults to `--gripper franka_hand`: the Robotiq was never driven by teleop.
  Use `--gripper robotiq --gripper-config example_fr3_config_robotiq.yaml`. The Robotiq driver is now
  installed on the NUC (`/workspace/ros2/nuc_setup/install_robotiq.sh`, into `/opt/ros/humble/franka`;
  rosdep also apt-installed ros-humble-ros2-control / joint-limits / transmission-interface).
- The teleop session starts its own camera publisher (`publish_zed_uvc.py ... --mode hd720 --fps 60`),
  which holds both ZEDs; pass an empty `--cameras` to keep them free for the viewer/recorder.
  ⚠️ teleop records at **60 fps**, calibration/viewer at 30 (same intrinsics; the deploy fps is open item 9).
- `joint_tap.py` dies with the teleop session; a "joints STALE" refusal means it is not running.
- `real2sim/deploy/live_crop_view.py` (new): live 224 policy crops of both cameras through the deploy
  `Rectifier`, red-object EDGE flags, `--needle x,y,zrot` target overlay, SPACE = raw frames + live q.
  Run it with `~/anaconda3/bin/python` (the simdist env's OpenCV is headless).
- A pure-rotation check (`tools/scene_rot_delta.py ref.png new.png`) is enough to re-pose a bumped
  scene camera without a board, as long as the residual stays at ~0.5 px.

## Files

| path | what |
|---|---|
| `threading_real_cell_20261014.yaml` | the cell (3 changes vs 10-13, documented in its header) |
| `data/layout_shot_000/` | needle at the target, arm at home: scene + wrist raw PNG, 224 policy view, live q |
| `data/touch_shot_006/` | one touch-off pose (scene raw + q) |
| `data/all_shots_joints.json` | live q of all 9 shots (0–3 layout, 4–8 touch-off) |
| `data/scene_after_bump/scene_after.png` | the scene view after the bump (reference for the next check) |
| `data/run_1014_1552/` | the 7-view wrist check (`extrinsics.json` is NOT used) |
| `tools/` | `scene_rot_delta.py`, `snap_layout.py`; needle silhouette fit `geo.py` + `fit.py`; touch-off `touch.py`, fingertip projection `tipcheck.py` (scratch scripts: paths point at `~/simdist/calib/`) |
| `figs/` | fingertip projection, needle target overlay, 224 policy views of the layout shots, wrist silhouette across sessions |

Raw runs: `~/simdist/calib/{run_1014_1552, layout_1014, layout_1014b, scene_bump_1014}` on the workstation.

## Still open (from `../calib_20261013/README.md`)

- Exposure / gain / white balance locked + frames under deploy lighting (item 8) — next.
- Robotiq position register → sim knuckle angle map (item 6): driver now installed.
- Home pose → DROID home with the Robotiq square to the table (item 7 TODO), real + sim together.
- Deploy capture mode / fps (item 9); clutter (bowl) at the lower left of the scene view.
