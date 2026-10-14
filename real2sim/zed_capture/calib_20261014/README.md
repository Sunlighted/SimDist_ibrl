# 2026-10-14 robot visit (Threading cell, Robotiq): everything the sim needs from the real cell

Builds on `../calib_20261013/` (same cameras: scene ZED 2i SN 31011113, wrist ZED-M SN 16478414,
HD720 raw left eye). Cell config: **`threading_real_cell_20261014.yaml`** = the 10-13 cell (git 2d57392)
with **four** measured changes, each documented where it is set in the yaml:

1. `cameras.agentview.quat`: the ZED 2i was bumped (0.30°, pos unchanged);
2. `gripper_mount_z` 0.007 → 0.0135: the real Robotiq is 6.5 mm longer than the sim's;
3. `layout_needle_*`: range re-centred on (0.55, −0.12, −100°) so it fits the fovy-50 scene crop;
4. `init_qpos`: the DROID home `[0, −π/5, 0, −4π/5, 0, 3π/5, 0]` (Robotiq vertical and square).

⚠️ `env/threading_variants.py` still defaults `Threading_RealCellRobotiq` to the **10-13** yaml; select this one with
`THREADING_ROBOTIQ_CELL_YAML=<path>` or change the default (first item of the sim checklist below).

## Status of the real side (all robot-side items of `../calib_20261013/README.md`)

| item | status |
|---|---|
| 1 wrist mount / `T_hand_wrist` | mount checked (the old drift was the Robotiq turning on the flange; it no longer turns). `T_hand_wrist` **kept from 10-13**; its translation is likely ~1 cm off (see Conclusions) |
| 2 layout with live joints | done (`data/layout_shot_000`, live q) |
| 3 work-surface height | done: touch-off 0.0287 ± 0.0012 = the cell's 0.0285; table tilt 0.60° / 0.27° |
| 4 needle vs the fovy-50 crop | done: placement and sim range moved to (0.55, −0.12, −100°), fovy stays 50 |
| 5 placement protocol | range centre + extents drawn live by `real2sim/deploy/live_crop_view.py --needle 0.55,-0.12,-100` |
| 6 Robotiq → sim gripper map | done: sim driver ≈ 0.83 × real knuckle; ⚠️ the sim Robotiq cannot close below ~14 mm |
| 7 home pose | done: DROID home, real + sim |
| 8 exposure / gain / WB | done: locked (275 / 200 / 3300), reference frames in `data/exposure_locked/` |
| 9 capture mode / fps | done: HD720 @ **30 fps everywhere** (teleop was 60). ⚠️ the owner runs the policy at **10 Hz**, the sim is 20 Hz |
| 10 housekeeping | USB3 checked on both cameras; ⬜ the bowl / balls are still at the lower left of the scene view |

## Sim-side checklist (do these before regenerating experts / demos / stage-1)

1. **Use the 10-14 cell.** Point `REAL_CELL_ROBOTIQ_YAML` in `env/threading_variants.py` at
   `calib_20261014/threading_real_cell_20261014.yaml` (and check `scripts/test_sim2real.py`'s Robotiq-cell asserts,
   which were written against the 10-13 values).
2. **Control rate.** The owner runs the real policy at 10 Hz (20 Hz was unstable). robosuite `control_freq`, the
   MimicGen generation, the world model's one-step dynamics, the teleop recorder (`run_teleop_session.py --rate`,
   default 20) and the policy runner must all use the same rate (iron rule 37: a 20 Hz model deployed at 10 Hz is a
   persistent 2× dynamics error). Decide first; everything below is regenerated anyway.
3. **Placement sampler.** In a quick check, 300 `reset()`s of ONE `Threading_RealCellRobotiq` instance kept the
   needle / tripod pose fixed; only a new instance changed it. Verify this is intended (fixed per-env layout schedule?)
   before generating data, otherwise every episode of a worker sees the same layout.
4. **Robotiq grasp.** The robosuite Robotiq85 stalls at ~0.55 rad with ~14 mm between the pads (the real one closes
   to 0). The 10 mm needle shaft cannot be pinched in the sim; the 40 mm handle can (the owner grasps the handle on
   the robot). Check where the MimicGen expert grasps. Obs map for deploy: sim driver = 0.83 × real knuckle,
   duplicated into both driver joints; the real gripper keeps closing until it meets something.
5. **Home pose.** `init_qpos` is now the DROID home. Experts / source demos built from the Franka-ready pose (and
   anything built on the Panda hand) are invalid. At this pose the wrist camera does not see the needle (it is
   behind the fingers), so the first frames carry needle information only in the scene view.
6. **Needle range** x [0.475, 0.625], y [−0.17, −0.07], z_rot [−130, −70]: re-check reachability / MimicGen
   feasibility with the tripod range (unchanged: centre (0.527, 0.176), z_rot [−45.6, −35.6]).
7. **Wrist camera DR.** The wrist hand-eye translation is probably ~1 cm off (it puts the needle 12 mm / 7° away
   from the validated scene camera) and the wrist has `randomize: false`. Consider ±1 cm position DR on the wrist
   camera only, or re-solve it with ≥ 15 poses on the next visit.
8. **Table tilt.** Measured 0.60° about y / 0.27° about x relative to the base; `dr_base_tilt_deg` 1° covers it,
   a nominal tilt could be added.
9. **Colour.** Render this cell and match it to `data/exposure_locked/` (render → measure → fix, iron rule 28). The
   real exposure is fixed now, so no auto-exposure DR is needed beyond small gain jitter.
10. **Deploy.** `policy_server.py`'s `ReferenceArm` is PegInsertion-only; a Threading policy needs the Robotiq cell
    (gripper map above, DROID home, the chosen rate). The Rectifier already takes the fovy and the wrist
    `real_intrinsics` from the cell. Cameras arrive locked at HD720@30 from `publish_zed_uvc.py`.

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
| needle range vs the fovy-50 scene crop | 10-13 centre (0.427, −0.156): range corners up to **33 px @224 outside** the crop (needle cut at the right edge). Feasible centres need x ≥ 0.51 | needle range re-centred on **(0.55, −0.12, −100°)**, widths unchanged (±75 / ±50 mm / ±30°): every corner ≥ 8 px inside the scene crop (the wrist sees it at the OLD home only; not at the DROID home) |
| real needle as placed | scene-camera fit (sim NeedleObject silhouette, IoU 0.77): **(0.553, −0.108)**, z_rot −101…−108° | — |
| wrist vs scene on the needle | the wrist (3 arm poses, live joints, IoU 0.86–0.91) puts the needle 12 mm / 7° away from the scene. NOT stale joints (live q this time), NOT the needle size (scaling the model drops IoU to 0.5–0.7), NOT the table height (0.0285 is now measured). The scene is validated by the fingertips ⇒ **the wrist hand-eye translation is ~1 cm off**. Today's 7-view wrist solve makes it worse (24 mm) | ⚠️ wrist has `randomize: false`; consider ±1 cm wrist position DR, or re-solve the wrist with ≥ 15 poses |

## Process notes (traps)

- `run_teleop_session.py` defaults to `--gripper franka_hand`: the Robotiq was never driven by teleop.
  Use `--gripper robotiq --gripper-config example_fr3_config_robotiq.yaml`. The Robotiq driver is now
  installed on the NUC (`/workspace/ros2/nuc_setup/install_robotiq.sh`, into `/opt/ros/humble/franka`;
  rosdep also apt-installed ros-humble-ros2-control / joint-limits / transmission-interface).
- The teleop session starts its own camera publisher (`publish_zed_uvc.py`), which holds both ZEDs; pass an
  empty `--cameras` to keep them free for the viewer/recorder. Its default was 60 fps; now 30 (see Exposure).
- Standalone ROS commands in the container need `CYCLONEDDS_URI=file:///workspace/ros2/cyclonedds_ws.xml`
  (the session sets it), otherwise nothing on the NUC is visible.
- Full teleop command now: `run_teleop_session.py --no-record --kill-stale --gripper robotiq
  --gripper-config example_fr3_config_robotiq.yaml --cameras` (it resets to the DROID home first).
- `joint_tap.py` dies with the teleop session; a "joints STALE" refusal means it is not running.
- `real2sim/deploy/live_crop_view.py` (new): live 224 policy crops of both cameras through the deploy
  `Rectifier`, red-object EDGE flags, `--needle x,y,zrot` target overlay, SPACE = raw frames + live q.
  Run it with `~/anaconda3/bin/python` (the simdist env's OpenCV is headless).
- A pure-rotation check (`tools/scene_rot_delta.py ref.png new.png`) is enough to re-pose a bumped
  scene camera without a board, as long as the residual stays at ~0.5 px.

## Files

| path | what |
|---|---|
| `threading_real_cell_20261014.yaml` | the cell (4 changes vs 10-13, documented in its header) |
| `data/home_droid/` | DROID home, needle on the target: scene + wrist raw, 224 policy view, q |
| `data/robotiq_gripper_log.json` | real Robotiq knuckle + GELLO target, open / closed / on the needle |
| `data/layout_shot_000/` | needle at the target, arm at home: scene + wrist raw PNG, 224 policy view, live q |
| `data/touch_shot_006/` | one touch-off pose (scene raw + q) |
| `data/all_shots_joints.json` | live q of all 9 shots (0–3 layout, 4–8 touch-off) |
| `data/scene_after_bump/scene_after.png` | the scene view after the bump (reference for the next check) |
| `data/run_1014_1552/` | the 7-view wrist check (`extrinsics.json` is NOT used) |
| `tools/gello_patch/` | `zed_controls.py` + `camera_controls.json` (locked capture settings) and the diffs applied to the gello `publish_zed_uvc.py`, `run_teleop_session.py`, `reset_arm.py`, `collect_demos.py` (gello is an upstream clone: re-apply after a pull) |
| `data/exposure_locked/` | reference scene + wrist frames with the locked settings, `summary.json` |
| `tools/` | `scene_rot_delta.py`, `snap_layout.py`; needle silhouette fit `geo.py` + `fit.py`; touch-off `touch.py`, fingertip projection `tipcheck.py` (scratch scripts: paths point at `~/simdist/calib/`) |
| `figs/` | fingertip projection, needle target overlay, 224 policy views of the layout shots, wrist silhouette across sessions |

Raw runs: `~/simdist/calib/{run_1014_1552, layout_1014, layout_1014b, scene_bump_1014}` on the workstation.

## Exposure / gain / white balance locked (item 8)

Deploy lighting = the room lights (LED). Both ZEDs now capture with FIXED settings,
`exposure_raw 275` (≈ 8.3 ms at 30 fps), `gain_raw 200`, white balance manual `3300`, at
**HD720 @ 30 fps everywhere** (teleop default changed 60 → 30).

| finding | consequence |
|---|---|
| Exposure/gain are not V4L2 controls on the ZED; they are sensor registers behind Stereolabs' UVC extension unit | `tools/gello_patch/zed_controls.py` (port of zed-open-capture's register access) |
| The registers answer only while streaming; for the first ~3 frames after an open they read 0, then the sensor's own init writes exposure ≈ 771 raw (over-exposed) | `apply(device, grab=...)` waits for the init, writes, streams 4 frames, verifies, retries. Must run after **every** open |
| A USB replug turns AEC back on | the publisher re-applies on every open anyway |
| Auto-exposure had chosen exposure 249 / gain 200–208 here; stable when the view is static, but it re-adapts when the content changes (gripper / arm in view) | fixed 275 / 200 |
| No light flicker: frame-to-frame mean sd 0.01–0.3 % even at 2 ms exposure | exposure needs no mains-period constraint |
| `white_balance_temperature` is coarse: 3000–3300 one image (cool, R/G 0.85–0.91), 3400–3500 another (warm, R/G 1.08–1.12) | 3300; the sim colour match happens afterwards (render → measure → fix), not in the camera |
| ⚠️ writing `white_balance_red_component` hung the ZED-M stream (uvcvideo `Failed to set UVC probe control: -71`) until a USB replug | never touch the `*_component` controls |
| Raw exposure units depend on the fps (max 1100 @30, 880 @60) | `camera_controls.json` carries its fps; `apply()` refuses another fps |
| Repeatability: 4 open→lock cycles per camera, white-board BGR within 0.6 grey levels; scene (227,211,196), wrist (195,196,170); < 0.2 % of the frame clipped | `data/exposure_locked/` = reference frames + `summary.json` |

Where it is applied (the gello repo is an upstream clone, so the changes are kept here as files/diffs in
`tools/gello_patch/`):
- `publish_zed_uvc.py`: new `lock_controls: bool = True` → `zed_controls.apply()` on every open (teleop recording and
  the deploy `robot_client` both read its topics);
- `run_teleop_session.py`: `camera_fps` 60 → 30;
- `real2sim/deploy/live_crop_view.py`: locks by default (`--auto_exposure` to opt out).
- `~/simdist/calib/record_calib.py` / `snap_layout.py` still capture on auto (corner detection does not care).

## Robotiq position → sim gripper joint (item 6)

Real: `/gripper/joint_states` (`robotiq_85_left_knuckle_joint`, rad; the PickNik driver maps the position
register linearly onto it), teleop with `--gripper robotiq --gripper-config example_fr3_config_robotiq.yaml`.
Log: `data/robotiq_gripper_log.json` (pos + GELLO target `/gripper/gripper_client/target_gripper_width_percent`).

| state | real knuckle (rad) | GELLO target width | sim driver joint (Threading_RealCellRobotiq) |
|---|---|---|---|
| fully open | **0.000** | 1.00 | 0.000 (pad gap ≈ 79 mm) |
| grasping the needle by its 40 mm handle | **0.440** (≈ 56 % of stroke ≈ 38 mm) | 0.215 | ≈ 0.37 gives a 40 mm pad gap |
| closed on nothing | **0.786** (pads touch) | 0.20 (trigger fully pressed) | **stalls at 0.52–0.556 with a 14–21 mm gap** |

- Map by finger opening, not by angle: anchors (open, 40 mm handle) ⇒ **sim ≈ 0.83 × real** (0 → 0, 0.440 → 0.365).
  The real side reports ONE knuckle; the sim obs has the 2 driver joints ⇒ feed the mapped value twice.
- The GELLO trigger tops out at a 0.20 target but the gripper keeps closing until it meets something, so the
  real gripper state is set by what it holds, not by how far the trigger is pressed.
- ⚠️ The robosuite Robotiq85 does not close: it stalls at ~0.55 rad with ~14 mm between the pads (the real one
  closes fully at 0.786). Anything thinner than ~14 mm (the 10 mm needle shaft) cannot be pinched in the sim.
  Fine if the sim expert grasps the 40 mm handle (as the owner did on the robot); check the MimicGen grasp.

## Home pose → DROID home (item 7)

`[0, −π/5, 0, −4π/5, 0, 3π/5, 0]` = DROID's `reset_joints` (`~/droid/droid/robot_env.py:27`, used on this
same FR3 + Robotiq for the pi0.5-DROID work). Owner's choice over the Franka ready pose
`[0, −π/4, 0, −3π/4, 0, π/2, π/4]`, whose j7 = π/4 was meant for the Franka Hand and left the Robotiq turned 45°.
- Real: reached via `run_teleop_session.py --start-pose ...` (its built-in reset; joints within 0.01 rad); the owner
  confirmed the Robotiq is square. `START_POSE` changed in `run_teleop_session.py`, `reset_arm.py`, `collect_demos.py`
  (diffs in `tools/gello_patch/`).
- Sim: cell `init_qpos` = the same; a reset lands within ~0.03 rad of it (joint noise).
- At this pose the **wrist camera does not see the needle**: the range centre projects to (82, 193) of the 224 wrist
  crop, behind the fingers (`data/home_droid/policy_view.png`, needle on the target). Tilting j6 +0.15 / +0.30 would
  lift it to row 146 / 101; the owner kept the plain DROID pose (gripper vertical), the scene camera localises the needle.

## ⚠️ Control rate: the owner runs the policy at 10 Hz

20 Hz was too fast on the real robot; 10 Hz is stable. The sim (robosuite `control_freq`), the world model's
one-step dynamics and the teleop recorder (`run_teleop_session.py --rate`, default 20) are all at 20 Hz today.
Deploying a 20 Hz-trained policy at 10 Hz is a persistent 2× dynamics error (CLAUDE.md iron rule 37).
**Decide the rate before regenerating experts / data / stage-1**, and use the same rate in the sim, the recorder
and the policy runner.

## Still open on the real side

- Clear the bowl / balls at the lower left of the scene view before collecting data.
- Next visit: re-solve the wrist hand-eye with ≥ 15 poses (translation), if the sim does not add wrist DR.
