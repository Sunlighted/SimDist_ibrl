# Threading sim2real: diagnosis of the failed transfer + camera re-setup plan

Written 2026-10-10. Scope: `Threading_RealCell` (pixel SimDist, r18@224, 2 cameras).
Background: `docs/infra.md` §14–§14f, `docs/real-teleop-stage2.md`,
`docs/real-teleop-requirements-EN.md`, `docs/simdist.md` §48.

> **Status 2026-10-14: the real side of steps 2–3 is done.** Cameras re-mounted and calibrated
> (`real2sim/zed_capture/calib_20261010/`, `calib_20261013/`, `calib_20261014/`), gate C passed, and the
> real cell measured: scene re-pose after a bump, Robotiq length (+6.5 mm), work-surface height and tilt,
> needle range for the fovy-50 crop, DROID home pose, locked exposure at HD720@30, Robotiq → sim gripper map.
> Current cell: `real2sim/zed_capture/calib_20261014/threading_real_cell_20261014.yaml`.
> **Before step 4, work through the sim-side checklist in `calib_20261014/README.md`**: switch the default
> cell, decide the control rate (the owner runs the policy at 10 Hz, the sim is 20 Hz), the placement
> sampler not re-sampling on `reset()`, the sim Robotiq not closing below ~14 mm, wrist-camera DR.

## 1. Why the previous transfer failed

### 1.1 Object placement on the bench did not match the sim — the schedule is stale
- §14f (2026-09-22) **swapped the layout**: ring/tripod moved to +y (camera side),
  needle to the right, `layout_tripod_z_rot_deg = -30`.
- But the schedule sent to the robot side, `release/cfgs/sim2real/threading_real_placement_seed0.csv`,
  and the table in `docs/real-teleop-requirements-EN.md` §5 still describe the
  **pre-swap** layout: needle 21–31 cm *left* of the centreline (y_base ≈ +0.30),
  tripod 8.4 cm to the *right*.
- Re-running `scripts/make_placement_schedule.py` on the current cell gives
  needle y_base ∈ [−0.13, −0.04] and tripod y_base ∈ [+0.24, +0.29] — left and right
  are reversed relative to the CSV.
- The EN doc already records that in the two good episodes the tripod was ~0.3 m
  (in y) away from the sim. A policy trained on the sim layout never saw the
  objects where they were.

### 1.2 The wrist camera was not just un-randomised — its pose is probably wrong
- `figs/sim2real/atm_qc/real_vs_sim_t0.jpg`: at t=0 the sim wrist view shows
  table / floor / fingertips; the real wrist view shows the pegboard and the arm
  body. The content does not match at all.
- The wrist extrinsic rests on the **B@45 hypothesis** ("ee" = flange, plus a 45°
  rotation about z of the robosuite hand frame), which was **never verified with a
  real photo** (§14e, real-teleop-stage2).
- The wrist frames in the training data were re-rendered at a **fixed** pose with no
  DR, while eval jitters both cameras — an asymmetry.
- Even with a correct pose, the ring and needle tip are **0.0% visible** in the wrist
  camera at the moment of insertion (§14e). That mount cannot provide alignment
  information.

### 1.3 The scene camera is too far
The ring hole is only ~5 px at 224² (§14f), while the Threading tolerance window is
~1.8 mm. That resolution cannot resolve the insertion.

### 1.4 The value failure is a downstream symptom, not the root cause
The value and dynamics heads read the **frozen** latent. Items 1.1–1.3, plus the
appearance gap (real frames ≈130 grey vs sim ≈205), make the real images OOD for
the encoder, so the value is garbage. §33 shows value cannot be rescued by
adaptation. **Fix the observations first; do not touch the value.**

## 2. Improvement plan

### A. Move the wrist camera to the back of the hand
- Goal: ring hole and needle tip visible during the last 20% of frames (insertion).
- **Search the pose in sim first, then build hardware.** Search space: translation
  on the back of the hand × pitch/yaw toward the fingertips. The only metric is the
  **segmentation pixel count at the insertion moment**, using the three guards from
  §14f: only the wrapper's `env.reset()`; re-fetch `m, d` after every reset; assert
  the camera pose equals the cell's value before reporting.
- Add a ZED Mini collision box (124×30×26 mm) to the sim and check that the
  insertion approach does not hit the tripod and that the `panda_joint7` margin does
  not get worse.
- Use a rigid 3D-printed mount whose **CAD transform goes straight into the sim** —
  no more guessing which frame "ee" is.
- Acceptance = hand-eye calibration (ChArUco) matching the CAD pose in **both
  translation and rotation** (iron law #27 — the 45° was missed by checking position only).
- Size the wrist DR from **remount repeatability**: remount several times, calibrate
  each, use the spread. Do not guess.
- Re-pick the wrist FOV from the hd720 native 50.02° (the current 45.656° is an HD2K
  carry-over that was never chosen).

### B. Move the scene camera closer
- Currently at base-frame (0.12, 0.50, 0.77), ~0.6–0.7 m from the ring. Moving to
  ~0.35–0.45 m roughly doubles the hole's pixel size.
- Keep the camera **opposite the arm** as seen from the ring (camera now at 149°,
  arm at −30°). The comment in `threading_real_cell_fixedyaw.yaml` notes that the
  optimal tripod yaw is camera-dependent ⇒ **re-sweep visibility and MimicGen
  generation rate** jointly with the new camera.
- Cost: less framing margin — recompute the worst case over the full layout DR.
  Also, the closer the camera, the more pixels a ±11 mm pose error becomes ⇒ the
  mount must be rigid and the extrinsic calibrated more tightly.

### C. A real→sim alignment gate instead of eyeballing
- Put the robot in 3–5 poses with the objects at known positions. Render the sim
  with the real joint angles and object poses; compute the **mask IoU** of the red
  objects and the gripper for each camera.
- Set a threshold: no data collection or training until it passes.
- For every real demo, estimate the object positions from the first frame (red
  segmentation) and reject episodes outside the sim distribution.

### D. Placement and appearance
- Regenerate the placement CSV from the current cell and update EN doc §5.
- Widen the sim layout slightly to cover hand-placement error (~±1 cm, ±5°), with
  the width taken from measured placement error.
- Lock exposure and white balance on the real cameras so the white table sits at
  180–220 grey (EN doc §4b).
- A back-of-hand wrist camera mostly sees table and objects, so background
  (pegboard etc.) shrinks — a side benefit.

### E. An offline gate before touching the robot (nearly free)
Feed the existing real teleop episodes through stage-1 and check:
- the distribution of latent distance to the nearest sim latent;
- whether the value rises monotonically along successful trajectories;
- agreement between the policy head's actions and the teleop actions.

Only go on the robot after these pass. The value failure can be caught here.

## 3. Execution order

1. Sim search for the wrist and scene camera poses, jointly re-sweeping tripod yaw (A, B).
2. Build mounts, install cameras, calibrate intrinsics/extrinsics, pass gate C.
3. Regenerate the placement schedule from the current cell; update the robot-side docs.
4. Re-render existing data with the new extrinsics and add wrist DR via the §14e
   replay contract (no new rollouts) — but first run `scripts/test_rerender.py` to
   confirm this data replays bit-exactly (§15m: `tex_rgb`/shadows were once not
   captured). Then retrain stage-1 with shift-only aug + dynens_h5 (§48 recipe; the
   encoder is frozen after stage-1, so a camera change requires retraining).
5. Pass gate E, then go on the robot.

## 4. Open issues independent of the cameras
- **Frame rate:** ZED HD2K is only 15 fps but training is at 20 Hz (iron law #37).
- **MPPI latency:** 58.8 ms per step, above the 50 ms budget at 20 Hz
  (3 iterations = 36.5 ms, 4 = 44.6 ms).
