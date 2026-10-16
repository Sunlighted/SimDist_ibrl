# 2026-10-15 robot visit: tracking-lag test, sim_ref, 10 Hz, wrist re-calibration, touch-offs, first demos

> **Day summary.** §0–§5 done, §6 started (5 demos). x stalled because the server re-seeded the reference at the measured q;
> porting the owner's `sim_ref` fixed it (x 0.1 → 2.7–6.2 cm/s). **Rate decided: 10 Hz** (the sim is retrained).
> The wrist camera moved 6.2 mm / 0.8° since 10-13 and is re-solved; the scene camera did not move. The recorder
> was dropping the Robotiq state (fixed). Sections below are in order of the day.

## (original title) tracking-lag test (NEXT_VISIT.md §0–§1) and what it means for the controller

Plan: `../calib_20261014/NEXT_VISIT.md`. Done today: §0 (sync + smoke) and §1 (tracking test at 20 and 10 Hz).
§2 (control rate) has a recommendation below, but the decision is still open. §3–§6 are not done yet.

## TL;DR

- **Overshoot is not what put the gripper into the table.** It is at most 1.6 mm at either rate.
- **The real arm follows the commanded target like a first-order lag with τ ≈ 100 ms.** The measured joints lag
  the 100 Hz command by 98 ms at 20 Hz and 103 ms at 10 Hz. Per step, the fraction of the motion completed is
  1 − e^(−dt/τ) = 0.39 at 20 Hz and 0.63 at 10 Hz; the z moves measured 0.33–0.45 and 0.48–0.63. The sim
  assumes 1.0.
- **Horizontal moves barely happen: x completes 1–4 % per step.** The cause is how we generate the target,
  not the controller.
  - Our server (`real2sim/deploy/policy_server.py`, `ReferenceArm`) restarts the sim arm at the **measured** q on
    every step. The commanded target therefore stays about 0.007 rad ahead of the arm, which is
    240 × 0.007 ≈ 1.7 N·m on j4. That is below the joint's static friction, so the arm does not move.
  - z works because gravity helps on the way down.
  - **The owner already measured and fixed this on 09-25**, in the docstring of
    `~/simdist/gello_software/SimDist_ibrl/real2sim/deploy/real_policy_server.py:208`: "sim_shadow kept the error at
    ~0.01 rad, ~2 Nm: below friction, so the arm moved ~11 % per step".
  - The fix was `--action_map sim_ref`, which every 09-24/25 Threading eval used (`~/simdist/real_runs/eval/*`).
    It only ever lived in that local copy of the repo, so the 10-15 Robotiq port of `policy_server.py` left it out.
- **Same controller then and now: no polymetis anywhere.** Policy inference and teleop both pause
  `gello_publisher` and stream `/gello/joint_states` at 100 Hz to the franka_ros2
  `franka_fr3_arm_controllers/joint_impedance_controller`.
  - That controller is a pure PD: `tau = K (q_goal − q) − D dq_filt`, with no integral and no friction
    compensation (`src/joint_impedance_controller.cpp:203`).
  - Gains: K = 240 240 240 240 100 60 20, D = 20 20 20 10 10 10 5.
  - The polymetis server running on the NUC today was DROID's, left over from someone else. It held the
    Robotiq serial port and was killed.

| | 20 Hz | 10 Hz | sim |
|---|---|---|---|
| done/step z (amp 0.2 / 0.5 / 1.0) | 1.29 / 0.45 / 0.33 | 0.80 / 0.63 / 0.48 | 1.00 |
| done/step x (amp 0.2 / 0.5 / 1.0) | **0.01 / 0.02 / 0.01** | **0.01 / 0.04 / 0.01** | 1.00 |
| real speed z, amp 1.0 | 2.3 cm/s (cmd 6.9) | 1.9 cm/s (cmd 3.9) | = cmd |
| overshoot (max) | 1.5 mm | 1.6 mm | 0 |
| joint lag vs the 100 Hz command | 98 ms | 103 ms | 0 |
| image age at the client, p50 (ROS receive time, a lower bound) | 7–11 ms | 17–19 ms | — |
| client step latency p50 / p95 / max | 34 / 55 / 82 ms | 35 / 55 / 88 ms | — |

Figure: `figs/track_20_vs_10hz_1015.jpg`. Raw data: `data/track{20,10}_*` (`steps.jsonl`, `joints.npz`,
`tracking_report.json`, client / server logs). Table: `data/summary_1015.txt`.

## Re-test with `sim_ref` (10-15 evening): passes at both rates

Same scripted moves, same stack, `--action_map sim_ref` (default now). Runs: `data/track{20,10}_sim_ref_*`,
figure `figs/track_sim_ref_1015.jpg`, table `data/summary_sim_ref_1015.txt`.

| | 20 Hz reseed (afternoon) | **20 Hz sim_ref** | **10 Hz sim_ref** |
|---|---|---|---|
| real speed x (amp 0.2 / 0.5 / 1.0), cm/s | 0.0 / 0.1 / 0.1 | **4.9 / 4.7 / 6.2** | **2.7 / 5.8 / 5.3** |
| real speed z, cm/s | ~0.5–2.3 | 5.4 / 7.2 / 8.0 | 2.5 / 6.6 / 7.5 |
| real / ref speed (x, z) | — | 0.93–1.06 | 0.96–1.23 |
| steady lag behind the target, x / z | — | 13–19 mm / 6–10 mm | 10–19 mm / 4–12 mm |
| past the final target (real overshoot) | ≤ 1.6 mm | ≤ 7.3 mm (z down, amp 1) | ≤ 6.2 mm (z down, amp 1) |
| speed-capped steps | 0 | 119 / 869 | 24 / 508 |
| lowest TCP / z_min | | 1.169 / 1.091 | 1.166 / 1.091 |

- **x now moves**: about 50× faster than with reseed, and the arm keeps up with the reference (`real/ref` ≈ 1).
- **The lag is now about 0.1 s in z and 0.25–0.4 s in x** (lag mm / speed). The joint fit gives
  222–240 ms (j6). x lags more than z, which fits friction rather than a pure first-order lag.
- **The real overshoot is now ≤ 7 mm**, mostly going down at amp 1. Near the table the policy run uses
  `--z_slow_band 0.06` (≤ 3 mm/step descent), which the scripted test turns off.
- **Rate (§2): recommend 10 Hz, to confirm with the owner.**
  - Both rates are capped at 8 cm/s, so per step the real arm moves about 2.5–4 mm at 20 Hz and 5–7 mm at 10 Hz.
    The sim's 50 ms OSC step moves farther than either, and 10 Hz is the closer match.
  - 10 Hz has slightly less overshoot, has 5× fewer capped steps, and is what the owner ran stably on 09-24/25.
  - Either way, the sim must model what the real arm does: the speed cap, the ~0.1–0.4 s lag and the
    1–2 step lead (Recommendation step 2). Without that, the dynamics gap is persistent, which is the bad kind
    (findings §27).

## Offline re-analysis (10-16): per-joint lag, a static z sag, and humans exceed the 8 cm/s cap

Inputs: the two reseed runs' `joints.npz` (the `track*_sim_ref_*` raw dirs were **not** brought back, only the figure)
and the 5 Robotiq demos (GELLO command `action/joint_positions` vs measured q at 10 Hz is a tracking test at
human speed with no server in the loop). Scripts: `tools/fit_joint_lag_1016.py`, `tools/demo_joint_lag_1016.py`,
`tools/demo_tcp_lag_1016.py` (Panda DH FK, flange + 0.17 m; only mm-scale differences are read from it).

1. **"τ ≈ 100 ms" is joint 4 only.** Delay + first-order fits, cmd → measured:

   | joint | reseed runs (20 / 10 Hz) | 5 demos (delay + τ) |
   |---|---|---|
   | j1–j4 | j4 τ 120 ms; j2 τ 200–240 ms | 60–100 ms + 0–100 ms, consistent over 5 demos |
   | j5, j6 | j6 τ 680–720 ms (pure-delay fit 390 ms) | τ 180–220 ms (j5 one at 420) |
   | j7 | — | τ 320–500 ms, residual 9–18 mrad (stick-slip) |

   Overdamped PD would give τ ≈ D/K = 42 / 83 / 167 ms for j4 / j2 / j6; the excess is friction. The wrist
   joints carry the orientation, which is what threading needs.
2. **At the TCP the demos lag the command by 0.13 s (x) / 0.16 s (y) / 0.13 s (z)** (regression of
   `TCP(q) − TCP(cmd)` on the command velocity, pooled). No x-vs-z difference ⇒ the 0.25–0.4 s "x lag" under
   sim_ref is probably the cap/lead mechanics, not the arm.
3. **Static sag: the measured TCP sits 6.3 mm below the command** (bias of the same regression; x −2.7,
   y −1.5 mm; residual sd 3–5 mm). It comes from j2 at **+9–10 mrad in all 5 demos** regardless of the per-episode
   GELLO offset ⇒ gravity compensation does not carry the Robotiq + ZED-M payload. The sim_ref figure shows the
   same thing at the high test pose (holds settle 2–3 mm below the target), so the "past target ≤ 7 mm" there is
   about half sag, half overshoot.
   - ⚠️ `--z_min` and `--z_slow_band` clamp the **reference** TCP (`policy_server.py:502`, `:545`). The real
     fingertip can be one sag (≈ 6 mm at the needle's reach) plus one overshoot (3–4 mm) below it. Raise `z_min`
     by ~1 cm or clamp on the measured TCP + sag; fixing the payload in the franka config is the clean fix (owner).
   - The sim has no sag. A pose-dependent persistent offset is the harmful kind of model error (findings §27).
4. **Humans move faster than the deploy cap.** Demo TCP speed p50 2.6, p90 12.0, p99 22.5 cm/s; **18 % of steps
   exceed 8 cm/s** (5 cm/s: 31 %). Last 3 cm of descent p90 2.8–4.7 cm/s (under the cap and the 3 mm/step slow
   band). ⇒ data derived from these demos (replay, MimicGen seeds, BC prior) must go through the same cap, or the
   policy learns transport speeds the real arm will not be allowed to execute.
5. Command → measured TCP distance: median 9–11 mm, max 30–62 mm (during fast moves).

## Sim replay of the 5 demos (10-16)

Converted with the ported `scripts/convert_real_teleop.py` (Robotiq cell `threading_real_cell_20261015.yaml`, 10 Hz,
prop from the deploy `ReferenceArm`) to `/user_data/Chenyang/real_stage2/threading_robotiq_1015/dataset.hdf5`, then
`scripts/replay_real_demos_sim.py` in the 10 Hz cell with every DR off. Objects at each demo's own frame-0 fit
(`demo_layout/layout_demos_1015.json`, `figs/fit_demos_frame0_1016.jpg`).

**Track mode** (OSC servo onto the measured grip-site path, gain 2.5): TCP error p50 0.6–1.2 mm, max 9–40 mm.

| demo (episode) | sim | needle in hand, wrist IoU sim/real, centroid Δ | sim tripod pushed | real tripod moved (frame 0 → last) |
|---|---|---|---|---|
| 0 (ep 1) | carried, misses the ring by 21.7 mm (z +20.8) | 0.18, 23 px: **slid ~2 cm in the sim grasp** | 13 mm | 21 mm, 23° |
| 1 (ep 2) | not grasped | — | 0 | 3 mm |
| 2 (ep 3) | needle knocked over | 0.04 | 0 | 5 mm |
| 3 (ep 4) | carried, misses by 19.9 mm sideways | 0.51, 6 px (matches) | 21.5 mm | **42 mm**, 14° |
| 4 (ep 5) | **success** at step 307 | 0.59, 4.5 px (matches) | 13.4 mm | 6.5 mm |

- The sim grasp of the needle handle is marginal: moving the needle 4 mm (layout shot vs the per-demo fit) flips
  demo 1 between lifted and not lifted.
- The operator pushes the tripod while threading (up to 42 mm / 23°) and corrects by eye. An open-loop replay cannot
  reproduce that contact, so a sim failure in the threading phase is not by itself a sim gap.
- **Open mode** with the old `cmd_fk` actions: 0/5, the arm drifts 0.5–0.8 m **up**. They carry the static sag:
  mean a = (+0.070, +0.044, +0.115) = (3.5, 2.2, 5.7) mm while the real TCP goes 12 cm down. Replaced by
  `ref_inverse` (follow-up below); the converter flags `cmd_fk` bias (`[action bias]`).
### Follow-up (10-16, later): the problem is the grasp; sag and the action convention are solved

**Initialisation matches** (`tools/demo_frame_check_1016.py`, `figs/demo_frame_check_1016.jpg`): at frame 0 the
scene needle / tripod IoU is 0.67–0.76 / 0.80–0.82 and the wrist gripper IoU 0.84. **Right before the close**, the
needle in the wrist view is at IoU 0.39–0.54, centroid 4–9 px ≈ 2–5 mm (demo 4: 18 px). The one success (demo 4)
has the largest offset, so a few mm before the close is not what decides the grasp.
- ⚠️ Mask trap: the handle cube in the gripper's shadow is RGB ~(20, 0, 12) and fails `R − max(G, B) > 22`
  (it first read IoU 0.03). The check uses `R − G > 12 and R ≥ B`.
- Object size matches: real needle = 4 cm cube handle, 16 cm overall (owner), sim = 40 mm cube, shaft −80..+40 mm
  = 160 mm. (§4 above calls it "the 2 cm handle cube": that is the half size.) The wrist-view handle widths agree
  (~70 px sim and real).

**The grasp is what fails, and per demo it is deterministic.** Needle perturbed ±2 mm / ±3° (9 variants × 5 demos):

| variant | lifted > 10 cm | demo 0 / 1 / 2 / 3 / 4 |
|---|---|---|
| sim as is | 22/45 | 9 / 2 / 0 / 9 / 2 |
| needle mass ×5 (sim 7.6 g) | 19/45 | 9 / 1 / 0 / 9 / 0 |
| condim 4 on pads + handle | 19/45 | 9 / 0 / 0 / 9 / 1 |
| both | 19/45 | 9 / 0 / 0 / 9 / 1 |

Real: 5/5. Mass and torsional friction are not it. In demos 1 and 4 the sim's inner knuckles touch the handle
top during the approach, before the close; in demo 1 the close squeezes the needle out (5–7 cm).
- Gripper timing: the real Robotiq is position-controlled from the GELLO trigger, so it starts closing 1–2 steps
  before the ±1 sign (cmd below mid-range) flips. Re-timing the sim close to the real knuckle onset (lead 0/1/2)
  moves single demos both ways (demo 2 lifts, demo 4 drops) and does not change the total.
- ⚠️ A kinematic penetration check is blind here: writing the two Robotiq driver joints does not move the passive
  linkage (equality constraints act only in the dynamics), so "no penetration at the real q" proved nothing.
- Still open: does the real gripper also touch the cube top before closing? A side photo of the real grasp
  depth, or the operator's account, decides between "sim Robotiq geometry / linkage" and "sim is right, the real
  fingers simply comply".

**Static sag = missing payload 0.30 kg** (`tools/fit_payload_sag_1016.py`, 183 still frames). j2 torque rms
2.43 → 0.94 N m with the fit; the CoM along the hand axis is not identifiable from these poses (the axis is near
vertical), the mass is. 0.30 kg ≈ Robotiq 2F-85 (~0.93) + ZED-M + mount (~0.1) − Franka Hand (0.73), i.e. the
controller is still configured for the Franka Hand. Predicted sag at the DROID home: TCP −4.0 mm z, −1.6 mm x.
Cross-check on the Franka-hand teleop data (same script, CoM fixed 60 mm along the hand axis): threading4
0.17 kg, threading6 0.22 kg missing ⇒ the sag predates the Robotiq; most likely the back-of-hand ZED-M + mount were
never in the load config, and the heavier Robotiq added ~0.1 kg on top. (The 09-22 `realdata/episode_000*` cannot
be used: their command carries an uncorrected GELLO offset of 0.1–0.2 rad.) The camera calibration sessions cannot
replace this: they store only the measured q (joint_tap), no command, and the link-side encoders already include the
sag, so FK(q) vs PnP cannot see it either. Any later teleop recording with still frames re-measures it
(`tools/fit_payload_sag_1016.py <dir>`); add a few tilted-wrist holds to also identify the CoM.
⇒ Fix at the source: set the end-effector load in Desk (or franka_ros2 set_load) to the Robotiq assembly
(~1.03–1.1 kg, CoM ~6 cm along the flange axis), then re-measure the sag. That fixes teleop and deploy alike.
Fallback without Desk access: a feed-forward `q_cmd += K⁻¹ τ(m = 0.30 kg)` in `policy_server` (not done).

**Action convention: `ref_inverse`** (converter default on the Robotiq cell). Per step, shoot the OSC action that
moves the sim REFERENCE arm (own q and qvel, as `--action_map sim_ref` steps it) onto the next GELLO command pose.
The reference follows the command to p50 0.08 / p99 3.1 / max 20 mm (clipped fast moves). Open-loop replay in the
sim no longer drifts: 12–26 mm from the measured TCP (= the command's lead + sag), was 0.2–0.8 m with `cmd_fk`.
The labels mean what the policy's actions mean at deploy; the sag stays a real-side dynamics property.

- Bug found on the way: patching `_get_initial_placement_bounds` does not fix the layout, because the wrapper's
  `apply_cell_layout` re-wraps it on the instance with the cell's full ranges. The replay places objects directly.

## §3 wrist re-calibration (10-15 evening): the wrist camera moved 6 mm / 0.8° since 10-13

Session `~/simdist/calib/run_1015_1915` (raw PNGs stay there, 105 MB; joints + solver outputs in
`data/wrist_run_1015_1915/`). Setup: ChArUco taped flat on the table, needle at about (0.556, −0.110) yaw −110°,
tripod removed (it hid the board). 28 samples, 27 used, **10 tilted ≥ 35°**, needle in the wrist view in 19.
Scene camera **not bumped**: `scene_rot_delta.py` vs the 10-14 `home_droid` frame gives 0.015°.

| | value |
|---|---|
| joint solve (`calib_20261010/tools/solve_calib.py`) | wrist per-view median 3.4 px, LOO median 4.5 px; wrist LOO ≤ 3.1 mm / 0.39° |
| ⚠️ its scene pose | **14.6 mm / 1.1° off the cell's**, although the camera did not move ⇒ the joint solve's base↔board chain carries ~1.5 cm of systematic error (FK / joint compliance; the per-view residuals reach 10 px while each view's own PnP is 0.1–0.7 px; 10-13: median 2.6, max 6.9 px) |
| **wrist with the scene FIXED** at the cell's validated pose (`tools/solve_wrist_fixed_scene.py`) | per-view median 2.9 px, LOO 4.1 px, **LOO ≤ 2.6 mm / 0.32°**; square 42.00 mm |
| it vs the joint solve | **0.9 mm / 0.06°**: the wrist result does not depend on the scene pose |
| **it vs the cell (10-13 solve)** | **6.2 mm / 0.79°**: in camera axes (x right, y down, z optical) −0.1 / −5.6 / −2.6 mm |
| new wrist pose (frame hand) | pos `[-0.07452, -0.03588, 0.01144]`, quat wxyz `[0.92782, -0.15101, 0.03893, -0.33888]` |

Held-out check (`tools/check_wrist_needle.py`, `figs/wrist_needle_check_1015.jpg`). The needle pose is fitted in
the scene camera (IoU 0.84), the wrist view is then rendered with each pose, and nothing here was used in the solve:
- **Gripper silhouette IoU: 0.690 (cell) → 0.755 (new)**, better in 24 of 27 views.
- **Needle in the wrist view: unchanged**, IoU median 0.39 → 0.40, centroid offset median ≈ 2.5–3 px at 224.
  The outlines sit on the real needle in most views; the 10–24 px cases are views where the needle is cut by the
  image edge or the fingers.
- ⇒ The 10-14 "needle 12 mm / 7° off in the wrist" is gone with either pose at this placement. The new pose is
  at least as good on the world and better on the gripper. The gripper IoU stays ≤ 0.85, so part of the
  remaining gripper misfit is the Robotiq geometry / mount, not the camera.
- Drift across sessions continues (10-10 → 10-10 late → 10-13 → 10-15): the mount is probably loosening.
  Tighten it before the demos, or re-check it with one board shot per session.

**Not changed yet (sim side, offline):** the cell's wrist nominal pose → the pose above. The wrist DR is
±(6, 10, 20) mm / ±(3, 3, 4.5)°, set wide on purpose (owner). The measured error is now ≤ 3 mm / 0.3° within a
session and 6 mm / 0.8° between sessions, so something like ±(6, 6, 8) mm / ±(1.5, 1.5, 2)° would cover it.
To decide with the owner.

## §4 touch-offs (10-15 evening, hand-guided): base ↔ scene confirmed, needle position confirmed, z_min not usable

Session `~/simdist/calib/touch_1015_1957`. Joints + results are in `data/touch_1015_1957/`; analysis
`tools/touch_1015.py`; figure `figs/touch_fingertips_on_scene_1015.jpg`.

**How it was done.** The arm was moved by hand in Desk Programming mode. The robot driver ran read-only (no
controller loaded, no cameras) through `simdist_runner/bringup_stack.py --cameras`, and `/franka/joint_states`
streamed at 1 kHz throughout. The fingertip is FK of the measured q on the cell model, with the Robotiq closed
and `gripper_mount_z 0.0135`. The board was re-detected by the scene camera at the cell's pose: pooled PnP
0.44 px, per-sample spread ≤ 3 mm.

| point | result |
|---|---|
| all 7, **FK gripper projected into the scene image** | it lies on the real gripper in all 7 photos, within a few px at 1280 ⇒ base ↔ scene camera ↔ FK agree again (as on 10-14) |
| needle_end_a / b | midpoint **(0.5564, −0.1098)** vs the scene-camera render fit (0.5564, −0.1097): **0.1 mm**. The line through the two touches is 14° off the fit's yaw; the "end" touches landed on the corner of the 4 cm handle cube (owner: 4 cm cube, 16 cm overall), so the render fit (IoU 0.84) is the better yaw |
| c1–c4 | ⚠️ touched at the clipboard / outer print corners, not at the inner chessboard corners (the near corners were out of reach) ⇒ 12–31 mm from the nearest outer chessboard corner, so **no xy / yaw number for base ↔ table from these**. The projection check above covers base ↔ scene |
| zmin | fingertip **0.0337** vs the work surface 0.0285 (+5.2 mm). On 10-14, five touches in execution mode gave **0.0287 ± 0.0012**. In the projection the red tip sits slightly above the real tip, so this hand-guided touch most likely did not reach the surface. **Keep 0.0285**, do not use this value |

⇒ For `dr_base_xy` / `dr_base_yaw_deg`, a future touch-off needs inner chessboard corners that are within reach,
with the board placed accordingly, or a GELLO-driven touch in execution mode for z.

## §2 decided (10-15 evening): **10 Hz**, the sim is retrained at 10 Hz

The decision is made with the user. The sim, the recorder (`run_teleop_session.py --rate 10`) and the runner
(`robot_client.py --hz 10`) all use 10 Hz. The sim needs `control_freq` 10 (100 ms per step), regenerated MimicGen
data and a new stage-1; the old 20 Hz data and checkpoints do not carry over (CLAUDE.md rule #37). The sim also
needs the real arm's behaviour around that step: the sim_ref speed cap (8 cm/s), the 0.1–0.4 s lag and the
1–2 step lead.

**Recorder fix before §6** (gello is an upstream clone; diffs in
`calib_20261014/tools/gello_patch/*_robotiq_state_1015.diff`, backups `*.bak_20261015` next to the scripts):
the teleop recorder subscribed to the Franka Hand topic and dropped any gripper state with fewer than two joints,
so **Robotiq demos would have had no gripper state at all**. `collect_demos.py` now stores the one knuckle angle as
`observation/gripper_knuckle` (rad, 0 open .. ~0.79 closed), and `run_teleop_session.py --gripper robotiq` passes
`--gripper-state-topic /gripper/joint_states`. Checked in the container: the extractor returns the knuckle, and
the recorder command carries the topic.

## §5 layout + exposure, §6 first demos (10-15 evening)

**Layout** (`~/simdist/calib/layout_1015_2008/sample_000`, `figs/layout_shot_1015_scene.jpg`, fit in
`data/layout_1015/layout_fit_1015.json`). Scene-camera render fit:
- needle (0.5561, −0.1108) yaw −110.2°, IoU 0.85, the same as in the wrist session;
- tripod (0.5204, 0.1832) yaw −36.0°, IoU 0.91, the same as on 10-14.

Both are inside the cell's ranges. The tripod yaw sits at the −35.6° edge, so centre it nearer −40° for the demos.
(The 8-DoF wrist correction this tool also prints is not used: one frame at DROID home barely sees the objects,
so the fit is ill-posed.)

**Exposure drift** (locked controls, empty board, centre-third mean grey):

| time | scene | wrist |
|---|---|---|
| 20:08 | 166.7 | 168.7 |
| 20:26 (after the demos) | 166.6 (−0.1 %) | 170.1 (+0.8 %) |

That is only 18 minutes apart, so it does not test the ~10 % seen across hours on 10-14. Repeat across a longer
session.

**Demos, first 5** (`~/simdist/gello_software/demos/threading_robotiq_1015/episode_000{1..5}.h5`, 230 MB, not in
git). Recorded by `run_teleop_session.py --gripper robotiq --rate 10` with the recorder fix:
- 21–31 s each, 100.0 ms per frame, no stale or dropped frames;
- the knuckle is recorded (0.0 open, ≈ 0.44 on the needle handle) and the gripper closes at frames 60–79;
- the GELLO command leads the measured joints by up to 7–12° during fast moves;
- the operator pressed SPACE before releasing the gripper, so the last frame = success, gripper still closed;
- the start pose was "forced", with up to 3.6° off the DROID home on j2.

The rest (to 20–30) was deferred on purpose: replay these 5 in the 10 Hz sim first, then decide the protocol.

## Next (offline, sim side)

The full checklist, in dependency order and with the numbers, is in **[`SIM_TODO.md`](SIM_TODO.md)**.

1. Sim `control_freq` 10, plus the real arm's behaviour around a step: the sim_ref cap of 8 cm/s, the
   0.1–0.4 s lag, the lead.
2. Convert and replay the 5 demos in the sim (`scripts/convert_real_teleop.py` needs the Robotiq knuckle and
   10 Hz). Where they fail is the real-vs-sim gap.
3. Cell: wrist nominal pose → the §3 pose. Wrist DR narrowing is the owner's call.
4. Then regenerate MimicGen at 10 Hz from the replayable demos, and run stage-1.

## Payload probe on the robot (10-16 evening): **≈ 0.3 kg missing, confirmed; the CoM is not identifiable**

`tools/payload_probe.py` was run (NEXT_VISIT §1): 15 poses, each held 4 s, 50 robot states + ~2500 joint messages
per pose. Raw json is on the workstation, `~/simdist/gello_software/simdist_runner/payload/payload_probe_1016_2120.json`
(25 MB). Per-pose means are in `payload_1016/payload_probe_1016_2120_summary.json`; the dry run (configured load
only) is `payload_dryrun_1016_2116.json` next to the raw file.

**Configured now** (robot_state): `inertia_ee` m **0.90 kg**, CoM (0, 0, 0.057) m, I = diag(0.00277, 0.00315,
0.00056); `inertia_load` 0 kg. (Not the 0.73 kg Franka Hand value; someone set 0.9 kg.)

| estimate | missing mass | ⇒ total | CoM of the total (hand frame, mm) |
|---|---|---|---|
| torque, all joints, no bias (`fit_payload_probe.py`) | 0.247 kg | 1.15 kg | (−1, −17, 70) |
| torque, all joints + per-joint bias | 0.364 kg | 1.26 kg | (−8, −2, 56) |
| torque, j2 + j4 only + bias (gravity-dominant, stiff joints) | 0.336–0.343 kg | 1.24 kg | z not consistent (−57 for the missing part) |
| static PD error, j1–j6 (position) | 0.294 kg | 1.19 kg | (0, −23, 74) |
| 10-16 offline sag fit (183 still frames, earlier) | 0.30 kg | 1.20 kg | — |

⇒ **The missing mass is 0.25–0.36 kg by every method; about 0.30 kg.** The CoM is NOT identified:
- **The wrist joints are dominated by stiction, not gravity.** The static error of j5 / j6 / j7 flips with the
  direction the pose was approached from: j5 +11 / −13 mrad, j6 ±13 mrad, j7 +20 / −27 mrad. At K = 100 / 60 / 20
  that is 0.4–1.3 N·m. `tau_ext_hat_filtered` on j5 follows the same sign (+0.7 … +0.9 vs −0.94 N·m), so the
  link-side torque sensor sees the holding torque too, and the cables (ZED-M, Robotiq) probably pull on the wrist
  as well. This is the same size as the gravity signal of a CoM offset, so CoM x/y/z swing between fits (cz from
  −57 to +119 mm for the missing part).
- j1 (vertical axis, no gravity torque) reads −0.8 N·m in 14 poses and +0.75 N·m at j1 +0.6: the same stiction
  pattern. That bounds what a bias term can absorb.
- Static error at the DROID home: j2 −5.4 mrad (q is lower than the command), j4 +6.0 mrad. ×240 ≈ 1.3–1.4 N·m:
  consistent with 0.3 kg at ~0.45 m.

**Recommendation (owner's call, Desk → end effector):** total mass **1.20 kg** (0.90 + 0.30), CoM keep on the
flange axis at **z ≈ 0.065 m** (between the fits' 0.056–0.074; physically Robotiq 2F-85 + coupling + ZED-M on the
back of the hand), keep x = y = 0 and the inertia. ⚠️ The hand frame here is robosuite's `robot0_right_hand`, 45° about
z from the FR3 flange: irrelevant while x = y = 0. Then **re-run the probe**. Fixed means a missing mass ≤ 0.03 kg
by the j2 + j4 fit and j2's static error at home < 2 mrad (now −5.4). The wrist stiction stays (a ±1 cm-class effect
at the TCP over the approach direction); it is a property of the real arm, not of the payload.

## Recommendation

**Keep the ROS 2 / franka_ros2 stack and do not switch inference to polymetis.** None of the latency measured
today is middleware:

| part | measured | would polymetis change it? |
|---|---|---|
| joint state at the client | 0–2 ms | no, it is already negligible |
| image age | 7–19 ms | no: the cameras stay on ROS / V4L2 |
| server step (preprocess + reference arm) | ~35 ms | no: it runs on the workstation |
| **joint lag ~100 ms** | **PD tracking + the per-period linear ramp of the client** | **no**: polymetis' default joint impedance is also PD; with the same gains it gives the same lag and the same friction deadband |

Switching would cost a new gRPC client (arm + Robotiq) with cameras still on ROS. It would also mean swapping stacks
between teleop / recording and inference: the two cannot own the arm together, which is exactly how today's
gripper-port clash happened. And it would add an untested variable on top of all the existing real data.

What to do instead, in this order:

1. **Port `sim_ref` into the main repo's `policy_server.py`** (default on for the real-cell tasks).
   - The sim reference arm keeps its own q and qvel and is stepped by the actions exactly as in training. The real
     arm is sent that trajectory, so a lagging arm builds up error, and therefore torque, until it moves.
   - It is re-synced to the measured q only if the two TCPs drift more than `resync_m` apart (09-25: 0.08 m, e.g.
     when blocked by contact).
   - Keep the 09-25 hard limits from `_speed_caps`: the target is never more than `max_lead` (0.03 m) ahead of the
     measured TCP, `max_tcp_speed` is 0.08 m/s, `max_rot_speed` is 0.4 rad/s, and a capped step pulls the reference
     back (no wind-up).
   - Then re-run `tools/run_tracking_test.sh`; the operator presses `g` once. Pass criterion: x done/step ≈ z
     done/step, and overshoot still ≤ a few mm.
2. **Model the lag in the sim, whatever the rate.**
   - Add a first-order lag of τ ≈ 100 ms between the OSC target and the arm, plus delay DR around it. The real speed
     is about 2 cm/s at both rates, so 10 Hz alone does not remove the gap; it only makes the per-step ratio look
     better.
   - Re-measure τ after step 1: part of the 100 ms is the client's one-period ramp and part is the deadband.
3. **Only if the lag is still large after 1–2:** stiffer gains. This is a shared teleop config, it changes the
   feel of teleop, and it needs the owner's OK. libfranka's own joint-impedance example uses roughly
   K 600 600 600 600 250 150 50; check that before using it. An integral term or friction compensation in the
   controller is the last resort: it is C++ rebuilt on the NUC.
4. If polymetis is still wanted, decide it with data: run the same scripted moves through DROID's polymetis
   (installed on the NUC) and compare the step response like for like.

Control rate (§2): no reason to prefer 20 Hz over 10 Hz was found today. Decide after step 1. With `sim_ref`
the rate mostly sets how far ahead the reference runs, and 10 Hz is what the owner found stable.

## What was fixed / added today (needed to get §1 to run)

- **`real2sim/deploy/robot_client.py`**
  - **Loop starvation.** The loop only spun ROS for whatever was left of the period. After one step slower than the
    period that was 0 s forever, so every input went stale together and the client stopped at step 32. It now
    always drains for ≥ 5 ms, and when it is late it skips the missed ticks instead of bursting.
  - **New `--activate`.** If the impedance controller is inactive, the client pauses the GELLO, holds the
    **measured** pose and then activates. Without this, the arm would jump to a misaligned GELLO (1.57 rad off
    today).
- **`tools/run_tracking_test.sh`**: §1 in one command, run on the host. It does the following:
  - brings up the stack;
  - loads the controller inactive (the robot launches with `spawn_controller:=false`, and only teleop's hand-over
    loads it);
  - per rate: `reset_arm --no-resume`, the scripted server, the live client with `--activate`, the analysis;
  - stops if overshoot exceeds 10 mm;
  - asks once, before the arm moves.

  `--down` tears the stack down.
- **`tools/bringup_stack.py`** (container, `/workspace/simdist_runner/`): the `run_teleop_session.py` bring-up
  (cameras, robot, Robotiq, GELLO) **without** reset, matching gate or controller activation.
- **`tools/wrist_camera_info.py`**: a stop-gap. The ZED-M **HID interface did not enumerate**, so its serial was
  unreadable and `publish_zed_uvc.py` sent no wrist `camera_info`. This script publishes it from
  `SN16478414.conf`. The server takes the wrist K / D from `calib_20261010/data/wrist_intrinsics.json`
  anyway, so only the image size comes from `camera_info`. A USB replug of the ZED-M should fix the root cause
  (exposure is re-locked on every open).
- **Deploy copy `~/simdist/deploy_simdist_1015/`** (host, not in git): the repo's code plus the old bundle's
  compiled `common_utils/build`. Overwriting the 09-23 `~/simdist/deploy_simdist` was refused. The client in the
  container is `/workspace/simdist_runner/robot_client_1015.py`; the old `robot_client.py` is untouched.

## Traps hit today

- **NUC: DROID's polymetis was left running** (started 14:20 NUC time as root: `launch_gripper.py` on
  `/dev/ttyUSB0`, plus `run_server -p 50051`). Symptom: the Robotiq driver aborts with
  `SerialException device reports readiness to read but returned no data`. Check with
  `fuser -v /dev/ttyUSB0` on the NUC. DROID's `scripts/server/run_server.py` (PID 4521) is still running.
- **DDS:** any standalone ROS command in the container needs
  `CYCLONEDDS_URI=file:///workspace/ros2/cyclonedds_ws.xml`. Without it the client sees nothing
  (`inputs missing after 15 s: joints=False ...`).
- **`reset_arm.py` returns early when the arm is already at the start pose** ("already at the start pose."),
  **without activating the controller**.
- **Teleop launched with an empty `--cameras`** publishes no cameras, and the client waits for both.

## Next step

1. ~~Port `sim_ref` + `_speed_caps` into `policy_server.py`~~ **done (10-15 evening)**:
   `--action_map sim_ref` (default; `reseed` = the 10-15 baseline), `--resync_m 0.08`, box *clamp* instead of
   refusal, `--z_slow_band 0.06 / --z_slow_step 0.003` (0 with `--scripted`), hard caps `--max_tcp_speed 0.08`,
   `--max_rot_speed 0.4`, `--max_lead 0.03` on both maps. `analyze_tracking.py` now also prints `ref cm/s`,
   `real/ref`, `lag mm` and `past tgt mm` (with sim_ref the target leads the arm, so read these, not done/step).
   - Offline check (fake arm: τ 100 ms + a static-friction deadband on the PD error, 20 Hz):

     | map | x real cm/s (amp 0.2 / 0.5 / 1.0) | z real cm/s | past target |
     |---|---|---|---|
     | reseed | 0.0 / 0.2 / 0.9 | 0.0 / 0.0 / 0.6 | ≤ 2 mm |
     | sim_ref | 5.1 / 4.7 / 7.1 | 5.3 / 7.0 / 7.9 | ≤ 4.4 mm, lead ≤ 18 mm |

     The deadband model is stricter than the real arm (real z moved under reseed, with gravity), so only the
     contrast counts. Dry run with the real inputs (arm not moving): runs, and the target stops 3 cm ahead (`capped(lead)`).
   - Note: the speed cap (8 cm/s) is below the sim's OSC speed at |a| ≥ 0.2 (≈ 20+ cm/s at 20 Hz), so on the real arm
     all three amplitudes run at the cap. That is the owner's 09-25 limit; the sim does not model it yet (step 2).
2. ~~On the robot: re-run~~ **done, passes** (section above). (`run_tracking_test.sh` now runs `sim_ref`; runs are named
   `runs/track{HZ}_sim_ref_*`). It stops between rates on `past tgt mm` > 10. Pass: x `real/ref` ≈ z `real/ref` ≈ 1,
   real x speed of several cm/s (10-15: 0.1 cm/s), `past tgt mm` ≤ a few mm.
3. Then continue `NEXT_VISIT.md`: §2 rate with the owner, §3 wrist re-calibration (teleop without cameras +
   `record_calib.py`; the stack from step 2 must be down first: `run_tracking_test.sh --down`),
   §4 touch-offs, §5 layout / exposure drift, §6 demos.
