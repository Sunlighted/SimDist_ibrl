# 2026-10-15 robot visit: tracking-lag test (NEXT_VISIT.md §0–§1) and what it means for the controller

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
| needle_end_a / b | midpoint **(0.5564, −0.1098)** vs the scene-camera render fit (0.5564, −0.1097): **0.1 mm**. The line through the two touches is 14° off the fit's yaw; the "end" touches landed on the corner of the 2 cm handle cube, so the render fit (IoU 0.84) is the better yaw |
| c1–c4 | ⚠️ touched at the clipboard / outer print corners, not at the inner chessboard corners (the near corners were out of reach) ⇒ 12–31 mm from the nearest outer chessboard corner, so **no xy / yaw number for base ↔ table from these**. The projection check above covers base ↔ scene |
| zmin | fingertip **0.0337** vs the work surface 0.0285 (+5.2 mm). On 10-14, five touches in execution mode gave **0.0287 ± 0.0012**. In the projection the red tip sits slightly above the real tip, so this hand-guided touch most likely did not reach the surface. **Keep 0.0285**, do not use this value |

⇒ For `dr_base_xy` / `dr_base_yaw_deg`, a future touch-off needs inner chessboard corners that are within reach,
with the board placed accordingly, or a GELLO-driven touch in execution mode for z.

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
