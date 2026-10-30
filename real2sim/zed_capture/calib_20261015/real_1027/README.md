# 10-27 robot visit: first stage-1 (exec_obs) policy on the robot — grasps, does not insert; cause = in-hand needle pose

Checkpoint `robotiq_s1_execobs_1027` (exec_obs 1, prop 36, 10 Hz, 400 steps, best at 150k), served from
`~/simdist/deploy_simdist_1015` (rsynced to c4a506b + the fixes below). Checklist was `../NEXT_VISIT_STAGE1.md`.

## TL;DR for the sim side

1. **Grasp + lift works on the robot: 100 %** (every episode). The approach phase agrees with human demos
   (open-loop cos 0.83–0.99).
2. **After the grasp the policy sweeps +y but with +x (+0.1…+0.2) and climbs 3–5 cm, then stalls** at
   grip − ring ≈ (+0.17…+0.20, −0.08…−0.02, +0.02…+0.05) m with |a| ≈ 0.02. Systematic, all 8 takeover episodes
   and both MPPI episodes. In sim (6/6 success, ~50–60 steps) and in the human demos the same sweep goes **−x**
   at ring height (insertion actions ≈ (−0.2, +0.13, 0)).
3. **Cause: the needle sits in the real gripper in a different, very repeatable pose** (6 policy grasps, fitted
   from wrist + scene silhouettes, wrist IoU 0.82–0.91). Gripper frame = `gripper0_grip_site`, z along the fingers
   (down), −y toward the needle tip:

   | | **real (n=6)** | sim (`results/inhand.json`) |
   |---|---:|---:|
   | handle centre along the shaft (+ = toward the tail) | **−12.0 ± 1.0 mm** (grips the tail half of the cube) | +11.9 mm (shaft half) |
   | handle centre below the grip point | **+15.3 ± 1.9 mm** (grips only the top of the cube) | ≈ 0 (cube centre) |
   | shaft pitch, tip down + | **+9.8 ± 0.9°** | +1.3° |
   | shaft yaw | **+6.9 ± 2.8°** | +1.4° |

   Full per-grasp poses (same convention as sim `inhand.json`: needle root body in the grip-site frame) and the
   mean: **`results/inhand_real_1027.json`**. Figures: `../figs/inhand_fit_1027_grasp04.jpg`,
   `../figs/inhand_side_photos_1027.jpg`.
4. **Closed loop on the cause:** all-sim images (no visual gap at all), real demo arm states, needle placed at the
   **real** mean in-hand pose ⇒ the post-grasp x action goes from −0.24 / −0.17 (sim in-hand pose) to
   **−0.04 / +0.01** — the real failure, reproduced in pure sim (`results/abl_realinhand_d{1,2}.json`).
   The policy is extremely sensitive to the in-hand pose: ±7–14° of pitch flips the sign of x (`abl6_*`).
5. **Sim needle rests flat with the tip 15 mm in the air** (`tools/needle_rest.py`: pitch +0.02°, handle cube on
   the table, shaft cantilevered); on the real table the tip rests on the table (≈7° tip-down by geometry).

### What the sim needs (owner: sim side)
- needle at rest tip-down (mass distribution so the CoM leaves the handle's footprint, or an initial ~7° pitch
  that settles);
- check the sim fingertip–table contact height vs the real one (real `z_min` 0.960 because the closed fingertips
  touch at eef z 0.9563; the sim estimate was 0.950) — part of the 15 mm depth gap may be this clamp;
- **in-hand pose randomization that covers the real distribution with margin**: along shaft ±25 mm,
  depth 0–20 mm, pitch 0–12°, yaw ±10°;
- acceptance test: in sim, put the needle at `inhand_real_1027.json`'s pose after a grasp and require insertion.

## How the cause was isolated (open loop, demos 1–2 = clean human demos at the tape position)

Same real state (q, dq, knuckle), sim render with the needle at the sim in-hand pose and the tripod at
(0.521, 0.183, −35°) (validated overlay), same prop/exec_obs; mean post-grasp action x (d1 / d2):

| input | x | cos to sim action |
|---|---:|---:|
| all sim | −0.24 / −0.17 | 1 |
| real scene + sim wrist | −0.18 / −0.14 | 0.94–0.97 |
| sim scene + real wrist | −0.03 / +0.03 | 0.42–0.57 |
| all real | −0.01 / +0.03 | 0.41–0.51 |

Wrist image factors on the (real scene + sim wrist) baseline: shadows off, colour matched, wrist camera ±1 cm on
each axis, realistic blur (σ≈0.9 px, measured by Laplacian variance) — **all ≤ 0.03**. Sim foreground on the real
background stays −x; **real foreground (gripper + needle) on the sim background goes +0.11 / +0.12**. Needle is the
larger part of the foreground effect. ⇒ led to the in-hand fit above.

Wrist camera alignment is fine: the sim finger silhouette at the real q overlays the real fingers within 2–3 px
(224 px). The server uses the refit wrist K,D (`real_intrinsics` in the cell), not the factory ones recorded in
the demo h5.

## Runs (`~/simdist/deploy_simdist_1015/runs/`, in the handoff tar)

| run | what |
|---|---|
| `s1_dry_1027_1855` | dry, board on the table, no objects |
| `s1_dry_1027_1904` | dry; correct directions (+x, −y, −z, open); also the z_min touch |
| `s1_policy_1027_1910` | ep1: `cartesian_reflex` (needle/gripper hit the tripod) — **wrong colour map** |
| `s1_policy_1027_1927` | 264 steps, unclear |
| `s1_policy_1027_1935` | 4 eps, all stall at grip ≈ (0.755, 0.12, 0.31) base; obs of ep0–1 overwritten (bug, fixed) |
| `s1_policy_1027_2014` | ep0 stall at (0.70, 0.14, 0.26); ep1 = operator moved the tripod (intervention, ignore) |
| `s1_mppi_1027_2028` | MPPI `init_std=0.3`, 2 eps, same failure; latency median 52 ms, **p95 71–75 ms (> 60 ms)** |
| `s1_policy_1027_2057` | 8 takeover eps (policy until stall, then human via GELLO); ep4–7 = demos 3–6 |
| `s1_inhand_1027_2145` | the 7 in-hand grasps (grasp_02 excluded by the operator) |

Demos `~/simdist/gello_software/demos/threading_robotiq_1027/episode_0001–0006.h5`: 1–2 clean human demos;
3–6 recorded from home ⇒ policy segment + human takeover. In-hand snapshots: `~/simdist/gello_software/inhand_1027/`
(raw scene/wrist JPEGs + q + knuckle), phone side photos IMG_5457/58/59/60/61/63 = grasp 01/03/04/05/06/07.
Board photos `~/simdist/calib/run_1027_1853` (5 samples, inconclusive at mm level: board-in-base consistent to
2–6 mm with the 10-15 wrist pose; the scene camera looks ~0.35° rotated vs `home_droid/shot_000` but with few inliers).

## Fixed / added today
- **`policy_server.py`**: obs images were overwritten every episode (`self.n` reset) ⇒ episode counter, files
  `obs_ep{ep:02d}_{n:05d}.png`, `episode` field in steps.jsonl.
- **`robot_client.py --no-gello`**: for a stack without the GELLO (implies activate, requires the controller
  inactive, no SIGSTOP; on exit the gripper keeps its last command, so a grasped needle stays in the hand).
- `../tools/` (container side, `/workspace/simdist_runner/`): `bringup_nogello.py` (teleop stack minus the GELLO,
  controller loaded inactive), `home_nogello.py` (open Robotiq, move to DROID home, controller off),
  `takeover.py` (hand the arm to the GELLO where it stands: match table, offsets = robot − GELLO, no jump — hold
  the GELLO trigger closed before Ctrl-C in the client), `grab_frames.py`, `grab_inhand.py` (scene+wrist ×5 + q +
  knuckle per grasp).
- Analysis scripts: `tools/` here (`query.py` open loop, `ablate*.py` image ablations, `inhand.py` sim in-hand
  pose, `fit.py` real in-hand fit, `needle_rest.py`, `fingers.py`, `roll2.py` sim rollouts). They run from
  `~/simdist/deploy_simdist_1015` with `MUJOCO_GL=egl`.

## 10-27 demos in the v8 sim (10-30 cell, done 10-30)

Six demos: ep1–2 clean human, ep3–6 = policy stalls, operator takes over with the GELLO. Converted to
`/user_data/Chenyang/real_stage2/threading_robotiq_1027_c1030/dataset.hdf5`. Layout per demo:
`../demo_layout/layout_demos_1027{,_wristrefit}.json` (scene fit, then the needle refit on the wrist camera).
Scripts, tables and videos: `viz/replay_1027_v8/` (mp4/png/npz are gitignored, on curie only).

- **Replay = OSC track of FK(real q)** (TCP error p50 0.2–0.3 mm while inserting). Writing q straight into the sim
  (`--mode kin`, spline per substep) drops the needle from the Robotiq on the lift in 6/6: not usable.
  Gripper: h5 command + the cell's 1–2 step delay, speed 0.006; in-hand DR off; tip-down rest on (6.9°).
- **As fitted: 2/6** (ep1, ep2). ep3–5 are *in* the ring (lateral 3–4 mm) but only 17–42 mm deep (success needs ≈50):
  the needle shoves the tripod 24–71 mm away. ep6 misses laterally (~20 mm) all the way.
- **Tripod offset sweep** (63 cells, dx/dy ±20/±15 mm): best at **dx +5…+10 mm (away from the base), dy −10…0**
  → 4/6 and the highest final wrist-camera red-mask IoU (0.789 vs 0.705). Selection-free check: shift each tripod
  by that demo's own needle scene→wrist correction (≈ +4, −8 mm, +6.5°) → also 4/6. ⇒ the scene-camera fit of the
  tripod likely carries the same bias the wrist refit removed from the needle.
- **Tripod mass (sim 18 g) is not the issue**: ×1/×3/×10 at (+10, 0) → 4/4/3 of 6; summed |sim − real push| over the
  6 demos 35/42/53 mm, best at ×1. ×10 at the fitted spot also gives 4/6 but freezes the tripod (ep3 1 mm vs real 32).
  Replays are deterministic (×1 reproduces the sweep cell bit for bit) but chaotic in the parameters.
- **No v9**: the corrected tripod (≈0.53, 0.18) is ≥13 mm inside `layout_tripod_*` (±25 mm), so training already
  covers it. Unresolved: ep5 never gets past ~30 mm, ep6 never (the real tripod turned −13° during it); the sim
  in-hand pose is still shallower/flatter than real. Next visit: weigh the tripod; one ChArUco shot at the needle
  spot seen by both cameras to fix the scene-camera bias.

## Traps hit today
- **Colour map omitted**: the NEXT_VISIT_STAGE1 commands had no `--color_json`, so the server used the Peg splat
  map `real2sim/deploy/color_correction.json` (+40 levels, 18 % saturated). Threading needs
  `--color_json real2sim/zed_capture/calib_20261014/color_correction_threading_1014.json`. Run 1910 and the dry
  runs used the wrong map.
- **z_min**: closed fingertips touch the table at eef z 0.9563 (sim estimate 0.950, ~6 mm off) ⇒ `--z_min 0.960`.
- **ZED-M (wrist) USB**: after the reflex stop it disconnected 21× in 7 min; when the HID interface (2b03:f681) is
  missing at bringup the serial reads None ⇒ no wrist camera_info ⇒ client fails with `camera_info=['scene']`.
  Fix: reseat the cable, restart bringup; `lsusb` must show f880, f881, f682, f681.
- A `cartesian_reflex` kills ros2_control (restart the stack).

## Commands that worked
```bash
# server (host)
cd ~/simdist/deploy_simdist_1015; ~/anaconda3/envs/simdist/bin/python real2sim/deploy/policy_server.py \
  --ckpt ckpts/robotiq_s1/model0.pt \
  --color_json real2sim/zed_capture/calib_20261014/color_correction_threading_1014.json \
  --z_min 0.960 --save_dir runs/s1_policy_$(date +%m%d_%H%M) --save_every 1      # MPPI: + --planner_set init_std=0.3
# client (container), no-GELLO stack
python3 /workspace/simdist_runner/bringup_nogello.py --gripper robotiq
python3 /workspace/simdist_runner/home_nogello.py
python3 /workspace/simdist_runner/robot_client_1018.py --gripper-state-topic /gripper/joint_states --hz 10 --steps 400 --live --no-gello
```
