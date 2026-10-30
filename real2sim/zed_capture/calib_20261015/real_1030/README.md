# 10-30 robot visit: v8 stage 1 (60k / 70k) — the 10-27 stall is gone; grasping is the new bottleneck

Package `v8_s1_1030` (sha256 OK), served from `~/simdist/deploy_simdist_1015` (rsynced to 3d086d8; that dir is not a
git repo). Server/client commands exactly as in the package README (Threading colour map, `--z_min 0.960`,
no-GELLO stack). Data: handoff tar `to_sim_1030.tar` (runs below).

## TL;DR for v9

1. **The 10-27 failure is fixed.** Open loop on the 10-27 human demos, post-grasp mean x: old ckpt −0.01 / +0.03,
   v8 60k −0.23 / −0.17, 70k −0.23 / −0.19. On the robot every v8 grasp that held went to the ring; no +x stall.
2. **60k ≫ 70k on the robot** (as the sim handover predicted, 7/8 vs 1/8): 70k 0/7 insertions; 60k reaches the ring
   with the needle in hand in most episodes. **60k: 5/12 inserted (~42 %); 70k: 0/7.**
3. **New bottleneck: the first grasp always misses, along the shaft.** At the first close command the handle sits at
   wrist-image x **202–215 px** (224 px image, centre 112; 60k, 11/11 episodes) — half out of the view. The sim
   policy closes "on the move" too, but with the handle at **159–199 px** (v8 60k, 6 sim episodes). The 2nd attempt
   lands at ~185–191 px, the 3rd at ~170 px; 60k recovers by re-grasping (2–5 extra attempts, 50–150 steps).
   Image-right = along the shaft (grip +y), not the finger closing axis.
   - At the first close the real arm still trails its target by ~2 cm (4–6 cm a few steps earlier; v8 descends at up
     to 0.28 m/s vs 0.09 for the old ckpt) and keeps moving 1.5–2.5 cm while the Robotiq closes.
4. **70k grasps off the handle's tail → the needle pivots tip-down beyond the sim DR.** Fitted in-hand pose (wrist +
   scene silhouettes, wrist IoU 0.88–0.94) for the 3 lifted 70k grasps: handle centre −11 mm along the shaft in all
   three, pitch in the gripper frame **+10 / +16 / +26°** (cell `inhand_dr_pitch_deg` is [−6, 18]). 4 of 7 70k
   grasps dropped the needle within 6 cm of lift. 60k in-hand fits are unreliable (handle hidden by the fingers,
   IoU 0.67) — roughly +4…+6°.

### v9 changes, in priority order
1. **Grasp timing / tracking at the bottom of the descent.** Compare, at the sim close command, eef − reference
   target and the eef travel between the close command and finger contact against the real numbers above (real:
   ~2 cm lag, 1.5–2.5 cm travel). If the sim arm is "on target" when it closes, the exec layer under-models lag at
   v8's descent speed; fix that (or DR it) before anything else. Optionally wider `gripper_cmd_delay` (1–3).
2. **Off-centre grasp physics.** In sim, grasp the handle 10–15 mm toward its tail and lift: the real needle pivots
   to +16…+26° tip-down. If the sim needle stays level, the finger contact is too sticky in torsion — calibrate it
   (this is the root cause; DR is the bandage).
3. **Wider in-hand grasp-pose DR** (operator, 10-30): pitch to ~[−6, +30]° (70k measured +26°, 60k ep3 near-vertical with a not-fully-closed gripper), and widen along/depth/yaw too — real grasps land wherever the first close happens.
4. Keep the re-grasp behaviour (it is what makes 60k work); don't select it away.
5. **Checkpoint selection:** the sim handover-with-real-in-hand-pose test ranked 60k/70k correctly; use it (with the
   10-27 and 10-30 in-hand poses) as a selection metric, not the in-training eval alone.
6. Planning: not tested and not worth it at stage 1 (sim planner ≈ policy head; the failures are sim-physics gaps
   the world model shares; MPPI p95 latency 71–75 ms > 60 ms on 10-27). Revisit after stage-3 real adaptation.

## Runs (`~/simdist/deploy_simdist_1015/runs/`)

| run | ckpt | eps | result |
|---|---|---:|---|
| `v8_s1_60k_dry_1030_1703` | 60k | dry | descent 1.239 → 0.967 in ~2 s (≤ 0.28 m/s); eef z min 0.9610 (target clamped at 0.960; closed fingertips touch at 0.9563 ⇒ 4.7 mm margin); 47 z-clamp warns at the bottom (searching for the absent needle); 1 re-sync during the fast descent; latency total median 42 / p95 61 ms |
| `v8_s1_60k_1030_1706` | 60k | 2 | ep0 inserted (end grip−ring (+0.039, −0.009, +0.001)); ep1 no grasp |
| `v8_s1_60k_1030_1714` | 60k | 2 | ep0 inserted (+0.040, −0.038, +0.005); ep1 at the ring 2.5 cm high at 400 steps |
| `v8_s1_70k_1030_1801` | 70k | 7 | 0 insertions; 3 lifted (pitch +10/+16/+26°), 4 dropped |
| `v8_s1_60k_1030_1818` | 60k | 8 | **3/8 inserted (ep2, ep6, ep7; confirmed by the operator)**; ep0/ep1 never held the needle (ep1: the arm swung toward the scene camera, operator stop); **ep3: the Robotiq was not fully closing (bad gripper initialisation, operator)** — the needle hung near-vertical and poked down beside the post; ep4 needle tilted past the loop (operator stop at 268); ep5 tip on the loop edge. ⚠️ Holding knuckle (median of the last 60 steps) is **0.88 in ep2–ep4 vs 0.44 in ep5–ep7** (0.42–0.44 on 10-27 and earlier today): the gripper state changed between ep4 and ep5 — account for it when replaying ep2–ep4 |

First-close handle position (wrist px) per 60k episode in `1818`: 215, 202, 207, 206, 211, 214, 206, 214; attempts
before a holding grasp 2–5. Session logs: only `franka_gripper_node died` (Franka hand node, unused with the
Robotiq) and one broadcaster realtime-lock error; no reflex today.

## Files
- `results/inhand_fit_v8_1030.json` (per grasp: dx/drot vs sim, pitch/yaw, handle centre in the grip frame, IoU),
  `results/sim_v8_60k_close.json` (6 sim episodes: success + grip − handle at the close command).
- `figs/close_sim_vs_real.jpg` (scene | wrist at the close command, sim top, real bottom),
  `figs/inhand_fit_70k_ep6.jpg`, `figs/60k_1818_last_frames.jpg`.
- `tools/fit_v8.py` (in-hand fit from server obs PNGs + steps.jsonl q/fingers),
  `tools/roll_v8.py` (sim rollouts of any ckpt, dumps the views and grip − handle at the close command).
