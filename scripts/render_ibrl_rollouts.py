"""Roll an IBRL (or BC) policy out on the Threading Robotiq cell and render the episodes (2026-10-21).

Same rollout as scripts/diagnose_threading_robotiq_failures.py (greedy, seed per episode, strict, sigma 0.1) with
sim_state recorded; afterwards each episode is re-reset from its seed (same DR) and the recorded states are rendered:
agentview | close-up on the ring | wrist camera, plus an info strip per frame (who won the IBRL argmax, gripper
command, needle lift, tip -> ring distance, tripod displacement, grasp angle). The outcome bucket is recomputed on THIS
rollout (the BC-RNN prior samples, so a seed can come out differently from an earlier run).

    python scripts/render_ibrl_rollouts.py --weight <model0.pt> --seeds 4,6,10 --out viz/ibrl_failures
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_threading_robotiq_failures import _probe, classify  # noqa: E402

H, W = 300, 400
# cv2.putText has no CJK glyphs
EN = {"success": "success", "1": "1 never lifted", "2": "2 dropped after lift", "3": "3 held, never near ring",
      "4": "4 near ring, not through", "5": "5 tripod moved >4cm", "6": "6 raw ok, strict failed",
      "7": "7 TABLE IMPACT (episode ended)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--mode", default="rl", choices=["rl", "bc"])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keyframes", type=int, default=8)
    ap.add_argument("--bc_ta", type=int, default=0, help="mode bc: re-plan every N steps (0 = the policy's own Ta)")
    args = ap.parse_args()
    import cv2
    import imageio
    import mujoco
    import torch
    import train_bc
    import train_rl
    from common_utils import ibrl_utils as utils
    from env.robosuite_wrapper import PixelRobosuite
    agent, _, env_params = (train_rl.load_model if args.mode == "rl" else train_bc.load_model)(args.weight, "cuda")
    if args.mode == "bc" and args.bc_ta > 0:
        agent.Ta = args.bc_ta
    env = PixelRobosuite(**dict(env_params, state_only_no_render=True, hand_init_perturb_joint_sigma=0.1,
                                record_sim_state=True))
    picked = {}
    if args.mode == "rl" and getattr(agent, "bc_policies", None):
        bcp = agent.bc_policies[0]
        _act = bcp.act

        def act(obs, **kw):
            a = _act(obs, **kw)
            picked["bc"] = a.detach().clone()
            return a
        bcp.act = act
    os.makedirs(args.out, exist_ok=True)
    summary = []
    for s in [int(x) for x in args.seeds.split(",")]:
        # ---- rollout
        with torch.no_grad(), utils.eval_mode(agent):
            np.random.seed(s)
            obs, _ = env.reset()
            e = env.env
            m = e.sim.model
            ids = dict(nb=m.body_name2id("needle_obj_root"),
                       rg=[m.geom_name2id(f"tripod_obj_ring_{i}") for i in range(e.tripod.num_ring_geoms)],
                       lf=m.body_name2id("gripper0_left_inner_finger"), rf=m.body_name2id("gripper0_right_inner_finger"))
            pn0, _, c0, _, _ = _probe(e, ids)
            states = [obs["sim_state"].numpy().copy()]
            rec = {k: [] for k in ("lift", "tip_d", "rad", "ang", "tdisp", "raw", "bc", "grip", "gcos", "knuckle")}
            kadr = m.get_joint_qpos_addr("gripper0_finger_joint")
            term, succ_any, queue = False, False, []
            while not term:
                if not queue:
                    chunk = (agent.act_chunk(obs, eval_mode=True) if hasattr(agent, "act_chunk")
                             else agent.act(obs, eval_mode=True).unsqueeze(0))
                    queue = [x for x in chunk]
                a = queue.pop(0)
                a = (a.cpu() if hasattr(a, "cpu") else torch.as_tensor(a)).reshape(-1)
                rec["bc"].append(bool("bc" in picked and torch.allclose(picked["bc"].reshape(-1).cpu(), a)))
                rec["grip"].append(float(a[6]))
                obs, r, term, succ, _ = env.step(a)
                succ_any |= bool(succ)
                states.append(obs["sim_state"].numpy().copy())
                pn, tip, c, rad, ang = _probe(e, ids)
                d = e.sim.data
                fl, fr = d.xpos[ids["lf"]], d.xpos[ids["rf"]]
                ax = (fr - fl) / max(np.linalg.norm(fr - fl), 1e-9)
                rec["gcos"].append(float(abs(ax @ d.xmat[ids["nb"]].reshape(3, 3)[:, 1])))
                rec["lift"].append(float(pn[2] - pn0[2]))
                rec["tip_d"].append(float(np.linalg.norm(tip - c)))
                rec["rad"].append(rad)
                rec["ang"].append(ang)
                rec["tdisp"].append(float(np.linalg.norm(c - c0)))
                rec["raw"].append(bool(e._check_success()))
                rec["knuckle"].append(float(d.qpos[kadr]))
        r = dict(success=succ_any, **{k: np.array(v) for k, v in rec.items()})
        bucket = classify(r)
        if getattr(env, "table_impact", False):      # sim2real cell table_impact_force ended the episode
            bucket = "7 table impact"
        tag = "success" if succ_any else "fail" + bucket.split()[0]
        # ---- render (re-reset from the seed: identical DR; then write the recorded states back)
        np.random.seed(s)
        env.reset()
        sim = env.env.sim
        c_ring = np.mean(sim.data.geom_xpos[ids["rg"]], axis=0)
        normal = sim.data.get_body_xmat("tripod_obj_root")[:, 0]
        rend = mujoco.Renderer(sim.model._model, H, W)
        vis = mujoco.MjvOption()
        vis.geomgroup[0] = 0      # hide collision geoms (robosuite's own renderer does; else the arm shows green)
        close = mujoco.MjvCamera()
        close.type = mujoco.mjtCamera.mjCAMERA_FREE
        close.lookat[:] = c_ring
        close.distance, close.elevation = 0.30, -15.0
        close.azimuth = float(np.degrees(np.arctan2(normal[1], normal[0]))) + 35.0
        frames = []
        for t, st in enumerate(states):
            sim.set_state_from_flattened(st)
            sim.forward()
            views = []
            for cam in ("agentview", close, "robot0_eye_in_hand"):
                rend.update_scene(sim.data._data, camera=cam, scene_option=vis)
                views.append(rend.render().copy())
            fr = np.concatenate(views, 1)
            info = np.full((44, fr.shape[1], 3), 24, np.uint8)
            k = min(t, len(rec["bc"]) - 1)
            who = "BC " if (rec["bc"][k] or args.mode == "bc") else "RL "
            line1 = f"seed {s}  t={t}/{len(states) - 1}  {EN[bucket.split()[0]]}  | next action from {who} grip {rec['grip'][k]:+.2f}"
            line2 = (f"lift {100 * rec['lift'][k]:5.1f} cm  knuckle {rec['knuckle'][k]:.3f}  tip->ring {100 * rec['tip_d'][k]:5.1f} cm  "
                     f"axis {rec['ang'][k]:4.1f} deg  tripod moved {1e3 * rec['tdisp'][k]:4.0f} mm  "
                     f"grasp |cos(fingers,shaft)| {rec['gcos'][k]:.2f}")
            for i, txt in enumerate((line1, line2)):
                cv2.putText(info, txt, (6, 17 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
            frames.append(np.concatenate([fr, info], 0))
        name = f"seed{s:03d}_{tag}"
        imageio.mimwrite(os.path.join(args.out, f"{name}.mp4"), frames, fps=10, quality=6)
        idx = np.linspace(0, len(frames) - 1, args.keyframes).round().astype(int)
        imageio.imwrite(os.path.join(args.out, f"{name}_sheet.jpg"), np.concatenate([frames[i] for i in idx], 0),
                        quality=85)
        line = (f"{name}: {bucket}, {len(states) - 1} steps, lift max {100 * r['lift'].max():.1f} cm, "
                f"tip->ring min {100 * r['tip_d'].min():.1f} cm, tripod moved max {1e3 * r['tdisp'].max():.0f} mm, "
                f"BC chosen {r['bc'].mean():.2f}")
        print(line, flush=True)
        summary.append(line)
    with open(os.path.join(args.out, "README.txt"), "a") as f:
        f.write(f"# {args.weight} (mode {args.mode})\n" + "\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
