"""Where do IBRL (or BC) episodes on the Threading Robotiq cell fail? (2026-10-21)

Rolls the policy out greedily, exactly as evaluate/eval.py does (same env_params from the ckpt, seed = seed0 + i,
strict success), and logs privileged sim quantities every step:
  lift      needle height above its reset height            (grasped / dropped)
  tip_d     needle tip -> ring centre distance
  rad, ang  offset of the ring centre from the needle's shaft line, and shaft-axis vs ring-normal angle
  tdisp     tripod (ring centre) displacement from reset
  raw       raw (non-strict) success;  bc  whether the IBRL argmax picked the prior's action
Each failed episode is then put in ONE bucket by the furthest stage it reached (see classify()).

    python scripts/diagnose_threading_robotiq_failures.py --weight <model0.pt> --mode rl --num 200 --workers 10 --out <npz>
    --grip bc|hold  counterfactual gripper (prior's command / keep closed once lifted) to price the drops
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIP_LOCAL = np.array([0.0, -0.08, 0.0])   # = scripts/threading_robotiq_expert.py


def _probe(e, ids):
    d = e.sim.data
    pn, Rn = d.xpos[ids["nb"]].copy(), d.xmat[ids["nb"]].reshape(3, 3).copy()
    P = np.array([d.geom_xpos[g] for g in ids["rg"]])
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][2]
    tip, axis = pn + Rn @ TIP_LOCAL, Rn[:, 1]
    v = c - tip
    rad = float(np.linalg.norm(v - (v @ axis) * axis))          # ring centre off the shaft line
    ang = float(np.degrees(np.arccos(min(1.0, abs(float(axis @ n))))))
    return pn, tip, c, rad, ang


def run(job):
    import torch
    import train_rl
    import train_bc
    from env.robosuite_wrapper import PixelRobosuite
    weight, mode, seeds, grip = job
    torch.cuda.set_device(0)
    loader = train_rl.load_model if mode == "rl" else train_bc.load_model
    agent, _, env_params = loader(weight, "cuda")
    env_params = dict(env_params, state_only_no_render=True, hand_init_perturb_joint_sigma=0.1)
    env = PixelRobosuite(**env_params)
    picked = {}
    if mode == "rl" and getattr(agent, "bc_policies", None):        # remember the prior's action to tell who won
        bcp = agent.bc_policies[0]
        _act = bcp.act

        def act(obs, **kw):
            a = _act(obs, **kw)
            picked["bc"] = a.detach().clone()
            return a
        bcp.act = act
    out = []
    from common_utils import ibrl_utils as utils
    with torch.no_grad(), utils.eval_mode(agent):
        for s in seeds:
            np.random.seed(s)
            obs, _ = env.reset()
            e = env.env
            m = e.sim.model
            ids = dict(nb=m.body_name2id("needle_obj_root"),
                       rg=[m.geom_name2id(f"tripod_obj_ring_{i}") for i in range(e.tripod.num_ring_geoms)],
                       lf=m.body_name2id("gripper0_left_inner_finger"), rf=m.body_name2id("gripper0_right_inner_finger"))
            pn0, _, c0, _, _ = _probe(e, ids)
            rec = {k: [] for k in ("lift", "tip_d", "rad", "ang", "tdisp", "raw", "bc", "grip", "gcos", "hold_d")}
            term, succ_any, queue = False, False, []
            while not term:
                if not queue:      # = evaluate/eval.py _policy_act_chunk_single (BC-RNN acts in open-loop chunks)
                    chunk = (agent.act_chunk(obs, eval_mode=True) if hasattr(agent, "act_chunk")
                             else agent.act(obs, eval_mode=True).unsqueeze(0))
                    queue = [x for x in chunk]
                a = queue.pop(0)
                a = (a.cpu() if hasattr(a, "cpu") else torch.as_tensor(a)).reshape(-1)
                # diagnostic gripper overrides (counterfactual, not a policy): bc = the prior's gripper command on
                # every step; hold = keep closed once the needle is off the table (privileged)
                if grip == "bc" and "bc" in picked:
                    a = a.clone(); a[6] = picked["bc"].reshape(-1)[6].cpu()
                elif grip == "hold" and rec["lift"] and rec["lift"][-1] > 0.02:
                    a = a.clone(); a[6] = 1.0
                rec["bc"].append(bool("bc" in picked and torch.allclose(picked["bc"].reshape(-1).cpu(), a.reshape(-1))))
                rec["grip"].append(float(a[6]))
                obs, r, term, succ, _ = env.step(a)
                succ_any |= bool(succ)
                pn, tip, c, rad, ang = _probe(e, ids)
                rec["lift"].append(float(pn[2] - pn0[2]))
                rec["tip_d"].append(float(np.linalg.norm(tip - c)))
                rec["rad"].append(rad)
                rec["ang"].append(ang)
                rec["tdisp"].append(float(np.linalg.norm(c - c0)))
                rec["raw"].append(bool(e._check_success()))
                # grasp geometry: |cos| of the finger closing axis vs the needle shaft (0 = across the shaft, the
                # expert's grasp; 1 = along it), and finger-midpoint -> needle-body distance
                d = e.sim.data
                fl, fr = d.xpos[ids["lf"]], d.xpos[ids["rf"]]
                ax = (fr - fl) / max(np.linalg.norm(fr - fl), 1e-9)
                rec["gcos"].append(float(abs(ax @ d.xmat[ids["nb"]].reshape(3, 3)[:, 1])))
                rec["hold_d"].append(float(np.linalg.norm((fl + fr) / 2 - d.xpos[ids["nb"]])))
            out.append(dict(seed=s, success=succ_any, T=env.time_step, **{k: np.array(v) for k, v in rec.items()}))
    return out


LIFT, NEAR = 0.02, 0.03


def classify(r):
    lift, tip_d = r["lift"], r["tip_d"]
    if r["success"]:
        return "success"
    if lift.max() < LIFT:
        return "1 没抓起"
    t_up = int(np.argmax(lift >= LIFT))
    if (lift[t_up:] < 0.005).any() and tip_d.min() >= NEAR:
        return "2 抓起后掉落"
    if r["tdisp"].max() > 0.04:
        return "5 推动了环 (>4 cm)"
    if tip_d.min() >= NEAR:
        return "3 抓着但没到环附近"
    if r["raw"].any():
        return "6 原始判据成立但 strict 没过"
    return "4 到了环附近但没穿过"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--mode", default="rl", choices=["rl", "bc"])
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grip", default="none", choices=["none", "bc", "hold"])
    args = ap.parse_args()
    import multiprocessing as mp
    seeds = list(range(args.seed0, args.seed0 + args.num))
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as p:
        res = [r for part in p.map(run, [(args.weight, args.mode, c, args.grip) for c in chunks]) for r in part]
    res.sort(key=lambda r: r["seed"])
    cls = [classify(r) for r in res]
    print(f"episodes {len(res)}  success {np.mean([r['success'] for r in res]):.3f}")
    for k in sorted(set(cls)):
        print(f"  {k:24s} {cls.count(k):4d}  ({cls.count(k) / len(res):.3f})")
    np.savez_compressed(args.out, res=np.array(res, dtype=object), cls=np.array(cls))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
