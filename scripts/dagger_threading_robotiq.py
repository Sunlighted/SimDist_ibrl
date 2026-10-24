"""DAgger-style takeover demos for the Threading Robotiq cell (2026-10-24).

Why: the v5 BC prior is as precise as the expert on its OWN trajectories (1 cm before the ring: 3.1 mm / 0.5 deg vs the
expert's 2.7 mm / 0.6 deg; 88 % success once it gets there) but cannot recover from the states IBRL brings it to:
IBRL arrives 13 mm off the shaft line 2 cm before the ring, and handing over to the prior there drops 0.70 -> 0.30.
The 3000 expert demos never contain such states (the expert is aligned 6 cm before the ring).

What: roll a policy out (IBRL or BC, optional Gaussian action noise), pick K takeover steps t0 ~ U[0, T), rebuild the
state at t0 bit-exactly (seeded reset + replay of the executed actions, checked against the recorded needle pose),
then let the scripted expert take over from that state (ThreadingExpert.takeover) and record obs/actions from t0 to the
end. Successful takeovers are written in collect_threading_robotiq_demos.py's format (one demo per takeover), so
train_bc.py reads them unchanged. The BC-RNN is open-loop from a single obs (Markov), so a demo starting mid-episode
needs no history.

    python scripts/dagger_threading_robotiq.py --weight <ckpt> --mode rl --episodes 100 --k 3 --noise 0.0 \\
        --seed0 2000000 --workers 20 --out_dir DIR/<name>
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_env(env_params, horizon):
    from env.robosuite_wrapper import PixelRobosuite
    p = dict(env_params, episode_length=horizon, camera_names=[], rl_cameras=[], state_only_no_render=True,
             hand_init_perturb_joint_sigma=0.1, end_on_success=True, record_sim_state=True)
    return PixelRobosuite(**p)


def _needle(e):
    return e.sim.data.xpos[e.sim.model.body_name2id("needle_obj_root")].copy()


def run(job):
    import torch
    import train_rl
    import train_bc
    from common_utils import ibrl_utils as utils
    from scripts.threading_robotiq_expert import ThreadingExpert
    weight, mode, seeds, k_per, noise, horizon, expert_steps, policy_env = job
    torch.cuda.set_device(0)
    loader = train_rl.load_model if mode == "rl" else train_bc.load_model
    agent, _, env_params = loader(weight, "cuda")
    if mode == "bc" and os.environ.get("BC_TA"):
        for mod in [agent] + list(agent.modules()):
            if hasattr(mod, "Ta"):
                mod.Ta = int(os.environ["BC_TA"])
    env = _make_env(env_params, horizon)
    out, stats = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        # ---- 1. policy rollout (episode length 400, as in eval), executed actions recorded
        np.random.seed(s)
        obs, _ = env.reset()
        if hasattr(agent, "reset_prior"):
            agent.reset_prior()
        acts, pos, queue, succ, term = [], [], [], False, False
        with torch.no_grad(), utils.eval_mode(agent):
            while not term and len(acts) < policy_env:
                if not queue:
                    chunk = (agent.act_chunk(obs, eval_mode=True) if hasattr(agent, "act_chunk")
                             else agent.act(obs, eval_mode=True).unsqueeze(0))
                    queue = [x for x in chunk]
                a = queue.pop(0)
                a = (a.cpu() if hasattr(a, "cpu") else torch.as_tensor(a)).reshape(-1).float()
                if noise > 0:
                    a = torch.clamp(a + torch.from_numpy(rng.normal(0, noise, a.shape[0]).astype(np.float32)), -1, 1)
                acts.append(a.clone())
                obs, _, term, succ, _ = env.step(a)
                pos.append(_needle(env.env))
        T = len(acts)
        t0s = sorted(set(int(t) for t in rng.integers(0, T, size=k_per)))
        # ---- 2. per t0: rebuild bit-exactly, expert takes over
        for t0 in t0s:
            np.random.seed(s)
            obs, _ = env.reset()
            for t in range(t0):
                obs, _, term, _, _ = env.step(acts[t])
            if t0 > 0:
                assert np.array_equal(_needle(env.env), pos[t0 - 1]), f"replay mismatch seed {s} t0 {t0}"
            ex = ThreadingExpert(env.env)
            phase0 = ex.takeover()
            ep_obs, ep_act, ep_rew = defaultdict(list), [], []
            ok, n = False, 0
            while n < expert_steps and env.time_step < horizon:
                for k, v in obs.items():
                    ep_obs[k].append(v.cpu().numpy())
                a, _ = ex.act()
                ep_act.append(a.astype(np.float32))
                obs, r, term, ok, _ = env.step(torch.from_numpy(a.astype(np.float32)))
                ep_rew.append(float(r))
                n += 1
                if ok or term:
                    break
            ok = bool(ok)
            stats.append(dict(seed=s, t0=t0, T=T, policy_success=bool(succ), phase=phase0, ok=ok, n=n,
                              impact=bool(getattr(env, "table_impact", False))))
            if ok:
                out.append(dict(obs=ep_obs, actions=ep_act, rewards=ep_rew, seed=s, t0=t0, phase=phase0))
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--mode", default="rl", choices=["rl", "bc"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--k", type=int, default=3, help="takeovers per policy episode")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise on the policy's actions")
    ap.add_argument("--seed0", type=int, default=2000000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--policy_len", type=int, default=400)
    ap.add_argument("--expert_steps", type=int, default=350)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    import multiprocessing as mp
    from scripts.collect_threading_robotiq_demos import write, sha256
    import env.threading_variants as tv
    horizon = args.policy_len + args.expert_steps
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as p:
        res = p.map(run, [(args.weight, args.mode, c, args.k, args.noise, horizon, args.expert_steps, args.policy_len)
                          for c in chunks])
    eps = sorted([e for r in res for e in r[0]], key=lambda e: (e["seed"], e["t0"]))
    st = sorted([x for r in res for x in r[1]], key=lambda x: (x["seed"], x["t0"]))
    ok = np.array([x["ok"] for x in st])
    print(f"policy episodes {args.episodes}, policy success {np.mean([x['policy_success'] for x in st]):.3f} "
          f"(per takeover) | takeovers {len(st)}, expert ok {ok.mean():.3f}, mean demo len "
          f"{np.mean([len(e['actions']) for e in eps]):.0f}")
    for ph, c in sorted(Counter(x["phase"] for x in st).items()):
        o = [x["ok"] for x in st if x["phase"] == ph]
        print(f"  takeover phase {ph:8s} n {c:4d}  expert ok {np.mean(o):.3f}")
    for lab, sel in (("policy succeeded", True), ("policy failed", False)):
        o = [x for x in st if x["policy_success"] == sel]
        if o:
            print(f"  {lab:17s} n {len(o):4d}  expert ok {np.mean([x['ok'] for x in o]):.3f}  "
                  f"(t0 >= 0.5T: {np.mean([x['ok'] for x in o if x['t0'] >= 0.5 * x['T']] or [np.nan]):.3f})")
    cell = tv.REAL_CELL_ROBOTIQ_YAML
    meta = dict(env_name="Threading_RealCellRobotiq", sim2real_cell=os.path.relpath(cell), cell_sha256=sha256(cell),
                control_freq=10.0, exec_layer=True, hand_init_perturb_joint_sigma=0.1, episode_length=horizon,
                use_state=True, source_weight=f"dagger takeover: policy {args.weight} ({args.mode}, noise {args.noise}) "
                "-> scripted expert (ThreadingExpert.takeover)", collect_action_noise=0.0,
                scripted_success_rate=float(ok.mean()), shard=args.seed0, takeover_t0=[e["t0"] for e in eps],
                takeover_phase=[e["phase"] for e in eps])
    path = write(args.out_dir, eps, meta)
    with open(os.path.join(args.out_dir, "takeovers.json"), "w") as f:
        json.dump(st, f)
    print(f"{len(eps)} demos -> {path}")


if __name__ == "__main__":
    main()
