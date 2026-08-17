"""Did stage-3 adapt actually change the Coffee world model, and in which direction?

CONTEXT. On Coffee_D0 / target C50, 100k transitions of stage-3 dynamics adapt
bought nothing once each checkpoint was evaluated near its OWN planner optimum
(zero-adapt 87% vs adapted 81-84%). This script asks the prerequisite question
the planner scores cannot answer: did the dynamics get better, worse, or just
different -- and where.

It measures H-step latent rollout error against the encoder's own encoding of
the real next states, on TARGET transitions, for any set of checkpoints:

    err_H = || f^H(z_t, a_t..a_{t+H-1}) - encode(s_{t+H}) ||   (mean over dims)

reported at H = 1..5, on two disjoint state pools:
  * demo pool     -- the 20 stage-2 demos (in-distribution for stage-2/3)
  * holdout pool  -- demos collected by a DIFFERENT policy (the weak/sat sets),
                     i.e. states the adapt run never visited

⚠️ READ THIS BEFORE TRUSTING ANY NUMBER HERE (CLAUDE.md 30):
prediction error does NOT predict planner score -- on Threading, adapt raised
held-out needle error 5x (1.69mm -> 8.37mm) while MPPI success went UP. So this
script cannot say "adapt helped/hurt". What it CAN say is:
  (a) whether adapt moved the model at all (if err is unchanged, stage-3 is a
      no-op and the search should move elsewhere);
  (b) whether it traded in-distribution accuracy for out-of-distribution error
      (the catastrophic-forgetting signature), which is the leading hypothesis
      for why a wider planner (init_std 2.0-2.5) becomes optimal after adapt.

Usage
  python scripts/diag_coffee_dyn_error.py \
      --ckpts stage1=exps/.../coffee_simdist_diversev1_h5_seed1/latest.pt \
              r0=exps/.../coffee_r0_anchor0.3/latest.pt \
              adapted=exps/.../coffee_adapt_C50_weak_a0.3_seed1/latest.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import pyrallis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


DEMOS = {
    "demo(400k, used by stage-2)":
        "release/data/mimicgen/coffee_d0_target_C50/dataset.hdf5",
    "holdout(weak200k)":
        "release/data/mimicgen/coffee_d0_target_C50_weak200k/dataset.hdf5",
    "holdout(sat1000k)":
        "release/data/mimicgen/coffee_d0_target_C50_sat1000k/dataset.hdf5",
}


def load_segments(path, H, max_seg=4000, seed=0):
    """(states[N,H+1,obs], actions[N,H,act]) sampled from a demo hdf5."""
    import h5py
    rng = np.random.RandomState(seed)
    S, A = [], []
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            g = f["data"][k]
            s = np.asarray(g["obs"]["state"], dtype=np.float32)
            a = np.asarray(g["actions"], dtype=np.float32)
            n = min(len(s) - 1, len(a))
            for t in range(0, n - H):
                S.append(s[t:t + H + 1])
                A.append(a[t:t + H])
    S, A = np.asarray(S), np.asarray(A)
    if len(S) > max_seg:
        idx = rng.choice(len(S), max_seg, replace=False)
        S, A = S[idx], A[idx]
    return torch.as_tensor(S), torch.as_tensor(A)


def rollout_err(model, S, A, H, device="cuda", bs=512):
    """Per-horizon mean |f^h - encode(s_h)| over latent dims."""
    out = np.zeros(H)
    n = 0
    for i in range(0, len(S), bs):
        s = S[i:i + bs].to(device)
        a = A[i:i + bs].to(device)
        with torch.no_grad():
            z = model.encode(s[:, 0])
            for h in range(H):
                z = model.next(z, a[:, h])
                tgt = model.encode(s[:, h + 1])
                out[h] += float((z - tgt).abs().mean(dim=-1).sum())
        n += len(s)
    return out / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="name=path pairs")
    ap.add_argument("--stage1_cfg_dir",
                    default="exps/mimicgen/coffee_simdist_diversev1_h5_seed1")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--max_seg", type=int, default=4000)
    args = ap.parse_args()

    import train_simdist
    from rl.simdist_agent import SimDistAgent

    cfg = pyrallis.load(train_simdist.MainConfig,  # type: ignore
                        open(os.path.join(args.stage1_cfg_dir, "cfg.yaml")))
    obs_shape, prop_shape, act_dim = (66,), (9,), 7

    pools = {}
    for name, path in DEMOS.items():
        if os.path.exists(path):
            pools[name] = load_segments(path, args.horizon, args.max_seg)
            log(f"[pool] {name}: {len(pools[name][0])} segments")

    log("\n" + "=" * 78)
    hdr = f"{'ckpt':<28}{'pool':<28}" + "".join(
        f"{'H=' + str(h + 1):>9}" for h in range(args.horizon))
    log(hdr)
    log("-" * len(hdr))

    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        agent = SimDistAgent(True, obs_shape, prop_shape, act_dim,
                             cfg.rl_camera, cfg.simdist)
        agent.load_state_dict(torch.load(path, map_location="cpu"))
        agent = agent.to("cuda")
        agent.train(False)
        for p in agent.parameters():
            p.requires_grad_(False)
        for pname, (S, A) in pools.items():
            e = rollout_err(agent.model, S, A, args.horizon)
            log(f"{name:<28}{pname:<28}" + "".join(f"{v:9.5f}" for v in e))
        del agent
        torch.cuda.empty_cache()

    log("\nCAVEAT (CLAUDE.md 30): rollout error does NOT predict planner score. "
        "Use this only to tell whether adapt MOVED the model, and whether it "
        "traded in-distribution accuracy for out-of-distribution error.")


if __name__ == "__main__":
    main()
