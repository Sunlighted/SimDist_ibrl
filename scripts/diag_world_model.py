"""Diagnose whether the SimDist world model is good enough for MPPI.

Tests, all env-free on real trajectories:
  A. multi-step dynamics: latent rollout w/ real actions -> decode -> state MSE
     per horizon step (does the model predict the future state accurately?)
  B. reward prediction: predict_reward vs real reward (esp. terminal r=1).
  C. value discrimination: value vs steps-to-success (Spearman) + success-traj
     vs failure-traj value gap. THIS is what MPPI ranks candidates by.
  D. return-ranking sanity (the direct MPPI test): from z0, model-return of the
     REAL expert future actions vs noised-expert vs random. The model SHOULD
     rank expert > noised > random; if not, planning selects garbage.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import argparse
import numpy as np
import torch
import h5py
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as e


def load_trajs(path, n, with_flags=False):
    f = h5py.File(path, "r")
    keys = list(f["data"].keys())[:n]
    out = []
    for k in keys:
        g = f["data"][k]
        out.append(dict(
            state=g["obs"]["state"][:].astype(np.float32),
            action=g["actions"][:].astype(np.float32),
            reward=g["rewards"][:].astype(np.float32),
            success=float(np.sum(g["rewards"][:]) > 0),
        ))
    f.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="exps/mimicgen/threading_simdist_polexpert_seed1/frozen_cmp.pt")
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--expert_path", default="release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5",
                    help="all-success trajs for dynamics/reward/return tests (A/B/D)")
    ap.add_argument("--mixed_path", default="release/data/mimicgen/threading_d0_source_replay/replay_buffer.hdf5",
                    help="success+failure trajs for value discrimination (C)")
    ap.add_argument("--n_expert", type=int, default=200)
    ap.add_argument("--n_mixed", type=int, default=400)
    args = ap.parse_args()
    H = args.H

    agent, ws = e.build(args.weight, use_planner=0)
    m = agent.model
    disc = 0.99

    print(f"expert_path = {args.expert_path}")
    print(f"mixed_path  = {args.mixed_path}")
    exp = load_trajs(args.expert_path, args.n_expert)
    div = load_trajs(args.mixed_path, args.n_mixed)

    # ---------- A. multi-step dynamics (state-space) ----------
    seg_s, seg_a = [], []
    for tr in exp:
        T = len(tr["action"])
        for t0 in range(0, max(1, T - H), 7):
            if t0 + H + 1 <= len(tr["state"]):
                seg_s.append(tr["state"][t0:t0 + H + 1])
                seg_a.append(tr["action"][t0:t0 + H])
    S = torch.tensor(np.stack(seg_s)).cuda()   # [N,H+1,obs]
    A = torch.tensor(np.stack(seg_a)).cuda()   # [N,H,act]
    with torch.no_grad():
        z = m.encode(S[:, 0])
        lat_mse, st_mse = [], []
        for t in range(H):
            z = m.next(z, A[:, t])
            z_true = m.encode(S[:, t + 1])
            lat_mse.append(float(((z - z_true) ** 2).mean()))
            st_mse.append(float(((m.decode(z) - S[:, t + 1]) ** 2).mean()))
        # baseline: how big is a typical state step (s_{t+1}-s_t) and state scale
        step_mse = float(((S[:, 1:] - S[:, :-1]) ** 2).mean())
        state_var = float(S.var())
    print(f"\n[A] multi-step dynamics  (N={S.shape[0]} segments, H={H})")
    print(f"    state scale: var(state)={state_var:.4f}, mean per-step move MSE={step_mse:.4f}")
    for t in range(H):
        print(f"    step {t+1}: rollout->decode state MSE={st_mse[t]:.4f}  "
              f"({100*st_mse[t]/state_var:5.1f}% of state var)   latent MSE={lat_mse[t]:.5f}")

    # ---------- B. reward prediction ----------
    with torch.no_grad():
        # terminal success steps (r=1) vs random non-terminal (r=0)
        z0 = m.encode(S[:, 0])
        # gather (s,a,r) pairs from expert trajs
        sa_s, sa_a, sa_r = [], [], []
        for tr in exp:
            n = min(len(tr["state"]), len(tr["action"]), len(tr["reward"]))
            sa_s.append(tr["state"][:n]); sa_a.append(tr["action"][:n]); sa_r.append(tr["reward"][:n])
        ss = torch.tensor(np.concatenate(sa_s)).cuda()
        aa = torch.tensor(np.concatenate(sa_a)).cuda()
        rr = torch.tensor(np.concatenate(sa_r)).cuda()
        rhat = m.predict_reward(m.encode(ss), aa).squeeze(-1)
        pos = rr > 0.5
        print(f"\n[B] reward prediction  (N={len(rr)} steps, {int(pos.sum())} with r=1)")
        print(f"    r=1 steps: pred mean={float(rhat[pos].mean()):.3f} (target 1.0)")
        print(f"    r=0 steps: pred mean={float(rhat[~pos].mean()):.3f} (target 0.0)")

    # ---------- C. value discrimination ----------
    with torch.no_grad():
        # success trajs: value vs steps-to-success
        vs, steps_to_go = [], []
        for tr in exp:
            T = len(tr["state"])
            v = m.predict_value(m.encode(torch.tensor(tr["state"]).cuda())).squeeze(-1).cpu().numpy()
            vs.append(v); steps_to_go.append(np.arange(T)[::-1])  # T-1..0
        v_all = np.concatenate(vs); stg = np.concatenate(steps_to_go)
        rho = spearmanr(v_all, -stg).correlation  # higher value <-> fewer steps to go
        # success vs failure final-state value (diverse)
        succ_v, fail_v = [], []
        for tr in div:
            vlast = float(m.predict_value(m.encode(torch.tensor(tr["state"][-5:]).cuda())).mean())
            (succ_v if tr["success"] else fail_v).append(vlast)
        print(f"\n[C] value discrimination")
        print(f"    Spearman(value, -steps_to_success) on expert trajs = {rho:+.3f}  (want >>0)")
        print(f"    diverse last-states: success value mean={np.mean(succ_v):.3f} (n={len(succ_v)}), "
              f"failure value mean={np.mean(fail_v):.3f} (n={len(fail_v)})  gap={np.mean(succ_v)-np.mean(fail_v):+.3f}")

    # ---------- D. return-ranking sanity (direct MPPI test) ----------
    with torch.no_grad():
        disc_w = (disc ** torch.arange(H, device="cuda")).view(H, 1)
        fin = disc ** H
        def model_return(z0, acts):  # acts [N,H,act]
            z = z0.clone(); rew = []
            for t in range(H):
                rew.append(m.predict_reward(z, acts[:, t]).squeeze(-1))
                z = m.next(z, acts[:, t])
            rew = torch.stack(rew, 0)  # [H,N]
            return (rew * disc_w).sum(0) + m.predict_value(z).squeeze(-1) * fin
        z0 = m.encode(S[:, 0])
        ret_exp = model_return(z0, A)
        ret_noise = model_return(z0, (A + 0.3 * torch.randn_like(A)).clamp(-1, 1))
        ret_rand = model_return(z0, torch.empty_like(A).uniform_(-1, 1))
        # per-sample: does expert beat random?
        win_rand = float((ret_exp > ret_rand).float().mean())
        win_noise = float((ret_exp > ret_noise).float().mean())
        print(f"\n[D] return-ranking sanity  (model-return of action seqs from same z0, N={z0.shape[0]})")
        print(f"    mean return: expert={float(ret_exp.mean()):.3f}  noised={float(ret_noise.mean()):.3f}  random={float(ret_rand.mean()):.3f}")
        print(f"    P(return_expert > return_random) = {win_rand:.2%}   (want ~100%)")
        print(f"    P(return_expert > return_noised) = {win_noise:.2%}   (want >50%)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
