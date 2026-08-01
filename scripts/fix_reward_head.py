"""Refit ONLY the reward head with a pos-weighted MSE so it stops collapsing to
0 on the rare success steps. Encoder/dynamics/value/policy/decoder are frozen;
the reward head learns r=reward_head([encode(s), a]) on (s,a,r) from the offline
buffer. Saves a new ckpt next to the input.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import argparse
import numpy as np
import torch
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as ev


def load_sar(paths, max_traj):
    S, A, R = [], [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            keys = list(f["data"].keys())[:max_traj]
            for k in keys:
                g = f["data"][k]
                n = min(len(g["actions"]), len(g["rewards"]))
                S.append(np.asarray(g["obs"]["state"][:n], np.float32))
                A.append(np.asarray(g["actions"][:n], np.float32))
                R.append(np.asarray(g["rewards"][:n], np.float32))
    return (np.concatenate(S), np.concatenate(A), np.concatenate(R))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="exps/mimicgen/threading_simdist_polexpert_seed1/frozen_cmp.pt")
    ap.add_argument("--out", default="exps/mimicgen/threading_simdist_polexpert_seed1/frozen_rewardfix.pt")
    ap.add_argument("--pos_weight", type=float, default=50.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_traj", type=int, default=8000)
    args = ap.parse_args()

    agent, ws = ev.build(args.weight, use_planner=0)
    m = agent.model
    paths = ["release/data/mimicgen/threading_d0_source_diverse/dataset.hdf5",
             "release/data/mimicgen/threading_d0_source_sigma0.1_n5000/dataset.hdf5"]
    S, A, R = load_sar(paths, args.max_traj)
    pos = (R > 0.5).mean()
    print(f"loaded {len(R)} transitions, success-step fraction = {pos*100:.2f}%")
    S = torch.tensor(S).cuda(); A = torch.tensor(A).cuda(); R = torch.tensor(R).cuda()

    # freeze everything, train only the reward head
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.reward.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(m.reward.parameters(), lr=args.lr)

    N = len(R)
    for it in range(args.steps):
        idx = torch.randint(0, N, (args.batch,), device="cuda")
        with torch.no_grad():
            z = m.encode(S[idx])
        pred = m.predict_reward(z, A[idx])
        r = R[idx]
        err = (pred - r) ** 2
        wt = torch.where(r > 0.5, args.pos_weight, 1.0)
        loss = (err * wt).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 2000 == 0:
            with torch.no_grad():
                p_all = m.predict_reward(m.encode(S), A)
                m1 = float(p_all[R > 0.5].mean()); m0 = float(p_all[R < 0.5].mean())
                # success-step ranking quality
                from scipy.stats import spearmanr
                rho = spearmanr(p_all.cpu().numpy(), R.cpu().numpy()).correlation
            print(f"step {it+1}: loss={loss.item():.4f}  pred(r=1)={m1:.3f}  pred(r=0)={m0:.3f}  "
                  f"spearman(pred,r)={rho:+.3f}")

    torch.save(agent.state_dict(), args.out)
    print(f"saved {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
