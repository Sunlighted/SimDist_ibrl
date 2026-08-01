"""Does the value-only model-return identify the PRECISE good first action,
and does a longer horizon help?

For real states s -> z0, take the policy action a_pi (the precise good action
that scores 0.70-0.75 in closed loop). Sample K candidate first-actions around
a_pi (+N(0,sigma)). For each, build an H-step sequence (first=candidate, rest=
a_pi broadcast, matching the planner's seed_mean_from_policy) and score it by the
VALUE-ONLY model return (gamma^H V(z_H)). Then ask, per state:
  - how far is argmax_candidate from a_pi?  (0 = value peaks AT the good action)
  - what percentile rank does a_pi itself get among the candidates? (100 = best)
Sweep H. If longer H makes argmax approach a_pi and a_pi's rank ->100, multistep
value is informative; if not, value cannot drive action selection at any H.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import numpy as np
import torch
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as e

WEIGHT = "exps/mimicgen/threading_simdist_polexpert_seed1/frozen_cmp.pt"
EXPERT = "release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5"


def main():
    agent, ws = e.build(WEIGHT, use_planner=0)
    m = agent.model
    disc = 0.99
    K = 256          # candidate first-actions per state
    sigma = 0.15     # search radius (same as the failing V4-refine/V5)
    N = 300          # states

    f = h5py.File(EXPERT, "r")
    keys = list(f["data"].keys())
    S = []
    for k in keys[:60]:
        S.append(f["data"][k]["obs"]["state"][:])
    f.close()
    S = np.concatenate(S, 0)
    idx = np.random.RandomState(0).choice(len(S), N, replace=False)
    states = torch.tensor(S[idx], dtype=torch.float32).cuda()  # [N,obs]

    with torch.no_grad():
        a_pi = m.policy_action(states)                          # [N,act]  precise good action
        for H in [1, 3, 5, 10]:
            fin = disc ** H
            argmax_dist, pi_rank = [], []
            for i in range(N):
                z0 = m.encode(states[i:i+1])                    # [1,z]
                cand = (a_pi[i:i+1] + sigma * torch.randn(K, a_pi.shape[1], device="cuda")).clamp(-1, 1)  # [K,act]
                # include a_pi itself as candidate 0 for ranking
                cand = torch.cat([a_pi[i:i+1], cand], 0)        # [K+1,act]
                z = z0.expand(cand.shape[0], -1).clone()
                for t in range(H):
                    z = m.next(z, cand)                          # broadcast same action over horizon
                ret = m.predict_value(z).squeeze(-1) * fin       # [K+1]  value-only return
                best = ret.argmax().item()
                argmax_dist.append(float((cand[best] - a_pi[i]).abs().mean()))
                # percentile rank of a_pi (candidate 0)
                pi_rank.append(float((ret < ret[0]).float().mean()) * 100)
            print(f"H={H:2d}: mean |argmax - a_pi| = {np.mean(argmax_dist):.3f}  "
                  f"(a_pi |.|~{a_pi.abs().mean():.2f}; 0=value peaks at good action)   "
                  f"a_pi percentile rank = {np.mean(pi_rank):5.1f}/100  "
                  f"(100 = value already ranks the good action best)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
