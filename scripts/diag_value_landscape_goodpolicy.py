"""(a) test on the GOODPOLICY WM + TARGET states: can the value surface even
IDENTIFY the good action a_pi among candidates around it, at any horizon?

For target states s -> z0, a_pi = policy_action(s) (the ~0.73 good action).
Sample K candidates = a_pi + N(0,sigma) (the MPPI candidate scale). Score each by
the VALUE-ONLY WM return gamma^H * V(z_H) (broadcast candidate over H). Report per
H: mean |argmax - a_pi| (0 = value peaks AT the good action) and a_pi percentile
rank (100 = value ranks the good action best; 50 = random).

Builds the WM from the stage-1 MainConfig + loads the goodpolicy adapted weights
(the adapt cfg.yaml isn't a MainConfig, so we can't use eval_simdist_policy_only.build).
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import h5py
import pyrallis

WM_CKPT = "exps/mimicgen/threading_simdist_adapt_goodpolicy_seed1/latest.pt"
WM_CFG_DIR = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1"
DATA = "release/data/mimicgen/threading_d0_target_diverse_v2/dataset.hdf5"


def main():
    from train_simdist import MainConfig
    from rl.world_model import WorldModel
    dev = "cuda"
    base_cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(WM_CFG_DIR, "cfg.yaml")))  # type: ignore
    wm = base_cfg.simdist.world_model
    f = h5py.File(DATA, "r"); g = f["data"]
    ids = sorted((k for k in g if k.startswith("demo_")), key=lambda k: int(k.split("_")[1]))
    obs_dim = g[ids[0]]["obs"]["state"].shape[1]
    act_dim = g[ids[0]]["actions"].shape[1]
    m = WorldModel(obs_dim, act_dim, wm).to(dev)
    sd = torch.load(WM_CKPT)
    msd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    m.load_state_dict(msd); m.eval()

    disc = 0.99; K = 256; sigma = 0.15; N = 300
    S = []
    for k in ids:
        S.append(np.array(g[k]["obs"]["state"], np.float32))
        if sum(len(x) for x in S) > 60000:
            break
    S = np.concatenate(S, 0)
    idx = np.random.RandomState(0).choice(len(S), N, replace=False)
    states = torch.tensor(S[idx], dtype=torch.float32).to(dev)

    print(f"=== value LANDSCAPE (a) | goodpolicy WM | TARGET states | N={N} K={K} sigma={sigma} ===")
    with torch.no_grad():
        a_pi = m.policy_action(states)
        for H in [1, 3, 5, 10]:
            fin = disc ** H
            argmax_dist, pi_rank = [], []
            for i in range(N):
                z0 = m.encode(states[i:i+1])
                cand = (a_pi[i:i+1] + sigma * torch.randn(K, a_pi.shape[1], device=dev)).clamp(-1, 1)
                cand = torch.cat([a_pi[i:i+1], cand], 0)
                z = z0.expand(cand.shape[0], -1).clone()
                for t in range(H):
                    z = m.next(z, cand)
                ret = m.predict_value(z).squeeze(-1) * fin
                best = ret.argmax().item()
                argmax_dist.append(float((cand[best] - a_pi[i]).abs().mean()))
                pi_rank.append(float((ret < ret[0]).float().mean()) * 100)
            print(f"H={H:2d}: mean |argmax - a_pi| = {np.mean(argmax_dist):.3f}  "
                  f"(a_pi |.|~{a_pi.abs().mean():.2f}; 0=value peaks at good action)   "
                  f"a_pi percentile rank = {np.mean(pi_rank):5.1f}/100  "
                  f"(100 = value ranks the good action best; 50 = random)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
