"""Step 8 (headline): does the conditional-density DGI stay ~0 on suboptimal /
flailing source states where the OLD H2O discriminator hallucinates a high gap?

Roll out a FLAILING policy in the SOURCE env recording sim_state; the true gap
per transition is the sim2sim counterfactual ||s'_tgt - s'_src|| (target sim
step from the same sim_state). In free-space flailing this is ≈0.
Compare, per transition:
  - GT gap (ground truth)
  - OLD discriminator score |log dyn-ratio|  (expect hallucinated-high on OOD)
  - CDM DGI                                  (expect ≈0, tracking GT)
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from cf_utils import REPO, TARGET_SHIFT, counterfactual_step, get_sim_state, get_env
from data_utils import state_gap, needle_ring_dist
from discriminator import DynamicsDiscriminator
from conditional_density import build_model
from common_utils import ibrl_utils as utils


def load_disc(path, device):
    ck = torch.load(path, map_location=device)
    d = DynamicsDiscriminator(ck["state_dim"], ck["action_dim"], ck["hidden"], ck["dropout"]).to(device)
    d.load_state_dict(ck["state_dict"]); d.eval(); return d


def load_cdm(path, device):
    ck = torch.load(path, map_location=device)
    m = build_model(ck["head"], ck["state_dim"], ck["action_dim"], ck["ds_dim"], hidden=ck["hidden"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


def flailing_rollouts(num_traj, horizon, seed, mode, behavior_weight=None):
    """Roll out in SOURCE env recording (s, a, sim_state). mode:
       'random' = uniform random actions; 'noisy_bc' = flow BC + big action noise."""
    src_env = get_env(REPO + "/exps/mimicgen/threading_ibrl_source_seed1/model0.pt",
                      domain="source", hand_init_sigma=0.1)
    agent = None
    if mode == "noisy_bc":
        import train_bc
        agent, _, _ = train_bc.load_model(behavior_weight, "cuda")
    trajs = []
    ctx = utils.eval_mode(agent) if agent is not None else _nullctx()
    with torch.no_grad(), ctx:
        for ti in range(num_traj):
            np.random.seed(seed + ti)
            obs, _ = src_env.reset()
            S, A, SIM = [], [], []
            for step in range(horizon):
                sim_t = get_sim_state(src_env)
                s_t = obs["state"].cpu().numpy()
                if mode == "random":
                    action = torch.from_numpy(np.random.uniform(-1, 1, 7).astype(np.float32))
                else:
                    a = agent.act(obs, eval_mode=True)
                    action = (a.cpu() + 0.5 * torch.randn(7)).clamp(-1, 1)
                obs, reward, terminal, _, _ = src_env.step(action)
                S.append(s_t); A.append(action.cpu().numpy()); SIM.append(sim_t)
                if terminal:
                    break
            trajs.append((np.asarray(S, np.float32), np.asarray(A, np.float32),
                          np.asarray(SIM, np.float64)))
    return trajs


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--disc", default=f"{os.path.dirname(__file__)}/discriminator_cf.pt")
    p.add_argument("--cdm", default=f"{os.path.dirname(__file__)}/cdm_gaussian.pt")
    p.add_argument("--mode", choices=["random", "noisy_bc"], default="random")
    p.add_argument("--behavior_weight", default=f"{REPO}/exps/mimicgen/threading_flow_bc_seed1/model0.pt")
    p.add_argument("--num_traj", type=int, default=6)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--seed", type=int, default=500)
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/step8_suboptimal.png")
    args = p.parse_args()
    dev = "cuda"

    disc = load_disc(args.disc, dev)
    cdm = load_cdm(args.cdm, dev)
    tgt_env = get_env(REPO + "/exps/mimicgen/threading_ibrl_source_seed1/model0.pt", domain="target")

    trajs = flailing_rollouts(args.num_traj, args.horizon, args.seed, args.mode, args.behavior_weight)

    results = []
    for S, A, SIM in trajs:
        # ground-truth gap via counterfactual target step from each sim_state
        STGT = np.stack([counterfactual_step(tgt_env, SIM[t], A[t])[0] for t in range(len(S))]).astype(np.float32)
        # s'_src = the actual source rollout next state (S[t+1]); align lengths
        SSRC = np.concatenate([S[1:], S[-1:]], 0)  # last step has no next; pad
        n = len(S) - 1
        S, A, SSRC, STGT = S[:n], A[:n], SSRC[:n], STGT[:n]
        gt = state_gap(SSRC, STGT)
        dist = needle_ring_dist(S)
        St, At, SSt = (torch.as_tensor(x, device=dev) for x in (S, A, SSRC))
        with torch.no_grad():
            disc_score = np.abs(disc.log_dynamics_ratio(St, At, SSt).cpu().numpy())
            cdm_dgi = cdm.dgi(St, At, SSt).cpu().numpy()
        results.append(dict(gt=gt, dist=dist, disc=disc_score, dgi=cdm_dgi))

    allgt = np.concatenate([r["gt"] for r in results])
    alldisc = np.concatenate([r["disc"] for r in results])
    alldgi = np.concatenate([r["dgi"] for r in results])
    print(f"mode={args.mode}, {len(allgt)} flailing transitions")
    print(f"  GT gap        : mean {allgt.mean():.4f} median {np.median(allgt):.4f} (≈0 expected free-space)")
    print(f"  OLD disc score: mean {alldisc.mean():.4f}  Spearman(disc,GT)={spearmanr(alldisc,allgt)[0]:.3f}")
    print(f"  CDM DGI       : mean {alldgi.mean():.4f}  Spearman(DGI ,GT)={spearmanr(alldgi ,allgt)[0]:.3f}")
    # normalized: how much does each track GT *shape*? report both raw means on low-gap subset
    lo = allgt < np.quantile(allgt, 0.8)
    print(f"  on low-GT (true≈no-gap) rows: disc mean {alldisc[lo].mean():.4f} vs DGI mean {alldgi[lo].mean():.4f}")

    # ---- plot per-traj (normalized for shape) + overall scatter ----
    n = len(results)
    fig, axes = plt.subplots(2, max(3, (n + 2) // 2 + 1), figsize=(16, 7)); axes = axes.ravel()
    for i, r in enumerate(results):
        ax = axes[i]; t = np.arange(len(r["gt"]))
        for key, c, lab in [("gt", "C3", "GT gap"), ("disc", "C1", "OLD disc"), ("dgi", "C0", "CDM DGI")]:
            v = r[key]; ax.plot(t, v / (v.max() + 1e-9), color=c, lw=1.2, label=lab if i == 0 else None)
        ax.set_title(f"traj{i} (norm)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    ax = axes[-1]
    ax.scatter(allgt, alldisc, s=5, alpha=0.3, color="C1", label="OLD disc")
    ax.scatter(allgt, alldgi, s=5, alpha=0.3, color="C0", label="CDM DGI")
    ax.set_xlabel("GT gap"); ax.set_ylabel("score (raw)"); ax.set_title("score vs GT gap"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
