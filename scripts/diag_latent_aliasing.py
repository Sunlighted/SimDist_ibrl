"""Is the FROZEN stage-1 latent a good feature space for THIS demo set's state->action map?

Code audit (CLAUDE.md 35): in these adapt runs the demo data enters the pipeline
through exactly ONE path -- `bc_finetune` -> `update_bc_policy` (`dyn_demo_frac`
and `selfdistill` are both off). So the entire 0.32 adapt spread between demo
sets is carried by the policy function those 1000 BC steps produce.

And `policy_action(s) = policy(encode(s))` with `encode` FROZEN and trained only
on SOURCE data. Every statistic measured so far lived in RAW STATE space; the BC
problem lives in the frozen LATENT space. Two demo sets can match on every
raw-state statistic and still differ in whether that fixed feature map SEPARATES
their state->action pairs:

  aliasing:  for each demo transition, take its k nearest neighbours IN LATENT
             space and measure how much their ACTIONS disagree. High aliasing
             means the frozen latent cannot tell those states apart, so BC is
             forced to average -- capping the policy's quality and (crucially)
             leaving its off-manifold behaviour underdetermined.

Reported alongside the raw-state version of the same quantity, so "the latent
aliases these states" can be separated from "these states genuinely have
inconsistent actions".

  latent_alias / state_alias  > 1  =>  the frozen encoder LOSES information that
                                       matters for this set's action map

No env needed.

Usage:
    python scripts/diag_latent_aliasing.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pyrallis
import torch

from rl.sequence_replay import SequenceReplayBuffer, load_demos_to_sequence_buffer
from rl.simdist_agent import SimDistAgent
from train_simdist import MainConfig

STAGE1 = "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1"
SETS = [  # label, demo dir, n, known adapt
    ("orig (LOST ckpt)", "threading_d0_target_dynamics", 20, 0.952),
    ("matched n20", "threading_d0_target_expert_matched_n20", 20, 0.897),
    ("prefix n20", "threading_d0_target_expert_n500", 20, 0.884),
    ("sig002_extra n20", "threading_d0_target_sig002_extra", 20, 0.715),
    ("snapA_200000 n20", "threading_d0_snapA_200000", 20, 0.680),
    ("sig01new n20", "threading_d0_target_sig01new", 20, 0.675),
    ("repro952 n20", "threading_d0_repro952_n20", 20, 0.630),
    ("sig01exp@002 n20", "threading_d0_target_sig01expert_at002", 20, 0.359),
]
KNN = 8


def log(*a):
    print(*a, file=sys.stderr, flush=True)


@torch.no_grad()
def alias(F, A, knn=KNN):
    """mean ||a_i - a_j|| over each point's knn nearest neighbours in F-space,
    normalised by the action set's own overall spread (so it is scale-free)."""
    sd = F.std(0) + 1e-6
    idx = torch.cdist(F / sd, F / sd).topk(knn + 1, largest=False).indices[:, 1:]
    d = float((A[idx] - A[:, None]).norm(dim=2).mean())
    spread = float((A - A.mean(0)).norm(dim=1).mean())
    return d, d / spread


def main():
    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(_REPO, STAGE1, "cfg.yaml")))
    H = cfg.simdist.world_model.horizon
    sd = torch.load(os.path.join(_REPO, STAGE1, "latest.pt"), map_location="cuda")
    ag = SimDistAgent(1, (37,), None, 7, "", cfg.simdist, device="cuda")
    ag.load_state_dict(sd); ag = ag.cuda(); ag.eval()

    log(f"{'set':>20}{'adapt':>7}{'n_tr':>7}"
        f"{'state_alias':>13}{'latent_alias':>14}{'latent/state':>14}{'lat_eff_dim':>12}")
    rows = []
    with torch.no_grad():
        for lab, path, n, ad in SETS:
            b = SequenceReplayBuffer(max_episodes=20000, device="cpu")
            load_demos_to_sequence_buffer(
                b, os.path.join(_REPO, "release/data/mimicgen", path, "dataset.hdf5"), num_data=n)
            S = torch.cat([torch.as_tensor(e["state"][:-1]) for e in b.episodes]).float().cuda()
            A = torch.cat([torch.as_tensor(e["action"]) for e in b.episodes]).float().cuda()
            Z = ag.model.encode(S)
            _, sa = alias(S, A)
            _, la = alias(Z, A)
            # effective dim of the latent set (participation ratio of the cov spectrum)
            Zz = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
            ev = torch.linalg.eigvalsh(torch.cov(Zz.T)).flip(0)
            ev = ev[ev > 1e-10]
            pr = float((ev.sum() ** 2) / (ev ** 2).sum())
            log(f"{lab:>20}{ad:>7.3f}{S.shape[0]:>7}{sa:>13.4f}{la:>14.4f}{la/sa:>14.3f}{pr:>12.2f}")
            rows.append((ad, sa, la, la / sa, pr))
    from scipy import stats
    Y = np.array([r[0] for r in rows])
    for i, k in enumerate(["state_alias", "latent_alias", "latent/state", "lat_eff_dim"], start=1):
        X = np.array([r[i] for r in rows])
        rho, p = stats.spearmanr(X, Y)
        log(f"  {k:>14} vs adapt: Spearman ρ={rho:+.3f} p={p:.4f}  (n={len(X)})")
    os._exit(0)


if __name__ == "__main__":
    main()
