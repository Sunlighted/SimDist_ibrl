"""Cheap OFFLINE probe (CLAUDE.md §31 early go/no-go, §29-style): does the dynamics
ensemble's geometric disagreement GROW off-distribution? If OOD disagreement ~=
in-distribution disagreement, the members collapsed to the same function and the
MPPI penalty is a near-constant (argmax-invariant no-op) — the §29 value-ensemble
fate. If OOD >> in-dist, the penalty has signal (avoids unconstrained regions).

Loads the current dynens ckpt + a few source demos; measures cross-head std of the
predicted next latent at: in-dist (s,a) / OOD action / OOD (noised) state / far
rollout under random actions.
"""
import sys
import h5py
import numpy as np
import torch

sys.path.insert(0, "/home/Chenyang/ibrl")
from rl.world_model import WorldModel, WorldModelConfig  # noqa: E402

CK = "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt"
DATA = "release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5"
DEV = "cuda"


def load_states(n_demo=60):
    f = h5py.File(DATA, "r")
    data = f["data"]
    S, A = [], []
    for k in list(data.keys())[:n_demo]:
        S.append(np.asarray(data[k]["obs"]["state"]))
        A.append(np.asarray(data[k]["actions"]))
    f.close()
    S = np.concatenate(S, 0)
    A = np.concatenate(A, 0)
    return torch.tensor(S, dtype=torch.float32), torch.tensor(A, dtype=torch.float32)


def main():
    sd = torch.load(CK, map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    cfg = WorldModelConfig(
        encoder_type="latent", latent_dim=512, mlp_hidden_dim=512, num_layer=2,
        discrete_heads=1, num_bins=101, shared_encoder=1, activation="mish",
        simnorm_dim=8, enc_lr_scale=0.3, reward_coef=0.1, value_coef=0.1, horizon=5,
        num_dynamics_heads=5)
    m = WorldModel(37, 7, cfg).to(DEV)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    unexp = [k for k in unexpected if not k.startswith("value")]
    assert not unexp, f"unexpected: {unexp[:5]}"
    m.eval()
    assert m.num_dynamics_heads() == 5

    S, A = load_states()
    idx = torch.randperm(S.shape[0])[:4000]
    S, A = S[idx].to(DEV), A[idx].to(DEV)

    @torch.no_grad()
    def dis(z, a):
        return m.dynamics_disagreement(z, a).mean().item()

    with torch.no_grad():
        z = m.encode(S)                                  # in-dist latent
        d_in = dis(z, A)                                 # (a) in-dist (s,a)
        a_rand = torch.rand_like(A) * 2 - 1              # U[-1,1]
        d_oa = dis(z, a_rand)                            # (b) OOD action
        # (c) OOD state: encode a noised state (push off the data manifold)
        d_os = dis(m.encode(S + 0.5 * torch.randn_like(S)), A)
        # (d) far rollout: roll ensemble mean 15 steps with RANDOM actions -> a
        # genuinely off-distribution imagined latent (what MPPI can wander into)
        zf = z.clone()
        for _ in range(15):
            zf = m.next(zf, torch.rand_like(A) * 2 - 1)
        d_far = dis(zf, torch.rand_like(A) * 2 - 1)      # (d) far OOD latent
        # scale reference: typical per-dim latent value + its spread
        zdim_mean = z.mean().item()
        zdim_std = z.std().item()

    print(f"latent per-dim: mean {zdim_mean:.4f} std {zdim_std:.4f} (SimNorm ~1/8=0.125)")
    print("--- dynamics ensemble geometric disagreement (cross-head std of next latent) ---")
    print(f"(a) in-dist   (s,a)        : {d_in:.5f}")
    print(f"(b) OOD action             : {d_oa:.5f}   ratio vs in-dist {d_oa/d_in:.2f}x")
    print(f"(c) OOD state (+0.5 noise) : {d_os:.5f}   ratio {d_os/d_in:.2f}x")
    print(f"(d) far rollout (15 rand)  : {d_far:.5f}   ratio {d_far/d_in:.2f}x")
    print()
    r = max(d_oa, d_os, d_far) / d_in
    if r < 1.5:
        print(f"VERDICT: COLLAPSE-like — OOD disagreement only {r:.2f}x in-dist. The "
              f"penalty is ~constant across candidates (argmax-invariant, §29 fate). "
              f"Relaunch with dynamics_prior_scale>0 or declare dead.")
    else:
        print(f"VERDICT: SIGNAL — OOD disagreement up to {r:.2f}x in-dist. Heads diverge "
              f"off-manifold; the penalty can steer argmax. Let the run continue; the "
              f"real go/no-go is post-distill Spearman(disagree, true ||zH-sH||).")


if __name__ == "__main__":
    main()
