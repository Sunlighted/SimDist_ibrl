"""Train an ENSEMBLE of value heads V_i(s) offline on the SOURCE replay, so we can
later ask (scripts/diag_value_ensemble_explain.py) whether ensemble DISAGREEMENT
std_i V_i(s) explains the terminal-value error that MPPI's argmax exploits.

Target = MC discounted return-to-go on the source demos (sparse 0/1 reward +
end_on_success => G_t = gamma^(t_success - t) on success demos, 0 otherwise),
computed by a reverse discounted cumsum so it's general.

Ensemble = N independent MLPs, each fit to a BOOTSTRAP resample of the (state, G)
pairs (bagging) with its own random init. Bagging + independent inits is the
honest deep-ensemble recipe: the members agree on the source manifold and diverge
where they must EXTRAPOLATE (OOD) -- exactly the disagreement pessimism relies on.

Saves value_ensemble.pt: {state_dicts, norm_mean, norm_std, cfg}.

Run:
  python scripts/train_value_ensemble.py \
      --replay release/data/mimicgen/threading_d0_source_replay/replay_buffer.hdf5 \
      --out exps/mimicgen/value_ensemble_source_seed1/value_ensemble.pt \
      --num_heads 10 --steps 40000
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


class VNet(nn.Module):
    def __init__(self, obs_dim, hidden=512, n_layer=3):
        super().__init__()
        layers, d = [], obs_dim
        for _ in range(n_layer):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_source_returns(replay_path, gamma):
    """-> states [M,37], returns G [M] (MC discounted return-to-go)."""
    f = h5py.File(replay_path, "r")
    d = f["data"]
    S, G = [], []
    for k in d.keys():
        g = d[k]
        s = g["obs"]["state"][()].astype(np.float32)      # [T,37]
        r = g["rewards"][()].astype(np.float32)           # [T]
        ret = np.zeros_like(r)
        acc = 0.0
        for t in range(len(r) - 1, -1, -1):
            acc = r[t] + gamma * acc
            ret[t] = acc
        S.append(s)
        G.append(ret)
    f.close()
    return np.concatenate(S, 0), np.concatenate(G, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default="release/data/mimicgen/threading_d0_source_replay/replay_buffer.hdf5")
    ap.add_argument("--out", default="exps/mimicgen/value_ensemble_source_seed1/value_ensemble.pt")
    ap.add_argument("--num_heads", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    states_np, rets_np = load_source_returns(args.replay, args.gamma)
    M, obs_dim = states_np.shape
    log(f"[data] {M} transitions, obs_dim={obs_dim}, "
        f"G range [{rets_np.min():.3f}, {rets_np.max():.3f}] mean {rets_np.mean():.3f} "
        f"frac(G>0)={np.mean(rets_np > 1e-4):.2f}")

    mean = states_np.mean(0)
    std = states_np.std(0).clip(1e-6, None)
    states = torch.tensor((states_np - mean) / std, device=dev)
    rets = torch.tensor(rets_np, device=dev)

    heads = [VNet(obs_dim, args.hidden, args.n_layer).to(dev) for _ in range(args.num_heads)]
    opts = [torch.optim.Adam(h.parameters(), lr=args.lr) for h in heads]
    # per-head bootstrap resample (bagging): independent index pool
    boots = [torch.randint(0, M, (M,), device=dev) for _ in range(args.num_heads)]

    for step in range(args.steps):
        losses = []
        for h, opt, boot in zip(heads, opts, boots):
            sel = boot[torch.randint(0, M, (args.batch,), device=dev)]
            pred = h(states[sel])
            loss = torch.mean((pred - rets[sel]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if step % 5000 == 0 or step == args.steps - 1:
            # ensemble disagreement on a random in-distribution minibatch (sanity)
            with torch.no_grad():
                sel = torch.randint(0, M, (2048,), device=dev)
                preds = torch.stack([h(states[sel]) for h in heads], 0)  # [N,2048]
                dis = preds.std(0).mean().item()
            log(f"[step {step}] mse={np.mean(losses):.4f}  in-dist disagree={dis:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dicts": [h.state_dict() for h in heads],
        "norm_mean": mean, "norm_std": std,
        "obs_dim": obs_dim, "hidden": args.hidden, "n_layer": args.n_layer,
        "num_heads": args.num_heads, "gamma": args.gamma,
    }, args.out)
    log(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
