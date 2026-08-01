"""Joint H-step dynamics vs autoregressive — does one-shot trajectory prediction
beat recursive rollout, especially in the contact phase?

Joint model: F([s0, a_{0:H-1}]) -> \hat{s}_{1:H} predicted JOINTLY in one forward
(D-MPC/Valdi style), an MLP of the same family/capacity as the autoregressive
world model. Trained on the SAME data / steps / horizon / rho as the ensemble
head `threading_dyn_ens_seed1` (100k, num_data 20000) for a controlled comparison.

At the end, evaluates BOTH the joint model and the autoregressive head on a
HELD-OUT set (trajectories 20000:40000, unseen by both) and reports per-step
needle error binned by the true target state's phase (free/approach/contact).
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from rl.world_model import WorldModel, WorldModelConfig
from rl.sequence_replay import SequenceReplayBuffer, load_source_replay_to_sequence_buffer

NEEDLE = slice(9, 12); RING = slice(23, 26)
def log(*a): print(*a, file=sys.stderr, flush=True)


class JointDyn(nn.Module):
    """[s0, a_{0:H-1}] -> normalized \hat{s}_{1:H}  (one-shot)."""
    def __init__(self, obs_dim, act_dim, H, hidden=512, nlayer=2, layer_norm=True):
        super().__init__()
        self.obs_dim, self.act_dim, self.H = obs_dim, act_dim, H
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("std", torch.ones(obs_dim))
        d_in, d_out = obs_dim + H * act_dim, H * obs_dim
        layers, d = [], d_in
        for _ in range(nlayer):
            layers += [nn.Linear(d, hidden)]
            if layer_norm: layers += [nn.LayerNorm(hidden)]
            layers += [nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def set_norm(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, s0, aseq):  # s0 [B,obs], aseq [B,H,act] -> [B,H,obs] normalized
        z0 = (s0 - self.mean) / self.std
        x = torch.cat([z0, aseq.reshape(aseq.shape[0], -1)], dim=1)
        return self.net(x).view(-1, self.H, self.obs_dim)

    def norm(self, s):
        return (s - self.mean) / self.std

    def denorm_needle(self, z_slice):  # z_slice [...,3] normalized needle -> meters
        return z_slice * self.std[NEEDLE] + self.mean[NEEDLE]


def load_range(path, lo, hi):
    """Load trajectories [lo:hi) via a temporary full load then slice episodes."""
    buf = SequenceReplayBuffer(max_episodes=10_000_000)
    load_source_replay_to_sequence_buffer(buf, path, num_data=hi)
    sub = SequenceReplayBuffer(max_episodes=hi - lo + 2)
    for ep in buf.episodes[lo:hi]:
        sub.add_episode(ep["state"], ep["action"], ep["reward"], success=ep.get("success", False))
    return sub


@torch.no_grad()
def eval_by_phase(joint, auto, val, H, dev, n_seg=40000):
    S, A = val.sample(n_seg, H, dev)[:2]  # [B,H+1,obs], [B,H,act]
    std = auto.norm_state_std
    # autoregressive rolled
    auto_err = torch.zeros(S.shape[0], H, device=dev)
    z = auto.encode(S[:, 0])
    for t in range(H):
        z = auto.next(z, A[:, t])
        auto_err[:, t] = (auto.decode(z)[:, NEEDLE] - S[:, t + 1, NEEDLE]).norm(dim=1) * 1000
    # joint one-shot
    zj = joint(S[:, 0], A)                        # [B,H,obs] normalized
    needle_pred = joint.denorm_needle(zj[:, :, NEEDLE])  # [B,H,3] meters
    joint_err = (needle_pred - S[:, 1:, NEEDLE]).norm(dim=2) * 1000  # [B,H]
    dist = (S[:, 1:, NEEDLE] - S[:, 1:, RING]).norm(dim=2)  # [B,H]
    phases = {"free(>=15cm)": dist >= 0.15, "approach": (dist >= 0.05) & (dist < 0.15),
              "contact(<5cm)": dist < 0.05}
    ae, je, dist = auto_err.cpu().numpy(), joint_err.cpu().numpy(), dist.cpu().numpy()
    log(f"\n==== HELD-OUT needle error by phase (RMS mm), auto vs joint ====")
    log(f"{'phase':16s} {'n':>8s} {'auto_Hstep':>11s} {'joint_Hstep':>12s} {'winner':>8s}")
    for nm, m in phases.items():
        m = m.cpu().numpy(); n = int(m.sum())
        if n < 100: log(f"{nm:16s} {n:>8d}  too few"); continue
        a = float(np.sqrt((ae[m] ** 2).mean())); j = float(np.sqrt((je[m] ** 2).mean()))
        log(f"{nm:16s} {n:>8d} {a:11.2f} {j:12.2f} {'JOINT' if j < a else 'auto':>8s}")
    # per rollout-depth in contact
    cm = (torch.from_numpy(dist) < 0.05).numpy()
    log("\ncontact-phase error by depth t:  auto vs joint")
    for t in range(H):
        sel = cm[:, t]; n = int(sel.sum())
        if n < 30: continue
        a = float(np.sqrt((ae[sel, t] ** 2).mean())); j = float(np.sqrt((je[sel, t] ** 2).mean()))
        log(f"  t={t+1}: n={n:>6d}  auto={a:6.2f}  joint={j:6.2f}")
    # overall
    log(f"\noverall RMS: auto={float(np.sqrt((ae**2).mean())):.2f}  joint={float(np.sqrt((je**2).mean())):.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5")
    p.add_argument("--auto_ckpt", default="exps/mimicgen/threading_dyn_ens_seed1/best.pt")
    p.add_argument("--num_data", type=int, default=20000)
    p.add_argument("--num_steps", type=int, default=100000)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--nlayer", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save_dir", default="exps/mimicgen/threading_dyn_joint_seed1")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    # train data = first num_data (same as ens heads); held-out = [num_data:2*num_data]
    train = SequenceReplayBuffer(max_episodes=10_000_000)
    load_source_replay_to_sequence_buffer(train, args.data_path, num_data=args.num_data)
    obs_dim = train.episodes[0]["state"].shape[1]; act_dim = train.episodes[0]["action"].shape[1]
    mean, std = train.state_stats()

    joint = JointDyn(obs_dim, act_dim, args.horizon, args.hidden, args.nlayer).to(dev)
    joint.set_norm(mean, std)
    log(f"[joint] in={obs_dim + args.horizon*act_dim} -> out={args.horizon*obs_dim}, "
        f"params={sum(p.numel() for p in joint.parameters())/1e3:.1f}K")
    opt = torch.optim.Adam(joint.parameters(), lr=args.lr)
    H, rho = args.horizon, args.rho

    pbar = tqdm(total=args.num_steps, desc="joint", dynamic_ncols=True, mininterval=2.0)
    for step in range(1, args.num_steps + 1):
        S, A = train.sample(args.batch_size, H, dev)[:2]
        pred = joint(S[:, 0], A)                       # [B,H,obs] normalized
        tgt = joint.norm(S[:, 1:])                     # [B,H,obs] normalized
        w = (rho ** torch.arange(H, device=dev)).view(1, H, 1)
        loss = (w * (pred - tgt) ** 2).sum() / (w.sum() * pred.shape[0] * obs_dim)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(joint.parameters(), 20.0); opt.step()
        pbar.update(1)
        if step % 4000 == 0: pbar.set_postfix(loss=f"{loss.item():.4f}")
    pbar.close()
    torch.save(joint.state_dict(), os.path.join(args.save_dir, "joint.pt"))

    # ---- comparison on held-out ----
    auto = WorldModel(obs_dim, act_dim, WorldModelConfig(
        encoder_type="identity", mlp_hidden_dim=512, num_layer=2, layer_norm=1,
        horizon=H, rho=rho, activation="relu")).to(dev)
    auto.load_state_dict(torch.load(args.auto_ckpt, map_location=dev), strict=False); auto.eval()
    val = load_range(args.data_path, args.num_data, 2 * args.num_data)
    log(f"[heldout] {val.size()} unseen trajectories")
    joint.eval()
    eval_by_phase(joint, auto, val, H, dev)
    os._exit(0)


if __name__ == "__main__":
    main()
