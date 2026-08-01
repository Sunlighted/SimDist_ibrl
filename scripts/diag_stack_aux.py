"""Test the user's idea: frame-STACK input + an AUXILIARY next-velocity head.

Motivation (why this is the best-designed velocity variant so far):
  * fd_predict (augment state with vel, predict+ROLL vel) HURT: predicted velocity
    dynamics is hard and its error, fed back, corrupts the pos rollout.
  * This design FIXES that: input = stacked pos frames (implicit velocity); the main
    head predicts Delta-pos and ROLLOUT uses only the pos history (velocity NEVER fed
    back); a separate AUX head predicts the NEXT TRUE velocity qvel_{t+1} -- which is
    the result of the step's contact forces (NOT a finite-diff of the inputs, so it is
    a NON-trivial target that forces the trunk to model contact = the binding constraint).
    The aux loss shapes the shared trunk but does not enter the rollout.

  KEY: qvel_{t+1} is genuine info only because we have TRUE velocity labels (the gather
  logs sim qvel). On position-only data the aux target degenerates to Delta-pos/dt.

Variants (same trunk/lr/steps, only input-stack + aux differ):
  A    : [pos_t, a]                 -> Dpos              (baseline, no velocity)
  S2   : [pos_t,pos_{t-1}, a]        -> Dpos             (stack2, no aux)
  S2A  : [pos_t,pos_{t-1}, a]        -> Dpos + aux qvel  (the idea, stack2)
  S3A  : [pos_t,..,pos_{t-2}, a]     -> Dpos + aux qvel  (the idea, stack3)

Metric: held-out AUTOREGRESSIVE rollout needle MAE (mm) at H=1/3/5, split overall +
contact phase. Beating baseline's needle error (esp. contact) => the aux-contact-force
signal helps; ~equal => it reduces to plain stacking.
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")
from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.domain_shift import DomainShiftConfig  # noqa: E402

ERR = sys.stderr
NEEDLE = slice(9, 12); RING = slice(23, 26)


def log(*a): print(*a, file=ERR, flush=True)


class StackAuxDyn(nn.Module):
    """Trunk over k stacked z-scored pos frames + action; main head -> Dpos(norm),
    optional aux head -> next qvel(norm). Rollout uses ONLY the main head (pos history)."""
    def __init__(self, pos_dim, qvel_dim, adim, stack, pmean, pstd, qmean, qstd, aux, h=256):
        super().__init__()
        self.pos_dim, self.stack, self.aux = pos_dim, stack, aux
        self.register_buffer("pmean", pmean); self.register_buffer("pstd", pstd.clamp_min(1e-6))
        self.register_buffer("qmean", qmean); self.register_buffer("qstd", qstd.clamp_min(1e-6))
        self.trunk = nn.Sequential(nn.Linear(pos_dim * stack + adim, h), nn.ReLU(),
                                   nn.Linear(h, h), nn.ReLU())
        self.main = nn.Linear(h, pos_dim)                       # Dpos (normalized)
        self.auxh = nn.Linear(h, qvel_dim) if aux else None     # next qvel (normalized)

    def np_(self, p): return (p - self.pmean) / self.pstd
    def dp(self, zp): return zp * self.pstd + self.pmean
    def nq(self, q): return (q - self.qmean) / self.qstd

    def forward(self, hist_norm, a):
        """hist_norm: [B, stack, pos_dim] normalized (newest first). Returns
        (next_pos_norm[B,pos_dim], aux_qvel_norm or None)."""
        x = torch.cat([hist_norm.reshape(hist_norm.shape[0], -1), a], -1)
        f = self.trunk(x)
        dpos = self.main(f)                                     # normalized Dpos
        next_pos = hist_norm[:, 0] + dpos
        aux = self.auxh(f) if self.aux else None
        return next_pos, aux


def gather(args):
    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1; cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/stackaux_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, args.ckpt)))
    model = ws.agent.to("cuda").model
    ep = dict(ws.eval_env_params)
    if args.shift == "source":
        ep["domain_shift"] = DomainShiftConfig(mode="none")
    log(f"[setup] shift={args.shift} ({ep['domain_shift'].mode})")
    env = PixelRobosuite(**ep)
    ep_len = int(ws.eval_env_params.get("episode_length", 400))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    episodes = []
    for e in range(args.episodes):
        obs, _ = env.reset(); pos_l, qvel_l, act_l = [], [], []
        for t in range(ep_len):
            qvel = env.env.sim.get_state().qvel.copy()
            with torch.no_grad():
                a = model.policy_action(obs["state"].view(1, -1)).view(-1).clamp(-1, 1)
            pos_l.append(obs["state"].cpu().numpy()); qvel_l.append(qvel)
            act_l.append(a.detach().cpu().numpy())
            obs, _, term, _, _ = env.step(a.detach().cpu())
            if term: break
        if len(pos_l) >= args.horizon + 3:
            episodes.append(dict(pos=np.array(pos_l, np.float32), qvel=np.array(qvel_l, np.float32),
                                 act=np.array(act_l, np.float32)))
    log(f"[gather] {len(episodes)} episodes, {sum(len(e['pos']) for e in episodes)} tr | "
        f"qvel_dim={episodes[0]['qvel'].shape[1]}")
    return episodes


def build_segs(eps, stack, H, dev):
    pad = stack - 1
    P, Q, A = [], [], []
    for e in eps:
        pos, qv, ac = e["pos"], e["qvel"], e["act"]
        T = len(pos)
        for i in range(pad, T - H):
            P.append(pos[i - pad:i + H + 1])     # [pad+H+1, pos]
            Q.append(qv[i:i + H + 1])            # [H+1, qvel]  (qvel_{t+1} = Q[.,1:])
            A.append(ac[i:i + H])
    return (torch.tensor(np.array(P), device=dev), torch.tensor(np.array(Q), device=dev),
            torch.tensor(np.array(A), device=dev))


def run_variant(name, stack, aux, tr, te, pos_dim, qvel_dim, adim, args, dev="cuda"):
    allp = np.concatenate([e["pos"] for e in tr], 0); allq = np.concatenate([e["qvel"] for e in tr], 0)
    pm, ps = torch.tensor(allp.mean(0), device=dev), torch.tensor(allp.std(0), device=dev)
    qm, qs = torch.tensor(allq.mean(0), device=dev), torch.tensor(allq.std(0), device=dev)
    net = StackAuxDyn(pos_dim, qvel_dim, adim, stack, pm, ps, qm, qs, aux).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    P, Q, A = build_segs(tr, stack, args.horizon, dev)
    Pte, Qte, Ate = build_segs(te, stack, args.horizon, dev)
    n = P.shape[0]; pad = stack - 1; H = args.horizon
    for it in range(args.num_steps):
        idx = torch.randint(0, n, (args.batch,), device=dev)
        seg, qseg, act = P[idx], Q[idx], A[idx]
        zp = net.np_(seg)                                       # normalized pos
        hist = torch.stack([zp[:, pad - k] for k in range(stack)], 1)
        loss = 0.0; w = 1.0
        for t in range(H):
            npos, aq = net(hist, act[:, t])
            loss = loss + w * ((npos - zp[:, pad + t + 1]) ** 2).mean()
            if aux:
                loss = loss + args.aux_weight * w * ((aq - net.nq(qseg[:, t + 1])) ** 2).mean()
            w *= args.rho
            hist = torch.cat([npos.unsqueeze(1), hist[:, :-1]], 1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 20.0); opt.step()
    # held-out rollout needle MAE (mm) at each horizon + contact/free split
    net.eval()
    with torch.no_grad():
        zp = net.np_(Pte); pad = stack - 1
        hist = torch.stack([zp[:, pad - k] for k in range(stack)], 1)
        # phase from the start state's needle-ring dist
        start_pos = Pte[:, pad]
        d = torch.norm(start_pos[:, NEEDLE] - start_pos[:, RING], dim=1).cpu().numpy()
        lo, hi = np.percentile(d, 15), np.percentile(d, 70)
        cm, fm = d < lo, d > hi
        errs = []
        for t in range(H):
            npos, _ = net(hist, Ate[:, t])
            pred = net.dp(npos); true = Pte[:, pad + t + 1]
            nd = (torch.norm(pred[:, NEEDLE] - true[:, NEEDLE], dim=1) * 1000.0).cpu().numpy()
            errs.append((nd.mean(), nd[cm].mean(), nd[fm].mean()))
            hist = torch.cat([npos.unsqueeze(1), hist[:, :-1]], 1)
    return errs   # list of (all, contact, free) per horizon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--shift", default="target", choices=["source", "target"])
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--num_steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--aux_weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    eps = gather(args)
    rng = np.random.RandomState(args.seed); n_te = max(1, len(eps) // 5)
    te_ids = set(rng.choice(len(eps), n_te, replace=False))
    tr = [e for i, e in enumerate(eps) if i not in te_ids]
    te = [e for i, e in enumerate(eps) if i in te_ids]
    pos_dim = eps[0]["pos"].shape[1]; qvel_dim = eps[0]["qvel"].shape[1]; adim = eps[0]["act"].shape[1]
    H = args.horizon
    log(f"[split] train {len(tr)} / test {len(te)} | aux_weight={args.aux_weight} H={H}\n")

    variants = [("A  [pos]", 1, False), ("S2 [stack2]", 2, False),
                ("S2A[stack2+aux]", 2, True), ("S3A[stack3+aux]", 3, True)]
    hdr = "".join(f"{'H='+str(h):>22s}" for h in (1, 3, H))
    log(f"{'variant':18s}{hdr}   (needle MAE mm: all/contact/free)")
    for name, stack, aux in variants:
        e = run_variant(name, stack, aux, tr, te, pos_dim, qvel_dim, adim, args)
        cells = "".join(f"{e[h][0]:>7.2f}/{e[h][1]:>5.2f}/{e[h][2]:>5.2f}" for h in (0, 2, H - 1))
        log(f"{name:18s}{cells}")
    log("\n===== READ =====")
    log("  S2A/S3A contact << A contact  => aux next-qvel head (contact-force signal) helps; pursue.")
    log("  S2A ~= S2                     => aux head adds nothing beyond stacking; velocity line closed.")
    os._exit(0)


if __name__ == "__main__":
    main()
