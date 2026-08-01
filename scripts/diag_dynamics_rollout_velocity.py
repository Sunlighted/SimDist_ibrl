"""Do velocity / frame-stacking reduce the DYNAMICS model's MULTI-STEP rollout error?

Follow-up to diag_velocity_recovers.py (single-step MLP proxy) — now the real thing:
train a residual dynamics model (same class as train_dynamics.py: z-scored raw state,
z' = z + MLP([z,a]), multi-step rho^t consistency) per input variant, and measure
HELD-OUT AUTOREGRESSIVE rollout needle-position error at H=1/3/5.

Variants (all trained/rolled on the SAME gathered episodes, same MLP/lr/steps):
  A  : predict D[pos(37)]                 from [pos, a]              -- current obs
  B1 : predict D[pos, robo_vel(9)]        from [pos, robo_vel, a]    -- realistic velocity
  B2 : predict D[pos, full_qvel]          from [pos, full_qvel, a]   -- max velocity info
  B3 : predict D[pos]                     from [pos_t, pos_{t-1}, a] -- 2-frame stack
  B4 : predict D[pos]                     from [pos_t,pos_{t-1},pos_{t-2}, a] -- 3-frame stack

KEY difference vs the single-step proxy: to ROLL a velocity-conditioned model (B1/B2)
autoregressively, the model must ALSO predict the next velocity (fed back into itself)
-- so velocity dynamics (= accelerations/contact forces, itself hard) re-enters the
error. This is exactly the multi-step question: does velocity's single-step gain survive
compounding?  needle = pos[9:12].
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
NEEDLE = slice(9, 12)


def log(*a): print(*a, file=ERR, flush=True)


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class ResidualDyn(nn.Module):
    """z' = z + MLP([z, a]) in z-scored state space; needle read from the pos block."""
    def __init__(self, sdim, adim, mean, std, stack=1, hidden=256):
        super().__init__()
        self.sdim, self.stack = sdim, stack
        self.register_buffer("mean", mean)      # [sdim]
        self.register_buffer("std", std.clamp_min(1e-6))
        self.net = mlp(sdim * stack + adim, sdim, hidden)   # input: stacked z + a -> Dz (of the FULL state)

    def norm(self, s): return (s - self.mean) / self.std
    def denorm(self, z): return z * self.std + self.mean

    def step(self, z_hist, a):
        """z_hist: [B, stack, sdim] (normalized, newest first). Returns next z (normalized)."""
        x = torch.cat([z_hist.reshape(z_hist.shape[0], -1), a], -1)
        return z_hist[:, 0] + self.net(x)


def gather(args):
    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1; cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/dynroll_ws"
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
        obs, _ = env.reset(); pos_l, rvel_l, qvel_l, act_l = [], [], [], []
        for t in range(ep_len):
            raw = env.env._get_observations(force_update=True)
            rvel = np.concatenate([np.asarray(raw["robot0_joint_vel"]).ravel(),
                                   np.asarray(raw["robot0_gripper_qvel"]).ravel()])
            qvel = env.env.sim.get_state().qvel.copy()
            with torch.no_grad():
                a = model.policy_action(obs["state"].view(1, -1)).view(-1).clamp(-1, 1)
            pos_l.append(obs["state"].cpu().numpy()); rvel_l.append(rvel)
            qvel_l.append(qvel); act_l.append(a.detach().cpu().numpy())
            obs, _, term, _, _ = env.step(a.detach().cpu())
            if term: break
        if len(pos_l) >= args.horizon + 2:
            episodes.append(dict(pos=np.array(pos_l, np.float32), rvel=np.array(rvel_l, np.float32),
                                 qvel=np.array(qvel_l, np.float32), act=np.array(act_l, np.float32)))
    log(f"[gather] {len(episodes)} usable episodes, "
        f"{sum(len(e['pos']) for e in episodes)} transitions | qvel dim={episodes[0]['qvel'].shape[1]}")
    return episodes


def make_state(episodes, kind):
    """Return per-episode augmented-state arrays [T, sdim] and pos-block index (needle at 9:12)."""
    out = []
    for e in episodes:
        if kind == "A":   s = e["pos"]
        elif kind == "B1": s = np.concatenate([e["pos"], e["rvel"]], 1)
        elif kind == "B2": s = np.concatenate([e["pos"], e["qvel"]], 1)
        else:             s = e["pos"]   # B3/B4 stack handled via stack>1 on the pos state
        out.append((s.astype(np.float32), e["act"]))
    return out


def train_and_eval(data_tr, data_te, sdim, adim, stack, H, args, dev="cuda"):
    allS = np.concatenate([s for s, _ in data_tr], 0)
    mean = torch.tensor(allS.mean(0), device=dev); std = torch.tensor(allS.std(0), device=dev)
    net = ResidualDyn(sdim, adim, mean, std, stack=stack).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    # segments [n, H+1, sdim] and actions [n, H, adim], plus (stack-1) frames of history before start
    def build_segs(data, need_pad=stack - 1):
        S, A = [], []
        for s, a in data:
            T = len(s)
            for i in range(need_pad, T - H):
                S.append(s[i - need_pad:i + H + 1]); A.append(a[i:i + H])
        return (torch.tensor(np.array(S), device=dev), torch.tensor(np.array(A), device=dev))
    Str, Atr = build_segs(data_tr); Ste, Ate = build_segs(data_te)
    n = Str.shape[0]
    for it in range(args.num_steps):
        idx = torch.randint(0, n, (args.batch,), device=dev)
        seg, act = Str[idx], Atr[idx]                     # seg [B, pad+H+1, sdim]
        z = net.norm(seg)                                  # normalized
        pad = stack - 1
        hist = [z[:, pad - k] for k in range(stack)]       # newest first
        hist = torch.stack(hist, 1)                        # [B, stack, sdim]
        loss = 0.0; w = 1.0
        for t in range(H):
            zpred = net.step(hist, act[:, t])
            ztrue = z[:, pad + t + 1]
            loss = loss + w * ((zpred - ztrue) ** 2).mean(); w *= args.rho
            hist = torch.cat([zpred.unsqueeze(1), hist[:, :-1]], 1)   # shift in prediction
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 20.0); opt.step()
    # ---- held-out autoregressive rollout: needle MAE (mm) at each horizon ----
    net.eval()
    with torch.no_grad():
        z = net.norm(Ste); pad = stack - 1
        hist = torch.stack([z[:, pad - k] for k in range(stack)], 1)
        errs = []
        for t in range(H):
            zpred = net.step(hist, Ate[:, t])
            pred_state = net.denorm(zpred); true_state = Ste[:, pad + t + 1]
            nd = torch.norm(pred_state[:, NEEDLE] - true_state[:, NEEDLE], dim=1) * 1000.0
            errs.append(nd.mean().item())
            hist = torch.cat([zpred.unsqueeze(1), hist[:, :-1]], 1)
    return errs   # needle MAE mm at H=1..H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--shift", default="target", choices=["source", "target"])
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--num_steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    episodes = gather(args)
    rng = np.random.RandomState(args.seed)
    n_te = max(1, len(episodes) // 5)
    te_ids = set(rng.choice(len(episodes), n_te, replace=False))
    tr = [e for i, e in enumerate(episodes) if i not in te_ids]
    te = [e for i, e in enumerate(episodes) if i in te_ids]
    adim = episodes[0]["act"].shape[1]; H = args.horizon
    log(f"[split] train {len(tr)} / test {len(te)} episodes | H={H}\n")

    variants = [("A",1),("B1",1),("B2",1),("B3",2),("B4",3)]
    log(f"{'variant':30s}" + "".join(f"{'H='+str(h):>10s}" for h in range(1, H + 1)))
    for kind, stack in variants:
        dtr = make_state(tr, kind); dte = make_state(te, kind)
        sdim = dtr[0][0].shape[1]
        errs = train_and_eval(dtr, dte, sdim, adim, stack, H, args)
        tag = {"A":"A  [pos]","B1":"B1 [pos+jointvel]","B2":"B2 [pos+FULLqvel]",
               "B3":"B3 [stack2 pos]","B4":"B4 [stack3 pos]"}[kind]
        log(f"{tag:30s}" + "".join(f"{e:>10.3f}" for e in errs) + "   (needle MAE mm)")
    log("\n===== READ: needle rollout MAE (mm) vs horizon; 1.8mm = target window =====")
    log("  B1/B2 << A at H=5  => velocity survives compounding; add velocity to obs.")
    log("  B1/B2 -> A as H grows => velocity's 1-step gain eaten by having to predict vel dynamics.")
    os._exit(0)


if __name__ == "__main__":
    main()
