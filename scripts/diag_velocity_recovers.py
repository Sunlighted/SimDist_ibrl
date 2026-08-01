"""Does ADDING velocity to the obs recover the ~1.5mm obs-conditional stochasticity?

diag_obs_stochasticity.py showed: fixing the 37-dim obs (which has NO velocity) but
varying the hidden qvel spreads the next needle pos by ~1.5mm (contact phase) -- so
the transition is stochastic-from-obs at ~the tolerance-window scale.

If that missing info is VELOCITY, then a model conditioned on (pos, vel) should predict
the next needle pos MUCH better than one conditioned on pos alone. This is a predictive
test (fair, same MLP arch, only the input feature set differs):

  A  : [pos_obs(37), action(7)]                       -- the current obs
  B1 : [pos_obs, joint_vel(7)+gripper_qvel(2), action] -- realistic velocity observables
  B2 : [pos_obs, FULL sim.data.qvel, action]           -- maximal velocity info (upper bound)

Report held-out one-step needle-position MAE (mm), split contact / free phase.
  A >> B  and B small  => velocity IS the missing info; ADD it to obs, keep MSE (Markov).
  A ~= B               => velocity is NOT enough; residual is arm null-space / contact
                          micro-state => generative head or irreducible.
"""
import os
import sys
import argparse

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
RING = slice(23, 26)


def log(*a):
    print(*a, file=ERR, flush=True)


def mlp(in_dim, out_dim, h=256):
    return nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                         nn.Linear(h, out_dim))


def train_predictor(Xtr, Ytr, Xte, in_dim, epochs=4000, bs=512, seed=0):
    """Fit Delta-needle from X; return per-sample test prediction (denormalized to meters)."""
    torch.manual_seed(seed)
    dev = "cuda"
    xm, xs = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    ym, ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-6)
    Xn = ((Xtr - xm) / xs).to(dev); Yn = ((Ytr - ym) / ysd).to(dev)
    Xten = ((Xte - xm) / xs).to(dev)
    net = mlp(in_dim, 3).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = Xn.shape[0]
    for ep in range(epochs):
        idx = torch.randint(0, n, (bs,), device=dev)
        pred = net(Xn[idx])
        loss = ((pred - Yn[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xten) * ysd.to(dev) + ym.to(dev)   # denormalize -> meters
    return pred.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--shift", default="target", choices=["source", "target"])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/velrec_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, args.ckpt)))
    model = ws.agent.to("cuda").model

    ep = dict(ws.eval_env_params)
    if args.shift == "source":
        ep["domain_shift"] = DomainShiftConfig(mode="none")
    log(f"[setup] shift={args.shift} ({ep['domain_shift'].mode})")
    env = PixelRobosuite(**ep)
    ep_len = int(ws.eval_env_params.get("episode_length", 400))

    # ---- gather transitions with (pos, robo_vel, full_qvel, action, next needle, phase, ep) ----
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    recs = []
    for e in range(args.episodes):
        obs, _ = env.reset()
        cur = obs["state"].cpu().numpy()
        for t in range(ep_len):
            raw = env.env._get_observations(force_update=True)
            robo_vel = np.concatenate([np.asarray(raw["robot0_joint_vel"]).ravel(),
                                       np.asarray(raw["robot0_gripper_qvel"]).ravel()])  # 9
            full_qvel = env.env.sim.get_state().qvel.copy()
            with torch.no_grad():
                a = model.policy_action(obs["state"].view(1, -1)).view(-1).clamp(-1, 1)
            a_np = a.detach().cpu().numpy()
            dist = float(np.linalg.norm(cur[NEEDLE] - cur[RING]))
            obs, _, term, _, _ = env.step(a.detach().cpu())
            nxt = obs["state"].cpu().numpy()
            recs.append(dict(pos=cur.copy(), robo_vel=robo_vel, qvel=full_qvel, act=a_np,
                             dneedle=(nxt[NEEDLE] - cur[NEEDLE]).copy(), dist=dist, ep=e))
            cur = nxt
            if term:
                break
    log(f"[gather] {len(recs)} transitions from {args.episodes} episodes")

    eps = np.array([r["ep"] for r in recs])
    rng = np.random.RandomState(args.seed)
    test_eps = set(rng.choice(args.episodes, max(1, args.episodes // 5), replace=False))
    is_te = np.array([r["ep"] in test_eps for r in recs])
    dists = np.array([r["dist"] for r in recs])
    lo, hi = np.percentile(dists, 15), np.percentile(dists, 70)
    log(f"[gather] test eps={len(test_eps)}/{args.episodes} ({is_te.sum()} trans) | "
        f"phase: contact<{lo:.3f} free>{hi:.3f} | full qvel dim={len(recs[0]['qvel'])}")

    def stack(key):
        return torch.tensor(np.stack([r[key] for r in recs]), dtype=torch.float32)
    pos, robo_vel, qvel, act, dY = stack("pos"), stack("robo_vel"), stack("qvel"), stack("act"), stack("dneedle")
    tr, te = ~torch.tensor(is_te), torch.tensor(is_te)

    # frame-stacking (state-based analog): prev-state within the same episode (pad with
    # self at episode start => zero implicit velocity). B3=2-frame, B4=3-frame.
    ep_arr = np.array([r["ep"] for r in recs]); N = len(recs)
    prev1 = np.arange(N); prev2 = np.arange(N)
    for i in range(N):
        prev1[i] = i - 1 if (i > 0 and ep_arr[i] == ep_arr[i - 1]) else i
        j = prev1[i]
        prev2[i] = j - 1 if (j > 0 and ep_arr[j] == ep_arr[j - 1]) else j
    pos_p1, pos_p2 = pos[torch.tensor(prev1)], pos[torch.tensor(prev2)]

    feats = {
        "A  [pos+act]":              torch.cat([pos, act], 1),
        "B1 [pos+jointvel+act]":     torch.cat([pos, robo_vel, act], 1),
        "B2 [pos+FULLqvel+act]":     torch.cat([pos, qvel, act], 1),
        "B3 [stack2 pos+act]":       torch.cat([pos, pos_p1, act], 1),
        "B4 [stack3 pos+act]":       torch.cat([pos, pos_p1, pos_p2, act], 1),
    }
    dist_te = dists[is_te]
    m_contact = dist_te < lo
    m_free = dist_te > hi
    log(f"[eval] held-out: contact={m_contact.sum()} free={m_free.sum()} all={is_te.sum()}\n")
    log(f"{'model':26s}{'contact MAE(mm)':>16s}{'free MAE(mm)':>15s}{'all MAE(mm)':>14s}")
    for name, X in feats.items():
        pred = train_predictor(X[tr], dY[tr], X[te], X.shape[1], seed=args.seed)
        errmm = (torch.norm(pred - dY[te], dim=1).numpy()) * 1000.0
        log(f"{name:26s}{errmm[m_contact].mean():>16.3f}{errmm[m_free].mean():>15.3f}{errmm.mean():>14.3f}")

    log("\n===== READ =====")
    log("  A >> B (B small)  => velocity recovers the obs-conditional spread; ADD velocity to obs (Markov, keep MSE).")
    log("  A ~= B            => velocity is NOT the missing info; residual = arm null-space / contact micro-state.")
    os._exit(0)


if __name__ == "__main__":
    main()
