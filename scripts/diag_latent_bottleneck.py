"""Does the learned dynamics DESTROY information, or merely DISTORT it?

Context (§27): with learned dynamics MPPI gets 0.825; feed it the REAL s_H and the
SAME distilled value head scores 100/100. So the value is fine -- the imagined
terminal z_H is simply the wrong place. The natural fix is to CALIBRATE the value
onto the imagined latent: supervise J(z_H^imagined) with what really happens.

That only works if z_H^imagined still CONTAINS the information. Two possibilities:
  * DISTORTED but injective -- different (s_0, a) land on different z_H, just at the
    wrong coordinates. Then a calibration head can undo it.
  * MANY-TO-ONE -- candidates with very different true outcomes get squashed onto the
    same z_H. Then no amount of post-training recovers them: the bottleneck destroyed
    the signal, and the only way out is to bypass the dynamics entirely (chunk-Q).

The target y is NOT a Monte-Carlo return (too expensive, and noisy): it is
    y = V_hat( encode(s_H^real) )
i.e. the SAME distilled value applied to the TRUE terminal state -- the exact scoring
function that was measured to reach 100/100 on the oracle arm. It is, by construction,
a sufficient ranking signal.

Three regressions on the same candidates:
  A  z_H^imagined            -> y     (ceiling of "calibrate the value in latent space")
  B  (s_0, a_{0:H-1})        -> y     (ceiling of chunk-Q: no dynamics at all)
  baseline: corr( V_hat(z_H^imagined), y )   -- what the planner uses TODAY, uncalibrated

  R2(A) ~= R2(B)  => dynamics distorts but preserves  => calibration is viable
  R2(A) <<  R2(B) => dynamics destroys                => drop the model, go chunk-Q

Also reports the WITHIN-CANDIDATE-SET Spearman (planner only ever ranks inside one
set of 32), which is the metric that actually matters.
"""
import os
import sys
import atexit
import argparse

import numpy as np
import torch
import torch.multiprocessing as mp
import pyrallis
from scipy.stats import spearmanr

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from scripts.eval_oracle_dyn_mppi import ProbePool, restore  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


def fit_mlp(X, Y, hidden=512, steps=4000, lr=1e-3, val_frac=0.2, seed=0, tag=""):
    """Plain MLP regression with a held-out split. Returns (val_R2, val_spearman)."""
    torch.manual_seed(seed)
    n = len(X)
    perm = torch.randperm(n)
    nv = int(n * val_frac)
    vi, ti = perm[:nv], perm[nv:]
    Xt, Yt, Xv, Yv = X[ti], Y[ti], X[vi], Y[vi]
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp(min=1e-6)
    Xt, Xv = (Xt - mu) / sd, (Xv - mu) / sd
    ym, ys = Yt.mean(), Yt.std().clamp(min=1e-6)

    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    ).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bs = min(512, len(Xt))
    for i in range(steps):
        idx = torch.randint(0, len(Xt), (bs,), device=X.device)
        pred = net(Xt[idx]).squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred, (Yt[idx] - ym) / ys)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pv = net(Xv).squeeze(-1) * ys + ym
    ss_res = ((pv - Yv) ** 2).sum().item()
    ss_tot = ((Yv - Yv.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-9)
    sp = spearmanr(pv.cpu().numpy(), Yv.cpu().numpy()).correlation
    return r2, sp, net, (mu, sd, ym, ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--probe_every", type=int, default=5)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--num_base", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--probe_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/bneck_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, "latest.pt")))
    agent = ws.agent.to("cuda")
    model, planner = agent.model, agent.planner
    p = planner.cfg
    p.num_samples, p.num_base_override, p.horizon = args.num_samples, args.num_base, args.horizon
    p.iterations, p.num_elites, p.use_reward = 2, 8, 0
    planner.discounts = (p.discount ** torch.arange(args.horizon, device="cuda")).view(args.horizon, 1)
    planner.final_discount = p.discount ** args.horizon

    ep = dict(ws.eval_env_params)
    main_env = PixelRobosuite(**ep)
    pool = ProbePool(ep, args.probe_workers)
    log(f"[setup] {args.episodes} eps, probe every {args.probe_every}, "
        f"{args.num_samples} cands/probe, H={args.horizon}, {args.probe_workers} workers")

    S0, A, ZI, Y = [], [], [], []       # s_0 / action seq / imagined z_H / target y
    VHAT_IMAG = []                       # what the planner uses today
    set_id = []
    ep_len = int(ep.get("episode_length", 400))
    sid = 0
    for e in range(args.episodes):
        obs, _ = main_env.reset()
        for t in range(ep_len):
            with torch.no_grad():
                if t % args.probe_every == 0:
                    s0 = obs["state"]
                    z0 = model.encode(s0.view(1, -1))
                    base = planner._policy_rollout(z0, args.num_base)
                    base = base + torch.randn_like(base) * p.base_policy_std
                    mean = torch.zeros(args.horizon, planner.action_dim, device="cuda")
                    std = torch.full((args.horizon, planner.action_dim), p.init_std, device="cuda")
                    noise = torch.randn(args.horizon, args.num_samples - args.num_base,
                                        planner.action_dim, device="cuda")
                    cands = torch.cat([(mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1),
                                       base.clamp(-1, 1)], dim=1)          # [H, N, act]
                    N = cands.shape[1]

                    # imagined terminal latent (what the planner scores today)
                    z = z0.expand(N, -1).clone()
                    for h in range(args.horizon):
                        z = model.next(z, cands[h])
                    z_imag = z                                             # [N, zdim]
                    v_imag = model.predict_value(z_imag, target=False).view(-1)

                    # REAL terminal state -> the verified 100/100 scoring signal
                    sim_state = main_env.env.sim.get_state().flatten().copy()
                    real = pool.rollout(sim_state, cands)                  # [H, N, obs]
                    y = model.predict_value(model.encode(real[-1]), target=False).view(-1)

                    S0.append(s0.view(1, -1).expand(N, -1).cpu())
                    A.append(cands.permute(1, 0, 2).reshape(N, -1).cpu())  # [N, H*act]
                    ZI.append(z_imag.cpu())
                    VHAT_IMAG.append(v_imag.cpu())
                    Y.append(y.cpu())
                    set_id.append(np.full(N, sid)); sid += 1
                    restore(main_env, sim_state)
                a = planner.plan(obs["state"], eval_mode=True)
            obs, _, term, _, _ = main_env.step(a.cpu())
            if term:
                break
        log(f"  ep{e}: {sid} candidate sets so far")
    pool.close()

    S0 = torch.cat(S0).cuda(); A = torch.cat(A).cuda(); ZI = torch.cat(ZI).cuda()
    Y = torch.cat(Y).cuda(); VH = torch.cat(VHAT_IMAG).cuda()
    set_id = np.concatenate(set_id)
    log(f"\n[data] {len(Y)} candidates from {sid} sets | z dim {ZI.shape[1]} | "
        f"(s0,a) dim {S0.shape[1] + A.shape[1]}")
    log(f"[data] y (= V_hat on REAL s_H): mean {Y.mean():.4f} std {Y.std():.4f}")

    def within_set_spearman(pred):
        rs = []
        pr, yy = pred.cpu().numpy(), Y.cpu().numpy()
        for s in range(sid):
            m = set_id == s
            if m.sum() > 2 and yy[m].std() > 0:
                r = spearmanr(pr[m], yy[m]).correlation
                if not np.isnan(r):
                    rs.append(r)
        return float(np.mean(rs)), len(rs)

    # ---------- baseline: what the planner uses TODAY (uncalibrated V_hat on z_imag)
    sp0, nsets = within_set_spearman(VH)
    log(f"\n=== BASELINE  planner 今天用的 V_hat(z_H^imagined)")
    log(f"    集内 Spearman vs y = {sp0:+.3f}   (over {nsets} sets)   <- 这就是 0.825 的来源")

    # ---------- A: calibrate onto the imagined latent
    r2a, spa, neta, na = fit_mlp(ZI, Y, tag="A")
    preda = (neta((ZI - na[0]) / na[1]).squeeze(-1) * na[3] + na[2]).detach()
    spa_set, _ = within_set_spearman(preda)
    log(f"\n=== A  z_H^imagined -> y   (你的方案: 把 value 校准到想象空间)")
    log(f"    held-out R2 = {r2a:.3f}   held-out Spearman = {spa:+.3f}   集内 Spearman = {spa_set:+.3f}")

    # ---------- B: bypass the dynamics entirely (chunk-Q ceiling)
    SA = torch.cat([S0, A], dim=1)
    r2b, spb, netb, nb = fit_mlp(SA, Y, tag="B")
    predb = (netb((SA - nb[0]) / nb[1]).squeeze(-1) * nb[3] + nb[2]).detach()
    spb_set, _ = within_set_spearman(predb)
    log(f"\n=== B  (s_0, a_0:H-1) -> y   (chunk-Q 上界: 完全不用 dynamics)")
    log(f"    held-out R2 = {r2b:.3f}   held-out Spearman = {spb:+.3f}   集内 Spearman = {spb_set:+.3f}")

    log(f"\n===== 判读 =====")
    log(f"  R2(A)={r2a:.3f}  vs  R2(B)={r2b:.3f}   (差 {r2b - r2a:+.3f})")
    log(f"  集内 Spearman: baseline {sp0:+.3f} -> A {spa_set:+.3f} -> B {spb_set:+.3f}")
    if r2b - r2a > 0.15:
        log("  ⇒ dynamics 明显 DESTROYS 信息: z_H 是瓶颈, 校准 value 救不回来 => 扔掉 dynamics, 上 chunk-Q")
    elif r2a > 0.6:
        log("  ⇒ dynamics 只是 DISTORTS: z_H 仍含足够信息 => 校准 value 到 imagined latent 可行")
    else:
        log("  ⇒ 两者都不高: 这批候选的 y 本身可能就难预测, 检查 y 的方差/覆盖")


if __name__ == "__main__":
    main()
