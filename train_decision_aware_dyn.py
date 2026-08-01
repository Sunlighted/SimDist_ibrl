"""Decision-aware dynamics finetune: only punish the error that argmax can SEE.

§27's decomposition:
    eps(a) = V_hat(f^H(phi(s0), a)) - y(a)  =  eps_bar(s0) + eps_tilde(s0, a)
             (within-set constant)             (within-set variation)
and  performance loss <= 2 * sup_a |eps_tilde|.  argmax is COMPLETELY IMMUNE to
eps_bar (a constant does not change a ranking).

Measured on 2978 candidate sets: eps_bar carries 46% of the total error variance --
so nearly half of what MSE training fights is something the planner does not care
about. And the signal-to-noise the planner actually faces is
    SNR = Var_a[y] / Var_a[eps_tilde] = 0.52   (<< 1  =>  argmax cannot rank)

So: replace the MSE objective
        min_f  E[ eps^2 ]  =  E[ Var_a(eps) ] + E[ eps_bar^2 ]
with the decision-aware one
        min_f  E_sets[ Var_{a in set}( V_hat(f^H(phi(s0),a)) - y ) ]
i.e. DROP the bias term. Let the model be arbitrarily wrong, as long as it is
wrong the SAME WAY for every candidate.

Only `dynamics` is trained (encoder / value / policy frozen).

Three arms (the third is mandatory, else "finetuning on this data at all" is a
confounder):
  none   -- the stock model                       (baseline 0.825)
  var    -- decision-aware: min Var_a(eps)
  mse    -- ordinary: min E[eps^2]  on the SAME data
"""
import os
import sys
import argparse

import numpy as np
import torch
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


@torch.no_grad()
def evaluate(planner, env_params, episodes, seed):
    env = PixelRobosuite(**env_params)
    succ = []
    torch.manual_seed(seed); np.random.seed(seed)
    ep_len = int(env_params.get("episode_length", 400))
    for e in range(episodes):
        obs, _ = env.reset()
        s = 0.0
        for t in range(ep_len):
            a = planner.plan(obs["state"], eval_mode=True)
            obs, r, term, _, _ = env.step(a.cpu())
            if r > 0:
                s = 1.0
            if term:
                break
        succ.append(s)
        if (e + 1) % 10 == 0:
            log(f"    eval {e+1}/{episodes}: running {np.mean(succ):.3f}")
    return float(np.mean(succ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--cache", default="/tmp/claude-1003/-home-Chenyang-ibrl/"
                                       "08677077-8099-4d03-9973-403b524caa19/scratchpad/calib_data2.pt")
    ap.add_argument("--loss", default="var", choices=["var", "mse", "none"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sets_per_batch", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--eval_episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/dadyn_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, "latest.pt")))
    agent = ws.agent.to("cuda")
    model, planner = agent.model, agent.planner
    p = planner.cfg
    p.num_samples, p.num_base_override, p.horizon = 32, 10, args.horizon
    p.iterations, p.num_elites, p.use_reward = 2, 8, 0
    planner.discounts = (p.discount ** torch.arange(args.horizon, device="cuda")).view(args.horizon, 1)
    planner.final_discount = p.discount ** args.horizon
    ep = dict(ws.eval_env_params)

    d = torch.load(args.cache)
    Y, sets = d["Y"].cuda(), d["sets"]
    S0, A = d["S0"].cuda(), d["A"].cuda()
    H, act_dim = args.horizon, A.shape[1] // args.horizon
    A = A.view(-1, H, act_dim)                       # [Ncand, H, act]
    bounds = np.cumsum([0] + list(sets))
    nsets = len(sets)
    # 按候选集划分 train/val
    perm = np.random.permutation(nsets)
    n_val = int(0.15 * nsets)
    vsets, tsets = perm[:n_val], perm[n_val:]
    log(f"[data] {len(Y)} cands / {nsets} sets (train {len(tsets)} / val {len(vsets)})")

    def set_slice(k):
        return slice(bounds[k], bounds[k + 1])

    def eps_of(ks, grad=True):
        """eps = V_hat(f^H(phi(s0), a)) - y, per candidate, for the given sets."""
        out = []
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for k in ks:
                sl = set_slice(k)
                with torch.no_grad():
                    z = model.encode(S0[sl])          # encoder FROZEN
                a = A[sl]                             # [N, H, act]
                for h in range(H):
                    z = model.next(z, a[:, h])        # <-- the ONLY thing that trains
                v = model.predict_value(z, target=False).view(-1)   # value FROZEN
                out.append(v - Y[sl])
        return out

    @torch.no_grad()
    def val_metrics():
        es = eps_of(vsets, grad=False)
        vt = torch.stack([e.var() for e in es]).mean().item()          # Var_a[eps_tilde]
        eb = torch.stack([e.mean() for e in es])
        vb = eb.var().item()                                            # Var_s[eps_bar]
        vy = torch.stack([Y[set_slice(k)].var() for k in vsets]).mean().item()
        return vt, vb, vy, vy / max(vt, 1e-12)

    vt0, vb0, vy0, snr0 = val_metrics()
    log(f"[val@init] Var[eps_tilde]={vt0:.5f}  Var[eps_bar]={vb0:.5f}  "
        f"Var[y]={vy0:.5f}  ★SNR={snr0:.3f}")

    base = evaluate(planner, ep, args.eval_episodes, args.seed)
    log(f"[BASELINE] stock dynamics: TARGET strict = {base:.3f}")

    if args.loss == "none":
        log(f"\n===== SUMMARY =====\n  stock: {base:.3f}")
        os._exit(0)

    # only the dynamics head trains
    for prm in model.parameters():
        prm.requires_grad_(False)
    for prm in model.dynamics.parameters():
        prm.requires_grad_(True)
    opt = torch.optim.Adam(model.dynamics.parameters(), lr=args.lr)

    log(f"\n===== finetune dynamics, loss = {args.loss} =====")
    for i in range(args.steps):
        ks = np.random.choice(tsets, args.sets_per_batch, replace=False)
        es = eps_of(ks, grad=True)
        if args.loss == "var":
            # decision-aware: ONLY the within-set variation. The bias E_a[eps] is free.
            loss = torch.stack([e.var() for e in es]).mean()
        else:
            # ordinary MSE: penalises Var_a[eps] + (E_a[eps])^2 -- half of which the
            # planner is immune to
            loss = torch.stack([(e ** 2).mean() for e in es]).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), 10.0)
        opt.step()
        if (i + 1) % 1000 == 0:
            vt, vb, vy, snr = val_metrics()
            log(f"  step {i+1}: loss {loss.item():.5f} | val Var[eps_tilde] {vt:.5f} "
                f"(init {vt0:.5f})  Var[eps_bar] {vb:.5f}  ★SNR {snr:.3f} (init {snr0:.3f})")

    vt, vb, vy, snr = val_metrics()
    log(f"\n[val@end] Var[eps_tilde] {vt0:.5f} -> {vt:.5f}   ★SNR {snr0:.3f} -> {snr:.3f}")
    score = evaluate(planner, ep, args.eval_episodes, args.seed)
    log(f"\n===== SUMMARY (loss={args.loss}) =====")
    log(f"  stock dynamics      : {base:.3f}")
    log(f"  finetuned dynamics  : {score:.3f}")
    log(f"  SNR {snr0:.3f} -> {snr:.3f}   (预测集内 Spearman "
        f"{np.sqrt(snr0/(1+snr0)):.3f} -> {np.sqrt(snr/(1+snr)):.3f})")
    os._exit(0)


if __name__ == "__main__":
    main()
