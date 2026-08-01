"""GO/NO-GO for value-ensemble pessimism (CLAUDE.md §27 follow-up).

Runs the DEPLOYED target MPPI (learned dynamics) and, at every committed plan,
captures BOTH the model's imagined terminal state z_H = decode(rollout of the
plan) AND the REAL terminal state s_H (same plan stepped in the true target sim).
Then evaluates a pre-trained value ENSEMBLE (scripts/train_value_ensemble.py) on
those pairs and asks the ONE question that decides whether making V an ensemble
(the paper's min-over-ensemble pessimism) can break the 0.85 ceiling:

  Does ensemble DISAGREEMENT std_i V_i(z_H) light up exactly where the plan is
  wrong?  Two "wrongness" targets:
    (a) geometric  ||z_H - s_H||     <- CONFOUND-FREE (oracle-exact), the ROOT error
    (b) value      V_bar(z_H) - V_bar(s_H)   <- the terminal-value error argmax exploits

  disagreement EXPLAINS the error (high Spearman/R^2)  => pessimism can SEE it,
      min-ensemble has real headroom -> worth plugging into the planner.
  disagreement is BLIND (low corr, error concentrated at LOW disagreement)
      => the dynamics error is on-manifold-plausible; NO value pessimism (however
      many independent heads) can fix it -> stop, the bottleneck is the model.

On-policy by construction (pairs come from the real deployed planner), so it
dodges the §27 fixed-cache distribution-shift trap.

Run:
  python scripts/diag_value_ensemble_explain.py \
      --run_dir exps/mimicgen/threading_simdist_adapt_long_h5_seed1 --ckpt model0.pt \
      --ensemble exps/mimicgen/value_ensemble_source_seed1/value_ensemble.pt \
      --episodes 20
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


def restore(env, sim_state):
    raw = env.env
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    env.time_step = 0
    if getattr(env, "_success_validator", None) is not None:
        env._success_validator.reset(raw, env._bindings)


class VNet(nn.Module):
    def __init__(self, obs_dim, hidden, n_layer):
        super().__init__()
        layers, d = [], obs_dim
        for _ in range(n_layer):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_ensemble(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    heads = []
    for sd in ck["state_dicts"]:
        h = VNet(ck["obs_dim"], ck["hidden"], ck["n_layer"]).to(dev)
        h.load_state_dict(sd)
        h.eval()
        heads.append(h)
    mean = torch.tensor(ck["norm_mean"], device=dev)
    std = torch.tensor(ck["norm_std"], device=dev)

    @torch.no_grad()
    def ens(states):                              # [M,37] -> [N,M]
        x = (states - mean) / std
        return torch.stack([h(x) for h in heads], 0)
    return ens, len(heads)


# ------------------------------------------------ stats helpers (no scipy dep)
def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x))
    return r


def spearman(a, b):
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def imagine_terminal(model, s0, plan):
    """s0 [37], plan [H,act] -> imagined terminal state [37] via learned dynamics."""
    z = model.encode(s0.view(1, -1))
    for t in range(plan.shape[0]):
        z = model.next(z, plan[t].view(1, -1))
    return model.decode(z).view(-1)


@torch.no_grad()
def real_terminal(env, sim_state, plan):
    """plan [H,act] stepped in the TRUE sim from sim_state -> terminal state [37]."""
    restore(env, sim_state)
    a = plan.cpu()
    obs = None
    for t in range(plan.shape[0]):
        obs, _, _, _, _ = env.step(a[t])
    return obs["state"].to(plan.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--ckpt", default="model0.pt")
    ap.add_argument("--ensemble", default="exps/mimicgen/value_ensemble_source_seed1/value_ensemble.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--stride", type=int, default=1, help="record every k-th control step")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0
    cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/209968a1-cfcd-4e75-a195-fbe839bb8788/scratchpad/veexp_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, args.ckpt)))
    agent = ws.agent.to(dev)
    agent.eval()
    model, planner = agent.model, agent.planner
    H = planner.cfg.horizon

    ep = dict(ws.eval_env_params)
    main_env = PixelRobosuite(**ep)
    probe_env = PixelRobosuite(**ep)
    probe_env.reset()   # CRITICAL: apply the domain shift (restore() only writes qpos/qvel)

    ens, N = load_ensemble(args.ensemble, dev)
    log(f"[setup] {args.run_dir}/{args.ckpt} | H={H} | ensemble N={N} | "
        f"{args.episodes} episodes, TARGET strict, sigma=0.1")

    Z, S = [], []      # imagined / real terminal states
    succ = []
    for e in range(args.episodes):
        obs, _ = main_env.reset()
        done_succ = 0.0
        for t in range(args.max_steps):
            with torch.no_grad():
                plan = planner.plan(obs["state"], eval_mode=True, return_full=True)  # [H,act]
            if t % args.stride == 0:
                sim_state = main_env.env.sim.get_state().flatten().copy()
                zH = imagine_terminal(model, obs["state"], plan)
                sH = real_terminal(probe_env, sim_state, plan)
                Z.append(zH.cpu().numpy())
                S.append(sH.cpu().numpy())
            obs, reward, terminal, _, _ = main_env.step(plan[0].detach().cpu())
            if reward > 0:
                done_succ = 1.0
            if terminal:
                break
        succ.append(done_succ)
        log(f"  ep{e}: succ={done_succ:.0f} pairs={len(Z)} (running {np.mean(succ):.3f})")

    Z = torch.tensor(np.array(Z), device=dev)   # [M,37]
    S = torch.tensor(np.array(S), device=dev)
    M = Z.shape[0]
    log(f"\n[collected] {M} committed-plan pairs | target strict success {np.mean(succ):.3f}")

    with torch.no_grad():
        Vz = ens(Z)                       # [N,M]
        Vs = ens(S)
    vbar_z = Vz.mean(0).cpu().numpy()
    vbar_s = Vs.mean(0).cpu().numpy()
    dis_z = Vz.std(0).cpu().numpy()       # ensemble disagreement at imagined terminal
    dis_s = Vs.std(0).cpu().numpy()
    vmin_z = Vz.min(0).values.cpu().numpy()
    geom = (Z - S).norm(dim=-1).cpu().numpy()          # ||z_H - s_H||  (oracle-exact)
    val_err = vbar_z - vbar_s                            # signed terminal-value error
    abs_ve = np.abs(val_err)

    def block(title):
        log("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)

    block("0. distributions")
    log(f"  geom ||z_H - s_H||   : mean {geom.mean():.4f}  med {np.median(geom):.4f}  p90 {np.percentile(geom,90):.4f}")
    log(f"  val_err (signed)     : mean {val_err.mean():+.4f}  std {val_err.std():.4f}  "
        f"frac overestimate(>0) {np.mean(val_err>0):.2f}")
    log(f"  disagree@z_H         : mean {dis_z.mean():.4f}  med {np.median(dis_z):.4f}  p90 {np.percentile(dis_z,90):.4f}")
    log(f"  disagree@s_H         : mean {dis_s.mean():.4f}   (real state; vs imagined above)")

    block("1. HEADLINE -- does disagreement SEE the error?")
    log(f"  Spearman(disagree@z_H, geom ||z_H-s_H||) = {spearman(dis_z, geom):+.3f}")
    log(f"  Pearson (disagree@z_H, geom)             = {pearson(dis_z, geom):+.3f}")
    log(f"  Spearman(disagree@z_H, |val_err|)        = {spearman(dis_z, abs_ve):+.3f}")
    log(f"  Pearson (disagree@z_H, |val_err|)        = {pearson(dis_z, abs_ve):+.3f}")
    log(f"  R^2(|val_err| ~ disagree@z_H, linear)    = {pearson(dis_z, abs_ve)**2:.3f}")

    block("2. is the IMAGINED point more OOD than the REAL one?")
    log(f"  mean disagree  imagined {dis_z.mean():.4f}  vs  real {dis_s.mean():.4f}  "
        f"(ratio {dis_z.mean()/max(dis_s.mean(),1e-9):.2f}x)")
    log("  (if ~1x, the ensemble can't even tell imagination from reality => structurally blind)")

    block("3. does MIN reduce the OVERESTIMATE? (the harmful direction)")
    over = val_err > 0
    if over.sum() > 0:
        e_mean = (vbar_z - vbar_s)[over]                 # overestimate with mean-ensemble
        e_min = (vmin_z - vbar_s)[over]                  # ...after taking min
        log(f"  overestimate subset: {over.sum()}/{M}")
        log(f"  mean overestimate  (mean-ensemble) = {e_mean.mean():+.4f}")
        log(f"  mean residual      (min-ensemble)  = {e_min.mean():+.4f}")
        log(f"  |err| reduced by min: {np.mean(np.abs(e_mean)) - np.mean(np.abs(e_min)):+.4f} "
            f"(positive = min helps)")
        # does the min DOWN-PUSH land where the overestimate is? corr over ALL points
        push = vbar_z - vmin_z                            # >=0 pessimism magnitude
        log(f"  Spearman(min-push, val_err) over all pts = {spearman(push, val_err):+.3f} "
            f"(want high +: push where overestimated)")

    block("4. BLIND-FRACTION -- how much error hides at LOW disagreement?")
    order = np.argsort(dis_z)                             # ascending disagreement
    lo = order[: M // 2]                                  # bottom-50% disagreement
    log(f"  bottom-50% disagreement holds "
        f"{100*np.sum(abs_ve[lo])/max(np.sum(abs_ve),1e-9):.1f}% of total |val_err|, "
        f"{100*np.sum(geom[lo])/max(np.sum(geom),1e-9):.1f}% of total geom error")
    log("  (a big share here = the error is at points the ensemble thinks are FINE => pessimism blind)")

    block("VERDICT")
    sp_g = spearman(dis_z, geom)
    sp_v = spearman(dis_z, abs_ve)
    blind = 100 * np.sum(abs_ve[lo]) / max(np.sum(abs_ve), 1e-9)
    if sp_g > 0.4 and sp_v > 0.4 and blind < 35:
        log("  => disagreement TRACKS the error and little hides at low disagreement:")
        log("     value-ensemble pessimism has REAL headroom. Worth plugging into the planner")
        log("     (then RE-measure on the NEW on-policy distribution -- §27 caveat).")
    else:
        log("  => disagreement is (largely) BLIND to the dynamics/value error, and/or the")
        log("     error concentrates where the ensemble is confident. No amount of value")
        log("     ensembling breaks the ceiling; the bottleneck is the MODEL, not the value.")
    log(f"  [Sp(geom)={sp_g:+.3f}  Sp(|val_err|)={sp_v:+.3f}  blind-frac={blind:.0f}%]")

    os._exit(0)


if __name__ == "__main__":
    main()
