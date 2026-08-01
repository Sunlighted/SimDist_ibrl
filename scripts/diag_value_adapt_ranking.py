"""Why did VALUE-adapt make SimDist WORSE? (CLAUDE.md §27/§31 follow-up)

The value-adapt arms (`train_simdist_headablate.py --head value`, and
`train_simdist_adapt.py --adapt_value 1`) fit their value head essentially
PERFECTLY on target data (held-out val_mse 3e-4, pred_mean == tgt_mean) yet MPPI
on target FELL from R=0.745 to ~0.40. So it is not underfitting / not an lr
problem. The suspected mechanism is a TRAINING-vs-QUERY distribution mismatch:

  * stage-1 distill trains the value on the ROLLED PREDICTED latent
    (rl/simdist_agent.py:301  `_value_head_loss(z_next_pred, ...)`)
  * the planner scores exactly there
    (rl/simdist_planner.py:209  `predict_value(z, target=False)`, z = f^H(z0,a))
  * but value-adapt trains on the REAL encoder latent
    (`update_value_only`     rl/simdist_agent.py:526  z = encode(states))
    (`update_dynamics_value` rl/simdist_agent.py:613  z_all = encode(states))

=> the retrained head is fit where the planner never looks, and EXTRAPOLATES
where it is actually used. A mean-type metric (MSE on real states) is blind to
what argmax needs: the ORDERING across the candidate set (§27 #5).

This script measures the ordering directly, ON-POLICY (each arm drives its own
rollout, so we dodge the §27 fixed-cache distribution-shift trap). At every
committed plan we grab the planner's REAL final-iteration CEM candidate set
(monkeypatched `_estimate_returns`, no code duplication), and for each candidate
action sequence a^i compute

    z_H^i  = f^H(encode(s0), a^i)      imagined terminal latent (learned dynamics)
    s_H^i  = true sim rollout of a^i   real terminal state (counterfactual replay)
    y^i    = V_expert(s_H^i)           reference ranking (target IBRL expert, 98%)

and then, for BOTH value heads h in {stock, retrained}, the two Spearmans

    rho_img [h] = corr_i( V_h(z_H^i)          , y^i )   <- what the planner USES
    rho_real[h] = corr_i( V_h(encode(s_H^i))  , y^i )   <- the TRAINING distribution

plus the §27 error decomposition eps_h^i = V_h(z_H^i) - V_h(encode(s_H^i)),
reporting the within-set fluctuation std_i(eps~) and sup|eps~| (the part argmax
is exposed to; a constant offset is harmless).

PREDICTION if the mismatch story is right:
    rho_real[retrained] >= rho_real[stock]   (it really did learn target value)
    rho_img [retrained] <<  rho_img [stock]  (but not where the planner reads it)
    std(eps~)[retrained] >> std(eps~)[stock]

Run:
  python scripts/diag_value_adapt_ranking.py --episodes 12 --stride 6
"""
import argparse
import os
import sys

import numpy as np
import torch
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

import common_utils  # noqa: E402
from train_simdist import MainConfig, Workspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402

ERR = sys.stderr


def log(*a):
    # AdaptWorkspace/Workspace hijack stdout via common_utils.Logger -> use stderr
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


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x))
    return r


def spearman(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def mean_se(x):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if len(x) == 0:
        return np.nan, np.nan, 0
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0, len(x)


def build_agent(base_cfg_dir, ckpt, sigma, dev):
    """Rebuild the SimDist agent + TARGET eval env params from the stage-1 cfg
    (mirrors train_simdist_headablate.AblateWorkspace.__init__, minus the data)."""
    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(base_cfg_dir, "cfg.yaml")))  # type: ignore
    cfg.simdist.use_planner = 1
    from rl.simdist_planner import apply_original_recipe
    apply_original_recipe(cfg.simdist.planner)
    if sigma is not None:
        cfg.eval_hand_init_perturb_joint_sigma = sigma
    cfg.use_wb = 0
    ws = Workspace(cfg, from_main=False)
    ws.agent.load_state_dict(torch.load(ckpt))
    agent = ws.agent.to(dev)
    agent.eval()
    return agent, ws.eval_env_params, cfg


def load_reference_value(ref_ckpt, env, dev):
    """Target-domain IBRL expert (98% strict) -> V_ref(s) = min-ensemble Q(s, pi(s))."""
    import train_rl
    ref_cfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(ref_ckpt), "cfg.yaml")))
    ref = train_rl.load_frozen_agent(ref_ckpt, env, ref_cfg, dev)

    @torch.no_grad()
    def vref(states):                      # [M, obs] -> [M]
        return ref.value({"state": states.to(dev)}).view(-1)
    return vref


@torch.no_grad()
def imagine_terminal_latents(model, s0, actions):
    """s0 [obs], actions [H, M, act] -> imagined terminal latents [M, z]."""
    M = actions.shape[1]
    z = model.encode(s0.view(1, -1)).expand(M, -1).contiguous()
    for t in range(actions.shape[0]):
        z = model.next(z, actions[t])
    return z


@torch.no_grad()
def real_terminals(env, sim_state, actions):
    """actions [H, M, act] stepped in the TRUE target sim -> terminal states [M, obs]."""
    H, M = actions.shape[0], actions.shape[1]
    a = actions.detach().cpu()
    out = []
    for i in range(M):
        restore(env, sim_state)
        obs = None
        for t in range(H):
            obs, _, _, _, _ = env.step(a[t, i])
        out.append(obs["state"])
    return torch.stack(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg_dir", default="exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1")
    ap.add_argument("--stock_ckpt", default="exps/mimicgen/threading_simdist_adapt_seed1/latest.pt")
    ap.add_argument("--retrained_ckpt",
                    default="exps/mimicgen/threading_simdist_ablate_val_ws_600k_seed1/latest.pt")
    ap.add_argument("--ref_ckpt",
                    default="exps/mimicgen/threading_ibrl_rlprior_prior_sig01_seed1/model0.pt")
    ap.add_argument("--sigma", type=float, default=-1.0,
                    help="eval hand-init sigma; <0 => inherit stage-1 cfg (what the ablate runs did)")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--stride", type=int, default=6, help="record every k-th control step")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--k_rand", type=int, default=12, help="uniform candidates from the CEM set")
    ap.add_argument("--k_top", type=int, default=12, help="top candidates by the arm's own return")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default="/tmp/claude-1003/-home-Chenyang-ibrl/"
                                          "bd3f0c47-1194-4d54-a030-1f490b87cf9f/scratchpad/valrank")
    args = ap.parse_args()

    sigma = None if args.sigma < 0 else args.sigma
    dev = "cuda"
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- two agents: IDENTICAL except model.value / model.value_target ----
    agentS, ep_params, _ = build_agent(args.base_cfg_dir, args.stock_ckpt, sigma, dev)
    agentR, _, _ = build_agent(args.base_cfg_dir, args.retrained_ckpt, sigma, dev)
    model = agentS.model            # encoder/dynamics are byte-identical in both
    H = agentS.planner.cfg.horizon

    main_env = PixelRobosuite(**ep_params)
    probe_env = PixelRobosuite(**ep_params)
    probe_env.reset()   # CRITICAL (§27 bug): applies the domain shift; restore() only writes qpos/qvel

    vref = load_reference_value(args.ref_ckpt, main_env, dev)
    log(f"[setup] H={H} | sigma={'inherit' if sigma is None else sigma} | "
        f"{args.episodes} eps/arm, stride {args.stride}, K={args.k_rand}rand+{args.k_top}top")
    log(f"        stock      = {args.stock_ckpt}")
    log(f"        retrained  = {args.retrained_ckpt}")
    log(f"        V_ref      = {args.ref_ckpt}")

    # value readouts (only the head differs between the two models)
    heads = {"stock": agentS.model, "retrained": agentR.model}

    @torch.no_grad()
    def vhead(m, z):
        return m.predict_value(z, target=False).view(-1)

    results = {}     # arm -> dict of per-decision-point lists
    for arm, arm_agent in (("stock", agentS), ("retrained", agentR)):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        planner = arm_agent.planner
        # monkeypatch to capture the FINAL CEM iteration's real candidate set
        cap = {}
        orig = planner._estimate_returns

        def wrapped(z0, actions, _orig=orig, _cap=cap):
            r = _orig(z0, actions)
            _cap["actions"] = actions
            _cap["returns"] = r
            return r
        planner._estimate_returns = wrapped

        acc = {k: [] for k in ("rho_img_s", "rho_img_r", "rho_real_s", "rho_real_r",
                               "eps_std_s", "eps_std_r", "eps_sup_s", "eps_sup_r",
                               "sig_std", "set_kind")}
        succ = []
        for e in range(args.episodes):
            obs, _ = main_env.reset()
            done_succ = 0.0
            for t in range(args.max_steps):
                cap.clear()
                with torch.no_grad():
                    plan = planner.plan(obs["state"], eval_mode=True, return_full=True)
                if t % args.stride == 0 and "actions" in cap:
                    cand = cap["actions"]                     # [H, N, act]
                    rets = cap["returns"].view(-1)            # [N]
                    N = cand.shape[1]
                    sim_state = main_env.env.sim.get_state().flatten().copy()
                    sets = []
                    if args.k_rand > 0:
                        idx = torch.from_numpy(
                            np.random.choice(N, size=min(args.k_rand, N), replace=False)).to(cand.device)
                        sets.append(("rand", idx))
                    if args.k_top > 0:
                        sets.append(("top", torch.topk(rets, min(args.k_top, N)).indices))
                    for kind, idx in sets:
                        a_sub = cand[:, idx]                                   # [H, K, act]
                        zH = imagine_terminal_latents(model, obs["state"], a_sub)   # [K, z]
                        sH = real_terminals(probe_env, sim_state, a_sub)            # [K, obs]
                        with torch.no_grad():
                            zH_real = model.encode(sH.to(dev))
                            y = vref(sH).cpu().numpy()
                        vi = {h: vhead(m, zH).cpu().numpy() for h, m in heads.items()}
                        vr = {h: vhead(m, zH_real).cpu().numpy() for h, m in heads.items()}
                        for h, tag in (("stock", "s"), ("retrained", "r")):
                            eps = vi[h] - vr[h]
                            eps_t = eps - eps.mean()            # within-set fluctuation
                            acc[f"rho_img_{tag}"].append(spearman(vi[h], y))
                            acc[f"rho_real_{tag}"].append(spearman(vr[h], y))
                            acc[f"eps_std_{tag}"].append(float(eps_t.std()))
                            acc[f"eps_sup_{tag}"].append(float(np.abs(eps_t).max()))
                        acc["sig_std"].append(float(np.std(y)))
                        acc["set_kind"].append(kind)
                obs, reward, terminal, _, _ = main_env.step(plan[0].detach().cpu())
                if reward > 0:
                    done_succ = 1.0
                if terminal:
                    break
            succ.append(done_succ)
            log(f"  [{arm}] ep{e}: succ={done_succ:.0f}  points={len(acc['sig_std'])}  "
                f"(running success {np.mean(succ):.3f})")
        planner._estimate_returns = orig
        acc["success"] = float(np.mean(succ))
        results[arm] = acc

    np.save(os.path.join(args.save_dir, "raw.npy"), results, allow_pickle=True)

    # ------------------------------------------------------------------ report
    def blk(t):
        log("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)

    for kind in ("all", "rand", "top"):
        blk(f"candidate set = {kind.upper()}   (ordering across the candidates argmax competes over)")
        log(f"{'on-policy arm':<14}{'value head':<12}{'rho_real (train dist)':>24}"
            f"{'rho_img (planner uses)':>26}")
        for arm in ("stock", "retrained"):
            a = results[arm]
            sel = np.array([k == kind for k in a["set_kind"]]) if kind != "all" \
                else np.ones(len(a["set_kind"]), bool)
            for h, tag in (("stock", "s"), ("retrained", "r")):
                rr = mean_se(np.array(a[f"rho_real_{tag}"])[sel])
                ri = mean_se(np.array(a[f"rho_img_{tag}"])[sel])
                log(f"{arm:<14}{h:<12}{rr[0]:>16.3f} ±{rr[1]:.3f}{ri[0]:>18.3f} ±{ri[1]:.3f}"
                    f"   (n={rr[2]})")

    blk("eps = V(imagined z_H) - V(encode(real s_H))   [§27: only the WITHIN-SET part hurts argmax]")
    log(f"{'on-policy arm':<14}{'value head':<12}{'std_i(eps~)':>14}{'sup|eps~|':>12}"
        f"{'std_i(y_ref)':>14}{'SNR=Var(y)/Var(eps~)':>24}")
    for arm in ("stock", "retrained"):
        a = results[arm]
        sg = mean_se(a["sig_std"])[0]
        for h, tag in (("stock", "s"), ("retrained", "r")):
            es = mean_se(a[f"eps_std_{tag}"])[0]
            sp = mean_se(a[f"eps_sup_{tag}"])[0]
            snr = (sg ** 2) / max(es ** 2, 1e-12)
            log(f"{arm:<14}{h:<12}{es:>14.4f}{sp:>12.4f}{sg:>14.4f}{snr:>24.2f}")

    blk("diag-run success (sanity vs the logged R=0.745 / ~0.40; small n, just a check)")
    for arm in ("stock", "retrained"):
        log(f"  {arm:<12} strict success = {results[arm]['success']:.3f} "
            f"({args.episodes} episodes)")

    blk("VERDICT")
    s_img = mean_se(results["stock"]["rho_img_s"])[0]
    r_img = mean_se(results["retrained"]["rho_img_r"])[0]
    s_real = mean_se(results["stock"]["rho_real_s"])[0]
    r_real = mean_se(results["retrained"]["rho_real_r"])[0]
    log(f"  stock     : rho_real {s_real:+.3f} -> rho_img {s_img:+.3f}")
    log(f"  retrained : rho_real {r_real:+.3f} -> rho_img {r_img:+.3f}")
    if r_real >= s_real - 0.02 and r_img < s_img - 0.05:
        log("  => CONFIRMED: the retrained value ranks target states BETTER on the")
        log("     real-latent distribution it was fit on, but WORSE at the imagined")
        log("     latents the planner scores. The bug is the training distribution")
        log("     (encode(s) instead of the rolled f^H(z,a)), not the lr.")
    elif r_img >= s_img - 0.05:
        log("  => NOT confirmed: the retrained head ranks imagined latents about as")
        log("     well as stock. The MPPI drop must come from elsewhere -- dig further")
        log("     (e.g. absolute-scale/terminal-weight interaction, reward-term balance).")
    else:
        log("  => partial: check the table -- ordering degraded on BOTH distributions,")
        log("     which points at the value TARGET semantics rather than the latent it")
        log("     is evaluated on.")

    os._exit(0)


if __name__ == "__main__":
    main()
