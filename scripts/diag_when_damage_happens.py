"""Does the annulus deficit happen WHERE the uncertainty gate would open?

State so far (CLAUDE.md §33). With oracle dynamics the wide1.5 domain splits cleanly:
core 1.000 (n=121) vs annulus 0.905+-0.022 (n=179), z~4.3, against a true ceiling of
0.987+-0.007 (the wide-trained IBRL expert, n=300). Probing the existing 10-head value
ensemble showed its disagreement rises with eccentricity ONLY at reset states
(annulus/core 1.58x, and the far-OOD control lifts it 2-5.7x so the heads are NOT
collapsed), but INVERTS over full on-policy trajectories (0.68x) -- because once the
needle is grasped and carried the state is back inside the source distribution (source
trajectories move the needle all over the workspace), and because raw disagreement is
~0.6% of V, so it mostly tracks the value level rather than novelty.

So the gate would only ever open during the first stretch of an episode. That is only
useful if the deficit is CAUSED there. This measures both, on the same rollouts:

  (A) WHERE THE GATE WOULD OPEN: relative disagreement (disagree / |V|, to remove the
      "proportional to V" artefact) as a function of episode step, core vs annulus.
  (B) WHERE THE DAMAGE HAPPENS: for episodes that FAIL, hand over to the wide-trained
      expert (0.987 ceiling) at t = 0, 20, 40, 80, ... and measure p_recover(t). The step
      at which p_recover collapses is when the episode became unrecoverable.

If (A) and (B) coincide early, a gated residual is aimed at the right place. If the damage
is already done before the gate opens, or long after it closes, the gate is useless no
matter how good the residual is.

Run:
  python scripts/diag_when_damage_happens.py --episodes 40 --restarts 3
"""
import argparse
import os
import sys

import numpy as np
import torch
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

ERR = sys.stderr
BOX_C, BOX_HW = np.array([-0.125, 0.2]), np.array([0.075, 0.05])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sut_run", default="exps/mimicgen/threading_simdist_dynens_adapt_wide1.5_seed1",
                    help="system under test (its learned MPPI generates the rollouts)")
    ap.add_argument("--sut_ckpt", default="model0.pt")
    ap.add_argument("--vens_run", default="exps/mimicgen/threading_simdist_vens_adapt_long_h5_seed1",
                    help="value ENSEMBLE used only as an uncertainty probe over raw states")
    ap.add_argument("--expert", default="exps/mimicgen/threading_ibrl_target_wide15_seed1/model0.pt",
                    help="wide-trained IBRL expert (0.987 ceiling) used for the takeover test")
    ap.add_argument("--needle_scale", type=float, default=1.5)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--takeover_at", default="0,20,40,80,140")
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--max_fail", type=int, default=12, help="cap the takeover cost")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    import train_rl
    from train_simdist import MainConfig, Workspace
    from train_simdist_adapt import AdaptConfig
    from env.domain_shift import InitDistConfig
    from env.robosuite_wrapper import PixelRobosuite

    def build(run, ckpt="model0.pt", planner=1):
        a = pyrallis.load(AdaptConfig, open(os.path.join(run, "cfg.yaml")))  # type: ignore
        s = pyrallis.load(MainConfig, open(  # type: ignore
            os.path.join(os.path.dirname(a.stage1_ckpt), "cfg.yaml")))
        s.use_wb = 0
        s.simdist.use_planner = planner
        if planner:
            from rl.simdist_planner import apply_original_recipe
            apply_original_recipe(s.simdist.planner)
        if a.eval_hand_init_perturb_joint_sigma is not None:
            s.eval_hand_init_perturb_joint_sigma = a.eval_hand_init_perturb_joint_sigma
        w = Workspace(s, from_main=False)
        w.agent.load_state_dict(torch.load(os.path.join(run, ckpt)))
        return w.agent.to(dev).eval(), w.eval_env_params, s

    sut, ep_params, _ = build(args.sut_run, args.sut_ckpt, planner=1)
    vens, _, _ = build(args.vens_run, "model0.pt", planner=0)     # probe only, no planning
    log(f"[sut ] {args.sut_run} (learned MPPI)")
    log(f"[vens] {args.vens_run} | heads={vens.model.num_value_heads()}")

    nsc = args.needle_scale
    ep = dict(ep_params)
    ep["init_distribution"] = InitDistConfig(
        enabled=True,
        needle_x=(BOX_C[0] - BOX_HW[0] * nsc, BOX_C[0] + BOX_HW[0] * nsc),
        needle_y=(BOX_C[1] - BOX_HW[1] * nsc, BOX_C[1] + BOX_HW[1] * nsc),
        needle_z_rot=(-2 * np.pi / 3, -np.pi / 3),
        # D0 pins the tripod; InitDistConfig defaults are the D1 RANDOM ranges, and
        # leaving them would silently randomize the hole = a different task.
        tripod_x=(0.0, 0.0), tripod_y=(-0.15, -0.15), tripod_z_rot=(np.pi / 2, np.pi / 2))
    env = PixelRobosuite(**ep)
    log(f"[env ] shift={ep['domain_shift'].mode} kp={ep['domain_shift'].gripper_kp_scale} "
        f"| needle box x{nsc} | tripod pinned")

    ecfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(args.expert), "cfg.yaml")))
    expert = train_rl.load_frozen_agent(args.expert, env, ecfg, dev)
    edesc = train_rl.attach_ibrl_prior(expert, args.expert, env, dev)
    expert.cfg.act_method = "ibrl" if edesc else "rl"
    log(f"[exp ] {args.expert} | prior={edesc or 'none'}")

    @torch.no_grad()
    def rel_disagree(states):
        """disagree / |V| -- the raw disagreement is ~0.6% of V, so it otherwise just
        tracks the value level instead of novelty."""
        V = vens.model.predict_value_all(vens.model.encode(states.to(dev)))   # [N,B]
        return (V.std(0) / V.mean(0).abs().clamp_min(1e-6)).cpu().numpy()

    # ---------------- roll the system under test ----------------
    eps = []
    for e in range(args.episodes):
        np.random.seed(args.seed + e)
        torch.manual_seed(args.seed + e)
        obs, _ = env.reset()
        n0 = obs["state"][9:12].cpu().numpy()[:2]
        ecc = float(np.max(np.abs((n0 - BOX_C) / (BOX_HW * nsc))))
        S, SS, ok = [], [], 0.0
        for t in range(400):
            with torch.no_grad():
                a = sut.act(obs, eval_mode=True)
            obs, reward, terminal, _, _ = env.step(a)
            S.append(obs["state"].cpu().numpy().copy())
            SS.append(env.env.sim.get_state().flatten().copy())
            if reward > 0:
                ok = 1.0
            if terminal:
                break
        eps.append(dict(S=np.stack(S), SS=SS, ok=ok, ecc=ecc))
        if (e + 1) % 10 == 0:
            log(f"    ep{e+1}: success {np.mean([x['ok'] for x in eps]):.2f}")
    succ = np.array([x["ok"] for x in eps])
    ecc = np.array([x["ecc"] for x in eps])
    ann = ecc > 1.0 / nsc
    log(f"[roll] success {succ.mean():.3f} | core {succ[~ann].mean():.3f} (n={(~ann).sum()}) "
        f"| annulus {succ[ann].mean():.3f} (n={ann.sum()})")

    def blk(t):
        log("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    # ---------------- (A) where would the gate open? ----------------
    blk("(A) relative disagreement  disagree/|V|  by episode step")
    bins = [(0, 20), (20, 40), (40, 80), (80, 140), (140, 400)]
    log(f"{'step':>12}{'core':>14}{'annulus':>14}{'ann/core':>11}")
    for lo, hi in bins:
        rc, ra = [], []
        for x in eps:
            idx = np.arange(len(x["S"]))
            m = (idx >= lo) & (idx < hi)
            if m.sum() == 0:
                continue
            d = rel_disagree(torch.from_numpy(x["S"][m]).float())
            (ra if x["ecc"] > 1.0 / nsc else rc).append(d.mean())
        if rc and ra:
            log(f"{f'{lo}-{hi}':>12}{np.mean(rc):>14.5f}{np.mean(ra):>14.5f}"
                f"{np.mean(ra)/max(np.mean(rc),1e-9):>11.2f}x")

    # ---------------- (B) where does the damage happen? ----------------
    blk("(B) takeover by the 0.987 expert at step t -> p_recover(t), FAILED annulus eps")
    fails = [x for x in eps if x["ok"] == 0.0 and x["ecc"] > 1.0 / nsc][: args.max_fail]
    log(f"  {len(fails)} failed annulus episodes")
    if fails:
        ts = [int(v) for v in args.takeover_at.split(",")]
        log(f"{'takeover t':>12}{'p_recover':>12}{'n':>6}")
        for t0 in ts:
            rec = []
            for x in fails:
                if t0 >= len(x["SS"]):
                    continue
                r = []
                for _ in range(args.restarts):
                    restore(env, x["SS"][t0])
                    obs, _ = env._extract_images(env.env._get_observations())
                    s = 0.0
                    for _ in range(400 - t0):
                        with torch.no_grad():
                            a = expert.act(obs, eval_mode=True)
                        obs, reward, terminal, _, _ = env.step(a)
                        if reward > 0:
                            s = 1.0
                            break
                        if terminal:
                            break
                    r.append(s)
                rec.append(np.mean(r))
            if rec:
                log(f"{t0:>12}{np.mean(rec):>12.3f}{len(rec):>6}")
        log("\n  p_recover high early then collapsing => the damage happens in that window.")
        log("  If p_recover is already ~0 at t=0 the episode was doomed by its INIT, and no")
        log("  value fix can help; if it stays high late, the damage is late and a gate that")
        log("  only opens early is aimed at the wrong place.")

    blk("VERDICT")
    log("  the gated-residual plan only makes sense if (A) high-disagreement window and")
    log("  (B) p_recover collapse window OVERLAP. Compare the two tables above.")
    os._exit(0)


if __name__ == "__main__":
    main()
