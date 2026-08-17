"""Does stage-3 adapt improve PREDICTION but degrade candidate RANKING?

THE PUZZLE. On Coffee/C50, 100k transitions of dynamics adapt lowered H-step
latent rollout error by 25-30% (held-out pools included, so not forgetting) yet
the planner got no better -- and got clearly WORSE whenever round-0 was already
good (a meta-analysis over 134 SimDist adapt runs gives corr(start, gain) =
-0.72, with every arm converging to ~0.7 regardless of where it started).

THE HYPOTHESIS. MPPI never consumes prediction accuracy; it consumes the ORDER
its value head assigns to candidates. stage-1 distills dynamics AND value
together, so V_hat is calibrated on the latents the stage-1 dynamics produces.
stage-3 trains dynamics ONLY (adapt_value=0), so f^H drifts into latent regions
V_hat was never fit on. Mean accuracy can improve while the within-candidate
ordering rots -- exactly CLAUDE.md 27's eps_bar (harmless, constant within a
candidate set) vs eps_tilde (the only part argmax feels) split, and the mirror
image of 33 (there the VALUE was swapped and the dynamics kept).

WHAT THIS MEASURES. For each probe state, build K candidate action sequences,
then score them two ways:
  model : V_hat( f^H(z, a) )                        -- what MPPI actually ranks by
  truth : V_hat( encode(s^H_real) )                 -- same value head, but the
          terminal state comes from replaying the SAME actions in the REAL
          target simulator (cold-start, qacc_warmstart=0)
and report
  * spearman(model, truth) per state, averaged   <- the ranking fidelity
  * top-k overlap between the two orderings      <- what elite selection sees
  * std_i(model) and std_i(truth)                <- the ranking SIGNAL magnitude
Isolating the value head this way means any difference is attributable to the
dynamics alone: both columns use the identical, frozen V_hat.

Reading it: if adapt lowers spearman / top-k overlap while lowering rollout
error, the "more accurate but worse-ranked" story is confirmed and the fix is to
make stage-3 optimise ordering (or re-fit the value alongside), not accuracy.
If spearman is unchanged, the degradation lives somewhere else and this line
should be dropped.

  python scripts/diag_coffee_rank_fidelity.py --ckpts stage1=... adapted=...
"""
import argparse
import os
import sys

import numpy as np
import torch
import pyrallis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


TARGET = dict(gripper_kp_scale=0.30, lid_damping_scale=50.0, mass_scale=3.0,
              friction_scale=0.5, pod_holder_friction_scale=2.0)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path")
    ap.add_argument("--stage1_cfg_dir",
                    default="exps/mimicgen/coffee_simdist_diversev1_h5_seed1")
    ap.add_argument("--demo",
                    default="release/data/mimicgen/coffee_d0_target_C50/dataset.hdf5")
    ap.add_argument("--num_states", type=int, default=60)
    ap.add_argument("--num_cand", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cand_std", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import h5py
    import train_simdist
    from rl.simdist_agent import SimDistAgent
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    rng = np.random.RandomState(args.seed)

    # ---- probe states: (sim_state, state) pairs from target demos -----------
    picks = []
    with h5py.File(args.demo, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            g = f["data"][k]
            ss = np.asarray(g["obs"]["sim_state"])
            st = np.asarray(g["obs"]["state"], dtype=np.float32)
            n = min(len(ss), len(st)) - args.horizon - 1
            if n <= 1:
                continue
            for t in rng.choice(n, size=min(4, n), replace=False):
                picks.append((ss[t], st[t]))
    rng.shuffle(picks)
    picks = picks[:args.num_states]
    log(f"[probe] {len(picks)} states from {os.path.basename(args.demo)}")

    env = PixelRobosuite("Coffee_D0", "Panda", episode_length=400, use_state=True,
                         device="cpu", rl_image_size=96, camera_names=["agentview"],
                         domain_shift=DomainShiftConfig(mode="fixed",
                                                        mask_hole_offset_in_obs=False,
                                                        **TARGET))
    env.reset()   # REQUIRED: this is where the shift is applied (CLAUDE.md 27)
    raw = env.env

    H, K, A = args.horizon, args.num_cand, env.action_dim
    cands = np.clip(rng.randn(len(picks), K, H, A) * args.cand_std, -1, 1).astype(np.float32)

    # ---- TRUE terminal states: replay every candidate in the real target sim
    log(f"[probe] rolling {len(picks)}x{K} candidates in the REAL target sim ...")
    true_terminal = np.zeros((len(picks), K, 66), dtype=np.float32)
    for i, (sim_state, _) in enumerate(picks):
        for j in range(K):
            raw.done = False
            raw.timestep = 0
            raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
            raw.sim.data.qacc_warmstart[:] = 0.0
            raw.sim.forward()
            ctrl = raw.robots[0].controller
            if hasattr(ctrl, "update"):
                ctrl.update(force=True)
            if hasattr(ctrl, "reset_goal"):
                ctrl.reset_goal()
            for h in range(H):
                obs, _, _, _ = raw.step(np.asarray(cands[i, j, h], dtype=np.float64))
            true_terminal[i, j] = np.concatenate(
                [obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                 obs["robot0_gripper_qpos"], obs["object-state"]])
        if (i + 1) % 20 == 0:
            log(f"   {i + 1}/{len(picks)}")

    cfg = pyrallis.load(train_simdist.MainConfig,  # type: ignore
                        open(os.path.join(args.stage1_cfg_dir, "cfg.yaml")))
    S0 = torch.as_tensor(np.stack([p[1] for p in picks]))
    CA = torch.as_tensor(cands)
    TT = torch.as_tensor(true_terminal)

    log("\n" + "=" * 96)
    log(f"{'ckpt':<16}{'spearman':>10}{'top%d ovl' % args.topk:>10}"
        f"{'std_i(model)':>14}{'std_i(truth)':>14}{'|bias|':>10}")
    log("-" * 96)

    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        agent = SimDistAgent(True, (66,), (9,), A, cfg.rl_camera, cfg.simdist)
        agent.load_state_dict(torch.load(path, map_location="cpu"))
        agent = agent.to("cuda").eval()
        for p in agent.parameters():
            p.requires_grad_(False)
        m = agent.model
        sp, ov, sm, st_, bi = [], [], [], [], []
        with torch.no_grad():
            for i in range(len(picks)):
                z = m.encode(S0[i:i + 1].to("cuda")).expand(K, -1).contiguous()
                for h in range(H):
                    z = m.next(z, CA[i, :, h].to("cuda"))
                v_model = m.predict_value(z).flatten().cpu().numpy()
                v_true = m.predict_value(
                    m.encode(TT[i].to("cuda"))).flatten().cpu().numpy()
                sp.append(spearman(v_model, v_true))
                a = set(np.argsort(-v_model)[:args.topk])
                b = set(np.argsort(-v_true)[:args.topk])
                ov.append(len(a & b) / args.topk)
                sm.append(v_model.std()); st_.append(v_true.std())
                bi.append(abs(v_model.mean() - v_true.mean()))
        log(f"{name:<16}{np.mean(sp):>10.3f}{np.mean(ov):>10.3f}"
            f"{np.mean(sm):>14.5f}{np.mean(st_):>14.5f}{np.mean(bi):>10.5f}")
        del agent
        torch.cuda.empty_cache()

    log("\nspearman/top-k = does the model ORDER candidates like the real sim does "
        "(same frozen V_hat on both sides, so this isolates the dynamics).")
    log("std_i(model) vs std_i(truth) = ranking SIGNAL; |bias| is a per-state "
        "constant and is INVISIBLE to CEM's topk+softmax (CLAUDE.md 33's eps_bar).")


if __name__ == "__main__":
    main()
