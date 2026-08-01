"""ROOT-CAUSE experiment for "value adapt makes SimDist worse" (7 observational
hypotheses refuted; see scripts/diag_value_adapt_*.py and diag_elite_action_geometry.py).

Switches from observation to INTERVENTION. The retrained (failing) value head differs
from the stock (working) one along exactly two recipe axes:

    latent  : it was fit at encode(s)      -- stage-1 fits at the ROLLED z_next_pred
              (rl/simdist_agent.py:526/613 vs :301)
    teacher : it regressed the target-expert critic Q (`values` in the diverse sets)
              -- stage-1 regressed V_src distilled from the SOURCE IBRL critic

plus a third possibility nobody has tested: that ANY re-fit of the value head breaks
the planner, i.e. the closed loop is hypersensitive to the value function and the
"recipe" is irrelevant.

That third possibility is testable FIRST and decides the whole design:

  E0 self-distill (the control): re-initialize the value head and regress the STOCK
     head's OWN output at the latents the planner queries. By construction the fitted
     function equals stock's to whatever precision we train to (~1e-5).
       E0 keeps ~0.75  => the collapse IS attributable to teacher/latent -> run the 2x2
       E0 collapses    => the collapse is CLOSED-LOOP FRAGILITY, not the adapt target:
                          a value function that differs by 1e-5 already kills it, while
                          the retrained head differs by ~0.05. That would also explain
                          the whole §27 series of failed value-side interventions.

  then the 2x2 (only if E0 passes):
       A  real   x expertq   (should reproduce the known failing arm, ~0.37)
       B  rolled x expertq   (isolates the latent axis)
       C  real   x vsrc      (isolates the teacher axis)
       D  rolled x vsrc      (the stock recipe re-run from scratch -> should be ~0.75)

Everything except the value head is FROZEN (encoder/dynamics/reward/policy loaded from
the stock adapt ckpt and never touched), so the value head is the only variable -- the
same 20/70 tensors that separate the two ckpts today.

Usage:
  # the decisive control first
  python scripts/value_head_recipe_ablate.py --arm E0 --steps 40000 --num_eval_episode 200
  # then, if E0 passes, the recipe 2x2
  python scripts/value_head_recipe_ablate.py --arm A --steps 100000
  ...
"""
import argparse
import os
import sys
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from evaluate import run_eval, run_eval_mp  # noqa: E402
from train_simdist import MainConfig, Workspace  # noqa: E402

ERR = sys.stderr
ARMS = {            # arm -> (latent, teacher)
    "E0": ("rolled", "stock_self"),
    "A": ("real", "expertq"),
    "B": ("rolled", "expertq"),
    "C": ("real", "vsrc"),
    "D": ("rolled", "vsrc"),
}
DATA = {
    # SOURCE-domain distillation data -- this is what stage-1 actually trained the
    # (working) value head on. Included so the "same teacher, different data" axis
    # can be tested directly.
    "src": "release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5",
    "v3": "release/data/mimicgen/threading_d0_target_diverse_v3/dataset.hdf5",
    "v4": "release/data/mimicgen/threading_d0_target_diverse_v4/dataset.hdf5",
    "v2": "release/data/mimicgen/threading_d0_target_diverse_v2/dataset.hdf5",
}


def log(*a):
    print(*a, file=ERR, flush=True)


BASE_CFG_DIR = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1"


def build(ckpt, sigma, dev="cuda", base_cfg_dir=None):
    cfg: MainConfig = pyrallis.load(MainConfig, open(  # type: ignore
        os.path.join(base_cfg_dir or BASE_CFG_DIR, "cfg.yaml")))
    cfg.simdist.use_planner = 1
    cfg.use_wb = 0
    from rl.simdist_planner import apply_original_recipe
    apply_original_recipe(cfg.simdist.planner)
    if sigma is not None:
        cfg.eval_hand_init_perturb_joint_sigma = sigma
    ws = Workspace(cfg, from_main=False)
    ws.agent.load_state_dict(torch.load(ckpt))
    return ws.agent.to(dev).eval(), ws.eval_env_params, cfg


class SegmentPool:
    """Flat GPU pool of (s_t, a_{t..t+H-1}, s_{t+H}, values_t, values_{t+H}) segments."""

    def __init__(self, paths, episodes_per, H, rng, dev, need_values):
        S0, A, SH, V0, VH = [], [], [], [], []
        for p in paths:
            if not os.path.exists(p):
                log(f"  !! missing {p}")
                continue
            with h5py.File(p, "r") as f:
                d = f["data"]
                keys = list(d.keys())
                sel = rng.choice(len(keys), size=min(episodes_per, len(keys)), replace=False)
                for ei in sel:
                    ep = d[keys[ei]]
                    s = np.asarray(ep["obs"]["state"][()], dtype=np.float32)
                    a = np.asarray(ep["actions"][()], dtype=np.float32)
                    T = min(s.shape[0], a.shape[0] + 1)
                    if T < H + 2:
                        continue
                    has_v = "values" in ep
                    if need_values and not has_v:
                        continue
                    v = (np.asarray(ep["values"][()], dtype=np.float32).reshape(-1)
                         if has_v else np.zeros(T, np.float32))
                    idx = np.arange(0, T - H - 1)
                    S0.append(s[idx]); SH.append(s[idx + H])
                    A.append(np.stack([a[i:i + H] for i in idx]))
                    V0.append(v[idx]); VH.append(v[idx + H])
        self.s0 = torch.from_numpy(np.concatenate(S0)).to(dev)
        self.sH = torch.from_numpy(np.concatenate(SH)).to(dev)
        self.a = torch.from_numpy(np.concatenate(A)).to(dev)
        self.v0 = torch.from_numpy(np.concatenate(V0)).to(dev)
        self.vH = torch.from_numpy(np.concatenate(VH)).to(dev)
        self.n = self.s0.shape[0]

    def sample(self, B, gen=None):
        i = torch.randint(0, self.n, (B,), device=self.s0.device, generator=gen)
        return self.s0[i], self.a[i], self.sH[i], self.vH[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--stock_ckpt", default="exps/mimicgen/threading_simdist_adapt_seed1/latest.pt")
    ap.add_argument("--source_agent", default="exps/mimicgen/threading_ibrl_source_seed1/model0.pt",
                    help="for teacher=vsrc: V_src(s) = min-ensemble Q_src(s, pi_src(s))")
    ap.add_argument("--base_cfg_dir", default=BASE_CFG_DIR,
                    help="stage-1 cfg dir defining the ARCH (e.g. the dynens stage-1 for the "
                         "dynamics-ensemble system)")
    ap.add_argument("--datasets", default="v3,v4")
    ap.add_argument("--episodes_per", type=int, default=400)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--explore_std", type=float, default=0.5,
                    help="for latent=rolled: extra action noise so the rolled latents COVER "
                         "the planner's candidate cloud, not just the dataset action")
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--num_eval_episode", type=int, default=200)
    ap.add_argument("--num_eval_proc", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=-1.0)
    ap.add_argument("--value_ema", type=float, default=0.0,
                    help="EMA decay for the value head weights (0 = off). The score swings "
                         "8-9 SE across checkpoints for external-target arms; EMA targets that "
                         "weight-wandering directly (evals + ckpts use the EMA weights).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save_dir", default="")
    args = ap.parse_args()

    latent, teacher = ARMS[args.arm]
    sigma = None if args.sigma < 0 else args.sigma
    dev = "cuda"
    save_dir = args.save_dir or f"exps/mimicgen/threading_valrecipe_{args.arm}_seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    agent, ep_params, cfg = build(args.stock_ckpt, sigma, dev, args.base_cfg_dir)
    stock_ref, _, _ = build(args.stock_ckpt, sigma, dev, args.base_cfg_dir)      # frozen reference copy
    model, H = agent.model, agent.planner.cfg.horizon
    wm = cfg.simdist.world_model
    log(f"[{args.arm}] latent={latent} teacher={teacher} | H={H} | "
        f"discrete_heads={wm.discrete_heads} | save_dir={save_dir}")

    # ---- teacher for vsrc: the frozen SOURCE IBRL critic's V ----
    vsrc = None
    if teacher == "vsrc":
        import train_rl
        from env.robosuite_wrapper import PixelRobosuite
        tmp_env = PixelRobosuite(**ep_params)
        src_cfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
            os.path.join(os.path.dirname(args.source_agent), "cfg.yaml")))
        src = train_rl.load_frozen_agent(args.source_agent, tmp_env, src_cfg, dev)

        @torch.no_grad()
        def vsrc(states):
            return src.value({"state": states}).view(-1)

    pool = SegmentPool([DATA[d] for d in args.datasets.split(",")], args.episodes_per,
                       H, rng, dev, need_values=(teacher == "expertq"))
    # TRUE held-out pool: a DIFFERENT episode draw, never sampled during training.
    # (the old in-loop "heldout_mse" resampled the TRAIN pool -> optimistic by 1.7-9x,
    #  see scripts/diag_valrecipe_generalization.py)
    val_pool = SegmentPool([DATA[d] for d in args.datasets.split(",")], 150, H,
                           np.random.default_rng(args.seed + 100000), dev,
                           need_values=(teacher == "expertq"))
    log(f"[{args.arm}] pool: {pool.n} segments | val pool (unseen eps): {val_pool.n}")

    # ---- reinit the value head; everything else frozen ----
    agent.reinit_head("value")
    vparams = [p for hd in agent.model.value_heads() for p in hd.parameters()]
    opt = torch.optim.Adam(vparams, lr=args.lr)
    frozen_before = {k: v.clone() for k, v in agent.model.state_dict().items()
                     if not k.startswith(("value", "value_target"))}

    # ---- optional EMA over the value head weights ----
    ema = ({k: v.detach().clone() for k, v in
            [(f"{i}.{n}", p) for i, hd in enumerate(agent.model.value_heads())
             for n, p in hd.named_parameters()]} if args.value_ema > 0 else None)

    def ema_update():
        if ema is None:
            return
        d = args.value_ema
        with torch.no_grad():
            for i, hd in enumerate(agent.model.value_heads()):
                for n, p in hd.named_parameters():
                    ema[f"{i}.{n}"].mul_(d).add_(p.detach(), alpha=1 - d)

    def ema_swap():
        """swap EMA weights in (and the online weights into the shadow) -- call twice."""
        if ema is None:
            return
        with torch.no_grad():
            for i, hd in enumerate(agent.model.value_heads()):
                for n, p in hd.named_parameters():
                    tmp = p.detach().clone()
                    p.copy_(ema[f"{i}.{n}"])
                    ema[f"{i}.{n}"] = tmp

    def eval_mppi(seed):
        st = np.random.get_state()
        try:
            sc = run_eval_mp(env_params=ep_params, agent=agent, num_proc=args.num_eval_proc,
                             num_game=args.num_eval_episode, seed=seed, verbose=False)
        finally:
            np.random.set_state(st)
        return float(np.mean(sc))

    @torch.no_grad()
    def latents_and_targets(B, src_pool=None):
        s0, a, sH, vH = (src_pool or pool).sample(B)
        z0 = model.encode(s0)
        if latent == "rolled":
            aa = a.clone()
            if args.explore_std > 0:                     # cover the candidate cloud
                m = torch.rand(B, 1, 1, device=dev) < 0.5
                aa = torch.where(m, (aa + torch.randn_like(aa) * args.explore_std).clamp(-1, 1), aa)
            z = z0
            for t in range(H):
                z = model.next(z, aa[:, t])
            zq = z                                        # query latent = rolled prediction
        else:
            zq = model.encode(sH)                         # query latent = real encoding
        if teacher == "stock_self":
            tgt = stock_ref.model.predict_value(zq, target=False).view(-1)
        elif teacher == "expertq":
            tgt = vH
        else:
            tgt = vsrc(sH)
        return zq, tgt

    best = -1.0
    for step in range(1, args.steps + 1):
        zq, tgt = latents_and_targets(args.batch_size)
        loss = agent._value_head_loss(zq, tgt, wm.discrete_heads, wm.num_bins, wm.vmin, wm.vmax)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(vparams, wm.grad_clip)
        opt.step()
        ema_update()
        with torch.no_grad():
            for src_h, tgt_h in zip(agent.model.value_heads(), agent.model.value_target_heads()):
                tgt_h.load_state_dict(src_h.state_dict())
        if step % 1000 == 0:
            with torch.no_grad():
                zq2, t2 = latents_and_targets(4096)
                pred = agent.model.predict_value(zq2, target=False).view(-1)
                mse = float(F.mse_loss(pred, t2))
                zq3, t3 = latents_and_targets(4096, val_pool)
                vmse = float(F.mse_loss(agent.model.predict_value(zq3, target=False).view(-1), t3))
            log(f"  [{args.arm}] step {step}: loss={float(loss):.6f} train_mse={mse:.6f} "
                f"VAL_mse={vmse:.6f} "
                f"pred={float(pred.mean()):.4f} tgt={float(t2.mean()):.4f} gn={float(gn):.3f}")
        if step % args.eval_every == 0 or step == args.steps:
            # sanity: nothing but the value head moved
            now = agent.model.state_dict()
            drift = max((now[k] - v).abs().max().item() for k, v in frozen_before.items())
            ema_swap()          # evaluate (and save) the EMA weights
            sc = eval_mppi(seed=0)
            log(f"  [{args.arm}] step {step}: MPPI_target={sc:.3f} "
                f"({args.num_eval_episode} games)  frozen_drift={drift:.2e}")
            if sc > best:
                best = sc
                torch.save(agent.state_dict(), os.path.join(save_dir, "best.pt"))
            torch.save(agent.state_dict(), os.path.join(save_dir, "latest.pt"))
            ema_swap()          # restore the online weights for further training
    log(f"\n[{args.arm}] DONE  best MPPI_target={best:.3f}  (stock reference ~0.745-0.80, "
        f"known failing retrained arm ~0.37)")
    os._exit(0)


if __name__ == "__main__":
    main()
