"""Dynamics PREDICTION error vs ensemble size (1 / 3 / 5 / 10 heads), measured
the way CLAUDE.md iron rule #5 demands -- on the planner's own on-policy state
distribution, against the TRUE simulator, for the planner's own candidate cloud.

Why not just report val-MSE: §30 showed mean mm-error has ~zero predictive power
for planner score, and §27's dose-response showed only the PERSISTENT
(state-dependent, common-to-all-candidates) part of the error hurts; iid-ish
candidate-to-candidate scatter is averaged out by the closed loop. So every error
is split into
    common  e_bar(s)       = mean over the K candidates at state s
    spread  e_k - e_bar(s) = candidate-idiosyncratic part
and the planner-facing quantity (the H-step return J that CEM ranks by) is split
the same way: eps = J_model - J_true, eps_bar(s) is argmax-immune, eps_tilde is
what can re-order candidates (§27#5: look at sup|eps_tilde| and Var_s[eps_bar]).

Protocol (all models see the IDENTICAL (s, action-sequence) pairs => paired):
  1. A behaviour agent (default: the §31 dynens adapted ckpt, the 0.962 planner)
     runs MPPI in the chosen domain. Every --stride steps we snapshot sim_state.
  2. At each snapshot, K candidate H-step action sequences are drawn exactly as
     the original-recipe MPPI's first iteration does: n_base policy rollouts
     (+ base_policy_std noise) and n_rand N(0, init_std) samples, clamped.
  3. Each candidate is replayed in the TRUE simulator from the snapshot
     (ProbePool from eval_oracle_dyn_mppi.py) -> s_1..s_H.
  4. Offline, for each model spec: z0=enc(s0), roll the candidates through the
     model (ensemble MEAN over the first `heads` heads, as the planner does),
     compare with the truth.

Error measures per model (reported at t=1 and t=H):
  * needle mm, dynamics-attributable: |dec(z_hat_t) - dec(enc(s_t))|_needle.
    Subtracting the decoder's own round trip isolates the dynamics; the raw
    |dec(z_hat_t) - s_t| and the round-trip floor are printed too.
  * latent relative: |z_hat_t - enc(s_t)| / |enc(s_t) - enc(s_0)|.
  * return: std_k(eps_tilde)/std_k(J_true) (noise/signal), per-state Spearman,
    top-1 regret in units of std_k(J_true), std_s(eps_bar).

Run:
  python scripts/diag_dynens_pred_error.py --collect_out <npz> [--domain target]
  python scripts/diag_dynens_pred_error.py --data <npz> --models "<spec>;<spec>"
  spec = name|ckpt|stage1_dir|heads
"""
import argparse
import atexit
import copy
import json
import os
import sys

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.multiprocessing as mp

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist import MainConfig, Workspace       # noqa: E402
from rl.simdist_agent import SimDistAgent             # noqa: E402
from rl.simdist_planner import apply_original_recipe  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite      # noqa: E402
from scripts.eval_oracle_dyn_mppi import restore  # noqa: E402

NEEDLE = slice(9, 12)
RING = slice(23, 26)

BEHAVIOR_CKPT = "exps/mimicgen/threading_simdist_dynens_adapt_long_h5_seed1/ablate_frozen.pt"
BEHAVIOR_S1 = "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1"


# ------------------------------------------------------------ exact snapshots
# eval_oracle_dyn_mppi.restore() writes only qpos/qvel. That is NOT a full state:
# PandaGripper.current_action is an ACCUMULATOR (robosuite panda_gripper.py:55),
# and qacc_warmstart seeds the solver. Replaying the executed action from a
# restore()-only snapshot misses the main env's next state by up to 7e-3
# (needle 0.22 mm) -- the same order as the errors measured here. With the fields
# below the replay matches to ~1e-6 (checked every run, see collect()).
CTRL_KEYS = ("goal_pos", "goal_ori", "torques", "delta", "relative_ori", "ori_ref")


def snapshot(env):
    raw = env.env
    robots = []
    for r in raw.robots:
        c = r.controller
        robots.append(dict(
            grip=copy.deepcopy(getattr(r.gripper, "current_action", None)),
            ctrl={k: copy.deepcopy(getattr(c, k)) for k in CTRL_KEYS if hasattr(c, k)}))
    return dict(sim=raw.sim.get_state().flatten().copy(),
                warm=raw.sim.data.qacc_warmstart.copy(), robots=robots)


def restore_full(env, snap):
    restore(env, snap["sim"])
    raw = env.env
    raw.sim.data.qacc_warmstart[:] = snap["warm"]
    for r, rs in zip(raw.robots, snap["robots"]):
        if rs["grip"] is not None:
            r.gripper.current_action = copy.deepcopy(rs["grip"])
        for k, v in rs["ctrl"].items():
            setattr(r.controller, k, copy.deepcopy(v))
    raw.sim.forward()


def _worker(conn, env_params):
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = PixelRobosuite(**{**env_params, "device": "cpu"})
    env.reset()      # the domain shift is applied inside reset()
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            snap, acts = msg
            H, n, _ = acts.shape
            out = np.zeros((H, n, env.observation_shape[0]), dtype=np.float32)
            for i in range(n):
                restore_full(env, snap)
                for t in range(H):
                    obs, _, _, _, _ = env.step(torch.from_numpy(acts[t, i]))
                    out[t, i] = obs["state"].numpy()
            conn.send(out)
    finally:
        conn.close()


class Pool:
    """ProbePool (eval_oracle_dyn_mppi.py) with full snapshots."""

    def __init__(self, env_params, n):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            pc, cc = ctx.Pipe()
            pr = ctx.Process(target=_worker, args=(cc, env_params), daemon=True)
            pr.start()
            cc.close()
            self.conns.append(pc)
            self.procs.append(pr)
        atexit.register(self.close)

    def rollout(self, snap, actions):
        a = actions.detach().cpu().numpy()
        chunks = np.array_split(np.arange(a.shape[1]), len(self.conns))
        live = [(c, idx) for c, idx in zip(self.conns, chunks) if len(idx)]
        for c, idx in live:
            c.send((snap, a[:, idx]))
        return torch.tensor(np.concatenate([c.recv() for c, _ in live], axis=1), device="cuda")

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_cfg(stage1_dir, sigma):
    cfg: MainConfig = pyrallis.load(  # type: ignore
        MainConfig, open(os.path.join(stage1_dir, "cfg.yaml"), "r"))
    cfg.simdist.use_planner = 1
    apply_original_recipe(cfg.simdist.planner)
    cfg.eval_hand_init_perturb_joint_sigma = sigma
    cfg.source_agent = ""          # never load the source expert here
    return cfg


def build_agent(cfg, env, ckpt, heads):
    agent = SimDistAgent(bool(cfg.use_state), env.observation_shape, env.prop_shape,
                         env.action_dim, cfg.rl_camera, cfg.simdist, device="cuda")
    agent.load_state_dict(torch.load(ckpt, map_location="cuda"))   # strict
    agent = agent.to("cuda").eval()
    m = agent.model
    total = m.num_dynamics_heads()
    if heads > 0:
        assert 1 <= heads <= total, f"heads={heads} but ckpt has {total}"
        m.dynamics_extra = nn.ModuleList(list(m.dynamics_extra)[: heads - 1])
        if len(m.dynamics_prior) > 0:
            m.dynamics_prior = nn.ModuleList(list(m.dynamics_prior)[:heads])
    return agent, total


def env_params_for(ws, cfg, domain):
    p = dict(ws.eval_env_params)
    if domain == "source":
        p["domain_shift"] = cfg.train_domain_shift
        p["init_distribution"] = cfg.train_init_dist
    return p


# --------------------------------------------------------------------- collect
@torch.no_grad()
def collect(args):
    cfg = load_cfg(args.behavior_stage1, args.sigma)
    ws = Workspace(cfg, from_main=False)
    ep = env_params_for(ws, cfg, args.domain)
    log(f"[env] domain={args.domain} shift={ep['domain_shift']} "
        f"init_dist={ep['init_distribution']} sigma={ep['hand_init_perturb_joint_sigma']} "
        f"strict={ep['strict_success']} ep_len={ep['episode_length']}")
    env = PixelRobosuite(**ep)
    agent, total = build_agent(cfg, env, args.behavior_ckpt, 0)
    planner = agent.planner
    pc = planner.cfg
    H, A = pc.horizon, env.action_dim
    nb, nr = args.n_base, args.n_rand
    log(f"[behavior] {args.behavior_ckpt} heads={total} H={H} "
        f"base_std={pc.base_policy_std} init_std={pc.init_std}  K={nb}+{nr}")
    pool = Pool(ep, args.workers)

    S0, SIM, ACT, REAL, EPI, TSTEP, SUCC = [], [], [], [], [], [], []
    checks = []
    n_succ = 0
    for e in range(args.episodes):
        seed = args.seed * 100000 + e
        np.random.seed(seed)
        torch.manual_seed(seed)
        planner.reset()
        obs, _ = env.reset()
        t, first_idx, success = 0, len(S0), False
        while True:
            state = obs["state"].to("cuda").float().view(1, -1)
            rec = (t % args.stride == 0)
            if rec:
                sim_state = snapshot(env)
                z0 = agent.model.encode(state)
                base = planner._policy_rollout(z0, nb)
                if pc.base_policy_std > 0:
                    base = base + torch.randn_like(base) * pc.base_policy_std
                rand = pc.init_std * torch.randn(H, nr, A, device="cuda")
                acts = torch.cat([base, rand], dim=1).clamp(-1, 1)   # [H, K, A]
                real = pool.rollout(sim_state, acts)                 # [H, K, obs]
                S0.append(state[0].cpu().numpy())
                SIM.append(sim_state["sim"])
                ACT.append(acts.cpu().numpy())
                REAL.append(real.cpu().numpy())
                EPI.append(e)
                TSTEP.append(t)
            a = agent.act(obs, eval_mode=True)
            if rec and len(checks) < 8:
                # replay determinism: the pool's 1-step rollout of the EXECUTED
                # action must reproduce the main env's next state (iron rule #7:
                # a wrong-domain / un-reset probe env would show up here).
                one = pool.rollout(sim_state, a.view(1, 1, -1).to("cuda"))[0, 0]
            obs, _, terminal, success, _ = env.step(a)
            if rec and len(checks) < 8:
                checks.append(float((one.cpu() - obs["state"].cpu().float()).abs().max()))
            t += 1
            if terminal:
                break
        success = bool(success)
        n_succ += success
        SUCC += [success] * (len(S0) - first_idx)
        log(f"[ep {e:3d}] len={t:3d} success={int(success)}  states={len(S0)}  "
            f"running success={n_succ / (e + 1):.3f}")
    pool.close()
    log(f"[check] replay max|diff| over {len(checks)} probes = {max(checks):.2e} "
        f"(must be ~0, else the probe env is not the same physics)")
    assert max(checks) < 1e-4, "probe env does not reproduce the main env"
    np.savez_compressed(
        args.collect_out, s0=np.stack(S0), sim=np.stack(SIM), acts=np.stack(ACT),
        real=np.stack(REAL), episode=np.array(EPI), tstep=np.array(TSTEP),
        success=np.array(SUCC), n_base=nb, n_rand=nr, domain=args.domain,
        behavior=args.behavior_ckpt, sigma=args.sigma,
        success_rate=n_succ / args.episodes)
    log(f"[saved] {args.collect_out}  states={len(S0)}  success={n_succ / args.episodes:.3f}")
    os._exit(0)


# --------------------------------------------------------------------- measure
def spearman_rows(x, y):
    """Row-wise Spearman over the last dim. x, y: [B, K]."""
    rx = x.argsort(-1).argsort(-1).float()
    ry = y.argsort(-1).argsort(-1).float()
    rx = rx - rx.mean(-1, keepdim=True)
    ry = ry - ry.mean(-1, keepdim=True)
    return (rx * ry).sum(-1) / (rx.norm(dim=-1) * ry.norm(dim=-1) + 1e-12)


@torch.no_grad()
def returns_from_latents(model, planner, zs, acts):
    """zs: [H+1, N, z] latent path (index 0 = z0), acts [H, N, A] -> J [N],
    the same expression as MppiPlanner._estimate_returns (reward on (z_t,a_t),
    terminal value on z_H) so model and truth are scored identically."""
    H = acts.shape[0]
    ret = model.predict_value(zs[H], target=False) * planner.final_discount
    if planner.cfg.use_reward:
        r = torch.stack([model.predict_reward(zs[t], acts[t]) for t in range(H)], 0)
        ret = ret + (r * planner.discounts).sum(0)
    return ret


@torch.no_grad()
def measure(args):
    d = np.load(args.data, allow_pickle=True)
    s0 = torch.tensor(d["s0"], device="cuda")            # [B, obs]
    acts = torch.tensor(d["acts"], device="cuda")        # [B, H, K, A]
    real = torch.tensor(d["real"], device="cuda")        # [B, H, K, obs]
    nb = int(d["n_base"])
    B, H, K, A = acts.shape
    log(f"[data] {args.data}: B={B} states, H={H}, K={K} (base {nb}), "
        f"domain={d['domain']}, behavior success={float(d['success_rate']):.3f}")
    dist0 = torch.norm(s0[:, NEEDLE] - s0[:, RING], dim=1)
    near = dist0 <= torch.quantile(dist0, 1 / 3)          # contact-ish tertile

    results = []
    per_all = {}
    env_cache = {}
    for spec in args.models.split(";"):
        name, ckpt, s1dir, heads = spec.split("|")
        heads = int(heads)
        cfg = load_cfg(s1dir, 0.1)
        if s1dir not in env_cache:
            ws = Workspace(cfg, from_main=False)
            env_cache[s1dir] = ws.eval_env
        agent, total = build_agent(cfg, env_cache[s1dir], ckpt, heads)
        model, planner = agent.model, agent.planner
        assert planner.cfg.horizon == H, (name, planner.cfg.horizon, H)
        assert planner.value_fn is None and planner.reward_fn is None

        a = acts.permute(1, 0, 2, 3).reshape(H, B * K, A)                 # [H, BK, A]
        z0 = model.encode(s0)                                             # [B, z]
        zhat = [z0.repeat_interleave(K, 0)]
        for t in range(H):
            zhat.append(model.next(zhat[-1], a[t]))
        zhat = torch.stack(zhat, 0)                                       # [H+1, BK, z]
        r = real.permute(1, 0, 2, 3).reshape(H, B * K, -1)                # [H, BK, obs]
        ztrue = torch.cat([zhat[:1], model.encode(r.reshape(H * B * K, -1)).view(H, B * K, -1)], 0)

        row = dict(name=name, heads=heads, total_heads=total)
        per = {}
        for t in sorted({1, H}):
            dec_hat = model.decode(zhat[t])[:, NEEDLE]
            dec_true = model.decode(ztrue[t])[:, NEEDLE]
            s_true = r[t - 1][:, NEEDLE]
            e = ((dec_hat - dec_true) * 1000.0).view(B, K, 3)             # mm, dyn-attributable
            ebar = e.mean(1, keepdim=True)
            common = ebar.squeeze(1).norm(dim=-1)                          # [B]
            spread = (e - ebar).norm(dim=-1)                               # [B, K]
            tot = e.norm(dim=-1)
            raw = ((dec_hat - s_true) * 1000.0).norm(dim=-1).mean()
            rt = ((dec_true - s_true) * 1000.0).norm(dim=-1).mean()
            zmove = (ztrue[t] - ztrue[0]).norm(dim=-1).clamp_min(1e-6)
            zrel = ((zhat[t] - ztrue[t]).norm(dim=-1) / zmove).view(B, K)
            per[f"t{t}_mm"] = tot.mean(1)
            per[f"t{t}_mm_common"] = common
            per[f"t{t}_mm_spread"] = spread.pow(2).mean(1).sqrt()
            per[f"t{t}_zrel"] = zrel.mean(1)
            per[f"t{t}_zrel_base"] = zrel[:, :nb].mean(1)
            row[f"t{t}"] = dict(
                mm_total=float(tot.mean()), mm_common_rms=float(common.pow(2).mean().sqrt()),
                mm_spread_rms=float(spread.pow(2).mean().sqrt()),
                mm_total_near=float(tot[near].mean()),
                mm_common_rms_near=float(common[near].pow(2).mean().sqrt()),
                mm_p95=float(torch.quantile(tot.flatten(), 0.95)),
                mm_raw=float(raw), mm_roundtrip=float(rt),
                latent_rel=float(zrel.mean()), latent_rel_base=float(zrel[:, :nb].mean()))

        Jm = returns_from_latents(model, planner, zhat, a).view(B, K)
        Jt = returns_from_latents(model, planner, ztrue, a).view(B, K)
        for tag, sl in (("all", slice(None)), ("base", slice(0, nb))):
            jm, jt = Jm[:, sl], Jt[:, sl]
            eps = jm - jt
            ebar = eps.mean(1)
            etil = eps - ebar[:, None]
            sig = jt.std(1)
            best = jt.max(1).values
            pick = jt.gather(1, jm.argmax(1, keepdim=True)).squeeze(1)
            per[f"J{tag}_spearman"] = spearman_rows(jm, jt)
            per[f"J{tag}_regret"] = (best - pick) / sig.clamp_min(1e-9)
            per[f"J{tag}_noise"] = etil.std(1)
            per[f"J{tag}_sup"] = etil.abs().max(1).values
            per[f"J{tag}_signal"] = sig
            per[f"J{tag}_ebar"] = ebar
            row[f"J_{tag}"] = dict(
                signal=float(sig.mean()),
                noise_over_signal=float(etil.std(1).mean() / sig.mean()),
                sup_eps_tilde_over_signal=float(etil.abs().max(1).values.mean() / sig.mean()),
                std_s_eps_bar_over_signal=float(ebar.std() / sig.mean()),
                spearman=float(spearman_rows(jm, jt).mean()),
                regret_over_signal=float(((best - pick) / sig.clamp_min(1e-9)).mean()),
                regret_near=float(((best - pick) / sig.clamp_min(1e-9))[near].mean()))
        results.append(row)
        per_all[name] = {k: v.detach().cpu().numpy() for k, v in per.items()}
        t5 = row[f"t{H}"]
        log(f"[{name}] heads={heads}/{total}  H={H}: needle dyn-err {t5['mm_total']:.2f} mm "
            f"(common {t5['mm_common_rms']:.2f} / spread {t5['mm_spread_rms']:.2f}); "
            f"J noise/signal {row['J_base']['noise_over_signal']:.3f}  "
            f"spearman {row['J_base']['spearman']:.3f}  regret {row['J_base']['regret_over_signal']:.3f}")
        del agent, model, planner
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(dict(data=args.data, B=B, H=H, K=K, n_base=nb, results=results), f, indent=1)
    flat = {f"{n}/{k}": v for n, dd in per_all.items() for k, v in dd.items()}
    np.savez(args.out.replace(".json", "_perstate.npz"), episode=d["episode"],
             near=near.cpu().numpy(), success=d["success"], **flat)
    log(f"[saved] {args.out}")
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect_out", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--domain", default="target", choices=["target", "source"])
    ap.add_argument("--behavior_ckpt", default=BEHAVIOR_CKPT)
    ap.add_argument("--behavior_stage1", default=BEHAVIOR_S1)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--n_base", type=int, default=24)
    ap.add_argument("--n_rand", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.collect_out:
        collect(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
