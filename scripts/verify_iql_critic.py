"""Verify steps ①+② of the proposed residual-simdist pipeline BEFORE committing
to the full run:

  ① collect N (default 30k) TARGET transitions by rolling the FROZEN adapted
     SimDist MPPI base (the same base the residual run uses), + load the 20
     TARGET demos.  (rolling the frozen base => the stored action IS the base
     action, so no base_action recompute is needed.)
  ② pretrain the residual agent's critic (rl/residual_rl_agent.EnsembleQ, the
     EXACT arch stage-3 uses: 10-head LayerNorm REDQ over z-scored (s,a)) with
     IQL (expectile V regression + TD Q), fully OFFLINE on ①.

Then DIAGNOSE critic quality (the go/no-go for the whole idea):
  A. value range   — V(s), Q(s,a) should sit in ~[0,1] (sparse 0/1 + end_on_success).
  B. task-value ranking — Spearman( V(s_t), -steps_to_success ) on the demos
     (the §18-style sanity gate: a good value function rises toward the goal).
  C. Q calibration — Q(s_t, a_t) vs the Monte-Carlo return gamma^steps_to_go
     on the demos (Pearson r + MAE).
  D. success/fail separation — mean V over SUCCESS vs FAIL base-rollout episodes.
  E. base-action advantage — mean [ Q(s, a_base) - V(s) ] on demo states.

Usage:
  python scripts/verify_iql_critic.py \
    --base_run exps/mimicgen/threading_residual_ibrlhp_h5base_seed1 \
    --n_transitions 30000 --iql_steps 50000 --expectile 0.7

The rollout data is cached to <save_dir>/rollouts.npz so re-runs skip collection.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import yaml
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from tqdm import tqdm

import common_utils
from common_utils import ibrl_utils as utils
import train_residual_simdist as T
from rl.residual_rl_agent import EnsembleQ
from env.robosuite_wrapper import STATE_KEYS


# --------------------------------------------------------------------------- V
class ValueMLP(nn.Module):
    """V(s) -> scalar. Same trunk shape as the critic (LayerNorm MLP)."""

    def __init__(self, obs_dim, hidden, n_layer, layer_norm):
        super().__init__()
        dims = [obs_dim] + [hidden] * n_layer
        layers: list[nn.Module] = []
        for i in range(n_layer):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s).squeeze(-1)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(); y = y - y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


# ------------------------------------------------------------------ step ① data
def collect_base_rollouts(ws, n_transitions, cache_path):
    """Roll the FROZEN MPPI base deterministically in the TARGET env until
    >= n_transitions collected (stop on a terminal). Returns a list of episodes,
    each a dict {states:[T+1,obs], actions:[T,act], rewards:[T], success:bool}."""
    if os.path.exists(cache_path):
        print(f"[collect] loading cached rollouts from {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        eps = list(d["episodes"])
        n = sum(len(e["actions"]) for e in eps)
        n_succ = sum(int(e["success"]) for e in eps)
        print(f"[collect] cached: {len(eps)} episodes, {n} transitions, "
              f"{n_succ} success ({n_succ/max(1,len(eps)):.2%})")
        return eps

    print(f"[collect] rolling frozen MPPI base for >= {n_transitions} transitions")
    episodes = []
    collected = 0
    pbar = tqdm(total=n_transitions, desc="collect (frozen base, TARGET)",
                dynamic_ncols=True, mininterval=1.0)
    with torch.no_grad(), utils.eval_mode(ws.agent):
        while collected < n_transitions:
            obs, _ = ws.train_env.reset()
            states = [obs["state"].cpu().numpy()]
            actions, rewards = [], []
            success = False
            while True:
                a = ws._base_actions(obs["state"].view(1, -1))[0].cpu()
                obs, reward, terminal, succ, _ = ws.train_env.step(a)
                states.append(obs["state"].cpu().numpy())
                actions.append(a.numpy()); rewards.append(float(reward))
                collected += 1; pbar.update(1)
                if terminal:
                    success = bool(succ)
                    break
            episodes.append(dict(
                states=np.stack(states).astype(np.float32),
                actions=np.stack(actions).astype(np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                success=success))
    pbar.close()
    n_succ = sum(int(e["success"]) for e in episodes)
    print(f"[collect] {len(episodes)} episodes, {collected} transitions, "
          f"{n_succ} success ({n_succ/max(1,len(episodes)):.2%})")
    np.savez(cache_path, episodes=np.array(episodes, dtype=object))
    print(f"[collect] cached -> {cache_path}")
    return episodes


def load_demos(demo_path, num_data, reward_scale=1.0):
    """Load the TARGET demos as episode dicts (truncated at first success)."""
    cfg_path = os.path.join(os.path.dirname(demo_path), "env_cfg.json")
    task_name = json.load(open(cfg_path, "r"))["env_name"]
    state_keys = STATE_KEYS[task_name]
    f = h5py.File(demo_path, "r")
    n_ep = len(list(f["data"].keys()))
    episodes = []
    for ep_id in range(n_ep):
        if num_data > 0 and ep_id >= num_data:
            break
        ep = f[f"data/demo_{ep_id}"]
        actions = np.clip(np.array(ep["actions"]).astype(np.float32), -1.0, 1.0)
        rewards = np.array(ep["rewards"]).astype(np.float32)
        if not (rewards == 1).any():
            continue
        states = np.concatenate(
            [np.array(ep["obs"][k]) for k in state_keys], axis=1).astype(np.float32)
        T_ = int(np.argmax(rewards == 1)) + 1
        # states[:T_] are s_0..s_{T_-1}; states[T_] would be next of last but the
        # demo ends at success — build an explicit terminal next state = last state.
        st = states[:T_ + 1] if states.shape[0] > T_ else np.concatenate(
            [states[:T_], states[T_ - 1:T_]], axis=0)
        episodes.append(dict(
            states=st.astype(np.float32),
            actions=actions[:T_],
            rewards=(rewards[:T_] * reward_scale).astype(np.float32),
            success=True))
    f.close()
    print(f"[demos] loaded {len(episodes)} demos from {demo_path}")
    return episodes


# ---------------------------------------------------------- flat transition set
def build_flat(episodes, gamma, device):
    """episodes -> flat 1-step tensors on device + per-transition steps_to_go
    (only meaningful for successful episodes)."""
    S, A, R, S2, D, STG = [], [], [], [], [], []
    for e in episodes:
        st, ac, rw = e["states"], e["actions"], e["rewards"]
        T_ = len(ac)
        for t in range(T_):
            S.append(st[t]); A.append(ac[t]); R.append(rw[t])
            S2.append(st[min(t + 1, len(st) - 1)])
            done = 1.0 if (t == T_ - 1 and e["success"]) else 0.0
            D.append(done)
            STG.append((T_ - 1 - t) if e["success"] else -1)
    to = lambda x, dt=torch.float32: torch.tensor(np.asarray(x), dtype=dt, device=device)
    return dict(
        s=to(S), a=to(A), r=to(R), s2=to(S2), done=to(D),
        stg=np.asarray(STG, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run", type=str,
                    default="exps/mimicgen/threading_residual_ibrlhp_h5base_seed1")
    ap.add_argument("--save_dir", type=str, default="")
    ap.add_argument("--n_transitions", type=int, default=30000)
    ap.add_argument("--iql_steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--iql_lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=5e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--demo_frac", type=float, default=0.5,
                    help="fraction of each IQL batch drawn from the 20 demos (RLPD-style)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--features", type=str, default="raw", choices=["raw", "latent"],
                    help="feature space for the pretrained critic/vf/actor. 'raw' "
                         "(default) = z-scored raw state, byte-identical to every "
                         "existing run. 'latent' = the FROZEN SimDist encoder's "
                         "z=encode(s); pair with train_residual_simdist's "
                         "--residual.latent_features 1 (the nets are latent-shaped, "
                         "so a latent ckpt will NOT load into a raw run and vice versa).")
    # optional: also IQL-warmup the RESIDUAL actor (AWR) -> iql_actor_critic.pt
    ap.add_argument("--train_actor", type=int, default=0)
    ap.add_argument("--replan_base_action", type=int, default=0,
                    help="1 if the cached rollouts were NOT collected by the frozen "
                         "base (e.g. the SimDist adapt buffer), so action != base_action")
    ap.add_argument("--actor_steps", type=int, default=40000)
    ap.add_argument("--awr_beta", type=float, default=3.0)
    ap.add_argument("--awr_clip", type=float, default=100.0)
    args = ap.parse_args()

    save_dir = args.save_dir or os.path.join(args.base_run, "iql_critic_verify")
    os.makedirs(save_dir, exist_ok=True)
    common_utils.set_all_seeds(args.seed)
    device = "cuda"

    # rebuild the residual-simdist config from the base run -> gives base ckpt,
    # target domain_shift, strict_success, hand_init sigma, residual/critic arch.
    cfg: T.MainConfig = pyrallis.load(  # type: ignore
        T.MainConfig, open(os.path.join(args.base_run, "cfg.yaml"), "r"))
    cfg.save_dir = save_dir
    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # Workspace(from_main=False) still reads it
    print(common_utils.wrap_ruler("building base + env (from_main=False)"))
    ws = T.Workspace(cfg, from_main=False)  # env + agent + frozen MPPI base

    # ----------------------------------------------------------------- step ①
    rollouts = collect_base_rollouts(
        ws, args.n_transitions, os.path.join(save_dir, "rollouts.npz"))
    demos = load_demos(cfg.offline_demo_path, cfg.offline_num_data, cfg.env_reward_scale)

    # z-score on the COMBINED state pool (what stage-3 would see)
    all_s = np.concatenate([e["states"][:-1] for e in rollouts + demos], axis=0)
    s_mean = torch.tensor(all_s.mean(0), dtype=torch.float32, device=device)
    s_std = torch.tensor(all_s.std(0), dtype=torch.float32, device=device).clamp_min(1e-6)
    print(f"[zscore] fit on {all_s.shape[0]} states (dim={all_s.shape[1]})")

    obs_dim = ws.train_env.observation_shape[0]
    action_dim = ws.train_env.action_dim
    rc = cfg.residual

    # ---- featurizer: the ONE thing that differs between the raw and latent arms.
    # It feeds sample()/AWR/diagnostics alike, so swapping it here swaps the whole
    # pretrain's feature space.
    if args.features == "latent":
        # the frozen SimDist base's world model doubles as the featurizer; take it
        # from the base the Workspace already attached (same object stage-3 uses).
        wm = ws.agent._base_policy[0].model
        assert not getattr(wm, "raw_state", False), (
            "--features latent is meaningless with a raw-state WorldModel "
            "(encoder_type='identity': encode() is just a z-score)")
        _enc = wm.encode
        def norm(s):
            with torch.no_grad():   # frozen encoder
                return _enc(s)
        with torch.no_grad():
            feat_dim = int(_enc(torch.zeros(1, obs_dim, device=device)).shape[-1])
        print(f"[features] latent: frozen SimDist encoder, dim={feat_dim} "
              f"(raw obs_dim={obs_dim}); NOT z-scored (SimNorm is already bounded)")
    else:
        norm = lambda s: (s - s_mean) / s_std
        feat_dim = obs_dim
        print(f"[features] raw: z-scored state, dim={feat_dim}")

    roll_flat = build_flat(rollouts, args.gamma, device)
    demo_flat = build_flat(demos, args.gamma, device)
    print(f"[data] rollout transitions={len(roll_flat['r'])}, "
          f"demo transitions={len(demo_flat['r'])}")

    critic = EnsembleQ(feat_dim, action_dim, rc.critic_hidden_dim, rc.critic_num_layer,
                       bool(rc.critic_layer_norm), rc.num_critics).to(device)
    import copy
    critic_target = copy.deepcopy(critic)
    for p in critic_target.parameters():
        p.requires_grad_(False)
    vf = ValueMLP(feat_dim, rc.critic_hidden_dim, rc.critic_num_layer,
                  bool(rc.critic_layer_norm)).to(device)
    q_opt = torch.optim.Adam(critic.parameters(), lr=args.iql_lr)
    v_opt = torch.optim.Adam(vf.parameters(), lr=args.iql_lr)

    def sample(flat, n):
        idx = torch.randint(0, len(flat["r"]), (n,), device=device)
        return (norm(flat["s"][idx]), flat["a"][idx], flat["r"][idx],
                norm(flat["s2"][idx]), flat["done"][idx])

    def sample_mixed(n):
        n_off = int(round(n * args.demo_frac))
        n_on = n - n_off
        parts = [sample(roll_flat, n_on)]
        if n_off > 0:
            parts.append(sample(demo_flat, n_off))
        return [torch.cat([p[i] for p in parts], 0) for i in range(5)]

    # ----------------------------------------------------------------- step ②
    print(common_utils.wrap_ruler(
        f"IQL critic pretrain ({args.iql_steps} steps, expectile={args.expectile})"))
    for step in tqdm(range(args.iql_steps), desc="IQL", mininterval=2.0):
        s, a, r, s2, done = sample_mixed(args.batch_size)
        # V: expectile regression toward min-subset target Q(s,a)
        with torch.no_grad():
            q_sa = critic_target(s, a).min(0).values  # [B]
        v = vf(s)
        diff = q_sa - v
        w = torch.where(diff > 0, args.expectile, 1.0 - args.expectile)
        v_loss = (w * diff**2).mean()
        v_opt.zero_grad(set_to_none=True); v_loss.backward(); v_opt.step()
        # Q: TD toward r + gamma*(1-done)*V(s'), clamp to [0,1] (sparse, unbiased)
        with torch.no_grad():
            y = (r + args.gamma * (1.0 - done) * vf(s2)).clamp(0.0, 1.0)
        q = critic(s, a)  # [K,B]
        q_loss = ((q - y.unsqueeze(0)) ** 2).mean()
        q_opt.zero_grad(set_to_none=True); q_loss.backward(); q_opt.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.mul_(1 - args.tau).add_(p, alpha=args.tau)
        if (step + 1) % max(1, args.iql_steps // 10) == 0:
            print(f"  step {step+1}: v_loss={v_loss.item():.4f} "
                  f"q_loss={q_loss.item():.5f} q_mean={q.mean().item():.3f} "
                  f"v_mean={v.mean().item():.3f}")

    torch.save({"critic": critic.state_dict(), "vf": vf.state_dict(),
                "s_mean": s_mean.cpu(), "s_std": s_std.cpu(),
                "features": args.features, "feat_dim": feat_dim},
               os.path.join(save_dir, "iql_critic.pt"))

    # -------------------- optional: AWR warmup of the RESIDUAL actor -----------
    # Train a ResidualActor (the SAME arch stage-3 uses) via advantage-weighted
    # regression toward the offline RESIDUAL (a_data - a_base): rollout data has
    # a_data==a_base => target 0; the 20 demos have a_data!=a_base => target the
    # advantage-weighted (demo - base) correction. So the actor starts pointing
    # at high-advantage demo-style corrections instead of zero.
    if args.train_actor:
        from rl.residual_rl_agent import ResidualActor
        ascale = rc.action_scale
        actor = ResidualActor(feat_dim, action_dim, rc.actor_hidden_dim,
                              rc.actor_num_layer, bool(rc.actor_layer_norm), ascale).to(device)
        a_opt = torch.optim.Adam(actor.parameters(), lr=args.iql_lr)
        # a_base per transition: rollout -> the executed (== base) action, UNLESS
        #                        the rollouts came from a non-frozen base (e.g. the
        #                        SimDist adapt buffer) -> re-plan with this base;
        #                        demo    -> the MPPI base action at that state.
        if args.replan_base_action:
            print("[abase] re-planning rollout base_action with the frozen base")
            chunk = 256
            roll_abase = torch.cat([
                ws._base_actions(roll_flat["s"][i:i + chunk])
                for i in tqdm(range(0, len(roll_flat["s"]), chunk), desc="replan")])
        else:
            roll_abase = roll_flat["a"]
        demo_abase = ws._base_actions(demo_flat["s"])  # [N, act] on cuda

        def sample_actor(n):
            n_off = int(round(n * args.demo_frac)); n_on = n - n_off
            io = torch.randint(0, len(roll_flat["r"]), (n_on,), device=device)
            S = [roll_flat["s"][io]]; A = [roll_flat["a"][io]]; AB = [roll_abase[io]]
            if n_off > 0:
                idf = torch.randint(0, len(demo_flat["r"]), (n_off,), device=device)
                S.append(demo_flat["s"][idf]); A.append(demo_flat["a"][idf]); AB.append(demo_abase[idf])
            return torch.cat(S), torch.cat(A), torch.cat(AB)

        print(common_utils.wrap_ruler(
            f"AWR residual-actor warmup ({args.actor_steps} steps, beta={args.awr_beta})"))
        for step in tqdm(range(args.actor_steps), desc="AWR actor", mininterval=2.0):
            s_raw, a, abase = sample_actor(args.batch_size)
            sn = norm(s_raw)
            with torch.no_grad():
                adv = critic_target(sn, a).min(0).values - vf(sn)
                wgt = torch.exp(args.awr_beta * adv).clamp(max=args.awr_clip)
                tgt = (a - abase).clamp(-ascale, ascale)
            pred = actor(sn, abase)
            a_loss = (wgt * ((pred - tgt) ** 2).sum(-1)).mean()
            a_opt.zero_grad(set_to_none=True); a_loss.backward(); a_opt.step()
            if (step + 1) % max(1, args.actor_steps // 10) == 0:
                print(f"  [AWR] step {step+1}: loss={a_loss.item():.5f} "
                      f"w_mean={wgt.mean().item():.2f} pred_abs={pred.abs().mean().item():.4f} "
                      f"tgt_abs={tgt.abs().mean().item():.4f}")
        torch.save({"critic": critic.state_dict(), "vf": vf.state_dict(),
                    "actor": actor.state_dict(),
                    "s_mean": s_mean.cpu(), "s_std": s_std.cpu(),
                    "features": args.features, "feat_dim": feat_dim},
                   os.path.join(save_dir, "iql_actor_critic.pt"))
        print(f"[save] iql_actor_critic.pt (critic + vf + AWR residual actor)")

    # -------------------------------------------------------------- diagnostics
    print(common_utils.wrap_ruler("CRITIC QUALITY DIAGNOSTICS"))
    critic.eval(); vf.eval()

    @torch.no_grad()
    def V(s_raw):  # s_raw np [N,obs]
        return vf(norm(torch.tensor(s_raw, dtype=torch.float32, device=device))).cpu().numpy()

    @torch.no_grad()
    def Q(s_raw, a_raw):
        sn = norm(torch.tensor(s_raw, dtype=torch.float32, device=device))
        an = torch.tensor(a_raw, dtype=torch.float32, device=device)
        return critic(sn, an).min(0).values.cpu().numpy()

    # gather demo transitions (s, a, steps_to_go, mc_return)
    ds, da, dstg = [], [], []
    for e in demos:
        T_ = len(e["actions"])
        for t in range(T_):
            ds.append(e["states"][t]); da.append(e["actions"][t]); dstg.append(T_ - 1 - t)
    ds = np.stack(ds); da = np.stack(da); dstg = np.array(dstg)
    mc_ret = args.gamma ** dstg
    v_demo = V(ds); q_demo = Q(ds, da)

    # A. value range
    print("\n[A] value range (should be ~[0,1]):")
    print(f"    V(demo):  min={v_demo.min():.3f} mean={v_demo.mean():.3f} max={v_demo.max():.3f}")
    print(f"    Q(demo):  min={q_demo.min():.3f} mean={q_demo.mean():.3f} max={q_demo.max():.3f}")

    # B. task-value ranking (the key gate)
    sp_global = spearman(v_demo, -dstg)
    per_demo = []
    i = 0
    for e in demos:
        T_ = len(e["actions"])
        vv = v_demo[i:i + T_]; gg = -dstg[i:i + T_]
        per_demo.append(spearman(vv, gg)); i += T_
    print("\n[B] task-value ranking  Spearman( V(s_t), -steps_to_success ):")
    print(f"    global = {sp_global:+.3f}   per-demo mean = {np.nanmean(per_demo):+.3f}")

    # C. Q calibration vs Monte-Carlo return
    r_cal = pearson(q_demo, mc_ret)
    mae = float(np.abs(q_demo - mc_ret).mean())
    print("\n[C] Q calibration  Q(s,a_demo) vs gamma^steps_to_go:")
    print(f"    pearson r = {r_cal:+.3f}   MAE = {mae:.3f}")

    # D. success/fail separation over base rollouts
    succ_v, fail_v = [], []
    for e in rollouts:
        v0 = V(e["states"][:1])[0]
        (succ_v if e["success"] else fail_v).append(v0)
    sv = np.mean(succ_v) if succ_v else float("nan")
    fv = np.mean(fail_v) if fail_v else float("nan")
    print("\n[D] success/fail separation  mean V(s_0) over base-rollout episodes:")
    print(f"    success (n={len(succ_v)}): {sv:.3f}   fail (n={len(fail_v)}): {fv:.3f}   "
          f"gap = {sv - fv:+.3f}")

    # E. base-action advantage on demo states
    a_base = ws._base_actions(torch.tensor(ds, dtype=torch.float32)).cpu().numpy()
    q_base = Q(ds, a_base)
    adv = q_base - v_demo
    print("\n[E] base-action advantage  mean[ Q(s,a_base) - V(s) ] on demo states:")
    print(f"    mean adv = {adv.mean():+.3f}  (Q_base mean={q_base.mean():.3f})")

    # verdict
    print(common_utils.wrap_ruler("VERDICT"))
    ok_rank = np.nanmean(per_demo) > 0.5
    ok_cal = r_cal > 0.6
    ok_range = (-0.1 <= v_demo.min()) and (v_demo.max() <= 1.15)
    ok_sep = (sv - fv) > 0.05 if (succ_v and fail_v) else True
    print(f"  [B] ranking  Spearman>0.5 : {'PASS' if ok_rank else 'FAIL'} "
          f"({np.nanmean(per_demo):+.3f})")
    print(f"  [C] calib    pearson>0.6  : {'PASS' if ok_cal else 'FAIL'} ({r_cal:+.3f})")
    print(f"  [A] range    ~[0,1]       : {'PASS' if ok_range else 'FAIL'}")
    print(f"  [D] succ>fail             : {'PASS' if ok_sep else 'n/a  '} (gap {sv-fv:+.3f})")
    overall = ok_rank and ok_cal and ok_range
    print(f"\n  ==> critic quality: {'GOOD (proceed to stage 3)' if overall else 'WEAK (investigate before stage 3)'}")
    sys.stdout.flush()
    os._exit(0)  # EGL teardown hangs


if __name__ == "__main__":
    main()
