"""Offline IQL pretraining of a **QAgent-shaped** critic + actor (for the IBRL
stage-3 branches), from the SimDist stage-1 adapt rollouts + the TARGET demos.

Counterpart of `scripts/verify_iql_critic.py`, which pretrains the RESIDUAL
agent's `EnsembleQ` (z-scored, concat-(s,a) MLP) for `train_residual_simdist.py`.
IBRL's `QAgent` uses a DIFFERENT critic class (`rl.critic.MultiFcQ`) and actor
(`rl.actor.FcActor`), and consumes RAW (un-normalized) states -- so the residual
checkpoint cannot be loaded into it. This script trains the same IQL objective
into QAgent's own nets, in raw-state space, so `train_ibrl_simdist.py` can load
them directly.

Two actor modes:
  --actor_mode awr    AWR on the ABSOLUTE action: pi(s) -> a, advantage-weighted.
  --actor_mode clone  plain BC on the rollout actions (the MPPI-clone actor that
                      `actor_bc_pretrain_steps` produces, but done offline here).
  --actor_mode none   critic only.

Diagnostics (the go/no-go gate before stage 3):
  A. value range   -- V(s), Q(s,a) in ~[0,1] (sparse 0/1 reward + end_on_success)
  B. task-value ranking -- Spearman( V(s_t), -steps_to_success ) on the demos
  C. Q vs discounted return -- Pearson( Q(s,a), gamma^steps_to_success )
  D. success/fail separation -- mean V over SUCCESS vs FAIL rollout episodes

Usage:
  python scripts/pretrain_iql_qagent.py \
      --rollouts exps/mimicgen/threading_simdist_adapt_30k_seed1/rollouts.npz \
      --ibrl_cfg exps/mimicgen/threading_ibrl_permmargin_seed1/cfg.yaml \
      --save_dir exps/mimicgen/threading_simdist_adapt_30k_seed1/iql_qagent \
      --actor_mode awr
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common_utils
from rl.actor import FcActor, FcActorConfig
from rl.critic import MultiFcQ, MultiFcQConfig
from scripts.verify_iql_critic import (
    ValueMLP,
    build_flat,
    load_demos,
    pearson,
    spearman,
)


def load_rollouts(path):
    d = np.load(path, allow_pickle=True)
    eps = list(d["episodes"])
    n = sum(e["actions"].shape[0] for e in eps)
    n_succ = sum(int(e["success"]) for e in eps)
    print(f"[rollouts] {len(eps)} episodes, {n} transitions, "
          f"{n_succ} success ({n_succ / max(1, len(eps)):.2%}) <- {path}")
    return eps


def build_nets(ibrl_cfg_path, obs_dim, action_dim, device):
    """Rebuild QAgent's state critic/actor from an ibrl_simdist cfg.yaml."""
    cfg = yaml.safe_load(open(ibrl_cfg_path, "r"))["q_agent"]
    ccfg = MultiFcQConfig(**cfg["state_critic"])
    acfg = FcActorConfig(**cfg["state_actor"])
    print(f"[arch] state_critic: {ccfg}")
    print(f"[arch] state_actor : {acfg}")
    critic = MultiFcQ((obs_dim,), action_dim, ccfg).to(device)
    actor = FcActor((obs_dim,), action_dim, acfg).to(device)
    return critic, actor, ccfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=str, required=True)
    ap.add_argument("--demo_path", type=str,
                    default="release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    ap.add_argument("--demo_num_data", type=int, default=20)
    ap.add_argument("--ibrl_cfg", type=str,
                    default="exps/mimicgen/threading_ibrl_permmargin_seed1/cfg.yaml")
    ap.add_argument("--save_dir", type=str, required=True)
    # Reuse an already-trained critic+vf (skip the IQL phase) and only re-train the
    # actor. Guarantees the IBRL branches share a BIT-IDENTICAL critic, so their
    # comparison isolates the actor init.
    ap.add_argument("--load_critic_from", type=str, default="")
    ap.add_argument("--iql_steps", type=int, default=80000)
    ap.add_argument("--actor_steps", type=int, default=20000)
    ap.add_argument("--actor_mode", type=str, default="rwr",
                    choices=["rwr", "awr", "clone", "none"])
    ap.add_argument("--batch_size", type=int, default=256)
    # 0.9 (not IQL's usual 0.7): with expectile 0.7 on this near-on-policy data
    # V(s) ~= Q(s,a) for the data action, so adv ~= 0 and the AWR weights collapse
    # to 1 (advantage weighting degenerates into plain BC). A higher expectile
    # pushes V toward max_a Q, giving the weights real spread.
    ap.add_argument("--expectile", type=float, default=0.9)
    ap.add_argument("--iql_lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=5e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--demo_frac", type=float, default=0.5)
    ap.add_argument("--awr_beta", type=float, default=3.0)
    ap.add_argument("--awr_clip", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=1)
    # go/no-go thresholds
    ap.add_argument("--gate_spearman", type=float, default=0.95)
    ap.add_argument("--gate_pearson", type=float, default=0.90)
    args = ap.parse_args()

    common_utils.set_all_seeds(args.seed)
    device = "cuda"
    os.makedirs(args.save_dir, exist_ok=True)

    roll_eps = load_rollouts(args.rollouts)
    demo_eps = load_demos(args.demo_path, args.demo_num_data)
    roll = build_flat(roll_eps, args.gamma, device)
    demo = build_flat(demo_eps, args.gamma, device)
    obs_dim = roll["s"].shape[1]
    action_dim = roll["a"].shape[1]
    print(f"[data] rollout {len(roll['r'])} tr | demo {len(demo['r'])} tr | "
          f"obs {obs_dim} act {action_dim}")

    critic, actor, ccfg = build_nets(args.ibrl_cfg, obs_dim, action_dim, device)
    critic_target = copy.deepcopy(critic)
    for p in critic_target.parameters():
        p.requires_grad_(False)
    vf = ValueMLP(obs_dim, ccfg.hidden_dim, ccfg.num_layer, bool(ccfg.layer_norm)).to(device)
    q_opt = torch.optim.Adam(critic.parameters(), lr=args.iql_lr)
    v_opt = torch.optim.Adam(vf.parameters(), lr=args.iql_lr)

    def sample(n):
        """RLPD-style mix: demo_frac from the 20 demos, rest from the rollouts."""
        n_off = int(round(n * args.demo_frac))
        n_on = n - n_off
        io = torch.randint(0, len(roll["r"]), (n_on,), device=device)
        out = {k: [roll[k][io]] for k in ("s", "a", "r", "s2", "done")}
        if n_off > 0:
            idf = torch.randint(0, len(demo["r"]), (n_off,), device=device)
            for k in out:
                out[k].append(demo[k][idf])
        return {k: torch.cat(v) for k, v in out.items()}

    # ------------------------------------------------------------------- IQL
    # Q is trained on RAW states (QAgent's own input convention). Reward is
    # sparse 0/1 with end_on_success => true Q in [0,1] => clamp the TD target.
    if args.load_critic_from:
        prev = torch.load(args.load_critic_from)
        critic.load_state_dict(prev["critic"])
        critic_target.load_state_dict(prev["critic"])
        vf.load_state_dict(prev["vf"])
        print(f"[reuse] critic+vf loaded from {args.load_critic_from}; skipping IQL")
        args.iql_steps = 0
    print(common_utils.wrap_ruler(
        f"IQL critic+V ({args.iql_steps} steps, expectile={args.expectile}, raw state)"))
    for step in tqdm(range(args.iql_steps), desc="IQL", mininterval=2.0):
        b = sample(args.batch_size)
        # V: expectile regression toward min-ensemble Q_target(s,a)
        with torch.no_grad():
            q_sa = critic_target(b["s"], b["a"]).min(-1).values  # [B]
        v = vf(b["s"])
        diff = q_sa - v
        v_loss = (torch.abs(args.expectile - (diff < 0).float()) * diff.pow(2)).mean()
        v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        v_opt.step()

        # Q: TD toward r + gamma*(1-done)*V(s'), clamped to the reward range
        with torch.no_grad():
            target = b["r"] + args.gamma * (1.0 - b["done"]) * vf(b["s2"])
            target = target.clamp(0.0, 1.0)
        q = critic(b["s"], b["a"])                       # [B, num_q]
        q_loss = ((q - target.unsqueeze(-1)) ** 2).mean()
        q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        q_opt.step()

        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.data.mul_(1 - args.tau).add_(args.tau * p.data)

        if (step + 1) % max(1, args.iql_steps // 10) == 0:
            print(f"  [IQL] step {step+1}: v_loss={v_loss.item():.5f} "
                  f"q_loss={q_loss.item():.5f} q_mean={q.mean().item():.3f} "
                  f"v_mean={v.mean().item():.3f}")

    ckpt = {"critic": critic.state_dict(), "vf": vf.state_dict()}

    # ----------------------------------------------------------------- actor
    if args.actor_mode != "none":
        a_opt = torch.optim.Adam(actor.parameters(), lr=args.iql_lr)
        tag = {"rwr": "RWR (Q-weighted, absolute action)",
               "awr": "AWR (advantage-weighted, absolute action)",
               "clone": "clone (plain BC)"}[args.actor_mode]
        print(common_utils.wrap_ruler(f"actor pretrain: {tag}, {args.actor_steps} steps"))
        for step in tqdm(range(args.actor_steps), desc="actor", mininterval=2.0):
            b = sample(args.batch_size)
            if args.actor_mode == "awr":
                # DEGENERATE on single-behavior-policy data: each state has exactly
                # one action, so V(s) fits Q(s,a) exactly and adv ~= 0 (measured
                # adv_std=0.011 vs Q_std=0.258). Kept for ablation only.
                with torch.no_grad():
                    adv = critic_target(b["s"], b["a"]).min(-1).values - vf(b["s"])
                    wgt = torch.exp(args.awr_beta * adv).clamp(max=args.awr_clip)
            elif args.actor_mode == "rwr":
                # Weight by Q itself (batch-centred), which DOES separate success
                # (Q~0.73) from failure (Q~0.36) transitions.
                with torch.no_grad():
                    q_sa = critic_target(b["s"], b["a"]).min(-1).values
                    wgt = torch.exp(args.awr_beta * (q_sa - q_sa.mean())).clamp(max=args.awr_clip)
            else:
                wgt = torch.ones(len(b["r"]), device=device)
            mu = actor.net(b["s"])                       # absolute action
            a_loss = (wgt * ((mu - b["a"]) ** 2).sum(-1)).mean()
            a_opt.zero_grad(set_to_none=True)
            a_loss.backward()
            a_opt.step()
            if (step + 1) % max(1, args.actor_steps // 10) == 0:
                # w_std/w_p10/w_p90 are the real diagnostic: a w_mean of 1.0 with
                # w_std ~= 0 means the advantage weighting has degenerated into BC.
                wq = torch.quantile(wgt, torch.tensor([0.1, 0.9], device=device))
                print(f"  [actor] step {step+1}: loss={a_loss.item():.5f} "
                      f"w_mean={wgt.mean().item():.3f} w_std={wgt.std().item():.3f} "
                      f"w_p10={wq[0].item():.3f} w_p90={wq[1].item():.3f} "
                      f"mu_abs={mu.abs().mean().item():.3f} "
                      f"a_abs={b['a'].abs().mean().item():.3f}")
        ckpt["actor"] = actor.state_dict()
        ckpt["actor_mode"] = args.actor_mode

    out = os.path.join(args.save_dir, "iql_qagent.pt")
    torch.save(ckpt, out)
    print(f"[save] {out}  (keys: {list(ckpt.keys())})")

    # ----------------------------------------------------------- diagnostics
    print(common_utils.wrap_ruler("CRITIC QUALITY DIAGNOSTICS (go/no-go gate)"))
    critic.eval(); vf.eval()

    def _chunked(fn, *tensors, bs: int = 65536):
        """Evaluate fn over a big tensor in slices.

        The rollout set is the whole offline buffer -- 2.1M transitions on TPA --
        and a single 5-head critic forward over it asked for 40 GB and OOMed
        AFTER training had finished and the checkpoint was already saved. The
        diagnostics are a read-only gate, so the batch size is free to pick.
        """
        out = []
        with torch.no_grad():
            for i in range(0, len(tensors[0]), bs):
                out.append(fn(*[t[i:i + bs] for t in tensors]).cpu())
        return torch.cat(out).numpy()

    v_demo = _chunked(vf, demo["s"])
    q_demo = _chunked(lambda s, a: critic(s, a).min(-1).values, demo["s"], demo["a"])
    v_roll = _chunked(vf, roll["s"])
    q_roll = _chunked(lambda s, a: critic(s, a).min(-1).values, roll["s"], roll["a"])

    print(f"[A] range   V(demo) [{v_demo.min():.3f}, {v_demo.max():.3f}]  "
          f"Q(demo) [{q_demo.min():.3f}, {q_demo.max():.3f}]")
    print(f"[A] range   V(roll) [{v_roll.min():.3f}, {v_roll.max():.3f}]  "
          f"Q(roll) [{q_roll.min():.3f}, {q_roll.max():.3f}]")

    stg = demo["stg"]
    ok = stg >= 0
    sp_global = spearman(v_demo[ok], -stg[ok].astype(np.float64))
    # per-demo Spearman, averaged
    per = []
    off = 0
    for e in demo_eps:
        T_ = len(e["actions"])
        seg = slice(off, off + T_)
        vv, ss = v_demo[seg], stg[seg]
        if (ss >= 0).all() and T_ > 3:
            per.append(spearman(vv, -ss.astype(np.float64)))
        off += T_
    sp_per = float(np.mean(per)) if per else float("nan")
    print(f"[B] Spearman(V, -steps_to_success): global {sp_global:+.3f}  "
          f"per-demo mean {sp_per:+.3f}  (n={len(per)})")

    gpow = args.gamma ** stg[ok].astype(np.float64)
    pr = pearson(q_demo[ok], gpow)
    mae = float(np.abs(q_demo[ok] - gpow).mean())
    print(f"[C] Pearson(Q, gamma^steps) {pr:+.3f}   MAE {mae:.3f}")

    v0_s, v0_f = [], []
    off = 0
    for e in roll_eps:
        T_ = len(e["actions"])
        (v0_s if e["success"] else v0_f).append(v_roll[off])
        off += T_
    if v0_s and v0_f:
        print(f"[D] V(s0) success {np.mean(v0_s):.3f} (n={len(v0_s)})  "
              f"fail {np.mean(v0_f):.3f} (n={len(v0_f)})")

    passed = (sp_per >= args.gate_spearman) and (pr >= args.gate_pearson)
    print(common_utils.wrap_ruler(
        f"GATE: {'PASS' if passed else 'FAIL'}  "
        f"(per-demo Spearman {sp_per:+.3f} >= {args.gate_spearman}, "
        f"Pearson {pr:+.3f} >= {args.gate_pearson})"))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
