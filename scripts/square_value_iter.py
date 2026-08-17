"""On-policy value iteration for MPPI-over-ORACLE-dynamics on Square target-B.

Why this exists (CLAUDE.md 33 + the 2x2 measured 2026-08-08):

    oracle dynamics, 50 games      pi_src (0.24)   pi_tgt (0.96)
      V_src                            0.900          0.420
      V_tgt                            0.420          1.000

Both mismatched cells collapse -- including V_src + pi_tgt, which drags a 96%
policy down to 42%. So what a planner needs is not "the most accurate value"
but a value CONSISTENT WITH ITS OWN CONTINUATION: V^pi answers "how much does
pi collect from here", and scoring pi_B's candidates with V_A systematically
over-rates the states A can rescue and B cannot -- which is exactly where
argmax then goes.

Every value arm in 33 regressed onto an EXTERNAL teacher (V_src, an expert Q,
another policy's MC returns), i.e. a function describing a different policy. So
the one thing never tried is fitting V to the PLANNER'S OWN returns. On
Threading that was infeasible: the CEM ranking signal was 0.0068, so sparse-0/1
MC noise (~0.5/sqrt(n)) needed ~5400 rollouts per state. Square measures
0.067-0.091 -- 10-13x stronger -- which is ~100x fewer samples, so it becomes
tractable here.

The loop (proposal is held FIXED at pi_src throughout -- moving the value and
the proposal together is the 33 "simultaneous update" cell, which scored 0.173):

    round k:  roll N episodes with (oracle dyn, V_k, pi_src)   <- this pass IS
                                                                  the closed-loop
                                                                  eval, no extra cost
              fit V_{k+1} on those rollouts' own MC returns
              repeat

Two arms, differing in EXACTLY one thing -- whether the fit is anchored:

  --arm residual : V(s) = V_src(s).detach() + lam * mlp(s), mlp last layer
                   zero-init  => day-0 output is bit-identical to V_src.
  --arm finetune : V(s) = mlp(s), where mlp is first DISTILLED to match V_src
                   (reported as day-0 fidelity) and then fine-tuned freely.

Same architecture, same data, same optimizer in both -- so the comparison is
"does the anchor do work", not "does a good initialisation do work".

  python scripts/square_value_iter.py --arm residual --rounds 4 --episodes 50
"""
import os
import sys
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/Chenyang/ibrl")
sys.path.insert(0, "/home/Chenyang/ibrl/scripts")
os.environ.setdefault("MUJOCO_GL", "egl")

from eval_square_oracle_mppi import (  # noqa: E402
    restore, ProbePool, serial_rollout, agent_action, TARGET_B,
)

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


class ValueHead(nn.Module):
    """The planner's terminal value, in one of the two arms.

    `src_fn` is the frozen source value (a closure, never trained). The MLP is
    identical in both arms; only how its output is combined differs."""

    def __init__(self, obs_dim, src_fn, arm, hidden=512, num_layer=3, lam=1.0):
        super().__init__()
        self.arm = arm
        self.lam = lam
        self._src = [src_fn]                      # list => not a submodule
        layers, d = [], obs_dim
        for _ in range(num_layer):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        last = nn.Linear(d, 1)
        if arm == "residual":
            # zero-init => delta is exactly 0 everywhere at day 0, so the planner
            # starts on the incumbent function rather than on a fresh guess.
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.mlp = nn.Sequential(*layers)

    def forward(self, s):
        out = self.mlp(s).view(-1)
        if self.arm == "residual":
            with torch.no_grad():
                base = self._src[0](s).view(-1)
            return base + self.lam * out
        return out


def mc_returns(rewards, discount):
    """Sparse 0/1 with end_on_success => return from t is discount^(steps to the
    terminal reward), or 0 for a failed episode. Computed by backward recursion
    so it stays correct if a reward ever lands mid-episode."""
    out = np.zeros(len(rewards), dtype=np.float32)
    run = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        run = rewards[t] + discount * run
        out[t] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="exps/mimicgen/square_ibrl_rotw1_seed1/model0.pt")
    ap.add_argument("--arm", default="residual", choices=["residual", "finetune"])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=50, help="per round; this pass is also the eval")
    ap.add_argument("--lam", type=float, default=1.0, help="residual arm only")
    ap.add_argument("--fit_steps", type=int, default=4000)
    ap.add_argument("--distill_steps", type=int, default=4000, help="finetune arm day-0 match")
    ap.add_argument("--fit_lr", type=float, default=3e-4)
    ap.add_argument("--fit_batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--num_layer", type=int, default=3)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--num_base", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--num_elites", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--init_std", type=float, default=0.15)
    ap.add_argument("--min_std", type=float, default=0.02)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--base_policy_std", type=float, default=0.1)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--probe_workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default="")
    ap.add_argument("--use_wb", type=int, default=1)
    args = ap.parse_args()

    import train_rl
    from env.domain_shift import DomainShiftConfig
    from env.robosuite_wrapper import PixelRobosuite

    save_dir = args.save_dir or f"exps/mimicgen/square_valiter_{args.arm}_seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    agent, _e, env_params = train_rl.load_model(args.weight, "cuda")
    agent.train(False)
    for p in agent.parameters():
        p.requires_grad_(False)

    env_params = dict(env_params)
    env_params["domain_shift"] = DomainShiftConfig(
        mode="fixed", mask_hole_offset_in_obs=False, **TARGET_B)
    idc = env_params.get("init_distribution", None)
    log(f"[setup] arm={args.arm} weight={args.weight}")
    log(f"[setup] target-B {TARGET_B} | nut_z_rot="
        f"{tuple(round(np.degrees(float(v)),1) for v in getattr(idc,'nut_z_rot',()))} deg "
        f"enabled={getattr(idc,'enabled',False)}")
    log(f"[setup] sigma train/eval = {env_params.get('hand_init_perturb_joint_sigma')}")

    main_env = PixelRobosuite(**env_params)
    ep_len = int(env_params.get("episode_length", 400))
    obs_dim = main_env.observation_shape[0]
    A = main_env.action_dim
    H = args.horizon
    dev = "cuda"
    gamma_H = args.discount ** H

    pool = ProbePool(env_params, args.probe_workers)
    log(f"[setup] ProbePool: {args.probe_workers} sim-only cpu workers")

    def src_value(states):
        return agent.value({"state": states}, use_ibrl=True).view(-1)

    vhead = ValueHead(obs_dim, src_value, args.arm, args.hidden,
                      args.num_layer, args.lam).to(dev)

    wb = None
    if args.use_wb:
        import wandb
        wb = wandb.init(project="mimicgen", group="square_value_iter",
                        name=f"valiter_{args.arm}_seed{args.seed}",
                        config=vars(args), dir=save_dir)

    # --- finetune arm: distil V_src into the free MLP so both arms start at the
    # same place. Needs states; use a short random-ish probe of the env.
    if args.arm == "finetune":
        log("[distill] collecting states to match V_src on ...")
        st = []
        for ep in range(6):
            np.random.seed(90000 + ep)
            obs, _ = main_env.reset()
            for _t in range(ep_len):
                s = obs["state"].to(dev)
                st.append(s.cpu().numpy())
                a = agent_action(agent, s.view(1, -1)).view(-1)
                obs, _r, term, succ, _ = main_env.step(a.cpu())
                if succ or term:
                    break
        S = torch.tensor(np.array(st), device=dev)
        with torch.no_grad():
            Y = src_value(S)
        opt = torch.optim.Adam(vhead.parameters(), lr=args.fit_lr)
        for i in range(args.distill_steps):
            idx = torch.randint(0, len(S), (args.fit_batch,), device=dev)
            loss = ((vhead(S[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            fid = ((vhead(S) - Y) ** 2).mean().item()
        log(f"[distill] day-0 fidelity to V_src: MSE={fid:.6f} over {len(S)} states "
            f"(V level ~{Y.mean().item():.3f})")
        if wb:
            wb.log({"distill/day0_mse": fid}, step=0)

    def plan(state, sim_state, signal_acc):
        a0 = agent_action(agent, state.view(1, -1)).view(1, 1, A)
        base = a0.expand(H, args.num_base, A).clone()
        if args.base_policy_std > 0:
            base = (base + torch.randn_like(base) * args.base_policy_std).clamp(-1, 1)
        mean = torch.zeros(H, A, device=dev)
        std = torch.full((H, A), args.init_std, device=dev)
        num_rand = args.num_samples - args.num_base
        for _ in range(args.iterations):
            noise = torch.randn(H, num_rand, A, device=dev)
            rand = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
            actions = torch.cat([rand, base], dim=1)
            states = pool.rollout(sim_state, actions)
            with torch.no_grad():
                returns = vhead(states[-1]) * gamma_H
            signal_acc.append(float(returns.std()))
            ev, ei = torch.topk(returns, args.num_elites)
            elite = actions[:, ei]
            sc = torch.exp(args.temperature * (ev - ev.max()))
            sc = sc / (sc.sum() + 1e-9)
            nm = (sc.view(1, -1, 1) * elite).sum(1)
            ns = torch.sqrt((sc.view(1, -1, 1) * (elite - nm.unsqueeze(1)) ** 2).sum(1))
            mean = args.momentum * mean + (1 - args.momentum) * nm
            std = ns.clamp(min=args.min_std)
        return mean[0]

    pool_S, pool_G = [], []          # accumulated states + MC returns
    for rd in range(args.rounds):
        succ, sig = 0, []
        rd_S, rd_G = [], []
        for ep in range(args.episodes):
            np.random.seed(args.seed + 10000 * rd + ep)
            torch.manual_seed(args.seed + 10000 * rd + ep)
            obs, _ = main_env.reset()
            sim = main_env.env.sim
            ep_s, ep_r = [], []
            done = False
            for _t in range(ep_len):
                s = obs["state"].to(dev)
                ep_s.append(s.cpu().numpy())
                act = plan(s, sim.get_state().flatten(), sig)
                obs, r, term, success, _ = main_env.step(act.cpu())
                ep_r.append(float(success))
                if success:
                    done = True
                    break
                if term:
                    break
            succ += int(done)
            rd_S.append(np.array(ep_s))
            rd_G.append(mc_returns(np.array(ep_r), args.discount))
            if (ep + 1) % 10 == 0:
                log(f"  [rd{rd}] {ep+1}/{args.episodes} running={succ/(ep+1):.3f}")

        score = succ / args.episodes
        se = (score * (1 - score) / args.episodes) ** 0.5
        sig_mean = float(np.mean(sig)) if sig else 0.0
        log(f"[ROUND {rd}] score={score:.3f} +- {se:.3f} ({succ}/{args.episodes})  "
            f"CEM signal std_i(g^H V)={sig_mean:.5f}")
        if wb:
            wb.log({"round": rd, "score": score, "score_se": se,
                    "cem_signal_std": sig_mean,
                    "n_states": int(sum(len(x) for x in rd_S))}, step=rd)

        pool_S += rd_S
        pool_G += rd_G
        S = torch.tensor(np.concatenate(pool_S), device=dev)
        G = torch.tensor(np.concatenate(pool_G), device=dev)
        log(f"[fit rd{rd}] {len(S)} states, return mean={G.mean():.3f} "
            f"frac>0={float((G>0).float().mean()):.3f}")

        # Fit on the planner's OWN returns. Same data / arch / optimiser in both
        # arms; the arm only changes whether the output is anchored to V_src.
        opt = torch.optim.Adam(vhead.parameters(), lr=args.fit_lr)
        for i in range(args.fit_steps):
            idx = torch.randint(0, len(S), (args.fit_batch,), device=dev)
            loss = ((vhead(S[idx]) - G[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            fit_mse = ((vhead(S) - G) ** 2).mean().item()
            drift = (vhead(S) - src_value(S)).abs().mean().item()
        log(f"[fit rd{rd}] train MSE={fit_mse:.5f}  mean|V - V_src|={drift:.4f}")
        if wb:
            wb.log({"fit/mse": fit_mse, "fit/drift_from_src": drift}, step=rd)
        torch.save(vhead.state_dict(), os.path.join(save_dir, f"vhead_rd{rd}.pt"))

    pool.close()
    if wb:
        wb.finish()
    os._exit(0)


if __name__ == "__main__":
    main()
