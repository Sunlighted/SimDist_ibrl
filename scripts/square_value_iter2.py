"""On-policy value iteration v2 for MPPI-over-ORACLE-dynamics on Square target-B.

v1 (scripts/square_value_iter.py) collapsed 0.940 -> 0.000 after a single fit, in
both arms. The post-mortem gives three fixes, all implemented here:

  (1) TD(0) instead of Monte-Carlo returns. v1's fit residual was RMS 0.123
      against a CEM ranking signal of 0.031 -- a ratio of 4.0, where CLAUDE.md
      33's dose-response already collapses at 0.93. MC's per-state target
      variance is irreducible at this budget: at H=3 the signal is 0.031, so MC
      needs ~(0.5/0.031)^2 ~= 260 samples per state and 50 episodes give ~1.
      TD bootstraps that variance away. It is also better conditioned in the
      ANCHORED arm specifically: the target r + gamma*V(s') already contains
      V_src's global shape, so the residual only has to fix local errors.

  (2) A real trust region, chosen by a FREE pre-gate. v1 pinned lam=1.0, so
      nothing could reject a bad fit. The gate must NOT measure "distance from
      V_src": a per-state constant offset is invariant under CEM's topk+softmax
      (33's epsilon-bar), so only the WITHIN-CANDIDATE-SET spread of the
      perturbation can hurt. We therefore stash candidate-terminal batches
      during collection and compute, offline and for free,

          perturb(lam) = mean_sets std_i(gamma^H * lam * delta)
          signal       = mean_sets std_i(gamma^H * V_src)

      then deploy the largest lam whose ratio is under --gate_ratio. The free
      arm gets the same gate applied to EARLY STOPPING (checkpoints every
      --ckpt_every steps, take the last one still under the gate), so both arms
      decide "how far to move" by one identical criterion.

  (3) A revert guard. v1's collapse was self-reinforcing: a broken value scores
      0, so every new return is 0, so the fit target goes flat (frac>0 went
      0.761 -> 0.153 -> 0.085 -> 0.051) and it never recovers. Here a round that
      does not beat the incumbent is rolled back.

Arms (unchanged in intent from v1): --arm residual  = V_src + lam*delta, delta
zero-init so day-0 is EXACTLY V_src; --arm finetune = a free MLP distilled onto
V_src first. v1's round 0 already showed the asymmetry that motivates this:
distilling V_src to MSE 4.5e-5 over 1027 pi_src states still dropped the planner
0.94 -> 0.10, because faithfulness on pi_src's states says nothing about the
candidate-terminal cloud MPPI actually ranks -- and that cloud's distribution
depends on the planner, so there is no distribution to distil on a priori. The
anchor sidesteps that entirely.

  python scripts/square_value_iter2.py --arm residual --rounds 5 --episodes 50
"""
import os
import sys
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/Chenyang/ibrl")
sys.path.insert(0, "/home/Chenyang/ibrl/scripts")
os.environ.setdefault("MUJOCO_GL", "egl")

from eval_square_oracle_mppi import ProbePool, agent_action, TARGET_B  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


class Delta(nn.Module):
    """The trainable part. Identical architecture in both arms."""

    def __init__(self, obs_dim, hidden, num_layer, zero_init):
        super().__init__()
        layers, d = [], obs_dim
        for _ in range(num_layer):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        last = nn.Linear(d, 1)
        if zero_init:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s).view(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="exps/mimicgen/square_ibrl_rotw1_seed1/model0.pt")
    ap.add_argument("--arm", default="residual", choices=["residual", "finetune"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--gate_ratio", type=float, default=0.05,
                    help="max allowed (within-candidate-set perturbation)/(ranking "
                         "signal). 0.25 was measured too loose: it admitted lam=1.0 at "
                         "ratio 0.207 and the closed-loop score fell 0.900 -> 0.720.")
    ap.add_argument("--gate_shrink", type=float, default=0.5,
                    help="on a reverted round, multiply gate_ratio by this. The gate "
                         "threshold is the only free knob and the closed-loop score is "
                         "the only trustworthy signal, so let the score calibrate it "
                         "instead of guessing a fixed value.")
    ap.add_argument("--lams", default="0,0.01,0.02,0.05,0.1,0.15,0.25,0.5,1.0")
    ap.add_argument("--fit_steps", type=int, default=4000)
    ap.add_argument("--ckpt_every", type=int, default=250, help="finetune arm early-stop grid")
    ap.add_argument("--distill_steps", type=int, default=4000)
    ap.add_argument("--fit_lr", type=float, default=3e-4)
    ap.add_argument("--fit_batch", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.005, help="TD target EMA")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--num_layer", type=int, default=3)
    ap.add_argument("--revert_tol", type=float, default=0.02)
    ap.add_argument("--cand_keep", type=int, default=400,
                    help="candidate-terminal batches stashed per round for the gate")
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
    ap.add_argument("--probe_workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default="")
    ap.add_argument("--use_wb", type=int, default=1)
    args = ap.parse_args()

    import train_rl
    from env.domain_shift import DomainShiftConfig
    from env.robosuite_wrapper import PixelRobosuite

    save_dir = args.save_dir or f"exps/mimicgen/square_valiter2_{args.arm}_seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    lams = [float(x) for x in args.lams.split(",")]

    agent, _e, env_params = train_rl.load_model(args.weight, "cuda")
    agent.train(False)
    for p in agent.parameters():
        p.requires_grad_(False)

    env_params = dict(env_params)
    env_params["domain_shift"] = DomainShiftConfig(
        mode="fixed", mask_hole_offset_in_obs=False, **TARGET_B)
    idc = env_params.get("init_distribution", None)
    log(f"[setup] v2 arm={args.arm} weight={args.weight}")
    log(f"[setup] target-B {TARGET_B} | nut_z_rot="
        f"{tuple(round(np.degrees(float(v)),1) for v in getattr(idc,'nut_z_rot',()))} deg")
    log(f"[setup] sigma={env_params.get('hand_init_perturb_joint_sigma')} "
        f"gate_ratio={args.gate_ratio} lams={lams}")

    main_env = PixelRobosuite(**env_params)
    ep_len = int(env_params.get("episode_length", 400))
    obs_dim = main_env.observation_shape[0]
    A, H, dev = main_env.action_dim, args.horizon, "cuda"
    gamma_H = args.discount ** H
    pool = ProbePool(env_params, args.probe_workers)
    log(f"[setup] ProbePool: {args.probe_workers} cpu workers")

    def src_value(s):
        return agent.value({"state": s}, use_ibrl=True).view(-1)

    anchored = args.arm == "residual"
    delta = Delta(obs_dim, args.hidden, args.num_layer, zero_init=anchored).to(dev)
    delta_tgt = copy.deepcopy(delta)
    for p in delta_tgt.parameters():
        p.requires_grad_(False)

    def value_of(s, lam, net=None, use_tgt=False):
        net = net or (delta_tgt if use_tgt else delta)
        if anchored:
            with torch.no_grad():
                base = src_value(s)
            return base + lam * net(s)
        return net(s)

    wb = None
    if args.use_wb:
        import wandb
        wb = wandb.init(project="mimicgen", group="square_value_iter2",
                        name=f"valiter2_{args.arm}_seed{args.seed}",
                        config=vars(args), dir=save_dir)

    # ---- finetune arm: distil V_src so day-0 is as close as a free net gets ----
    if not anchored:
        st = []
        for ep in range(8):
            np.random.seed(90000 + ep)
            obs, _ = main_env.reset()
            for _t in range(ep_len):
                s = obs["state"].to(dev)
                st.append(s.cpu().numpy())
                obs, _r, term, succ, _ = main_env.step(
                    agent_action(agent, s.view(1, -1)).view(-1).cpu())
                if succ or term:
                    break
        S0 = torch.tensor(np.array(st), device=dev)
        Y0 = src_value(S0).detach()
        o = torch.optim.Adam(delta.parameters(), lr=args.fit_lr)
        for _ in range(args.distill_steps):
            i = torch.randint(0, len(S0), (args.fit_batch,), device=dev)
            l = ((delta(S0[i]) - Y0[i]) ** 2).mean()
            o.zero_grad(); l.backward(); o.step()
        delta_tgt.load_state_dict(delta.state_dict())
        log(f"[distill] day-0 MSE={((delta(S0)-Y0)**2).mean().item():.7f} over {len(S0)} states")

    def plan(state, sim_state, lam, cand_store, sig_acc):
        a0 = agent_action(agent, state.view(1, -1)).view(1, 1, A)
        base = a0.expand(H, args.num_base, A).clone()
        if args.base_policy_std > 0:
            base = (base + torch.randn_like(base) * args.base_policy_std).clamp(-1, 1)
        mean = torch.zeros(H, A, device=dev)
        std = torch.full((H, A), args.init_std, device=dev)
        nr = args.num_samples - args.num_base
        for _ in range(args.iterations):
            noise = torch.randn(H, nr, A, device=dev)
            rand = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
            actions = torch.cat([rand, base], dim=1)
            states = pool.rollout(sim_state, actions)
            with torch.no_grad():
                ret = value_of(states[-1], lam) * gamma_H
            sig_acc.append(float(ret.std()))
            # Stash a subsample of candidate terminals -- this is the ONLY
            # distribution on which the gate (and any honest offline check of a
            # value edit) is meaningful.
            if len(cand_store) < args.cand_keep and np.random.rand() < 0.05:
                cand_store.append(states[-1].detach().clone())
            ev, ei = torch.topk(ret, args.num_elites)
            el = actions[:, ei]
            sc = torch.exp(args.temperature * (ev - ev.max()))
            sc = sc / (sc.sum() + 1e-9)
            nm = (sc.view(1, -1, 1) * el).sum(1)
            ns = torch.sqrt((sc.view(1, -1, 1) * (el - nm.unsqueeze(1)) ** 2).sum(1))
            mean = args.momentum * mean + (1 - args.momentum) * nm
            std = ns.clamp(min=args.min_std)
        return mean[0]

    def run_round(rd, lam, cand_store):
        succ, sig = 0, []
        S, R, S2, D = [], [], [], []
        for ep in range(args.episodes):
            np.random.seed(args.seed + 10000 * rd + ep)
            torch.manual_seed(args.seed + 10000 * rd + ep)
            obs, _ = main_env.reset()
            sim = main_env.env.sim
            traj_s, traj_r = [], []
            done = False
            for _t in range(ep_len):
                s = obs["state"].to(dev)
                traj_s.append(s.cpu().numpy())
                act = plan(s, sim.get_state().flatten(), lam, cand_store, sig)
                obs, _r, term, success, _ = main_env.step(act.cpu())
                traj_r.append(float(success))
                if success:
                    done = True
                    break
                if term:
                    break
            succ += int(done)
            traj_s.append(obs["state"].cpu().numpy())      # s_{T+1} for bootstrapping
            ts = np.array(traj_s)
            n = len(traj_r)
            S.append(ts[:n]); S2.append(ts[1:n + 1])
            R.append(np.array(traj_r, dtype=np.float32))
            d = np.zeros(n, dtype=np.float32)
            if done:
                d[-1] = 1.0                                 # success terminates
            else:
                # timeout: truncation, not an absorbing state. Drop the final
                # transition rather than teach V that the horizon is terminal.
                S[-1] = S[-1][:-1]; S2[-1] = S2[-1][:-1]
                R[-1] = R[-1][:-1]; d = d[:-1]
            D.append(d)
            if (ep + 1) % 10 == 0:
                log(f"  [rd{rd} lam={lam}] {ep+1}/{args.episodes} running={succ/(ep+1):.3f}")
        return (succ / args.episodes, float(np.mean(sig)) if sig else 0.0,
                np.concatenate(S), np.concatenate(R), np.concatenate(S2), np.concatenate(D))

    def gate(net, cand_store, lam_list):
        """Free, and the only honest offline check: how big is the perturbation
        WITHIN a candidate set, relative to the ranking signal there?"""
        if not cand_store:
            return {l: float("inf") for l in lam_list}
        with torch.no_grad():
            sig, per = [], {l: [] for l in lam_list}
            for cs in cand_store:
                v_src = src_value(cs) * gamma_H
                sig.append(float(v_src.std()))
                d = net(cs) * gamma_H
                for l in lam_list:
                    per[l].append(float((l * d).std()) if anchored
                                  else float((d - v_src).std()))
            s = float(np.mean(sig)) + 1e-12
            return {l: float(np.mean(per[l])) / s for l in lam_list}

    # ---- round 0: the incumbent (lam=0 => exactly V_src for the anchored arm) --
    cands = []
    best_lam = 0.0
    score, sig, S, R, S2, D = run_round(0, best_lam, cands)
    log(f"[ROUND 0] score={score:.3f} (incumbent) CEM signal={sig:.5f} "
        f"| {len(S)} transitions, frac r>0={float((R>0).mean()):.3f}")
    if wb:
        wb.log({"round": 0, "score": score, "lam": best_lam,
                "cem_signal_std": sig, "gate_ratio_used": 0.0}, step=0)
    best_score = score
    gate_thresh = args.gate_ratio
    best_state = copy.deepcopy(delta.state_dict())
    pool_S, pool_R, pool_S2, pool_D = [S], [R], [S2], [D]

    for rd in range(1, args.rounds):
        S = torch.tensor(np.concatenate(pool_S), device=dev)
        R = torch.tensor(np.concatenate(pool_R), device=dev)
        S2 = torch.tensor(np.concatenate(pool_S2), device=dev)
        D = torch.tensor(np.concatenate(pool_D), device=dev)
        log(f"[fit rd{rd}] TD(0) on {len(S)} transitions, frac r>0={float((R>0).float().mean()):.3f}")

        opt = torch.optim.Adam(delta.parameters(), lr=args.fit_lr)
        ckpts = []
        for i in range(args.fit_steps):
            idx = torch.randint(0, len(S), (args.fit_batch,), device=dev)
            with torch.no_grad():
                y = R[idx] + args.discount * (1 - D[idx]) * value_of(S2[idx], 1.0, use_tgt=True)
            loss = ((value_of(S[idx], 1.0) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for p, pt in zip(delta.parameters(), delta_tgt.parameters()):
                    pt.mul_(1 - args.tau).add_(args.tau * p)
            if not anchored and (i + 1) % args.ckpt_every == 0:
                ckpts.append((i + 1, copy.deepcopy(delta.state_dict())))
        log(f"[fit rd{rd}] TD loss={loss.item():.6f}")

        # --- pick how far to move, by the gate ---
        if anchored:
            ratios = gate(delta, cands, lams)
            ok = [l for l in lams if ratios[l] <= gate_thresh]
            pick = max(ok) if ok else 0.0
            log(f"[gate thresh={gate_thresh:.4f}] "
                + "  ".join(f"{l}:{ratios[l]:.3f}" for l in lams)
                + f"  -> lam={pick}")
            chosen_ratio = ratios[pick]
        else:
            pick = 1.0
            chosen_ratio, best_i = float("inf"), None
            for (i, sd) in ckpts:
                tmp = Delta(obs_dim, args.hidden, args.num_layer, False).to(dev)
                tmp.load_state_dict(sd)
                r = gate(tmp, cands, [1.0])[1.0]
                if r <= gate_thresh:
                    best_i, chosen_ratio = (i, sd), r
            if best_i is None:
                log("[gate] every early-stop ckpt exceeds the gate -> keep incumbent")
                delta.load_state_dict(best_state)
                chosen_ratio = 0.0
            else:
                log(f"[gate] early stop at step {best_i[0]}, ratio={chosen_ratio:.3f}")
                delta.load_state_dict(best_i[1])

        cands = []
        score, sig, s_, r_, s2_, d_ = run_round(rd, pick, cands)
        log(f"[ROUND {rd}] score={score:.3f} (best {best_score:.3f}) lam={pick} "
            f"gate_ratio={chosen_ratio:.3f} CEM signal={sig:.5f}")
        if wb:
            wb.log({"round": rd, "score": score, "lam": pick,
                    "gate_ratio_used": chosen_ratio, "cem_signal_std": sig,
                    "td_loss": loss.item(), "best_score": max(best_score, score)}, step=rd)
        pool_S.append(s_); pool_R.append(r_); pool_S2.append(s2_); pool_D.append(d_)

        if score >= best_score - args.revert_tol:
            best_score = max(best_score, score)
            best_state = copy.deepcopy(delta.state_dict())
            torch.save(best_state, os.path.join(save_dir, "delta_best.pt"))
        else:
            gate_thresh *= args.gate_shrink
            log(f"[revert] {score:.3f} < {best_score:.3f}-{args.revert_tol} "
                f"-> roll back; gate thresh -> {gate_thresh:.4f}")
            delta.load_state_dict(best_state)
            delta_tgt.load_state_dict(best_state)

    log(f"\n[FINAL] arm={args.arm} best={best_score:.3f}")
    pool.close()
    if wb:
        wb.finish()
    os._exit(0)


if __name__ == "__main__":
    main()
