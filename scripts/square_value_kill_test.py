"""Can an on-policy value update rescue ANY of the planner's failures? (kill test)

The iteration runs (scripts/square_value_iter{,2}.py) could not answer this,
because at 50 episodes per round the SE is 0.042 while the effect of interest is
a few points: the rounds where the value was left essentially untouched
(lam=0.0 twice, lam=0.01 once) themselves spanned 0.880-0.940, so every accept /
revert decision was reacting to noise. Detecting +0.03 at p=0.9 head-on needs
~775 episodes per arm (~26 h). Worse, the revert-guard-plus-shrink loop is a
ONE-WAY RATCHET against a noisy score, so it drives lam -> 0 whether or not the
update helps -- it is not a neutral test.

This is the cheap, high-power version. Failures are only ~10% of episodes, so
restrict the comparison to them:

  phase 1  run the incumbent V_src on N seeds, record which seeds FAIL.
  phase 2  re-run ONLY those seeds (plus a matched success subsample as a
           side-effect check) under several lam, paired by seed.

On the failure subset the base rate is 0 by construction, so ANY rescue is
signal -- power goes up by one to two orders of magnitude versus measuring a
0.90 mean. The lam=0 arm in phase 2 is a determinism control: same env seed and
same torch seed, so it must reproduce phase 1 EXACTLY. If it does not, the whole
paired design is invalid and nothing else here can be read.

Read-out:
  rescued > 0 at some lam  -> the update carries real information; worth pursuing
                              with a proper powered comparison.
  rescued == 0 everywhere  -> dead: the fitted value cannot convert a single
                              failure, so the ceiling is not the value's to lift.

  python scripts/square_value_kill_test.py --episodes 200 --lams 0,0.15,0.5,1.0
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/Chenyang/ibrl")
sys.path.insert(0, "/home/Chenyang/ibrl/scripts")
os.environ.setdefault("MUJOCO_GL", "egl")

from eval_square_oracle_mppi import ProbePool, agent_action, TARGET_B  # noqa: E402
from square_value_iter2 import Delta  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="exps/mimicgen/square_ibrl_rotw1_seed1/model0.pt")
    ap.add_argument("--delta", default="exps/mimicgen/square_valiter2b_residual_seed0/delta_best.pt")
    ap.add_argument("--episodes", type=int, default=200, help="phase-1 screen")
    ap.add_argument("--lams", default="0,0.15,0.5,1.0")
    ap.add_argument("--n_success_check", type=int, default=20)
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
    ap.add_argument("--probe_workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import train_rl
    from env.domain_shift import DomainShiftConfig
    from env.robosuite_wrapper import PixelRobosuite

    lams = [float(x) for x in args.lams.split(",")]
    agent, _e, env_params = train_rl.load_model(args.weight, "cuda")
    agent.train(False)
    for p in agent.parameters():
        p.requires_grad_(False)
    env_params = dict(env_params)
    env_params["domain_shift"] = DomainShiftConfig(
        mode="fixed", mask_hole_offset_in_obs=False, **TARGET_B)

    main_env = PixelRobosuite(**env_params)
    ep_len = int(env_params.get("episode_length", 400))
    obs_dim = main_env.observation_shape[0]
    A, H, dev = main_env.action_dim, args.horizon, "cuda"
    gamma_H = args.discount ** H
    pool = ProbePool(env_params, args.probe_workers)

    delta = Delta(obs_dim, args.hidden, args.num_layer, zero_init=True).to(dev)
    delta.load_state_dict(torch.load(args.delta))
    delta.eval()
    for p in delta.parameters():
        p.requires_grad_(False)
    log(f"[setup] delta <- {args.delta} | lams={lams} | screen={args.episodes} eps")

    def value_of(s, lam):
        with torch.no_grad():
            return agent.value({"state": s}, use_ibrl=True).view(-1) + lam * delta(s)

    def plan(state, sim_state, lam):
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
            ret = value_of(states[-1], lam) * gamma_H
            ev, ei = torch.topk(ret, args.num_elites)
            el = actions[:, ei]
            sc = torch.exp(args.temperature * (ev - ev.max()))
            sc = sc / (sc.sum() + 1e-9)
            nm = (sc.view(1, -1, 1) * el).sum(1)
            ns = torch.sqrt((sc.view(1, -1, 1) * (el - nm.unsqueeze(1)) ** 2).sum(1))
            mean = args.momentum * mean + (1 - args.momentum) * nm
            std = ns.clamp(min=args.min_std)
        return mean[0]

    def episode(ep_seed, lam):
        # The pairing depends on this being a pure function of (ep_seed, lam):
        # env reset from np, candidate noise from torch, MuJoCo replay from an
        # exact sim_state. The lam=0 arm in phase 2 checks that empirically.
        np.random.seed(ep_seed)
        torch.manual_seed(ep_seed)
        obs, _ = main_env.reset()
        sim = main_env.env.sim
        for _t in range(ep_len):
            s = obs["state"].to(dev)
            act = plan(s, sim.get_state().flatten(), lam)
            obs, _r, term, success, _ = main_env.step(act.cpu())
            if success:
                return 1
            if term:
                break
        return 0

    # ---- phase 1: screen with the incumbent -------------------------------
    seeds = [args.seed + i for i in range(args.episodes)]
    res0 = {}
    for i, sd in enumerate(seeds):
        res0[sd] = episode(sd, 0.0)
        if (i + 1) % 25 == 0:
            log(f"  [screen] {i+1}/{len(seeds)} success={np.mean(list(res0.values())):.3f}")
    fails = [sd for sd in seeds if res0[sd] == 0]
    succs = [sd for sd in seeds if res0[sd] == 1][: args.n_success_check]
    base_rate = float(np.mean(list(res0.values())))
    log(f"\n[phase 1] V_src on {len(seeds)} seeds: success={base_rate:.3f}  "
        f"FAILURES={len(fails)}  (success subsample for the side-effect check: {len(succs)})")

    # ---- phase 2: re-run those seeds under each lam ------------------------
    log(f"\n{'lam':>6} {'rescued/fail':>14} {'kept/success':>14}  note")
    for lam in lams:
        r = [episode(sd, lam) for sd in fails]
        k = [episode(sd, lam) for sd in succs]
        rescued, kept = int(sum(r)), int(sum(k))
        note = ""
        if lam == 0.0:
            ok = (rescued == 0 and kept == len(succs))
            note = ("determinism control OK" if ok else
                    "!! CONTROL BROKEN -- the run is not a pure function of the "
                    "seed, so every paired number below is meaningless")
        log(f"{lam:>6} {rescued:>7}/{len(fails):<6} {kept:>7}/{len(succs):<6}  {note}")

    log("\n[read-out] rescued==0 at every lam => the fitted value cannot convert a "
        "single failure => this line is dead.")
    pool.close()
    os._exit(0)


if __name__ == "__main__":
    main()
