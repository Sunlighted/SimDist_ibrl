"""Square: SOURCE-trained value + ORACLE dynamics, evaluated on a target domain.

The Square analogue of the `oracle` arm in scripts/eval_oracle_dyn_mppi.py
(CLAUDE.md §27/§31), with NO SimDist world model anywhere in the loop:

  value    = the SOURCE-domain IBRL expert's min-ensemble Q(s, pi_ibrl(s)),
             applied DIRECTLY to the real terminal state (no encoder, no
             distilled V head).
  proposal = the same expert, rolled closed-loop in the real sim for H steps,
             replicated with gaussian jitter into `num_base` seeds.
  dynamics = the REAL target simulator (candidates are replayed from the
             current sim_state), i.e. zero model error.

So everything learned is source-domain and the ONLY oracle ingredient is the
dynamics -- exactly the decomposition that gave Threading 99.3% (§31), which
established that a stale source value is not what caps a planner.

Arms:
  policy  -- the expert executed directly (IBRL hybrid max-Q). The bar.
  oracle  -- MPPI/CEM over real rollouts scored by the source value.

The nut z-rotation window comes from the ckpt's own cfg.yaml (eval_init_dist),
so source and target share an identical init distribution by construction.

  python scripts/eval_square_oracle_mppi.py \
      --weight exps/mimicgen/square_ibrl_rotw1_seed1/model0.pt \
      --shift target --episodes 100 --probe_workers 8
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


# Target-B (CLAUDE.md §32): the moderate Square target. Everything else stays
# nominal -- DomainShiftConfig defaults every unset scale to 1.0.
TARGET_B = dict(gripper_kp_scale=0.30, peg_size_scale=1.20, peg_friction_scale=2.0)

# (std_i, mean_i, range_i) of gamma^H*V per CEM call -- see the note at the
# append site. This is the tolerance budget for any edit to the value function.
SIGNAL = []


def restore(env, sim_state):
    """Rewind a probe env to an exact sim_state. qacc_warmstart=0 makes the
    replay deterministic (same cold-start convention as the Threading script)."""
    raw = env.env
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    env.time_step = 0
    if getattr(env, "_success_validator", None) is not None:
        env._success_validator.reset(raw, env._bindings)


def _probe_worker(conn, env_params):
    """Sim-only worker (device="cpu" => no model/obs GPU buffer). Replays
    candidate action sequences from a sim_state, returns the real states."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as _np
    import torch as _torch
    from env.robosuite_wrapper import PixelRobosuite as _PR

    env = _PR(**{**env_params, "device": "cpu"})
    env.reset()
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            sim_state, acts = msg                      # acts: [H, n, act]
            H, n, _ = acts.shape
            out = _np.zeros((H, n, env.observation_shape[0]), dtype=_np.float32)
            for i in range(n):
                restore(env, sim_state)
                for t in range(H):
                    obs, _, _, _, _ = env.step(_torch.from_numpy(acts[t, i]))
                    out[t, i] = obs["state"].numpy()
            conn.send(out)
    finally:
        conn.close()


class ProbePool:
    """Fan candidate rollouts across sim-only workers. They all start from the
    same sim_state and never interact, so results re-concatenate in candidate
    order and the CEM math is identical to the serial path."""

    def __init__(self, env_params, num_workers):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(num_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_probe_worker, args=(b, env_params), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def rollout(self, sim_state, actions):
        acts = actions.detach().cpu().numpy()
        n = acts.shape[1]
        chunks = np.array_split(np.arange(n), len(self.conns))
        live = [(c, idx) for c, idx in zip(self.conns, chunks) if len(idx)]
        for c, idx in live:
            c.send((sim_state, acts[:, idx]))
        parts = [c.recv() for c, _ in live]
        out = np.concatenate(parts, axis=1)
        return torch.from_numpy(out).to("cuda")

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)


def serial_rollout(env, sim_state, actions):
    H, n, _ = actions.shape
    acts = actions.detach().cpu().numpy()
    out = np.zeros((H, n, env.observation_shape[0]), dtype=np.float32)
    for i in range(n):
        restore(env, sim_state)
        for t in range(H):
            obs, _, _, _, _ = env.step(torch.from_numpy(acts[t, i]))
            out[t, i] = obs["state"].numpy()
    return torch.from_numpy(out).to("cuda")


def agent_action(agent, state):
    """One deterministic IBRL-hybrid action for a [1, obs] state."""
    return agent.act({"state": state}, eval_mode=True, stddev=0.0, cpu=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="exps/mimicgen/square_ibrl_rotw1_seed1/model0.pt")
    ap.add_argument("--shift", default="target", choices=["source", "target"])
    ap.add_argument("--arms", default="policy,oracle")
    ap.add_argument("--episodes", type=int, default=100)
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
    ap.add_argument("--probe_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simdist_ckpt", default="",
                    help="Score terminals with a SimDist world model's DISTILLED "
                         "value head V_hat(encode(s)) instead of the expert critic, "
                         "leaving the proposal and everything else alone. This is a "
                         "fidelity test: V_hat regresses onto the very V_src this arm "
                         "otherwise uses, so a drop means distillation lost the LOCAL "
                         "candidate-ranking structure (which no offline MSE reports).")
    ap.add_argument("--proposal_simdist_ckpt", default="",
                    help="Take the PROPOSAL from this SimDist checkpoint's policy "
                         "head instead of --weight's expert, leaving the env, the "
                         "value and the planner alone. --weight still defines the "
                         "environment (init_distribution, sigma, strict) and the "
                         "default terminal value, so pass the SOURCE expert there. "
                         "This is what makes the oracle arm comparable to the "
                         "SimDist adapt runs, whose proposal is the stage-2 policy "
                         "head rather than an expert.")
    ap.add_argument("--simdist_base_rollout", type=int, default=0,
                    help="With --proposal_simdist_ckpt: build the base candidate "
                         "SEQUENCES the way SimDist's own planner does -- roll the "
                         "policy closed-loop H steps through the LEARNED model "
                         "(rl/simdist_planner.py::_policy_rollout) -- instead of "
                         "holding one action for H steps. Only the SCORING rollout "
                         "then differs between this arm and a SimDist adapt run, "
                         "which is what isolates dynamics error. Default 0 keeps "
                         "the held-action seed bit-for-bit.")
    ap.add_argument("--base_real_rollout", type=int, default=0,
                    help="Build the base candidate SEQUENCES by rolling the "
                         "proposal closed-loop through the REAL simulator (a "
                         "probe env stepped in this process), instead of through "
                         "a learned world model (--simdist_base_rollout) or by "
                         "holding one action for H steps. Takes precedence over "
                         "--simdist_base_rollout. Two reasons to prefer it here: "
                         "(1) --simdist_base_rollout leaves a LEARNED component "
                         "inside an arm whose entire point is zero model error; "
                         "(2) a BC proposal has no world model, so it is the only "
                         "construction that a SimDist head and a BC head can "
                         "BOTH use -- which is what makes 'swap the proposal' a "
                         "single-variable change. Costs H sequential sim steps "
                         "per plan call plus one extra GL context.")
    ap.add_argument("--proposal_bc_ckpt", default="",
                    help="Take the PROPOSAL from a plain BC policy (train_bc.py) "
                         "instead of an expert or a SimDist head -- e.g. a policy "
                         "BC'd on the 20 TARGET demos, whose bare score is ~0.2. "
                         "This is the WEAK-proposal regime: the ceiling cells are "
                         "all within 0.06 of 1.0, where an additive null is "
                         "arithmetically forced (CLAUDE.md iron law #18), so a "
                         "proposal with real headroom is the only way to see "
                         "whether the planner is buying anything. NOTE: a BC "
                         "policy has no world model, so --simdist_base_rollout is "
                         "unavailable and the base seeds fall back to the "
                         "held-action construction -- pass --base_policy_std > 0 "
                         "or all `num_base` seeds are identical copies.")
    ap.add_argument("--value_weight", default="",
                    help="Take the terminal value from THIS agent instead of the "
                         "source expert (proposal still comes from --weight). "
                         "Pointing it at a target-trained expert is the go/no-go "
                         "probe for 'is the source value what caps the oracle arm'.")
    args = ap.parse_args()

    import train_rl
    from env.domain_shift import DomainShiftConfig
    from env.robosuite_wrapper import PixelRobosuite

    agent, _eval_env, env_params = train_rl.load_model(args.weight, "cuda")
    agent.train(False)
    for p in agent.parameters():
        p.requires_grad_(False)

    env_params = dict(env_params)
    if args.shift == "target":
        env_params["domain_shift"] = DomainShiftConfig(
            mode="fixed", mask_hole_offset_in_obs=False, **TARGET_B)
    else:
        env_params["domain_shift"] = DomainShiftConfig(mode="none")

    idc = env_params.get("init_distribution", None)
    log(f"[setup] weight={args.weight}")
    log(f"[setup] shift={args.shift} "
        f"{TARGET_B if args.shift == 'target' else '(nominal)'}")
    log(f"[setup] init_dist enabled={getattr(idc, 'enabled', False)} "
        f"nut_z_rot={tuple(round(float(v), 4) for v in getattr(idc, 'nut_z_rot', ()))} "
        f"= {tuple(round(np.degrees(float(v)), 1) for v in getattr(idc, 'nut_z_rot', ()))} deg")
    log(f"[setup] prior attached: {len(getattr(agent, 'bc_policies', []))} "
        f"| act_method={agent.cfg.act_method}")
    # Planner settings were NOT logged before, so the eval_sw_* sweep logs do not
    # record their own configuration -- and CLAUDE.md 40c/40d say these knobs move
    # the score by tens of points and transfer across neither tasks nor
    # oracle-vs-learned dynamics. Two numbers are only comparable if this line matches.
    log(f"[setup] planner N={args.num_samples} (base {args.num_base}) H={args.horizon} "
        f"iters={args.iterations} elites={args.num_elites} temp={args.temperature} "
        f"init_std={args.init_std} min_std={args.min_std} momentum={args.momentum} "
        f"base_policy_std={args.base_policy_std}"
        + ("   [base seeds = SimDist closed-loop policy rollout]"
           if args.simdist_base_rollout and args.proposal_simdist_ckpt
           else "   [base candidates are IDENTICAL copies]" if args.base_policy_std == 0 else ""))
    log(f"[setup] hand_init sigma={env_params.get('hand_init_perturb_joint_sigma')} "
        f"| strict={getattr(env_params.get('strict_success'), 'enabled', False)} "
        f"| ep_len={env_params.get('episode_length')}")

    main_env = PixelRobosuite(**env_params)
    ep_len = int(env_params.get("episode_length", 400))
    arms = [a for a in args.arms.split(",") if a]

    pool = None
    probe = None
    if "oracle" in arms:
        if args.probe_workers > 0:
            pool = ProbePool(env_params, args.probe_workers)
            log(f"[setup] ProbePool: {args.probe_workers} sim-only cpu workers")
        else:
            probe = PixelRobosuite(**{**env_params, "device": "cpu"})
            probe.reset()

    # The base-seed rollout is closed-loop, so it cannot go through the pool
    # (workers only replay a fixed action sequence); it needs its own env here.
    base_probe = None
    if args.base_real_rollout and "oracle" in arms:
        base_probe = probe if probe is not None else PixelRobosuite(
            **{**env_params, "device": "cpu"})
        base_probe.reset()
        log("[setup] BASE SEEDS = proposal rolled closed-loop in the REAL sim")

    A = main_env.action_dim
    H = args.horizon
    dev = "cuda"
    gamma_H = args.discount ** H

    # Terminal value. Defaults to the SOURCE expert (the whole point of the arm);
    # --value_weight swaps in another agent's critic while leaving the proposal
    # alone, which isolates "is the value the binding constraint".
    value_agent = agent
    wm = None
    if args.simdist_ckpt:
        import pyrallis as _pyr
        import train_simdist
        from rl.simdist_agent import SimDistAgent
        sdir = os.path.dirname(args.simdist_ckpt)
        import yaml as _yaml0                      # adapt runs write an AdaptConfig
        _r0 = _yaml0.safe_load(open(os.path.join(sdir, "cfg.yaml")))
        if isinstance(_r0, dict) and _r0.get("stage1_ckpt"):
            sdir = os.path.dirname(_r0["stage1_ckpt"])
            log(f"[setup] value cfg follows stage1_ckpt -> {sdir}")
        scfg = _pyr.load(train_simdist.MainConfig,  # type: ignore
                         open(os.path.join(sdir, "cfg.yaml")))
        sd_agent = SimDistAgent(True, main_env.observation_shape, main_env.prop_shape,
                                main_env.action_dim, scfg.rl_camera, scfg.simdist)
        sd_agent.load_state_dict(torch.load(args.simdist_ckpt))
        sd_agent.train(False)
        for p in sd_agent.parameters():
            p.requires_grad_(False)
        wm = sd_agent.to("cuda").model
        log(f"[setup] VALUE = distilled V_hat from {args.simdist_ckpt} "
            f"(proposal from {'--proposal_simdist_ckpt' if args.proposal_simdist_ckpt else '--proposal_bc_ckpt' if args.proposal_bc_ckpt else 'the expert'})")
    prop_wm = None
    if args.proposal_simdist_ckpt:
        import pyrallis as _pyr2
        import train_simdist as _ts2
        from rl.simdist_agent import SimDistAgent as _SDA2
        # A stage-3 adapt run writes an AdaptConfig, not the stage-1 MainConfig
        # that describes the architecture (CLAUDE.md 24). Follow its stage1_ckpt
        # back to the dir that does hold a MainConfig; a stage-1 dir is used as is.
        pdir = os.path.dirname(args.proposal_simdist_ckpt)
        import yaml as _yaml
        _raw = _yaml.safe_load(open(os.path.join(pdir, "cfg.yaml")))
        if isinstance(_raw, dict) and _raw.get("stage1_ckpt"):
            pdir = os.path.dirname(_raw["stage1_ckpt"])
            log(f"[setup] proposal cfg follows stage1_ckpt -> {pdir}")
        pcfg = _pyr2.load(_ts2.MainConfig,  # type: ignore
                          open(os.path.join(pdir, "cfg.yaml")))
        p_agent = _SDA2(True, main_env.observation_shape, main_env.prop_shape,
                        main_env.action_dim, pcfg.rl_camera, pcfg.simdist)
        p_agent.load_state_dict(torch.load(args.proposal_simdist_ckpt))
        p_agent.train(False)
        for p_ in p_agent.parameters():
            p_.requires_grad_(False)
        prop_wm = p_agent.to("cuda").model
        log(f"[setup] PROPOSAL = SimDist policy head from "
            f"{args.proposal_simdist_ckpt} (env + value still from --weight)")

    prop_bc = None
    if args.proposal_bc_ckpt:
        assert not args.proposal_simdist_ckpt, \
            "--proposal_bc_ckpt and --proposal_simdist_ckpt both set a proposal"
        import train_bc as _tbc
        # Reuse main_env for the obs/action shapes; building a second
        # PixelRobosuite would cost another GL context for nothing
        # (memory gotcha-gpu-density-mppi-runs).
        prop_bc = _tbc._load_model(args.proposal_bc_ckpt, main_env, "cuda")
        prop_bc.train(False)
        for p_ in prop_bc.parameters():
            p_.requires_grad_(False)
        if args.simdist_base_rollout:
            log("[setup] WARNING --simdist_base_rollout ignored: a BC proposal "
                "has no world model to roll through; base seeds are held-action.")
        if args.base_policy_std <= 0:
            log("[setup] WARNING base_policy_std=0 with a held-action base => all "
                f"{args.num_base} base candidates are IDENTICAL copies.")
        log(f"[setup] PROPOSAL = BC policy from {args.proposal_bc_ckpt} "
            f"(env + value still from --weight)")

    if args.value_weight:
        value_agent, _ve, _vp = train_rl.load_model(args.value_weight, "cuda")
        value_agent.train(False)
        for p in value_agent.parameters():
            p.requires_grad_(False)
        log(f"[setup] VALUE from {args.value_weight} "
            f"(proposal still from {args.weight})")

    def base_seq_real(state2d, num, H, sim_state):
        """[H, num, A] base seeds from a closed-loop rollout in the REAL sim.

        Same closed loop as SimDist's own _policy_rollout, with the learned
        model replaced by the simulator: query the proposal, step, re-query at
        the state the simulator actually produced. Every proposal can do this
        (a BC head has no world model to roll through), and it keeps the oracle
        arm free of learned dynamics everywhere, not just in the scoring."""
        restore(base_probe, sim_state)
        s = state2d
        acts = []
        for _ in range(H):
            a = proposal_action(s)                       # [1, A]
            acts.append(a)
            obs, _, _, _, _ = base_probe.step(a.view(-1).detach().cpu())
            s = obs["state"].to(dev).view(1, -1)
        return torch.stack(acts, dim=0).expand(H, num, A).clone()

    def base_seq(state2d, num, H):
        """[H, num, A] base candidate sequences.

        SimDist seeds CEM with a closed-loop policy rollout through its own model
        (decode the rolled latent, re-query the policy, step again); this script
        historically held a single action for H steps, which with base_policy_std=0
        makes all `num` base candidates identical. Matching the construction is
        what makes an oracle-dynamics number comparable to a SimDist adapt run."""
        if args.simdist_base_rollout and prop_wm is not None:
            z = prop_wm.encode(state2d).expand(num, -1).clone()
            shared = bool(getattr(prop_wm.cfg, "shared_encoder", 0))
            acts = []
            for _ in range(H):
                a = (prop_wm.policy_from_latent(z) if shared
                     else prop_wm.policy_action(prop_wm.decode(z)))
                acts.append(a)
                z = prop_wm.next(z, a)
            return torch.stack(acts, dim=0)
        return proposal_action(state2d).view(1, 1, -1).expand(H, num, -1).clone()

    def proposal_action(state2d):                # [B, obs] -> [B, A]
        """The action the planner seeds its base candidates with, and what the
        `policy` arm reports bare. Routed through one function so the two can
        never drift apart."""
        if prop_wm is not None:
            return prop_wm.policy_action(state2d)
        if prop_bc is not None:
            return prop_bc.forward({"state": state2d}).detach()
        return agent_action(agent, state2d)

    def value_fn(states):                       # [N, obs] -> [N]
        if wm is not None:
            return wm.predict_value(wm.encode(states), target=False).view(-1)
        return value_agent.value({"state": states}, use_ibrl=True).view(-1)

    def rollout_real(sim_state, actions):
        return (pool.rollout(sim_state, actions) if pool is not None
                else serial_rollout(probe, sim_state, actions))

    def plan(state, sim_state):
        """CEM over REAL rollouts, scored by the source-domain terminal value."""
        # base seeds: expert action at s0, held for H steps, then jittered. (A
        # closed-loop expert rollout would need a probe env stepped outside the
        # pool; the held action is the same open-loop seed SimDist's policy
        # rollout provides, and CEM refines it over `iterations`.)
        base = (base_seq_real(state.view(1, -1), args.num_base, H, sim_state)
                if base_probe is not None
                else base_seq(state.view(1, -1), args.num_base, H))
        if args.base_policy_std > 0:
            base = (base + torch.randn_like(base) * args.base_policy_std).clamp(-1, 1)

        mean = torch.zeros(H, A, device=dev)
        std = torch.full((H, A), args.init_std, device=dev)
        num_rand = args.num_samples - args.num_base
        for _ in range(args.iterations):
            noise = torch.randn(H, num_rand, A, device=dev)
            rand = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
            actions = torch.cat([rand, base], dim=1)
            states = rollout_real(sim_state, actions)          # [H, N, obs]
            returns = value_fn(states[-1]) * gamma_H
            # §33's ruler: what CEM actually ranks candidates by is the WITHIN-call
            # spread of gamma^H * V. On Threading that was 0.0068 = 1.4% of V's
            # range, which is why any refit whose residual was of the same order
            # rewrote the elite set and collapsed. Record it so the tolerance for
            # a value edit on THIS system is measured, not assumed.
            SIGNAL.append((float(returns.std()), float(returns.mean()),
                           float(returns.max() - returns.min())))
            elite_val, elite_idx = torch.topk(returns, args.num_elites)
            elite = actions[:, elite_idx]
            score = torch.exp(args.temperature * (elite_val - elite_val.max()))
            score = score / (score.sum() + 1e-9)
            new_mean = (score.view(1, -1, 1) * elite).sum(1)
            new_std = torch.sqrt(
                (score.view(1, -1, 1) * (elite - new_mean.unsqueeze(1)) ** 2).sum(1))
            mean = args.momentum * mean + (1 - args.momentum) * new_mean
            std = new_std.clamp(min=args.min_std)
        return mean[0]

    def yaw_deg(q):
        w, x, y, z = q
        return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))) % 360

    def phase(rec):
        """§32's failure taxonomy, same thresholds so the numbers are comparable:
        grasp   = never lifted the nut 3 cm;
        transport = lifted but never got within 5 cm of the peg;
        insertion = reached the peg but never seated."""
        if rec["max_lift"] < 0.03:
            return "grasp"
        if rec["min_pegdist"] > 0.05:
            return "transport"
        return "insertion"

    results, records = {}, {}
    for arm in arms:
        succ = 0
        rows = []
        for ep in range(args.episodes):
            np.random.seed(args.seed + ep)
            torch.manual_seed(args.seed + ep)
            obs, _ = main_env.reset()
            # hard_reset rebuilds the sim each reset -> re-fetch model/data + ids
            sim, mdl = main_env.env.sim, main_env.env.sim.model
            nut_bid = mdl.body_name2id("SquareNut_main")
            peg_bid = mdl.body_name2id("peg1")
            peg_xy = sim.data.body_xpos[peg_bid][:2].copy()
            nut0 = sim.data.body_xpos[nut_bid].copy()
            yaw0 = yaw_deg(sim.data.body_xquat[nut_bid].copy())
            z0 = nut0[2]
            max_lift, min_pegdist = 0.0, 1e9
            done = False
            for _t in range(ep_len):
                state = obs["state"].to(dev)
                if arm == "policy":
                    act = proposal_action(state.view(1, -1)).view(-1)
                else:
                    sim_state = sim.get_state().flatten()
                    act = plan(state, sim_state)
                obs, _r, terminal, success, _ = main_env.step(act.cpu())
                nut = sim.data.body_xpos[nut_bid]
                max_lift = max(max_lift, float(nut[2] - z0))
                min_pegdist = min(min_pegdist, float(np.linalg.norm(nut[:2] - peg_xy)))
                if success:
                    done = True
                    break
                if terminal:
                    break
            succ += int(done)
            rows.append(dict(ep=ep, succ=int(done), yaw=float(yaw0),
                             x=float(nut0[0]), y=float(nut0[1]),
                             max_lift=max_lift, min_pegdist=min_pegdist))
            if (ep + 1) % 10 == 0:
                log(f"  [{arm}] {ep+1}/{args.episodes}  running={succ/(ep+1):.3f}")
        records[arm] = rows
        fails = [r for r in rows if not r["succ"]]
        from collections import Counter
        fc = Counter(phase(r) for r in fails)
        log(f"  [{arm}] failure phases: {dict(fc)}  "
            f"(grasp=never lifted 3cm / transport=lifted but >5cm from peg / "
            f"insertion=reached peg, never seated)")
        for r in fails:
            log(f"      ep{r['ep']:<3d} yaw={r['yaw']:6.1f} y={r['y']:+.3f} "
                f"lift={r['max_lift']:.3f} pegdist={r['min_pegdist']:.3f} "
                f"-> {phase(r)}")
        rate = succ / args.episodes
        se = (rate * (1 - rate) / args.episodes) ** 0.5
        results[arm] = (rate, se, succ)
        log(f"[RESULT] {arm:7s} {args.shift:6s}  {succ}/{args.episodes} = "
            f"{rate:.3f} +- {se:.3f}")

    if SIGNAL:
        import numpy as _np
        a = _np.array(SIGNAL)
        log(f"\n[signal] CEM ranking signal over {len(a)} plan calls: "
            f"std_i(g^H V) mean={a[:,0].mean():.5f}  median={_np.median(a[:,0]):.5f}")
        log(f"[signal] V level mean={a[:,1].mean():.4f}  within-call range mean={a[:,2].mean():.5f}")
        log(f"[signal] std/level = {a[:,0].mean()/max(abs(a[:,1].mean()),1e-9):.4f}  "
            f"(Threading was 0.0068 abs / ~1.4% of V range, §33)")
    log("\n" + "=" * 60)
    for arm, (rate, se, succ) in results.items():
        log(f"{arm:8s} {args.shift:6s}  {rate:.3f} +- {se:.3f}  ({succ}/{args.episodes})")
    if pool is not None:
        pool.close()
    os._exit(0)


if __name__ == "__main__":
    main()
