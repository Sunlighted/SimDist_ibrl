"""Coffee: SOURCE-trained value + ORACLE dynamics, evaluated on the target domain.

The Coffee_D0 analogue of scripts/eval_square_oracle_mppi.py (itself the
standalone version of the `oracle` arm in scripts/eval_oracle_dyn_mppi.py,
CLAUDE.md 27/31). NO SimDist world model anywhere in the loop:

  value    = the SOURCE-domain IBRL expert's min-ensemble Q(s, pi_ibrl(s)),
             applied DIRECTLY to the real terminal state (no encoder, no
             distilled V head).
  proposal = the same expert, rolled closed-loop in the real sim for H steps,
             replicated with gaussian jitter into `num_base` seeds.
  dynamics = the REAL target simulator (candidates are replayed from the
             current sim_state), i.e. ZERO model error.

So everything learned is source-domain and the ONLY oracle ingredient is the
dynamics. On Threading this decomposition gave 99.3% and established that a
stale source value is not what caps a planner (CLAUDE.md 31); this asks the
same question on a third task, whose target is a genuine TWO-stage gap.

Arms:
  policy  -- the expert executed directly (IBRL hybrid max-Q). The bar.
  oracle  -- MPPI/CEM over real rollouts scored by the source value.

  python scripts/eval_coffee_oracle_mppi.py \
      --weight exps/mimicgen/coffee_ibrl_source_sig01_seed1/model0.pt \
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


# Coffee target C75kp25 (CLAUDE.md 40b): chosen by measuring the DYNAMICS GAP
# directly, not by reading the expert's success rate. Everything else stays
# nominal -- DomainShiftConfig defaults every unset scale to 1.0.
TARGET_B = dict(gripper_kp_scale=0.25, lid_damping_scale=75.0, mass_scale=3.0,
                friction_scale=0.5, pod_holder_friction_scale=2.0)

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
    ap.add_argument("--weight", default="exps/mimicgen/coffee_ibrl_source_sig01_seed1/model0.pt")
    ap.add_argument("--shift", default="target", choices=["source", "target"])
    # Per-field overrides of TARGET_B, so alternative target recipes can be
    # probed without editing the file. -1 = keep the C75kp25 default.
    ap.add_argument("--target_kp", type=float, default=-1)
    ap.add_argument("--target_lid", type=float, default=-1)
    ap.add_argument("--target_mass", type=float, default=-1)
    ap.add_argument("--target_pfric", type=float, default=-1)
    ap.add_argument("--target_hfric", type=float, default=-1)
    ap.add_argument("--arms", default="policy,oracle",
                    help="policy = proposal executed bare; oracle = CEM over REAL "
                         "sim rollouts; learned = the SAME CEM rolled through the "
                         "world model from --simdist_ckpt and scored by its own "
                         "V_hat off the terminal LATENT. oracle - learned at a "
                         "matched budget is the dynamics term (tasks.md 41c).")
    ap.add_argument("--shift_from_cfg", type=int, default=0,
                    help="Take the target shift from the SimDist stage-1 cfg that "
                         "--simdist_ckpt / --proposal_simdist_ckpt descends from, "
                         "instead of this file's TARGET_B constant. Use it: "
                         "TARGET_B here is the C75kp25 recipe (kp 0.25 / lid 75 / "
                         "mass 3), NOT the G5x one that tasks.md 40e settled on "
                         "(kp 0.30 / lid 5 / mass 8 / friction 0.2 / pod_holder 5), "
                         "so the default silently evaluates a different target. The "
                         "stage-1 cfg is the target's only source of truth "
                         "(CLAUDE.md iron law 19b).")
    ap.add_argument("--proposal_simdist_ckpt", default="",
                    help="Take the PROPOSAL from this SimDist ckpt's policy head "
                         "instead of --weight's expert (env + value still from "
                         "--weight unless overridden).")
    ap.add_argument("--proposal_bc_ckpt", default="",
                    help="Take the PROPOSAL from a plain BC policy (train_bc.py), "
                         "e.g. one BC'd from scratch on the 20 target demos.")
    ap.add_argument("--decode_roundtrip", type=int, default=0,
                    help="Control for the learned arm's --value_weight cells: score "
                         "with the model's OWN V_hat but push the latent through "
                         "decode() and back through encode() first, so the gap to "
                         "the plain V_hat cell is the decode cost ALONE. On Square "
                         "that round trip cost -0.260, MORE than the value swap it "
                         "was being confused with.")
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
    tgt = dict(TARGET_B)
    for _cli, _fld in (("target_kp", "gripper_kp_scale"),
                       ("target_lid", "lid_damping_scale"),
                       ("target_mass", "mass_scale"),
                       ("target_pfric", "friction_scale"),
                       ("target_hfric", "pod_holder_friction_scale")):
        _v = getattr(args, _cli)
        if _v >= 0:
            tgt[_fld] = _v
    if args.shift_from_cfg:
        # Follow --proposal_simdist_ckpt / --simdist_ckpt back to the stage-1 dir
        # that holds the MainConfig, and take its eval_domain_shift verbatim.
        import yaml as _y
        _ck = args.proposal_simdist_ckpt or args.simdist_ckpt
        assert _ck, "--shift_from_cfg needs --simdist_ckpt or --proposal_simdist_ckpt"
        _d = os.path.dirname(_ck)
        _raw = _y.safe_load(open(os.path.join(_d, "cfg.yaml")))
        if isinstance(_raw, dict) and _raw.get("stage1_ckpt"):
            _d = os.path.dirname(_raw["stage1_ckpt"])
            _raw = _y.safe_load(open(os.path.join(_d, "cfg.yaml")))
        _ds = dict(_raw.get("eval_domain_shift") or {})
        _ds.pop("mode", None)
        _ds = {k: (tuple(v) if isinstance(v, list) else v) for k, v in _ds.items()}
        tgt = {k: v for k, v in _ds.items() if k != "mask_hole_offset_in_obs"}
        log(f"[setup] SHIFT FROM CFG {_d}: "
            + ", ".join(f"{k}={v}" for k, v in sorted(tgt.items())
                        if not str(k).startswith("dr_") and v not in (1.0, 0.0)))
    if args.shift == "target":
        env_params["domain_shift"] = DomainShiftConfig(
            mode="fixed", mask_hole_offset_in_obs=False, **tgt)
    else:
        env_params["domain_shift"] = DomainShiftConfig(mode="none")

    idc = env_params.get("init_distribution", None)
    log(f"[setup] weight={args.weight}")
    log(f"[setup] shift={args.shift} "
        f"{tgt if args.shift == 'target' else '(nominal)'}")
    log(f"[setup] hand_init_joint_sigma="
        f"{env_params.get('hand_init_perturb_joint_sigma', None)}")
    log(f"[setup] prior attached: {len(getattr(agent, 'bc_policies', []))} "
        f"| act_method={agent.cfg.act_method}")

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

    A = main_env.action_dim
    H = args.horizon
    dev = "cuda"
    gamma_H = args.discount ** H
    log(f"[setup] planner N={args.num_samples} (base {args.num_base}) H={H} "
        f"iters={args.iterations} elites={args.num_elites} temp={args.temperature} "
        f"init_std={args.init_std} min_std={args.min_std} momentum={args.momentum} "
        f"base_policy_std={args.base_policy_std}")

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
        # An adapt run writes an AdaptConfig, and parsing that as MainConfig dies
        # on `target_num_data: null` (memory gotcha-eval-adapt-ckpt-cfg-collision).
        # Follow stage1_ckpt back to the dir that holds the real MainConfig.
        import yaml as _yaml0
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
            f"(proposal: {'--proposal_simdist_ckpt' if args.proposal_simdist_ckpt else '--proposal_bc_ckpt' if args.proposal_bc_ckpt else 'the expert'})")
    if args.value_weight:
        value_agent, _ve, _vp = train_rl.load_model(args.value_weight, "cuda")
        value_agent.train(False)
        for p in value_agent.parameters():
            p.requires_grad_(False)
        log(f"[setup] VALUE from {args.value_weight} "
            f"(proposal: {'--proposal_simdist_ckpt' if args.proposal_simdist_ckpt else '--proposal_bc_ckpt' if args.proposal_bc_ckpt else args.weight})")

    prop_wm = None
    if args.proposal_simdist_ckpt:
        import pyrallis as _pyr2
        import train_simdist as _ts2
        from rl.simdist_agent import SimDistAgent as _SDA2
        pdir = os.path.dirname(args.proposal_simdist_ckpt)
        import yaml as _yaml2                   # an adapt run writes an AdaptConfig
        _r2 = _yaml2.safe_load(open(os.path.join(pdir, "cfg.yaml")))
        if isinstance(_r2, dict) and _r2.get("stage1_ckpt"):
            pdir = os.path.dirname(_r2["stage1_ckpt"])
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
        log(f"[setup] PROPOSAL = SimDist policy head from {args.proposal_simdist_ckpt}")

    prop_bc = None
    if args.proposal_bc_ckpt:
        assert not args.proposal_simdist_ckpt, "two proposals given"
        import train_bc as _tbc
        prop_bc = _tbc._load_model(args.proposal_bc_ckpt, main_env, "cuda")
        prop_bc.train(False)
        for p_ in prop_bc.parameters():
            p_.requires_grad_(False)
        log(f"[setup] PROPOSAL = BC policy from {args.proposal_bc_ckpt}")

    def proposal_action(state2d):               # [B, obs] -> [B, A]
        """The proposal, routed through one function so the `policy` arm and the
        planner's base seeds can never drift apart."""
        if prop_wm is not None:
            return prop_wm.policy_action(state2d)
        if prop_bc is not None:
            return prop_bc.forward({"state": state2d}).detach()
        return agent_action(agent, state2d)

    def value_fn(states):                       # [N, obs] -> [N]
        """Terminal value on RAW states. --value_weight wins over --simdist_ckpt:
        with both given the world model supplies dynamics only."""
        if args.value_weight:
            return value_agent.value({"state": states}, use_ibrl=True).view(-1)
        if wm is not None:
            return wm.predict_value(wm.encode(states), target=False).view(-1)
        return value_agent.value({"state": states}, use_ibrl=True).view(-1)

    def value_fn_latent(z):                     # [N, z] -> [N]
        """Terminal value on IMAGINED latents (the learned arm). With
        --value_weight the latent must be decoded, a path the planner never uses
        under shared_encoder=1, so --decode_roundtrip exists to price it."""
        if args.value_weight:
            return value_agent.value({"state": wm.decode(z)}, use_ibrl=True).view(-1)
        if args.decode_roundtrip:
            return wm.predict_value(wm.encode(wm.decode(z)), target=False).view(-1)
        return wm.predict_value(z, target=False).view(-1)

    def rollout_real(sim_state, actions):
        return (pool.rollout(sim_state, actions) if pool is not None
                else serial_rollout(probe, sim_state, actions))

    def plan(state, sim_state):
        """CEM over REAL rollouts, scored by the source-domain terminal value."""
        # base seeds: expert action at s0, held for H steps, then jittered. (A
        # closed-loop expert rollout would need a probe env stepped outside the
        # pool; the held action is the same open-loop seed SimDist's policy
        # rollout provides, and CEM refines it over `iterations`.)
        a0 = proposal_action(state.view(1, -1)).view(1, 1, A)
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
        """Coffee failure taxonomy. Unlike Threading/Square this task has TWO
        stages, so there is a 4th bucket for "pod is in, lid never closed":
        grasp     = never lifted the pod 3 cm;
        transport = lifted but never got within 5 cm of the holder;
        insertion = reached the holder but the pod never descended into the cup
                    (a seated pod sits ~5 mm BELOW the holder body origin; one
                    merely resting on the rim is ~+51 mm, so +10 mm separates
                    them cleanly);
        close-lid = pod seated, but the lid never reached the 15 deg criterion."""
        if rec["max_lift"] < 0.03:
            return "grasp"
        if rec["min_holderdist"] > 0.05:
            return "transport"
        if rec["min_pod_above_holder"] > 0.01:
            return "insertion"
        return "close-lid"

    if "learned" in arms:
        assert wm is not None, "--arms learned needs --simdist_ckpt (the world model)"
        if args.value_weight:
            log("[setup] WARNING learned arm + external critic => terminal latents "
                "go through decode(z_H), never used at plan time under "
                "shared_encoder=1. Price it with --decode_roundtrip before reading "
                "the Delta as a value effect (on Square the decode cost -0.260).")

    def plan_learned(state):
        """The same CEM as plan(), with the learned world model in place of the
        simulator. Candidate sampling, base seeds, elites, temperature and
        iterations are shared code, so oracle - learned isolates model error."""
        base = proposal_action(state.view(1, -1)).view(1, 1, A).expand(
            H, args.num_base, A).clone()
        if args.base_policy_std > 0:
            base = (base + torch.randn_like(base) * args.base_policy_std).clamp(-1, 1)
        mean = torch.zeros(H, A, device=dev)
        std = torch.full((H, A), args.init_std, device=dev)
        num_rand = args.num_samples - args.num_base
        z0 = wm.encode(state.view(1, -1))
        for _ in range(args.iterations):
            noise = torch.randn(H, num_rand, A, device=dev)
            rand = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
            actions = torch.cat([rand, base], dim=1)
            z = z0.expand(actions.shape[1], -1).clone()
            for t in range(H):
                z = wm.next(z, actions[t])
            returns = value_fn_latent(z) * gamma_H
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
            pod_bid = mdl.body_name2id("coffee_pod_main")
            hold_bid = mdl.body_name2id("coffee_machine_pod_holder_root")
            lid_adr = mdl.jnt_qposadr[
                mdl.joint_name2id("coffee_machine_lid_main_joint0")]
            hold_xyz = sim.data.body_xpos[hold_bid].copy()
            pod0 = sim.data.body_xpos[pod_bid].copy()
            yaw0 = yaw_deg(sim.data.body_xquat[pod_bid].copy())
            z0 = pod0[2]
            max_lift, min_holderdist = 0.0, 1e9
            min_pod_above_holder, min_lid_deg = 1e9, 1e9
            done = False
            for _t in range(ep_len):
                state = obs["state"].to(dev)
                if arm == "policy":
                    act = proposal_action(state.view(1, -1)).view(-1)
                elif arm == "learned":
                    act = plan_learned(state)
                else:
                    sim_state = sim.get_state().flatten()
                    act = plan(state, sim_state)
                obs, _r, terminal, success, _ = main_env.step(act.cpu())
                pod = sim.data.body_xpos[pod_bid]
                max_lift = max(max_lift, float(pod[2] - z0))
                min_holderdist = min(
                    min_holderdist, float(np.linalg.norm(pod[:2] - hold_xyz[:2])))
                min_pod_above_holder = min(
                    min_pod_above_holder, float(pod[2] - hold_xyz[2]))
                min_lid_deg = min(min_lid_deg,
                                  float(np.degrees(sim.data.qpos[lid_adr])))
                if success:
                    done = True
                    break
                if terminal:
                    break
            succ += int(done)
            rows.append(dict(ep=ep, succ=int(done), yaw=float(yaw0),
                             x=float(pod0[0]), y=float(pod0[1]),
                             max_lift=max_lift, min_holderdist=min_holderdist,
                             min_pod_above_holder=min_pod_above_holder,
                             min_lid_deg=min_lid_deg))
            if (ep + 1) % 10 == 0:
                log(f"  [{arm}] {ep+1}/{args.episodes}  running={succ/(ep+1):.3f}")
        records[arm] = rows
        fails = [r for r in rows if not r["succ"]]
        from collections import Counter
        fc = Counter(phase(r) for r in fails)
        log(f"  [{arm}] failure phases: {dict(fc)}  "
            f"(grasp=never lifted 3cm / transport=lifted but >5cm from holder / "
            f"insertion=reached holder, pod never went in / close-lid=pod in, "
            f"lid never closed)")
        for r in fails:
            log(f"      ep{r['ep']:<3d} x={r['x']:+.3f} y={r['y']:+.3f} "
                f"lift={r['max_lift']:.3f} hdist={r['min_holderdist']:.3f} "
                f"podz={r['min_pod_above_holder']:+.3f} "
                f"lid={r['min_lid_deg']:5.1f}deg -> {phase(r)}")
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
