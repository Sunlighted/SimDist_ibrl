"""A1: give MPPI a PERFECT dynamics model and see if it helps.

Single variable. Everything about the planner is unchanged -- same CEM loop,
same candidate sampling, same policy-seeded base proposals, same learned reward
head r_hat and learned terminal value V_hat, same temperature/elites. The ONLY
change: instead of rolling candidates through the LEARNED latent dynamics, we
roll them through the REAL TARGET SIMULATOR (reset to the current sim_state,
step the candidate's H actions), and score with

    J(a) = sum_t gamma^t r_hat(encode(s_t^real)) + gamma^H V_hat(encode(s_H^real))

so z_t comes from the true next states rather than the model's guesses.

  oracle arm >> learned arm  =>  model error was the bottleneck (H1, optimizer's
                                 curse). Fix = pessimism / uncertainty penalty.
  oracle arm ~= learned arm  =>  a PERFECT model does not help: the objective
                                 (r_hat + V_hat over H steps) is the bottleneck (H2).

Budgets are shrunk (N=32, H=3, iters=2) so the oracle arm is affordable; the
learned arm runs at the SAME budget so the comparison is paired.
"""
import os
import sys
import atexit
import argparse

import numpy as np
import torch
import torch.multiprocessing as mp
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.domain_shift import DomainShiftConfig  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


def restore(env, sim_state):
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
    """A sim-only worker: holds ONE probe env on the CPU (device="cpu" => it does
    NOT allocate a model/obs GPU buffer), replays candidate action sequences from a
    given sim_state, and returns the real terminal states as numpy."""
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
            sim_state, acts = msg              # acts: numpy [H, n, act]
            H, n, _ = acts.shape
            # all H steps, not just the terminal: --terminal q needs s_{H-1}
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
    """Fan the candidates' TRUE rollouts out over N sim-only workers.

    The oracle arm's cost is entirely MuJoCo stepping (iters * num_samples * H sim
    steps per control step), which is CPU-bound and perfectly parallel across
    candidates -- they all start from the same sim_state and never interact. Workers
    run device="cpu" envs so they add no model GPU memory (only a small EGL context
    each). Results are re-concatenated in candidate order, so the CEM math is
    bit-identical to the serial path (MuJoCo is deterministic given sim_state +
    qacc_warmstart=0)."""

    def __init__(self, env_params, num_workers: int):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(num_workers):
            pc, cc = ctx.Pipe()
            proc = ctx.Process(target=_probe_worker, args=(cc, env_params), daemon=True)
            proc.start()
            cc.close()
            self.conns.append(pc)
            self.procs.append(proc)
        self.n = num_workers
        # Belt and braces: an exception anywhere between here and close() would
        # otherwise orphan the workers, and they are NOT findable by `pkill -f
        # <this script>` (spawn re-execs them as `multiprocessing.spawn`), so they
        # sit there at 100% CPU holding an EGL context until someone hunts the PIDs.
        atexit.register(self.close)

    @torch.no_grad()
    def rollout(self, sim_state, actions):
        """actions [H, N, act] (cuda) -> the candidates' REAL states [H, N, obs] (cuda).

        Identical output to the serial `real_rollout_states`: candidates are split
        into contiguous index blocks, so re-concatenating preserves candidate order
        and the CEM math is unchanged."""
        a = actions.detach().cpu().numpy()
        H, N, _ = a.shape
        chunks = np.array_split(np.arange(N), self.n)
        for conn, idx in zip(self.conns, chunks):
            conn.send((sim_state, a[:, idx]))
        outs = [conn.recv() for conn in self.conns]          # each [H, n_i, obs]
        return torch.tensor(np.concatenate(outs, axis=1), device="cuda")

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


@torch.no_grad()
def real_rollout_states(env, sim_state, actions):
    """actions [H, N, act] -> real next-states [H, N, obs] by stepping the TRUE sim."""
    H, N, _ = actions.shape
    out = torch.zeros(H, N, env.observation_shape[0], device="cuda")
    a_cpu = actions.cpu()
    for i in range(N):
        restore(env, sim_state)
        for t in range(H):
            obs, _, _, _, _ = env.step(a_cpu[t, i])
            out[t, i] = obs["state"]
    return out


@torch.no_grad()
def oracle_returns(model, planner, states, actions=None, s0=None):
    """states [H, N, obs] (REAL next-states). J = gamma^H * V(s_H).

    V is either the world model's distilled head (V_hat(encode(s_H))) or, when an
    external value_fn is attached (an expert QAgent's min-ensemble Q(s, pi(s))),
    that function applied DIRECTLY to the real s_H -- no encoder in the loop.

    Both arms run with use_reward=0: the learned reward head is ~0 everywhere on
    this sparse task (§17), so the planning return is the terminal value anyway.
    Keeping it off makes the two arms differ ONLY in where s_H came from."""
    q_fn = getattr(planner, "q_fn", None)
    if q_fn is not None:
        # TERMINAL Q: score the candidate by the expert critic's Q(s_{H-1}, a_{H-1}),
        # i.e. feed the REAL second-to-last state together with the candidate's OWN
        # last action -- no actor in the loop. Unlike V(s)=min_k Q_k(s, pi(s)), whose
        # action is in-distribution (the actor's own), this exposes the critic to the
        # candidates' OOD actions (22/32 are random) -- the classic offline-RL failure
        # mode. gamma^(H-1) because Q already carries the last step's reward + tail.
        H, N, _ = states.shape
        s_prev = states[-2] if H >= 2 else s0.view(1, -1).expand(N, -1)
        return q_fn(s_prev, actions[-1]) * (planner.cfg.discount ** (H - 1))
    if planner.value_fn is not None:
        return planner.value_fn(states[-1]) * planner.final_discount
    z_H = model.encode(states[-1])
    return model.predict_value(z_H, target=False) * planner.final_discount


@torch.no_grad()
def plan_randomJ(planner, model, state):
    """CEM identical to MppiPlanner.plan, but J is REPLACED BY NOISE. Isolates how
    much of MPPI's gain comes from ranking-by-value vs from averaging the noised
    policy-proposal cloud (the smoother hypothesis)."""
    cfg = planner.cfg
    H, A = cfg.horizon, planner.action_dim
    dev = planner.device
    z0 = model.encode(state.view(1, -1))
    num_base = planner._num_base()
    base_actions = planner._policy_rollout(z0, num_base) if num_base > 0 else None
    if base_actions is not None and cfg.base_policy_std > 0:
        base_actions = base_actions + torch.randn_like(base_actions) * cfg.base_policy_std
    mean = torch.zeros(H, A, device=dev)
    std = torch.full((H, A), cfg.init_std, device=dev)
    num_rand = cfg.num_samples - num_base
    for _ in range(cfg.iterations):
        noise = torch.randn(H, num_rand, A, device=dev)
        rand_actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
        actions = (torch.cat([rand_actions, base_actions.clamp(-1, 1)], dim=1)
                   if base_actions is not None else rand_actions)
        returns = torch.randn(actions.shape[1], device=dev)   # <-- J := noise
        elite_val, elite_idx = torch.topk(returns, cfg.num_elites)
        elite_acts = actions[:, elite_idx]
        score = torch.exp(cfg.temperature * (elite_val - elite_val.max()))
        score = score / (score.sum() + 1e-9)
        new_mean = (score.view(1, -1, 1) * elite_acts).sum(1)
        new_std = torch.sqrt((score.view(1, -1, 1) * (elite_acts - new_mean.unsqueeze(1)) ** 2).sum(1))
        mean = cfg.momentum * mean + (1 - cfg.momentum) * new_mean
        std = new_std.clamp(min=cfg.min_std)
    return mean[0]


@torch.no_grad()
def plan_oracle(planner, model, state, probe_env, sim_state, agent, pool=None,
                noise_fn=None):
    """CEM identical to MppiPlanner.plan, but returns are computed on REAL rollouts.

    noise_fn (optional): perturbs the REAL rollout states [H,N,obs] BEFORE they are
    scored, emulating a dynamics model whose estimate of s_t is off by epsilon. The
    injected noise is drawn from a SEPARATE generator (never the global stream), so
    the candidate actions + env resets are bit-identical across noise levels -- the
    only thing that differs at t=0 is the perturbation (paired dose-response)."""
    cfg = planner.cfg
    H, A = cfg.horizon, planner.action_dim
    dev = planner.device
    z0 = model.encode(state.view(1, -1))

    num_base = planner._num_base()
    base_actions = planner._policy_rollout(z0, num_base) if num_base > 0 else None
    if base_actions is not None and cfg.base_policy_std > 0:
        base_actions = base_actions + torch.randn_like(base_actions) * cfg.base_policy_std

    mean = torch.zeros(H, A, device=dev)
    std = torch.full((H, A), cfg.init_std, device=dev)
    num_rand = cfg.num_samples - num_base

    for _ in range(cfg.iterations):
        noise = torch.randn(H, num_rand, A, device=dev)
        rand_actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
        actions = (torch.cat([rand_actions, base_actions.clamp(-1, 1)], dim=1)
                   if base_actions is not None else rand_actions)
        real_states = (pool.rollout(sim_state, actions) if pool is not None
                       else real_rollout_states(probe_env, sim_state, actions))
        if noise_fn is not None:
            real_states = noise_fn(real_states)
        returns = oracle_returns(model, planner, real_states, actions=actions, s0=state)
        elite_val, elite_idx = torch.topk(returns, cfg.num_elites)
        elite_acts = actions[:, elite_idx]
        score = torch.exp(cfg.temperature * (elite_val - elite_val.max()))
        score = score / (score.sum() + 1e-9)
        new_mean = (score.view(1, -1, 1) * elite_acts).sum(1)
        new_std = torch.sqrt((score.view(1, -1, 1) * (elite_acts - new_mean.unsqueeze(1)) ** 2).sum(1))
        mean = cfg.momentum * mean + (1 - cfg.momentum) * new_mean
        std = new_std.clamp(min=cfg.min_std)
    return mean[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--num_base", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--num_elites", type=int, default=8)
    ap.add_argument("--arms", default="learned,oracle,policy")
    ap.add_argument("--value_agent", default="",
                    help="path to an expert QAgent ckpt; its V(s)=min-ensemble Q(s,pi(s)) "
                         "REPLACES the world model's distilled value head")
    ap.add_argument("--terminal", default="v", choices=["v", "q"],
                    help="v = score by V(s_H) (needs an actor); q = score by "
                         "Q(s_{H-1}, a_{H-1}) with the candidate's OWN last action "
                         "(no actor; exposes the critic to OOD candidate actions). "
                         "q is ORACLE-arm only.")
    ap.add_argument("--value_ibrl", type=int, default=1,
                    help="1 = V uses the expert's IBRL HYBRID policy (max-Q over "
                         "{RL actor, prior}) — the policy it actually executes. "
                         "0 = bare RL actor (the old V_src definition).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe_workers", type=int, default=0,
                    help="parallelize the oracle arm's candidate rollouts over N sim-only "
                         "worker processes (device=cpu => no model GPU memory). 0 = serial. "
                         "Output is bit-identical (MuJoCo is deterministic from sim_state).")
    ap.add_argument("--q_reduce", default="min", choices=["min", "mean"],
                    help="how to reduce the expert critic's Q-ensemble. min = the "
                         "pessimistic min-ensemble the agent was TRAINED with; mean = the "
                         "ensemble mean. Pessimism helps TD learning but may destroy the "
                         "RANKING signal a planner needs on OOD candidates (§27).")
    ap.add_argument("--jam_friction", type=float, default=5.0,
                    help="--shift jam: ring sliding-friction scale (probe: 5.0 jams but stays "
                         "solvable at 33%; 8.0 drove the source expert to 0%)")
    ap.add_argument("--jam_ring", type=float, default=0.85,
                    help="--shift jam: ring hole scale")
    ap.add_argument("--shift", default="keep", choices=["keep", "source", "target", "jam"],
                    help="physics for BOTH main_env + probe rollouts. keep = the run's "
                         "eval shift (TARGET for an adapt run); source = mode=none "
                         "(wide 3mm tolerance window); target = the dynamics-only shift.")
    ap.add_argument("--oracle_noise_sigma", default="0",
                    help="comma-separated list of noise levels (in per-dim state-std "
                         "units) injected into the oracle arm's REAL s_t before scoring. "
                         "e.g. '0,0.25,0.5,1.0,2.0'. 0 = clean oracle.")
    ap.add_argument("--oracle_noise_mode", default="iid", choices=["iid", "smooth"],
                    help="iid = fresh Gaussian per candidate (pure in-set variation "
                         "eps_tilde; UPPER BOUND on damage, unlike a real model). "
                         "smooth = a FIXED random field of state (correlated across "
                         "nearby candidates => eps_bar-heavy, argmax-immune; closer to "
                         "a real learned model's error). Same per-dim magnitude, "
                         "different cross-candidate correlation (§27).")
    ap.add_argument("--smooth_freq", type=float, default=0.3,
                    help="smooth-field frequency scale (in 1/state-std). Smaller = "
                         "longer correlation length => more of the error is a constant "
                         "offset within the candidate cloud (eps_bar).")
    ap.add_argument("--hole_offset", default="0,0,0",
                    help="object-init bias A: shift the tripod (hole) freejoint qpos by "
                         "'dx,dy,dz' (m) every reset, UNMASKED in obs (so the value sees "
                         "the OOD state). D0's hole is FIXED, so any nonzero shift is pure "
                         "OOD. Applied to BOTH main + probe env. e.g. '0.01,0.01,0'.")
    ap.add_argument("--force_init_dist", type=int, default=0,
                    help="apply the InitDistConfig monkeypatch even at needle_scale=1.0, so a "
                         "wide-vs-narrow comparison uses ONE code path (the native robosuite "
                         "sampler and the monkeypatch are NOT equivalent: same nominal bounds "
                         "measured 0.933 vs 1.000)")
    ap.add_argument("--needle_scale", default="1.0",
                    help="WIDEN the needle placement box: half-widths x massed by this "
                         "factor around the D0 center (1.0=D0, 2.0=2x wider box). Tests "
                         "source-value robustness to a broader START distribution.")
    ap.add_argument("--tripod_range", default="0",
                    help="RANDOMIZE the (D0-fixed) hole in a +/-R (m) box around nominal. "
                         "0=fixed (D0). Tests robustness to a broader GOAL distribution.")
    ap.add_argument("--dump_per_ep", default="",
                    help="path to a .npz to dump per-episode (arm, success, needle "
                         "init xy relative to the placement-box center, normalized by "
                         "the CURRENT box half-width). Lets you bucket the results by "
                         "where in the init box the needle started: |u|,|v| <= 1/scale "
                         "is the original D0 core, beyond that is the widened annulus. "
                         "Empty => no dump (default; byte-identical to before).")
    ap.add_argument("--no_strict", type=int, default=0,
                    help="1 => disable strict-success gating on the eval env (raw "
                         "position-only success). Diagnostic: isolates whether a "
                         "hole-offset drop is a strict-gate artifact vs a real "
                         "value/proposal OOD failure.")
    ap.add_argument("--needle_shift", default="0,0",
                    help="object-init bias B: shift the needle placement BOX center by "
                         "'dx,dy' (m), keeping the D0 width + z_rot. Injected via "
                         "InitDistConfig on both envs. e.g. '0.03,0' slides the needle "
                         "spawn box +3cm in x.")
    args = ap.parse_args()

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0
    cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/oracle_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, args.ckpt)))
    agent = ws.agent.to("cuda")
    model, planner = agent.model, agent.planner

    # shrink the planner budget (both arms identical)
    p = planner.cfg
    p.num_samples, p.num_base_override = args.num_samples, args.num_base
    p.iterations, p.num_elites = args.iterations, args.num_elites
    p.horizon = args.horizon
    planner.discounts = (p.discount ** torch.arange(args.horizon, device="cuda")).view(args.horizon, 1)
    planner.final_discount = p.discount ** args.horizon
    p.use_reward = 0  # score by terminal value only (sparse r_hat ~ 0); same for both arms

    ep_nr = dict(ws.eval_env_params)
    if args.shift == "jam":
        # JAM domain: NOMINAL contact physics + an extreme ring rim so a MISALIGNED
        # insertion sticks instead of sliding in. Unlike the SS12 target this changes
        # no task geometry at all (probe: min needle-ring distance 0.122 nominal vs
        # 0.124 jam, deep-reach 86.7% in both) -- only the VALUE of the same geometry,
        # which is the one thing a frozen value cannot detect from observation.
        ep_nr["domain_shift"] = DomainShiftConfig(
            mode="fixed", ring_friction_scale=args.jam_friction,
            ring_size_scale=args.jam_ring, mask_hole_offset_in_obs=False)
        log(f"[setup] SHIFT=jam -> ring_friction x{args.jam_friction}, "
            f"ring_size x{args.jam_ring} (nominal contact physics otherwise)")
    elif args.shift == "source":
        ep_nr["domain_shift"] = DomainShiftConfig(mode="none")
        log("[setup] SHIFT=source -> mode=none (nominal physics, wide tolerance window)")
    elif args.shift == "target":
        log(f"[setup] SHIFT=target -> {ep_nr['domain_shift'].mode} (the run's eval shift)")
    else:
        log(f"[setup] SHIFT=keep -> {ep_nr['domain_shift'].mode}")

    # --- object-initialization bias (orthogonal to the physics shift) -------
    ho = tuple(float(x) for x in args.hole_offset.split(","))
    if any(abs(v) > 0 for v in ho):
        import dataclasses
        ds = ep_nr["domain_shift"]
        if ds.mode == "none":  # source: promote to a fixed shift carrying only the offset
            ds = DomainShiftConfig(mode="fixed", friction_scale=1.0, mass_scale=1.0,
                                   solref_scale=1.0, solimp_scale=1.0, gripper_kp_scale=1.0,
                                   ring_size_scale=1.0, ring_friction_scale=1.0)
        ep_nr["domain_shift"] = dataclasses.replace(
            ds, hole_offset=ho, mask_hole_offset_in_obs=False)
        log(f"[setup] HOLE OFFSET (init bias A) = {ho} m, UNMASKED "
            f"(||={ (ho[0]**2+ho[1]**2+ho[2]**2)**0.5*1000:.1f} mm; success radius 12 mm)")
    ns = tuple(float(x) for x in args.needle_shift.split(","))
    nsc = float(args.needle_scale)
    trg = float(args.tripod_range)
    # D0 nominal needle box: x center -0.125 hw 0.075, y center 0.2 hw 0.05.
    # --needle_scale widens the half-width around the center; --needle_shift slides it.
    # Hoisted out of the branch below so --dump_per_ep can always express the needle
    # init in box coordinates (with nsc=1 these are exactly the D0 bounds).
    ncx, ncy, nhx, nhy = -0.125 + ns[0], 0.2 + ns[1], 0.075 * nsc, 0.05 * nsc
    if any(abs(v) > 0 for v in ns) or nsc != 1.0 or trg > 0 or args.force_init_dist:
        from env.domain_shift import InitDistConfig
        PI = 3.141592653589793
        # D0 tripod is FIXED at (0, -0.15); --tripod_range randomizes the hole in a +/-R box.
        ep_nr["init_distribution"] = InitDistConfig(
            enabled=True,
            needle_x=(ncx - nhx, ncx + nhx), needle_y=(ncy - nhy, ncy + nhy),
            needle_z_rot=(-2.0 * PI / 3.0, -PI / 3.0),
            tripod_x=(-trg, trg), tripod_y=(-0.15 - trg, -0.15 + trg),
            tripod_z_rot=(PI / 2.0, PI / 2.0))
        log(f"[setup] INIT DIST: needle box x=({ncx-nhx:.3f},{ncx+nhx:.3f}) "
            f"y=({ncy-nhy:.3f},{ncy+nhy:.3f}) [scale {nsc}x, shift {ns}]; "
            f"tripod range +/-{trg*100:.0f}cm (D0 hole is fixed)")

    if args.no_strict:
        ep_nr["strict_success"] = None
        log("[setup] STRICT DISABLED (raw position-only success)")

    main_env = PixelRobosuite(**ep_nr)
    probe_env = PixelRobosuite(**ep_nr)
    # CRITICAL: the domain shift is applied inside reset() (robosuite's hard_reset
    # rebuilds the sim, so the mutations must be re-applied every reset). A freshly
    # constructed env is still NOMINAL/SOURCE physics, and `restore()` only writes
    # qpos/qvel -- it never touches the model. Without this reset the counterfactual
    # rollouts silently run in the WRONG domain.
    probe_env.reset()
    pool = (ProbePool(ep_nr, args.probe_workers) if args.probe_workers > 0 else None)
    if pool is not None:
        log(f"[setup] ProbePool: {args.probe_workers} sim-only workers (device=cpu)")

    if args.value_agent:
        import copy
        import train_rl
        import train_bc
        from rl.q_agent import QAgent
        vdir = os.path.dirname(args.value_agent)
        vsrc = pyrallis.load(train_rl.MainConfig, open(os.path.join(vdir, "cfg.yaml")))  # type: ignore
        vagent = QAgent(True, main_env.observation_shape, main_env.prop_shape,
                        main_env.action_dim, vsrc.rl_camera, vsrc.q_agent)
        vagent.load_state_dict(torch.load(args.value_agent))
        vagent.train(False)
        for prm in vagent.parameters():
            prm.requires_grad_(False)
        vagent = vagent.to("cuda")

        # These experts were TRAINED with act_method="ibrl", i.e. they execute the
        # max-Q pick over {RL actor, prior}. Scoring with the bare RL actor would
        # evaluate a policy they never run (and induce a different V ranking), so
        # re-attach whichever prior the expert actually used. Two flavours exist:
        #   bc_policy         -> a trained BC head            (e.g. snapshots_seed1_v2)
        #   sgft_source_agent -> a frozen RL agent as prior   (e.g. sig01_snapshots)
        prior_desc = "none (bare RL actor)"
        if args.value_ibrl:
            if vsrc.bc_policy:
                bc_policy, _, _ = train_bc.load_model(vsrc.bc_policy, "cuda")
                vagent.add_bc_policy(copy.deepcopy(bc_policy))
                prior_desc = f"BC head <- {vsrc.bc_policy}"
            elif vsrc.sgft_source_agent:
                prior_agent = train_rl.load_frozen_agent(
                    vsrc.sgft_source_agent, main_env, vsrc, "cuda")
                vagent.add_bc_policy(train_rl.FrozenAgentBCPolicy(prior_agent))
                prior_desc = f"frozen RL agent <- {vsrc.sgft_source_agent}"
            else:
                raise ValueError(
                    f"--value_ibrl 1 but {vdir}/cfg.yaml declares no prior "
                    f"(bc_policy and sgft_source_agent both empty)")
            vagent.cfg.act_method = "ibrl"
            vagent.prior_margin = 0.0
        else:
            vagent.cfg.act_method = "rl"

        def _reduce(q):                        # q: [N, num_q]
            return q.min(-1)[0] if args.q_reduce == "min" else q.mean(-1)

        if args.terminal == "q":
            # Q(s, a) straight from the expert critic -- the candidate's own action
            # goes in, so the actor (and the IBRL prior) never runs.
            planner.q_fn = lambda st, act: _reduce(vagent.critic.forward_k(st, act)).view(-1)
        else:
            # raw state [N, obs] -> V [N].  oracle arm feeds the REAL s_H (no encoder,
            # no decoder in the loop); the learned arm reaches it via decode(z).
            # q_reduce=mean bypasses QAgent.value() (which hard-codes the min-ensemble).
            if args.q_reduce == "mean":
                def _v_mean(st):
                    a_rl = vagent._act_default(obs={"state": st}, eval_mode=True,
                                               stddev=0.0, clip=None, use_target=False)
                    q = vagent.critic.forward_k(st, a_rl).mean(-1)
                    if args.value_ibrl:
                        a_bc = vagent.bc_policies[0].act({"state": st}, cpu=False)
                        q = torch.maximum(q, vagent.critic.forward_k(st, a_bc).mean(-1))
                    return q.view(-1)
                planner.value_fn = _v_mean
            else:
                planner.value_fn = lambda st: vagent.value(
                    {"state": st}, use_ibrl=bool(args.value_ibrl)).view(-1)
        log(f"[setup] EXPERT value = {args.value_agent} "
            f"(train shift mode={vsrc.train_domain_shift.mode}) | "
            f"terminal={'Q(s_H-1, a_H-1) [candidate action, no actor]' if args.terminal == 'q' else ('V(s_H) via IBRL hybrid max-Q' if args.value_ibrl else 'V(s_H) via bare RL actor')} | "
            f"prior = {prior_desc}")
    ep_len = int(ws.eval_env_params.get("episode_length", 400))
    log(f"[setup] {args.run_dir}/{args.ckpt} | N={p.num_samples} (base {args.num_base}) "
        f"H={p.horizon} iters={p.iterations} elites={p.num_elites} use_reward={p.use_reward}")
    log(f"[setup] {args.episodes} episodes/arm, strict, hand_init sigma=0.1")

    # ---- noise-injection setup (oracle arm sigma sweep) ----
    obs_dim = main_env.observation_shape[0]
    sigmas = [float(s) for s in str(args.oracle_noise_sigma).split(",") if s != ""]
    # per-dim state std from a short policy rollout -> puts sigma in std units.
    _sb = []
    _o, _ = main_env.reset()
    for _ in range(400):
        with torch.no_grad():
            _a = model.policy_action(_o["state"].view(1, -1)).view(-1).clamp(-1, 1)
        _o, _, _term, _, _ = main_env.step(_a.detach().cpu())
        _sb.append(_o["state"])
        if _term:
            _o, _ = main_env.reset()
    state_batch = torch.stack(_sb).to("cuda")            # [M, obs]
    state_std = state_batch.std(0).clamp_min(1e-6)       # [obs]
    needle_std_m = float(state_std[9:12].mean().item())  # needle body pos = state[9:12] (§17)

    # smooth field: a FIXED random function g(s), normalized to per-dim ~unit std on
    # the state distribution. Nearby candidates -> nearly identical perturbation
    # (correlated => eps_bar-heavy, argmax-immune); mimics a real model's smooth error.
    field_gen = torch.Generator(device="cuda"); field_gen.manual_seed(20260719)
    K = 256
    Wf = torch.randn(obs_dim, K, generator=field_gen, device="cuda") * args.smooth_freq
    bf = torch.rand(K, generator=field_gen, device="cuda") * (2 * np.pi)
    Rf = torch.randn(K, obs_dim, generator=field_gen, device="cuda")
    with torch.no_grad():
        _feat = torch.cos((state_batch / state_std) @ Wf + bf)   # [M, K]
        _raw = _feat @ Rf                                        # [M, obs]
        Rf = Rf / _raw.std(0, keepdim=True).clamp_min(1e-6)      # per-dim ~unit std

    def _smooth_field(states):        # [..., obs] -> per-dim ~unit-std perturbation
        feat = torch.cos((states / state_std) @ Wf + bf)
        return feat @ Rf

    def make_noise_fn(sigma, mode, gen):
        if sigma <= 0:
            return None
        scale = sigma * state_std     # [obs] per-dim magnitude (SAME for both modes)
        if mode == "iid":
            def fn(real_states):      # [H, N, obs], fresh Gaussian per candidate
                eps = torch.randn(real_states.shape, generator=gen, device=real_states.device)
                return real_states + eps * scale
            return fn
        def fn(real_states):          # smooth: deterministic field of state
            return real_states + _smooth_field(real_states) * scale
        return fn

    log(f"[setup] state_std collected (needle-pos std {needle_std_m*1000:.1f} mm); "
        f"oracle noise mode={args.oracle_noise_mode} smooth_freq={args.smooth_freq}")
    for s in sigmas:
        log(f"[setup]   sigma={s:g}  ->  ~{s*needle_std_m*1000:.2f} mm needle-pos noise")

    # job list: the oracle arm expands over the sigma sweep
    jobs = []
    for arm in args.arms.split(","):
        if arm == "oracle":
            for s in sigmas:
                jobs.append(("oracle" if s == 0 else f"oracle_s{s:g}", "oracle", s))
        else:
            jobs.append((arm, arm, 0.0))

    results = {}
    per_ep = []   # (label, ep, success, u, v) when --dump_per_ep is set
    for label, kind, sigma in jobs:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # injection noise uses a SEPARATE generator (fixed seed across sigma) so the
        # global stream -- candidate actions + env resets -- is identical at every
        # noise level; only the perturbation magnitude differs (nested dose-response).
        noise_gen = torch.Generator(device="cuda"); noise_gen.manual_seed(777)
        noise_fn = (make_noise_fn(sigma, args.oracle_noise_mode, noise_gen)
                    if kind == "oracle" else None)
        succ = []
        for ep in range(args.episodes):
            obs, _ = main_env.reset()
            u = v = float("nan")
            if args.dump_per_ep:
                # needle body pos is state[9:12] (world); express it in the placement
                # box's own frame so the D0 core / widened annulus split is exact.
                npos = obs["state"][9:12].detach().cpu().numpy().astype(float)
                toff = np.asarray(main_env.env.table_offset, dtype=float)
                u = (npos[0] - (toff[0] + ncx)) / nhx
                v = (npos[1] - (toff[1] + ncy)) / nhy
            done_succ = 0.0
            for t in range(ep_len):
                with torch.no_grad():
                    if kind == "policy":
                        a = model.policy_action(obs["state"].view(1, -1)).view(-1).clamp(-1, 1)
                    elif kind == "learned":
                        a = planner.plan(obs["state"], eval_mode=True)
                    elif kind == "random":
                        a = plan_randomJ(planner, model, obs["state"])
                    else:  # oracle dynamics (optionally noised)
                        sim_state = main_env.env.sim.get_state().flatten().copy()
                        a = plan_oracle(planner, model, obs["state"], probe_env, sim_state,
                                        agent, pool=pool, noise_fn=noise_fn)
                obs, reward, terminal, _, _ = main_env.step(a.detach().cpu())
                if reward > 0:
                    done_succ = 1.0
                if terminal:
                    break
            succ.append(done_succ)
            if args.dump_per_ep:
                per_ep.append((label, ep, done_succ, u, v))
            log(f"  [{label}] ep{ep}: {done_succ:.0f}  (running {np.mean(succ):.3f})")
        results[label] = float(np.mean(succ))
        log(f"[RESULT] {label:14s}: {results[label]:.3f}  (n={args.episodes})")

    if pool is not None:
        pool.close()
    log(f"\n===== ORACLE + NOISE dose-response  (shift={args.shift}, "
        f"mode={args.oracle_noise_mode}, n={args.episodes}) =====")
    for k, v in results.items():
        extra = ""
        if k.startswith("oracle_s"):
            s = float(k.split("_s")[1])
            extra = f"   (~{s*needle_std_m*1000:.2f} mm needle noise)"
        log(f"  {k:14s}: {v:.3f}{extra}")
    if "oracle" in results and "learned" in results:
        d = results["oracle"] - results["learned"]
        log(f"  clean oracle - learned = {d:+.3f}")

    if args.dump_per_ep and per_ep:
        labels = np.array([r[0] for r in per_ep])
        eps    = np.array([r[1] for r in per_ep], dtype=int)
        sc     = np.array([r[2] for r in per_ep], dtype=float)
        uu     = np.array([r[3] for r in per_ep], dtype=float)
        vv     = np.array([r[4] for r in per_ep], dtype=float)
        np.savez(args.dump_per_ep, label=labels, ep=eps, success=sc, u=uu, v=vv,
                 needle_scale=nsc, box_center=np.array([ncx, ncy]),
                 box_half=np.array([nhx, nhy]))
        log(f"\n[dump] per-episode -> {args.dump_per_ep}  (n={len(per_ep)})")
        # D0 core = the ORIGINAL box, i.e. |u|,|v| <= 1/needle_scale in the widened frame.
        r = np.maximum(np.abs(uu), np.abs(vv))
        thr = 1.0 / nsc if nsc > 0 else 1.0
        log(f"===== bucket by needle init (core = original D0 box, "
            f"max(|u|,|v|) <= {thr:.3f}) =====")
        for lab in dict.fromkeys(labels):
            m = labels == lab
            core, ann = m & (r <= thr), m & (r > thr)
            def _f(mask):
                if mask.sum() == 0:
                    return "   n/a      "
                p = sc[mask].mean()
                se = (p * (1 - p) / mask.sum()) ** 0.5
                return f"{p:.3f}+-{se:.3f} (n={int(mask.sum())})"
            log(f"  {lab:14s} core {_f(core)}   annulus {_f(ann)}")


if __name__ == "__main__":
    main()
