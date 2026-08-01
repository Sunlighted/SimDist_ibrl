"""Is distillation a LOW-PASS FILTER on the expert's value? (§27 hypothesis)

The claim: the distilled V_hat beats the native expert critic V_exp at PLANNING
(oracle-dyn 1.000 vs 0.675-0.925) not because it is more accurate -- it cannot be,
it was regressed ONTO the expert -- but because regression with finite capacity
SMOOTHS AWAY the high-frequency spikes that TD bootstrapping leaves in sparsely
visited regions. argmax eats exactly those spikes.

Three falsifiable predictions, measured on TWO distributions:
  A) demo states               -- in-distribution (the expert's own support)
  B) MPPI candidate terminals  -- what the planner ACTUALLY scores (真实 rollout,
                                  含 22/32 随机候选推出去的 OOD 状态)

  P1  Spearman(V_hat, V_exp) is HIGH on A       (same large-scale structure)
  P2  local roughness ||d V / d s|| is SMALLER for V_hat, and the gap WIDENS on B
      (the spikes live off-distribution)
  P3  within each candidate set, V_exp's max is more of an OUTLIER than V_hat's
      ((max - median) / IQR) -- i.e. V_exp offers argmax a sharper spike to bite.

If P2/P3 come out flat or reversed, the low-pass story is wrong.
"""
import os
import sys
import argparse

import numpy as np
import torch
import pyrallis
import h5py
from scipy.stats import spearmanr

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402

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


@torch.no_grad()
def real_terminals(env, sim_state, actions):
    """actions [H,N,act] -> the REAL terminal states s_H [N, obs] (true sim rollout)."""
    H, N, _ = actions.shape
    out = torch.zeros(N, env.observation_shape[0], device="cuda")
    a_cpu = actions.cpu()
    for i in range(N):
        restore(env, sim_state)
        for t in range(H):
            obs, _, _, _, _ = env.step(a_cpu[t, i])
        out[i] = obs["state"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--value_agent",
                    default="exps/mimicgen/threading_ibrl_snapshots_seed1_v2/model0.pt")
    ap.add_argument("--episodes", type=int, default=3, help="episodes of MPPI to harvest candidates from")
    ap.add_argument("--probe_every", type=int, default=10)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--num_base", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/vsmooth_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, "latest.pt")))
    agent = ws.agent.to("cuda")
    model, planner = agent.model, agent.planner
    p = planner.cfg
    p.num_samples, p.num_base_override, p.horizon = args.num_samples, args.num_base, args.horizon
    p.iterations, p.num_elites, p.use_reward = 2, 8, 0
    planner.discounts = (p.discount ** torch.arange(args.horizon, device="cuda")).view(args.horizon, 1)
    planner.final_discount = p.discount ** args.horizon

    main_env = PixelRobosuite(**dict(ws.eval_env_params))
    probe_env = PixelRobosuite(**dict(ws.eval_env_params))
    # CRITICAL: the domain shift is applied inside reset() (robosuite's hard_reset
    # rebuilds the sim, so the mutations must be re-applied every reset). A freshly
    # constructed env is still NOMINAL/SOURCE physics, and `restore()` only writes
    # qpos/qvel -- it never touches the model. Without this reset the counterfactual
    # rollouts silently run in the WRONG domain.
    probe_env.reset()

    # the native expert critic
    import train_rl
    from rl.q_agent import QAgent
    vdir = os.path.dirname(args.value_agent)
    vsrc = pyrallis.load(train_rl.MainConfig, open(os.path.join(vdir, "cfg.yaml")))  # type: ignore
    vagent = QAgent(True, main_env.observation_shape, main_env.prop_shape,
                    main_env.action_dim, vsrc.rl_camera, vsrc.q_agent).to("cuda")
    vagent.load_state_dict(torch.load(args.value_agent))
    vagent.train(False)

    # ---- the two value functions, DIFFERENTIABLE w.r.t. the raw state
    def v_hat(s):                      # distilled world-model head (reads the latent)
        return model.predict_value(model.encode(s), target=False).view(-1)

    def v_exp(s):                      # native expert critic: min_k Q_k(s, pi(s))
        a = vagent.actor({"state": s}, 0.0).mean
        return vagent.critic.forward_k(s, a).min(-1)[0].view(-1)

    def grad_norm(fn, S, chunk=256):
        gs = []
        for i in range(0, len(S), chunk):
            s = S[i:i + chunk].clone().requires_grad_(True)
            v = fn(s).sum()
            g = torch.autograd.grad(v, s)[0]
            gs.append(g.norm(dim=-1).detach())
        return torch.cat(gs)

    # ================= distribution A: demo states (in-distribution) =================
    A = []
    with h5py.File("release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5", "r") as f:
        for k in list(f["data"].keys()):
            A.append(f[f"data/{k}/obs/state"][:])
    A = torch.tensor(np.concatenate(A), dtype=torch.float32, device="cuda")

    # ============ distribution B: REAL terminal states of MPPI's candidates ==========
    # (this is exactly what the oracle arm scores: true s_H of all 32 candidates)
    B_sets = []   # list of [N, obs] — keep the SET structure for the spike test
    ep_len = int(ws.eval_env_params.get("episode_length", 400))
    for ep in range(args.episodes):
        obs, _ = main_env.reset()
        for t in range(ep_len):
            with torch.no_grad():
                if t % args.probe_every == 0:
                    state = obs["state"]
                    z0 = model.encode(state.view(1, -1))
                    base = planner._policy_rollout(z0, args.num_base)
                    base = base + torch.randn_like(base) * p.base_policy_std
                    mean = torch.zeros(args.horizon, planner.action_dim, device="cuda")
                    std = torch.full((args.horizon, planner.action_dim), p.init_std, device="cuda")
                    noise = torch.randn(args.horizon, args.num_samples - args.num_base,
                                        planner.action_dim, device="cuda")
                    cands = torch.cat([(mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1),
                                       base.clamp(-1, 1)], dim=1)
                    sim_state = main_env.env.sim.get_state().flatten().copy()
                    B_sets.append(real_terminals(probe_env, sim_state, cands))
                    restore(main_env, sim_state)
                a = planner.plan(obs["state"], eval_mode=True)
            obs, _, term, _, _ = main_env.step(a.cpu())
            if term:
                break
        log(f"  ep{ep}: harvested {len(B_sets)} candidate sets")
    B = torch.cat(B_sets, dim=0)

    # =============================== measurements ===================================
    out = {}
    for name, S in [("A: demo states (in-dist)", A), ("B: MPPI candidate terminals", B)]:
        with torch.no_grad():
            vh, ve = v_hat(S), v_exp(S)
        gh, ge = grad_norm(v_hat, S), grad_norm(v_exp, S)
        # normalize the gradient by each V's own spread -> scale-free roughness
        rh = (gh / vh.std()).mean().item()
        re = (ge / ve.std()).mean().item()
        sp = spearmanr(vh.cpu().numpy(), ve.cpu().numpy()).correlation
        log(f"\n=== {name}   (n={len(S)})")
        log(f"  V_hat : mean {vh.mean():.4f}  std {vh.std():.4f}")
        log(f"  V_exp : mean {ve.mean():.4f}  std {ve.std():.4f}")
        log(f"  P1 Spearman(V_hat, V_exp) = {sp:+.4f}")
        log(f"  P2 归一化局部粗糙度 ||dV/ds||/std :  V_hat {rh:8.2f}   V_exp {re:8.2f}"
            f"   (V_exp / V_hat = {re / rh:.2f}x)")
        out[name] = (rh, re)

    # P3: within each candidate set, how much of an OUTLIER is the max?
    log(f"\n=== P3 尖峰突出度 (每个候选集合内部, {len(B_sets)} 个集合 × {args.num_samples} 候选)")
    def spike(v):
        med = v.median()
        q75, q25 = v.quantile(0.75), v.quantile(0.25)
        iqr = (q75 - q25).clamp(min=1e-6)
        return ((v.max() - med) / iqr).item()
    sh = [spike(v_hat(S)) for S in B_sets]
    se = [spike(v_exp(S)) for S in B_sets]
    with torch.no_grad():
        sh = np.array([spike(v_hat(S)) for S in B_sets])
        se = np.array([spike(v_exp(S)) for S in B_sets])
    log(f"  (max - median) / IQR :  V_hat {sh.mean():.2f} ± {sh.std():.2f}"
        f"    V_exp {se.mean():.2f} ± {se.std():.2f}   (V_exp / V_hat = {se.mean()/sh.mean():.2f}x)")
    log(f"  V_exp 的尖峰更突出的集合占比: {100*(se > sh).mean():.1f}%")

    log("\n===== 判读 =====")
    ra, rb = out["A: demo states (in-dist)"], out["B: MPPI candidate terminals"]
    log(f"  P2 粗糙度比 (V_exp/V_hat):  在 A(in-dist) = {ra[1]/ra[0]:.2f}x"
        f"   在 B(planner 查询分布) = {rb[1]/rb[0]:.2f}x")
    log(f"  ⇒ 低通滤波假说要求: 比值 > 1, 且在 B 上比在 A 上更大 (尖峰住在分布外)")
    log(f"  P3 要求: V_exp 的 (max-med)/IQR 更大 (给 argmax 更尖的钩子)")


if __name__ == "__main__":
    main()
