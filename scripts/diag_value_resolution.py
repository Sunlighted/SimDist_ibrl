"""Is the LOCAL value resolution missing in V_src itself, or lost in the
distillation to V_wm? And does a long horizon keep improving the WM score?

Ground truth = Monte-Carlo: from each candidate action's TRUE next state, roll
the source RL policy in the source env to success; MC return = gamma^(steps to
success), else 0. Then compare, over a 2-D action grid at a real insertion state:
  Q_src(s0,a)            source critic action-value
  V_src(s1_true)         source critic value of the true next state
  V_wm(enc(s1_true))     distilled WM value head on the true next state
  W_H(a)                 WM value-return at horizon H (H=1..40)
vs the MC ground truth. Spearman correlations tell us:
  - corr(Q_src/V_src, MC) high  -> the RL critic HAS local resolution
  - corr(V_wm, MC) low while corr(V_src, MC) high -> distillation lost it
  - corr(W_H, MC) vs H -> does longer horizon keep helping or saturate?
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as e
import classifier_test.cf_utils as cf
import pyrallis

SIMDIST = "exps/mimicgen/threading_simdist_polexpert_seed1/frozen_cmp.pt"
SOURCE = "exps/mimicgen/threading_ibrl_source_seed1/model0.pt"
DEMOS = "release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5"
GAMMA = 0.99
GRID = 11
RANGE = 0.5
DIMS = (0, 1)
MC_STEPS = 30
HS = [1, 3, 5, 10, 20, 30, 40]


def load_source_agent(env):
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(SOURCE), "cfg.yaml")))
    return train_rl.load_frozen_agent(SOURCE, env, cfg, "cuda")


def mc_return(env, sim0, a, policy):
    """Reset to sim0, step a, then roll the source policy; return gamma^k at the
    first success step k (1-indexed from sim0), else 0."""
    raw = env.env
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    obs, r, done, _ = raw.step(np.asarray(a, dtype=np.float64))
    if r > 0.5:
        return GAMMA ** 1
    for k in range(2, MC_STEPS + 1):
        if done:
            break
        st = torch.tensor(cf.extract_state(obs), dtype=torch.float32).cuda()
        with torch.no_grad():
            act = policy.act({"state": st}, eval_mode=True).cpu().numpy()
        obs, r, done, _ = raw.step(np.asarray(act, dtype=np.float64))
        if r > 0.5:
            return GAMMA ** k
    return 0.0


def main():
    agent, ws = e.build(SIMDIST, use_planner=0)
    m = agent.model
    env = cf.get_env(SOURCE, domain="source")
    src = load_source_agent(env)

    eps = cf.load_demo_episodes(DEMOS, max_episodes=30)
    ep = next(x for x in eps if x["rewards"].sum() > 0)
    T = len(ep["action"])
    t0 = max(0, T - 10)
    s0 = ep["state"][t0].astype(np.float32)
    a_star = ep["action"][t0].astype(np.float32)
    sim0 = ep["sim_state"][t0]

    dx, dy = DIMS
    xs = np.linspace(a_star[dx] - RANGE, a_star[dx] + RANGE, GRID)
    ys = np.linspace(a_star[dy] - RANGE, a_star[dy] + RANGE, GRID)
    A = np.tile(a_star, (GRID * GRID, 1))
    gx, gy = np.meshgrid(xs, ys)
    A[:, dx] = gx.reshape(-1)
    A[:, dy] = gy.reshape(-1)
    A = np.clip(A, -1, 1)
    At = torch.tensor(A, dtype=torch.float32).cuda()
    s0t = torch.tensor(s0, dtype=torch.float32).cuda().unsqueeze(0)
    N = GRID * GRID

    with torch.no_grad():
        z0 = m.encode(s0t).expand(N, -1).clone()
        WH = {}
        z = z0.clone()
        for h in range(1, max(HS) + 1):
            z = m.next(z, At)
            if h in HS:
                WH[h] = (m.predict_value(z) * GAMMA ** h).cpu().numpy()
        Qsrc = src.critic.forward_k(s0t.expand(N, -1), At).min(-1)[0].cpu().numpy()

    # ground truth + true-state values
    s1_true = np.zeros((N, s0.shape[0]), dtype=np.float32)
    MC = np.zeros(N, dtype=np.float32)
    for i in range(N):
        s1_true[i], _, _ = cf.counterfactual_step(env, sim0, A[i])
        MC[i] = mc_return(env, sim0, A[i], src)
    s1t = torch.tensor(s1_true).cuda()
    with torch.no_grad():
        Vsrc_true = src.value({"state": s1t}).cpu().numpy()
        Vwm_true = m.predict_value(m.encode(s1t)).cpu().numpy()

    def sp(a, b):
        c = spearmanr(a, b).correlation
        return 0.0 if np.isnan(c) else c

    print(f"\nstate t0={t0}/{T},  MC success fraction over grid = {(MC>0).mean():.2f}")
    print("dynamic range (std) on the grid:")
    print(f"   Q_src={Qsrc.std():.4f}  V_src(true)={Vsrc_true.std():.4f}  "
          f"V_wm(true)={Vwm_true.std():.4f}  MC={MC.std():.4f}")
    print("\nSpearman vs MC ground-truth return:")
    print(f"   Q_src(s0,a)         {sp(Qsrc, MC):+.3f}   <- does the RL critic resolve good actions?")
    print(f"   V_src(s1_true)      {sp(Vsrc_true, MC):+.3f}   <- does V_src resolve good next states?")
    print(f"   V_wm(enc(s1_true))  {sp(Vwm_true, MC):+.3f}   <- did the distilled head keep it?")
    print(f"\n   corr(V_wm_true, V_src_true) = {sp(Vwm_true, Vsrc_true):+.3f}  (distillation fidelity, local)")
    print("\nWM value-return vs MC and vs Q_src, over horizon H:")
    for h in HS:
        print(f"   H={h:2d}:  corr(W_H, MC)={sp(WH[h], MC):+.3f}   corr(W_H, Q_src)={sp(WH[h], Qsrc):+.3f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
