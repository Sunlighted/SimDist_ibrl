"""Cleanly attribute the MPPI scoring error to VALUE vs DYNAMICS by swapping each
to ground truth on an action grid (H-step, action broadcast over the horizon).

For each candidate action a (broadcast H times):
  s_H_true  = roll the REAL source env H steps from s0 with a   (true dynamics)
  z_H_wm    = roll the WM latent dynamics H steps                (WM dynamics)
  s_H_pred  = decode(z_H_wm)
Four scores:
  GT          = V_src(s_H_true)              true dyn + true value   (the target)
  val_only    = V_wm(enc(s_H_true))          true dyn + WM value     (isolates value)
  dyn_only    = V_src(s_H_pred)              WM dyn  + true value     (isolates dynamics)
  mppi        = V_wm(z_H_wm)                 WM dyn  + WM value       (what MPPI uses)
corr(val_only, GT) low  -> the VALUE head is the bottleneck.
corr(dyn_only, GT) low  -> the DYNAMICS is the bottleneck.
Averaged over several insertion states.
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
H = 5
N_STATES = 5


def load_source_agent(env):
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(SOURCE), "cfg.yaml")))
    return train_rl.load_frozen_agent(SOURCE, env, cfg, "cuda")


def env_rollout_H(raw, sim0, a):
    """Roll the REAL env H steps from sim0 with action a broadcast; return s_H."""
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    obs, _, _, _ = raw.step(np.asarray(a, dtype=np.float64))
    for _ in range(H - 1):
        obs, _, done, _ = raw.step(np.asarray(a, dtype=np.float64))
        if done:
            break
    return cf.extract_state(obs)


def sp(a, b):
    c = spearmanr(a, b).correlation
    return 0.0 if np.isnan(c) else c


def main():
    agent, ws = e.build(SIMDIST, use_planner=0)
    m = agent.model
    env = cf.get_env(SOURCE, domain="source")
    src = load_source_agent(env)
    raw = env.env

    eps = cf.load_demo_episodes(DEMOS, max_episodes=40)
    succ = [x for x in eps if x["rewards"].sum() > 0]

    rows = {"val_only": [], "dyn_only": [], "mppi": [], "dyn_err": [], "move": []}
    for si in range(N_STATES):
        ep = succ[si]
        T = len(ep["action"])
        t0 = max(0, T - 12)
        s0 = ep["state"][t0].astype(np.float32)
        a_star = ep["action"][t0].astype(np.float32)
        sim0 = ep["sim_state"][t0]
        dx, dy = DIMS
        xs = np.linspace(a_star[dx] - RANGE, a_star[dx] + RANGE, GRID)
        ys = np.linspace(a_star[dy] - RANGE, a_star[dy] + RANGE, GRID)
        A = np.tile(a_star, (GRID * GRID, 1))
        gx, gy = np.meshgrid(xs, ys)
        A[:, dx] = gx.reshape(-1); A[:, dy] = gy.reshape(-1)
        A = np.clip(A, -1, 1)
        At = torch.tensor(A, dtype=torch.float32).cuda()
        s0t = torch.tensor(s0, dtype=torch.float32).cuda().unsqueeze(0)
        Npt = GRID * GRID

        with torch.no_grad():
            z = m.encode(s0t).expand(Npt, -1).clone()
            for _ in range(H):
                z = m.next(z, At)
            z_H_wm = z
            s_H_pred = m.decode(z_H_wm)
            mppi = (m.predict_value(z_H_wm) * GAMMA ** H).cpu().numpy()

        s_H_true = np.stack([env_rollout_H(raw, sim0, A[i]) for i in range(Npt)])
        sHt = torch.tensor(s_H_true, dtype=torch.float32).cuda()
        with torch.no_grad():
            GT = (src.value({"state": sHt}) * GAMMA ** H).cpu().numpy()        # true dyn + true val
            val_only = (m.predict_value(m.encode(sHt)) * GAMMA ** H).cpu().numpy()  # true dyn + WM val
            dyn_only = (src.value({"state": s_H_pred}) * GAMMA ** H).cpu().numpy()   # WM dyn + true val
            dyn_err = np.linalg.norm(s_H_pred.cpu().numpy() - s_H_true, axis=1).mean()
            move = np.linalg.norm(s_H_true - s0, axis=1).mean()

        rows["val_only"].append(sp(val_only, GT))
        rows["dyn_only"].append(sp(dyn_only, GT))
        rows["mppi"].append(sp(mppi, GT))
        rows["dyn_err"].append(dyn_err)
        rows["move"].append(move)
        print(f"state {si}: corr(val_only,GT)={sp(val_only,GT):+.2f}  "
              f"corr(dyn_only,GT)={sp(dyn_only,GT):+.2f}  corr(mppi,GT)={sp(mppi,GT):+.2f}  "
              f"dyn_err={dyn_err:.3f} (move={move:.3f})")

    print(f"\n=== averaged over {N_STATES} insertion states (H={H}) ===")
    print(f"  corr(val_only , GT) = {np.mean(rows['val_only']):+.3f}   "
          f"<- TRUE dynamics + WM value  (isolates the VALUE head)")
    print(f"  corr(dyn_only , GT) = {np.mean(rows['dyn_only']):+.3f}   "
          f"<- WM dynamics + TRUE value  (isolates the DYNAMICS)")
    print(f"  corr(mppi     , GT) = {np.mean(rows['mppi']):+.3f}   <- what MPPI actually uses")
    print(f"  WM dynamics error after {H} steps = {np.mean(rows['dyn_err']):.4f}  "
          f"(vs true state move {np.mean(rows['move']):.4f})")
    lo = "VALUE head" if np.mean(rows['val_only']) < np.mean(rows['dyn_only']) else "DYNAMICS"
    print(f"\n  => the lower-correlation (more broken) component is: {lo}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
