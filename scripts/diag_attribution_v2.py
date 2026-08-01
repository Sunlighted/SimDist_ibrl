"""Re-do the value-vs-dynamics attribution with REALISTIC MPPI candidates: scan
the FIRST action a0, then CONTINUE with the policy (not broadcast a0), matching
how MPPI scores a candidate (first step + sensible continuation). For each a0:
  s_H_true = env: step a0, then source policy H-1 steps           (true dyn)
  z_H_wm   = WM:  next(z0,a0), then WM-policy via decode H-1 steps (WM dyn)
Four scores vs GT=V_src(s_H_true): val_only (WM val on true s), dyn_only (true
val on WM s), mppi (WM val on WM z). Also MC truth + dynamics error.
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
GRID = 9
RANGE = 0.5
DIMS = (0, 1)
H = 5
N_STATES = 5
MC_STEPS = 30


def load_source_agent(env):
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(SOURCE), "cfg.yaml")))
    return train_rl.load_frozen_agent(SOURCE, env, cfg, "cuda")


def env_first_then_policy(raw, sim0, a0, src, mc=False):
    """Step a0, then source policy; return s_H (or MC return if mc)."""
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0; raw.sim.forward()
    obs, r, done, _ = raw.step(np.asarray(a0, dtype=np.float64))
    if mc and r > 0.5:
        return GAMMA
    last = cf.extract_state(obs)
    steps = MC_STEPS if mc else H
    for k in range(2, steps + 1):
        if done:
            break
        st = torch.tensor(cf.extract_state(obs), dtype=torch.float32).cuda()
        with torch.no_grad():
            a = src.act({"state": st}, eval_mode=True).cpu().numpy()
        obs, r, done, _ = raw.step(a)
        if k == H:
            last = cf.extract_state(obs)
        if mc and r > 0.5:
            return GAMMA ** k
    return 0.0 if mc else last


def wm_first_then_policy(m, s0t, a0t):
    """next(z0,a0), then WM policy via decode; return z_H [B,z]."""
    with torch.no_grad():
        z = m.next(m.encode(s0t).expand(a0t.shape[0], -1), a0t)
        for _ in range(H - 1):
            a = m.policy_action(m.decode(z))
            z = m.next(z, a)
    return z


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

    R = {"val_only": [], "dyn_only": [], "mppi": [], "dyn_err": [],
         "val_mc": [], "mppi_mc": []}
    for si in range(N_STATES):
        ep = succ[si]; T = len(ep["action"]); t0 = max(0, T - 12)
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

        z_H = wm_first_then_policy(m, s0t, At)
        with torch.no_grad():
            s_H_pred = m.decode(z_H)
            mppi = (m.predict_value(z_H) * GAMMA ** H).cpu().numpy()
        s_H_true = np.stack([env_first_then_policy(raw, sim0, A[i], src) for i in range(Npt)])
        MC = np.array([env_first_then_policy(raw, sim0, A[i], src, mc=True) for i in range(Npt)])
        sHt = torch.tensor(s_H_true, dtype=torch.float32).cuda()
        with torch.no_grad():
            GT = (src.value({"state": sHt}) * GAMMA ** H).cpu().numpy()
            val_only = (m.predict_value(m.encode(sHt)) * GAMMA ** H).cpu().numpy()
            dyn_only = (src.value({"state": s_H_pred}) * GAMMA ** H).cpu().numpy()
            dyn_err = np.linalg.norm(s_H_pred.cpu().numpy() - s_H_true, axis=1).mean()
        R["val_only"].append(sp(val_only, GT)); R["dyn_only"].append(sp(dyn_only, GT))
        R["mppi"].append(sp(mppi, GT)); R["dyn_err"].append(dyn_err)
        R["val_mc"].append(sp(val_only, MC)); R["mppi_mc"].append(sp(mppi, MC))
        print(f"state {si}: val_only={sp(val_only,GT):+.2f} dyn_only={sp(dyn_only,GT):+.2f} "
              f"mppi={sp(mppi,GT):+.2f} | vs MC: val={sp(val_only,MC):+.2f} mppi={sp(mppi,MC):+.2f} "
              f"| dyn_err={dyn_err:.3f}")

    print(f"\n=== per-step candidates (first step + policy continuation), H={H}, {N_STATES} states ===")
    print(f"  corr(val_only, GT)  = {np.mean(R['val_only']):+.3f}   (true dyn + WM value -> isolates VALUE)")
    print(f"  corr(dyn_only, GT)  = {np.mean(R['dyn_only']):+.3f}   (WM dyn + true value -> isolates DYNAMICS)")
    print(f"  corr(mppi,     GT)  = {np.mean(R['mppi']):+.3f}   (MPPI actual)")
    print(f"  WM dynamics error H={H}: {np.mean(R['dyn_err']):.3f}")
    print(f"  vs MC truth: corr(val_only,MC)={np.mean(R['val_mc']):+.3f}  corr(mppi,MC)={np.mean(R['mppi_mc']):+.3f}")
    lo = "VALUE" if np.mean(R['val_only']) < np.mean(R['dyn_only']) else "DYNAMICS"
    print(f"  => more-broken component: {lo}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
