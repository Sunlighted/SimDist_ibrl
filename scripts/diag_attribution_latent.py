"""Attribution done ENTIRELY in latent space (how MPPI actually scores).
MPPI return = V_wm(z_H), z_H rolled by the latent dynamics. No decode.
Per-step candidates (first action scanned, policy continuation).

  z_H_wm   = WM latent rollout
  z_H_true = encode(s_H_true)   (the latent of the REAL next state)
  A = V_wm(z_H_true)   value head on the TRUE latent   -> isolates the VALUE head
  B = V_wm(z_H_wm)     MPPI actual score
  GT = V_src(s_H_true) ground-truth value
  corr(A, GT)  : value head fidelity (on correct latents)
  corr(B, A)   : how much the latent DYNAMICS error corrupts the score
  corr(B, GT)  : MPPI actual
  latent err   : ||z_H_wm - z_H_true||  (SimNorm latent, bounded)
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


def load_source_agent(env):
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(SOURCE), "cfg.yaml")))
    return train_rl.load_frozen_agent(SOURCE, env, cfg, "cuda")


def env_first_then_policy(raw, sim0, a0, src):
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0; raw.sim.forward()
    obs, r, done, _ = raw.step(np.asarray(a0, dtype=np.float64))
    last = cf.extract_state(obs)
    for k in range(2, H + 1):
        if done:
            break
        st = torch.tensor(cf.extract_state(obs), dtype=torch.float32).cuda()
        with torch.no_grad():
            a = src.act({"state": st}, eval_mode=True).cpu().numpy()
        obs, r, done, _ = raw.step(a)
        if k == H:
            last = cf.extract_state(obs)
    return last


def wm_first_then_policy(m, s0t, a0t):
    with torch.no_grad():
        z = m.next(m.encode(s0t).expand(a0t.shape[0], -1), a0t)
        for _ in range(H - 1):
            z = m.next(z, m.policy_action(m.decode(z)))
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

    R = {"val": [], "dyn_impact": [], "mppi": [], "lat_err": [], "lat_scale": []}
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

        z_wm = wm_first_then_policy(m, s0t, At)
        s_true = np.stack([env_first_then_policy(raw, sim0, A[i], src) for i in range(Npt)])
        sTt = torch.tensor(s_true, dtype=torch.float32).cuda()
        with torch.no_grad():
            z_true = m.encode(sTt)
            A_score = (m.predict_value(z_true) * GAMMA ** H).cpu().numpy()   # WM val on TRUE latent
            B_score = (m.predict_value(z_wm) * GAMMA ** H).cpu().numpy()      # MPPI actual
            GT = (src.value({"state": sTt}) * GAMMA ** H).cpu().numpy()
            lat_err = (z_wm - z_true).norm(dim=1).mean().item()
            lat_scale = z_true.norm(dim=1).mean().item()
        R["val"].append(sp(A_score, GT))
        R["dyn_impact"].append(sp(B_score, A_score))
        R["mppi"].append(sp(B_score, GT))
        R["lat_err"].append(lat_err); R["lat_scale"].append(lat_scale)
        print(f"state {si}: corr(val_head,GT)={sp(A_score,GT):+.2f}  "
              f"corr(mppi,val_head)={sp(B_score,A_score):+.2f}  corr(mppi,GT)={sp(B_score,GT):+.2f}  "
              f"latent_err={lat_err:.3f}/{lat_scale:.2f}")

    print(f"\n=== LATENT-space attribution (per-step candidates, H={H}, {N_STATES} states) ===")
    print(f"  corr(value_head, GT)      = {np.mean(R['val']):+.3f}   "
          f"<- WM value head on the TRUE latent vs ground truth  (the VALUE head)")
    print(f"  corr(mppi, value_head)    = {np.mean(R['dyn_impact']):+.3f}   "
          f"<- WM latent vs true latent, same value head  (the latent DYNAMICS impact)")
    print(f"  corr(mppi, GT)            = {np.mean(R['mppi']):+.3f}   <- MPPI actual")
    print(f"  latent dynamics error     = {np.mean(R['lat_err']):.3f}  "
          f"(of latent norm {np.mean(R['lat_scale']):.2f}  ->  {100*np.mean(R['lat_err'])/np.mean(R['lat_scale']):.0f}%)")
    print("\n  reading: if corr(value_head,GT) is the low one -> VALUE head limits MPPI;")
    print("           if corr(mppi,value_head) is the low one -> latent DYNAMICS limits MPPI.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
