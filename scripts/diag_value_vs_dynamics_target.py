"""TARGET-ADAPTED version of diag_value_vs_dynamics: is the ADAPTED target
dynamics the bottleneck (not the source WM)?

WM       = goodpolicy adapt ckpt (target-adapted dynamics), built from the
           stage-1 MainConfig + loaded adapt weights.
env      = TARGET physics (dynamics-only shift) -> env_rollout gives TRUE target
           next-states.
GT value = TARGET expert critic (threading_ibrl_target_snapshots_rlprior, ~0.98
           on target) V(s)=min-ensemble Q(s,pi(s)).
states   = insertion phase (last ~12 steps) of successful TARGET expert demos
           (target_expert_n500, has sim_state).

Per candidate action a (broadcast over H):
  GT       = V_tgt(s_H_true)        true target dyn + true value    (target)
  val_only = V_wm(encode(s_H_true)) true dyn + WM value             (isolates VALUE)
  dyn_only = V_tgt(decode(z_H_wm))  WM dyn + true value             (isolates DYNAMICS)
  mppi     = V_wm(z_H_wm)           WM dyn + WM value               (what MPPI uses)
Low corr(dyn_only,GT) => the ADAPTED dynamics is the broken link.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pyrallis
from scipy.stats import spearmanr
import classifier_test.cf_utils as cf

WM_CKPT = "exps/mimicgen/threading_simdist_adapt_goodpolicy_seed1/latest.pt"
WM_CFG_DIR = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1"
TGT_CRITIC = "exps/mimicgen/threading_ibrl_target_snapshots_rlprior_seed1/model0.pt"
DEMOS = "release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5"
GAMMA = 0.99
GRID = 11
RANGE = 0.5
DIMS = (0, 1)
N_STATES = 8
HS = [1, 3, 5]


def sp(a, b):
    c = spearmanr(a, b).correlation
    return 0.0 if np.isnan(c) else c


def build_wm():
    from train_simdist import MainConfig
    from rl.world_model import WorldModel
    import h5py
    base_cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(WM_CFG_DIR, "cfg.yaml")))  # type: ignore
    wm = base_cfg.simdist.world_model
    f = h5py.File(DEMOS, "r"); k0 = sorted(f["data"].keys())[0]
    obs_dim = f["data"][k0]["obs"]["state"].shape[1]
    act_dim = f["data"][k0]["actions"].shape[1]; f.close()
    m = WorldModel(obs_dim, act_dim, wm).to("cuda")
    sd = torch.load(WM_CKPT)
    msd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    m.load_state_dict(msd); m.eval()
    return m


def env_rollout_H(raw, sim0, a, H):
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


def main():
    m = build_wm()
    env = cf.get_env(TGT_CRITIC, domain="target")   # TARGET physics
    raw = env.env
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(TGT_CRITIC), "cfg.yaml")))
    src = train_rl.load_frozen_agent(TGT_CRITIC, env, cfg, "cuda")   # TARGET expert critic = GT value

    eps = cf.load_demo_episodes(DEMOS, max_episodes=60)
    succ = [x for x in eps if x["rewards"].sum() > 0]
    print(f"=== TARGET-ADAPTED value-vs-dynamics | WM={WM_CKPT} | TARGET env | GT=target expert critic ===")
    print(f"    {len(succ)} successful target demos; {N_STATES} insertion states; grid {GRID}x{GRID}")

    for H in HS:
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
            s_H_true = np.stack([env_rollout_H(raw, sim0, A[i], H) for i in range(Npt)])
            sHt = torch.tensor(s_H_true, dtype=torch.float32).cuda()
            with torch.no_grad():
                GT = (src.value({"state": sHt}) * GAMMA ** H).cpu().numpy()
                val_only = (m.predict_value(m.encode(sHt)) * GAMMA ** H).cpu().numpy()
                dyn_only = (src.value({"state": s_H_pred}) * GAMMA ** H).cpu().numpy()
                dyn_err = np.linalg.norm(s_H_pred.cpu().numpy() - s_H_true, axis=1).mean()
                move = np.linalg.norm(s_H_true - s0, axis=1).mean()
            rows["val_only"].append(sp(val_only, GT))
            rows["dyn_only"].append(sp(dyn_only, GT))
            rows["mppi"].append(sp(mppi, GT))
            rows["dyn_err"].append(dyn_err); rows["move"].append(move)
        print(f"\n--- H={H} (averaged over {N_STATES} target insertion states) ---")
        print(f"  corr(val_only , GT) = {np.mean(rows['val_only']):+.3f}   <- TRUE tgt dyn + WM value  (VALUE head)")
        print(f"  corr(dyn_only , GT) = {np.mean(rows['dyn_only']):+.3f}   <- ADAPTED WM dyn + TRUE value  (DYNAMICS)")
        print(f"  corr(mppi     , GT) = {np.mean(rows['mppi']):+.3f}   <- what MPPI uses")
        print(f"  adapted-WM dyn error after {H} steps = {np.mean(rows['dyn_err']):.4f}  (true move {np.mean(rows['move']):.4f})")
        lo = "VALUE head" if np.mean(rows['val_only']) < np.mean(rows['dyn_only']) else "DYNAMICS"
        print(f"  => more broken component: {lo}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
