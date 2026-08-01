"""Clean up the decoder confound: quantify (1) decoder reconstruction error and
(2) LATENT-SPACE dynamics error (no decoder), on the goodpolicy target-adapted WM.

For the same target insertion states + action grid:
  recon_rel      = ||decode(encode(s)) - s|| / ||s_H_true - s0||   (decoder quality vs signal)
  latent_dyn_rel = ||z_H_wm - encode(s_H_true)|| / ||encode(s_H_true) - encode(s0)||
                   (WM latent prediction drift vs the TRUE latent move; >1 => the
                    latent dynamics error exceeds the signal, decoder-free)
Also reprints corr(val_only,GT) vs corr(mppi,GT) (both WM-value, no decoder) as
the clean dynamics isolation.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, pyrallis
from scipy.stats import spearmanr
import classifier_test.cf_utils as cf

WM_CKPT = "exps/mimicgen/threading_simdist_adapt_goodpolicy_seed1/latest.pt"
WM_CFG_DIR = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1"
TGT_CRITIC = "exps/mimicgen/threading_ibrl_target_snapshots_rlprior_seed1/model0.pt"
DEMOS = "release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5"
GAMMA, GRID, RANGE, DIMS, N_STATES, HS = 0.99, 11, 0.5, (0, 1), 8, [1, 3, 5]


def sp(a, b):
    c = spearmanr(a, b).correlation
    return 0.0 if np.isnan(c) else c


def build_wm():
    from train_simdist import MainConfig
    from rl.world_model import WorldModel
    import h5py
    bc = pyrallis.load(MainConfig, open(os.path.join(WM_CFG_DIR, "cfg.yaml")))
    f = h5py.File(DEMOS, "r"); k0 = sorted(f["data"].keys())[0]
    od = f["data"][k0]["obs"]["state"].shape[1]; ad = f["data"][k0]["actions"].shape[1]; f.close()
    m = WorldModel(od, ad, bc.simdist.world_model).to("cuda")
    sd = torch.load(WM_CKPT); m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
    m.eval(); return m


def roll(raw, sim0, a, H):
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0; raw.sim.forward()
    obs, _, _, _ = raw.step(np.asarray(a, np.float64))
    for _ in range(H - 1):
        obs, _, d, _ = raw.step(np.asarray(a, np.float64))
        if d: break
    return cf.extract_state(obs)


def main():
    m = build_wm()
    env = cf.get_env(TGT_CRITIC, domain="target"); raw = env.env
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(TGT_CRITIC), "cfg.yaml")))
    src = train_rl.load_frozen_agent(TGT_CRITIC, env, cfg, "cuda")
    eps = [x for x in cf.load_demo_episodes(DEMOS, max_episodes=60) if x["rewards"].sum() > 0]
    print(f"=== latent dyn err + decoder recon | goodpolicy target-adapted WM | {N_STATES} states ===")
    for H in HS:
        recon_rel, ldyn_rel, cvo, cmp = [], [], [], []
        for si in range(N_STATES):
            ep = eps[si]; T = len(ep["action"]); t0 = max(0, T - 12)
            s0 = ep["state"][t0].astype(np.float32); a_star = ep["action"][t0].astype(np.float32)
            sim0 = ep["sim_state"][t0]
            dx, dy = DIMS
            A = np.tile(a_star, (GRID * GRID, 1))
            gx, gy = np.meshgrid(np.linspace(a_star[dx]-RANGE, a_star[dx]+RANGE, GRID),
                                 np.linspace(a_star[dy]-RANGE, a_star[dy]+RANGE, GRID))
            A[:, dx] = gx.reshape(-1); A[:, dy] = gy.reshape(-1); A = np.clip(A, -1, 1)
            At = torch.tensor(A, dtype=torch.float32).cuda()
            s0t = torch.tensor(s0, dtype=torch.float32).cuda().unsqueeze(0)
            N = GRID * GRID
            with torch.no_grad():
                z0 = m.encode(s0t)
                z = z0.expand(N, -1).clone()
                for _ in range(H): z = m.next(z, At)
                z_H_wm = z
                mppi = (m.predict_value(z_H_wm) * GAMMA**H).cpu().numpy()
            s_H_true = np.stack([roll(raw, sim0, A[i], H) for i in range(N)])
            sHt = torch.tensor(s_H_true, dtype=torch.float32).cuda()
            with torch.no_grad():
                z_true = m.encode(sHt)                        # latent of the TRUE next state
                GT = (src.value({"state": sHt}) * GAMMA**H).cpu().numpy()
                val_only = (m.predict_value(z_true) * GAMMA**H).cpu().numpy()
                recon = m.decode(z_true)                      # decode(encode(s_true))
                recon_err = torch.norm(recon - sHt, dim=1).cpu().numpy()
                # latent-space dynamics error (no decoder): predicted latent vs true-state latent
                ldyn = torch.norm(z_H_wm - z_true, dim=1).cpu().numpy()
                lmove = torch.norm(z_true - z0, dim=1).clamp(min=1e-6).cpu().numpy()
                move = np.linalg.norm(s_H_true - s0, axis=1)
            recon_rel.append((recon_err / np.clip(move, 1e-6, None)).mean())
            ldyn_rel.append((ldyn / lmove).mean())
            cvo.append(sp(val_only, GT)); cmp.append(sp(mppi, GT))
        print(f"\n--- H={H} ---")
        print(f"  decoder recon err / true move          = {np.mean(recon_rel):.3f}   (<<1 => decoder fine)")
        print(f"  LATENT dyn err / true latent move      = {np.mean(ldyn_rel):.3f}   (>1 => latent dynamics error exceeds signal, DECODER-FREE)")
        print(f"  corr(val_only,GT)={np.mean(cvo):+.3f}  vs  corr(mppi,GT)={np.mean(cmp):+.3f}   (both WM-value, no decoder)")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
