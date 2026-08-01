"""For the SAME MPPI candidates (base policy + per-step noise), which value
prediction is closer to the TRUE Monte-Carlo return:
  A = V_wm(encode(s_H_true))   real env rollout -> encode -> value  (perfect dyn)
  B = V_wm(z_H_wm)             latent-space rollout -> value        (MPPI actual)
MC = from s_H_true, roll source policy to success -> gamma^k        (ground truth)
If A >> B in correlation with MC, the latent DYNAMICS is what breaks MPPI; if A
is also low, the VALUE head is the bottleneck.
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
H = 5
K = 40            # candidates per state
SIGMA = 0.15      # per-step noise around the base policy (MPPI candidate scale)
N_STATES = 8
MC_STEPS = 30


def load_source_agent(env):
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(SOURCE), "cfg.yaml")))
    return train_rl.load_frozen_agent(SOURCE, env, cfg, "cuda")


def base_seq(raw, sim0, src):
    """Source-policy closed-loop H-step action sequence (the base proposal)."""
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0; raw.sim.forward()
    obs = raw._get_observations(force_update=True)
    seq = []
    for _ in range(H):
        st = torch.tensor(cf.extract_state(obs), dtype=torch.float32).cuda()
        with torch.no_grad():
            a = src.act({"state": st}, eval_mode=True).cpu().numpy()
        seq.append(a)
        obs, _, done, _ = raw.step(a)
    return np.array(seq)


def exec_and_mc(raw, sim0, seq, src):
    """Execute the candidate seq -> s_H_true, then policy to success -> MC return."""
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0; raw.sim.forward()
    obs = r = None
    for h in range(H):
        obs, r, done, _ = raw.step(np.asarray(seq[h], dtype=np.float64))
        if done:
            break
    s_H = cf.extract_state(obs)
    if r is not None and r > 0.5:
        return s_H, GAMMA ** H
    for k in range(H + 1, H + MC_STEPS + 1):
        if done:
            break
        st = torch.tensor(cf.extract_state(obs), dtype=torch.float32).cuda()
        with torch.no_grad():
            a = src.act({"state": st}, eval_mode=True).cpu().numpy()
        obs, r, done, _ = raw.step(a)
        if r > 0.5:
            return s_H, GAMMA ** k
    return s_H, 0.0


def wm_latent(m, s0t, seq):
    with torch.no_grad():
        z = m.encode(s0t)
        for h in range(len(seq)):
            a = torch.tensor(seq[h], dtype=torch.float32).cuda().unsqueeze(0)
            z = m.next(z, a)
    return z  # [1,zdim]


def main():
    agent, ws = e.build(SIMDIST, use_planner=0)
    m = agent.model
    env = cf.get_env(SOURCE, domain="source")
    src = load_source_agent(env)
    raw = env.env
    eps = cf.load_demo_episodes(DEMOS, max_episodes=40)
    succ = [x for x in eps if x["rewards"].sum() > 0]
    rng = np.random.RandomState(0)

    cA, cB, allA, allB, allMC = [], [], [], [], []
    for si in range(N_STATES):
        ep = succ[si]; T = len(ep["action"]); t0 = max(0, T - 12)
        s0 = ep["state"][t0].astype(np.float32)
        sim0 = ep["sim_state"][t0]
        s0t = torch.tensor(s0, dtype=torch.float32).cuda().unsqueeze(0)
        base = base_seq(raw, sim0, src)
        A, B, MC = [], [], []
        for _ in range(K):
            seq = np.clip(base + SIGMA * rng.randn(H, 7).astype(np.float32), -1, 1)
            s_H, mc = exec_and_mc(raw, sim0, seq, src)
            z_wm = wm_latent(m, s0t, seq)
            with torch.no_grad():
                a_val = float(m.predict_value(m.encode(torch.tensor(s_H).cuda().unsqueeze(0))) * GAMMA ** H)
                b_val = float(m.predict_value(z_wm) * GAMMA ** H)
            A.append(a_val); B.append(b_val); MC.append(mc)
        A, B, MC = map(np.array, (A, B, MC))
        ra = spearmanr(A, MC).correlation
        rb = spearmanr(B, MC).correlation
        ra = 0.0 if np.isnan(ra) else ra; rb = 0.0 if np.isnan(rb) else rb
        cA.append(ra); cB.append(rb)
        allA += list(A); allB += list(B); allMC += list(MC)
        print(f"state {si}: MC success frac={np.mean(MC>0):.2f}  "
              f"corr(A=real->enc->V, MC)={ra:+.2f}   corr(B=latent->V, MC)={rb:+.2f}")

    gA = spearmanr(allA, allMC).correlation
    gB = spearmanr(allB, allMC).correlation
    print(f"\n=== same candidates (base policy + sigma={SIGMA} noise), H={H}, "
          f"{N_STATES} states x {K} candidates ===")
    print(f"  A  real-rollout -> encode -> value :  per-state corr {np.mean(cA):+.3f} | pooled {gA:+.3f}")
    print(f"  B  latent-rollout -> value (MPPI)  :  per-state corr {np.mean(cB):+.3f} | pooled {gB:+.3f}")
    print(f"\n  => the value head on the TRUE encoded state is "
          f"{'MUCH better' if np.mean(cA)-np.mean(cB) > 0.2 else 'similar'} -> "
          f"{'latent DYNAMICS is the main breakage' if np.mean(cA)-np.mean(cB) > 0.2 else 'value head also weak'}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
