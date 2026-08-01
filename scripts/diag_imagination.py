"""Can a policy roll out PURELY in latent space and reach success? And is that
imagined data usable for augmentation (no real env)?

For real start states, compare:
  (a) IMAGINED: z=encode(s0); loop a=policy(decode(z)); z=next(z,a)  [open-loop latent]
  (b) REAL    : same policy closed-loop in the source env
Track: ||decode(z_t) - s_real_t|| (does imagination stay faithful?), the WM
value + reward along the imagined rollout (does the model think it succeeds?),
and the real success.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as e
import classifier_test.cf_utils as cf
import pyrallis

SIMDIST = "exps/mimicgen/threading_simdist_polexpert_seed1/frozen_cmp.pt"
SOURCE = "exps/mimicgen/threading_ibrl_source_seed1/model0.pt"
DEMOS = "release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5"
T_ROLL = 25
N_STATES = 8


def imagined(m, s0):
    s0t = torch.tensor(s0, dtype=torch.float32).cuda().unsqueeze(0)
    with torch.no_grad():
        z = m.encode(s0t)
        dec, val, rew = [], [], []
        for _ in range(T_ROLL):
            a = m.policy_action(m.decode(z))
            rew.append(float(m.predict_reward(z, a)))
            z = m.next(z, a)
            dec.append(m.decode(z).squeeze(0).cpu().numpy())
            val.append(float(m.predict_value(z)))
    return np.array(dec), np.array(val), np.array(rew)


def real_rollout(raw, sim0, m):
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0; raw.sim.forward()
    obs = raw._get_observations(force_update=True)
    sts, succ = [], False
    for _ in range(T_ROLL):
        st = torch.tensor(cf.extract_state(obs), dtype=torch.float32).cuda()
        with torch.no_grad():
            a = m.policy_action(st.unsqueeze(0)).squeeze(0).cpu().numpy()
        obs, r, done, _ = raw.step(a)
        sts.append(cf.extract_state(obs))
        if r > 0.5:
            succ = True
        if done:
            break
    sts = np.array(sts)
    if len(sts) < T_ROLL:
        sts = np.vstack([sts, np.repeat(sts[-1:], T_ROLL - len(sts), 0)])
    return sts, succ


def main():
    agent, ws = e.build(SIMDIST, use_planner=0)
    m = agent.model
    env = cf.get_env(SOURCE, domain="source")
    raw = env.env
    eps = cf.load_demo_episodes(DEMOS, max_episodes=40)
    succ = [x for x in eps if x["rewards"].sum() > 0]

    divs, rsucc = [], []
    ex = None
    for si in range(N_STATES):
        ep = succ[si]; T = len(ep["action"]); t0 = max(0, T - 22)
        s0 = ep["state"][t0].astype(np.float32)
        sim0 = ep["sim_state"][t0]
        dec, val, rew = imagined(m, s0)
        real_s, ok = real_rollout(raw, sim0, m)
        div = np.linalg.norm(dec - real_s, axis=1)
        move = np.linalg.norm(real_s - s0, axis=1)
        divs.append(div); rsucc.append(ok)
        if si == 0:
            ex = (dec, val, rew, real_s, div, move, ok)
        print(f"state {si}: real success(policy)={ok}  "
              f"imagined-vs-real state div @T={div[-1]:.3f} (real move {move[-1]:.3f})  "
              f"imagined WM value end={val[-1]:.3f}  max imagined reward={rew.max():.3f}")

    divs = np.array(divs)
    print(f"\n=== pure-latent imagination vs real (T={T_ROLL}, {N_STATES} states) ===")
    print(f"  imagined-vs-real state divergence: t=5 {divs[:,4].mean():.3f}  "
          f"t=15 {divs[:,14].mean():.3f}  t={T_ROLL} {divs[:,-1].mean():.3f}")
    print(f"  real policy success rate = {np.mean(rsucc):.2f}")

    dec, val, rew, real_s, div, move, ok = ex
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(div, "r.-", label="||decode(z_t) - s_real_t||")
    ax[0].plot(move, "g.-", label="real move ||s_real_t - s0||")
    ax[0].set_title("imagination drift from reality"); ax[0].set_xlabel("latent step"); ax[0].legend(fontsize=8)
    ax[1].plot(val, "b.-")
    ax[1].set_title(f"WM value along imagined rollout (real success={ok})")
    ax[1].set_xlabel("latent step"); ax[1].set_ylabel("V_wm(z_t)")
    ax[2].plot(rew, "m.-")
    ax[2].axhline(1.0, ls="--", c="k", lw=1, label="reward=1 (success)")
    ax[2].set_title("WM predicted reward along imagined rollout")
    ax[2].set_xlabel("latent step"); ax[2].legend(fontsize=8)
    fig.suptitle("Pure-latent imagination: does the model reach (a real) success?", fontsize=12)
    fig.tight_layout()
    fig.savefig("scripts/diag_imagination.png", dpi=110)
    print("saved scripts/diag_imagination.png")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
