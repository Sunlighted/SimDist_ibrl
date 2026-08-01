"""Where does the WM dynamics error come from? Perturb the expert action by
noise of scale sigma, broadcast it over the horizon, and measure the WM
prediction error vs the REAL env at each horizon step.

  sigma=0      : expert action repeated (on-distribution ACTION, but a repeated
                 action is an off-distribution SEQUENCE) -> isolates "broadcast".
  sigma>0      : adds the MPPI candidate noise -> isolates "perturbed action".
  err at H=1   : is a SINGLE step of a perturbed action already wrong?
  err vs H     : how fast does it compound?
Compared to the true state move, to judge "is the prediction basically wrong".
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as e
import classifier_test.cf_utils as cf
import pyrallis

SIMDIST = "exps/mimicgen/threading_simdist_polexpert_seed1/frozen_cmp.pt"
SOURCE = "exps/mimicgen/threading_ibrl_source_seed1/model0.pt"
DEMOS = "release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5"
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.4]
HMAX = 10
K = 32
N_STATES = 4
HS_REPORT = [1, 2, 3, 5, 10]


def load_source_agent(env):
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(SOURCE), "cfg.yaml")))
    return train_rl.load_frozen_agent(SOURCE, env, cfg, "cuda")


def env_rollout(raw, sim0, a, H):
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    states = []
    obs = None
    for h in range(H):
        obs, _, done, _ = raw.step(np.asarray(a, dtype=np.float64))
        states.append(cf.extract_state(obs))
        if done:
            while len(states) < H:
                states.append(states[-1])
            break
    return np.stack(states)  # [H, 37]


def main():
    agent, ws = e.build(SIMDIST, use_planner=0)
    m = agent.model
    env = cf.get_env(SOURCE, domain="source")
    src = load_source_agent(env)
    raw = env.env

    eps = cf.load_demo_episodes(DEMOS, max_episodes=40)
    succ = [x for x in eps if x["rewards"].sum() > 0]
    rng = np.random.RandomState(0)

    # err[sigma][h] accumulates over states/candidates
    err = {s: np.zeros(HMAX) for s in SIGMAS}
    cnt = {s: 0 for s in SIGMAS}
    move = np.zeros(HMAX); mcnt = 0

    for si in range(N_STATES):
        ep = succ[si]; T = len(ep["action"]); t0 = max(0, T - 12)
        s0 = ep["state"][t0].astype(np.float32)
        a_star = ep["action"][t0].astype(np.float32)
        sim0 = ep["sim_state"][t0]
        s0t = torch.tensor(s0, dtype=torch.float32).cuda().unsqueeze(0)

        for sigma in SIGMAS:
            kk = 1 if sigma == 0 else K
            A = np.clip(a_star[None] + sigma * rng.randn(kk, 7).astype(np.float32), -1, 1)
            At = torch.tensor(A, dtype=torch.float32).cuda()
            # WM rollout, decode at each h
            with torch.no_grad():
                z = m.encode(s0t).expand(kk, -1).clone()
                s_pred = []
                for h in range(HMAX):
                    z = m.next(z, At)
                    s_pred.append(m.decode(z).cpu().numpy())
            s_pred = np.stack(s_pred, 1)  # [kk, HMAX, 37]
            # env rollout per candidate
            for i in range(kk):
                s_true = env_rollout(raw, sim0, A[i], HMAX)  # [HMAX,37]
                err[sigma] += np.linalg.norm(s_pred[i] - s_true, axis=1)
                cnt[sigma] += 1
                if sigma == 0:
                    move += np.linalg.norm(s_true - s0, axis=1); mcnt += 1

    print(f"WM dynamics prediction error  ||decode(rollout_h) - s_true_h||  "
          f"(avg over {N_STATES} states)\n")
    hdr = "  sigma \\ H |" + "".join(f"  H={h:<2d}" for h in HS_REPORT)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for sigma in SIGMAS:
        e_h = err[sigma] / max(cnt[sigma], 1)
        tag = "(expert, broadcast)" if sigma == 0 else "(expert+noise)"
        print(f"  {sigma:>5.2f}    |" + "".join(f"  {e_h[h-1]:.3f}" for h in HS_REPORT) + f"   {tag}")
    mv = move / max(mcnt, 1)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  true move |" + "".join(f"  {mv[h-1]:.3f}" for h in HS_REPORT)
          + "   (||s_true_h - s0||, the scale to beat)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
