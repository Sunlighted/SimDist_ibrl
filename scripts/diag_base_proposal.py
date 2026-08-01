"""Is the MPPI base-policy proposal (policy via decode-rollout) on-distribution
for the dynamics, or already OOD? Compare WM-vs-env dynamics error for three
H-step action sequences at the same insertion state:
  A. on-policy closed-loop   : source policy re-decides each step on the REAL
                               state (truly on-distribution).
  B. base-proposal open-loop : WM policy via the decode-rollout (z -> decode ->
                               policy -> next), the SEQUENCE MPPI actually scores.
  C. base-proposal + noise   : B plus base_policy_std=0.1 gaussian (the candidates).
For each: execute the action sequence in the REAL env -> s_true_H; the WM rolls
the SAME actions -> s_pred_H; error = ||decode(s_pred_H) - s_true_H||.
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
H = 5
N_STATES = 5
NOISE = 0.1


def load_source_agent(env):
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(SOURCE), "cfg.yaml")))
    return train_rl.load_frozen_agent(SOURCE, env, cfg, "cuda")


def env_exec(raw, sim0, aseq):
    """Execute the action sequence in the real env from sim0; return states [H,37]."""
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    sts = []
    for h in range(len(aseq)):
        obs, _, done, _ = raw.step(np.asarray(aseq[h], dtype=np.float64))
        sts.append(cf.extract_state(obs))
        if done:
            while len(sts) < len(aseq):
                sts.append(sts[-1])
            break
    return np.stack(sts)


def wm_predict(m, s0t, aseq):
    """WM rolls the SAME action sequence; return decoded states [H,37]."""
    with torch.no_grad():
        z = m.encode(s0t)
        out = []
        for h in range(len(aseq)):
            a = torch.tensor(aseq[h], dtype=torch.float32).cuda().unsqueeze(0)
            z = m.next(z, a)
            out.append(m.decode(z).squeeze(0).cpu().numpy())
    return np.stack(out)


def base_proposal_seq(m, s0t, noise=0.0, rng=None):
    """Policy-via-decode rollout: z -> decode -> policy_action -> a -> next."""
    with torch.no_grad():
        z = m.encode(s0t)
        seq = []
        for _ in range(H):
            a = m.policy_action(m.decode(z))
            seq.append(a.squeeze(0).cpu().numpy())
            z = m.next(z, a)
    seq = np.array(seq)
    if noise > 0:
        seq = np.clip(seq + noise * rng.randn(*seq.shape).astype(np.float32), -1, 1)
    return seq


def main():
    agent, ws = e.build(SIMDIST, use_planner=0)
    m = agent.model
    env = cf.get_env(SOURCE, domain="source")
    src = load_source_agent(env)
    raw = env.env
    eps = cf.load_demo_episodes(DEMOS, max_episodes=40)
    succ = [x for x in eps if x["rewards"].sum() > 0]
    rng = np.random.RandomState(0)

    errs = {"A_on_policy": [], "B_base_proposal": [], "C_base_noise": []}
    moves = []
    for si in range(N_STATES):
        ep = succ[si]; T = len(ep["action"]); t0 = max(0, T - 12)
        s0 = ep["state"][t0].astype(np.float32)
        sim0 = ep["sim_state"][t0]
        s0t = torch.tensor(s0, dtype=torch.float32).cuda().unsqueeze(0)

        # A. on-policy closed-loop action sequence (decided on REAL states)
        raw.done = False; raw.timestep = 0
        raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
        raw.sim.data.qacc_warmstart[:] = 0.0; raw.sim.forward()
        obs = raw._get_observations(force_update=True)
        a_on = []
        for h in range(H):
            st = torch.tensor(cf.extract_state(obs), dtype=torch.float32).cuda()
            with torch.no_grad():
                a = src.act({"state": st}, eval_mode=True).cpu().numpy()
            a_on.append(a)
            obs, _, done, _ = raw.step(a)
        a_on = np.array(a_on)

        b_seq = base_proposal_seq(m, s0t)
        c_seq = base_proposal_seq(m, s0t, noise=NOISE, rng=rng)

        for name, seq in [("A_on_policy", a_on), ("B_base_proposal", b_seq), ("C_base_noise", c_seq)]:
            s_true = env_exec(raw, sim0, seq)
            s_pred = wm_predict(m, s0t, seq)
            errs[name].append(np.linalg.norm(s_pred - s_true, axis=1))   # [H]
        moves.append(np.linalg.norm(env_exec(raw, sim0, a_on) - s0, axis=1))

    HS = [1, 2, 3, 5]
    print(f"\nWM dynamics error ||decode(WM roll) - env|| over the SAME action seq "
          f"(avg {N_STATES} states):")
    print("   seq \\ H            |" + "".join(f"  H={h}" for h in HS))
    for name in ["A_on_policy", "B_base_proposal", "C_base_noise"]:
        eh = np.mean(errs[name], 0)
        print(f"   {name:18s}|" + "".join(f"  {eh[h-1]:.3f}" for h in HS))
    mv = np.mean(moves, 0)
    print(f"   {'true move':18s}|" + "".join(f"  {mv[h-1]:.3f}" for h in HS))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
