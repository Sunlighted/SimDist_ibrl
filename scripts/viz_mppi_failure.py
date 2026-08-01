"""Visualize WHY value-based MPPI fails: scan a 2-D action slice at a real
insertion state and compare the world model's scoring landscape against the
ground truth, decomposing the error into dynamics vs value.

Panels (heatmaps over a 2-D action grid centered on the expert action a*):
  1. Q_src(s0, a)            GROUND-TRUTH action-value (source critic). The
                             landscape MPPI *should* climb; peak ~ a*.
  2. WM return H=1           gamma * V_wm(next(z0,a))           (what MPPI ranks by)
  3. WM return H=10          gamma^10 * V_wm(rollout_10(z0,a))
  4. dynamics error          ||decode(next(z0,a)) - s1_true(a)|| (env counterfactual)
  5. V_wm(encode(s1_true))   value head on the TRUE next state  (isolates value)
  6. V_src(s1_true)          GROUND-TRUTH value of the true next state

Reading it:
  - 1 vs 2: does the WM score peak where the GT action-value peaks? If not, the
    scoring is wrong -> then 4/5 say whether it's dynamics or value.
  - 4: if dynamics error is large/structured off a*, next(z0,a) is OOD-wrong for
    perturbed actions -> the H=1 value-return is scored on garbage latents.
  - 5 vs 6: if these agree, the value head is fine on TRUE states (error is
    dynamics); if they differ, the value head itself is wrong.
  - 3 vs 2: does a longer horizon align the landscape with GT (why multistep helps)?
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
GAMMA = 0.99
GRID = 25
RANGE = 0.6   # scan a*[d] +- RANGE
DIMS = (0, 1)  # action dims to scan (eef dx, dy)


def load_source_agent(env):
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(SOURCE), "cfg.yaml")))
    return train_rl.load_frozen_agent(SOURCE, env, cfg, "cuda")


def main():
    agent, ws = e.build(SIMDIST, use_planner=0)
    m = agent.model
    env = cf.get_env(SOURCE, domain="source")
    src = load_source_agent(env)

    eps = cf.load_demo_episodes(DEMOS, max_episodes=30)
    # pick a successful episode + an insertion-phase step (near the end)
    ep = next(e_ for e_ in eps if e_["rewards"].sum() > 0)
    T = len(ep["action"])
    t0 = max(0, T - 10)   # ~insertion phase
    s0 = ep["state"][t0].astype(np.float32)
    a_star = ep["action"][t0].astype(np.float32)
    sim0 = ep["sim_state"][t0]
    print(f"episode T={T}, t0={t0}, a*={np.round(a_star,3)}")

    dx, dy = DIMS
    xs = np.linspace(a_star[dx] - RANGE, a_star[dx] + RANGE, GRID)
    ys = np.linspace(a_star[dy] - RANGE, a_star[dy] + RANGE, GRID)
    # build the [GRID*GRID, act] action grid
    A = np.tile(a_star, (GRID * GRID, 1))
    gx, gy = np.meshgrid(xs, ys)             # [GRID,GRID]
    A[:, dx] = gx.reshape(-1)
    A[:, dy] = gy.reshape(-1)
    A = np.clip(A, -1, 1)
    At = torch.tensor(A, dtype=torch.float32).cuda()
    s0t = torch.tensor(s0, dtype=torch.float32).cuda().unsqueeze(0)

    with torch.no_grad():
        N = GRID * GRID
        # --- world model scores ---
        z0 = m.encode(s0t).expand(N, -1).clone()
        z1 = m.next(z0, At)
        W1 = (m.predict_value(z1) * GAMMA).cpu().numpy()
        z = z0.clone()
        for _ in range(10):
            z = m.next(z, At)
        W10 = (m.predict_value(z) * GAMMA ** 10).cpu().numpy()
        decode_next = m.decode(z1).cpu().numpy()   # predicted next state (H=1)
        # --- ground-truth action-value (source critic) ---
        s0_rep = s0t.expand(N, -1)
        Qsrc = src.critic.forward_k(s0_rep, At).min(-1)[0].cpu().numpy()

    # --- ground-truth next state via env counterfactual (per grid point) ---
    s1_true = np.zeros((N, s0.shape[0]), dtype=np.float32)
    r_true = np.zeros(N, dtype=np.float32)
    for i in range(N):
        s1_true[i], r_true[i], _ = cf.counterfactual_step(env, sim0, A[i])
    s1t = torch.tensor(s1_true).cuda()
    with torch.no_grad():
        dyn_err = np.linalg.norm(decode_next - s1_true, axis=1)
        val_on_true = m.predict_value(m.encode(s1t)).cpu().numpy()     # WM value head on TRUE state
        Vsrc_true = src.value({"state": s1t}).cpu().numpy()            # GT value of true next state

    def grid(v):
        return v.reshape(GRID, GRID)

    panels = [
        ("1. GT action-value  Q_src(s0,a)", grid(Qsrc)),
        ("2. WM return  H=1", grid(W1)),
        ("3. WM return  H=10", grid(W10)),
        ("4. dynamics error ||decode(next(z0,a))-s1_true||", grid(dyn_err)),
        ("5. WM value on TRUE next state  V_wm(enc(s1_true))", grid(val_on_true)),
        ("6. GT value of true next state  V_src(s1_true)", grid(Vsrc_true)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    for ax, (title, Z) in zip(axes.ravel(), panels):
        im = ax.imshow(Z, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        # mark a* and argmax
        ax.plot(a_star[dx], a_star[dy], "r*", ms=16, label="a* (expert)")
        ai, aj = np.unravel_index(np.argmax(Z) if "error" not in title else np.argmin(Z), Z.shape)
        ax.plot(xs[aj], ys[ai], "wo", ms=8, mfc="none", label="argmax" if "error" not in title else "argmin")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"action[{dx}]"); ax.set_ylabel(f"action[{dy}]")
        ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"value-MPPI landscape vs ground truth @ insertion state (t0={t0}/{T})", fontsize=13)
    fig.tight_layout()
    out = "scripts/viz_mppi_failure.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")

    # quantitative summary
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    print(f"\n[scoring vs GT action-value]  corr(W1, Qsrc)  = {corr(W1, Qsrc):+.3f}")
    print(f"[scoring vs GT action-value]  corr(W10, Qsrc) = {corr(W10, Qsrc):+.3f}")
    print(f"[value head on TRUE state]    corr(V_wm_true, V_src_true) = {corr(val_on_true, Vsrc_true):+.3f}")
    print(f"[dynamics]  mean ||decode(next)-s1_true|| = {dyn_err.mean():.4f}  "
          f"(at a*: {dyn_err[np.argmin(np.abs(A-a_star).sum(1))]:.4f})")
    amax_W1 = A[np.argmax(W1)]; amax_Q = A[np.argmax(Qsrc)]
    print(f"[argmax]  a*={np.round(a_star[[dx,dy]],3)}  "
          f"Q_src peak={np.round(amax_Q[[dx,dy]],3)}  W1 peak={np.round(amax_W1[[dx,dy]],3)}  "
          f"W10 peak={np.round(A[np.argmax(W10)][[dx,dy]],3)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
