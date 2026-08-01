"""Visualize that the value lacks LOCAL action resolution, and show why MPPI,
climbing the WM value, picks the wrong action.

Panels:
  (0,0) GROUND-TRUTH MC-return landscape over a 2-D action slice (where the good
        actions actually are).
  (0,1) WM value-return W_H=5 landscape (what MPPI ranks by) — note the peak is
        elsewhere.
  (0,2) WM value-return W_H=20 — long horizon drifts OOD -> garbage.
  (1,0) scatter: WM score vs MC truth per candidate. A flat/uncorrelated cloud =
        the value cannot tell good actions from bad.
  (1,1) ALONG the direction MPPI moves (a* -> WM-value argmax): WM value keeps
        RISING ("I'm improving") while the true MC return does NOT -> MPPI sees
        value go up and walks off the good action.
  (1,2) along a real expert rollout: V_wm, V_src, and true progress gamma^(T-t)
        all rise GLOBALLY (value has progress sense) — the failure is purely
        LOCAL action discrimination, not global progress.
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
GRID = 15
RANGE = 0.5
DIMS = (0, 1)
MC_STEPS = 30


def load_source_agent(env):
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(SOURCE), "cfg.yaml")))
    return train_rl.load_frozen_agent(SOURCE, env, cfg, "cuda")


def mc_return(env, sim0, a, policy):
    raw = env.env
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    obs, r, done, _ = raw.step(np.asarray(a, dtype=np.float64))
    if r > 0.5:
        return GAMMA
    for k in range(2, MC_STEPS + 1):
        if done:
            break
        st = torch.tensor(cf.extract_state(obs), dtype=torch.float32).cuda()
        with torch.no_grad():
            act = policy.act({"state": st}, eval_mode=True).cpu().numpy()
        obs, r, done, _ = raw.step(np.asarray(act, dtype=np.float64))
        if r > 0.5:
            return GAMMA ** k
    return 0.0


def wm_return(m, s0t, A, H):
    with torch.no_grad():
        z = m.encode(s0t).expand(A.shape[0], -1).clone()
        for _ in range(H):
            z = m.next(z, A)
        return (m.predict_value(z) * GAMMA ** H).cpu().numpy()


def main():
    agent, ws = e.build(SIMDIST, use_planner=0)
    m = agent.model
    env = cf.get_env(SOURCE, domain="source")
    src = load_source_agent(env)

    eps = cf.load_demo_episodes(DEMOS, max_episodes=30)
    ep = next(x for x in eps if x["rewards"].sum() > 0)
    T = len(ep["action"])
    t0 = max(0, T - 10)
    s0 = ep["state"][t0].astype(np.float32)
    a_star = ep["action"][t0].astype(np.float32)
    sim0 = ep["sim_state"][t0]
    dx, dy = DIMS

    xs = np.linspace(a_star[dx] - RANGE, a_star[dx] + RANGE, GRID)
    ys = np.linspace(a_star[dy] - RANGE, a_star[dy] + RANGE, GRID)
    A = np.tile(a_star, (GRID * GRID, 1))
    gx, gy = np.meshgrid(xs, ys)
    A[:, dx] = gx.reshape(-1)
    A[:, dy] = gy.reshape(-1)
    A = np.clip(A, -1, 1)
    At = torch.tensor(A, dtype=torch.float32).cuda()
    s0t = torch.tensor(s0, dtype=torch.float32).cuda().unsqueeze(0)
    N = GRID * GRID

    W5 = wm_return(m, s0t, At, 5)
    W20 = wm_return(m, s0t, At, 20)
    MC = np.array([mc_return(env, sim0, A[i], src) for i in range(N)], dtype=np.float32)

    def G(v):
        return v.reshape(GRID, GRID)

    a_mc = A[np.argmax(MC)]
    a_w5 = A[np.argmax(W5)]

    # ---- (1,1) line from a* toward the WM-value argmax (the way MPPI moves) ----
    nL = 11
    ts = np.linspace(0, 1.4, nL)   # extend past the WM argmax
    line_W5, line_MC = [], []
    for t in ts:
        a = a_star.copy()
        a[[dx, dy]] = a_star[[dx, dy]] + t * (a_w5[[dx, dy]] - a_star[[dx, dy]])
        a = np.clip(a, -1, 1)
        line_W5.append(wm_return(m, s0t, torch.tensor(a, dtype=torch.float32).cuda().unsqueeze(0), 5)[0])
        line_MC.append(mc_return(env, sim0, a, src))
    line_W5 = np.array(line_W5); line_MC = np.array(line_MC)

    # ---- (1,2) value along the real expert rollout ----
    St = torch.tensor(ep["state"], dtype=torch.float32).cuda()
    with torch.no_grad():
        Vwm_traj = m.predict_value(m.encode(St)).cpu().numpy()
        Vsrc_traj = src.value({"state": St}).cpu().numpy()
    prog = GAMMA ** (np.arange(T)[::-1])   # gamma^(steps to success)

    def norm(v):
        v = np.asarray(v, dtype=float)
        return (v - v.min()) / (v.max() - v.min() + 1e-9)

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    def heat(a, Z, title):
        im = a.imshow(G(Z), origin="lower", extent=extent, aspect="auto", cmap="viridis")
        a.plot(a_star[dx], a_star[dy], "r*", ms=18, label="a* expert")
        a.plot(a_mc[dx], a_mc[dy], "g^", ms=11, label="MC-best (true)")
        a.plot(a_w5[dx], a_w5[dy], "wo", ms=10, mfc="none", label="WM-best (W5)")
        a.set_title(title, fontsize=10); a.set_xlabel(f"action[{dx}]"); a.set_ylabel(f"action[{dy}]")
        a.legend(loc="upper right", fontsize=7); fig.colorbar(im, ax=a, fraction=0.046)

    heat(ax[0, 0], MC, "(0,0) GROUND-TRUTH  MC return  (where good actions are)")
    heat(ax[0, 1], W5, "(0,1) WM value-return  H=5  (what MPPI ranks by)")
    heat(ax[0, 2], W20, "(0,2) WM value-return  H=20  (long horizon -> OOD)")

    ax[1, 0].scatter(MC, W5, c=np.abs(A[:, [dx, dy]] - a_star[[dx, dy]]).sum(1), cmap="plasma", s=28)
    ax[1, 0].set_xlabel("true MC return"); ax[1, 0].set_ylabel("WM score W5")
    from scipy.stats import spearmanr
    rho = spearmanr(MC, W5).correlation
    ax[1, 0].set_title(f"(1,0) WM score vs truth  (Spearman={rho:+.2f})\nflat/uncorrelated = no local resolution", fontsize=10)

    ax[1, 1].plot(ts, norm(line_W5), "o-", color="crimson", label="WM value W5 (MPPI climbs this)")
    ax[1, 1].plot(ts, norm(line_MC), "s-", color="navy", label="TRUE MC return")
    ax[1, 1].axvline(1.0, ls="--", c="gray", lw=1, label="WM argmax")
    ax[1, 1].set_xlabel("move from a* (0) toward WM-value argmax (1)")
    ax[1, 1].set_ylabel("normalized")
    ax[1, 1].set_title("(1,1) along MPPI's move: WM value RISES, truth does NOT", fontsize=10)
    ax[1, 1].legend(fontsize=8)

    ax[1, 2].plot(norm(Vwm_traj), label="V_wm(s_t)")
    ax[1, 2].plot(norm(Vsrc_traj), label="V_src(s_t)")
    ax[1, 2].plot(norm(prog), "k--", label="true progress gamma^(T-t)")
    ax[1, 2].axvline(t0, ls=":", c="r", label="t0 (analyzed state)")
    ax[1, 2].set_xlabel("rollout step t"); ax[1, 2].set_ylabel("normalized")
    ax[1, 2].set_title("(1,2) along real rollout: value HAS global progress sense", fontsize=10)
    ax[1, 2].legend(fontsize=8)

    fig.suptitle("Value lacks LOCAL action resolution: MPPI climbs WM value but walks off the good action", fontsize=13)
    fig.tight_layout()
    out = "scripts/viz_value_resolution.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    print(f"a*={np.round(a_star[[dx,dy]],3)}  MC-best={np.round(a_mc[[dx,dy]],3)}  WM-best={np.round(a_w5[[dx,dy]],3)}")
    print(f"WM value along move: {np.round(norm(line_W5),2)}")
    print(f"TRUE MC   along move: {np.round(norm(line_MC),2)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
