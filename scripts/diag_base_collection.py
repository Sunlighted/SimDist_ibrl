"""What DATA does each stage-2 base policy make stage-3 collect?

scripts/diag_base_policy_quality.py measured the policy in isolation (intrinsic
properties + policy-only rollouts) and found NOTHING that predicts the adapt
outcome (max |rho| = 0.60 over 9 metrics, n=6 -- pure noise). But stage-3 does
not roll the policy: it rolls MPPI, in which the policy only supplies 100 of 350
candidates. So the object to measure is the MPPI-with-this-proposal rollout
distribution, on the UNADAPTED stage-1 dynamics (= round 0 of stage-3).

Per base policy, roll MPPI in the TARGET env and report what the dynamics
adapter would be fed:
    mppi_succ      round-0 MPPI success (cross-check vs the run's logged round 0)
    ep_len         mean episode length  (short = succeeds fast = less data/ep)
    contact_steps  steps per episode with needle->tripod < contact_thresh
    contact_frac   that as a fraction of the episode
    cov_all        sqrt(det cov) of needle->tripod rel. pose over ALL frames
    cov_contact    same over CONTACT frames only  (insertion-phase coverage)
    novel_frac     fraction of collected states whose nearest neighbour among the
                   TARGET demo states is farther than the demo set's own median
                   NN distance -- i.e. how much of the collected data is OUTSIDE
                   the demo manifold (where the frozen value head is least
                   reliable, CLAUDE.md 33)
    base_elite     mean fraction of the 64 CEM elites that came from the policy-
                   seeded candidates (how much the planner listens to it)

Usage:
    python scripts/diag_base_collection.py --num_game 25
"""
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pyrallis
import torch

import common_utils  # noqa: F401
from env.robosuite_wrapper import PixelRobosuite
from rl.sequence_replay import SequenceReplayBuffer, load_demos_to_sequence_buffer
from rl.simdist_agent import SimDistAgent
from rl.simdist_planner import apply_original_recipe
from train_simdist import MainConfig
from scripts.diag_base_policy_quality import BASES, STAGE1, HOLDOUT, NEEDLE, TRIPOD, swap_policy


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def rollout_mppi(agent, env, num_game, seed, thresh, demo_S, demo_nn):
    pl = agent.planner
    pl.record_origin = True
    agent.cfg.use_planner = 1
    rs = np.random.get_state(); np.random.seed(seed)
    succ, lens, csteps, allf, contf, states, elite = [], [], [], [], [], [], []
    for _ in range(num_game):
        obs, _ = env.reset()
        pl._origin_records.clear()
        s, done, n, c = 0.0, False, 0, 0
        while not done:
            a = agent.act(obs, eval_mode=True)
            st = obs["state"]
            st = st.cpu().numpy() if torch.is_tensor(st) else np.asarray(st)
            d = st[NEEDLE] - st[TRIPOD]
            allf.append(d); states.append(st)
            if np.linalg.norm(d) < thresh:
                contf.append(d); c += 1
            obs, _, terminal, sc, _ = env.step(a)
            s = max(s, float(sc)); n += 1; done = bool(terminal)
        succ.append(s > 0.5); lens.append(n); csteps.append(c)
        elite.extend([r[-1][0] for r in pl._origin_records if r])
    pl.record_origin = False
    np.random.set_state(rs)

    def vol(x):
        x = np.asarray(x)
        return float(np.sqrt(max(np.linalg.det(np.cov(x.T)), 0)) * 1e6) if len(x) > 10 else 0.0

    S = torch.as_tensor(np.stack(states), dtype=torch.float32, device="cuda")
    sd = demo_S.std(0) + 1e-6
    nn = torch.cdist(S / sd, demo_S / sd).min(dim=1).values
    novel = float((nn > demo_nn).float().mean())

    return dict(mppi_succ=float(np.mean(succ)), ep_len=float(np.mean(lens)),
                contact_steps=float(np.mean(csteps)),
                contact_frac=float(np.mean(csteps) / np.mean(lens)),
                cov_all=vol(allf), cov_contact=vol(contf),
                novel_frac=novel, base_elite=float(np.mean(elite)) if elite else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_game", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--contact_thresh", type=float, default=0.12)
    p.add_argument("--only", default="")
    args = p.parse_args()

    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(_REPO, STAGE1, "cfg.yaml")))
    cfg.simdist.use_planner = 1
    apply_original_recipe(cfg.simdist.planner)      # the recipe stage-3 actually uses

    buf = SequenceReplayBuffer(max_episodes=10000, device="cpu")
    load_demos_to_sequence_buffer(buf, os.path.join(_REPO, HOLDOUT), num_data=100)
    demo_S = torch.cat([e["state"][:-1] for e in buf.episodes]).float().cuda()
    sd = demo_S.std(0) + 1e-6
    dd = torch.cdist(demo_S / sd, demo_S / sd)
    dd.fill_diagonal_(float("inf"))
    demo_nn = float(dd.min(dim=1).values.median())
    log(f"[manifold] {demo_S.shape[0]} demo states, median NN dist = {demo_nn:.4f}")

    env = PixelRobosuite(
        env_name=cfg.task_name, robots=["Panda"], episode_length=cfg.episode_length,
        reward_shaping=False, image_size=cfg.image_size, rl_image_size=cfg.rl_image_size,
        camera_names=["agentview"], rl_cameras=["agentview"], use_state=True,
        obs_stack=cfg.obs_stack, state_stack=cfg.state_stack, prop_stack=cfg.prop_stack,
        strict_success=cfg.strict_success, domain_shift=cfg.eval_domain_shift,
        hand_init_perturb_joint_sigma=0.1,
    )

    agent = SimDistAgent(1, (37,), None, 7, "", cfg.simdist, device="cuda")
    agent.load_state_dict(torch.load(os.path.join(_REPO, STAGE1, "latest.pt"), map_location="cuda"))
    agent = agent.cuda()

    rows = {}
    for name in (args.only.split(",") if args.only else list(BASES)):
        swap_policy(agent, BASES[name][0])
        agent.eval()
        r = dict(adapt=BASES[name][1])
        r.update(rollout_mppi(agent, env, args.num_game, args.seed,
                              args.contact_thresh, demo_S, demo_nn))
        rows[name] = r
        log(f"{name:>14} " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))

    keys = ["adapt", "mppi_succ", "ep_len", "contact_steps", "contact_frac",
            "cov_all", "cov_contact", "novel_frac", "base_elite"]
    log("\n" + "=" * 110)
    log(f"{'base':>14} " + "".join(f"{k:>13}" for k in keys))
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["adapt"]):
        log(f"{n:>14} " + "".join(f"{r[k]:>13.4f}" for k in keys))
    from scipy.stats import spearmanr
    y = [r["adapt"] for r in rows.values()]
    log(f"\nSpearman vs adapt outcome (n={len(rows)}):")
    for k in keys[1:]:
        rho, pv = spearmanr([r[k] for r in rows.values()], y)
        log(f"  {k:>13}: rho={rho:+.3f}  p={pv:.3f}")
    os._exit(0)


if __name__ == "__main__":
    main()
