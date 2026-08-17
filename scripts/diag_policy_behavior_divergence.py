"""Closed-loop behavioural comparison of stage-2 BC policies from IDENTICAL starts.

`diag_offmanifold_divergence.py` showed these policies agree on the demos but
disagree off-manifold by ~0.98, i.e. ~93% of the action's own across-state
spread. That is a STATIC measurement (same state fed to both). This one is
DYNAMIC: reset to the same initial state (same numpy seed) and let each policy
drive its own episode, then ask

  * WHEN do the trajectories separate (divergence vs control step)?
  * WHERE do they separate (which task phase: reach / grasp / transport / insert)?
  * WHERE do they end up (visited-state distribution, novelty vs their own demos)?
  * what are the outcomes (success, length, failure mode)?

Phases are keyed off the needle: grasp = needle lifted off the table
(needle_z > z0 + 2cm), insert = needle within `insert_thresh` of the tripod.

Run:
    python scripts/diag_policy_behavior_divergence.py --num_game 20
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
from train_simdist import MainConfig

STAGE1 = "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1"
BASES = {
    "P_orig":  ("threading_d0_target_dynamics", 20, "LOST @220k  (adapt 0.95)"),
    "P_repro": ("threading_d0_repro952_n20", 20, "snap @220k  (adapt 0.63*)"),
    "P_snapA": ("threading_d0_snapA_200000", 20, "snap @200k  (adapt 0.68)"),
}
EEF = slice(0, 3)
NEEDLE = slice(9, 12)
TRIPOD = slice(23, 26)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def bc(cfg, sd, path, n, H, steps=1000):
    b = SequenceReplayBuffer(max_episodes=20000, device="cpu")
    load_demos_to_sequence_buffer(
        b, os.path.join(_REPO, "release/data/mimicgen", path, "dataset.hdf5"), num_data=n)
    torch.manual_seed(1); np.random.seed(1)
    ag = SimDistAgent(1, (37,), None, 7, "", cfg.simdist, device="cuda")
    ag.load_state_dict(sd); ag = ag.cuda()
    for _ in range(steps):
        st, ac, _ = b.sample(256, H, "cuda")
        ag.update_bc_policy(st, ac, lr=3e-4)
    ag.eval(); ag.cfg.use_planner = 0
    demo_S = torch.cat([torch.as_tensor(e["state"][:-1]) for e in b.episodes]).float().cuda()
    return ag, demo_S


@torch.no_grad()
def rollout(ag, env, seed, max_len=400):
    np.random.seed(seed)
    obs, _ = env.reset()
    S, A, succ = [], [], 0.0
    for _ in range(max_len):
        s = obs["state"]
        S.append((s.cpu().numpy() if torch.is_tensor(s) else np.asarray(s)).copy())
        a = ag.act(obs, eval_mode=True)
        A.append(a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a))
        obs, _, term, sc, _ = env.step(a)
        succ = max(succ, float(sc))
        if term:
            break
    return np.stack(S), np.stack(A), succ > 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_game", type=int, default=20)
    p.add_argument("--seed0", type=int, default=500)
    p.add_argument("--insert_thresh", type=float, default=0.12)
    args = p.parse_args()

    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(_REPO, STAGE1, "cfg.yaml")))
    cfg.simdist.use_planner = 0
    H = cfg.simdist.world_model.horizon
    sd = torch.load(os.path.join(_REPO, STAGE1, "latest.pt"), map_location="cuda")

    env = PixelRobosuite(
        env_name=cfg.task_name, robots=["Panda"], episode_length=400,
        reward_shaping=False, image_size=cfg.image_size, rl_image_size=cfg.rl_image_size,
        camera_names=["agentview"], rl_cameras=["agentview"], use_state=True,
        obs_stack=cfg.obs_stack, state_stack=cfg.state_stack, prop_stack=cfg.prop_stack,
        strict_success=cfg.strict_success, domain_shift=cfg.eval_domain_shift,
        hand_init_perturb_joint_sigma=0.1)

    agents, demoS = {}, {}
    for lab, (path, n, who) in BASES.items():
        agents[lab], demoS[lab] = bc(cfg, sd, path, n, H)
        log(f"{lab:>8}  {who}")

    keys = list(BASES)
    trajs = {k: [] for k in keys}
    for g in range(args.num_game):
        for k in keys:
            trajs[k].append(rollout(agents[k], env, args.seed0 + g))

    log("\n=== outcome ===")
    for k in keys:
        S = trajs[k]
        log(f"  {k:>8}  success={np.mean([t[2] for t in S]):.2f}  "
            f"len={np.mean([len(t[0]) for t in S]):6.1f}  "
            f"reach_insert={np.mean([float((np.linalg.norm(t[0][:, NEEDLE]-t[0][:, TRIPOD], axis=1) < args.insert_thresh).any()) for t in S]):.2f}")

    log("\n=== 从同一初始状态出发, 轨迹分离 vs 控制步 (eef 位置, mm) ===")
    log(f"{'step':>6} " + "".join(f"{a+' vs '+b:>22}" for a, b in
        [("orig", "repro"), ("orig", "snapA"), ("repro", "snapA")]))
    for t in (0, 5, 10, 20, 30, 40, 60, 80):
        row = []
        for a, b in [("P_orig", "P_repro"), ("P_orig", "P_snapA"), ("P_repro", "P_snapA")]:
            d = [np.linalg.norm(trajs[a][g][0][t, EEF] - trajs[b][g][0][t, EEF]) * 1000
                 for g in range(args.num_game)
                 if len(trajs[a][g][0]) > t and len(trajs[b][g][0]) > t]
            row.append(f"{np.mean(d):8.2f} (n={len(d):2d})" if d else "      -      ")
        log(f"{t:>6} " + "".join(f"{r:>22}" for r in row))

    log("\n=== 分相位: 各 policy 到达每个阶段的比例 / 首次到达步 ===")
    for k in keys:
        gr, ins, grt, inst = [], [], [], []
        for S, A, sc in trajs[k]:
            z0 = S[0, NEEDLE][2]
            lifted = np.where(S[:, NEEDLE][:, 2] > z0 + 0.02)[0]
            near = np.where(np.linalg.norm(S[:, NEEDLE] - S[:, TRIPOD], axis=1) < args.insert_thresh)[0]
            gr.append(len(lifted) > 0); ins.append(len(near) > 0)
            if len(lifted): grt.append(lifted[0])
            if len(near): inst.append(near[0])
        log(f"  {k:>8}  抓起={np.mean(gr):.2f}(首次@{np.mean(grt) if grt else -1:5.1f})  "
            f"到插入相={np.mean(ins):.2f}(首次@{np.mean(inst) if inst else -1:5.1f})")

    log("\n=== 访问状态相对【自己的】demo 流形的新颖度 ===")
    with torch.no_grad():
        for k in keys:
            V = torch.as_tensor(np.concatenate([t[0] for t in trajs[k]]),
                                dtype=torch.float32, device="cuda")
            D = demoS[k]; s = D.std(0) + 1e-6
            nn = torch.cdist(V / s, D / s).min(1).values
            dd = torch.cdist(D / s, D / s); dd.fill_diagonal_(float("inf"))
            ref = float(dd.min(1).values.median())
            log(f"  {k:>8}  访问 {V.shape[0]:>6} 状态 | 到 demo 最近邻距离 中位={float(nn.median()):.3f} "
                f"(demo 自身最近邻中位={ref:.3f}, 比值={float(nn.median())/ref:.2f}x) | "
                f"超出该阈值的占比={float((nn > ref).float().mean())*100:.1f}%")
    os._exit(0)


if __name__ == "__main__":
    main()
