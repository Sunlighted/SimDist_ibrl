"""Noise floor: how different are the stage-2 policies of three replicate runs?

`long_h5`, `long_h5_repro`, `long_h5_repro2` all ran the SAME stage-1 ckpt +
the SAME 20 demos (`threading_d0_target_dynamics`) + the same seed, and reached
adapt 0.952 / 0.925 / 0.950. Their stage-2 policy heads are therefore three
samples of "same data, same seed, only GPU nondeterminism", i.e. the FLOOR for
any policy-disagreement measurement.

Compare that floor against:
  same network, different demos   (snapA_200000  vs repro952_n20)
  different network, same recipe  (long_h5       vs repro952_n20)

so the numbers from `diag_offmanifold_divergence.py` (0.773 / 0.978 off-manifold)
can be read against a calibrated zero. All heads are lifted from checkpoints
(`model.policy.*` is frozen after stage-2), so no BC re-training is involved.

Run:
    python scripts/diag_replicate_policy_spread.py --num_game 15
"""
import argparse
import itertools
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

STAGE1 = "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1"
A = "exps/mimicgen/threading_simdist_dynens_adapt_"
HEADS = {   # label -> (ckpt with the frozen stage-2 head, demos it was BC'd on)
    "R1 long_h5":       (A + "long_h5_seed1/latest.pt",        "原始 20 条"),
    "R2 long_h5_repro": (A + "long_h5_repro_seed1/latest.pt",  "原始 20 条"),
    "R3 long_h5_repro2": (A + "long_h5_repro2_seed1/latest.pt", "原始 20 条"),
    "repro952_n20":     (A + "repro952_n20_seed1/latest.pt",   "我的 220k 快照, 匹配 20 条"),
    "snapA_200000":     (A + "snapA_200000_seed1/latest.pt",   "我的 200k 快照, 随机 20 条"),
}
DEMOS = "release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_game", type=int, default=15)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(_REPO, STAGE1, "cfg.yaml")))
    cfg.simdist.use_planner = 1
    apply_original_recipe(cfg.simdist.planner)
    H = cfg.simdist.world_model.horizon
    base_sd = torch.load(os.path.join(_REPO, STAGE1, "latest.pt"), map_location="cuda")

    agent = SimDistAgent(1, (37,), None, 7, "", cfg.simdist, device="cuda")
    agent.load_state_dict(base_sd); agent = agent.cuda(); agent.eval()

    # ---- lift each frozen policy head out of its checkpoint ----
    heads = {}
    for lab, (ck, who) in HEADS.items():
        path = os.path.join(_REPO, ck)
        if not os.path.exists(path):
            log(f"  [skip] {lab}: {ck} 不存在"); continue
        sd = torch.load(path, map_location="cuda")
        heads[lab] = {k: v.clone() for k, v in sd.items() if k.startswith("model.policy.")}
        log(f"  {lab:<18} ({who})  {len(heads[lab])} 个 policy 张量")

    labs = list(heads)
    log("\n=== 权重层面: policy 头的 max|Δ| ===")
    log(f"{'':>18}" + "".join(f"{l.split()[0]:>16}" for l in labs))
    for a in labs:
        row = ""
        for b in labs:
            m = max(float((heads[a][k] - heads[b][k]).abs().max()) for k in heads[a])
            row += f"{m:>16.3e}"
        log(f"{a:>18}" + row)

    # ---- state sets ----
    buf = SequenceReplayBuffer(max_episodes=20000, device="cpu")
    load_demos_to_sequence_buffer(buf, os.path.join(_REPO, DEMOS), num_data=20)
    demo_S = torch.cat([torch.as_tensor(e["state"][:-1]) for e in buf.episodes]).float().cuda()

    env = PixelRobosuite(
        env_name=cfg.task_name, robots=["Panda"], episode_length=cfg.episode_length,
        reward_shaping=False, image_size=cfg.image_size, rl_image_size=cfg.rl_image_size,
        camera_names=["agentview"], rl_cameras=["agentview"], use_state=True,
        obs_stack=cfg.obs_stack, state_stack=cfg.state_stack, prop_stack=cfg.prop_stack,
        strict_success=cfg.strict_success, domain_shift=cfg.eval_domain_shift,
        hand_init_perturb_joint_sigma=0.1)

    def put(lab):
        cur = agent.state_dict()
        for k, v in heads[lab].items():
            cur[k] = v
        agent.load_state_dict(cur)

    put(labs[0]); agent.cfg.use_planner = 1
    log(f"\n用 {labs[0]} 的 MPPI 采集流形外状态 ({args.num_game} 局)...")
    S = []
    rs = np.random.get_state(); np.random.seed(args.seed)
    with torch.no_grad():
        for _ in range(args.num_game):
            obs, _ = env.reset(); done = False
            while not done:
                a = agent.act(obs, eval_mode=True)
                s = obs["state"]
                S.append((s.cpu().numpy() if torch.is_tensor(s) else np.asarray(s)).copy())
                obs, _, done, _, _ = env.step(a); done = bool(done)
    np.random.set_state(rs)
    off_S = torch.as_tensor(np.stack(S), dtype=torch.float32, device="cuda")
    log(f"  demo 状态 {demo_S.shape[0]}, 流形外状态 {off_S.shape[0]}")

    # ---- action disagreement ----
    with torch.no_grad():
        acts = {}
        for lab in labs:
            put(lab)
            acts[lab] = (agent.model.policy_action(demo_S), agent.model.policy_action(off_S))
        spread = float(np.mean([float((acts[l][1] - acts[l][1].mean(0)).norm(dim=1).mean()) for l in labs]))
        log(f"\n=== 动作分歧 mean||a_i - a_j|| (参考: 流形外动作自身跨状态散布 = {spread:.4f}) ===")
        log(f"{'pair':<40}{'demo 上':>12}{'流形外':>12}")
        for a, b in itertools.combinations(labs, 2):
            same_demos = (HEADS[a][1] == HEADS[b][1])
            tag = "★同数据同seed(噪声地板)" if same_demos else ""
            d0 = float((acts[a][0] - acts[b][0]).norm(dim=1).mean())
            d1 = float((acts[a][1] - acts[b][1]).norm(dim=1).mean())
            log(f"{a.split()[0]+' vs '+b.split()[0]:<40}{d0:>12.4f}{d1:>12.4f}  {tag}")
    os._exit(0)


if __name__ == "__main__":
    main()
