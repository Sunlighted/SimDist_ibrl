"""Do two stage-2 BC policies that agree ON THE DEMOS diverge OFF-manifold?

Context (CLAUDE.md 35): `threading_d0_target_dynamics` (20 demos from a LOST
checkpoint) adapts to 0.93-0.95 across 3 replicates, while demos collected from
the *same recipe at the same training step* -- statistically indistinguishable on
~12 measures (len distribution incl. std, contact_frac, |da|, per-dim |a|,
saturation, gripper behaviour, grasp geometry, set-level effective dimension,
pairwise trajectory distance, BC loss) -- adapt to only 0.63-0.80.

All those measures are computed ON the demos. But `novel_frac` is 0.9986-1.0000
for every base: essentially 100% of the states stage-3 collects lie OUTSIDE the
demo manifold. Two policies can fit the same 20 sparse trajectories equally well
(both BC losses ~0.005-0.006) and still behave completely differently away from
them -- and away from them is where all of the adaptation data comes from. No
joint training is needed for that; 20 trajectories are simply a very sparse
constraint on a 37-dim state space.

This measures policy-pair disagreement on three state sets:
    demo_orig    the original 20 demos' states        (in-manifold for P_orig)
    demo_repro   the repro952 20 demos' states        (in-manifold for P_repro)
    offmanifold  states visited by MPPI collection    (where stage-3 gets data)

and reports it relative to the actions' own across-state spread, so "big" means
big compared to how much the policy varies at all.

CONTROL: two policies trained on DIFFERENT demo sets from the SAME network
(repro952_n20 vs snapA_200000). If cross-network disagreement is no larger than
this same-network baseline, the network identity is NOT the carrier.

Also reports roll_drift: ||imagined H-step latent - encode(true s_{t+H})||, the
coherence of the policy's own rollout, which is what seeds the MPPI base
candidates (`_policy_rollout` queries the head H times recursively).

Usage:
    python scripts/diag_offmanifold_divergence.py --num_game 15
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

STAGE1 = "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1"
BASES = {  # label -> (demo dir, n, which network collected it)
    "P_orig":  ("threading_d0_target_dynamics", 20, "LOST ckpt @220k"),
    "P_repro": ("threading_d0_repro952_n20", 20, "snap ckpt @220k"),
    "P_snapA": ("threading_d0_snapA_200000", 20, "snap ckpt @200k"),
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_buf(path, n):
    b = SequenceReplayBuffer(max_episodes=20000, device="cpu")
    load_demos_to_sequence_buffer(b, os.path.join(_REPO, "release/data/mimicgen", path,
                                                  "dataset.hdf5"), num_data=n)
    return b


def bc_train(cfg, sd, buf, H, steps=1000):
    torch.manual_seed(1); np.random.seed(1)
    ag = SimDistAgent(1, (37,), None, 7, "", cfg.simdist, device="cuda")
    ag.load_state_dict(sd); ag = ag.cuda()
    for _ in range(steps):
        st, ac, _ = buf.sample(256, H, "cuda")
        m = ag.update_bc_policy(st, ac, lr=3e-4)
    ag.eval()
    return ag, m["bc/policy_loss"]


@torch.no_grad()
def collect_offmanifold(ag, env, num_game, seed):
    ag.cfg.use_planner = 1
    rs = np.random.get_state(); np.random.seed(seed)
    S = []
    for _ in range(num_game):
        obs, _ = env.reset()
        done = False
        while not done:
            a = ag.act(obs, eval_mode=True)
            s = obs["state"]
            S.append((s.cpu().numpy() if torch.is_tensor(s) else np.asarray(s)).copy())
            obs, _, done, _, _ = env.step(a)
            done = bool(done)
    np.random.set_state(rs)
    return torch.as_tensor(np.stack(S), dtype=torch.float32, device="cuda")


@torch.no_grad()
def flat_states(buf):
    return torch.cat([torch.as_tensor(e["state"][:-1]) for e in buf.episodes]).float().cuda()


@torch.no_grad()
def roll_drift(ag, buf, H):
    st, _, _ = buf.sample(256, H, "cuda")
    m = ag.model
    z = m.encode(st[:, 0])
    for _ in range(H):
        z = m.next(z, m.policy_from_latent(z) if getattr(m.cfg, "shared_encoder", 0)
                   else m.policy_action(m.decode(z)))
    return float((z - m.encode(st[:, H])).norm(dim=1).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_game", type=int, default=15)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(_REPO, STAGE1, "cfg.yaml")))
    cfg.simdist.use_planner = 1
    apply_original_recipe(cfg.simdist.planner)
    H = cfg.simdist.world_model.horizon
    sd = torch.load(os.path.join(_REPO, STAGE1, "latest.pt"), map_location="cuda")

    env = PixelRobosuite(
        env_name=cfg.task_name, robots=["Panda"], episode_length=cfg.episode_length,
        reward_shaping=False, image_size=cfg.image_size, rl_image_size=cfg.rl_image_size,
        camera_names=["agentview"], rl_cameras=["agentview"], use_state=True,
        obs_stack=cfg.obs_stack, state_stack=cfg.state_stack, prop_stack=cfg.prop_stack,
        strict_success=cfg.strict_success, domain_shift=cfg.eval_domain_shift,
        hand_init_perturb_joint_sigma=0.1)

    agents, bufs = {}, {}
    for lab, (path, n, who) in BASES.items():
        bufs[lab] = load_buf(path, n)
        agents[lab], bl = bc_train(cfg, sd, bufs[lab], H)
        log(f"{lab:>8} ({who:<15}) BC loss={bl:.5f}  roll_drift={roll_drift(agents[lab], bufs[lab], H):.4f}")

    # state sets
    sets = {f"demo_{k}": flat_states(bufs[k]) for k in BASES}
    log(f"\ncollecting off-manifold states with P_orig's MPPI ({args.num_game} games)...")
    sets["offmanifold"] = collect_offmanifold(agents["P_orig"], env, args.num_game, args.seed)
    for k, v in sets.items():
        log(f"  {k:<14} {v.shape[0]:>6} states")

    log("\n=== policy-pair disagreement:  mean||a_i(s) - a_j(s)||  (7-dim action) ===")
    log(f"{'state set':<14} " + "".join(f"{k:>26}" for k in
        ["orig vs repro (跨网络)", "repro vs snapA (同网络)"]) + f"{'a 自身跨状态散布':>20}")
    with torch.no_grad():
        for sk, S in sets.items():
            A = {k: agents[k].model.policy_action(S) for k in BASES}
            d_cross = float((A["P_orig"] - A["P_repro"]).norm(dim=1).mean())
            d_same = float((A["P_repro"] - A["P_snapA"]).norm(dim=1).mean())
            spread = float(np.mean([float((A[k] - A[k].mean(0)).norm(dim=1).mean()) for k in BASES]))
            log(f"{sk:<14} {d_cross:>26.4f}{d_same:>26.4f}{spread:>20.4f}")
    log("\n判据: 若 offmanifold 上的分歧 >> demo 上的分歧, 且 跨网络 >> 同网络,")
    log("      则 carrier = 稀疏 demo 未约束住的流形外行为, 且与生成网络绑定。")
    os._exit(0)


if __name__ == "__main__":
    main()
