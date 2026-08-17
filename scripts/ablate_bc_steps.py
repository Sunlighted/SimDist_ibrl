"""bc_steps ablation: is stage-2 BC over/under-trained at the fixed 1000 steps?

`train_simdist_adapt.py` uses a FIXED `bc_steps` (1000) for every demo set, but
the sets differ in transition count by 3x (1327 .. 4150), so the number of
EPOCHS varies 62..193. Measured across the 8 snapshot arms, closed-loop
`policy_only` is an inverted U in epochs (62 -> 0.03, 115-144 -> 0.42-0.47,
193 -> 0.13) while the BC train loss keeps falling -- i.e. the fixed step count
under-trains the long-demo sets and over-trains the short-demo ones.

This trains ONE policy incrementally and evaluates `policy_only` (mp eval,
sigma=0.1 + strict, TARGET shift -- the same protocol the adapt runs log) at a
series of step counts, so the whole curve costs one BC run plus K evals.

Usage:
    python scripts/ablate_bc_steps.py --demos threading_d0_snapB_400000 --n 20 \
        --steps 150,300,600,1000,2000,4000 --num_game 100
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
import torch.nn.functional as F

import common_utils  # noqa: F401
from evaluate import run_eval_mp
from rl.sequence_replay import SequenceReplayBuffer, load_demos_to_sequence_buffer
from rl.simdist_agent import SimDistAgent
from train_simdist import MainConfig

STAGE1 = "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--steps", default="150,300,600,1000,2000,4000")
    p.add_argument("--num_game", type=int, default=100)
    p.add_argument("--num_proc", type=int, default=5)
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="episode-level held-out split for a val loss (0 = off)")
    p.add_argument("--bc_lr", type=float, default=3e-4,
                   help="stage-2 BC lr. NOTE `update_bc_policy` builds its Adam LAZILY "
                        "on the first call, so the lr is fixed there and later values are "
                        "ignored -- one lr per process.")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    checkpoints = sorted(int(s) for s in args.steps.split(","))

    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(_REPO, STAGE1, "cfg.yaml")))
    cfg.simdist.use_planner = 0          # policy_only
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    H = cfg.simdist.world_model.horizon
    sd = torch.load(os.path.join(_REPO, STAGE1, "latest.pt"), map_location="cuda")

    full = SequenceReplayBuffer(max_episodes=20000, device="cpu")
    load_demos_to_sequence_buffer(
        full, os.path.join(_REPO, "release/data/mimicgen", args.demos, "dataset.hdf5"),
        num_data=args.n)
    eps = list(full.episodes)
    n_val = int(round(len(eps) * args.val_frac))
    if n_val:
        # only shuffle when we actually hold episodes out -- with val_frac=0 the
        # episode ORDER must stay as loaded so the BC sampling sequence matches
        # `train_simdist_adapt.py::bc_finetune` (same seed + batch size + data
        # => same trajectory), which is what makes a chosen step count transfer.
        np.random.RandomState(0).shuffle(eps)
    val_eps, tr_eps = eps[:n_val], eps[n_val:]
    tr = SequenceReplayBuffer(max_episodes=20000, device="cpu"); tr.episodes = tr_eps
    ntr = sum(e["action"].shape[0] for e in tr_eps)
    log(f"[{args.demos}] train {len(tr_eps)} ep / {ntr} trans, val {len(val_eps)} ep")

    def flat(ep_list):
        if not ep_list:
            return None, None
        S = torch.cat([torch.as_tensor(e["state"][:-1]) for e in ep_list]).float().cuda()
        A = torch.cat([torch.as_tensor(e["action"]) for e in ep_list]).float().cuda()
        return S, A
    Str, Atr = flat(tr_eps)
    Sv, Av = flat(val_eps)

    env_params = dict(
        env_name=cfg.task_name, robots=["Panda"], episode_length=cfg.episode_length,
        reward_shaping=False, image_size=cfg.image_size, rl_image_size=cfg.rl_image_size,
        camera_names=[], rl_cameras=[], use_state=True, obs_stack=cfg.obs_stack,
        state_stack=cfg.state_stack, prop_stack=cfg.prop_stack,
        strict_success=cfg.strict_success, domain_shift=cfg.eval_domain_shift,
        hand_init_perturb_joint_sigma=0.1)

    torch.manual_seed(1); np.random.seed(1)
    ag = SimDistAgent(1, (37,), None, 7, "", cfg.simdist, device="cuda")
    ag.load_state_dict(sd); ag = ag.cuda()

    log(f"\n[lr={args.bc_lr:g}]  {'steps':>7}{'epochs':>8}{'train':>10}{'val':>10}{'policy_only':>13}")
    done = 0
    for ck in checkpoints:
        while done < ck:
            st, ac, _ = tr.sample(256, H, "cuda")
            ag.update_bc_policy(st, ac, lr=args.bc_lr)
            done += 1
        ag.eval()
        with torch.no_grad():
            trl = float(F.mse_loss(ag.model.policy_action(Str), Atr))
            vl = float(F.mse_loss(ag.model.policy_action(Sv), Av)) if Sv is not None else float("nan")
        rs = np.random.get_state()
        try:
            sc = run_eval_mp(env_params=env_params, agent=ag, num_proc=args.num_proc,
                             num_game=args.num_game, seed=args.seed, verbose=False)
        finally:
            np.random.set_state(rs)
        ag.train()
        log(f"{ck:>7}{ck*256/ntr:>8.0f}{trl:>10.5f}{vl:>10.5f}{float(np.mean(sc)):>13.3f}")
    os._exit(0)


if __name__ == "__main__":
    main()
