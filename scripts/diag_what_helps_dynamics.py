"""WHICH collection behaviours help the stage-3 dynamics adapt?

Derived from what stage-3 actually does, not from scanning:
`update_dynamics_only` trains ONLY the dynamics head on a multi-step latent
consistency loss over segments of the MPPI-collected buffer. MPPI then QUERIES
that dynamics at (z_t, candidate action sequences), where the candidates are
100 policy-rollout+N(0,0.1) plus 250 ~N(mean, init_std=1.0) -- i.e. nearly the
whole action cube on the first CEM iteration.

That asymmetry (collection executes ONE action per state; the planner evaluates
hundreds there) gives two mechanistically-motivated requirements:

  A  contact-phase data must exist          -> contact_steps
     (CLAUDE.md 27: the ceiling is set by contact-phase model error)
  B  varied actions near the SAME state     -> act_spread_local   <-- never measured
     (otherwise the dynamics extrapolates over most of the candidate set, and
      argmax specifically seeks out those extrapolation errors)
  C  executed-action range overlaps the candidate range -> act_range_contact
  D  control, expected flat                 -> novel_frac

PRE-SPECIFIED: these four only. No scanning. Reported with n and the
multiple-comparison caveat (this project has been burned by a 9-metrics-on-6-
points fishing expedition before, CLAUDE.md 35).

Usage:
    python scripts/diag_what_helps_dynamics.py --bases <name>[,<name>...] --num_game 20
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
NEEDLE = slice(9, 12)
TRIPOD = slice(23, 26)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


@torch.no_grad()
def measure(ag, env, num_game, seed, thresh, demo_S, demo_nn, knn=8):
    """Roll MPPI, then compute the four pre-specified quantities."""
    ag.cfg.use_planner = 1
    rs = np.random.get_state(); np.random.seed(seed)
    S, A, C, lens = [], [], [], []
    for _ in range(num_game):
        obs, _ = env.reset(); n = 0; done = False
        while not done:
            a = ag.act(obs, eval_mode=True)
            s = obs["state"]
            s = s.cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
            S.append(s.copy())
            A.append(a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a))
            C.append(np.linalg.norm(s[NEEDLE] - s[TRIPOD]) < thresh)
            obs, _, done, _, _ = env.step(a); n += 1; done = bool(done)
        lens.append(n)
    np.random.set_state(rs)
    S = np.stack(S); A = np.stack(A); C = np.asarray(C)

    # A: contact-phase steps per episode
    contact_steps = float(C.sum() / num_game)

    # C: executed-action range within the contact phase (per-dim std, averaged)
    act_range = float(A[C].std(0).mean()) if C.sum() > 10 else float("nan")

    # B: action spread among STATE-NEIGHBOURS inside the contact phase --
    #    "does the dynamics see varied actions where MPPI will query varied ones"
    if C.sum() > knn + 1:
        Sc = torch.as_tensor(S[C], dtype=torch.float32, device="cuda")
        Ac = torch.as_tensor(A[C], dtype=torch.float32, device="cuda")
        sd = Sc.std(0) + 1e-6
        idx = torch.cdist(Sc / sd, Sc / sd).topk(knn + 1, largest=False).indices[:, 1:]
        # mean ||a_i - a_j|| over each point's knn state-neighbours
        act_spread_local = float((Ac[idx] - Ac[:, None]).norm(dim=2).mean())
    else:
        act_spread_local = float("nan")

    # D: control -- novelty of collected states vs the demo manifold
    V = torch.as_tensor(S, dtype=torch.float32, device="cuda")
    sd = demo_S.std(0) + 1e-6
    nn = torch.cdist(V / sd, demo_S / sd).min(1).values
    novel_frac = float((nn > demo_nn).float().mean())

    return dict(contact_steps=contact_steps, act_spread_local=act_spread_local,
                act_range_contact=act_range, novel_frac=novel_frac,
                ep_len=float(np.mean(lens)), n_contact=int(C.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bases", required=True,
                   help="comma-separated: <demo_dir>:<n>:<adapt>  (adapt = known outcome)")
    p.add_argument("--num_game", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--contact_thresh", type=float, default=0.12)
    p.add_argument("--bc_steps", type=int, default=1000)
    args = p.parse_args()

    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(_REPO, STAGE1, "cfg.yaml")))
    cfg.simdist.use_planner = 1
    apply_original_recipe(cfg.simdist.planner)
    H = cfg.simdist.world_model.horizon
    sd = torch.load(os.path.join(_REPO, STAGE1, "latest.pt"), map_location="cuda")

    # common demo manifold reference (the sigma=0.1 pool), so novel_frac is comparable
    ref = SequenceReplayBuffer(max_episodes=20000, device="cpu")
    load_demos_to_sequence_buffer(ref, os.path.join(
        _REPO, "release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5"), num_data=100)
    demo_S = torch.cat([torch.as_tensor(e["state"][:-1]) for e in ref.episodes]).float().cuda()
    s = demo_S.std(0) + 1e-6
    dd = torch.cdist(demo_S / s, demo_S / s); dd.fill_diagonal_(float("inf"))
    demo_nn = float(dd.min(1).values.median())

    env = PixelRobosuite(
        env_name=cfg.task_name, robots=["Panda"], episode_length=cfg.episode_length,
        reward_shaping=False, image_size=cfg.image_size, rl_image_size=cfg.rl_image_size,
        camera_names=["agentview"], rl_cameras=["agentview"], use_state=True,
        obs_stack=cfg.obs_stack, state_stack=cfg.state_stack, prop_stack=cfg.prop_stack,
        strict_success=cfg.strict_success, domain_shift=cfg.eval_domain_shift,
        hand_init_perturb_joint_sigma=0.1)

    for spec in args.bases.split(","):
        path, n, adapt = spec.split(":")
        torch.manual_seed(1); np.random.seed(1)
        ag = SimDistAgent(1, (37,), None, 7, "", cfg.simdist, device="cuda")
        ag.load_state_dict(sd); ag = ag.cuda()
        buf = SequenceReplayBuffer(max_episodes=20000, device="cpu")
        load_demos_to_sequence_buffer(
            buf, os.path.join(_REPO, "release/data/mimicgen", path, "dataset.hdf5"), num_data=int(n))
        for _ in range(args.bc_steps):
            st, ac, _ = buf.sample(256, H, "cuda")
            ag.update_bc_policy(st, ac, lr=3e-4)
        ag.eval()
        r = measure(ag, env, args.num_game, args.seed, args.contact_thresh, demo_S, demo_nn)
        log(f"WHAT {path}:{n} adapt={adapt} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()))
        del ag, buf
        torch.cuda.empty_cache()
    os._exit(0)


if __name__ == "__main__":
    main()
