"""What makes a stage-2 BC policy a GOOD base for SimDist stage-3 adaptation?

Standalone success rate does not predict the adapt outcome (measured: a 0.59
policy adapts to 0.770 while a 0.39 policy adapts to 0.952). This probes a
battery of other policy properties and lines them up against the known adapt
outcomes, to find one that does.

The stage-2 policy head is FROZEN after stage-2, so it can be lifted verbatim
out of any checkpoint of a run (`model.policy.*`) and dropped onto the shared
stage-1 encoder (byte-identical across all these runs -- verified). No BC replay
needed, no re-training, no seed drift.

Two measurement groups:

  INTRINSIC (no env, on a common held-out target-demo state set)
    act_mse       MSE vs the demo action                (accuracy)
    jac           ||da/ds||_F, finite-diff              (sharpness / jaggedness)
    a_disp        ||a(s) - mean_s a(s)||                (does it react to state,
                                                         or has it collapsed to
                                                         a near-constant action?)
    roll_drift    ||imagined H-step latent - encode(true s_{t+H})||
                                                        (is its own rollout, which
                                                         seeds the MPPI candidates,
                                                         self-consistent?)

  CLOSED-LOOP (policy-only rollouts in the TARGET env; sigma=0.1 + strict)
    success       sanity-check against the run's logged policy_only
    reach_frac    fraction of episodes that ever bring the needle within
                  `contact_thresh` of the tripod (i.e. get to the insertion phase)
    contact_steps mean steps per episode inside that band
    contact_cov   sqrt(det(cov)) of the needle->tripod relative pose over ALL
                  contact-phase frames -- the *volume* of insertion-phase
                  configurations this policy would feed the dynamics adapter
    jerk          mean |a_t - a_{t-1}| of the EXECUTED actions

Usage:
    python scripts/diag_base_policy_quality.py --num_game 40
    python scripts/diag_base_policy_quality.py --only "sigma0.02,matched_n50"
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
HOLDOUT = "release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5"

# base policy -> (checkpoint holding its frozen stage-2 policy head, adapt outcome
# over the 50-100k window of ITS OWN run; None = not comparable at that budget)
BASES = {
    "sigma0.02_n20": ("exps/mimicgen/threading_simdist_dynens_adapt_long_h5_seed1/latest.pt", 0.952),
    "matched_n50":   ("exps/mimicgen/threading_simdist_dynens_adapt_matchn50_seed1/latest.pt", 0.925),
    "matched_n20":   ("exps/mimicgen/threading_simdist_dynens_adapt_matchn20_seed1/latest.pt", 0.897),
    "prefix_n20":    ("exps/mimicgen/threading_simdist_dynens_adapt_n20_seed1/latest.pt", 0.884),
    "nobc":          ("exps/mimicgen/threading_simdist_dynens_adapt_long_h5_nobc_seed1/latest.pt", 0.823),
    "prefix_n50":    ("exps/mimicgen/threading_simdist_dynens_adapt_n50_seed1/latest.pt", 0.770),
    # collected by me (known provenance); adapt = 30-50k window, runs still going
    "sig002new_n20": ("exps/mimicgen/threading_simdist_dynens_adapt_sig002new_n20_seed1/latest.pt", 0.620),
    "sig01new_n20":  ("exps/mimicgen/threading_simdist_dynens_adapt_sig01new_n20_seed1/latest.pt", 0.580),
    "sig01new_n50":  ("exps/mimicgen/threading_simdist_dynens_adapt_sig01new_n50_seed1/latest.pt", 0.460),
}

NEEDLE = slice(9, 12)
TRIPOD = slice(23, 26)


def build_agent(cfg):
    a = SimDistAgent(1, (37,), None, 7, "", cfg.simdist, device="cuda")
    a.load_state_dict(torch.load(os.path.join(_REPO, STAGE1, "latest.pt"), map_location="cuda"))
    return a.cuda()


def swap_policy(agent, ckpt):
    """Load ONLY model.policy.* from `ckpt` (the frozen stage-2 head)."""
    sd = torch.load(os.path.join(_REPO, ckpt), map_location="cuda")
    cur = agent.state_dict()
    n = 0
    for k in cur:
        if k.startswith("model.policy."):
            cur[k] = sd[k]; n += 1
    assert n > 0
    agent.load_state_dict(cur)
    return n


@torch.no_grad()
def intrinsic(agent, S, A, H):
    """S[N,obs] states, A[N,act] demo actions, plus H-step segments for drift."""
    m = agent.model
    a = m.policy_action(S)
    act_mse = float(((a - A) ** 2).mean())
    a_disp = float((a - a.mean(0)).norm(dim=1).mean())
    # finite-difference Jacobian norm (random probe directions, scaled by state std)
    sd = S.std(0) + 1e-6
    eps = 1e-2
    js = []
    for _ in range(8):
        d = torch.randn_like(S) * sd * eps
        js.append(((m.policy_action(S + d) - a).norm(dim=1) / d.norm(dim=1)).mean())
    jac = float(torch.stack(js).mean())
    return dict(act_mse=act_mse, a_disp=a_disp, jac=jac)


@torch.no_grad()
def rollout_drift(agent, seg_s, H):
    """seg_s[B,H+1,obs]: how far does the policy's OWN imagined H-step latent
    rollout (which seeds the MPPI base candidates) end up from the true one?"""
    m = agent.model
    z = m.encode(seg_s[:, 0])
    for _ in range(H):
        z = m.next(z, m.policy_from_latent(z) if getattr(m.cfg, "shared_encoder", 0)
                   else m.policy_action(m.decode(z)))
    return float((z - m.encode(seg_s[:, H])).norm(dim=1).mean())


def rollout(agent, env, num_game, seed, thresh):
    agent.cfg.use_planner = 0
    succ, reach, csteps, jerks, cframes = [], [], [], [], []
    rs = np.random.get_state(); np.random.seed(seed)
    for g in range(num_game):
        obs, _ = env.reset()
        s, done, acts, rel = 0.0, False, [], []
        while not done:
            a = agent.act(obs, eval_mode=True)
            st = obs["state"]
            st = st.cpu().numpy() if torch.is_tensor(st) else np.asarray(st)
            d = st[NEEDLE] - st[TRIPOD]
            if np.linalg.norm(d) < thresh:
                rel.append(d)
            acts.append(a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a))
            obs, _, terminal, sc, _ = env.step(a)
            s = max(s, float(sc)); done = bool(terminal)
        succ.append(s > 0.5); reach.append(len(rel) > 0); csteps.append(len(rel))
        cframes.extend(rel)
        A = np.stack(acts)
        jerks.append(np.abs(np.diff(A, axis=0)).mean() if len(A) > 1 else 0.0)
    np.random.set_state(rs)
    cf = np.asarray(cframes)
    cov = np.sqrt(max(np.linalg.det(np.cov(cf.T)), 0)) if len(cf) > 10 else 0.0
    return dict(success=float(np.mean(succ)), reach_frac=float(np.mean(reach)),
                contact_steps=float(np.mean(csteps)), contact_cov=float(cov) * 1e6,
                jerk=float(np.mean(jerks)), n_contact_frames=len(cf))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_game", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n_probe", type=int, default=512)
    p.add_argument("--contact_thresh", type=float, default=0.12,
                   help="needle->tripod obs distance (m) defining the insertion "
                        "phase; the successful terminal value is ~0.094")
    p.add_argument("--only", default="")
    args = p.parse_args()

    def log(*a): print(*a, file=sys.stderr, flush=True)

    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(_REPO, STAGE1, "cfg.yaml")))
    cfg.simdist.use_planner = 0
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    H = cfg.simdist.world_model.horizon

    # common held-out probe states (demos 440-479 of the sigma=0.1 pool)
    buf = SequenceReplayBuffer(max_episodes=10000, device="cpu")
    load_demos_to_sequence_buffer(buf, os.path.join(_REPO, HOLDOUT), num_data=480)
    buf.episodes = buf.episodes[440:]
    torch.manual_seed(0)
    seg_s, seg_a, _ = buf.sample(args.n_probe, H, "cuda")
    S, A = seg_s[:, 0], seg_a[:, 0]
    log(f"[probe] {S.shape[0]} held-out states, H={H}, contact_thresh={args.contact_thresh}")

    env = PixelRobosuite(
        env_name=cfg.task_name, robots=["Panda"], episode_length=cfg.episode_length,
        reward_shaping=False, image_size=cfg.image_size, rl_image_size=cfg.rl_image_size,
        camera_names=["agentview"], rl_cameras=["agentview"], use_state=True,
        obs_stack=cfg.obs_stack, state_stack=cfg.state_stack, prop_stack=cfg.prop_stack,
        strict_success=cfg.strict_success, domain_shift=cfg.eval_domain_shift,
        hand_init_perturb_joint_sigma=0.1,
    )

    agent = build_agent(cfg)
    rows = {}
    todo = args.only.split(",") if args.only else list(BASES)
    for name in todo:
        ckpt, adapt = BASES[name]
        swap_policy(agent, ckpt)
        agent.eval()
        r = dict(adapt=adapt)
        r.update(intrinsic(agent, S, A, H))
        r["roll_drift"] = rollout_drift(agent, seg_s, H)
        r.update(rollout(agent, env, args.num_game, args.seed, args.contact_thresh))
        rows[name] = r
        log(f"{name:>14} " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))

    keys = ["adapt", "success", "reach_frac", "contact_steps", "contact_cov",
            "jerk", "act_mse", "a_disp", "jac", "roll_drift"]
    log("\n" + "=" * 118)
    log(f"{'base':>14} " + "".join(f"{k:>12}" for k in keys))
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["adapt"]):
        log(f"{n:>14} " + "".join(f"{r[k]:>12.4f}" for k in keys))
    log("\nSpearman vs adapt outcome (n=%d):" % len(rows))
    from scipy.stats import spearmanr
    y = [r["adapt"] for r in rows.values()]
    for k in keys[1:]:
        x = [r[k] for r in rows.values()]
        rho, pv = spearmanr(x, y)
        log(f"  {k:>14}: rho={rho:+.3f}  p={pv:.3f}")
    os._exit(0)


if __name__ == "__main__":
    main()
