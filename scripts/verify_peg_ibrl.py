"""Is a PegInsertion IBRL policy's success REAL? Physical checks per episode.

The env's success test is purely geometric (tip inside the pocket column,
15 mm below the top), so it cannot see clip-through or a peg lying sideways.
This rolls the ckpt out in the env rebuilt from ITS OWN cfg.yaml and logs:

  env      : the env params actually used (sigma, board jitter, ep length)
  success  : rate + first-success step
  board    : sampled board offset/yaw per episode (is the jitter live?)
  penetr.  : deepest peg-vs-pocket-wall penetration over the episode (mm;
             MuJoCo contact.dist < 0). Clip-through would show up as >> 1 mm.
  tilt     : peg long-axis angle from vertical at first success
  tip      : tip position in the board frame at success (inside +-hole_half?)
  actor%   : fraction of steps where IBRL executed the ACTOR's action rather
             than the BC prior's (argmax over Q). ~0 => RL learned nothing.

    python scripts/verify_peg_ibrl.py --weight exps/.../model0.pt --n 100
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_rl  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from common_utils import ibrl_utils as utils  # noqa: E402
from scripts.smoke_peg_insertion import scripted_expert_action  # noqa: E402

WALLS = ("wall_left", "wall_right", "wall_front", "wall_back")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=77000)   # disjoint from training eval seeds
    p.add_argument("--out", default=None, help="optional .npz of per-episode metrics")
    # baseline for the measurement itself: roll the SCRIPTED expert in the same
    # env (same params, same seeds) instead of the policy
    p.add_argument("--expert", type=int, default=0)
    # save the first K episodes' sim states in robomimic format, so
    # scripts/render_mimicgen_demos.py (with the pocket close-up) can replay them
    p.add_argument("--save_states", default=None)
    p.add_argument("--save_k", type=int, default=8)
    # JSON merged into env_kwargs, e.g. '{"board_solref": [0.004, 1], "insert_depth": 0.03}'
    p.add_argument("--env_kw", default=None)
    args = p.parse_args()

    agent, _, env_params = train_rl.load_model(args.weight, "cuda")
    env_params = dict(env_params)
    env_params["state_only_no_render"] = True
    if args.env_kw:
        import json as _json
        env_params["env_kwargs"] = {**(env_params.get("env_kwargs") or {}), **_json.loads(args.env_kw)}
    print("[env] task={env_name} ep_len={episode_length} sigma={hand_init_perturb_joint_sigma} "
          "env_kwargs={env_kwargs} shift={domain_shift}".format(
              **{k: env_params.get(k) for k in ("env_name", "episode_length",
                                                 "hand_init_perturb_joint_sigma",
                                                 "env_kwargs", "domain_shift")}))
    env = PixelRobosuite(**env_params)
    bc = agent.bc_policies[0] if getattr(agent, "bc_policies", None) else None
    print(f"[agent] act_method={agent.cfg.act_method} prior={'yes' if bc is not None else 'NO'}")

    rows, saved = [], []
    with torch.no_grad(), utils.eval_mode(agent):
        for ep in range(args.n):
            np.random.seed(args.seed + ep)
            obs, _ = env.reset()
            raw = env.env
            sim = raw.sim
            bid = raw.obj_body_id["board"]
            b_pos0 = np.array(sim.data.body_xpos[bid])
            b_R0 = np.array(sim.data.body_xmat[bid]).reshape(3, 3)
            b_yaw = np.degrees(np.arctan2(b_R0[1, 0], b_R0[0, 0]))
            wall_ids = {sim.model.geom_name2id(g) for g in
                        [n for n in sim.model.geom_names if n and n.endswith(WALLS)]}
            peg_body = raw.obj_body_id["peg"]
            peg_geoms = {i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == peg_body}

            n_actor, n_steps, min_dist, t_succ, tilt, tip_b = 0, 0, 0.0, None, None, None
            xml, states, acts = sim.model.get_xml(), [], []
            depth = -np.inf   # deepest tip reach below the board top (m), while in the pocket column
            terminal = False
            ctl = {}
            while not terminal:
                if args.expert:
                    action = torch.from_numpy(scripted_expert_action(raw, ctl))
                else:
                    action = agent.act(obs, eval_mode=True)
                if bc is not None and not args.expert:
                    a_bc = bc.act(obs, cpu=False)
                    n_actor += int(not torch.allclose(action.flatten().cpu(), a_bc.flatten().cpu(), atol=1e-6))
                states.append(np.array(sim.get_state().flatten()))
                acts.append(np.asarray(action.cpu()).reshape(-1))
                obs, reward, terminal, success, _ = env.step(action)
                n_steps += 1
                for c in sim.data.contact[: sim.data.ncon]:
                    pair = {c.geom1, c.geom2}
                    if pair & peg_geoms and pair & wall_ids:
                        min_dist = min(min_dist, float(c.dist))
                d = b_R0.T @ (raw._peg_tip_pos() - b_pos0)
                if abs(d[0]) < raw.board.hole_half_x and abs(d[1]) < raw.board.hole_half_y:
                    depth = max(depth, raw.board.half_z - d[2])
                if success and t_succ is None:
                    t_succ = n_steps
                    R = np.array(sim.data.body_xmat[peg_body]).reshape(3, 3)
                    tilt = np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))
                    tip_b = b_R0.T @ (raw._peg_tip_pos() - b_pos0)
            if args.save_states and ep < args.save_k:
                saved.append(dict(states=np.stack(states), actions=np.stack(acts), xml=xml,
                                  succ=t_succ is not None))
            rows.append(dict(
                succ=t_succ is not None, t=t_succ or n_steps, actor=n_actor / max(n_steps, 1),
                pen_mm=-min_dist * 1000, tilt=tilt if tilt is not None else np.nan,
                tip=tip_b if tip_b is not None else np.full(3, np.nan),
                bx=b_pos0[0], by=b_pos0[1], byaw=b_yaw, depth=depth * 1000))
            if ep < 10 or ep % 25 == 0:
                r = rows[-1]
                print(f"ep {ep:3d} succ={int(r['succ'])} t={r['t']:3d} actor={r['actor']:.2f} "
                      f"pen={r['pen_mm']:.2f}mm depth={r['depth']:.1f}mm tilt={r['tilt']:.1f} board=({r['bx']:.4f},{r['by']:.4f},{r['byaw']:.2f}deg)")

    S = np.array([r["succ"] for r in rows])
    pen = np.array([r["pen_mm"] for r in rows])
    act = np.array([r["actor"] for r in rows])
    bx, by, byaw = (np.array([r[k] for r in rows]) for k in ("bx", "by", "byaw"))
    tilt = np.array([r["tilt"] for r in rows])
    tip = np.stack([r["tip"] for r in rows])
    t = np.array([r["t"] for r in rows])
    dep = np.array([r["depth"] for r in rows])
    se = np.sqrt(S.mean() * (1 - S.mean()) / len(S))
    print("\n================ SUMMARY ================")
    print(f"success {S.mean():.3f} +- {se:.3f} (n={len(S)})   mean steps to success {t[S].mean():.1f}")
    print(f"board jitter live? x range {(bx.max()-bx.min())*1000:.1f} mm, "
          f"y range {(by.max()-by.min())*1000:.1f} mm, yaw [{byaw.min():.2f}, {byaw.max():.2f}] deg")
    print(f"penetration mm: median {np.median(pen):.2f}  p95 {np.percentile(pen, 95):.2f}  max {pen.max():.2f}"
          f"   (succ max {pen[S].max() if S.any() else np.nan:.2f})")
    fin = np.isfinite(dep)
    if fin.any():
        print(f"max insertion depth below board top (mm, 41 = through): median {np.median(dep[fin]):.1f} "
              f"min {dep[fin].min():.1f} max {dep[fin].max():.1f}  (episodes entering the column: {fin.mean():.2f})")
    if S.any():
        print(f"tilt at success deg: median {np.nanmedian(tilt[S]):.1f} max {np.nanmax(tilt[S]):.1f}")
        print(f"tip in board frame at success (mm): |x| max {np.nanmax(np.abs(tip[S,0]))*1000:.1f} "
              f"(half {17.78}), |y| max {np.nanmax(np.abs(tip[S,1]))*1000:.1f} (half {11.885}), "
              f"z max {np.nanmax(tip[S,2])*1000:.1f} (must be < {20.49-15:.2f})")
    print(f"actor share of executed actions: mean {act.mean():.3f} (succ {act[S].mean() if S.any() else np.nan:.3f}, "
          f"fail {act[~S].mean() if (~S).any() else np.nan:.3f})")
    # success vs board jitter magnitude
    r = np.hypot(bx - bx.mean(), by - by.mean()) * 1000
    for lo, hi in [(0, 5), (5, 10), (10, 15)]:
        m = (r >= lo) & (r < hi)
        if m.any():
            print(f"  board offset {lo:2d}-{hi:2d} mm: success {S[m].mean():.3f} (n={m.sum()})")
    for lo, hi in [(0, 2.5), (2.5, 5.1)]:
        m = (np.abs(byaw) >= lo) & (np.abs(byaw) < hi)
        if m.any():
            print(f"  board |yaw| {lo}-{hi} deg: success {S[m].mean():.3f} (n={m.sum()})")
    if args.save_states:
        import h5py, json
        with h5py.File(args.save_states, "w") as f:
            g = f.create_group("data")
            ek = dict(robots=["Panda"], controller_configs=env.ctrl_config,
                      initialization_noise={"magnitude": env_params["hand_init_perturb_joint_sigma"], "type": "gaussian"},
                      control_freq=20, reward_shaping=False, has_renderer=False,
                      has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True,
                      ignore_done=True, **(env_params.get("env_kwargs") or {}))
            g.attrs["env_args"] = json.dumps(dict(env_name=env_params["env_name"], type=1, env_kwargs=ek))
            for i, d in enumerate(saved):
                e = g.create_group(f"demo_{i}")
                e.create_dataset("states", data=d["states"])
                e.create_dataset("actions", data=d["actions"])
                e.attrs["model_file"] = d["xml"]
                e.attrs["success"] = int(d["succ"])
        print("saved", len(saved), "episodes ->", args.save_states)
    if args.out:
        np.savez(args.out, succ=S, pen=pen, actor=act, bx=bx, by=by, byaw=byaw, tilt=tilt, tip=tip, t=t)


if __name__ == "__main__":
    main()
