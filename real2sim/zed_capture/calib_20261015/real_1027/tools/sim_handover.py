"""Closed loop in the sim from a real demo's state: a ckpt takes over where the 10-27 policy stalled (instead of the GELLO).

The arm path of a real demo is replayed in the sim first (viz/replay_1027_v8/tripod_shift/replay_kin.py --mode track
--dump_at K: OSC servo on FK(real q), the needle grasped by the sim gripper, the tripod at the demo's layout); the full
sim state after step K is loaded into a policy env built as in training (same cell, execution layer + exec_obs on,
DR off), the execution layer re-latched on the arm, the gripper (and its command-delay queue) kept closed, and the
policy runs closed loop on sim renders. Optionally the needle is re-placed in the hand at the REAL mean in-hand pose
of 10-27 (results/inhand_real_1027.json; the wrapper's in-hand DR placement, so the convention is the training one).

  python sim_handover.py --ckpt X/model0.pt --state D/demo_2_state_t173.npz --out D/run.json [--real_inhand] [--video D/x.mp4]
"""
import argparse, dataclasses, json, os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, REPO); os.chdir(REPO)
import torch  # noqa: E402
import real2sim.deploy.policy_server as ps  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.sim2real import Sim2RealConfig, load_sim2real  # noqa: E402
from env.strict_success import StrictSuccessConfig  # noqa: E402

REAL_INHAND = dict(along=-0.01202, depth=0.01525, pitch=np.radians(9.81), yaw=np.radians(6.93))   # inhand_real_1027.json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--cell", default="real2sim/zed_capture/calib_20261015/threading_real_cell_20261030.yaml",
                    help="must be the cell the state was dumped in (same model)")
    ap.add_argument("--real_inhand", action="store_true")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", default="")
    a = ap.parse_args()

    agent, cfg = ps.load_agent(a.ckpt, "cuda")
    cams = cfg.rl_camera.split("+")
    cell = load_sim2real(a.cell)
    off = {f.name: f.default for f in dataclasses.fields(Sim2RealConfig) if f.name.startswith("dr_")}
    cell = dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras),
                               **off)
    cell = dataclasses.replace(cell, inhand_dr_prob=0.0, gripper_speed=0.006, gripper_cmd_delay=(1, 1))
    env = PixelRobosuite(env_name=cfg.task_name, robots="Panda", episode_length=10 ** 6, image_size=224,
                         rl_image_size=224, camera_names=cams, rl_cameras=cams, use_state=False, extra_state=True,
                         device="cpu", strict_success=StrictSuccessConfig(enabled=True, dwell_steps=3, max_tripod_disp=0.04),
                         sim2real=cell, exec_obs=bool(getattr(cfg, "exec_obs", 0)), hand_init_perturb_joint_sigma=0.0)
    np.random.seed(0); torch.manual_seed(0)
    env.reset()
    S = np.load(a.state)
    raw = env.env; sim = raw.sim; m, d = sim.model, sim.data
    names = [m.body_id2name(i) or "" for i in range(m.nbody)]
    assert int(S["nq"]) == m.nq and list(S["body_names"]) == names, "the state comes from another model"
    rob = raw.robots[0]
    nb = m.body_name2id("needle_obj_root")
    z_rest = float(d.body_xpos[nb][2])        # the needle at rest on this table (reset), before the state is written
    d.qpos[:] = S["qpos"]; d.qvel[:] = 0.0
    rob.gripper.current_action = np.array(S["grip_action"], copy=True)
    sim.forward()
    rob.controller.update(force=True)
    rob.controller.reset_goal()
    if env._grip_queue is not None:
        env._grip_queue = type(env._grip_queue)([1.0] * len(env._grip_queue))
    env._last_action = np.zeros_like(env._last_action)
    env._inhand_pending = False
    in_hand = d.body_xpos[nb][2] - z_rest > cell.inhand_dr_lift_m
    if a.real_inhand:
        env._inhand_draw = dict(REAL_INHAND)
        if in_hand:                       # handover after the grasp: re-place it now
            env._inhand_pending, env._inhand_z0 = True, -1e9
            assert env._maybe_inhand_dr(np.r_[np.zeros(6), 1.0])
        else:                             # from the table: the wrapper places it when the policy first lifts it
            env._inhand_pending, env._inhand_z0 = True, z_rest
    obs, _ = env._extract_images(raw._get_observations(force_update=True))

    base = d.body_xpos[m.body_name2id(rob.robot_model.root_body)].copy()
    gs = m.site_name2id("gripper0_grip_site")
    rg = [i for i in range(m.ngeom) if (m.geom_id2name(i) or "").startswith("tripod_obj_ring_")
          and not m.geom_id2name(i).endswith("_vis")]
    ng, hg = m.geom_name2id("needle_obj_needle"), m.geom_name2id("needle_obj_handle")
    tb = m.body_name2id("tripod_obj_root")
    trip0 = d.body_xpos[tb].copy()
    traj, frames, succ_any, raw_any = [], [], False, False
    for k in range(a.steps):
        o = {c: obs[c] for c in cams}; o["prop"] = obs["prop"]
        if a.video:
            frames.append(np.concatenate([obs[c].numpy().transpose(1, 2, 0) for c in cams], 1))
        with torch.no_grad():
            act = np.clip(agent.act(o, eval_mode=True).cpu().numpy().astype(np.float64), -1, 1)
        obs, r, term, succ, _ = env.step(torch.from_numpy(act).float())
        nc, hc = d.geom_xpos[ng] - base, d.geom_xpos[hg] - base
        ring = d.geom_xpos[rg].mean(0) - base
        ok = bool(raw._check_success())
        succ_any |= bool(succ); raw_any |= ok
        traj.append(dict(k=k, grip=(d.site_xpos[gs] - base).round(4).tolist(), ring=ring.round(4).tolist(),
                         needle=nc.round(4).tolist(), handle=hc.round(4).tolist(), a=act.round(3).tolist(),
                         succ=bool(succ), raw=ok))
        if term:
            break
    dist = [np.linalg.norm(np.array(t["needle"]) - np.array(t["ring"])) for t in traj]
    res = dict(ckpt=a.ckpt, state=a.state, real_inhand=a.real_inhand, in_hand_at_start=bool(in_hand),
               inhand_placed=getattr(env, "last_inhand_dr", None), steps=len(traj), success=succ_any,
               raw_success=raw_any, first_raw=next((t["k"] for t in traj if t["raw"]), None),
               min_needle_ring_mm=round(1e3 * min(dist), 1), tripod_push_mm=round(1e3 * float(np.linalg.norm((d.body_xpos[tb] - trip0)[:2])), 1),
               table_impact=bool(getattr(env, "table_impact", False)), traj=traj)
    json.dump(res, open(a.out, "w"))
    print(json.dumps({k: v for k, v in res.items() if k != "traj"}), flush=True)
    if a.video and frames:
        import imageio
        imageio.mimwrite(a.video, [f.astype(np.uint8) for f in frames], fps=10)


if __name__ == "__main__":
    main()
