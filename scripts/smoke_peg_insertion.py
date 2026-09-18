"""Smoke test for the real2sim PegInsertion_D0 env (env/peg_insertion.py).

Task: peg starts UPRIGHT ON THE TABLE; full grasp -> transport -> insert.

Runs entirely on CPU, no rendering (state_only_no_render), no GPU needed:
    python scripts/smoke_peg_insertion.py

Checks, in order:
  1. construction + registration + 37-dim state
  2. reset: peg upright on the table inside the sampler bounds, board static
  3. grasp: scripted expert reaches the transport phase with the peg lifted
  4. gripper open after grasp -> peg FALLS (held by contact, not welded)
  5. success detector: teleport peg into / next to the hole
  6. full scripted episodes (grasp + insert) -- also the demo-collection expert
"""

import argparse
import sys

import numpy as np
import torch  # noqa: F401  (import order: torch before mujoco per repo habit)

sys.path.insert(0, ".")

from env.peg_insertion import PEG_HALF, HOLE_HALF_X, HOLE_HALF_Y, BOARD_HALF_Z
from env.robosuite_wrapper import PixelRobosuite

from env.peg_insertion import MEASURED_TABLE_OFFSET
TABLE_Z = MEASURED_TABLE_OFFSET[2]      # measured work-surface height
GRIP_DEPTH = 0.035  # grip point below the peg top (short lever, see docs §5)


def make_env(sigma=0.1, episode_length=300):
    return PixelRobosuite(
        env_name="PegInsertion_D0",
        robots="Panda",
        episode_length=episode_length,
        use_state=True,
        state_only_no_render=True,
        device="cpu",
        hand_init_perturb_joint_sigma=sigma,
        end_on_success=True,
    )


def _grip_site_pose(raw_env):
    sid = raw_env.sim.model.site_name2id(
        raw_env.robots[0].gripper.important_sites["grip_site"]
    )
    pos = np.array(raw_env.sim.data.site_xpos[sid])
    R = np.array(raw_env.sim.data.site_xmat[sid]).reshape(3, 3)
    return pos, R


def _finger_qpos(raw_env):
    sim = raw_env.sim
    out = []
    for a in raw_env.robots[0].gripper.actuators:
        jid = int(sim.model.actuator_trnid[sim.model.actuator_name2id(a)][0])
        out.append(float(sim.data.qpos[sim.model.jnt_qposadr[jid]]))
    return np.array(out)


def _axis_align_err(R, col, target):
    """Cross-product error rotating R[:, col] onto target (small-angle)."""
    cur = R[:, col] / np.linalg.norm(R[:, col])
    return np.cross(cur, target)


def _yaw_err(a_xy, b_xy):
    """Signed angle (about z) from a to b, wrapped to [-pi/2, pi/2] (both axes
    are pi-symmetric: fingers and peg faces)."""
    a = np.arctan2(a_xy[1], a_xy[0])
    b = np.arctan2(b_xy[1], b_xy[0])
    e = b - a
    while e > np.pi / 2:
        e -= np.pi
    while e < -np.pi / 2:
        e += np.pi
    return e


def _peg_orient_err(R_peg):
    """so(3) error to the insertion orientation: long axis vertical -- toward
    the NEAREST of world +-z (a peg grasped upright off the table has +z UP;
    targeting -z unconditionally commands a 180 deg flip, which rolled the
    wrist over at saturated rate) -- and the 33 mm axis along the nearest
    world +-x (board is axis-aligned; both are pi-symmetric)."""
    zt = np.array([0.0, 0.0, 1.0 if R_peg[2, 2] >= 0 else -1.0])
    xt = np.array([1.0, 0.0, 0.0]) if R_peg[0, 0] >= 0 else np.array([-1.0, 0.0, 0.0])
    yt = np.cross(zt, xt)
    Rt = np.stack([xt, yt, zt], axis=1)
    return 0.5 * (
        np.cross(R_peg[:, 0], Rt[:, 0])
        + np.cross(R_peg[:, 1], Rt[:, 1])
        + np.cross(R_peg[:, 2], Rt[:, 2])
    )


def scripted_expert_action(raw_env, ctl):
    """One step of the scripted grasp+insert expert (P-control on OSC deltas).

    ctl is a dict carrying phase state between calls; pass {} at episode start.
    Phases: hover (above the peg, yaw-align the fingers to the 21.2 mm faces,
    gripper held at its half-open reset width via action 0) -> descend ->
    close (position servo ramps at gripper speed 0.01/step, so ~50 steps) ->
    lift -> seek (tip over the hole, full-orientation servo) -> insert
    (latched gentle push; retreat to seek if jammed at the mouth).
    """
    sim = raw_env.sim
    board = np.array(sim.data.body_xpos[raw_env.obj_body_id["board"]])
    hole_top = board[2] + BOARD_HALF_Z
    peg_pos = np.array(sim.data.body_xpos[raw_env.obj_body_id["peg"]])
    R_peg = np.array(sim.data.body_xmat[raw_env.obj_body_id["peg"]]).reshape(3, 3)
    tip = raw_env._peg_tip_pos()
    site_pos, site_R = _grip_site_pose(raw_env)

    mode = ctl.get("mode", "hover")
    grip = 1.0
    dxy = np.zeros(2)
    dz = 0.0
    # keep the hand vertical throughout; pre-grasp phases add finger-yaw align
    rot = 3.0 * _axis_align_err(site_R, 2, np.array([0.0, 0.0, -1.0]))

    if mode in ("hover", "descend"):
        grip = 0.0  # sign(0)=0: hold the half-open reset width (41.6 mm)
        finger_axis = raw_env.finger_travel_axis()
        peg_y = R_peg[:, 1]
        rot[2] += 2.0 * _yaw_err(finger_axis[:2], peg_y[:2])
        peg_top = peg_pos[2] + PEG_HALF[2]
        exy = peg_pos[:2] - site_pos[:2]
        if mode == "hover":
            ez = (peg_top + 0.05) - site_pos[2]
            if np.linalg.norm(exy) < 0.003 and abs(rot[2]) < 0.06:
                ctl["ok"] = ctl.get("ok", 0) + 1
                if ctl["ok"] >= 2:
                    mode, ctl["ok"] = "descend", 0
            else:
                ctl["ok"] = 0
        else:
            ez = (peg_top - GRIP_DEPTH) - site_pos[2]
            if abs(ez) < 0.004 and np.linalg.norm(exy) < 0.003:
                mode = "close"
                ctl["close_steps"], ctl["fq"] = 0, _finger_qpos(raw_env)
        dxy = np.clip(exy * 30.0, -0.6, 0.6)
        dz = float(np.clip(ez * 30.0, -0.4, 0.4))

    elif mode == "close":
        ctl["close_steps"] += 1
        # The gripper action ramps the position servo at speed 0.01/step, so
        # the CONTACT force builds over ~100 steps. Fingers touch the peg after
        # ~50 steps but at ~0.2 N -- lifting then lets the peg pivot (measured
        # tilt 3->85 deg in transit). Hold until the servo TARGETS are fully
        # closed (ctrl ~= 0 for both fingers => ~10 N/finger), not until the
        # finger qpos stops moving.
        act_ids = [
            sim.model.actuator_name2id(a)
            for a in raw_env.robots[0].gripper.actuators
        ]
        ctrl_gap = max(abs(float(sim.data.ctrl[a])) for a in act_ids)
        if ctl["close_steps"] >= 20 and ctrl_gap < 5e-4:
            mode = "lift"

    elif mode == "lift":
        lift_to = raw_env.table_offset[2] + 0.16
        dz = float(np.clip((lift_to - site_pos[2]) * 30.0, -0.4, 0.4))
        if site_pos[2] > lift_to - 0.008:
            mode = "seek"

    elif mode == "seek":
        rot = np.clip(3.0 * _peg_orient_err(R_peg), -0.25, 0.25)
        exy = board[:2] - tip[:2]
        far = np.linalg.norm(exy) > 0.015
        # travel HIGH (30 mm above the board) until nearly over the hole: at
        # 10 mm, a few degrees of in-grip swing dips a peg corner below the
        # board top and the edge catch pivots the peg ~90 deg in the fingers
        z_ref = hole_top + (0.030 if far else 0.010)
        tilt_ok = abs(R_peg[2, 2]) > np.cos(np.radians(2.0))
        ac = ctl.get("align_count", 0)
        ac = ac + 1 if (not far and np.linalg.norm(exy) < 0.001 and tilt_ok) else 0
        ctl["align_count"] = ac
        if ac >= 3:
            mode = "insert"
            ctl["last_depth"], ctl["stall"] = hole_top - tip[2], 0
        dxy = np.clip(exy * 30.0, -0.4, 0.4)
        dz = float(np.clip((z_ref - tip[2]) * 30.0, -0.4, 0.4))

    elif mode == "insert":
        rot = np.clip(3.0 * _peg_orient_err(R_peg), -0.4, 0.4)
        exy = board[:2] - tip[:2]
        depth = hole_top - tip[2]
        dxy = np.clip(exy * 30.0, -0.2, 0.2)
        dz = -0.15
        if depth - ctl["last_depth"] < 0.0005:
            ctl["stall"] += 1
        else:
            ctl["stall"], ctl["last_depth"] = 0, depth
        if ctl["stall"] > 15 and depth < 0.005:
            mode, ctl["align_count"] = "seek", 0  # jammed at the mouth: retry

    ctl["mode"] = mode
    rot = np.clip(rot, -0.4, 0.4)
    return np.concatenate([dxy, [dz], rot, [grip]]).astype(np.float32)


def scripted_episode(env, verbose=False):
    """Full scripted-expert episode. Returns (success, steps)."""
    env.reset()
    raw = env.env
    ctl = {}
    for t in range(env.env.horizon):
        action = scripted_expert_action(raw, ctl)
        _, _, terminal, success, _ = env.step(torch.from_numpy(action))
        if verbose and t % 10 == 0:
            tip = raw._peg_tip_pos()
            print(f"    t={t} {ctl['mode']:7s} tip z={tip[2]:.4f} success={success}")
        if success:
            return True, t + 1
        if terminal:
            break
    return False, env.env.horizon


# ------------------------------------------------------------------ checks
def t1_construct():
    env = make_env()
    rl_obs = env.reset()[0]
    assert rl_obs["state"].shape == (37,), rl_obs["state"].shape
    assert rl_obs["prop"].shape == (9,), rl_obs["prop"].shape
    assert env.action_dim == 7, env.action_dim
    print(f"[1] construct OK: state {tuple(rl_obs['state'].shape)}, "
          f"prop {tuple(rl_obs['prop'].shape)}, action_dim {env.action_dim}")
    return env

def t2_on_table(env):
    for _ in range(3):
        env.reset()
        raw = env.env
        p = np.array(raw.sim.data.body_xpos[raw.obj_body_id["peg"]])
        R = np.array(raw.sim.data.body_xmat[raw.obj_body_id["peg"]]).reshape(3, 3)
        b = np.array(raw.sim.data.body_xpos[raw.obj_body_id["board"]])
        assert abs(p[2] - (TABLE_Z + PEG_HALF[2])) < 0.005, p
        assert R[2, 2] > 0.99, "peg not upright at reset"
        assert -0.11 < p[0] < 0.11 and 0.09 < p[1] < 0.21, p
        assert np.allclose(b[:2], [0.0, -0.15], atol=1e-6), b
    print(f"[2] on-table reset OK: peg upright at z={p[2]:.4f}, board fixed")

def t3_grasp(env):
    env.reset()
    raw = env.env
    ctl = {}
    for t in range(150):
        env.step(torch.from_numpy(scripted_expert_action(raw, ctl)))
        if ctl["mode"] == "seek":
            break
    p = np.array(raw.sim.data.body_xpos[raw.obj_body_id["peg"]])
    lift = p[2] - (TABLE_Z + PEG_HALF[2])
    assert ctl["mode"] == "seek", f"grasp did not complete (mode={ctl['mode']})"
    assert lift > 0.03, f"peg lifted only {lift:.3f} m"
    print(f"[3] grasp OK: reached transport in {t + 1} steps, peg lifted {lift:.3f} m")
    return env, raw

def t4_drop(env, raw):
    OPEN = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
    z0 = float(raw.sim.data.body_xpos[raw.obj_body_id["peg"]][2])
    for _ in range(60):
        env.step(torch.from_numpy(OPEN))
    z1 = float(raw.sim.data.body_xpos[raw.obj_body_id["peg"]][2])
    assert z0 - z1 > 0.03, f"peg only fell {z0 - z1:.4f} m after opening gripper"
    print(f"[4] drop OK: peg fell {z0 - z1:.3f} m after gripper open")

def t5_success_detector(env):
    env.reset()
    raw = env.env
    sim = raw.sim
    board_pos = np.array(sim.data.body_xpos[raw.obj_body_id["board"]])
    tip_z = board_pos[2] + BOARD_HALF_Z - 0.020
    center_in = np.array([board_pos[0], board_pos[1], tip_z + PEG_HALF[2]])
    quat_vert = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz: 180 deg about x -> +z down
    sim.data.set_joint_qpos(raw.peg.joints[0], np.concatenate([center_in, quat_vert]))
    sim.forward()
    assert raw._check_success(), "teleported-in peg not detected as success"
    for off in ([0.05, 0, 0], [0, 0, 0.10]):
        c = center_in + np.array(off)
        sim.data.set_joint_qpos(raw.peg.joints[0], np.concatenate([c, quat_vert]))
        sim.forward()
        assert not raw._check_success(), f"offset {off} wrongly detected as success"
    print("[5] success detector OK (in=True, side=False, above=False)")

def t6_scripted(env, n_episodes):
    results = [scripted_episode(env) for _ in range(n_episodes)]
    succ = [r[0] for r in results]
    steps = [r[1] for r in results if r[0]]
    rate = float(np.mean(succ))
    avg = float(np.mean(steps)) if steps else float("nan")
    print(f"[6] scripted grasp+insert: {sum(succ)}/{n_episodes} success "
          f"(rate {rate:.2f}), avg steps to success {avg:.0f}")
    assert rate >= 0.8, f"scripted expert rate {rate:.2f} < 0.8 -- env regressed"
    return rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    env = t1_construct()
    t2_on_table(env)
    env2, raw2 = t3_grasp(env)
    t4_drop(env2, raw2)
    t5_success_detector(env)
    if args.verbose:
        scripted_episode(env, verbose=True)
    rate = t6_scripted(env, args.episodes)
    print(f"\nALL CHECKS PASSED (scripted rate {rate:.2f}; clearance "
          f"x/y = {(HOLE_HALF_X - PEG_HALF[0]) * 1000:.2f}/"
          f"{(HOLE_HALF_Y - PEG_HALF[1]) * 1000:.2f} mm/side)")


if __name__ == "__main__":
    main()
