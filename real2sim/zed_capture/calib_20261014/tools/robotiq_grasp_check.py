"""Scripted top-down grasp of the needle HANDLE on Threading_RealCellRobotiq (10-14 cell)."""
import numpy as np, robosuite, env.threading_variants  # noqa
from scipy.spatial.transform import Rotation as R

e = robosuite.make("Threading_RealCellRobotiq", robots="Panda", has_renderer=False,
                   has_offscreen_renderer=False, use_camera_obs=False, control_freq=20,
                   controller_configs=robosuite.load_controller_config(default_controller="OSC_POSE"))
res = []
for ep in range(12):
    np.random.seed(100 + ep)
    o = e.reset()
    m, d = e.sim.model, e.sim.data
    nid = m.body_name2id("needle_obj_root")
    hid = m.geom_name2id("needle_obj_handle")
    lo = m.body_name2id("gripper0_left_inner_finger"); ro = m.body_name2id("gripper0_right_inner_finger")
    def step(target, yaw_t, grip, n):
        for _ in range(n):
            eef = e.sim.data.site_xpos[m.site_name2id("gripper0_grip_site")]
            Re = R.from_matrix(e.sim.data.site_xmat[m.site_name2id("gripper0_grip_site")].reshape(3, 3))
            dp = np.clip((target - eef) * 10, -1, 1)
            # finger closing axis in world xy
            ax = d.xpos[ro] - d.xpos[lo]
            cur = np.arctan2(ax[1], ax[0])
            err = (yaw_t - cur + np.pi / 4) % (np.pi / 2) - np.pi / 4
            rot = np.array([0, 0, np.clip(err * 4, -1, 1)])
            # keep vertical: correct tilt of approach (eef z) towards world -z
            z = Re.as_matrix()[:, 2]
            tilt = np.cross(z, [0, 0, -1]) * 3
            rot[:2] = np.clip(tilt[:2], -1, 1)
            e.step(np.concatenate([dp, rot, [grip] * e.robots[0].gripper.dof]))
    hp = d.geom_xpos[hid].copy()
    nyaw = R.from_matrix(d.xmat[nid].reshape(3, 3)).as_euler("xyz")[2]
    step(hp + [0, 0, 0.12], nyaw, -1, 60)
    step(hp + [0, 0, 0.0], nyaw, -1, 50)
    step(hp + [0, 0, 0.0], nyaw, 1, 25)
    q_closed = o_q = [d.qpos[m.get_joint_qpos_addr(n)] for n in env.threading_variants.ROBOTIQ_DRIVER_JOINTS]
    z0 = d.xpos[nid][2]
    step(hp + [0, 0, 0.12], nyaw, 1, 50)
    lift = d.xpos[nid][2] - z0
    gap = np.linalg.norm(d.xpos[ro] - d.xpos[lo])
    res.append(lift)
    print(f"ep{ep} needle yaw {np.degrees(nyaw):7.1f}  driver {np.round(q_closed,3)}  lift {1000*lift:6.1f} mm")
res = np.array(res)
print(f"lifted >8 cm: {np.mean(res > 0.08):.2f}  median lift {1000*np.median(res):.1f} mm")
