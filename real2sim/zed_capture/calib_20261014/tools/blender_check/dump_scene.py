"""ibrl env (MuJoCo 2.3.2): build the 10-14 Robotiq cell at NOMINAL colours (no DR), set the real home q,
dump the MJCF + every runtime-modified model field (by name) + qpos, and a MuJoCo render for reference."""
import sys, json, dataclasses
import numpy as np, cv2
sys.path.insert(0, "/home/Chenyang/ibrl")
from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import load_sim2real
import env.threading_variants as tv
OUT = sys.argv[1]
cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
zero = {f.name: f.default for f in dataclasses.fields(cell) if f.name.startswith("dr_") and not callable(f.default)}
zero = {k: v for k, v in zero.items() if not isinstance(v, dataclasses._MISSING_TYPE)}
nom = dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras), **zero)
S = 224
env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=S, rl_image_size=S,
                     camera_names=["agentview", "robot0_eye_in_hand"], rl_cameras=["agentview", "robot0_eye_in_hand"],
                     use_state=True, obs_stack=1, state_stack=1, prop_stack=1, flip_image=True, device="cpu",
                     hand_init_perturb_joint_sigma=0.0, sim2real=nom)
np.random.seed(0); env.reset()
e = env.env; m, d = e.sim.model, e.sim.data
q = json.load(open("/home/Chenyang/ibrl/real2sim/zed_capture/calib_20261014/data/home_droid/state.json"))["q"]
for i in range(7):
    d.qpos[m.get_joint_qpos_addr(f"robot0_joint{i+1}")] = q[i]
e.sim.forward()
open(f"{OUT}/scene.xml", "w").write(m.get_xml())
import mujoco as _mj
_T = dict(joint=_mj.mjtObj.mjOBJ_JOINT, body=_mj.mjtObj.mjOBJ_BODY, geom=_mj.mjtObj.mjOBJ_GEOM, camera=_mj.mjtObj.mjOBJ_CAMERA)
def names(kind, n):
    return [_mj.mj_id2name(m._model, _T[kind], i) for i in range(n)]
dump = dict(
    qpos=d.qpos.tolist(),
    jnt=names("joint", m.njnt), jnt_qposadr=m.jnt_qposadr.tolist(),
    body=names("body", m.nbody), body_pos=m.body_pos.tolist(), body_quat=m.body_quat.tolist(),
    geom=names("geom", m.ngeom), geom_rgba=m.geom_rgba.tolist(), geom_size=m.geom_size.tolist(),
    geom_pos=m.geom_pos.tolist(), geom_quat=m.geom_quat.tolist(), geom_matid=m.geom_matid.tolist(),
    geom_xpos=d.geom_xpos.tolist(),
    cam=names("camera", m.ncam), cam_pos=m.cam_pos.tolist(), cam_quat=m.cam_quat.tolist(), cam_fovy=m.cam_fovy.tolist(),
    cam_xpos=d.cam_xpos.tolist(), cam_xmat=d.cam_xmat.tolist(),
    light_pos=m.light_pos.tolist(), light_dir=m.light_dir.tolist(), light_diffuse=m.light_diffuse.tolist(),
    light_ambient=m.light_ambient.tolist(), light_castshadow=m.light_castshadow.tolist(),
    light_xpos=d.light_xpos.tolist(), light_xdir=d.light_xdir.tolist(),
)
json.dump(dump, open(f"{OUT}/state.json", "w"))
for cam in ["agentview", "robot0_eye_in_hand"]:
    rgb = e.sim.render(camera_name=cam, width=S, height=S)[::-1]
    cv2.imwrite(f"{OUT}/mujoco_{cam}.png", rgb[..., ::-1])
print("ngeom", m.ngeom, "nlight", m.nlight, "cams", dump["cam"])
