import sys, json, dataclasses
import numpy as np, cv2
sys.path.insert(0, "/home/Chenyang/ibrl"); sys.path.insert(0, "/home/Chenyang/ibrl/real2sim/deploy")
from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import load_sim2real
import env.threading_variants as tv
from real2sim.deploy.preprocess import Rectifier
from live_crop_view import load_cams
B = sys.argv[1]; S = 224
D = "/home/Chenyang/ibrl/real2sim/zed_capture/calib_20261014/data/home_droid"
for ros, c in load_cams(tv.REAL_CELL_ROBOTIQ_YAML).items():
    raw = cv2.imread(f"{D}/{ros}_left.png")
    cv2.imwrite(f"{B}/real_{c['sim']}.png", Rectifier(c["K"], c["D"], raw.shape[1], raw.shape[0], S, c["fovy"])(raw))
cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
st = json.load(open(f"{B}/state.json"))
env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=S, rl_image_size=S,
                     camera_names=["agentview"], rl_cameras=["agentview"], use_state=True, obs_stack=1, state_stack=1,
                     prop_stack=1, flip_image=True, device="cpu", hand_init_perturb_joint_sigma=0.0,
                     sim2real=dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras)))
np.random.seed(0); env.reset()
e = env.env; m, d = e.sim.model, e.sim.data
import mujoco as mj
# restore the dumped model fields + qpos exactly (same geometry as the Blender frame)
for f, kind, names in (("body_pos", mj.mjtObj.mjOBJ_BODY, st["body"]), ("body_quat", mj.mjtObj.mjOBJ_BODY, st["body"]),
                       ("geom_pos", mj.mjtObj.mjOBJ_GEOM, st["geom"]), ("geom_quat", mj.mjtObj.mjOBJ_GEOM, st["geom"]),
                       ("geom_size", mj.mjtObj.mjOBJ_GEOM, st["geom"]),
                       ("cam_pos", mj.mjtObj.mjOBJ_CAMERA, st["cam"]), ("cam_quat", mj.mjtObj.mjOBJ_CAMERA, st["cam"]),
                       ("cam_fovy", mj.mjtObj.mjOBJ_CAMERA, st["cam"])):
    arr = getattr(m, f)
    for i, n in enumerate(names):
        if n: arr[mj.mj_name2id(m._model, kind, n)] = st[f][i]
d.qpos[:] = st["qpos"]; e.sim.forward()
cls = np.zeros(m.ngeom, np.int8)
for i in range(m.ngeom):
    n = mj.mj_id2name(m._model, mj.mjtObj.mjOBJ_GEOM, i) or ""
    cls[i] = 1 if n in ("table_visual", "table_collision") else 2 if n.startswith(("needle", "tripod")) else 0
for cam in ("agentview", "robot0_eye_in_hand"):
    seg = e.sim.render(camera_name=cam, width=S, height=S, segmentation=True)[::-1][..., 1]
    np.save(f"{B}/seg_{cam}.npy", np.where(seg >= 0, cls[np.clip(seg, 0, None)], 0))
    err = np.abs(e.sim.render(camera_name=cam, width=S, height=S)[::-1].astype(int) - cv2.imread(f"{B}/mujoco_{cam}.png")[..., ::-1]).mean()
    print(cam, "seg classes", np.bincount(np.load(f"{B}/seg_{cam}.npy").ravel(), minlength=3), "re-render vs dump mean|diff|", round(err, 2))
