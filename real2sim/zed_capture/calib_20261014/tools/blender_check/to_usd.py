"""bridge venv (MuJoCo 3): rebuild the dumped scene, verify poses against the 2.3.2 dump, export USD."""
import sys, json
import numpy as np, mujoco
from mujoco.usd import exporter as ux
from pxr import UsdGeom
B = sys.argv[1]
st = json.load(open(f"{B}/state.json"))
m = mujoco.MjModel.from_xml_path(f"{B}/scene.xml")
d = mujoco.MjData(m)
def by(kind, names, field, arr):
    for i, n in enumerate(names):
        if n is None: continue
        j = mujoco.mj_name2id(m, kind, n)
        assert j >= 0, (field, n)
        getattr(m, field)[j] = arr[i]
O = mujoco.mjtObj
for f in ("body_pos", "body_quat"): by(O.mjOBJ_BODY, st["body"], f, st[f])
for f in ("geom_rgba", "geom_size", "geom_pos", "geom_quat"): by(O.mjOBJ_GEOM, st["geom"], f, st[f])
for f in ("cam_pos", "cam_quat", "cam_fovy"): by(O.mjOBJ_CAMERA, st["cam"], f, st[f])
assert m.nlight == len(st["light_pos"])
for f in ("light_pos", "light_dir", "light_diffuse", "light_ambient", "light_castshadow"):
    getattr(m, f)[:] = np.array(st[f])
assert m.nq == len(st["qpos"]); d.qpos[:] = st["qpos"]
mujoco.mj_forward(m, d)
for n, want in zip(st["cam"], st["cam_xpos"]):
    got = d.cam_xpos[mujoco.mj_name2id(m, O.mjOBJ_CAMERA, n)]
    assert np.abs(got - want).max() < 1e-6, (n, got, want)
err = max(np.abs(d.geom_xpos[mujoco.mj_name2id(m, O.mjOBJ_GEOM, n)] - np.array(w)).max()
          for n, w in zip(st["geom"], st["geom_xpos"]) if n)
print(f"poses match the 2.3.2 dump: cams exact, named geoms max {1e3*err:.4f} mm")
cams = ["agentview", "robot0_eye_in_hand"]
ex = ux.USDExporter(model=m, output_directory="usd", output_directory_root=B, camera_names=cams, verbose=False)
ex.update_scene(d)
UsdGeom.SetStageMetersPerUnit(ex.stage, 1.0)
ex.save_scene(filetype="usdc")
# geom name per exported prim + camera intrinsics/poses, for the Blender side
geoms = {}
for i in range(m.ngeom):
    n = mujoco.mj_id2name(m, O.mjOBJ_GEOM, i)
    geoms[i] = dict(name=n, rgba=m.geom_rgba[i].tolist(), matid=int(m.geom_matid[i]),
                    group=int(m.geom_group[i]), body=mujoco.mj_id2name(m, O.mjOBJ_BODY, int(m.geom_bodyid[i])))
camd = {n: dict(pos=d.cam_xpos[mujoco.mj_name2id(m, O.mjOBJ_CAMERA, n)].tolist(),
                mat=d.cam_xmat[mujoco.mj_name2id(m, O.mjOBJ_CAMERA, n)].tolist(),
                fovy=float(m.cam_fovy[mujoco.mj_name2id(m, O.mjOBJ_CAMERA, n)])) for n in cams}
lights = dict(pos=d.light_xpos.tolist(), dir=d.light_xdir.tolist(), diffuse=m.light_diffuse.tolist(),
              ambient=m.light_ambient.tolist(), castshadow=m.light_castshadow.tolist())
json.dump(dict(geoms=geoms, cams=camd, lights=lights), open(f"{B}/meta.json", "w"))
import glob; print(glob.glob(f"{B}/usd/**/*.usdc", recursive=True)[:3])
