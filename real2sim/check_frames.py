"""Frame-alignment gate: do PolaRiS's per-link splats live in the SAME local
frame as robosuite's MuJoCo body frames?

Pure model-space check -- no GPU, no simulation, no rendering. For each mapped
link we compare the splat's centroid/extent (in its stored local frame) against
the MuJoCo body's own visual-geom centroid/extent (in body-local frame). If the
two conventions agree these should sit within ~1-2 cm (they differ only because
a splat covers the visible surface while geoms are collision/visual primitives).
A convention mismatch shows up as 10s of cm, or a wrong-axis extent.
"""
import os, sys, glob
os.environ.setdefault("MUJOCO_GL", "disable")  # model-space only; no GL context needed
import numpy as np
import mujoco
sys.path.insert(0, "/scratch/yifanr/realsim_survey/_GaussianRenderer/src")
from gaussian_renderer.core.util_gau import load_ply

SEG = "/scratch/yifanr/realsim_survey/polaris/hf-hub/nvidia_droid/SEGMENTED"

# PolaRiS link token -> robosuite MuJoCo body name.
MAP = {f"panda_link{i}": f"robot0_link{i}" for i in range(8)}
MAP.update({
    "Gripper-Robotiq_2F_85-base_link":          "gripper0_robotiq_85_adapter_link",
    "Gripper-Robotiq_2F_85-left_inner_finger":  "gripper0_left_inner_finger",
    "Gripper-Robotiq_2F_85-left_inner_knuckle": "gripper0_left_inner_knuckle",
    "Gripper-Robotiq_2F_85-left_outer_knuckle": "gripper0_left_outer_knuckle",
    "Gripper-Robotiq_2F_85-right_inner_finger": "gripper0_right_inner_finger",
    "Gripper-Robotiq_2F_85-right_inner_knuckle":"gripper0_right_inner_knuckle",
    "Gripper-Robotiq_2F_85-right_outer_knuckle":"gripper0_right_outer_knuckle",
})
# robosuite's Robotiq85 merges outer_finger into outer_knuckle -- no counterpart body.
UNMAPPED = ["Gripper-Robotiq_2F_85-left_outer_finger",
            "Gripper-Robotiq_2F_85-right_outer_finger"]


def splat_stats():
    out = {}
    for f in sorted(glob.glob(f"{SEG}/*.ply")):
        stem = os.path.basename(f)[:-4]
        tok = stem.split("-Defeatured")[0]
        if tok.startswith("panda_link"):
            tok = tok.split("-")[0]
        g = load_ply(f)
        out.setdefault(tok, []).append(g.xyz)
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def main():
    import robosuite
    from robosuite.controllers import load_controller_config
    cc = load_controller_config(default_controller="OSC_POSE")
    env = robosuite.make(env_name="Lift", robots="Panda", gripper_types="Robotiq85Gripper",
                         controller_configs=cc, has_renderer=False,
                         has_offscreen_renderer=False, use_camera_obs=False, horizon=10)
    m = env.sim.model
    sp = splat_stats()

    print(f"{'polaris link':<42s} {'mj body':<34s} {'splat c (m)':<24s} "
          f"{'mj geom c (m)':<24s} {'d(cm)':>7s} {'splat ext':<20s} {'mj ext':<20s}")
    ds = []
    for tok, body in MAP.items():
        if tok not in sp:
            print(f"{tok:<42s} {'-- NO PLY --'}"); continue
        try:
            bid = m.body_name2id(body)
        except Exception:
            print(f"{tok:<42s} {body:<34s} -- NO SUCH MJ BODY --"); continue
        xyz = sp[tok]
        sc, sext = xyz.mean(0), xyz.max(0) - xyz.min(0)
        # Compare against the body's VISUAL MESH VERTICES in body-local frame.
        # geom_pos/geom_size are the mesh FRAME ORIGIN and bounding half-extent,
        # not the shape -- using them as a centroid proxy is wrong (it put
        # link0 15.8 cm out on the first pass). Use group==1 (visual) geoms only,
        # since a splat is a visual representation.
        pts = []
        for i in range(m.ngeom):
            if m.geom_bodyid[i] != bid or m.geom_group[i] != 1:
                continue
            if m.geom_type[i] != 7 or m.geom_dataid[i] < 0:   # 7 = mjGEOM_MESH
                continue
            did = m.geom_dataid[i]
            a, n = m.mesh_vertadr[did], m.mesh_vertnum[did]
            v = np.array(m.mesh_vert[a:a + n]).reshape(-1, 3)
            q = m.geom_quat[i]                                # wxyz
            R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
            pts.append(v @ R.T + m.geom_pos[i])
        if not pts:
            print(f"{tok:<42s} {body:<34s} -- no visual mesh geoms --"); continue
        V = np.concatenate(pts, 0)
        gc = V.mean(0)
        gext = V.max(0) - V.min(0)
        d = np.linalg.norm(sc - gc) * 100
        ds.append((d, tok))
        print(f"{tok:<42s} {body:<34s} {np.round(sc,3)!s:<24s} {np.round(gc,3)!s:<24s} "
              f"{d:7.2f} {np.round(sext,3)!s:<20s} {np.round(gext,3)!s:<20s}")
    for tok in UNMAPPED:
        if tok in sp:
            print(f"{tok:<42s} {'(no robosuite body -- merged into outer_knuckle)'}")
    if ds:
        ds.sort()
        arr = np.array([d for d, _ in ds])
        print(f"\nmapped={len(ds)}  median={np.median(arr):.2f} cm  "
              f"max={arr.max():.2f} cm ({ds[-1][1]})")
    env.close()


if __name__ == "__main__":
    main()
