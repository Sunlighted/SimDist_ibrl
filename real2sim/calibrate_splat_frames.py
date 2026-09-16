"""Solve a per-body rigid correction aligning PolaRiS's per-link splats to
robosuite's MuJoCo visual meshes, and report the residual so we know whether to
trust each link.

Why this is needed: check_frames.py showed panda_link1..7 already agree to ~1-3 cm
(and their EXTENTS match to ~1 cm, which independently confirms the splats are
metric), but the Robotiq parts carry a consistent translation + probable axis
permutation, and panda_link0's splat is contaminated with surrounding structure
(extent 2-3.5x the mesh).

Method per body: coarse PCA-axis alignment over the 4 sign-consistent rotations,
then TRIMMED ICP (keep the best `trim` fraction of correspondences). Trimming is
what makes link0 tractable -- contaminated points simply never enter the fit.
Output: T (4x4) per body + RMS residual, written to splat_frame_calib.json.
"""
import os, sys, glob, json
os.environ.setdefault("MUJOCO_GL", "disable")
import numpy as np
import mujoco
from scipy.spatial import cKDTree

sys.path.insert(0, "/scratch/yifanr/realsim_survey/_GaussianRenderer/src")
from gaussian_renderer.core.util_gau import load_ply

SEG = "/scratch/yifanr/realsim_survey/polaris/hf-hub/nvidia_droid/SEGMENTED"
MAP = {f"panda_link{i}": f"robot0_link{i}" for i in range(8)}
MAP.update({
    "Gripper-Robotiq_2F_85-base_link":           "gripper0_robotiq_85_adapter_link",
    "Gripper-Robotiq_2F_85-left_inner_finger":   "gripper0_left_inner_finger",
    "Gripper-Robotiq_2F_85-left_inner_knuckle":  "gripper0_left_inner_knuckle",
    "Gripper-Robotiq_2F_85-left_outer_knuckle":  "gripper0_left_outer_knuckle",
    "Gripper-Robotiq_2F_85-right_inner_finger":  "gripper0_right_inner_finger",
    "Gripper-Robotiq_2F_85-right_inner_knuckle": "gripper0_right_inner_knuckle",
    "Gripper-Robotiq_2F_85-right_outer_knuckle": "gripper0_right_outer_knuckle",
})


def kabsch(P, Q, w=None):
    """Rigid R,t minimising |R@P + t - Q|."""
    if w is None:
        w = np.ones(len(P))
    w = w / w.sum()
    pc, qc = (P * w[:, None]).sum(0), (Q * w[:, None]).sum(0)
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def pca_frame(X):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    if np.linalg.det(Vt.T) < 0:
        Vt[-1] *= -1
    return Vt.T


def trimmed_icp(src, dst, trim=0.7, iters=60):
    tree = cKDTree(dst)
    R = np.eye(3); t = np.zeros(3)
    best = None
    Us, Ud = pca_frame(src), pca_frame(dst)
    # 4 det=+1 sign flips of the PCA axes disambiguate symmetry
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        S = np.diag([sx, sy, sx * sy])
        R0 = Ud @ S @ Us.T
        t0 = dst.mean(0) - R0 @ src.mean(0)
        R, t = R0, t0
        for _ in range(iters):
            P = src @ R.T + t
            d, idx = tree.query(P)
            k = max(10, int(trim * len(P)))
            keep = np.argsort(d)[:k]
            R, t = kabsch(src[keep], dst[idx[keep]])
        P = src @ R.T + t
        d, _ = tree.query(P)
        k = max(10, int(trim * len(P)))
        rms = float(np.sqrt((np.sort(d)[:k] ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, R.copy(), t.copy())
    return best


def visual_verts(m, bid):
    pts = []
    for i in range(m.ngeom):
        if m.geom_bodyid[i] != bid or m.geom_group[i] != 1:
            continue
        if m.geom_type[i] != 7 or m.geom_dataid[i] < 0:
            continue
        did = m.geom_dataid[i]
        a, n = m.mesh_vertadr[did], m.mesh_vertnum[did]
        v = np.array(m.mesh_vert[a:a + n]).reshape(-1, 3)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[i]); R = R.reshape(3, 3)
        pts.append(v @ R.T + m.geom_pos[i])
    return np.concatenate(pts, 0) if pts else None


def main():
    import robosuite
    from robosuite.controllers import load_controller_config
    cc = load_controller_config(default_controller="OSC_POSE")
    env = robosuite.make(env_name="Lift", robots="Panda", gripper_types="Robotiq85Gripper",
                         controller_configs=cc, has_renderer=False,
                         has_offscreen_renderer=False, use_camera_obs=False, horizon=10)
    m = env.sim.model

    splats = {}
    for f in sorted(glob.glob(f"{SEG}/*.ply")):
        tok = os.path.basename(f)[:-4].split("-Defeatured")[0]
        if tok.startswith("panda_link"):
            tok = tok.split("-")[0]
        splats.setdefault(tok, []).append(load_ply(f).xyz)
    splats = {k: np.concatenate(v, 0) for k, v in splats.items()}

    out = {}
    print(f"{'link':<42s} {'rms(mm)':>8s} {'|t|(cm)':>8s} {'rot(deg)':>9s}  verdict")
    for tok, body in MAP.items():
        if tok not in splats:
            continue
        V = visual_verts(m, m.body_name2id(body))
        if V is None:
            print(f"{tok:<42s} {'-- no visual mesh --'}"); continue
        S = splats[tok]
        sub = S[np.random.default_rng(0).choice(len(S), min(4000, len(S)), replace=False)]
        rms, R, t = trimmed_icp(sub, V)
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        verdict = "OK" if rms < 0.005 else ("MARGINAL" if rms < 0.010 else "POOR")
        print(f"{tok:<42s} {rms*1000:8.2f} {np.linalg.norm(t)*100:8.2f} {ang:9.2f}  {verdict}")
        out[tok] = {"mj_body": body, "R": R.tolist(), "t": t.tolist(),
                    "rms_m": rms, "rot_deg": float(ang), "verdict": verdict,
                    "n_splat": int(len(S))}
    with open("/scratch/yifanr/gsbench/splat_frame_calib.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote splat_frame_calib.json ({len(out)} links)")
    env.close()


if __name__ == "__main__":
    main()
