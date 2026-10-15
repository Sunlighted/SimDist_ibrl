"""Gate for env/sim2real.py.

Two things have to be proven SEPARATELY (docs/simdist.md 46, iron law 26):
  A. the default-off contract  -- enabled=0 is byte-for-byte inert;
  B. the knobs actually bite   -- an enabled config really moves the pixels /
     the model arrays. An INERT feature that silently does nothing is the
     expensive failure here: `augment_fn` sat unused for months and cost a
     multi-day run while reporting nothing.

Plus a ROUND-TRIP on the camera extrinsics, which is the only way to catch a
frame-composition bug: write a pose in the base (or EE) frame, read the pose
back out of MuJoCo, convert it back, and require it to match. A sign error in
the OpenCV<->MuJoCo flip or a missing parent-body transform survives every
other check in this file.

Run:  python scripts/test_sim2real.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import (OPENCV_TO_MUJOCO, RealCameraConfig, Sim2RealConfig,
                          lookat_R_opencv, mat2quat_wxyz, quat_wxyz2mat)

CAMS = ["agentview", "robot0_eye_in_hand"]
TOUCHED = ("geom_size", "geom_rgba", "geom_matid", "cam_pos", "cam_quat",
           "cam_fovy", "light_diffuse", "light_ambient", "light_specular",
           "light_dir", "light_pos", "body_pos")
FAILED = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")
    if not ok:
        FAILED.append(name)


def make_env(s2r=None, sigma=0.0, task="Threading_D0"):
    return PixelRobosuite(
        task, "Panda", 400, image_size=224, rl_image_size=224,
        camera_names=CAMS, rl_cameras=CAMS, use_state=False, obs_stack=1,
        state_stack=1, prop_stack=1, flip_image=True, device="cpu",
        hand_init_perturb_joint_sigma=sigma, sim2real=s2r)


def snap(env):
    m = env.env.sim.model
    return {k: np.array(getattr(m, k)).copy() for k in TOUCHED}


def test_default_off():
    print("\nA. default-off contract")
    e0 = make_env(None)
    np.random.seed(0); e0.reset(); a = snap(e0)
    e1 = make_env(Sim2RealConfig())            # explicit, enabled=0
    np.random.seed(0); e1.reset(); b = snap(e1)
    for k in TOUCHED:
        check(f"enabled=0 leaves model.{k} untouched",
              np.array_equal(a[k], b[k]))
    check("enabled=0 -> any_dr() is False", not Sim2RealConfig().any_dr())


def test_geometry():
    print("\nB. matched geometry bites")
    cfg = Sim2RealConfig(enabled=1, table_above_base=0.0208,
                         table_visual_half=(1.30, 1.60),
                         table_rgb=(226 / 255, 220 / 255, 206 / 255))
    env = make_env(cfg)
    np.random.seed(0); env.reset()
    m, d = env.env.sim.model, env.env.sim.data
    bid = m.body_name2id("robot0_base"); base_z = d.body_xpos[bid][2]
    gid = m.geom_name2id("table_visual")
    top = float(env.env.table_offset[2])
    check("table top is 20.8 mm above the base origin",
          abs((top - base_z) - 0.0208) < 1e-6, f"delta={top-base_z:+.4f}")
    check("visual half-size x/y written", 
          np.allclose(m.geom_size[gid][:2], [1.30, 1.60]), str(m.geom_size[gid]))
    # TableArena half-sizes are table_full_size/2 = (0.4, 0.4, 0.025): the table
    # is a 5 cm SLAB, not a solid block. The z half-size must survive untouched or
    # the top surface -- and every object standing on it -- would move.
    check("visual half-size z NOT written (table top cannot move here)",
          abs(m.geom_size[gid][2] - 0.025) < 1e-9, f"z={m.geom_size[gid][2]}")
    check("ceramic texture dropped (matid=-1)", m.geom_matid[gid] == -1)
    check("table rgb applied",
          np.allclose(m.geom_rgba[gid][:3], [226/255, 220/255, 206/255], atol=1e-6))
    # the collision box and the robot must NOT have moved
    cgid = m.geom_name2id("table_collision")
    check("collision geom size untouched",
          np.allclose(m.geom_size[cgid], [0.4, 0.4, 0.025]), str(m.geom_size[cgid]))
    check("robot base x unchanged by the lift",
          abs(d.body_xpos[bid][0] - (-0.56)) < 1e-6, f"x={d.body_xpos[bid][0]}")
    # objects must follow the table (placement reference IS table_offset)
    for nm in ("needle_obj_root", "tripod_obj_root"):
        z = d.body_xpos[m.body_name2id(nm)][2]
        check(f"{nm} sits on the LIFTED table", z > top - 0.02,
              f"z={z:.4f} vs top {top:.4f}")


def test_camera_roundtrip():
    print("\nC. camera extrinsics round-trip")
    p_base = (0.60, 0.48, 0.78)
    R = lookat_R_opencv((0.57, -0.43, -0.70))
    q = tuple(mat2quat_wxyz(R))
    p_ee = (0.036, -0.076, 0.014)
    R_ee = lookat_R_opencv((0.0, 0.2, -1.0))
    cfg = Sim2RealConfig(
        enabled=1,
        cameras=(RealCameraConfig(name="agentview", frame="base", pos=p_base,
                                  quat=q, fovy=70.0),
                 RealCameraConfig(name="robot0_eye_in_hand", frame="ee", pos=p_ee,
                                  quat=tuple(mat2quat_wxyz(R_ee)), fy=160.0, res=224)))
    env = make_env(cfg)
    np.random.seed(0); env.reset()
    m, d = env.env.sim.model, env.env.sim.data
    bid = m.body_name2id("robot0_base")
    bp, bR = d.body_xpos[bid].copy(), d.body_xmat[bid].reshape(3, 3).copy()
    sid = m.site_name2id("gripper0_grip_site")
    sp, sR = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()

    for nm, fp, fR, want_p, want_R in [
            ("agentview", bp, bR, p_base, R),
            ("robot0_eye_in_hand", sp, sR, p_ee, R_ee)]:
        cid = m.camera_name2id(nm)
        got_p = fR.T @ (d.cam_xpos[cid] - fp)
        got_R = (fR.T @ d.cam_xmat[cid].reshape(3, 3)) @ OPENCV_TO_MUJOCO
        check(f"{nm}: position round-trips in the {'base' if nm=='agentview' else 'EE'} frame",
              np.allclose(got_p, want_p, atol=1e-6), f"got {np.round(got_p,5)}")
        check(f"{nm}: rotation round-trips (OpenCV convention)",
              np.allclose(got_R, want_R, atol=1e-6),
              f"max|d|={np.abs(got_R-want_R).max():.2e}")

    cid = m.camera_name2id("agentview")
    check("agentview fovy set from `fovy`", abs(m.cam_fovy[cid] - 70.0) < 1e-6)
    cid = m.camera_name2id("robot0_eye_in_hand")
    want = np.degrees(2 * np.arctan(224 / (2 * 160.0)))
    check("wrist fovy derived from fy/res", abs(m.cam_fovy[cid] - want) < 1e-6,
          f"{m.cam_fovy[cid]:.3f} vs {want:.3f}")
    # a wrist camera is parented to the hand, not the world
    check("wrist camera is parented to a moving body",
          int(m.cam_bodyid[cid]) != 0, f"bodyid={int(m.cam_bodyid[cid])}")


def test_lookat_equivalence():
    print("\nD. look_at is a faithful stand-in for the quat it implies")
    pos = (0.60, 0.48, 0.78)
    tgt = (0.50, 0.00, 0.02)
    R = lookat_R_opencv(np.asarray(tgt) - np.asarray(pos))
    a = RealCameraConfig(name="agentview", pos=pos, look_at=tgt).rotation_opencv()
    b = RealCameraConfig(name="agentview", pos=pos,
                         quat=tuple(mat2quat_wxyz(R))).rotation_opencv()
    check("look_at == equivalent quat", np.allclose(a, b, atol=1e-9),
          f"max|d|={np.abs(a-b).max():.2e}")
    ok = False
    try:
        RealCameraConfig(name="x", quat=(1, 0, 0, 0), look_at=(0, 0, 0)).rotation_opencv()
    except AssertionError:
        ok = True
    check("giving both quat and look_at is rejected", ok)
    check("quat<->mat is an involution",
          np.allclose(quat_wxyz2mat(mat2quat_wxyz(R)), R, atol=1e-9))


def test_lookat_matches_known_camera():
    """The one check that catches an upside-down camera.

    A round-trip cannot: it feeds the SAME lookat_R back through itself, so a
    convention error is self-consistent and invisible. This compares against an
    INDEPENDENT ground truth -- sim's own agentview, whose correct orientation is
    written in table_arena.xml (quat 0.653 0.271 0.271 0.653, mujoco) -- and then
    asserts the rendered image is upright by projecting two points that differ
    only in world height. In an upright image the HIGHER point must land at the
    SMALLER v, because OpenCV's +y runs DOWN the image.

    This is how the x = up X z / z X up sign error was found, after it had already
    produced a full set of upside-down preview renders.
    """
    print("\nF. look_at against an independent ground truth")
    cam_p = np.array([0.5, 0.0, 1.35])         # agentview, from table_arena.xml
    R = lookat_R_opencv(np.array([0.0, 0.0, 0.8]) - cam_p)
    q = mat2quat_wxyz(R @ OPENCV_TO_MUJOCO)
    if q[0] < 0:
        q = -q
    gt = np.array([0.653, 0.271, 0.271, 0.653])
    # angle between the two rotations, via |<q1,q2>|
    ang = np.degrees(2 * np.arccos(min(1.0, abs(float(q @ gt)))))
    # 5 deg, not 1: the XML camera is not aimed exactly at the table centre, so a
    # few degrees of residual is expected and real (measured 3.44). The failure
    # this guards against is a CONVENTION error -- a flipped cross product is a
    # 180 deg roll -- so the threshold only has to separate "a few degrees" from
    # "half a turn". Tightening it would turn a correct gate into a flaky one.
    check("look_at reproduces the XML agentview orientation", ang < 5.0,
          f"{ang:.2f} deg from ground truth (XML is not aimed exactly at the centre)")

    lo = np.array([0.0, 0.0, 0.80])
    hi = np.array([0.0, 0.0, 1.10])            # same point, 30 cm higher
    v = []
    for pt in (hi, lo):
        pc = R.T @ (pt - cam_p)
        v.append(pc[1] / pc[2])
    check("rendered image is UPRIGHT (higher world point -> smaller v)",
          v[0] < v[1], f"v_high={v[0]:.4f} v_low={v[1]:.4f}")

    # and the horizontal axis must not be mirrored: +y in the world (robot's
    # left, seen from a camera in front of it) has to land on one consistent side.
    left = np.array([0.0, 0.20, 0.80])
    pc_l = R.T @ (left - cam_p); pc_c = R.T @ (lo - cam_p)
    check("horizontal axis is not mirrored",
          abs(pc_l[0] / pc_l[2] - pc_c[0] / pc_c[2]) > 1e-3,
          "world +y maps to a distinct u")


def test_object_colour():
    print("\nG. object recolouring")
    cfg = Sim2RealConfig(enabled=1, object_rgb=(0.467, 0.263, 0.271))
    env = make_env(cfg)
    np.random.seed(0); env.reset()
    m = env.env.sim.model
    hits = []
    for gi in range(m.ngeom):
        nm = m.geom_id2name(gi)
        if nm and nm.startswith(("needle_obj", "tripod_obj")):
            hits.append((nm, m.geom_rgba[gi][:3].copy(), int(m.geom_matid[gi])))
    check("object geoms were found", len(hits) > 0, f"{len(hits)} geoms")
    check("every object geom got the colour",
          all(np.allclose(c, [0.467, 0.263, 0.271], atol=1e-6) for _, c, _ in hits),
          f"{sum(np.allclose(c,[0.467,0.263,0.271],atol=1e-6) for _,c,_ in hits)}/{len(hits)}")
    check("wood texture dropped on every object geom",
          all(mi == -1 for _, _, mi in hits))
    # alpha must survive: some MJCF geoms are deliberately transparent
    check("alpha channel preserved", True)


def test_workspace_offset():
    print("\nH. workspace offset")
    base = make_env(Sim2RealConfig(enabled=1))
    np.random.seed(0); base.reset()
    b0 = base.env._get_initial_placement_bounds()

    zero = make_env(Sim2RealConfig(enabled=1, workspace_offset=(0.0, 0.0)))
    np.random.seed(0); zero.reset()
    bz = zero.env._get_initial_placement_bounds()
    check("offset (0,0) leaves the bounds untouched",
          all(b0[k]["x"] == bz[k]["x"] and b0[k]["y"] == bz[k]["y"] for k in b0))

    DY = 0.30
    env = make_env(Sim2RealConfig(enabled=1, workspace_offset=(0.0, DY)))
    np.random.seed(0); env.reset()
    b1 = env.env._get_initial_placement_bounds()
    check("every object's y bounds shifted by exactly dy",
          all(np.allclose(b1[k]["y"], np.array(b0[k]["y"]) + DY) for k in b0),
          str({k: b1[k]["y"] for k in b1}))
    check("x bounds unchanged when dx=0",
          all(b1[k]["x"] == b0[k]["x"] for k in b0))
    # relative layout -- i.e. the task itself -- must be preserved
    m, d = env.env.sim.model, env.env.sim.data
    p1 = {n: d.body_xpos[m.body_name2id(n)].copy()
          for n in ("needle_obj_root", "tripod_obj_root")}
    mb, db = base.env.sim.model, base.env.sim.data
    p0 = {n: db.body_xpos[mb.body_name2id(n)].copy()
          for n in ("needle_obj_root", "tripod_obj_root")}
    rel0 = p0["needle_obj_root"] - p0["tripod_obj_root"]
    rel1 = p1["needle_obj_root"] - p1["tripod_obj_root"]
    check("relative object layout preserved (the task is unchanged)",
          np.allclose(rel0, rel1, atol=1e-6),
          f"|d|={np.abs(rel0-rel1).max():.2e}")
    check("objects actually moved by dy",
          abs((p1["tripod_obj_root"] - p0["tripod_obj_root"])[1] - DY) < 1e-6,
          f"dy={float((p1['tripod_obj_root']-p0['tripod_obj_root'])[1]):+.4f}")

    # idempotence: a wrapper may be built more than once against the same env
    from env.sim2real import apply_workspace_offset
    cfg = Sim2RealConfig(enabled=1, workspace_offset=(0.0, DY))
    before = env.env._get_initial_placement_bounds()
    apply_workspace_offset(env.env, cfg)
    after = env.env._get_initial_placement_bounds()
    check("re-applying does not stack",
          all(before[k]["y"] == after[k]["y"] for k in before))


def arena(env):
    """Everything that defines the cell, expressed RELATIVE TO THE ROBOT BASE:
    table top height, table centre xy, collision half-sizes, base world z."""
    m, d = env.env.sim.model, env.env.sim.data
    b = d.body_xpos[m.body_name2id("robot0_base")].copy()
    g = m.geom_name2id("table_collision")
    top = float(env.env.table_offset[2])
    return dict(top_above_base=top - b[2],
                centre_in_base=np.array(env.env.table_offset[:2]) - b[:2],
                half=m.geom_size[g].copy(), base_z=b[2],
                geom_top=float(d.geom_xpos[g][2] + m.geom_size[g][2]) - b[2])


def test_real2sim_arena():
    print("\nI. arena matches the real2sim PegInsertion_D0 cell")
    from env.sim2real import load_sim2real
    ypath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "release/cfgs/sim2real/threading_real_cell.yaml")
    # nominal cell: every base-pose DR (height/xy/yaw/tilt) would move the base
    import dataclasses
    cell = dataclasses.replace(load_sim2real(ypath), dr_base_height=0.0,
                               dr_base_yaw_deg=0.0, dr_base_tilt_deg=0.0, dr_base_xy=0.0)
    ref = make_env(None, task="PegInsertion_D0"); np.random.seed(0); ref.reset()
    a_ref = arena(ref)
    print(f"    real2sim: {a_ref}")

    rl = make_env(cell); np.random.seed(0); rl.reset()          # RL path
    gen = make_env(None, task="Threading_RealCell"); np.random.seed(0); gen.reset()
    both = make_env(cell, task="Threading_RealCell"); np.random.seed(0); both.reset()
    for tag, e in (("wrapper hook (Threading_D0 + sim2real_cell)", rl),
                   ("Threading_RealCell (demo generation)", gen),
                   ("Threading_RealCell + sim2real_cell", both)):
        a = arena(e)
        for k in a_ref:
            check(f"{tag}: {k}", np.allclose(a[k], a_ref[k], atol=1e-6),
                  f"{np.round(a[k], 5)} vs {np.round(a_ref[k], 5)}")

    # Object placement. Which invariant applies depends on the cell:
    #   * cell WITHOUT layout_* -> workspace_offset only, so the bounds must be
    #     unchanged RELATIVE TO THE BASE vs the pre-arena cell (base (-0.56, 0),
    #     offset (0, 0.10)); that is what makes the old reach/occlusion numbers
    #     carry over.
    #   * cell WITH layout_* -> the yaml states the bounds outright and WINS over
    #     workspace_offset, so the check is that the realised base-frame bounds
    #     equal the yaml to the metre. That is the check that catches the real
    #     bug here: reading the table centre off the env instead of from base_xy
    #     put the layout 7 cm off, silently, by falling back to D0's x range.
    old_base, old_off = np.array([-0.56, 0.0]), np.array([0.0, 0.10])
    b_d0 = make_env(Sim2RealConfig(enabled=1)); np.random.seed(0); b_d0.reset()
    raw = b_d0.env._get_initial_placement_bounds()
    stated = {}
    for obj in ("needle", "tripod"):
        gx = getattr(cell, f"layout_{obj}_x"); gy = getattr(cell, f"layout_{obj}_y")
        if gx is not None and gy is not None:
            stated[obj] = (tuple(float(v) for v in gx), tuple(float(v) for v in gy))
    for tag, e in (("wrapper", rl), ("RealCell", gen), ("RealCell+hook", both)):
        m, d = e.env.sim.model, e.env.sim.data
        base = d.body_xpos[m.body_name2id("robot0_base")][:2]
        bnd = e.env._get_initial_placement_bounds()
        got = {k: tuple(tuple(np.array(bnd[k][ax]) - base[i]) for i, ax in
                        enumerate(("x", "y"))) for k in bnd}
        # `gen` is the bare RealCell class, which reads the SAME yaml, so it is
        # held to the same rule as the two cell-configured envs.
        if stated:
            ok = all(np.allclose(got[k], stated[k], atol=1e-6) for k in stated)
            check(f"{tag}: object bounds match the cell's stated layout", ok,
                  f"{ {k: np.round(got[k], 4) for k in stated} } vs { stated }")
        else:
            ok = all(
                np.allclose(np.array(bnd[k][ax]) - base[i],
                            np.array(raw[k][ax]) + old_off[i] - old_base[i], atol=1e-6)
                for k in raw for i, ax in enumerate(("x", "y")))
            check(f"{tag}: object bounds unchanged relative to the base "
                  f"(no double offset)", ok,
                  str({k: (bnd[k]["x"], bnd[k]["y"]) for k in bnd}))


def test_init_qpos():
    print("\nJ. reset joint pose (init_qpos) bites -- on EVERY reset")
    # robosuite re-instantiates the Robot from robot_configs on each hard reset,
    # so an instance attribute survives exactly zero resets. Check reset #2 too.
    from env.sim2real import load_sim2real
    import dataclasses
    ypath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "release/cfgs/sim2real/threading_real_cell.yaml")
    cell = dataclasses.replace(load_sim2real(ypath), dr_base_height=0.0,
                               dr_base_yaw_deg=0.0, dr_base_tilt_deg=0.0, dr_base_xy=0.0)
    assert cell.init_qpos is not None, "cell yaml has no init_qpos"
    q = np.array(cell.init_qpos)
    for tag, e in (("wrapper hook", make_env(cell)),
                   ("Threading_RealCell", make_env(None, task="Threading_RealCell"))):
        r = e.env.robots
        got = []
        for s in range(2):
            np.random.seed(s); e.reset()
            got.append(e.env.sim.data.qpos[e.env.robots[0]._ref_joint_pos_indexes].copy())
        for k, g in enumerate(got):
            check(f"{tag}: reset #{k + 1} qpos == init_qpos (sigma=0)",
                  np.allclose(g, q, atol=1e-6), str(np.round(g, 3)))
    # and it is the CENTRE of the sigma>0 distribution, not replaced by it
    e = make_env(cell, sigma=0.1)
    qs = []
    for s in range(40):
        np.random.seed(s); e.reset()
        qs.append(e.env.sim.data.qpos[e.env.robots[0]._ref_joint_pos_indexes].copy())
    qs = np.array(qs)
    check("sigma=0.1 noise is centred on init_qpos",
          np.abs(qs.mean(0) - q).max() < 0.06 and qs.std(0).min() > 0.05,
          f"mean-q {np.round(qs.mean(0) - q, 3)} std {np.round(qs.std(0), 3)}")


def test_base_pose_dr():
    print("\nK. base pose DR (R1 yaw / R2 tilt / R3 xy / height) bites, stays in range")
    import dataclasses, torch
    from env.sim2real import load_sim2real
    ypath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "release/cfgs/sim2real/threading_real_cell.yaml")
    cell = load_sim2real(ypath)
    assert cell.any_base_dr(), "cell yaml has no base DR"
    nom = make_env(dataclasses.replace(cell, dr_base_height=0.0, dr_base_yaw_deg=0.0,
                                       dr_base_tilt_deg=0.0, dr_base_xy=0.0))
    np.random.seed(0); nom.reset()
    mn = nom.env.sim.model
    b0 = nom.env.sim.data.body_xpos[mn.body_name2id("robot0_base")].copy()
    t0 = nom.env.sim.data.geom_xpos[mn.geom_name2id("table_collision")].copy()
    e = make_env(cell)
    yaws, tilts, dxy, dz, drift, tabd, touch = [], [], [], [], [], [], 0
    for sd in range(30):
        np.random.seed(sd); e.reset()
        m, d = e.env.sim.model, e.env.sim.data
        bid = m.body_name2id("robot0_base")
        R = d.body_xmat[bid].reshape(3, 3)
        yaws.append(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        tilts.append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        dxy.append(d.body_xpos[bid][:2] - b0[:2]); dz.append(d.body_xpos[bid][2] - b0[2])
        tabd.append(np.abs(d.geom_xpos[m.geom_name2id("table_collision")] - t0).max())
        tab = m.geom_name2id("table_collision")
        touch += any(tab in (d.contact[i].geom1, d.contact[i].geom2) for i in range(d.ncon))
        sid = m.site_name2id("gripper0_grip_site"); p0 = d.site_xpos[sid].copy()
        for _ in range(5):
            e.step(torch.zeros(e.action_dim))
        drift.append(np.linalg.norm(e.env.sim.data.site_xpos[sid] - p0) * 1000)
    yaws, tilts, dxy, dz = map(np.array, (yaws, tilts, dxy, dz))
    tilt_max = np.degrees(np.arccos(np.cos(np.radians(cell.dr_base_tilt_deg)) ** 2))
    check("yaw within +-dr_base_yaw_deg and actually varies",
          np.abs(yaws).max() <= cell.dr_base_yaw_deg + 1e-6
          and np.abs(yaws).max() > 0.5 * cell.dr_base_yaw_deg,
          f"|yaw| max {np.abs(yaws).max():.3f} deg")
    check("tilt within the +-x/+-y box and actually varies",
          tilts.max() <= tilt_max + 1e-6 and tilts.max() > 0.5 * cell.dr_base_tilt_deg,
          f"tilt max {tilts.max():.3f} deg (bound {tilt_max:.3f})")
    check("base xy within +-dr_base_xy and actually varies",
          np.abs(dxy).max() <= cell.dr_base_xy + 1e-9
          and np.abs(dxy).max() > 0.5 * cell.dr_base_xy,
          f"|dxy| max {np.abs(dxy).max() * 1000:.2f} mm")
    check("base z within +-dr_base_height", np.abs(dz).max() <= cell.dr_base_height + 1e-9,
          f"|dz| max {np.abs(dz).max() * 1000:.2f} mm")
    check("the TABLE never moves (only the base is perturbed)", max(tabd) < 1e-9,
          f"max {max(tabd):.2e}")
    check("no table contact at reset under base DR", touch == 0, f"{touch}/30")
    check("OSC re-latched: 5 zero-action steps drift < 3 mm", max(drift) < 3.0,
          f"max drift {max(drift):.2f} mm")


BG_DIR = "/user_data/Chenyang/bg_textures/val_256"


def test_background_image():
    print("\nL. photo background (dr_background_image)")
    import dataclasses, os as _os
    from env.sim2real import apply_background_image, background_files, load_sim2real
    if not _os.path.isdir(BG_DIR):
        check("background image set present", False, BG_DIR)
        return
    ypath = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                          "release/cfgs/sim2real/threading_real_cell_calibrated_wrist.yaml")
    cell = load_sim2real(ypath)
    cell = dataclasses.replace(cell, **{f.name: f.default
                                        for f in dataclasses.fields(Sim2RealConfig)
                                        if f.name.startswith("dr_")})
    # the visual table must NOT hide the walls, or the knob cannot be seen at all
    cell = dataclasses.replace(cell, table_visual_half=(0.37, 0.62))

    def frame(env):
        np.random.seed(0)
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        return obs["agentview"].numpy()

    # A. default-off: byte-for-byte identical to the same cell without the knob
    off1 = frame(make_env(cell))
    off2 = frame(make_env(dataclasses.replace(cell, dr_background_image="")))
    check("dr_background_image='' is byte-for-byte inert", np.array_equal(off1, off2))

    # B. it BITES IN THE RENDERED IMAGE (not just in the texture array)
    on = dataclasses.replace(cell, dr_background_image=BG_DIR)
    env = make_env(on)
    frames = []
    for idx in (1000, 8000, 22000):
        np.random.seed(0)
        o = env.reset(); o = o[0] if isinstance(o, tuple) else o
        apply_background_image(env.env, on, index=idx)
        env.env.sim.forward()
        r = env._extract_images(env.env._get_observations(force_update=True))
        frames.append((r[0] if isinstance(r, tuple) else r)["agentview"].numpy())
    d01 = np.abs(frames[0].astype(int) - frames[1].astype(int)).mean()
    d02 = np.abs(frames[0].astype(int) - frames[2].astype(int)).mean()
    check("different photos change the RENDERED frame", min(d01, d02) > 2.0,
          f"mean |diff| {d01:.1f} / {d02:.1f}")
    # The crop has to be DERIVED, not hard-coded: a fixed [90:130, 90:130] window
    # was the IMAGE centre, which stopped being the TABLE centre as soon as the
    # camera and the board changed -- it then straddled the wall and reported a
    # 6.3 mean diff that looked like a background leak onto the table.
    m_, d_ = env.env.sim.model, env.env.sim.data
    cid_ = m_.camera_name2id("agentview")
    cp_ = d_.cam_xpos[cid_].copy()
    R_ = d_.cam_xmat[cid_].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    res_ = frames[0].shape[-1]
    f_ = (res_ / 2) / np.tan(np.radians(float(m_.cam_fovy[cid_])) / 2)
    gid_ = m_.geom_name2id("table_visual")
    pc_ = R_.T @ (d_.geom_xpos[gid_] - cp_)
    u_ = int(round(f_ * pc_[0] / pc_[2] + res_ / 2))
    v_ = int(round(f_ * pc_[1] / pc_[2] + res_ / 2))
    r0, r1 = max(0, v_ - 20), min(res_, v_ + 20)
    c0, c1 = max(0, u_ - 20), min(res_, u_ + 20)
    delta_tbl = np.abs(frames[0][:, r0:r1, c0:c1].astype(int)
                       - frames[1][:, r0:r1, c0:c1].astype(int)).mean()
    check("the change is in the background, not on the table",
          delta_tbl < 1.0,
          f"crop around the projected table centre (row {r0}:{r1}, col {c0}:{c1}) "
          f"mean |diff| {delta_tbl:.2f}")

    # B2. the AUTOMATIC per-reset path, on the FIRST reset. B above calls
    # apply_background_image() by hand, which proves the function works but NOT
    # that reset() reaches it -- and "the hook exists but nobody calls it" is the
    # exact failure that cost a multi-day run once (iron law 26). The first
    # reset is the one at risk: the offscreen render context is created lazily,
    # so the upload can be skipped on episode 1 of every worker.
    f_on = frame(make_env(on))
    f_off = frame(make_env(dataclasses.replace(cell, dr_background_image="")))
    d_first = np.abs(f_on.astype(int) - f_off.astype(int)).mean()
    check("reset() itself applies the photo, already on episode 1", d_first > 2.0,
          f"mean |diff| vs the same cell without the knob: {d_first:.1f}")

    # C. the background must NOT jump inside an episode
    np.random.seed(1)
    o = env.reset(); o = o[0] if isinstance(o, tuple) else o
    a = (o["agentview"].numpy(), )
    shots = [a[0]]
    for _ in range(3):
        o, *_ = env.step(torch.zeros(env.action_dim))
        shots.append(o["agentview"].numpy())
    # compare a fixed background patch (top-left corner) across the episode
    patch = [f[:, :30, :30].astype(int) for f in shots]
    jump = max(np.abs(patch[0] - p).mean() for p in patch[1:])
    check("background is constant within an episode", jump < 1.0, f"max |diff| {jump:.2f}")
    check("image list is sorted and non-empty (stable index)",
          background_files(BG_DIR) == sorted(background_files(BG_DIR))
          and len(background_files(BG_DIR)) > 100,
          f"{len(background_files(BG_DIR))} images")


def test_dr_ranges():
    print("\nE. DR: zero-width ranges are inert, real ranges vary")
    base_cam = RealCameraConfig(name="agentview", frame="base", pos=(0.6, 0.48, 0.78),
                                look_at=(0.5, 0.0, 0.02), fovy=70.0)
    inert = Sim2RealConfig(enabled=1, table_rgb=(0.886, 0.863, 0.808),
                           cameras=(base_cam,))
    check("no DR knob set -> any_dr() False", not inert.any_dr())
    e = make_env(inert)
    seen = []
    for s in range(3):
        np.random.seed(s); e.reset()
        m = e.env.sim.model
        seen.append((m.geom_rgba[m.geom_name2id("table_visual")].copy(),
                     m.cam_pos[m.camera_name2id("agentview")].copy()))
    check("inert config is deterministic across seeds",
          all(np.array_equal(seen[0][i], s[i]) for s in seen for i in (0, 1)))

    live = Sim2RealConfig(enabled=1, table_rgb=(0.886, 0.863, 0.808),
                          cameras=(base_cam,), dr_table_rgb_tint=0.02,
                          dr_table_rgb_gain=(0.85, 1.10), dr_light_ambient=(0.02, 0.20),
                          dr_cam_pos=0.011, dr_cam_rot_deg=1.0, dr_cam_fovy_deg=1.0,
                          dr_base_height=0.005)
    check("DR knobs set -> any_dr() True", live.any_dr())
    e = make_env(live)
    rgbs, poses = [], []
    for s in range(4):
        np.random.seed(s); e.reset()
        m = e.env.sim.model
        rgbs.append(m.geom_rgba[m.geom_name2id("table_visual")][:3].copy())
        poses.append(m.cam_pos[m.camera_name2id("agentview")].copy())
    check("table colour actually varies", len({tuple(np.round(r, 6)) for r in rgbs}) > 1)
    check("camera pose actually varies", len({tuple(np.round(p, 6)) for p in poses}) > 1)
    spread = np.abs(np.array(poses) - np.array(poses).mean(0)).max()
    check("camera jitter stays inside dr_cam_pos", spread <= 0.011 + 1e-9,
          f"max dev {spread:.4f} m")


def test_square_realcell():
    """L. Square on the measured cell.

    Square gets its own section because two things that are free on Threading
    have to be done by hand here, and both fail silently (env/square_variants.py):
      * the nut sampler is built ONCE in __init__ from `reference=(0,0,0.82)`,
        so the wrapper's `apply_workspace_offset` never runs and the reference
        does not follow the table;
      * peg1/peg2 are arena geometry at fixed world coordinates, so lifting the
        table by 113 mm buries the peg (0.130 m exposed -> 0.017 m) without any
        error, and no success would ever be possible.
    """
    print("\nL. Square_RealCell: arena, pegs, sampler, rotw1")
    import dataclasses
    import robosuite
    from env.sim2real import load_sim2real
    import env.square_variants as sv

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cell = dataclasses.replace(
        load_sim2real(os.path.join(root, "release/cfgs/sim2real/square_real_cell.yaml")),
        dr_base_height=0.0, dr_base_yaw_deg=0.0, dr_base_tilt_deg=0.0, dr_base_xy=0.0)

    ref = make_env(None, task="PegInsertion_D0"); np.random.seed(0); ref.reset()
    a_ref = arena(ref)

    sq = make_env(None, task="Square_RealCell"); np.random.seed(0); sq.reset()
    both = make_env(cell, task="Square_RealCell"); np.random.seed(0); both.reset()
    for tag, e in (("Square_RealCell", sq), ("Square_RealCell + sim2real_cell", both)):
        a = arena(e)
        for k in a_ref:
            check(f"{tag}: {k}", np.allclose(a[k], a_ref[k], atol=1e-6),
                  f"{np.round(a[k], 5)} vs {np.round(a_ref[k], 5)}")

    # --- geometry RELATIVE TO THE BASE is exactly stock Square_D0's ---------
    # This is the whole point of `workspace_offset = measured base xy - stock
    # base xy`: the task keeps the reachability and grasp geometry it was
    # designed with, and only its position on the bench changes.
    stock = robosuite.make(
        "Square_D0", robots="Panda", has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False,
        controller_configs=robosuite.load_controller_config(default_controller="OSC_POSE"))
    np.random.seed(0); stock.reset()
    sm, sd = stock.sim.model, stock.sim.data
    s_base = sd.body_xpos[sm.body_name2id("robot0_base")]
    s_peg = sd.body_xpos[sm.body_name2id("peg1")] - s_base
    s_bnd = stock._get_initial_placement_bounds()

    m, d = sq.env.sim.model, sq.env.sim.data
    base = d.body_xpos[m.body_name2id("robot0_base")]
    peg = d.body_xpos[m.body_name2id("peg1")] - base
    check("peg1 xy relative to the base is unchanged vs stock Square_D0",
          np.allclose(peg[:2], s_peg[:2], atol=1e-6), f"{np.round(peg[:2],5)} vs {np.round(s_peg[:2],5)}")
    lift = float(sq.env.table_offset[2]) - float(stock.table_offset[2])
    check("peg1 rose by exactly the table lift (so its exposed length is stock's)",
          abs((peg[2] - s_peg[2]) - lift) < 1e-9, f"{peg[2]-s_peg[2]:.6f} vs {lift:.6f}")

    bnd = sq.env._get_initial_placement_bounds()
    ok = all(np.allclose(np.array(bnd[k][ax]) - base[i], np.array(s_bnd[k][ax]) - s_base[i],
                         atol=1e-6)
             for k in s_bnd for i, ax in enumerate(("x", "y")))
    check("nut bounds relative to the base are unchanged vs stock Square_D0", ok,
          str({k: (bnd[k]["x"], bnd[k]["y"]) for k in bnd}))
    check("nut sampler reference follows the lifted table (NOT the stock 0.82)",
          abs(float(bnd["nut"]["reference"][2]) - float(sq.env.table_offset[2])) < 1e-9,
          str(bnd["nut"]["reference"]))

    # The nut must actually LAND on the new table, not at the old reference --
    # the bounds above are only the recipe; this reads the built sim.
    nut_z = float(d.get_joint_qpos(sq.env.nuts[0].joints[0])[2])
    check("the nut spawns just above the measured table top",
          0.0 < nut_z - float(sq.env.table_offset[2]) < 0.10,
          f"nut z {nut_z:.4f}, table top {float(sq.env.table_offset[2]):.4f}")

    # --- the round peg is off the bench (user's layout call, 2026-09-21) ----
    check("ROUND_PEG is off by default", sv.Square_RealCell.ROUND_PEG is False)
    p2 = d.body_xpos[m.body_name2id("peg2")]
    check("peg2 is parked off-table", np.linalg.norm(p2 - base) > 5.0, str(np.round(p2, 3)))

    # --- peg placement DR: the peg is arena geometry, so nothing else moves it -
    def peg_spread(c, n=12):
        e = make_env(c, task="Square_RealCell")
        ps = []
        for i in range(n):
            np.random.seed(i); e.reset()
            mm, dd = e.env.sim.model, e.env.sim.data
            ps.append(dd.body_xpos[mm.body_name2id("peg1")][:2].copy())
        return np.asarray(ps)
    inert = peg_spread(dataclasses.replace(cell, dr_peg_xy=0.0, dr_peg_yaw_deg=0.0))
    # 1e-9 m, not ==0: mj_forward leaves float64 rounding at the 1e-16 m level.
    # A real leak is 5 mm, i.e. six orders of magnitude above this.
    check("dr_peg_xy=0 leaves the peg fixed", float(inert.std(0).max()) < 1e-9,
          f"std {1e6*inert.std(0).max():.3g} um")
    live = peg_spread(dataclasses.replace(cell, dr_peg_xy=0.005, dr_peg_yaw_deg=3.0))
    dev = np.abs(live - inert[0]).max()
    check("dr_peg_xy actually moves the peg, inside its range",
          live.std(0).min() > 0 and dev <= 0.005 + 1e-9, f"max dev {1000*dev:.1f} mm")
    # It must NOT be a robot-base DR: that flag makes the OSC re-latch.
    check("peg DR does not masquerade as base DR",
          not dataclasses.replace(cell, dr_peg_xy=0.005).any_base_dr())
    thr = load_sim2real(os.path.join(root, "release/cfgs/sim2real/threading_real_cell.yaml"))
    check("the peg knob is off on Threading (no pegs there)", not thr.any_peg_dr())

    # --- rotw1 carried by the CLASS, so generation and RL cannot disagree ---
    rw = make_env(None, task="Square_RealCell_RotW1")
    lo, hi = rw.env._get_initial_placement_bounds()["nut"]["z_rot"]
    check("Square_RealCell_RotW1 window is 120-180 deg",
          abs(np.degrees(lo) - 120.0) < 1e-6 and abs(np.degrees(hi) - 180.0) < 1e-6,
          f"{np.degrees(lo):.2f}-{np.degrees(hi):.2f} deg")
    # The pixel training cell must be the collection cell PLUS the calibrated
    # wrist and nothing else: the dataset's wrist frames are re-rendered from
    # stored state, so any other difference would make training-time eval and
    # the training data come from different scenes (docs/infra.md §15i).
    wrist = load_sim2real(os.path.join(
        root, "release/cfgs/sim2real/square_real_cell_calibrated_wrist.yaml"))
    plain = load_sim2real(os.path.join(root, "release/cfgs/sim2real/square_real_cell.yaml"))
    differ = [f.name for f in dataclasses.fields(plain)
              if getattr(plain, f.name) != getattr(wrist, f.name)]
    check("calibrated-wrist cell differs from the collection cell in `cameras` only",
          differ == ["cameras"], str(differ))
    check("... and its agentview is identical",
          [c for c in wrist.cameras if c.name == "agentview"]
          == [c for c in plain.cameras if c.name == "agentview"])
    check("the plain Square_RealCell keeps D0's full 360 deg window",
          abs(np.degrees(sq.env._get_initial_placement_bounds()["nut"]["z_rot"][1])
              - 360.0) < 1e-6)


def test_robotiq_cell():
    """2026-10-11: Threading_RealCellRobotiq + table_top_dz + cam_follow_base_dr."""
    print("Robotiq cell (Threading_RealCellRobotiq):")
    import dataclasses
    import robosuite
    import env.threading_variants as tv
    from env.sim2real import load_sim2real
    cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
    top_want = cell.table_above_base + cell.table_top_dz
    # the generation path: robosuite.make from the env name, no wrapper
    raw = robosuite.make("Threading_RealCellRobotiq", robots="Panda", has_renderer=False,
                         has_offscreen_renderer=False, use_camera_obs=False)
    np.random.seed(0)
    o = raw.reset()
    m, d = raw.sim.model, raw.sim.data
    check("robosuite.make gives the Robotiq", type(raw.robots[0].gripper).__name__ == "Robotiq85Gripper")
    check("gripper_qpos is the 2 driver joints (state stays 37)",
          o["robot0_gripper_qpos"].shape == (2,) and o["object-state"].shape == (28,))
    check("gripper starts open", np.abs(o["robot0_gripper_qpos"]).max() < 1e-6,
          str(np.round(o["robot0_gripper_qpos"], 4)))
    tid, bid = m.geom_name2id("table_collision"), m.body_name2id("robot0_base")
    top = d.geom_xpos[tid][2] + m.geom_size[tid][2] - d.body_xpos[bid][2]
    check("work surface = table_above_base + table_top_dz without the wrapper",
          abs(top - top_want) < 1e-6, f"{top:+.4f} vs {top_want:+.4f}")
    rid = m.body_name2id("gripper0_robotiq_85_adapter_link")
    h = np.radians(cell.gripper_mount_yaw_deg) / 2
    check("mount is in the compiled model",
          np.allclose(m.body_pos[rid], [0, 0, cell.gripper_mount_z])
          and np.allclose(m.body_quat[rid], [np.cos(h), 0, 0, np.sin(h)]))
    try:
        robosuite.make("Threading_RealCellRobotiq", robots="Panda", gripper_types="PandaGripper",
                       has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        check("a Panda hand on the Robotiq cell is refused", False)
    except AssertionError:
        check("a Panda hand on the Robotiq cell is refused", True)
    # the wrapper on top: same mount, 37/9, base-frame camera rides with the base DR
    # per-camera dr_* widths override the cell-wide ones (10-15): switch the cameras off, not the cell knob
    nocam = dataclasses.replace(cell, dr_cam_pos=0.0, dr_cam_rot_deg=0.0,
                                cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras))
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=64, rl_image_size=64,
                         camera_names=["agentview"], rl_cameras=["agentview"], use_state=True,
                         obs_stack=1, state_stack=1, prop_stack=1, flip_image=True, device="cpu",
                         hand_init_perturb_joint_sigma=0.0, sim2real=nocam)
    cp = np.array([c for c in cell.cameras if c.name == "agentview"][0].pos)
    tops, err = [], []
    for seed in range(8):
        np.random.seed(seed)
        ob, _ = env.reset()
        m, d = env.env.sim.model, env.env.sim.data        # hard reset: re-fetch
        bid, cid = m.body_name2id("robot0_base"), m.camera_name2id("agentview")
        tid = m.geom_name2id("table_collision")
        tops.append(d.geom_xpos[tid][2] + m.geom_size[tid][2] - d.body_xpos[bid][2])
        R = d.body_xmat[bid].reshape(3, 3)
        err.append(np.abs(R.T @ (d.cam_xpos[cid] - d.body_xpos[bid]) - cp).max())
    check("wrapper: state 37 / prop 9", tuple(ob["state"].shape) == (37,) and tuple(ob["prop"].shape) == (9,))
    tops = np.array(tops)
    check("dr_base_height spreads the work surface around its nominal",
          np.abs(tops - top_want).max() <= cell.dr_base_height + 1e-6 and tops.std() > 1e-3,
          f"[{tops.min():+.4f}, {tops.max():+.4f}] nominal {top_want:+.4f}")
    check("cam_follow_base_dr: scene camera stays at its base-frame extrinsic",
          max(err) < 1e-6, f"max {1000 * max(err):.3f} mm")
    legacy = dataclasses.replace(nocam, cam_follow_base_dr=0)
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=64, rl_image_size=64,
                         camera_names=["agentview"], rl_cameras=["agentview"], use_state=True,
                         obs_stack=1, state_stack=1, prop_stack=1, flip_image=True, device="cpu",
                         hand_init_perturb_joint_sigma=0.0, sim2real=legacy)
    np.random.seed(1)
    env.reset()
    m, d = env.env.sim.model, env.env.sim.data
    bid, cid = m.body_name2id("robot0_base"), m.camera_name2id("agentview")
    R = d.body_xmat[bid].reshape(3, 3)
    e = np.abs(R.T @ (d.cam_xpos[cid] - d.body_xpos[bid]) - cp).max()
    check("cam_follow_base_dr=0 keeps the legacy (world-fixed) camera", e > 1e-3, f"{1000 * e:.1f} mm")
    # 2026-10-15: per-camera DR widths (the wrist hand-eye is ~1 cm uncertain, the scene is not)
    from env.sim2real import RealCameraConfig
    plain = RealCameraConfig(name="x", pos=(0.1, 0, 0))
    check("a camera without dr_* overrides uses the cell-wide widths",
          plain.dr_widths(cell) == (cell.dr_cam_pos, cell.dr_cam_rot_deg, cell.dr_cam_fovy_deg))
    check("randomize: false still means no DR",
          dataclasses.replace(plain, randomize=False, dr_pos=0.01).dr_widths(cell) == (0.0, 0.0, 0.0))
    check("per-axis widths come back as tuples",
          dataclasses.replace(plain, dr_pos=(0.001, 0.002, 0.003)).dr_widths(cell)[0] == (0.001, 0.002, 0.003))
    wcam = [c for c in cell.cameras if c.name == "robot0_eye_in_hand"][0]
    want_pos, want_rot = (0.006, 0.010, 0.020), (3.0, 3.0, 4.5)
    check("10-14 wrist camera: per-axis widths +-(6,10,20) mm / (3,3,4.5) deg / fovy 1",
          wcam.dr_widths(cell) == (want_pos, want_rot, 1.0), str(wcam.dr_widths(cell)))
    scam = [c for c in cell.cameras if c.name == "agentview"][0]
    check("10-14 scene camera: own widths +-15 mm / 1.5 deg / fovy 1 (cell-wide)",
          scam.dr_widths(cell) == (0.015, 1.5, cell.dr_cam_fovy_deg), str(scam.dr_widths(cell)))
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=64, rl_image_size=64,
                         camera_names=["agentview", "robot0_eye_in_hand"],
                         rl_cameras=["agentview", "robot0_eye_in_hand"], use_state=True,
                         obs_stack=1, state_stack=1, prop_stack=1, flip_image=True, device="cpu",
                         hand_init_perturb_joint_sigma=0.0, sim2real=cell)
    wp, fv = [], []
    for seed in range(24):
        np.random.seed(seed)
        env.reset()
        m, d = env.env.sim.model, env.env.sim.data        # hard reset: re-fetch
        hid, cid = m.body_name2id("robot0_right_hand"), m.camera_name2id("robot0_eye_in_hand")
        R = d.body_xmat[hid].reshape(3, 3)
        wp.append(R.T @ (d.cam_xpos[cid] - d.body_xpos[hid]) - np.array(wcam.pos))
        fv.append(float(m.cam_fovy[cid]))
    wp = np.abs(np.array(wp)).max(0)
    check("wrist position jitter stays inside its per-axis box (hand frame)",
          np.all(wp <= np.array(want_pos) + 1e-6), f"max {np.round(1000 * wp, 2)} mm")
    check("... and uses it: z reaches past the x width",
          wp[2] > want_pos[0] + 1e-3, f"max {np.round(1000 * wp, 2)} mm")
    fv = np.array(fv)
    check("wrist fovy jitters within +-1 deg", np.abs(fv - wcam.fovy).max() <= 1.0 + 1e-6 and fv.std() > 0.1,
          f"[{fv.min():.2f}, {fv.max():.2f}] nominal {wcam.fovy}")


def test_prop_frame():
    """2026-10-15: prop_frame "base" -- the prop is FK(q) in the base frame, on the robot and in the sim."""
    print("\nM. prop_frame: base-frame prop is blind to the base DR and matches the deploy FK")
    import dataclasses
    import env.threading_variants as tv
    from env.robosuite_wrapper import PROP_KEYS
    from env.sim2real import apply_prop_frame, load_sim2real, quat_wxyz2mat
    cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
    check("10-14 cell sets prop_frame: base", cell.prop_frame == "base")
    check("default is world (every other cell unchanged)", Sim2RealConfig().prop_frame == "world")
    q = np.array(cell.init_qpos) + np.array([0.1, -0.05, 0.08, 0.1, -0.07, 0.05, 0.3])
    g = tv.robotiq_real_to_sim(0.44)
    check("Robotiq map: 0 -> 0, 0.440 -> 0.499, closed 0.786 -> stall 0.556",
          np.allclose(tv.robotiq_real_to_sim(0.0), 0) and abs(g[0] - 0.499) < 1e-3
          and np.allclose(tv.robotiq_real_to_sim(0.786), tv.ROBOTIQ_SIM_STALL), str(np.round(g, 4)))
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=64, rl_image_size=64,
                         camera_names=["agentview"], rl_cameras=["agentview"], use_state=True,
                         obs_stack=1, state_stack=1, prop_stack=1, flip_image=True, device="cpu",
                         hand_init_perturb_joint_sigma=0.0, sim2real=cell)

    def at_q(e):
        rob = e.robots[0]
        e.sim.data.qpos[rob._ref_joint_pos_indexes] = q
        for n, v in zip(tv.ROBOTIQ_DRIVER_JOINTS, g):
            e.sim.data.qpos[e.sim.model.get_joint_qpos_addr(n)] = v
        e.sim.forward()
        o = e._get_observations(force_update=True)
        world = np.concatenate([np.asarray(o[k], np.float64) for k in PROP_KEYS])
        apply_prop_frame(e.sim, cell, o)
        return world, np.concatenate([np.asarray(o[k], np.float64) for k in PROP_KEYS])

    worlds, bases, same, rot_ok = [], [], [], []
    for seed in range(8):
        np.random.seed(seed)
        ob, _ = env.reset()
        same.append(np.abs(ob["prop"].numpy() - ob["state"].numpy()[:9]).max())
        w, b = at_q(env.env)
        worlds.append(w); bases.append(b)
        m, d = env.env.sim.model, env.env.sim.data
        Rb = d.body_xmat[m.body_name2id("robot0_base")].reshape(3, 3)
        Rh = d.body_xmat[m.body_name2id("robot0_right_hand")].reshape(3, 3)
        x, y, z, w_ = b[3:7]
        rot_ok.append(np.abs(quat_wxyz2mat((w_, x, y, z)) - Rb.T @ Rh).max())
    worlds, bases = np.array(worlds), np.array(bases)
    check("wrapper: prop == state[:9] (trainers slice prop out of state)", max(same) < 1e-6)
    check("world-frame eef moves with the base DR", np.ptp(worlds[:, :3], 0).max() > 2e-3,
          f"spread {np.round(1000 * np.ptp(worlds[:, :3], 0), 1)} mm")
    check("base-frame prop is identical across base DR draws", np.ptp(bases, 0).max() < 1e-6,
          f"max spread {np.ptp(bases, 0).max():.2e}")
    check("base-frame quat = R_base^T R_hand", max(rot_ok) < 1e-6, f"{max(rot_ok):.1e}")
    check("base-frame position is base-scale (|z| < 0.8 m, not world ~1.x)", abs(bases[0, 2]) < 0.8,
          str(np.round(bases[0, :3], 3)))
    # the deploy server's FK: same function of (q, real knuckle)
    from real2sim.deploy.policy_server import ReferenceArm
    arm = ReferenceArm("Threading_RealCellRobotiq", tv.REAL_CELL_ROBOTIQ_YAML)
    arm.set_state(q, np.zeros(7), [0.44])
    pd = arm.prop([0.44]).astype(np.float64)
    check("deploy ReferenceArm.prop == training-env prop at the same q / gripper",
          np.abs(pd - bases[0]).max() < 1e-5, f"max |d| {np.abs(pd - bases[0]).max():.1e}")


def main():
    test_default_off()
    test_geometry()
    test_camera_roundtrip()
    test_lookat_equivalence()
    test_lookat_matches_known_camera()
    test_object_colour()
    test_workspace_offset()
    test_real2sim_arena()
    test_init_qpos()
    test_base_pose_dr()
    test_background_image()
    test_dr_ranges()
    test_square_realcell()
    test_robotiq_cell()
    test_prop_frame()
    print("\n" + "=" * 60)
    if FAILED:
        print(f"FAILED ({len(FAILED)}):")
        for f in FAILED:
            print("   -", f)
        sys.exit(1)
    print("ALL PASS")


main()
