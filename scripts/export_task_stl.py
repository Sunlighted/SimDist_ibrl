"""Export task objects (Square / Threading) from the MuJoCo model to STL for 3D printing.

Both tasks' objects are *procedural* -- robosuite's SquareNut and MimicGen's
NeedleObject / RingTripodObject are composites of box/capsule primitives, so
there is no mesh file on disk to grab. This script builds the real env (so any
domain_shift geometry knob -- ring_size_scale, needle_size_scale,
square_peg_size_scale -- is baked in), reads the collision geoms of the
requested body out of mjModel, converts each primitive to a triangle mesh in
the *body-local* frame, and writes one STL per body.

Units: MuJoCo is metres; STL is emitted in millimetres by default (--scale).

Examples:
    python scripts/export_task_stl.py --task Square --list
    python scripts/export_task_stl.py --task Square   --out_dir stl/square
    python scripts/export_task_stl.py --task Threading --out_dir stl/threading
    python scripts/export_task_stl.py --task Threading --ring_size_scale 0.9 \
        --out_dir stl/threading_ring090
"""

import argparse
import os
import re

import numpy as np
import trimesh

import robosuite  # noqa: F401  (registers envs)
import mimicgen  # noqa: F401  (registers MimicGen envs)
from robosuite.controllers import load_controller_config


# mjtGeom
PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = range(8)

# body -> which task, and a friendly output filename
TASK_BODIES = {
    "Square": [
        ("SquareNut_main", "square_nut"),
        ("peg1", "square_peg"),
    ],
    "Threading": [
        ("needle_obj_root", "needle"),
        ("tripod_obj_root", "ring_tripod"),
    ],
}

TASK_ENV = {
    "Square": "NutAssemblySquare",
    "Threading": "Threading_D0",
}


def quat_to_mat(q):
    """MuJoCo wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def geom_to_mesh(model, g, sections=48):
    """One mjModel geom -> trimesh in the geom's own frame (centred at origin)."""
    gtype = int(model.geom_type[g])
    size = np.asarray(model.geom_size[g], dtype=float)

    if gtype == BOX:
        return trimesh.creation.box(extents=2.0 * size[:3])
    if gtype == SPHERE:
        return trimesh.creation.icosphere(subdivisions=3, radius=float(size[0]))
    if gtype == CYLINDER:
        return trimesh.creation.cylinder(
            radius=float(size[0]), height=2.0 * float(size[1]), sections=sections
        )
    if gtype == CAPSULE:
        m = trimesh.creation.capsule(
            radius=float(size[0]), height=2.0 * float(size[1]), count=[sections // 2, sections]
        )
        # trimesh's capsule starts at the origin and runs +z; re-centre it the
        # way MuJoCo defines a capsule (midpoint of the cylinder at the origin).
        m.apply_translation(-m.bounds.mean(axis=0))
        return m
    if gtype == ELLIPSOID:
        m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        m.apply_scale(size[:3])
        return m
    if gtype == MESH:
        mid = int(model.geom_dataid[g])
        vadr, vnum = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fadr, fnum = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        verts = np.asarray(model.mesh_vert[vadr:vadr + vnum], dtype=float).reshape(-1, 3)
        faces = np.asarray(model.mesh_face[fadr:fadr + fnum], dtype=np.int64).reshape(-1, 3)
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return None  # plane / hfield: not printable


def collect_body_geoms(model, body_name, collision_only=True, name_filter=None):
    """All geoms whose body is `body_name`, as meshes in the *body* frame."""
    bid = None
    for i in range(model.nbody):
        if model.body_id2name(i) == body_name:
            bid = i
            break
    if bid is None:
        raise KeyError(f"body '{body_name}' not in model")

    parts, rows = [], []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        gname = model.geom_id2name(g) or f"<geom{g}>"
        is_col = int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0
        if collision_only and not is_col:
            continue
        if name_filter and not re.search(name_filter, gname):
            continue
        m = geom_to_mesh(model, g)
        if m is None:
            continue
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(model.geom_quat[g])
        T[:3, 3] = model.geom_pos[g]
        m.apply_transform(T)
        parts.append(m)
        rows.append((gname, int(model.geom_type[g]), np.asarray(model.geom_size[g])[:3].copy()))
    return parts, rows



# ---- 环孔放大 (打印用) -------------------------------------------------
# env/domain_shift.py 的 apply_ring_shift 只支持 *缩小* 孔径: 它把每个 cell 的
# y/z 半尺寸 += 8*(1-s) mm, cell 半尺寸只有 2 mm, 所以 s>1.25 会得到负尺寸,
# 而 s=1.15 会把壁厚从 4 mm 削到 1.6 mm -- 对实物是反效果。
# 这里改用"整个方框往外长": 壁厚恒为 4 mm, 孔心高度不变, 孔径 = 16*s。
RING_CELL = 4.0        # mm, 单个 cell 边长 (= 环壁厚)
RING_HOLE = 16.0       # mm, 名义孔径


def grow_ring_hole(model, body_name, scale):
    """把 6x6 环 pattern 重排成孔径 16*scale、壁厚 4 mm 的方框 (原地改 mjModel)。

    环由 6x6 网格挖掉内 4x4 得到的 20 个 cell 组成。沿每根轴的 6 个槽位重映射为
        槽 0   : 外壁, 半宽 2 mm,      中心 -(8s+2)
        槽 1-4 : 内跨, 半宽 2s mm,     四等分 [-8s, +8s]
        槽 5   : 外壁, 半宽 2 mm,      中心 +(8s+2)
    s=1 时逐值还原为原始 pattern (中心 -10/-6/-2/+2/+6/+10, 半宽全 2)。
    """
    s = float(scale)
    if s == 1.0:
        return
    bid = [i for i in range(model.nbody) if model.body_id2name(i) == body_name][0]
    ring = [g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) == bid and "ring" in (model.geom_id2name(g) or "")]
    if not ring:
        raise RuntimeError(f"no ring geoms on {body_name}")

    # 用名义坐标反推每个 cell 的 (行, 列) 槽位索引
    nominal = np.array([-10.0, -6.0, -2.0, 2.0, 6.0, 10.0]) / 1000.0
    h = RING_HOLE * s / 2000.0        # 半孔径 (m)
    w = RING_CELL / 2000.0            # 壁半厚 (m)

    def remap(slot):
        if slot == 0:
            return -(h + w), w
        if slot == 5:
            return +(h + w), w
        q = (RING_HOLE * s / 4.0) / 1000.0          # 内跨每格宽
        return -h + q * (slot - 1) + q / 2.0, q / 2.0

    zc = np.mean([model.geom_pos[g][2] for g in ring])   # 孔心高度, 保持不变
    for g in ring:
        py, pz = float(model.geom_pos[g][1]), float(model.geom_pos[g][2]) - zc
        jy = int(np.argmin(np.abs(nominal - py)))
        jz = int(np.argmin(np.abs(nominal - pz)))
        cy, sy = remap(jy)
        cz, sz = remap(jz)
        model.geom_pos[g][1] = cy
        model.geom_pos[g][2] = cz + zc
        model.geom_size[g][1] = sy
        model.geom_size[g][2] = sz



def weld_tripod_legs(model, body_name, overlap=3.0):
    """把悬空的三条腿接到底板上 (MimicGen 资产 bug 的修复)。

    ring_tripod.py 用【未旋转】的 capsule 高度 2*(r+h) 定位底板, 但腿随后被转了
    30 度, 竖直半高从 (h+r) 缩到 (h*cos30 + r), 于是腿顶离底板差
    h*(1-cos30) = 4.02 mm -- 同一刚体里 MuJoCo 不在乎, 打印出来是散件。

    修法: 沿腿自身轴线把半轴长 h 加 delta, 同时把 capsule 中心也沿轴上移
    delta。两者相等时【底端 z 完全不动】(上端则抬 2*delta*cos(tilt)), 所以
    三个落地点、footprint、以及环相对桌面的高度都逐点不变 —— 只有埋进底板的
    那一截变长。
    """
    bid = [i for i in range(model.nbody) if model.body_id2name(i) == body_name][0]
    gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid]
    caps = [g for g in gs if int(model.geom_type[g]) == 3]
    if not caps:
        return 0.0
    # 主体底面 = 所有非 capsule geom 的最低 z
    body_bottom = min(float(model.geom_pos[g][2]) - float(model.geom_size[g][2])
                      for g in gs if int(model.geom_type[g]) == 6)
    moved = 0.0
    for g in caps:
        R = quat_to_mat(model.geom_quat[g])
        axis = R[:, 2]                              # capsule 沿自身 z
        if axis[2] < 0:
            axis = -axis                            # 取朝上的一端
        r = float(model.geom_size[g][0]); h = float(model.geom_size[g][1])
        cz = float(model.geom_pos[g][2])
        top = cz + h * abs(axis[2]) + r
        rise = (body_bottom - top) + overlap / 1000.0
        if rise <= 0:
            continue
        delta = rise / (2.0 * abs(axis[2]))
        model.geom_size[g][1] = h + delta
        model.geom_pos[g] = np.asarray(model.geom_pos[g]) + delta * axis
        moved = delta * 1000.0
    return moved



# ---- 腿: 平底脚 + 拆件 --------------------------------------------------
LEG_STUB_R   = 0.004     # m, 定位凸台半径 (直径 8 mm)
LEG_STUB_LEN = 0.007     # m, 凸台伸出长度
LEG_HOLE_R   = 0.0042    # m, 盲孔半径 (单边 0.2 mm 配合间隙)
LEG_HOLE_LEN = 0.0075    # m, 盲孔深度


def leg_axes(model, body_name, collision_only=True):
    """三条腿的 (geom id, 朝上的单位轴, 中心, r, 半轴长) —— 全部米制、body 系。

    ⚠️ 每条腿有 `tripod_N` 和 `tripod_N_vis` 两个 capsule (碰撞 + 纯视觉, 几何完全
    重合)。做几何查询时必须只取碰撞的那 3 个, 否则会拿到 6 条"腿"; 而改 mjModel
    的那两个函数 (weld / extend) 要传 collision_only=False, 让视觉副本跟着一起动。
    """
    bid = [i for i in range(model.nbody) if model.body_id2name(i) == body_name][0]
    out = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid or int(model.geom_type[g]) != CAPSULE:
            continue
        if collision_only and not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
            continue
        axis = quat_to_mat(model.geom_quat[g])[:, 2]
        if axis[2] < 0:
            axis = -axis
        out.append((g, axis, np.asarray(model.geom_pos[g], float),
                    float(model.geom_size[g][0]), float(model.geom_size[g][1])))
    return out


def body_min_z(model, body_name):
    """整个 body 的最低 z (米, body 系) —— 即落在桌面上的那个高度。"""
    bid = [i for i in range(model.nbody) if model.body_id2name(i) == body_name][0]
    lo = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        if int(model.geom_type[g]) == CAPSULE:
            axis = quat_to_mat(model.geom_quat[g])[:, 2]
            r, h = float(model.geom_size[g][0]), float(model.geom_size[g][1])
            lo.append(float(model.geom_pos[g][2]) - h * abs(axis[2]) - r)
        elif int(model.geom_type[g]) == BOX:
            lo.append(float(model.geom_pos[g][2]) - float(model.geom_size[g][2]))
    return min(lo)


def extend_legs_down(model, body_name, depth):
    """把每条腿沿自身轴线【向下】延长, 使 depth mm 的料露在原最低面以下。

    随后在原最低面处平切, 就得到一个有限面积的接地平台, 而【落地高度不变】
    —— 环相对桌面的 189.00 mm 是间隙预算的一部分, 不能被切平动到。
    只加长下端: 半轴长 h += d/2 且中心沿轴下移 d/2, 上端 z 不动。
    """
    d = depth / 1000.0
    for g, axis, pos, r, h in leg_axes(model, body_name, collision_only=False):
        delta = d / abs(axis[2])          # 沿轴要伸出的长度
        model.geom_size[g][1] = h + delta / 2.0
        model.geom_pos[g] = pos - (delta / 2.0) * axis


def _axis_point_at_z(axis, center, z):
    """腿的轴线与水平面 z 的交点。"""
    return center + ((z - center[2]) / axis[2]) * axis


def _rod(axis, base, length, radius, extra_back=0.02):
    """沿 axis、从 base 起长 length 的圆柱 (向后多伸 extra_back 以保证布尔干净)。"""
    total = length + extra_back
    m = trimesh.creation.cylinder(radius=radius, height=total, sections=64)
    T = np.eye(4)
    T[:3, :3] = trimesh.geometry.align_vectors([0, 0, 1], axis)[:3, :3]
    T[:3, 3] = base + axis * (length - extra_back) / 2.0
    m.apply_transform(T)
    return m


def _halfspace_below(ref_mesh, z, pad=0.5):
    """一个盖住 ref_mesh 横截面、z 方向从很低一直到 z 的大盒子 (用于平切)。"""
    lo, hi = ref_mesh.bounds
    ext = [hi[0] - lo[0] + 2 * pad, hi[1] - lo[1] + 2 * pad, (z - (lo[2] - pad))]
    m = trimesh.creation.box(extents=ext)
    m.apply_translation([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] - pad + z) / 2])
    return m


def _boolean(op, meshes):
    return getattr(trimesh.boolean, op)(meshes, engine="manifold")


def split_tripod(model, body_name, parts, rows, cut_z, plate_z):
    """拆成 (主体, 单条腿)。主体 = 底板+立柱+环 挖三个定位盲孔; 腿 = 一条, 打 3 份。

    两件的【外表面几何完全来自原模型】, 新增的只有埋在材料内部的凸台/盲孔。
    """
    caps = [i for i, (_, gt, _) in enumerate(rows) if gt == CAPSULE]
    boxes = [i for i in range(len(rows)) if i not in caps]
    body = _boolean("union", [parts[i] for i in boxes])

    legs_info = leg_axes(model, body_name)
    holes, stubs = [], []
    for _, axis, center, _, _ in legs_info:
        base = _axis_point_at_z(axis, center, plate_z)
        holes.append(_rod(axis, base, LEG_HOLE_LEN, LEG_HOLE_R))
        stubs.append(_rod(axis, base, LEG_STUB_LEN, LEG_STUB_R))
    body = _boolean("difference", [body] + holes)

    # 一条腿: 上端切到底板底面, 下端切出平底, 再接上定位凸台
    leg = parts[caps[0]]
    leg = _boolean("difference", [leg, _halfspace_below(leg, cut_z)])          # 平底
    top_cut = _halfspace_below(leg, plate_z)
    leg = _boolean("intersection", [leg, top_cut])                             # 切掉埋进底板的一截
    leg = _boolean("union", [leg, stubs[0]])
    leg = _boolean("difference", [leg, _halfspace_below(leg, cut_z)])          # 凸台不该越过底面

    # 把腿转正: 轴朝 +Z, 平底落在 z=0
    axis = legs_info[0][1]
    T = np.eye(4)
    T[:3, :3] = trimesh.geometry.align_vectors(axis, [0, 0, 1])[:3, :3]
    leg.apply_transform(T)
    leg.apply_translation([0, 0, -leg.bounds[0][2]])
    return body, leg


def build_env(task, shift_kwargs):
    """Build the robosuite env and apply the project's domain-shift knobs."""
    env = robosuite.make(
        env_name=TASK_ENV[task],
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_offscreen_renderer=False,
        use_camera_obs=False,
        has_renderer=False,
        horizon=10,
    )
    env.reset()

    active = {k: v for k, v in shift_kwargs.items() if v is not None}
    if active:
        from env import domain_shift as ds

        bindings = ds.resolve_bindings(env)
        if "ring_size_scale" in active:
            ds.apply_ring_shift(env, bindings, ring_size_scale=active["ring_size_scale"])
        if "needle_size_scale" in active:
            ds.apply_needle_shift(env, bindings, needle_size_scale=active["needle_size_scale"])
        if active.get("weld_legs"):
            d = weld_tripod_legs(env.sim.model, "tripod_obj_root",
                                 overlap=active.get("leg_overlap", 3.0))
            print(f"[weld_legs] 每条腿沿轴延长 {d:.3f} mm (底端不动)")
        if "flat_feet" in active:
            z0 = body_min_z(env.sim.model, "tripod_obj_root")
            extend_legs_down(env.sim.model, "tripod_obj_root", active["flat_feet"])
            env._stl_cut_z = z0
            print(f"[flat_feet] 腿向下补长 {active['flat_feet']:.2f} mm, "
                  f"在 z={z0*1000:.3f} mm 处平切 (落地高度不变)")
        if "ring_hole_scale" in active:
            grow_ring_hole(env.sim.model, "tripod_obj_root", active["ring_hole_scale"])
        if "square_peg_size_scale" in active:
            ds.apply_square_peg_shift(env, bindings, peg_size_scale=active["square_peg_size_scale"])
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=sorted(TASK_BODIES), required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--scale", type=float, default=1000.0, help="m -> mm")
    p.add_argument("--list", action="store_true", help="just print bodies/geoms and exit")
    p.add_argument("--union", action="store_true", help="boolean-union the parts (needs manifold3d)")
    p.add_argument("--all_geoms", action="store_true", help="include visual-only geoms too")
    p.add_argument("--ring_size_scale", type=float, default=None,
                   help="sim 的收缩旋钮 (apply_ring_shift); 只支持 <1")
    p.add_argument("--weld_legs", action="store_true",
                   help="修复 MimicGen 资产 bug: 三条腿悬空 4.02mm, 打印会散")
    p.add_argument("--leg_overlap", type=float, default=3.0,
                   help="weld_legs 时腿埋进底板的深度 mm (默认 3)")
    p.add_argument("--flat_feet", type=float, default=None, metavar="MM",
                   help="把球面脚切出平台 (深 MM), 同时补长使落地高度不变")
    p.add_argument("--split_legs", action="store_true",
                   help="拆成 tripod_body.stl + tripod_leg.stl(打3份), 避开底板悬垂支撑")
    p.add_argument("--ring_hole_scale", type=float, default=None,
                   help="放大环孔到 16*s mm, 壁厚保持 4mm (打印用, 见 grow_ring_hole)")
    p.add_argument("--needle_size_scale", type=float, default=None)
    p.add_argument("--square_peg_size_scale", type=float, default=None)
    args = p.parse_args()

    env = build_env(args.task, {
        "ring_size_scale": args.ring_size_scale,
        "ring_hole_scale": args.ring_hole_scale,
        "weld_legs": args.weld_legs or None,
        "flat_feet": args.flat_feet,
        "leg_overlap": args.leg_overlap if args.weld_legs else None,
        "needle_size_scale": args.needle_size_scale,
        "square_peg_size_scale": args.square_peg_size_scale,
    })
    model = env.sim.model
    cut_z = getattr(env, "_stl_cut_z", None)

    if args.list:
        print(f"--- bodies in {TASK_ENV[args.task]} ---")
        for i in range(model.nbody):
            n = model.body_id2name(i)
            ng = int((model.geom_bodyid == i).sum())
            if ng and n and not n.startswith(("robot0", "gripper0", "world")):
                print(f"  {n:<28s} geoms={ng}")
        return

    out_dir = args.out_dir or f"stl/{args.task.lower()}"
    os.makedirs(out_dir, exist_ok=True)

    for body_name, nice in TASK_BODIES[args.task]:
        parts, rows = collect_body_geoms(model, body_name, collision_only=not args.all_geoms)
        if not parts:
            print(f"[skip] {body_name}: no printable geoms")
            continue

        print(f"\n=== {body_name} -> {nice}.stl  ({len(parts)} geoms) ===")
        tname = {BOX: "box", CAPSULE: "capsule", CYLINDER: "cylinder", SPHERE: "sphere", MESH: "mesh"}
        shown = rows if len(rows) <= 12 else rows[:6] + [("...", -1, np.zeros(3))] + rows[-3:]
        for gname, gt, sz in shown:
            if gt < 0:
                print(f"    ... ({len(rows) - 9} more)")
                continue
            print(f"    {gname:<26s} {tname.get(gt, gt):<9s} full-size(mm) = "
                  f"{np.array2string(2000.0 * sz, precision=2, suppress_small=True)}")

        is_tripod = body_name == "tripod_obj_root"
        plate_z = None
        if is_tripod:
            plate_z = min(float(model.geom_pos[g][2]) - float(model.geom_size[g][2])
                          for g in range(model.ngeom)
                          if model.geom_id2name(g) and "post" in model.geom_id2name(g)
                          and int(model.geom_type[g]) == BOX
                          and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])))

        outputs = []
        if is_tripod and args.split_legs:
            b, l = split_tripod(model, body_name, parts, rows,
                                cut_z if cut_z is not None else body_min_z(model, body_name),
                                plate_z)
            outputs = [("tripod_body", b), ("tripod_leg", l)]
        else:
            mesh = trimesh.util.concatenate(parts)
            if args.union or args.flat_feet is not None:
                try:
                    mesh = trimesh.boolean.union(parts, engine="manifold")
                except Exception as e:  # noqa: BLE001
                    print(f"    [warn] union failed ({e}); falling back to concatenate")
            if is_tripod and cut_z is not None:
                mesh = _boolean("difference", [mesh, _halfspace_below(mesh, cut_z)])
            outputs = [(nice, mesh)]

        for nice_i, mesh in outputs:
            _report_and_write(mesh, args.scale, out_dir, nice_i,
                              copies=3 if nice_i == "tripod_leg" else 1)


def _report_and_write(mesh, scale, out_dir, nice, copies=1):
        mesh = mesh.copy()
        mesh.apply_scale(scale)
        ncomp = len(mesh.split(only_watertight=False))
        flag = "" if ncomp == 1 else f"  <<< 警告: {ncomp} 个分离块, 打印会散!"
        ext = mesh.bounds[1] - mesh.bounds[0]
        cp = f"  [打 {copies} 份]" if copies > 1 else ""
        print(f"    {nice}: bbox(mm) = {np.array2string(ext, precision=2)}   "
              f"watertight={mesh.is_watertight}  vol(cm^3)={mesh.volume / 1000.0:.2f}  "
              f"连通块={ncomp}{flag}{cp}")

        path = os.path.join(out_dir, f"{nice}.stl")
        mesh.export(path)
        print(f"    wrote {path}")


if __name__ == "__main__":
    main()
