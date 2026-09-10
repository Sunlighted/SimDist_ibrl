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
    p.add_argument("--ring_hole_scale", type=float, default=None,
                   help="放大环孔到 16*s mm, 壁厚保持 4mm (打印用, 见 grow_ring_hole)")
    p.add_argument("--needle_size_scale", type=float, default=None)
    p.add_argument("--square_peg_size_scale", type=float, default=None)
    args = p.parse_args()

    env = build_env(args.task, {
        "ring_size_scale": args.ring_size_scale,
        "ring_hole_scale": args.ring_hole_scale,
        "needle_size_scale": args.needle_size_scale,
        "square_peg_size_scale": args.square_peg_size_scale,
    })
    model = env.sim.model

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

        mesh = trimesh.util.concatenate(parts)
        if args.union:
            try:
                mesh = trimesh.boolean.union(parts, engine="manifold")
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] union failed ({e}); falling back to concatenate")
        mesh.apply_scale(args.scale)

        ext = mesh.bounds[1] - mesh.bounds[0]
        print(f"    bbox(mm) = {np.array2string(ext, precision=2)}   "
              f"watertight={mesh.is_watertight}  vol(cm^3)={mesh.volume / 1000.0:.2f}")

        path = os.path.join(out_dir, f"{nice}.stl")
        mesh.export(path)
        print(f"    wrote {path}")


if __name__ == "__main__":
    main()
