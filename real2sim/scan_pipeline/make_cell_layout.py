"""Emit cell_layout.json: every cell element's pose in the ROBOT BASE frame.

Reproducible chain (all committed, no hand-entered numbers):
  scan_run2 raw captures -> ground_scan_run2.py -> ground_transforms.json
                                                -> board_model.json
  this script            -> cell_layout.json (+ human-readable printout)

Frames: "base" = Franka base (polymetis FK frame). The intermediate "world"
= ChArUco frame (corner plane z=0.8, center (0.20,0.30) -- a convention that
this script eliminates by re-expressing everything in base).
"""
import json, os
import numpy as np

GS = os.environ.get("GSDIR", "/scratch/yifanr/gsbench")


def inv(M):
    R, t = M[:3, :3], M[:3, 3]
    out = np.eye(4); out[:3, :3] = R.T; out[:3, 3] = -R.T @ t
    return out


def fmt(M):
    t = M[:3, 3]
    z = M[:3, :3] @ [0, 0, 1]
    return f"t=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m, +z_axis->({z[0]:+.3f},{z[1]:+.3f},{z[2]:+.3f})"


def main():
    g = json.load(open(f"{GS}/ground_transforms.json"))
    T_w_base = np.array(g["base"]["T_world_base"])   # base pose in board-world
    T_base_w = inv(T_w_base)                          # board-world -> base

    board = {eval(k): np.array(v) for k, v in json.load(open(f"{GS}/board_model.json")).items()}
    corners_w = np.stack(list(board.values()))
    center_w = np.append(corners_w.mean(0), 1.0)

    T_w_zed2i = np.array(g["T_world_zed2i"])
    T_base_zed2i = T_base_w @ T_w_zed2i
    T_ee_zedm = np.array(g["base"].get("T_ee_zedm", np.eye(4)))

    # ChArUco pose in base frame: same rotation convention as world, recentered
    T_w_charuco = np.eye(4); T_w_charuco[:3, 3] = center_w[:3]
    T_base_charuco = T_base_w @ T_w_charuco

    # table surface = the plane the ChArUco lies on (world z=0.8): express in base
    n_base = T_base_w[:3, :3] @ [0, 0, 1.0]
    p_base = (T_base_w @ [0, 0, 0.8, 1.0])[:3]
    plane_d = float(n_base @ p_base)  # plane: n.x = d in base coords
    tilt = float(np.degrees(np.arccos(abs(n_base[2]))))

    layout = {
        "frame": "franka_base (polymetis FK)",
        "robot_base": np.eye(4).tolist(),
        "charuco_board": {"T_base_board": T_base_charuco.tolist(),
                          "note": "origin = mean of marker corners; corner map in board_model.json"},
        "table_surface": {"normal_in_base": n_base.tolist(), "offset_d": plane_d,
                          "tilt_from_base_z_deg": tilt,
                          "note": "plane {x: n.x = d}; the surface the ChArUco lies on"},
        "zed2i_external": {"T_base_cam": T_base_zed2i.tolist(),
                           "convention": "CV: +z optical axis, +y down"},
        "zedm_wrist_mount": {"T_ee_cam": T_ee_zedm.tolist(),
                             "convention": "CV; rigid EE->camera from hand-eye"},
        "quality": {"base_spread_deg_mm": g["base"].get("spread_deg_mm"),
                    "handeye_method": g["base"].get("method"),
                    "n_poses": g.get("n_poses_used")},
        "task_board": "NOT grounded yet: per-session placement; locate via cameras "
                      "or the scene reconstruction when needed",
        "chain": "scan_run2 -> ground_scan_run2.py -> ground_transforms.json -> this",
    }
    out = f"{GS}/cell_layout.json"
    json.dump(layout, open(out, "w"), indent=2)

    print("=== cell layout, ROBOT BASE frame ===")
    print(f"robot base   : identity (anchor)")
    print(f"ChArUco board: {fmt(T_base_charuco)}")
    print(f"table surface: normal ({n_base[0]:+.3f},{n_base[1]:+.3f},{n_base[2]:+.3f}), "
          f"d={plane_d:+.4f} m, tilt {tilt:.2f} deg from base z")
    print(f"zed2i camera : {fmt(T_base_zed2i)}")
    la = T_base_zed2i[:3, :3] @ [0, 0, 1.0]
    print(f"               optical axis in base: ({la[0]:+.3f},{la[1]:+.3f},{la[2]:+.3f})")
    print(f"zedm mount   : {fmt(T_ee_zedm)} (in EE frame)")
    q = layout["quality"]
    print(f"quality      : base spread {q['base_spread_deg_mm']}, "
          f"method {q['handeye_method']}, poses {q['n_poses']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
