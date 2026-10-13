# 2026-10-13 camera calibration (Threading cell, NEW scene camera)

The external ZED 2i was **replaced**: SN 37597132 → **SN 31011113** (same 2.1 mm model). The wrist
ZED-M (SN 16478414) is unchanged. Mode for both: HD720 @ 30 fps, raw (unrectified) left eye.
Supersedes `../calib_20261010/` for the scene camera; tools and the wrist intrinsics are reused from
there (`../calib_20261010/tools/`, `../calib_20261010/data/wrist_intrinsics.json`).

Cell config: `threading_real_cell_20261013.yaml` (= the 10-10 cell with only `cameras` replaced).

## Files

| path | what |
|---|---|
| `data/run_1013_1757/extrinsics.json` | `T_base_scene` (ZED 2i in `robot0_base`), `T_hand_wrist` (ZED-M in `robot0_right_hand`), OpenCV optical axes. 25 of 27 robot poses (2 had < 10 wrist corners). |
| `data/run_1013_1757/sample_000`, `sample_020` | scene + wrist raw PNG + FR3 joints. |
| `data/SN31011113_factory.conf` | Stereolabs factory calibration of the new camera (downloaded with `ZED_Explorer --dc 31011113`). Its LEFT_CAM_HD / VGA / FHD values are now the `scene` entry of `release/cfgs/sim2real/zed_factory_intrinsics.json`; the old camera is kept there as `scene_SN37597132`. |
| `data/scene_intrinsics_refit.json` | Hand-held refit of the new ZED 2i (27 views, 9 tilted ≥ 35°), kept for reference only. |
| `figs/` | gate C overlays (`tools/overlay_real_scene.py`) and the fovy-50 / 224 / max-square crops of the new view. |

## Numbers

| | value |
|---|---|
| new ZED 2i, factory vs hand-held refit | fx 534.9 vs 540.0, cx 629.4 vs 624.2, cy 346.2 vs 343.7; bootstrap sd of the refit fx 3.6 / cx 4.1 / cy 3.1 px ⇒ not distinguishable; own-PnP 0.36 px (factory) vs 0.30 px (refit) ⇒ **factory used** |
| scene reprojection | median 0.33 px, p90 0.75 px; the board did not move during the run (every view 0.27–0.58 px against the pooled pose) |
| scene pose, leave-one-out | ≤ 1.4 mm / 0.12° |
| wrist pose, leave-one-out | ≤ 1.4 mm / 0.16° |
| board square size (solved) | 42.02 mm |

## ⚠️ The wrist camera mount is drifting

`T_hand_wrist` solved from three sessions with the **same** wrist intrinsics, relative to today's,
in camera axes:

| session | rotation (deg, x y z) | translation (mm) |
|---|---|---|
| run_1010 (10-10) | [-0.39, 0.35, **-1.91**] | [5.5, -8.4, -1.3] |
| run_1010_2327 (10-10 late) | [-0.19, 0.50, **-0.83**] | [0.5, -3.4, -1.6] |
| run_1013_1757 (10-13) | 0 | 0 |

Each session is internally consistent to ~1.5 mm / 0.16°, but the pose moves **monotonically**,
almost entirely as a roll about the optical axis: the camera is very likely turning on its mount
(loosening screw). Tighten / pin the mount, then re-check with ~10 robot poses; until then the
wrist pose is only good to ~1°, and the cell has `randomize: false` for the wrist camera.

## Notes

- The new ZED 2i first enumerated on USB 2 (480M ⇒ VGA only, and the ZED SDK does not list it);
  the fix was the cable / Type-C seating, not the port. Check `.../speed` = 5000.
- The 65.8° crop of the new view shows clutter (balls, bowls) at the lower left that the sim does
  not have; at fovy 50 only the bowl corner enters. Clear it before collecting data.
