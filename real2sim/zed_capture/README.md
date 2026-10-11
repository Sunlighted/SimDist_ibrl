# Real ZED captures (real2sim grounding inputs)

The images the grounding chain is actually solved from. Committed as **lossless
PNG, byte-identical copies of the originals** — see `.gitignore` for why JPEG was
rejected (q95 moved the external-camera PnP solution 2.3 mm mean / 4.3 mm max,
the same order as the whole chain's 6.1 mm).

## What is here

| path | what | used for |
|---|---|---|
| `scan_run2/pose_XXX/state.json` | joints + `ee_pos`/`ee_quat` (14 poses) | hand-eye |
| `scan_run2/pose_XXX/zedm_left.png` | **wrist** ZED Mini | hand-eye `T_ee_zedm`, wrist-camera pose |
| `scan_run2/pose_XXX/zed2i_left.png` | **external** ZED2i | `T_board_zed2i` |
| `scan_run2/{intrinsics,manifest}.json` | native 2208x1242 calibration | all of the above |
| `calib_20261010/` | 2026-10-10 re-mounted cell: wrist ZED-M refitted intrinsics, scene + wrist extrinsics (two runs), a few raw HD720 PNGs + FR3 joints | `release/cfgs/sim2real/threading_real_cell_20261010.yaml`; see the README inside |
| `Explorer_HD2K_*.png` | external-camera still, side-by-side stereo | independent `T_board_zed2i` check |

## What is NOT here (and where it lives)

`/home/yifanr/scan_run2` — right-eye images, `*_depth.npy`, `*_confidence.npy`
(586 MB). Excluded because the calibration is **2D PnP only**: ZED stereo depth
fits the board to just 18 mm (external) / 33 mm (wrist), against a board model
whose own self-consistency is sub-millimetre, so depth is the weak link and is no
longer used. Also `/home/yifanr/scan_run1` (superseded).

## Traps

- **`ee_quat` is xyzw.**
- **Intrinsics must be scaled from the manifest's OWN `resolution` field**, never
  a hardcoded 1280. scan_run1 stores 1280x720 intrinsics, scan_run2 stores native
  2208x1242 (fx 1057.6). The `width/1280` bug inflated fx to 1824 — a 92.5 deg
  camera described as 62 deg — and PnP absorbs that by moving the camera ~1.7x too
  far, which silently cost this project three separate solves.
- **`zed2i_left.png` has the ROBOT in frame, in a different pose in each of the 14
  captures.** They are calibration inputs ONLY. Never feed them to 3DGS: training
  on them bakes 14 overlapping arm ghosts into the scene splat. The scene splat is
  built from the phone sweeps with robot frames excluded (`docs/real2sim.md` 2f).
  The wrist frames (`zedm_left.png`) are robot-free — verified by cross-frame
  variance, 0.8% static pixels, no row above 40%.
- `polymetis` reports the **flange**, not the TCP: FK(joints) matches `ee_pos` to
  **0.5 mm** at `robot0_right_hand`, vs 96.5 mm at the grip site. So `T_ee_zedm`
  survives a gripper swap (Robotiq -> Panda) as long as the camera bracket itself
  is not disturbed.

## Solutions derived from this data

`real2sim/scan_pipeline/ground_run2_solved.json` (2D PnP; `T_base_board` 6.1 mm /
0.11 deg over 14 poses) and `ground_transforms.json` (depth-based, independent).
The two `T_ee_zedm` agree to **2.1 mm / 0.22 deg**.
