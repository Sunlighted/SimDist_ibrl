# Vertical slice: PolaRiS splats -> robosuite/MuJoCo -> gsplat. **PASSES**, with caveats.
2026-09-13. Jobs 5418508 (agentview), 5436955 (frontview). Code in `/scratch/yifanr/gsbench/`.

## Verdict
The toolchain works end to end: PolaRiS per-link splats + per-body frame calibration +
_GaussianRenderer batched gsplat + robosuite MuJoCo body poses -> a recognisable, correctly
scaled, correctly posed photoreal robot. See `slice/side_link1id_nolink0.png`.
Residual: haze/floaters + per-link rotation error. NOT yet deployment-grade.

## Numbers (frontview, 512^2, silhouette IoU vs MuJoCo's own segmentation)
| config | IoU | splat px | ref px |
|---|---:|---:|---:|
| all_identity | 0.527 | 19,519 | 11,408 |
| all_icp | 0.564 | 19,031 | 11,408 |
| no_link0 | 0.722 | 13,918 | 11,408 |
| **link1=identity + no_link0** | **0.733** | 14,117 | 11,408 |

## What was established
1. **The splats are METRIC.** Link extents match robosuite visual meshes to ~1 cm
   (link5: splat [0.113 0.199 0.312] vs mesh [0.110 0.185 0.311], centroids 0.46 cm apart).
   This matters because PolaRiS ships NO metric-scale code (their issue #23 closed unanswered)
   -- but their *assets* are nevertheless correctly scaled.
2. **Assets are MIT** (hub README front-matter). No licence obstacle.
3. **2DGS surfels are handled**: the SEGMENTED per-link plys carry 2 scales, and
   `util_gau.py:83-85` pads the third with 1e-9. The combined splat.ply is 3-scale.
4. **PolaRiS link frames != robosuite body frames.** Needs a per-body rigid correction.
   The Robotiq parts are decisive: identity RMS is 19-39x worse than the ICP fit
   (42-122 mm vs 1.8-3.2 mm).
5. **panda_link0 must be DROPPED.** Its splat is contaminated with the scanned mounting
   base/surroundings (extent 2-3.5x the mesh). Dropping it moves splat_px 19,333 -> 13,000
   against a 13,092 reference and lifts IoU 0.44 -> 0.53 (agentview) / 0.56 -> 0.72 (frontview).
   ⇒ It is the static base anyway, so it belongs in the BACKGROUND splat, not the per-link set.
6. **panda_link1's ICP solution is a symmetric ALIAS.** 95.6 deg rotation buying only 1.11x
   over identity, and `link1=identity` beats `all_icp` in the render (0.5695 vs 0.5643).

## ⚠ Three methodological traps hit while doing this (all produced confident wrong answers)
- **`geom_pos`/`geom_size` are NOT a mesh centroid/extent.** They are the mesh frame origin and
  bounding half-extent. Using them put link0 15.8 cm out spuriously. Use visual (group==1)
  mesh VERTICES.
- **ICP RMS cannot distinguish a correct fit from a symmetric alias.** 12/15 links scored
  "OK" (<5 mm) including a 178.5 deg flip. The discriminator is RMS-vs-IDENTITY ratio, then a render.
- **Per-link IoU is biased low by occlusion asymmetry.** Rendering one link alone gives it no
  occluders, while the reference mask holds only its *visible* pixels in the full scene =>
  splat_px exceeds ref_px 2-5x for reasons unrelated to alignment. Only whole-robot IoU is fair.
- **Camera choice can silently void a test.** On agentview only the gripper is in frame, so
  `link1_identity` scored bit-identical to `all_icp` -- the config was never exercised.

## Recommendation
**Scan our own arm rather than fighting the generic asset.** PolaRiS's splat is the DROID CAD
unit, registered here against robosuite's Panda+Robotiq85 MJCF -- a third model again. Some
residual is model-vs-model difference, not a solvable transform. We have the ZEDs and exact FK,
so per-link segmentation is cheap and gives our actual hardware's appearance.
Keep PolaRiS's assets as the fallback + the format/convention reference.

**If we do want to refine the generic asset**: gsplat is DIFFERENTIABLE, so per-link (R,t) can be
optimised by gradient descent on silhouette/photometric match against MuJoCo's rasterised render,
rather than by geometric ICP. That sidesteps the symmetry-alias problem entirely, because the
objective is the thing we actually care about.

## Artifacts
- `gs_mujoco.py` -- reusable bridge. NOTE it rotates gaussian ORIENTATIONS as well as positions;
  rotating only xyz leaves every ellipsoid mis-oriented (invisible to a centroid check, very
  visible in a render).
- `check_frames.py` / `calibrate_splat_frames.py` / `render_slice.py` / `splat_frame_calib.json`
- Combined venv at `/scratch/yifanr/gsbench/.venv`: mujoco 2.3.2 + robosuite 1.4.1 +
  numpy 1.26.4 (matching production) + gsplat 1.5.3 + torch 2.7.0+cu128.
  Isolated from `~/venvs/ibrl` so live jobs are untouched.
