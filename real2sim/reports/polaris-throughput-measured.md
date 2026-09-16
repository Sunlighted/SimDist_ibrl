# PolaRiS throughput — MEASURED on Killarney (job 5394634, 2026-09-11)

Supersedes every estimated per-step figure in `polaris.md` §3 and in the
conversation that preceded it. Hardware: 1x L40S (kn136), driver 580.159.03.
Scene `DROID-FoodBussing`, 2 cameras, random actions perturbing +-0.025 rad
around `default_joint_pos` (ActionCfg is `use_default_offset=False`, i.e.
ABSOLUTE joint targets -- commanding noise around 0 would benchmark the contact
solver thrashing against joint limits). 60 steps timed after 15 warmup steps,
around `torch.cuda.synchronize()`. Raw: `/scratch/yifanr/polaris_bench/results.jsonl`.

| mode | res | envs | ms/step | env steps/s | transitions/s |
|---|---|---:|---:|---:|---:|
| full (as shipped) | 720x1280 | 1 | 251.0 | 3.98 | 3.98 |
| full, rubric detached | 720x1280 | 1 | 250.7 | 3.99 | 3.99 |
| rtx only (no splat) | 720x1280 | 1 | 177.9 | 5.62 | 5.62 |
| full | 224x224 | 1 | 211.3 | 4.73 | 4.73 |
| rtx only | 224x224 | 1 | 172.8 | 5.79 | 5.79 |
| rtx only | 224x224 | 4 | 205.6 | 4.86 | 19.46 |
| rtx only | 224x224 | 16 | 350.0 | 2.86 | 45.71 |

## 1. Cost decomposition (single env)

| component | 720x1280 | 224x224 |
|---|---:|---:|
| splat rasterize + CPU composite (full - rtx) | 73.1 ms | 38.4 ms |
| Isaac RTX render + PhysX | 177.9 ms | 172.9 ms |
| rubric (full - norub) | 0.3 ms | - |

- **The splat is NOT the bottleneck: 29% of the shipped path.** Deleting it
  entirely buys 3.98 -> 5.62 steps/s.
- **~173 ms/step is a floor independent of resolution AND of the splat.**
  PhysX on convexDecomposition collision + USD/fabric writes + sensor update.
- **Resolution is a weak lever.** 18x fewer pixels buys 16% (full) / 2.8% (rtx).
  The splat path IS pixel-sensitive (73 -> 38 ms, sub-linear: Gaussian count
  dominates projection/sorting); Isaac's RTX path is essentially flat.
- **Rubric is free here but that is a FLOOR, not its true cost**: `base.py:52-63`
  short-circuits dependent criteria to False without calling them, and
  `checkers.py:43-46` returns early while the gripper is open. Under a random
  policy the expensive `is_within_xy` bbox path almost never runs.

## 2. Parallel scaling (rtx 224x224, splat off)

| envs | transitions/s | speedup | efficiency |
|---:|---:|---:|---:|
| 1 | 5.79 | 1.00x | 100% |
| 4 | 19.46 | 3.36x | 84% |
| 16 | 45.71 | 7.89x | 49% |

Scales, but decays; not swept past 16. `utils.py:99-100` already parameterises
`num_envs`, so this needed no code change. NOTE: `custom_render` copies only
`data.output["rgb"][0]` -- env 0 -- so per-env GPU->CPU transfer (~301 KB/env/step
at 224^2 x 2 cams) and any encode/write are NOT in these numbers. Upper bound on
collection throughput, not a measurement of it.

## 3. The decisive tension

The splat path is `env_0`-hardcoded (one `big_model`, one camera per rasterizer
call, CPU `np.where` composite), so it cannot ride the parallel scaling. Thus:

- **photoreal (splat): 3.98 steps/s, single env.**
- **parallel (rtx only): 45.7 tps at 16 envs -- but renders the raytraced
  generated meshes**, which PolaRiS's own visual-fidelity ablation ranks below
  splat, and visual fidelity is our UNREPAIRABLE axis (stage 3 refits dynamics
  only). See `stage3-repairs-dynamics-not-vision`.

Cost of a SimDist-scale source replay (18M transitions, ref `gen_source_replay.slrm`):

| path | single process |
|---|---:|
| PolaRiS shipped (splat) | **52 days** |
| PolaRiS rtx-only, 16 envs | 4.6 days |
| our MuJoCo pipeline (295 tps, 10 workers) | **17 h** |

## 4. Install notes (all three are login-vs-compute-node traps)

1. `flatdict==4.0.1` sdist imports `pkg_resources` without declaring setuptools
   => `[tool.uv.extra-build-dependencies] flatdict = ["setuptools<81"]`.
2. PolaRiS marks the nvidia index `explicit = true` and routes only `isaacsim`
   to it => all 25 `isaacsim-*` subpackages resolve from PyPI as wheel-STUBS,
   whose build backend computes tags with `packaging`, which on the login node
   emits NO manylinux tags => rejects every real wheel. Drop `explicit`.
3. `libGLU.so.1` is absent on the Ubuntu 24.04 compute nodes (present only in
   the CVMFS gentoo tree). Without it: libneuray -> MDL-SDK -> rtx.neuraylib all
   fail and the RTX raytracing shaders NEVER BUILD -- the renderer is silently
   broken but the job still runs to completion and emits well-formed timings.
   Fix = shim dir with ONLY libGLU/libOpenGL/libGLdispatch from Ubuntu's own
   debs (never the gentoo lib64: it carries libc/libstdc++ and would shadow
   Isaac Sim's stack). Ubuntu's libGLdispatch needs GLIBC_2.38, which the login
   node (2.37) lacks and the compute node (2.39) has -- so the job PROBES both
   shim dirs with the venv interpreter rather than assuming.
   ⚠ The job now aborts if it ever sees `Failed to load MDL-SDK` or
   `buildShaderAsync() failed`. Without that guard this produces fast, plausible,
   entirely invalid numbers.

Env: torch 2.9.1+cu130, isaacsim 5.1.0.0, isaaclab 2.3.0, warp 1.17.0,
16 GB venv at `/scratch/yifanr/polaris_bench/gh-polaris/.venv`.
Rasterizer JIT-compiles on FIRST IMPORT and requires a live GPU (216 s cold,
cached in `~/.cache/torch_extensions/`) -- it cannot be prebuilt on the login node.
