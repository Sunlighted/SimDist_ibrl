# _GaussianRenderer (discoverse-dev) throughput — MEASURED on Killarney
Job 5416656, 2026-09-12, 1x L40S (46 GB). Sweep took 5:14.
torch 2.7.0+cu128 / gsplat 1.5.3 — **our exact production env**, no Isaac Sim.
Raw: `/scratch/yifanr/gsbench/results.jsonl`. Harness: `/scratch/yifanr/gsbench/bench_gsrenderer.py`.

RENDERER ONLY — physics deliberately excluded (our MuJoCo step is already
measured at 11.82 ms, docs/infra.md:383). Fixes all four weaknesses of the
committed benchmark: 3 cameras not 1, 224^2 not 640x480, **background rasterised
every step** (not precomputed to bg_imgs), cameras AND bodies jittered per step,
GPU recorded, and gaussian count swept to real-reconstruction scale.

| gaussians | envs | cams | ms/step | trans/s | img/s | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|
| 76,046 | 1 | 3 | 1.98 | 504.8 | 1514.5 | 0.09 |
| 76,046 | 16 | 3 | 4.00 | 3996.3 | 11989.0 | 0.97 |
| 76,046 | 64 | 3 | 13.56 | 4721.2 | 14163.7 | 3.80 |
| 76,046 | 256 | 3 | 53.90 | 4749.8 | 14249.3 | 15.12 |
| **763,293** | 1 | 3 | 3.63 | 275.7 | 827.2 | 0.79 |
| **763,293** | 16 | 3 | 43.51 | 367.8 | 1103.3 | 9.66 |
| **763,293** | 64 | 3 | 171.98 | 372.1 | 1116.4 | 38.06 |
| **763,293** | 64 | 1 | 62.14 | 1029.9 | 1029.9 | 13.73 |
| **763,293** | 64 | 2 | 114.79 | 557.5 | 1115.1 | 25.89 |

## Findings

1. **At real-reconstruction scale the renderer delivers a near-CONSTANT ~1,100
   images/s**, however you split them across cameras (1030 / 1115 / 1116).
   So `transitions/s = 1100 / n_cams`. Cameras are linear and expensive, exactly
   as the code-level prediction in `gs-playground.md:96-99` said.
2. **Throughput saturates from 16 envs** at real scale (367.8 -> 372.1 from 16 to
   64 envs). Past that you buy only VRAM. 16 envs @ 9.66 GB is the operating point.
3. **Cost is LINEAR in gaussian count once the GPU is saturated.** 10x gaussians
   at 16 envs = 10.9x time (4.00 -> 43.51 ms) and 10x VRAM (0.97 -> 9.66 GB).
   ⚠ At 1 env it looks sub-linear (1.98 -> 3.63 ms = 1.83x) — that is a
   GPU-UNDERUTILISATION ARTIFACT, not an algorithmic property. Do not quote it.
4. **VRAM ~12 GB per camera** at 64 envs / 763k. It is the binding constraint,
   not time. 3 cams @ 64 envs = 38 GB of a 46 GB L40S.
5. Committed benchmark said 2210 img/s (1 cam, 640x480, 76k gaussians, bg
   excluded, unknown GPU). We measure 14,249 img/s at 76k/224^2 on an L40S with
   the background rasterised — the shipped number was pessimistic for our config,
   but only because it never tested ours.

## Head-to-head vs PolaRiS (both measured by us, same GPU class, same res)

| | gaussians | cams | envs | ms/step |
|---|---:|---:|---:|---:|
| PolaRiS splat component (full − rtx, 224^2) | ~730k | 2 | 1 | **38.4** |
| _GaussianRenderer (224^2) | 763k | 3 | 1 | **3.63** |

**~10x faster with 50% more cameras (~16x per camera).** Mechanism is in the
source: PolaRiS calls the 2DGS surfel rasteriser once per camera then does a CPU
`np.where` composite with a GPU->CPU round-trip per camera per step; this batches
cameras into one gsplat call and never leaves the GPU.

And the comparison UNDERSTATES the gap, because PolaRiS's 38.4 ms is only its
splat component — its true floor is the 173 ms/step of Isaac physics + USD/fabric
overhead that no configuration change touches (see polaris-throughput-measured.md).

## Time for an 18M-transition source replay

Renderer capacity, 2 cams @224 (our current pixel arm), 763k gaussians:
**557 tps per GPU (MEASURED)** => 18M / 557 = **9.0 h on one GPU**.

⚠ Composition with physics is a CALCULATION, not a measurement. Feeding 557 tps
needs 557 env-steps/s of MuJoCo = ~6.6 workers at 85 steps/s; a node has 64 cores
and 4 GPUs, so physics is not the limiter and 4 GPUs could plausibly sustain
~2200 tps. With JPEG + hdf5 IO on top, budget **4-12 h/node**. MUST BE MEASURED
END-TO-END before being relied on.

| path | one 18M dataset |
|---|---|
| PolaRiS as shipped (splat) | **2.9-13 node-days** |
| our current MuJoCo pipeline (no splat) | 17 h ✓measured end-to-end |
| **MuJoCo + _GaussianRenderer** | **~4-12 h/node** (render 9.0 h/GPU measured) |

=> The photoreal option is NOT slower than what we already run. That is the
decision: PolaRiS's throughput problem is not intrinsic to Gaussian splatting,
it is intrinsic to PolaRiS's renderer and to Isaac Sim's per-step overhead.

## Install notes
Deps are numpy/torch/scipy/plyfile/trimesh/gsplat — ~2 min, no Isaac Sim, no
16 GB venv. ONE trap: `gsplat/cuda/_backend.py:15` imports `packaging` without
declaring it => `pip install packaging`. gsplat JIT-compiles its kernels on first
use and needs nvcc (cudacore 12.9 for the cu128 wheel) + ninja.
