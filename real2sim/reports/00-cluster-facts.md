# Killarney compute-node facts (MEASURED 2026-09-10, jobs 5349433 / 5349449)
These gate every candidate's property G. Do not re-derive; cite this file.

| fact | value | consequence |
|---|---|---|
| nodes | `kn001-168`: 64 CPU / 515 GB / **4x L40S (46068 MiB)**; `kn169-178`: 48+ CPU / 2060 GB / **8x H100** | 46 GB VRAM/GPU clears Isaac Lab's 16 GB floor with room for tiled render |
| OS | Ubuntu 24.04.4 LTS (noble) | ✓ |
| glibc | **2.37** (Gentoo/CVMFS toolchain) | ✓ clears `isaacsim` pip wheel's >=2.35 requirement |
| NVIDIA driver | **580.159.03**, CUDA 13.0 driver-level | ✓ clears Isaac Lab's recommended >=580.95.05 |
| **Vulkan NVIDIA ICD** | `/usr/share/vulkan/icd.d/nvidia_icd.json` present; **`vulkaninfo --summary` enumerates GPU0 = NVIDIA L40S, DRIVER_ID_NVIDIA_PROPRIETARY, apiVersion 1.4.312, PHYSICAL_DEVICE_TYPE_DISCRETE_GPU** | ★ **the single most important gate: SAPIEN/ManiSkill3 GPU rendering and Isaac Sim RTX are both viable HEADLESS on this cluster.** This was the classic blocker and it is CLEAR. |
| EGL | `libEGL_nvidia.so.580.159.03` + `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` | ✓ (also what our current MUJOCO_GL=egl uses) |
| DISPLAY | unset on compute nodes | headless only; anything needing a GUI (GSWorld real2sim steps 3-5, PolaRiS Composer) must run elsewhere or under Xvfb |
| non-NVIDIA Vulkan ICDs | intel/nouveau/radeon/lvp/virtio all fail to load (missing libwayland-client.so.0, libLLVM.so.19.1) | harmless noise, but **set `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`** to silence loader errors and force the NVIDIA ICD |
| SLURM | `--account=aip-gigor` REQUIRED (multiple allocations) | every sbatch needs it |
| queue reality (2026-09-10) | a 5-min 1-GPU probe started in <1 min | short jobs schedule fast; iron law #16 still applies |
