# 环境与数据准备

conda env / 依赖 pin / submodule / MimicGen 任务注册 / 数据转换。装机或换机器时读。

> 从 `CLAUDE.md` 拆出（2026-08-14）。速查表、方法论铁律、文件地图仍在 `CLAUDE.md`；已归档分支见 `CLAUDE_ARCHIVE.md`。

---

## 0. Project overview

Forked from `https://github.com/Chenyang Cao/ibrl` (state-based RL + BC for
robosuite/MetaWorld). Modifications cluster around four axes:

1. **MimicGen task integration** — 10 D0 tasks registered in the wrapper, BC
   data conversion tools patched.
2. **Domain shift framework** — sgft-style contact-phase shifts for
   `Threading_D0` with `fixed` and `dr` modes, plumbed through `train_rl.py`.
3. **Flow-matching BC policy** — robomimic-aligned `StateFlowPolicy` with
   action chunking, EMA, and pred>action horizon.
4. **Chunked-eval infrastructure** — `evaluate/` rewired to support per-worker
   action queues with `is_first_step` episode-boundary signaling.

Each section below: **What / Why / Where / How / TODO**.

---

## 1. Environment setup

### What
A conda env called `ibrl` (Blackwell-compatible, deliberately diverges from
the README's torch 2.1 / cu121 pin).

### Where
- Conda env actually lives at `/home/Chenyang/.conda/envs/ibrl`
  (NOT `/home/Chenyang/anaconda3/envs/ibrl` despite where anaconda is).
- Source repos vendored as **git submodules** under `deps/`:
  - `deps/robosuite` — pinned commit `b9d8d3d` (== v1.4.1, MimicGen requirement)
  - `deps/robomimic` — at upstream HEAD `e10526b` (**v0.5.0**, 2026-02). The
    MimicGen-recommended pin `d0b37cf` is preserved as the git tag
    `mimicgen-pinned-rollback` on that submodule if v0.5 ever causes issues.
  - `deps/mimicgen` — at HEAD (`72bd767`).
  - Legacy clones at `/home/Chenyang/ibrl_deps/` may still exist for backward
    compat (the editable install was originally there); safe to delete once
    the new `deps/` install is verified.
- C++ extension `common_utils/rela` is compiled (`cd common_utils && make`).
  Rebuild if torch is upgraded.

### Setup from a fresh clone
```bash
# 1) clone with submodules
git clone --recursive https://github.com/Sunlighted/ibrl.git
cd ibrl

# 2) create env + install torch (Blackwell)
conda create -n ibrl python=3.10 -y
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ibrl
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 3) install pinned mujoco + deps in submodules
python -m pip install "numpy<2" "mujoco==2.3.2"
python -m pip install -e deps/robosuite -e deps/robomimic -e deps/mimicgen

# 4) install other ibrl deps
python -m pip install Cython==0.29.36 einops==0.7.0 fasteners==0.19 pyrallis==0.3.1 \
  tabulate==0.9.0 typing-inspect==0.9.0 wandb==0.16.4 cloudpickle "gym==0.26.2"

# 5) build the C++ extension
cd common_utils && make -j8 && cd ..

# 6) activate / set paths each session
source set_env.sh

# 7) point release/data at YOUR datasets (per-machine, NOT in git)
#    Every config path is written as release/data/... but the datasets are
#    hundreds of GB, so the repo carries no `release/data` -- you create it.
#    Either a symlink to wherever your storage is:
ln -s /your/storage/simdist/data release/data
#    ...or a real directory, if the data lives next to the checkout:
mkdir -p release/data
#    `source set_env_drac.sh` prints `data -> <resolved path>`; check that line
#    before launching anything. A dangling link reports exists=False while
#    still being a link, which reads as "my data vanished".

# 8) install the storage-link guard (see "Data-loss guard" below) -- 3 commands,
#    do NOT skip: this is what stops step 7 from being re-committed.
git config core.hooksPath scripts/githooks
git config --add simdist.storagelink "release/data=/your/storage/simdist/data"
git config --add simdist.storagelink "exps=/your/storage/simdist/exps"
```

> ⚠️ **Never commit `release/data`.** It was committed once as a symlink into one
> developer's personal scratch (`/scratch/<user>/simdist/data`). Because
> `.gitignore`'s `data/` rule has a trailing slash it matches directories only, so
> it did not stop the symlink from being added -- but it *does* match a real
> `release/data/` directory, which makes its contents **ignored** rather than
> untracked. Git refuses to clobber untracked files but **silently deletes ignored
> ones** to make room for a tracked path. Everyone who pulled had their local
> dataset directory deleted with no warning and `rc=0`, and was left with either a
> link to someone else's data or a dangling link. `release/data` is now in
> `.gitignore` explicitly.

### Data-loss guard (`scripts/githooks/`, `scripts/check_storage_links.sh`)

`.gitignore` documents the incident above but cannot prevent it: `git add -f`
bypasses it, and a branch that *already* tracks the path clobbers the link on
checkout regardless. Two hooks close that, installed with
`git config core.hooksPath scripts/githooks` (step 8 above -- **the hooks do
nothing until you run that**, git never enables a repo's hooks automatically).

- **`pre-commit`** refuses to stage (a) the known per-machine storage paths
  `data` / `exps` / `release/data` / `release/model`, (b) **any** path
  `.gitignore` matches -- that is the exact wipe mechanism, since git deletes
  ignored files to make room for a tracked path -- and (c) any symlink with an
  absolute target, which is per-machine config by definition.
  Escape hatch, for when you genuinely mean it: `git commit --no-verify`.
- **`post-checkout` / `post-merge` / `post-rewrite`** run
  `scripts/check_storage_links.sh` after anything that can move the working
  tree (checkout, merge, rebase, `commit --amend`). It never blocks the git
  operation; it only reports, and it re-creates a link that went missing.

The expected targets are *not* in git -- they are per-machine, which is how this
went wrong the first time. They live in local git config as
`simdist.storagelink = <path>=<target>` entries (step 8), so
`scripts/check_storage_links.sh` can be run standalone at any time:

```bash
./scripts/check_storage_links.sh   # exit 0 = fine; 1 = needs a human
```

Behaviour per failure mode (all five are covered by hand-tested cases):

| situation | what happens |
|---|---|
| link missing, target present | **restored** automatically, warns, exit 0 |
| link points somewhere else | warns with both paths, exit 1 |
| a **real dir/file** sits where the link belongs | **left untouched** (it may be data), exit 1 |
| link intact but target gone (scratch purged) | warns, exit 1 |
| the path is **TRACKED by git** (the original incident) | warns with the `git rm --cached` fix, exit 1 |

⚠️ These links point into `/scratch`, which the cluster purges on an
inactivity timer. The guard detects a purged target but cannot undo it --
anything you cannot regenerate does not belong only in `/scratch`.

### How
```bash
# activate
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh && conda activate ibrl
# or
cd /home/Chenyang/ibrl && source set_env.sh   # also sets PYTHONPATH + MUJOCO_GL
```

### Why diverge from README
- README pins **torch 2.1.0 + cu121** → silently fails on RTX Pro 6000 Blackwell
  (sm_120). Replaced with **torch 2.7.0 + cu128**.
- README pins **mujoco 3.1.3** → MimicGen needs 2.3.2, used that instead.
- Skipped **mujoco-py 2.1 + metaworld** entirely (old, painful on new system,
  not needed for MimicGen work). `env/metaworld_wrapper.py` will fail to
  import if anything tries to use it — accepted.
- `numpy<2` pinned (wandb 0.16.4 uses removed `np.float_`).
- `huggingface_hub==0.23.4` pinned (robomimic v0.5 hard pins).
- `robosuite_task_zoo` **NOT** installed → `HammerCleanup_D0` / `Kitchen_D0`
  unavailable, other 10 MimicGen tasks fine.

### Gotchas
- **`pip` has a stale shebang** pointing at `/home/Chenyang/anaconda3/envs/ibrl/bin/python3.10`
  (which doesn't exist). Always use **`python -m pip install ...`** in this env.
- 3× RTX Pro 6000 Blackwell available (sm_120, 97 GB VRAM each).

### TODO / known issues
- [ ] None known. If torch is bumped, remember to `cd common_utils && make` again.

---

## 2. MimicGen integration

### What
10 MimicGen D0 tasks registered in the robosuite wrapper, dataset conversion
tools patched to also register MimicGen envs.

### Where
- `env/robosuite_wrapper.py`
  - L12-22: `try: import mimicgen` triggers task registration on robosuite import
  - L25-67: `STATE_KEYS` dict extended for all 10 D0 tasks (all use `DEFAULT_STATE_KEYS`)
  - L68-85: `STATE_SHAPE` dict — precomputed (9 prop + object-state dim)
- `tools/dataset_states_to_obs.py`
  - L61-64: `import mimicgen` added (needed for `EnvUtils.create_env_for_data_processing` to find `Threading_D0` etc.)
  - L144-153: empty-cameras bug fix — `np.concatenate([])` crash when no `--camera_names` passed
- `tools/robomimic_dataset_conversion.py`
  - L20-23: `import mimicgen`

### Tasks available
Threading_D0, Stack_D0, StackThree_D0, Square_D0, Coffee_D0,
CoffeePreparation_D0, ThreePieceAssembly_D0, PickPlace_D0, NutAssembly_D0,
MugCleanup_D0. To add another, probe `obs["object-state"].shape[0]` once
and add the entry — see `STATE_SHAPE` block.

### How (state-only dataset prep)
```bash
cd /home/Chenyang/ibrl && source set_env.sh

# 1) download MimicGen-generated demos
python deps/mimicgen/mimicgen/scripts/download_datasets.py \
  --dataset_type core --tasks threading_d0 \
  --download_dir release/data/mimicgen

# 2) convert to ibrl format (state mode — no --camera_names)
python tools/dataset_states_to_obs.py \
  --dataset release/data/mimicgen/core/threading_d0.hdf5 \
  --output_folder release/data/mimicgen/core/threading_d0 \
  --done_mode 2
# Output: release/data/mimicgen/core/threading_d0/processed_data96.hdf5
```

For pixel mode add `--camera_names agentview robot0_eye_in_hand` (much slower:
re-renders every demo step at 96×96).

### Dataset facts (Threading_D0)
- MimicGen "core" release: **1000 demos**, **99.6% success**, ~224 mean
  steps/demo, **state shape = 37** (eef_pos[3] + eef_quat[4] +
  gripper_qpos[2] + object-state[28]), action_dim=7 (OSC_POSE + gripper)
- File downloaded by the user was at `/home/Chenyang/ibrl/dataset/processed_data96.hdf5`
  as of 2026-05-24 but **was removed** by 2026-05-25 — YAMLs hard-code this
  path and will fail until file reappears.

### Demo configs (BC + RL state)
- `release/cfgs/mimicgen/stack_d0_state_bc.yaml`
- `release/cfgs/mimicgen/stack_d0_state_rlpd.yaml`
- (Threading-specific configs documented in §3 and §4)

### TODO
- [ ] Re-download Threading dataset or update YAML paths once file reappears.
- [ ] Pixel-mode configs (currently only state mode is wired).
- [ ] Generalize `STATE_KEYS` to other MimicGen variants (D1, D2 — broader
      initial state distributions).
