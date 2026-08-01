import os
from dataclasses import dataclass, replace
from functools import cached_property
from collections import namedtuple, defaultdict
from typing import Optional
import pprint
import json
import h5py
import torch
import numpy as np

import common_utils
from common_utils import ibrl_utils as utils
from env.robosuite_wrapper import PixelRobosuite, STATE_KEYS, PROP_KEYS


Batch = namedtuple("Batch", ["obs", "action"])


def _compute_norm_stats(dataset_index_pairs, use_state):
    """Fit normalization stats over a set of (data, index_list) pairs.

    Returns a dict with `state_mean` / `state_std` (z-score; None in image
    mode) and `action_min` / `action_max` (min/max → [-1,1]). Computed on the
    stacked per-step entries so it automatically matches `obs_shape` even with
    state_stack > 1. Injected into the policy via set_normalization().
    """
    states, actions = [], []
    for data, index_list in dataset_index_pairs:
        for (episode_idx, step_idx) in index_list:
            entry = data[episode_idx][step_idx]
            actions.append(entry["action"])
            if use_state:
                states.append(entry["state"])
    actions = torch.stack(actions)
    stats = {
        "action_min": actions.min(dim=0)[0],
        "action_max": actions.max(dim=0)[0],
        "state_mean": None,
        "state_std": None,
    }
    if use_state:
        states = torch.stack(states)
        stats["state_mean"] = states.mean(dim=0)
        stats["state_std"] = states.std(dim=0)
    return stats


@dataclass
class DatasetConfig:
    path: str = ""
    rl_camera: str = "robot0_eye_in_hand"
    num_data: int = -1
    max_len: int = -1
    eval_episode_len: int = 400
    use_state: int = 0
    prop_stack: int = 1
    norm_action: int = 0
    obs_stack: int = 1
    state_stack: int = 1
    real_data: int = 0
    # When > 1, each entry also stores `action_chunk` of shape (action_horizon,
    # action_dim) covering steps [i, i+H), padded with the last action at
    # episode ends. Required by StateFlowPolicy / any chunked-action BC.
    action_horizon: int = 1
    # When 1, also stores `state_chunk` of shape (action_horizon, state_dim)
    # using the SAME window [i, i+H) padded with the last state. Required by
    # closed-loop RNN training (StateRnnGmmPolicy with obs_horizon=action_horizon).
    # Flow / Diffusion can ignore it. Gated by action_horizon > 1.
    build_state_chunk: int = 0
    # Fraction of episodes held out as validation (0.0 = no val split, legacy
    # behavior). Episodes are partitioned with a fixed RNG seed so the split
    # is stable across runs. Validation samples are accessed via sample_val.
    val_ratio: float = 0.0
    val_split_seed: int = 0
    # Independent val set drawn from the SAME hdf5 but DISJOINT episodes:
    # train = the first `num_data` kept episodes, val = the next `val_num_data`
    # kept episodes. Use when the file holds many more same-distribution demos
    # than you train on (e.g. source_sigma0.1 has 2000; train on 200, validate
    # on another 200). When >0 it OVERRIDES val_ratio (no in-train-set split).
    # Requires num_data > 0. Default 0 = disabled (legacy val_ratio behavior).
    val_num_data: int = 0
    # Robot arm init-state perturbation. Threaded into env_params used for
    # eval-time env construction. Three regimes:
    #   None (default)       → use robosuite's built-in init noise
    #                          (σ=0.02 gaussian on joint qpos, matches the
    #                          distribution MimicGen used when collecting
    #                          demos — so eval is in-distribution with the
    #                          training data).
    #   (0.0, 0.0, 0.0)      → fully deterministic init.
    #   (rx, ry, rz) >0      → IK an EE offset sampled uniformly from
    #                          [-rx,+rx]×[-ry,+ry]×[-rz,+rz] (sgft style).
    hand_init_perturb: Optional[tuple] = None
    # Robosuite joint-space gaussian σ override (rad). Set to scale up the
    # demo-style init noise without going through IK (which only perturbs
    # position and lets orientation drift). Mutually exclusive with
    # hand_init_perturb. None = robosuite's σ=0.02 default. 0.05 → ~2.5×
    # demo, 0.1 → ~5× demo (still controllable; perturbs both pos + ori
    # consistently).
    hand_init_perturb_joint_sigma: Optional[float] = None
    # ---- IL co-training: mix a second (target) dataset with the primary
    # (source) `path`. When `cotrain_path` is set, train_bc builds a
    # CoTrainDataset that draws EACH batch as `cotrain_ratio` fraction from
    # `path` (source) + (1 - cotrain_ratio) from `cotrain_path` (target).
    # `num_data` caps the source episodes, `cotrain_num_data` the target ones.
    # Example (95% source / 5% target, 380 source + 20 target episodes):
    #   path=<source.hdf5>  num_data=380
    #   cotrain_path=<target.hdf5>  cotrain_num_data=20  cotrain_ratio=0.95
    cotrain_path: str = ""
    cotrain_num_data: int = -1
    cotrain_ratio: float = 0.95
    # Optional per-demo `ep_type` filter (comma-separated, e.g. "C" or "C,D").
    # Only episodes whose HDF5 `ep_type` attr is in this set are loaded; empty =
    # keep all. Used to pull only the recovery (C) segments out of the
    # collect_recovery_data.py datasets (which store A/B/C/D per demo).
    ep_type_filter: str = ""

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")

    @cached_property
    def env_config(self) -> dict:
        if self.real_data:
            env_config = {
                "env_name": "real",
                "env_kwargs": {"robots": ["panda"], "controller_configs": {"control_delta": True}},
            }
            return env_config

        with h5py.File(self.path) as f:
            if "env_args" in f["data"].attrs:
                return json.loads(f["data"].attrs["env_args"])  # type: ignore

        # collect_target_demos.py format: no `env_args` attr; an `env_cfg.json`
        # sidecar sits next to the hdf5. Synthesize a minimal env_config. The
        # `_sidecar` marker tells RobomimicDataset to skip the strict
        # controller-config equality check (the sidecar doesn't store it).
        sidecar = os.path.join(os.path.dirname(self.path), "env_cfg.json")
        if os.path.exists(sidecar):
            with open(sidecar) as jf:
                meta = json.load(jf)
            return {
                "env_name": meta["env_name"],
                "env_kwargs": {
                    "robots": ["Panda"],
                    "controller_configs": {"control_delta": True},
                },
                "_sidecar": True,
            }
        raise AssertionError(
            f"{self.path}: no `env_args` attr and no env_cfg.json sidecar"
        )

    @cached_property
    def task_name(self):
        return self.env_config["env_name"]

    @cached_property
    def robot(self):
        return self.env_config["env_kwargs"]["robots"]

    @cached_property
    def ctrl_delta(self):
        return self.env_config["env_kwargs"]["controller_configs"]["control_delta"]


class RobomimicDataset:
    def __init__(self, cfg: DatasetConfig, build_env: bool = True):
        # build_env=False skips constructing the eval PixelRobosuite (and the
        # controller-cfg check). Used when this dataset is a co-training
        # sub-dataset — only one shared eval env is needed (built by the
        # CoTrainDataset), and these sub-datasets just provide sampled batches.
        self.cfg = cfg

        self.data = []
        datafile = h5py.File(cfg.path)
        num_episode: int = len(list(datafile["data"].keys()))  # type: ignore
        print(f"Raw Dataset size (#episode): {num_episode}")

        self.idx2entry = []  # store idx -> (episode_idx, timestep_idx)
        episode_lens = []
        all_actions = []  # for # logging purpose
        # When an independent val set is requested, load num_data (train) PLUS
        # val_num_data (val) kept episodes; the split below slices them apart.
        if cfg.val_num_data > 0:
            assert cfg.num_data > 0, "val_num_data>0 requires num_data>0"
            load_cap = cfg.num_data + cfg.val_num_data
        else:
            load_cap = cfg.num_data
        for episode_id in range(num_episode):
            if load_cap > 0 and len(episode_lens) >= load_cap:
                break

            episode_tag = f"demo_{episode_id}"
            episode = datafile[f"data/{episode_tag}"]

            # Optional ep_type filter (recovery datasets tag each demo A/B/C/D).
            # Skipped demos don't count toward num_data (the cap is on KEPT eps).
            if self.cfg.ep_type_filter:
                allowed = {t for t in self.cfg.ep_type_filter.split(",") if t}
                ep_type = episode.attrs.get("ep_type", None)
                if ep_type is not None and ep_type not in allowed:
                    continue

            if self.cfg.real_data:
                rewards = np.array(episode["rewards"]).astype(np.float32)  # type: ignore
                assert rewards.sum() == 1, f"wrong reward for episode {rewards.sum()}"

            actions = np.array(episode["actions"]).astype(np.float32)  # type: ignore
            actions = torch.from_numpy(actions)
            if self.cfg.ctrl_delta:
                # Policy-collected demos (collect_target_demos.py) can have a
                # few actions slightly outside [-1, 1] (the collecting policy
                # wasn't clipped). With OSC_POSE delta control the controller
                # clips to [-1, 1] anyway, so the EXECUTED — and thus the BC
                # target — action is the clipped one. No-op for already-in-range
                # MimicGen demos.
                actions = actions.clamp(-1.0, 1.0)
            all_actions.append(actions)

            episode_data: dict = {"action": actions}

            if episode_id == 0:
                common_utils.wrap_ruler("all avail obs", 60)
                for k, v in episode["obs"].items():  # type: ignore
                    print(k, np.array(v).shape)
                common_utils.wrap_ruler("", 60)

            if self.cfg.use_state:
                # Prefer a precomputed `obs/state` (collect_target_demos.py
                # format) — it is already the wrapper's STATE_KEYS concat, so it
                # matches the eval env's state ordering. Fall back to building
                # from STATE_KEYS for the standard MimicGen processed format.
                if "state" in episode["obs"]:  # type: ignore
                    episode_data["state"] = np.array(episode["obs"]["state"]).astype(np.float32)  # type: ignore
                else:
                    all_states = []
                    for key in STATE_KEYS[self.cfg.task_name]:
                        all_states.append(episode["obs"][key])  # type: ignore
                    episode_data["state"] = np.concatenate(all_states, axis=1).astype(np.float32)
            else:
                for camera in self.cfg.rl_cameras:
                    obses: np.ndarray = episode[f"obs/{camera}_image"]  # type: ignore
                    assert obses.shape[0] == actions.shape[0]
                    episode_data[camera] = obses

            # proprioception states
            if "obs/prop" in episode:  # type: ignore
                prop = np.array(episode["obs/prop"])  # type: ignore
                episode_data["prop"] = prop.astype(np.float32)
            else:
                robot_locs = []
                for key in PROP_KEYS:
                    robot_locs.append(episode["obs"][key])  # type: ignore
                episode_data["prop"] = np.concatenate(robot_locs, axis=1).astype(np.float32)

            episode_len = actions.shape[0]
            assert (
                episode_len < cfg.eval_episode_len
            ), f"found episode len {episode_len} > {cfg.eval_episode_len} (eval_len)"
            if self.cfg.max_len > 0 and episode_len > self.cfg.max_len:
                print(f"removing {episode_tag} because it is too long {episode_len}")
                continue

            # convert the data to list of dict
            episode_entries = []
            H = cfg.action_horizon
            for i in range(episode_len):
                entry = {"action": episode_data["action"][i]}
                if self.cfg.ctrl_delta:
                    assert entry["action"].min() >= -1
                    assert entry["action"].max() <= 1

                if H > 1:
                    # action chunk over [i, i+H); pad with last action if past episode end
                    end = min(i + H, episode_len)
                    chunk = episode_data["action"][i:end]                # (k, D), k <= H
                    if chunk.shape[0] < H:
                        pad = chunk[-1:].expand(H - chunk.shape[0], -1)  # repeat last action
                        chunk = torch.cat([chunk, pad], dim=0)
                    entry["action_chunk"] = chunk                        # (H, D)

                    if cfg.build_state_chunk and cfg.use_state:
                        # parallel state window [i, i+H), padded by repeating last state
                        s_arr = episode_data["state"]                    # (T, S) numpy
                        s_end = min(i + H, episode_len)
                        s_chunk_np = s_arr[i:s_end]
                        if s_chunk_np.shape[0] < H:
                            pad = np.repeat(s_chunk_np[-1:], H - s_chunk_np.shape[0], axis=0)
                            s_chunk_np = np.concatenate([s_chunk_np, pad], axis=0)
                        entry["state_chunk"] = torch.from_numpy(s_chunk_np)  # (H, S)

                entry["prop"] = utils.concat_obs(i, episode_data["prop"], cfg.prop_stack)
                if cfg.use_state:
                    entry["state"] = utils.concat_obs(i, episode_data["state"], cfg.state_stack)
                else:
                    for camera in cfg.rl_cameras:
                        entry[camera] = utils.concat_obs(i, episode_data[camera], cfg.obs_stack)

                self.idx2entry.append((len(self.data), len(episode_entries)))
                episode_entries.append(entry)

            episode_lens.append(len(episode_entries))
            self.data.append(episode_entries)
        datafile.close()

        if cfg.use_state:
            self.obs_shape = self.data[-1][-1]["state"].size()
        else:
            self.obs_shape = self.data[-1][-1][cfg.rl_cameras[0]].size()

        self.prop_shape = self.data[-1][-1]["prop"].size()
        self.action_dim = self.data[-1][-1]["action"].size()[0]
        print(f"Dataset size: {len(self.data)} episodes, {len(self.idx2entry)} steps")
        print(f"average length {np.mean(episode_lens):.1f}")
        print(f"max length: {sorted(episode_lens)[::-1][:5]}")
        print(f"obs shape:", self.obs_shape, "; prop shape:", self.prop_shape)

        # Validation split: partition episodes (not timesteps) so a val
        # demo's frames never leak into train. With val_ratio=0 the val
        # arrays are empty and sample_bc behavior is unchanged.
        num_episodes = len(self.data)
        if cfg.val_num_data > 0:
            # Independent val set: train = first num_data kept episodes,
            # val = the next (up to) val_num_data kept episodes. Disjoint by
            # construction, same distribution (same hdf5).
            val_episodes = set(range(cfg.num_data, num_episodes))
        elif cfg.val_ratio > 0:
            rng = np.random.RandomState(cfg.val_split_seed)
            shuffled = list(range(num_episodes))
            rng.shuffle(shuffled)
            n_val = max(1, int(round(num_episodes * cfg.val_ratio)))
            val_episodes = set(shuffled[:n_val])
        else:
            val_episodes = set()
        self.idx2entry_train = [(e, s) for (e, s) in self.idx2entry if e not in val_episodes]
        self.idx2entry_val = [(e, s) for (e, s) in self.idx2entry if e in val_episodes]
        if cfg.val_ratio > 0 or cfg.val_num_data > 0:
            print(f"Train/val split: {num_episodes - len(val_episodes)}/{len(val_episodes)} episodes, "
                  f"{len(self.idx2entry_train)}/{len(self.idx2entry_val)} steps")

        all_actions = torch.cat(all_actions, dim=0)
        action_mins = all_actions.min(dim=0)[0]
        action_maxs = all_actions.max(dim=0)[0]
        for i in range(self.action_dim):
            print(f"action dim {i}: [{action_mins[i].item():.2f}, {action_maxs[i].item():.2f}]")

        # Normalization stats fit on the TRAIN split only (no val leakage).
        self.norm_stats = _compute_norm_stats(
            [(self.data, self.idx2entry_train or self.idx2entry)], cfg.use_state
        )

        if cfg.real_data:
            self.env = None
            return

        self.env_params: dict = dict(
            env_name=self.cfg.task_name,
            robots=self.cfg.robot,
            episode_length=cfg.eval_episode_len,
            reward_shaping=False,
            image_size=224,
            rl_image_size=self.obs_shape[-1] if not cfg.use_state else 96,
            camera_names=cfg.rl_cameras,
            rl_cameras=cfg.rl_cameras,
            device="cuda",
            use_state=cfg.use_state,
            obs_stack=self.cfg.obs_stack,
            state_stack=self.cfg.state_stack,
            prop_stack=cfg.prop_stack,
            ctrl_delta=bool(self.cfg.ctrl_delta),
            hand_init_perturb=cfg.hand_init_perturb,
            hand_init_perturb_joint_sigma=cfg.hand_init_perturb_joint_sigma,
        )
        if not build_env:
            # Co-training sub-dataset: env_params is still exposed, but the live
            # env (and its strict controller check) is left to the parent.
            self.env = None
            return
        self.env = PixelRobosuite(**self.env_params)
        # The sidecar (collect_target_demos) format doesn't store the original
        # controller_configs, so there's nothing to assert against — skip it.
        if not self.cfg.env_config.get("_sidecar"):
            self._check_controller_cfg()

    def _check_controller_cfg(self):
        assert self.env is not None
        ref_controller_cfg = self.cfg.env_config["env_kwargs"]["controller_configs"]
        # rename for consistency
        ref_controller_cfg["damping_ratio"] = ref_controller_cfg["damping"]
        ref_controller_cfg["damping_ratio_limits"] = ref_controller_cfg["damping_limits"]
        ref_controller_cfg.pop("damping")
        ref_controller_cfg.pop("damping_limits")
        assert ref_controller_cfg == self.env.ctrl_config
        assert self.env.env.control_freq == self.cfg.env_config["env_kwargs"]["control_freq"]

        print(common_utils.wrap_ruler("config when the data was collected"))
        self.cfg.env_config["env_kwargs"].pop("controller_configs")
        pprint.pprint(self.cfg.env_config["env_kwargs"])
        print(common_utils.wrap_ruler(""))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _convert_to_batch(self, samples, device):
        batch = {}
        for k, v in samples.items():
            batch[k] = torch.stack(v).to(device)

        action = {"action": batch.pop("action")}
        if "action_chunk" in batch:
            action["action_chunk"] = batch.pop("action_chunk")
        ret = Batch(obs=batch, action=action)
        return ret

    def _sample_from_index(self, index_list, batchsize, device):
        """Sample a batch from a given (episode_idx, step_idx) index list."""
        if len(index_list) == 0:
            raise ValueError("sample called on empty index (val_ratio=0?)")
        choices = np.random.choice(len(index_list), batchsize)
        samples = defaultdict(list)
        for idx in choices:
            episode_idx, step_idx = index_list[idx]
            entry = self.data[episode_idx][step_idx]
            for k, v in entry.items():
                samples[k].append(v)
        return self._convert_to_batch(samples, device)

    def get_normalization_stats(self):
        """Dataset-fit stats for policy.set_normalization() (see _compute_norm_stats)."""
        return self.norm_stats

    def sample_bc(self, batchsize, device):
        return self._sample_from_index(self.idx2entry_train, batchsize, device)

    def sample_val(self, batchsize, device):
        """Sample a validation batch (held-out episodes). Empty if val_ratio=0."""
        return self._sample_from_index(self.idx2entry_val, batchsize, device)

    def has_val(self) -> bool:
        return len(self.idx2entry_val) > 0


def _merge_batches(batches: list) -> Batch:
    """Concatenate a list of Batch namedtuples along the batch dim (dim 0).

    All batches must share the same obs / action keys (guaranteed here because
    both co-training sub-datasets are the same task with the same cfg)."""
    if len(batches) == 1:
        return batches[0]
    obs = {k: torch.cat([b.obs[k] for b in batches], dim=0) for k in batches[0].obs}
    action = {k: torch.cat([b.action[k] for b in batches], dim=0) for k in batches[0].action}
    return Batch(obs=obs, action=action)


class CoTrainDataset:
    """IL co-training over a primary (source) + secondary (target) dataset.

    Loads `cfg.path` (source, capped at `cfg.num_data`) and `cfg.cotrain_path`
    (target, capped at `cfg.cotrain_num_data`) as two RobomimicDatasets, then
    draws EACH batch as `cfg.cotrain_ratio` from source + the rest from target
    — a fixed per-batch mixing ratio (the standard co-training knob). This is
    robust to the two datasets having different episode lengths/counts, unlike
    pooling-then-uniform sampling. Exposes the same surface train_bc expects of
    a RobomimicDataset (`sample_bc`, `sample_val`, `has_val`, `env_params`,
    `obs_shape`, `prop_shape`, `action_dim`, `cfg`).
    """

    def __init__(self, cfg: DatasetConfig):
        assert cfg.cotrain_path, "CoTrainDataset requires cfg.cotrain_path (target data)"
        assert 0.0 <= cfg.cotrain_ratio <= 1.0, f"cotrain_ratio out of [0,1]: {cfg.cotrain_ratio}"
        self.cfg = cfg

        print(common_utils.wrap_ruler("co-train: SOURCE dataset"))
        self.source = RobomimicDataset(cfg, build_env=False)
        target_cfg = replace(
            cfg, path=cfg.cotrain_path, num_data=cfg.cotrain_num_data, cotrain_path=""
        )
        print(common_utils.wrap_ruler("co-train: TARGET dataset"))
        self.target = RobomimicDataset(target_cfg, build_env=False)

        # Shapes/dims come from the source (both share the task → identical).
        self.obs_shape = self.source.obs_shape
        self.prop_shape = self.source.prop_shape
        self.action_dim = self.source.action_dim
        assert self.target.action_dim == self.action_dim, "source/target action_dim mismatch"
        assert self.target.obs_shape == self.obs_shape, "source/target obs_shape mismatch"

        # Build ONE eval env from the source env_params (the eval-time domain
        # shift is injected by train_bc.evaluate, not baked in here).
        self.env_params = self.source.env_params
        self.env = PixelRobosuite(**self.env_params) if not cfg.real_data else None

        # Normalization stats fit over the COMBINED source+target train split
        # (the policy sees a mix of both per batch).
        self.norm_stats = _compute_norm_stats(
            [(self.source.data, self.source.idx2entry_train or self.source.idx2entry),
             (self.target.data, self.target.idx2entry_train or self.target.idx2entry)],
            cfg.use_state,
        )

        n_src_eps, n_tgt_eps = len(self.source.data), len(self.target.data)
        print(common_utils.wrap_ruler("co-train mix"))
        print(f"source episodes: {n_src_eps} (from {cfg.path})")
        print(f"target episodes: {n_tgt_eps} (from {cfg.cotrain_path})")
        print(f"per-batch ratio: {cfg.cotrain_ratio:.2f} source / "
              f"{1 - cfg.cotrain_ratio:.2f} target")

    def get_normalization_stats(self):
        return self.norm_stats

    def _split(self, batchsize: int) -> tuple[int, int]:
        n_src = int(round(batchsize * self.cfg.cotrain_ratio))
        n_src = max(0, min(batchsize, n_src))
        return n_src, batchsize - n_src

    @staticmethod
    def _tag_domain(batch: Batch, domain_id: int, device) -> Batch:
        # Inject `domain_id` (0=source, 1=target) into batch.obs so CFG-aware
        # policies can read per-sample domain labels. Other policies index
        # batch.obs by name (state / prop / camera), so adding this key is a
        # no-op for them.
        n = next(iter(batch.obs.values())).shape[0]
        tag = torch.full((n,), domain_id, dtype=torch.long, device=device)
        new_obs = dict(batch.obs)
        new_obs["domain_id"] = tag
        return Batch(obs=new_obs, action=batch.action)

    def sample_bc(self, batchsize, device):
        n_src, n_tgt = self._split(batchsize)
        parts = []
        if n_src > 0:
            b = self.source._sample_from_index(self.source.idx2entry_train, n_src, device)
            parts.append(self._tag_domain(b, 0, device))
        if n_tgt > 0:
            b = self.target._sample_from_index(self.target.idx2entry_train, n_tgt, device)
            parts.append(self._tag_domain(b, 1, device))
        return _merge_batches(parts)

    def has_val(self) -> bool:
        return self.source.has_val() or self.target.has_val()

    def sample_val(self, batchsize, device):
        n_src, n_tgt = self._split(batchsize)
        parts = []
        if n_src > 0 and self.source.has_val():
            parts.append(self._tag_domain(
                self.source.sample_val(n_src, device), 0, device))
        if n_tgt > 0 and self.target.has_val():
            parts.append(self._tag_domain(
                self.target.sample_val(n_tgt, device), 1, device))
        if not parts:
            raise ValueError("sample_val called but neither split has a val set")
        return _merge_batches(parts)

    def __len__(self):
        return len(self.source) + len(self.target)
