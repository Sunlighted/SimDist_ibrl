from dataclasses import dataclass, field
from typing import Optional
import os
import sys
import yaml
import pyrallis
import numpy as np
import torch

from tqdm import tqdm
import common_utils
from bc.dataset import DatasetConfig, RobomimicDataset, CoTrainDataset
from bc.bc_policy import StateBcPolicy, StateBcPolicyConfig
from bc.bc_policy import BcPolicy, BcPolicyConfig
from bc.flow_policy import StateFlowPolicy, StateFlowPolicyConfig
from bc.rnn_gmm_policy import StateRnnGmmPolicy, StateRnnGmmPolicyConfig
from bc.diffusion_policy import StateDiffusionPolicy, StateDiffusionPolicyConfig
from bc.chunk_mlp_policy import StateChunkMlpPolicy, StateChunkMlpPolicyConfig
from bc.darp_policy import StateDARPPolicy, StateDARPPolicyConfig
from evaluate import run_eval_mp


def darp_pool_from_dataset(dataset, idx_list):
    """Stack (state, action) transitions from a RobomimicDataset index list into
    a DARP retrieval pool (state_stack must be 1 so entry['state'] is the raw obs)."""
    states = torch.stack([dataset.data[e][s]["state"] for (e, s) in idx_list])
    actions = torch.stack([dataset.data[e][s]["action"] for (e, s) in idx_list])
    return states, actions
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig


@dataclass
class MainConfig(common_utils.RunConfig):
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    state_policy: StateBcPolicyConfig = field(default_factory=lambda: StateBcPolicyConfig())
    policy: BcPolicyConfig = field(default_factory=lambda: BcPolicyConfig())
    flow_policy: StateFlowPolicyConfig = field(default_factory=lambda: StateFlowPolicyConfig())
    rnn_policy: StateRnnGmmPolicyConfig = field(default_factory=lambda: StateRnnGmmPolicyConfig())
    diffusion_policy: StateDiffusionPolicyConfig = field(default_factory=lambda: StateDiffusionPolicyConfig())
    chunk_mlp_policy: StateChunkMlpPolicyConfig = field(default_factory=lambda: StateChunkMlpPolicyConfig())
    darp_policy: StateDARPPolicyConfig = field(default_factory=lambda: StateDARPPolicyConfig())
    # Policy-head selection (exactly one of these may be 1):
    #   use_flow=1      → StateFlowPolicy (rectified flow, MLP)
    #   use_rnn=1       → StateRnnGmmPolicy (MimicGen BC-RNN+GMM, open-loop)
    #   use_diffusion=1 → StateDiffusionPolicy (robomimic DDIM, UNet1D)
    #   use_chunk_mlp=1 → StateChunkMlpPolicy (deterministic MLP, action chunk)
    # Default (all 0, use_state=1) → StateBcPolicy (deterministic MLP, single-step).
    use_flow: int = 0
    use_rnn: int = 0
    use_diffusion: int = 0
    use_chunk_mlp: int = 0
    # use_darp=1 → StateDARPPolicy (retrieval-conditioned MLP; pool = the dataset
    # transitions, set after construction). Single-step (action_horizon must be 1).
    use_darp: int = 0
    # training
    seed: int = 1
    load_model: str = "none"
    num_epoch: int = 20
    epoch_len: int = 10000
    batch_size: int = 256
    lr: float = 1e-4
    grad_clip: float = 5
    weight_decay: float = 0
    # eval
    num_eval_episode: int = 50
    # Run rollout-eval only every N epochs. With small epoch_len (e.g. 100,
    # MimicGen-style), evaling every epoch dominates wall clock; set to 50
    # to match MimicGen's `experiment.rollout.rate`. Default 1 preserves the
    # old "eval every epoch" behavior for backward compat.
    eval_period: int = 1
    # Validation-loss log frequency (epochs). Requires dataset.val_ratio > 0.
    # 0 disables. Cheap (one forward, no env), but enable only when you have
    # a val split. log key: `val/loss`.
    val_period: int = 0
    val_batches: int = 5
    # Target-domain spec. Determines the post-training "target" eval (always
    # run when mode != "none") AND, when `eval_target_during_training=1`,
    # also drives in-training periodic evals.
    #   mode="none"   → no target; in-training + final eval are source-only
    #   mode="fixed"  → target = fixed shift; final eval is dual source+target
    #   mode="dr"     → unused for BC eval (DR is a training-time concept)
    eval_domain_shift: DomainShiftConfig = field(default_factory=DomainShiftConfig)
    # If 1, in-training eval applies `eval_domain_shift` (so the topk saver
    # ranks by target score, and `score` in wandb is target-domain).
    # If 0 (default), in-training eval is ALWAYS source — the saved
    # `model0.pt` is the source-best ckpt, which is the conventional
    # "source-best → target eval" sim2real protocol. Either way, the
    # post-training final eval runs source AND (if mode != "none") target
    # on the source-best ckpt, reporting the gap.
    eval_target_during_training: int = 0
    # Robot arm init-state perturbation used at EVAL time. Overrides
    # DatasetConfig.hand_init_perturb when not None. Same three regimes:
    #   None (default)      → use robosuite's built-in σ=0.02 gaussian init
    #                         (matches MimicGen demo-collection distribution).
    #   (0.0, 0.0, 0.0)     → fully deterministic init (eval variance comes
    #                         purely from the policy itself).
    #   (rx, ry, rz) > 0    → IK an EE offset from uniform half-range box.
    # (Recall: BC training doesn't touch env — both fields only affect eval.)
    eval_hand_init_perturb: Optional[tuple] = None
    # Joint-space σ override for eval (matches DatasetConfig field). When
    # not None, takes precedence over dataset.hand_init_perturb_joint_sigma.
    eval_hand_init_perturb_joint_sigma: Optional[float] = None
    # Strict-success gate (Threading-only). BC training never touches the env,
    # so this only affects how eval episodes are scored — enable it to measure
    # the honest success rate (no clip-through / shove-the-ring false positives).
    # Enable in YAML with `strict_success.enabled: true`.
    strict_success: StrictSuccessConfig = field(default_factory=StrictSuccessConfig)
    # to be overwritten by run() to facilitate model loading
    task_name: str = ""
    robots: list[str] = field(default_factory=lambda: [])
    image_size: int = -1
    rl_image_size: int = -1
    # log
    save_dir: str = "exps/bc/run1"
    use_wb: int = 0
    save_per: int = -1

    @property
    def prop_stack(self):
        return self.dataset.prop_stack


def run(cfg: MainConfig, policy):
    if cfg.dataset.cotrain_path:
        # IL co-training: mix source (cfg.dataset.path) + target
        # (cfg.dataset.cotrain_path) at cfg.dataset.cotrain_ratio per batch.
        dataset = CoTrainDataset(cfg.dataset)
    else:
        dataset = RobomimicDataset(cfg.dataset)
    if not cfg.dataset.real_data:
        cfg.task_name = dataset.cfg.task_name
        cfg.robots = dataset.cfg.robot
        cfg.image_size = dataset.env_params["image_size"]
        cfg.rl_image_size = dataset.env_params["rl_image_size"]

    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
    print(common_utils.wrap_ruler("config"))
    with open(cfg.cfg_path, "r") as f:
        print(f.read(), end="")
    cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

    if policy is None:
        assert cfg.use_flow + cfg.use_rnn + cfg.use_diffusion + cfg.use_chunk_mlp + cfg.use_darp <= 1, (
            "set at most one of use_flow / use_rnn / use_diffusion / use_chunk_mlp / use_darp"
        )
        if cfg.use_flow:
            assert cfg.dataset.use_state, "use_flow=1 currently requires dataset.use_state=1"
            assert cfg.dataset.action_horizon == cfg.flow_policy.pred_horizon, (
                f"dataset.action_horizon ({cfg.dataset.action_horizon}) must match "
                f"flow_policy.pred_horizon ({cfg.flow_policy.pred_horizon})"
            )
            assert cfg.flow_policy.action_horizon <= cfg.flow_policy.pred_horizon, (
                f"flow_policy.action_horizon ({cfg.flow_policy.action_horizon}) must be "
                f"<= flow_policy.pred_horizon ({cfg.flow_policy.pred_horizon})"
            )
            if cfg.flow_policy.cfg_enabled:
                assert cfg.dataset.cotrain_path, (
                    "flow_policy.cfg_enabled=1 requires a co-training dataset "
                    "(set dataset.cotrain_path) — CFG conditions on the source / "
                    "target domain label that CoTrainDataset attaches to batches."
                )
            policy = StateFlowPolicy(dataset.obs_shape, dataset.action_dim, cfg.flow_policy)
            # Install dataset-fit normalization (state z-score + action min/max
            # → [-1,1]) into the policy buffers; persisted in the ckpt so eval
            # auto-normalizes. Identity if the dataset can't provide stats.
            if hasattr(dataset, "get_normalization_stats"):
                policy.set_normalization(**dataset.get_normalization_stats())
        elif cfg.use_rnn:
            assert cfg.dataset.use_state, "use_rnn=1 currently requires dataset.use_state=1"
            assert cfg.dataset.action_horizon == cfg.rnn_policy.pred_horizon, (
                f"dataset.action_horizon ({cfg.dataset.action_horizon}) must match "
                f"rnn_policy.pred_horizon ({cfg.rnn_policy.pred_horizon})"
            )
            assert cfg.rnn_policy.action_horizon <= cfg.rnn_policy.pred_horizon
            policy = StateRnnGmmPolicy(dataset.obs_shape, dataset.action_dim, cfg.rnn_policy)
        elif cfg.use_diffusion:
            assert cfg.dataset.use_state, "use_diffusion=1 currently requires dataset.use_state=1"
            assert cfg.dataset.action_horizon == cfg.diffusion_policy.pred_horizon, (
                f"dataset.action_horizon ({cfg.dataset.action_horizon}) must match "
                f"diffusion_policy.pred_horizon ({cfg.diffusion_policy.pred_horizon})"
            )
            assert cfg.diffusion_policy.action_horizon <= cfg.diffusion_policy.pred_horizon
            policy = StateDiffusionPolicy(
                dataset.obs_shape, dataset.action_dim, cfg.diffusion_policy
            )
        elif cfg.use_chunk_mlp:
            assert cfg.dataset.use_state, "use_chunk_mlp=1 requires dataset.use_state=1"
            assert cfg.dataset.action_horizon == cfg.chunk_mlp_policy.pred_horizon, (
                f"dataset.action_horizon ({cfg.dataset.action_horizon}) must match "
                f"chunk_mlp_policy.pred_horizon ({cfg.chunk_mlp_policy.pred_horizon})"
            )
            assert cfg.chunk_mlp_policy.action_horizon <= cfg.chunk_mlp_policy.pred_horizon
            policy = StateChunkMlpPolicy(dataset.obs_shape, dataset.action_dim, cfg.chunk_mlp_policy)
        elif cfg.use_darp:
            assert cfg.dataset.use_state, "use_darp=1 requires dataset.use_state=1"
            assert cfg.dataset.action_horizon == 1, "use_darp is single-step (set dataset.action_horizon=1)"
            policy = StateDARPPolicy(dataset.obs_shape, dataset.action_dim, cfg.darp_policy)
            # retrieval pool = this dataset's (train) transitions
            idx = dataset.idx2entry_train or dataset.idx2entry
            states, actions = darp_pool_from_dataset(dataset, idx)
            policy.set_pool(states, actions)
        elif cfg.dataset.use_state:
            policy = StateBcPolicy(dataset.obs_shape, dataset.action_dim, cfg.state_policy)
        else:
            policy = BcPolicy(
                dataset.obs_shape,
                dataset.prop_shape,
                dataset.action_dim,
                dataset.cfg.rl_cameras,
                cfg.policy,
            )

    policy = policy.to("cuda")
    print(common_utils.wrap_ruler("policy weights"))
    print(policy)

    common_utils.count_parameters(policy)
    if cfg.weight_decay == 0:
        print("Using Adam optimzer")
        optim = torch.optim.Adam(policy.parameters(), cfg.lr)
    else:
        print("Using AdamW optimzer")
        optim = torch.optim.AdamW(policy.parameters(), cfg.lr, weight_decay=cfg.weight_decay)

    stat = common_utils.MultiCounter(
        cfg.save_dir,
        bool(cfg.use_wb),
        wb_exp_name=cfg.wb_exp,
        wb_run_name=cfg.wb_run,
        wb_group_name=cfg.wb_group,
        config=cfg_dict,
    )

    saver = common_utils.TopkSaver(cfg.save_dir, 2)
    stopwatch = common_utils.Stopwatch()
    best_score = 0
    optim_step = 0
    # tqdm cosmetics: keep the inner bar contained so the per-epoch summary
    # (printed by stat.summary) and final eval lines stay visible above.
    # `leave=False` clears the inner bar after each epoch; outer epoch bar
    # stays as a persistent at-a-glance progress.
    epoch_iter = tqdm(range(cfg.num_epoch), desc="train", unit="epoch",
                      dynamic_ncols=True)
    for epoch in epoch_iter:
        stopwatch.reset()

        step_iter = tqdm(range(cfg.epoch_len), desc=f"  epoch {epoch}",
                         leave=False, unit="step", dynamic_ncols=True)
        # Lightweight rolling means for the postfix so the displayed numbers
        # don't jitter on every step. Window = last 100 steps.
        recent_loss: list = []
        recent_gn: list = []
        for _ in step_iter:
            with stopwatch.time("sample"):
                batch = dataset.sample_bc(cfg.batch_size, "cuda:0")

            with stopwatch.time("train"):
                loss = policy.loss(batch)

                optim.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(  # type: ignore
                    policy.parameters(), max_norm=cfg.grad_clip
                )
                optim.step()
                # Update EMA shadow params if the policy exposes the hook
                # (StateFlowPolicy with cfg.use_ema=1). No-op for plain
                # StateBcPolicy / BcPolicy.
                if hasattr(policy, "update_ema"):
                    policy.update_ema()
                loss_v = loss.item(); gn_v = grad_norm.item()
                stat["train/loss"].append(loss_v)
                stat["train/grad_norm"].append(gn_v)
                optim_step += 1

                recent_loss.append(loss_v); recent_gn.append(gn_v)
                if len(recent_loss) > 100:
                    recent_loss.pop(0); recent_gn.pop(0)
                # Refresh postfix every 20 steps to keep stdout cheap
                if optim_step % 20 == 0:
                    step_iter.set_postfix(
                        loss=f"{np.mean(recent_loss):.4f}",
                        gn=f"{np.mean(recent_gn):.3f}",
                        refresh=False,
                    )

        epoch_time = stopwatch.elapsed_time_since_reset
        speed = cfg.epoch_len / epoch_time
        stat["other/speed"].append(speed)
        # (outer-bar postfix updated at end of epoch body, after eval/save)

        # Validation loss (cheap: forward only, no env). Gated by val_period
        # AND availability of a val split. Logs `val/loss`.
        if cfg.val_period > 0 and (epoch + 1) % cfg.val_period == 0 and dataset.has_val():
            policy.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(cfg.val_batches):
                    vbatch = dataset.sample_val(cfg.batch_size, "cuda:0")
                    val_losses.append(policy.loss(vbatch).item())
                stat["val/loss"].append(float(np.mean(val_losses)))
            policy.train()

        # In-training rollout eval is gated by eval_period (default 1 = every
        # epoch, MimicGen-compatible value is ~50). Skipped epochs still
        # save `latest.pt` but don't update the topk ranking.
        do_eval = (cfg.eval_period > 0
                   and ((epoch + 1) % cfg.eval_period == 0
                        or epoch == cfg.num_epoch - 1))
        # In-training eval domain — gated by both eval_target_during_training
        # flag AND mode being set. Default flag=0 ⇒ source eval during training,
        # so topk picks the source-best ckpt and final target eval runs on it.
        in_train_shift = (cfg.eval_domain_shift
                          if (cfg.eval_target_during_training
                              and cfg.eval_domain_shift.mode != "none")
                          else None)
        if cfg.dataset.real_data:
            saved = saver.save(policy.state_dict(), epoch, save_latest=True)
            if cfg.save_per > 0 and (epoch + 1) % cfg.save_per == 0:
                saver.save(policy.state_dict(), epoch, force_save_name=f"epoch{epoch+1}")
        elif do_eval:
            with stopwatch.time("eval"):
                seed = epoch * cfg.num_eval_episode + 1
                # In-training eval uses whatever eval_domain_shift says
                # (mode="none" → source eval; mode="fixed" → target eval).
                # NOTE: saver scoring tracks this single number, so if you flip
                # mode mid-training the topk ranking will be inconsistent.
                scores = evaluate(policy, dataset, seed=seed,
                                  num_game=cfg.num_eval_episode,
                                  domain_shift=in_train_shift,
                                  strict_success=cfg.strict_success,
                                  hand_init_perturb=cfg.eval_hand_init_perturb,
                                  hand_init_perturb_joint_sigma=cfg.eval_hand_init_perturb_joint_sigma)
                score = float(np.mean(scores))
                saved = saver.save(policy.state_dict(), score, save_latest=True)

            best_score = max(best_score, score)
            stat["score"].append(score)
            stat["score(best)"].append(best_score)

            if (epoch + 1) % (5 * max(cfg.eval_period, 1)) == 0 or (epoch == cfg.num_epoch - 1):
                # eval the last checkpoint
                scores = evaluate(policy, dataset, num_game=100, seed=1,
                                  domain_shift=in_train_shift,
                                  strict_success=cfg.strict_success,
                                  hand_init_perturb=cfg.eval_hand_init_perturb,
                                  hand_init_perturb_joint_sigma=cfg.eval_hand_init_perturb_joint_sigma)
                stat["last_ckpt_score"].append(np.mean(scores))
        else:
            # No eval this epoch: only refresh latest.pt; perf=None skips topk
            saved = saver.save(policy.state_dict(), None, save_latest=True)

        stat.summary(epoch)
        stopwatch.summary()
        if saved:
            print("model saved!")

        # Outer bar postfix: now best_score is up to date for this epoch.
        epoch_iter.set_postfix(
            loss=f"{np.mean(recent_loss):.4f}",
            speed=f"{speed:.0f}st/s",
            best=f"{best_score:.3f}",
            refresh=False,
        )
    epoch_iter.close()

    if not cfg.dataset.real_data:
        # Post-training: load the topk-best ckpt and evaluate on BOTH source
        # (no shift) and target (eval_domain_shift). The "best" ckpt's ranking
        # came from in-training eval, which is source-domain when
        # eval_target_during_training=0 (default) and target-domain when =1.
        # So with the default flag, this implements "source-best ckpt → target
        # eval" — the conventional sim2real protocol.
        best_model = saver.get_best_model()
        policy.load_state_dict(torch.load(best_model))
        selection_domain = "target" if cfg.eval_target_during_training else "source"

        src_scores = evaluate(policy, dataset, num_game=100, seed=1,
                              domain_shift=None,
                              strict_success=cfg.strict_success,
                              hand_init_perturb=cfg.eval_hand_init_perturb,
                              hand_init_perturb_joint_sigma=cfg.eval_hand_init_perturb_joint_sigma)
        src_score = float(np.mean(src_scores))
        stat["best_ckpt_source"].append(src_score)
        print(f"[final eval | source]  score = {src_score:.4f}  ({best_model})"
              f"  (ckpt selected on {selection_domain}; "
              f"hand_init_perturb={cfg.eval_hand_init_perturb})")

        if cfg.eval_domain_shift.mode != "none":
            tgt_scores = evaluate(policy, dataset, num_game=100, seed=1,
                                  domain_shift=cfg.eval_domain_shift,
                                  strict_success=cfg.strict_success,
                                  hand_init_perturb=cfg.eval_hand_init_perturb,
                                  hand_init_perturb_joint_sigma=cfg.eval_hand_init_perturb_joint_sigma)
            tgt_score = float(np.mean(tgt_scores))
            stat["best_ckpt_target"].append(tgt_score)
            print(f"[final eval | target]  score = {tgt_score:.4f}  "
                  f"(shift mode={cfg.eval_domain_shift.mode}, "
                  f"hole_offset={cfg.eval_domain_shift.hole_offset})")
            print(f"[final eval | gap   ]  source - target = {src_score - tgt_score:+.4f}")
        else:
            print("[final eval | target]  skipped (eval_domain_shift.mode == 'none')")

        # Also keep the legacy "best_ckpt_score" key pointing at source for
        # backward compat with any downstream tooling reading wandb logs.
        stat["best_ckpt_score"].append(src_score)
        stat.summary(cfg.num_epoch)

    # quit!
    assert False


def evaluate(policy, dataset: RobomimicDataset, seed, num_game, *,
             domain_shift: "DomainShiftConfig | None" = None,
             strict_success: "StrictSuccessConfig | None" = None,
             hand_init_perturb=None,
             hand_init_perturb_joint_sigma=None):
    """Eval the policy on `num_game` episodes via run_eval_mp.

    Optional overrides let the caller eval the same policy under different
    env conditions without rebuilding the dataset:
      - domain_shift                   : injected into env_params
      - hand_init_perturb              : 3-tuple (rx, ry, rz) EE per-axis
                                         half-range (m), overrides dataset
      - hand_init_perturb_joint_sigma  : float, joint-space gaussian σ (rad);
                                         mutually exclusive with above

    If the policy exposes EMA hooks (StateFlowPolicy w/ use_ema=1), the
    online weights are swapped with EMA weights for the duration of eval.
    """
    env_params = dict(dataset.env_params)
    if domain_shift is not None:
        env_params["domain_shift"] = domain_shift
    if strict_success is not None:
        env_params["strict_success"] = strict_success
    if hand_init_perturb is not None:
        env_params["hand_init_perturb"] = tuple(hand_init_perturb)
        env_params["hand_init_perturb_joint_sigma"] = None
    if hand_init_perturb_joint_sigma is not None:
        env_params["hand_init_perturb_joint_sigma"] = float(hand_init_perturb_joint_sigma)
        env_params["hand_init_perturb"] = None

    ema_applied = False
    if hasattr(policy, "apply_ema"):
        ema_applied = bool(policy.apply_ema())
    try:
        return run_eval_mp(
            env_params, policy, num_game=num_game, seed=seed, num_proc=10, verbose=False
        )
    finally:
        if ema_applied:
            policy.restore_from_ema()


def _load_model(weight_file, env: PixelRobosuite, device, cfg: Optional[MainConfig] = None):
    if cfg is None:
        cfg_path = os.path.join(os.path.dirname(weight_file), f"cfg.yaml")
        cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore

    print("observation shape: ", env.observation_shape)
    if cfg.use_flow:
        policy = StateFlowPolicy(env.observation_shape, env.action_dim, cfg.flow_policy)
    elif cfg.use_rnn:
        policy = StateRnnGmmPolicy(env.observation_shape, env.action_dim, cfg.rnn_policy)
    elif cfg.use_diffusion:
        policy = StateDiffusionPolicy(
            env.observation_shape, env.action_dim, cfg.diffusion_policy
        )
    elif cfg.use_chunk_mlp:
        policy = StateChunkMlpPolicy(env.observation_shape, env.action_dim, cfg.chunk_mlp_policy)
    elif cfg.use_darp:
        policy = StateDARPPolicy(env.observation_shape, env.action_dim, cfg.darp_policy)
    elif cfg.dataset.use_state:
        policy = StateBcPolicy(env.observation_shape, env.action_dim, cfg.state_policy)
    else:
        policy = BcPolicy(
            env.observation_shape, env.prop_shape, env.action_dim, env.rl_cameras, cfg.policy
        )
    policy.load_state_dict(torch.load(weight_file, map_location=device))
    return policy.to(device)


# function to load bc models
def load_model(weight_file, device, *, verbose=True):
    run_folder = os.path.dirname(weight_file)
    cfg_path = os.path.join(run_folder, f"cfg.yaml")
    if verbose:
        print(common_utils.wrap_ruler("config of loaded agent"))
        with open(cfg_path, "r") as f:
            print(f.read(), end="")
        print(common_utils.wrap_ruler(""))

    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore

    assert not cfg.dataset.real_data
    env_params = dict(
        env_name=cfg.task_name,
        robots=cfg.robots,
        episode_length=cfg.dataset.eval_episode_len,
        reward_shaping=False,
        image_size=cfg.image_size,
        rl_image_size=cfg.rl_image_size,
        camera_names=cfg.dataset.rl_cameras,
        rl_cameras=cfg.dataset.rl_cameras,
        device=device,
        use_state=cfg.dataset.use_state,
        obs_stack=cfg.dataset.obs_stack,
        state_stack=cfg.dataset.state_stack,
        prop_stack=cfg.dataset.prop_stack,
    )
    # demos collected in a sim2real cell (env_cfg.json sidecar names it): rebuild the eval env IN that cell, or the
    # policy gets e.g. world-frame prop where it was trained on base-frame prop (bc/dataset.py, 2026-10-19)
    sidecar = os.path.join(os.path.dirname(cfg.dataset.path), "env_cfg.json")
    if os.path.exists(sidecar):
        import json as _json
        cell = _json.load(open(sidecar)).get("sim2real_cell")
        if cell:
            from env.sim2real import load_sim2real
            env_params["sim2real"] = load_sim2real(cell)
            if verbose:
                print(f"[load_model] eval env in the demos' sim2real cell {cell}")
    env = PixelRobosuite(**env_params)  # type: ignore

    if cfg.dataset.use_state:
        print(f"state_stack: {cfg.dataset.state_stack}, observation shape: {env.observation_shape}")
    else:
        print(f"obs_stack: {cfg.dataset.obs_stack}, observation shape: {env.observation_shape}")

    policy = _load_model(weight_file, env, device, cfg)
    return policy, env, env_params


if __name__ == "__main__":
    import rich.traceback

    # make logging more beautiful
    rich.traceback.install()
    torch.set_printoptions(linewidth=100)

    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    common_utils.set_all_seeds(cfg.seed)
    log_path = os.path.join(cfg.save_dir, "train.log")
    sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)

    if cfg.load_model is not None and cfg.load_model != "none":
        policy = load_model(cfg.load_model, "cuda")[0]
    else:
        policy = None

    run(cfg, policy=policy)
