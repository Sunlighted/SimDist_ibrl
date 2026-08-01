"""State-based Diffusion Policy for ibrl (DDIM, switchable backbone).

Two backbones supported via `cfg.backbone`:
  - "unet1d"      (default): robomimic's `ConditionalUnet1D` — FiLM-cond 1D
    U-Net, sinusoidal step embed, GroupNorm + Mish. Matches `bc/flow_policy`
    default so flow-vs-diffusion ablation isolates the training objective.
  - "transformer":           our port of real-stanford/diffusion_policy's
    `TransformerForDiffusion` — GPT-style encoder-decoder, causal self-attn
    on the action axis, time + obs tokens as cross-attention memory. Matches
    the diffusion_policy paper's lowdim transformer config.

Other features (backbone-agnostic):
  - DDIM/DDPM scheduler from `diffusers` with the same `prediction_type /
    beta_schedule / clip_sample` knobs as robomimic.
  - Tp prediction horizon vs Ta execution horizon (the slice at inference).
  - EMA over weights via `diffusers.training_utils.EMAModel(power=...)` —
    same warmup-decay EMA robomimic's DiffusionPolicyUNet uses; applied at
    eval time and persisted in the checkpoint.
  - To = 1 (single obs conditioning), to match StateFlowPolicy and let the
    same chunked dataset drive training.
  - Normalization (default ON, §15): obs z-score + action min/max→[-1,1], fit
    on the train split and stored as buffers (identity until set, so old
    checkpoints load byte-for-byte). Matches robomimic (hdf5_normalize_obs +
    action_config min_max) / the original diffusion_policy LinearNormalizer.

Loss:    standard ε-prediction MSE — sample t∈{0..T-1}, q(x_t|x_0,ε), predict ε.
Inference: DDIM (default) for `num_inference_steps` denoise steps from x_T~N(0,I).
"""

import copy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

# Reuse the robomimic UNet — it's self-contained (no robomimic-specific
# infra in its forward) and ships with the pinned submodule.
from robomimic.models.diffusion_policy_nets import ConditionalUnet1D

from bc.diffusion_transformer import TransformerForDiffusion


@dataclass
class StateDiffusionPolicyConfig:
    # ---- backbone choice ----
    backbone: str = "unet1d"     # "unet1d" | "transformer"

    # ---- Obs encoder (used by both backbones) ----
    # "mlp"  : small MLP over the flat state -> obs_feat_dim (ibrl default).
    # "none" : raw state IS the conditioning vector (faithful robomimic lowdim
    #          — its ObservationGroupEncoder has no learnable net for low_dim
    #          modality, just flattens). global_cond_dim = state_dim * To.
    obs_encoder: str = "none"    # "none" (faithful lowdim) | "mlp"
    obs_feat_dim: int = 128
    obs_mlp_layers: tuple = (256,)

    # ---- UNet1D backbone (only used when backbone="unet1d") ----
    # robomimic defaults are [256,512,1024] (=65M params) — oversized for
    # state-only. We default to a moderate size for state.
    unet_down_dims: tuple = (64, 128, 256)
    diffusion_step_embed_dim: int = 256       # robomimic default
    kernel_size: int = 5
    n_groups: int = 8

    # ---- Transformer backbone (only used when backbone="transformer") ----
    # Defaults mirror the diffusion_policy paper's published lowdim config
    # (n_layer=8, n_head=4, n_emb=256, p_drop_attn=0.3, causal_attn=True,
    # n_cond_layers=0). ~2-3M params for Tp=16, action_dim=7.
    tf_n_layer: int = 8
    tf_n_head: int = 4
    tf_n_emb: int = 256
    tf_p_drop_emb: float = 0.0
    tf_p_drop_attn: float = 0.3
    tf_causal_attn: int = 1
    tf_n_cond_layers: int = 0

    # ---- Horizons (must match dataset.action_horizon == pred_horizon) ----
    pred_horizon: int = 16       # Tp
    action_horizon: int = 8      # Ta

    # ---- Noise scheduler — DDIM by default (fewer inference steps, ~same quality) ----
    scheduler: str = "ddim"      # ddim | ddpm
    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    beta_schedule: str = "squaredcos_cap_v2"  # robomimic default
    clip_sample: int = 1
    prediction_type: str = "epsilon"          # epsilon | sample
    # DDIM-only
    set_alpha_to_one: int = 1
    steps_offset: int = 0

    # ---- EMA (diffusers EMAModel warmup-decay, matches robomimic) ----
    # decay ramps up with optimization step: power=0.75 reaches 0.999 at ~10K
    # steps, 0.9999 at ~215K (crowsonkb's schedule, capped at max_value 0.9999).
    use_ema: int = 1
    ema_power: float = 0.75
    # DEPRECATED + UNUSED: kept only so old checkpoint cfg.yaml dumps (which
    # carry a diffusion_policy block with ema_decay) still parse under pyrallis.
    # The EMA now uses ema_power (diffusers EMAModel warmup schedule), not this.
    ema_decay: float = 0.9999

    # ---- input/output normalization (matches robomimic + the original
    # diffusion_policy LinearNormalizer; mirrors StateFlowPolicy §15) ----
    # obs   -> z-score (gaussian), like robomimic's hdf5_normalize_obs.
    # action-> min/max mapped to [-1, 1], like robomimic's action_config
    #          normalization: "min_max" (diffusion requires [-1,1] actions).
    # Stats are fit on the train split by the dataset and injected via
    # set_normalization() at construction; persisted as buffers so eval
    # auto-normalizes. Identity until set (no-op), so pre-normalization
    # checkpoints load + behave byte-for-byte.
    normalize_obs: int = 1
    normalize_action: int = 1


class StateDiffusionPolicy(nn.Module):
    """Diffusion Policy specialized to flat-state obs.

    Public surface (matches StateFlowPolicy):
        loss(batch)                  -> scalar
        act(obs, eval_mode=True)     -> (B, D) or (D,)
        act_chunk(obs, eval_mode)    -> (B, Ta, D) or (Ta, D)
        update_ema()
        apply_ema() / restore_from_ema()
    """

    def __init__(self, obs_shape, action_dim: int, cfg: StateDiffusionPolicyConfig):
        super().__init__()
        assert len(obs_shape) == 1, f"StateDiffusionPolicy expects flat state, got {obs_shape}"
        assert cfg.action_horizon <= cfg.pred_horizon
        assert cfg.backbone in ("unet1d", "transformer"), f"unknown backbone: {cfg.backbone}"
        assert cfg.obs_encoder in ("mlp", "none"), f"unknown obs_encoder: {cfg.obs_encoder}"
        self.cfg = cfg
        self.state_dim = obs_shape[0]
        self.action_dim = action_dim
        self.Tp = cfg.pred_horizon
        self.Ta = cfg.action_horizon
        self.backbone = cfg.backbone

        # ---- obs encoder (shared across backbones) ----
        # "none" => raw state is the conditioning vector (faithful robomimic
        # lowdim). "mlp" => small MLP -> obs_feat_dim (ibrl default).
        if cfg.obs_encoder == "none":
            self.obs_enc = nn.Identity()
            cond_dim = self.state_dim
        else:
            layers, prev = [], self.state_dim
            for h in cfg.obs_mlp_layers:
                layers += [nn.Linear(prev, h), nn.Mish()]
                prev = h
            layers += [nn.Linear(prev, cfg.obs_feat_dim)]
            self.obs_enc = nn.Sequential(*layers)
            cond_dim = cfg.obs_feat_dim

        # ---- noise prediction net ----
        if self.backbone == "unet1d":
            # FiLM-cond 1D UNet, To=1 → global_cond = (B, obs_feat_dim) flat.
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=action_dim,
                global_cond_dim=cond_dim * 1,
                diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
                down_dims=list(cfg.unet_down_dims),
                kernel_size=cfg.kernel_size,
                n_groups=cfg.n_groups,
            )
        else:
            # GPT-style encoder-decoder, To=1 → cond = (B, 1, cond_dim).
            self.noise_pred_net = TransformerForDiffusion(
                input_dim=action_dim,
                output_dim=action_dim,
                horizon=self.Tp,
                cond_dim=cond_dim,
                n_obs_steps=1,
                n_layer=cfg.tf_n_layer,
                n_head=cfg.tf_n_head,
                n_emb=cfg.tf_n_emb,
                p_drop_emb=cfg.tf_p_drop_emb,
                p_drop_attn=cfg.tf_p_drop_attn,
                causal_attn=bool(cfg.tf_causal_attn),
                n_cond_layers=cfg.tf_n_cond_layers,
            )

        # ---- scheduler ----
        if cfg.scheduler == "ddim":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=cfg.num_train_timesteps,
                beta_schedule=cfg.beta_schedule,
                clip_sample=bool(cfg.clip_sample),
                set_alpha_to_one=bool(cfg.set_alpha_to_one),
                steps_offset=cfg.steps_offset,
                prediction_type=cfg.prediction_type,
            )
        elif cfg.scheduler == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.num_train_timesteps,
                beta_schedule=cfg.beta_schedule,
                clip_sample=bool(cfg.clip_sample),
                prediction_type=cfg.prediction_type,
            )
        else:
            raise ValueError(f"unknown scheduler: {cfg.scheduler}")

        # ---- Normalization buffers (identity until set_normalization()) ----
        # Registered buffers ⇒ saved in state_dict, moved by .to(device), and
        # NOT touched by EMA (the EMA ModuleList only wraps obs_enc +
        # noise_pred_net, and EMAModel.step iterates that module's params).
        # Identity defaults make the transform a no-op for back-compat.
        self.register_buffer("norm_state_mean", torch.zeros(self.state_dim))
        self.register_buffer("norm_state_std", torch.ones(self.state_dim))
        self.register_buffer("norm_action_min", -torch.ones(action_dim))
        self.register_buffer("norm_action_max", torch.ones(action_dim))

        # ---- EMA (diffusers EMAModel, matches robomimic DiffusionPolicyUNet) ----
        # The EMA wraps the trainable nets (obs_enc + noise_pred_net). We hold a
        # ModuleList over them inside a plain python list so it is NOT registered
        # as a submodule (no duplicate params in self.state_dict / optimizer) and
        # the EMAModel's averaged_model (a deepcopy) stays out of self.parameters().
        # EMAModel is built lazily on the first update_ema() so its averaged_model
        # lands on whatever device the params are on by then. Serialized via the
        # state_dict / load_state_dict overrides below so standalone eval picks up
        # the same averaged weights in-training eval used.
        self._ema_src_holder = [nn.ModuleList([self.obs_enc, self.noise_pred_net])]
        self.ema: "EMAModel | None" = None
        self._ema_backup = None

        # Range check is done once per training run (matches robomimic).
        self._action_check_done = False

    # =================================================================
    # core forward / loss
    # =================================================================

    def _obs_cond(self, state: torch.Tensor) -> torch.Tensor:
        """Encode flat state → conditioning vector (B, obs_feat_dim)."""
        return self.obs_enc(state)

    def _predict_noise(
        self,
        noisy_a: torch.Tensor,
        timesteps: torch.Tensor,
        obs_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Backbone dispatch: returns model output (noise pred or sample, per scheduler).

        UNet1D expects `global_cond` as a flat (B, obs_feat_dim) vector;
        the transformer expects `cond` as (B, To=1, obs_feat_dim) per-step.
        """
        if self.backbone == "unet1d":
            return self.noise_pred_net(
                sample=noisy_a, timestep=timesteps, global_cond=obs_feat,
            )
        # transformer
        return self.noise_pred_net(
            sample=noisy_a, timestep=timesteps, cond=obs_feat.unsqueeze(1),
        )

    # =================================================================
    # input/output normalization (mirrors StateFlowPolicy §15)
    # =================================================================

    @torch.no_grad()
    def set_normalization(self, state_mean=None, state_std=None,
                          action_min=None, action_max=None) -> None:
        """Install dataset-fit normalization stats into the buffers.

        Called once at train time from the dataset stats. No-op for any arg
        left None (e.g. image mode has no state stats). Std is floored to
        avoid divide-by-zero on (near-)constant state dims.
        """
        if self.cfg.normalize_obs and state_mean is not None:
            self.norm_state_mean.copy_(state_mean.to(self.norm_state_mean))
            self.norm_state_std.copy_(state_std.to(self.norm_state_std).clamp_min(1e-6))
        if self.cfg.normalize_action and action_min is not None:
            self.norm_action_min.copy_(action_min.to(self.norm_action_min))
            self.norm_action_max.copy_(action_max.to(self.norm_action_max))

    def _normalize_state(self, s: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_obs:
            return s
        return (s - self.norm_state_mean) / self.norm_state_std

    def _normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        """Map raw action [min, max] → [-1, 1] (broadcasts over the chunk axis)."""
        if not self.cfg.normalize_action:
            return a
        rng = (self.norm_action_max - self.norm_action_min).clamp_min(1e-6)
        return 2.0 * (a - self.norm_action_min) / rng - 1.0

    def _denormalize_action(self, a: torch.Tensor) -> torch.Tensor:
        """Inverse of _normalize_action: [-1, 1] → raw [min, max]."""
        if not self.cfg.normalize_action:
            return a
        rng = (self.norm_action_max - self.norm_action_min).clamp_min(1e-6)
        return (a + 1.0) * 0.5 * rng + self.norm_action_min

    def loss(self, batch):
        state = self._normalize_state(batch.obs["state"])   # (B, S)
        a = self._normalize_action(batch.action["action_chunk"])  # (B, Tp, D)
        assert a.shape[1] == self.Tp, (
            f"action_chunk Tp={a.shape[1]} doesn't match policy pred_horizon={self.Tp} "
            "— check DatasetConfig.action_horizon"
        )
        assert a.shape[2] == self.action_dim

        if not self._action_check_done:
            # Diffusion Policy expects actions in [-1, 1] (clip_sample + DDIM).
            # With normalize_action ON this holds by construction; with it OFF
            # this asserts the raw actions are already normalized.
            in_range = ((-1 - 1e-4 <= a) & (a <= 1 + 1e-4)).all().item()
            assert in_range, (
                "actions out of [-1, 1] — Diffusion Policy requires normalized actions. "
                "Enable normalize_action (default) or pre-normalize the action space."
            )
            self._action_check_done = True

        B = a.shape[0]
        obs_cond = self._obs_cond(state)                    # (B, obs_feat_dim)

        # sample noise + diffusion timestep
        noise = torch.randn_like(a)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=a.device, dtype=torch.long,
        )
        noisy_a = self.noise_scheduler.add_noise(a, noise, timesteps)  # (B, Tp, D)

        # predict noise residual (UNet1D or transformer, dispatched by backbone)
        noise_pred = self._predict_noise(noisy_a, timesteps, obs_cond)

        return F.mse_loss(noise_pred, noise)

    # =================================================================
    # inference (DDIM denoise loop)
    # =================================================================

    @torch.no_grad()
    def _sample_full_chunk(self, state: torch.Tensor) -> torch.Tensor:
        """Run the reverse diffusion process → (B, Tp, D) in NORMALIZED action space."""
        B = state.shape[0]
        obs_cond = self._obs_cond(self._normalize_state(state))
        x = torch.randn(B, self.Tp, self.action_dim,
                        device=state.device, dtype=state.dtype)
        self.noise_scheduler.set_timesteps(self.cfg.num_inference_steps)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self._predict_noise(x, k, obs_cond)
            x = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=x,
            ).prev_sample
        return x

    def act_chunk(self, obs, *, eval_mode=True):
        assert eval_mode
        assert not self.training
        state = obs["state"]
        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True
        chunk = self._sample_full_chunk(state)[:, : self.Ta].contiguous()
        chunk = self._denormalize_action(chunk)             # → raw action scale
        if unsqueezed:
            chunk = chunk.squeeze(0)
        return chunk.cpu()

    def act(self, obs, *, eval_mode=True, cpu=True):
        assert eval_mode
        assert not self.training
        state = obs["state"]
        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True
        chunk = self._sample_full_chunk(state)
        action = self._denormalize_action(chunk[:, 0])      # → raw action scale
        if unsqueezed:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

    # =================================================================
    # EMA (diffusers EMAModel(power=...), matches robomimic)
    # =================================================================

    def _ema_src(self) -> nn.Module:
        """The (unregistered) ModuleList over the trainable nets EMA tracks."""
        return self._ema_src_holder[0]

    def update_ema(self) -> None:
        if not self.cfg.use_ema:
            return
        if self.ema is None:
            # Built lazily so averaged_model lands on the params' current device.
            self.ema = EMAModel(model=self._ema_src(), power=self.cfg.ema_power)
        self.ema.step(self._ema_src())

    @torch.no_grad()
    def apply_ema(self) -> bool:
        """Swap live weights for the EMA average (state_dict swap incl. buffers)."""
        if not self.cfg.use_ema or self.ema is None:
            return False
        src = self._ema_src()
        self._ema_backup = copy.deepcopy(src.state_dict())
        device = next(src.parameters()).device
        avg_sd = {k: v.to(device) for k, v in self.ema.averaged_model.state_dict().items()}
        src.load_state_dict(avg_sd)
        return True

    @torch.no_grad()
    def restore_from_ema(self) -> None:
        if not self._ema_backup:
            return
        self._ema_src().load_state_dict(self._ema_backup)
        self._ema_backup = None

    # =================================================================
    # state_dict / load_state_dict: persist the EMA averaged weights
    # =================================================================
    _EMA_PREFIX = "_ema_shadow__"

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars,
        )
        if self.ema is not None:
            for name, t in self.ema.averaged_model.state_dict().items():
                sd[f"{prefix}{self._EMA_PREFIX}{name}"] = t.detach().clone()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        state_dict = dict(state_dict)
        # Tolerate pre-normalization checkpoints: backfill any missing
        # normalization buffers with the current (identity) values so a strict
        # load doesn't fail (identity ⇒ normalization is a no-op).
        own = nn.Module.state_dict(self)
        for bname in ("norm_state_mean", "norm_state_std",
                      "norm_action_min", "norm_action_max"):
            if bname not in state_dict and bname in own:
                state_dict[bname] = own[bname]
        ema_keys = [k for k in state_dict if self._EMA_PREFIX in k]
        new_ema: "dict[str, torch.Tensor]" = {}
        for k in ema_keys:
            param_name = k.split(self._EMA_PREFIX, 1)[1]
            new_ema[param_name] = state_dict.pop(k).clone()
        ret = super().load_state_dict(state_dict, strict=strict)
        if new_ema:
            if self.ema is None:
                self.ema = EMAModel(model=self._ema_src(), power=self.cfg.ema_power)
            device = next(self.parameters()).device
            self.ema.averaged_model.load_state_dict(
                {n: t.to(device) for n, t in new_ema.items()}
            )
        return ret
