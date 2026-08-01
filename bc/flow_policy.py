"""State-based flow-matching BC policy for ibrl.

Mirrors the design of
  - robomimic/algo/diffusion_policy.py (action chunking, EMA, pred>exec horizons)
  - openpi flow_policy.py (rectified flow / linear OT, Fourier time)

What's different vs robomimic's diffusion policy (when backbone="unet1d"):
  - Continuous time t in [0,1] instead of 100 discrete timesteps
  - Linear OT path x_t = (1-t)*z + t*a, target field v = a - z (no DDPM scheduler)
  - Euler ODE integration at inference (fixed step count)
  This is now an apples-to-apples comparison: same `ConditionalUnet1D` backbone,
  same MSE-mean loss reduction, same EMA / chunking. Only the
  sampling-method/training-target differ. With backbone="mlp" you get the
  legacy lightweight MLP variant.

Horizon convention (mirrors robomimic):
  - pred_horizon (Tp): chunk length the network predicts (and trains on)
  - action_horizon (Ta): chunk length actually executed per re-prediction (Ta <= Tp)
  Predicting more than executed lets the network "look ahead" Tp - Ta steps
  beyond the executed horizon, improving smoothness at chunk boundaries.

EMA: a shadow copy of parameters is updated every train step via
  shadow <- decay*shadow + (1-decay)*param. At eval time, online params are
  swapped with the shadow via apply_ema()/restore_from_ema(). Standard
  diffusion/flow trick — markedly improves stability and final perf.

Integration: chunked eval relies on `act_chunk(obs)` returning (B, Ta, D).
evaluate/multi_process_eval.py maintains a per-worker queue refilled from
act_chunk; a non-chunked `act(obs)` fallback exists for back-compat.
"""

from dataclasses import dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F

# Reuse robomimic's ConditionalUnet1D so the architectural side of the
# flow-vs-diffusion comparison is exactly matched.
from robomimic.models.diffusion_policy_nets import ConditionalUnet1D


@dataclass
class StateFlowPolicyConfig:
    # ---- backbone choice ----
    backbone: str = "unet1d"   # "unet1d" (default, matches diffusion) | "mlp"

    # ---- MLP backbone (used only when backbone="mlp") ----
    # Defaults bumped to 1024×6 (~5M params) so the MLP variant is in the
    # same ballpark as the UNet1D variant when you want to ablate
    # conditioning style (concat vs FiLM) at matched capacity.
    num_layer: int = 6
    hidden_dim: int = 1024
    state_proj_dim: int = 128
    action_proj_dim: int = 256
    time_embed_dim: int = 128
    fourier_dim: int = 64
    dropout: float = 0.0
    layer_norm: int = 1

    # ---- UNet1D backbone (used only when backbone="unet1d") ----
    # Same defaults as bc/diffusion_policy.py so swapping `use_flow` <->
    # `use_diffusion` gives identical network capacity.
    obs_feat_dim: int = 128
    obs_mlp_layers: tuple = (256,)
    unet_down_dims: tuple = (64, 128, 256)
    diffusion_step_embed_dim: int = 128
    kernel_size: int = 5
    n_groups: int = 8
    # SinusoidalPosEmb inside ConditionalUnet1D was designed for the DDPM
    # convention where the diffusion step is an integer in [0, T). Our flow
    # time is continuous t ∈ [0, 1], so we multiply by `time_scale` before
    # feeding the UNet — otherwise the sinusoidal frequencies only cover a
    # tiny corner of their range and the time embedding collapses.
    time_scale: float = 1000.0

    # ---- Horizons (robomimic convention) ----
    pred_horizon: int = 16     # Tp — chunk length predicted/trained; must match dataset.action_horizon
    action_horizon: int = 8    # Ta — chunk length executed per re-prediction; Ta <= Tp
    num_inference_steps: int = 10  # Euler ODE steps at inference time

    # ---- EMA for eval-time weights ----
    use_ema: int = 1
    ema_decay: float = 0.9999

    # ---- Input/output normalization (fit on the training data) ----
    # State z-score (s-mean)/std, action min/max → [-1,1]. Filled via
    # set_normalization() from dataset stats at train time and persisted as
    # buffers in the checkpoint (so eval auto-normalizes, no recompute).
    # Generative heads (flow/diffusion) are far more sensitive to input/
    # output scale than the deterministic MLP baselines — the obs encoder
    # otherwise burns capacity aligning the 37-dim state's heterogeneous
    # scales (eef_pos ~1e-2 m, quat ~1, object-state mixed), which hurts most
    # in the low-data regime. Default ON; buffers stay identity until
    # set_normalization() is called, so pre-normalization checkpoints (no
    # buffers / no call) load and behave byte-for-byte identically.
    normalize_obs: int = 1
    normalize_action: int = 1

    # ---- Classifier-Free Guidance (CFG) for IL co-training ----
    # Implements the CFG arm of CFG-ADDA (Lei et al. 2026, "Science of
    # Co-Training"): concat a one-hot domain label `c` to the obs encoder
    # output so the score is explicitly conditioned on source vs target;
    # randomly dropout `c → ∅` during training; at inference interpolate
    # `s̃ = (1+λ)·s(c) − λ·s(∅)` to control how much of the per-domain
    # specificity is transferred. Default OFF — when disabled the policy is
    # byte-for-byte the original Flow BC (no extra params, no extra forward).
    # Only meaningful with CoTrainDataset (provides batch.obs["domain_id"]).
    cfg_enabled: int = 0
    cfg_num_domains: int = 2          # source=0, target=1
    cfg_dropout_p: float = 0.2        # train: prob of replacing c with ∅
    cfg_guidance_lambda: float = -0.5 # paper default; <0 transfers "averaged" knowledge
    cfg_inference_domain_id: int = 1  # which one-hot to condition on at inference (1=target)


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal Fourier features for continuous flow time t in [0,1].

    Mirrors pi0.5's design: project t through fixed linear frequencies, take
    sin/cos, then a small MLP. Output dim = `out_dim`. Used by the MLP
    backbone only — the UNet1D backbone has its own SinusoidalPosEmb.
    """

    def __init__(self, fourier_dim: int, out_dim: int):
        super().__init__()
        assert fourier_dim % 2 == 0
        half = fourier_dim // 2
        # Fixed linear frequency basis (non-learnable for simplicity).
        freqs = torch.arange(half, dtype=torch.float32) + 1.0
        self.register_buffer("freqs", freqs * math.pi)
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape (B,) or (B,1); broadcast to (B, half)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        proj = t * self.freqs  # (B, half)
        emb = torch.cat([proj.sin(), proj.cos()], dim=-1)  # (B, fourier_dim)
        return self.proj(emb)  # (B, out_dim)


class StateFlowPolicy(nn.Module):
    """Rectified-flow BC policy over action chunks (Tp steps).

    Loss:    sample t ~ U(0,1), z ~ N(0,I), x_t = (1-t) z + t a;
             predict v(x_t, t, state) ≈ a - z; MSE (mean reduction).
    Inference: integrate dx/dt = v(x, t, state) from t=0 (x=z) to t=1 using
               `num_inference_steps` Euler steps; slice first Ta actions.

    Backbones:
      - "unet1d": robomimic's ConditionalUnet1D (FiLM-cond 1D U-Net over the
        Tp axis). Matches StateDiffusionPolicy on architecture, so the
        flow-vs-diffusion difference is purely the training objective.
      - "mlp":    legacy MLP-on-flattened-chunk variant. Cheap and self-
        contained but lacks temporal inductive bias along Tp.

    Public surface (matches StateBcPolicy where compatible):
        loss(batch)                  -> scalar
        act(obs, eval_mode=True)     -> (B, D)        — single action, back-compat
        act_chunk(obs, eval_mode)    -> (B, Ta, D)    — chunked, used by evaluate
        update_ema()                 — call after every optim.step()
        apply_ema() / restore_from_ema() — wrap eval to use shadow weights
    """

    def __init__(self, obs_shape, action_dim: int, cfg: StateFlowPolicyConfig):
        super().__init__()
        assert len(obs_shape) == 1, f"StateFlowPolicy expects flat state, got {obs_shape}"
        assert cfg.action_horizon <= cfg.pred_horizon, (
            f"action_horizon ({cfg.action_horizon}) must be <= pred_horizon "
            f"({cfg.pred_horizon}) — we predict Tp and execute the first Ta."
        )
        assert cfg.backbone in ("unet1d", "mlp"), f"unknown backbone: {cfg.backbone}"
        self.cfg = cfg
        self.action_dim = action_dim
        self.Tp = cfg.pred_horizon
        self.Ta = cfg.action_horizon
        self.state_dim = obs_shape[0]
        self.flat_action_dim = self.Tp * action_dim
        self.backbone = cfg.backbone

        if self.backbone == "unet1d":
            self._build_unet1d_backbone(cfg)
        else:
            self._build_mlp_backbone(cfg)

        # ---- Normalization buffers (identity until set_normalization()) ----
        # Registered buffers ⇒ saved in state_dict, moved by .to(device), and
        # NOT touched by EMA (which only iterates named_parameters). Identity
        # defaults make the transform a no-op for back-compat.
        self.register_buffer("norm_state_mean", torch.zeros(self.state_dim))
        self.register_buffer("norm_state_std", torch.ones(self.state_dim))
        self.register_buffer("norm_action_min", -torch.ones(action_dim))
        self.register_buffer("norm_action_max", torch.ones(action_dim))

        # ---- EMA shadow state (lazily initialized on first update_ema()) ----
        # Stored as plain dicts (not as nn.Parameter/buffer). Persisted via
        # the state_dict / load_state_dict overrides below, so checkpoints
        # *do* include the EMA shadow (resumable training; standalone eval
        # automatically loads EMA weights matching what in-training eval saw).
        self._ema_initialized = False
        self._ema_state: "dict[str, torch.Tensor]" = {}
        self._ema_backup: "dict[str, torch.Tensor]" = {}

    # =================================================================
    # input/output normalization
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

    # =================================================================
    # backbone construction
    # =================================================================

    def _build_mlp_backbone(self, cfg: StateFlowPolicyConfig) -> None:
        self.state_proj = nn.Sequential(
            nn.Linear(self.state_dim, cfg.state_proj_dim), nn.GELU(),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(self.flat_action_dim, cfg.action_proj_dim), nn.GELU(),
        )
        self.time_embed = FourierTimeEmbedding(cfg.fourier_dim, cfg.time_embed_dim)

        cfg_extra = cfg.cfg_num_domains if cfg.cfg_enabled else 0
        in_dim = cfg.state_proj_dim + cfg.action_proj_dim + cfg.time_embed_dim + cfg_extra
        layers = []
        prev = in_dim
        for _ in range(cfg.num_layer):
            layers.append(nn.Linear(prev, cfg.hidden_dim))
            layers.append(nn.GELU())
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(cfg.hidden_dim))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = cfg.hidden_dim
        layers.append(nn.Linear(prev, self.flat_action_dim))  # velocity head
        self.net = nn.Sequential(*layers)

    def _build_unet1d_backbone(self, cfg: StateFlowPolicyConfig) -> None:
        # Small obs encoder (matches bc/diffusion_policy.py).
        layers, prev = [], self.state_dim
        for h in cfg.obs_mlp_layers:
            layers += [nn.Linear(prev, h), nn.Mish()]
            prev = h
        layers += [nn.Linear(prev, cfg.obs_feat_dim)]
        self.obs_enc = nn.Sequential(*layers)

        # FiLM-conditioned 1D UNet over the Tp axis. global_cond = obs_feat
        # (concat with one-hot domain label when CFG is enabled — see
        # forward_velocity for where the concat happens).
        cfg_extra = cfg.cfg_num_domains if cfg.cfg_enabled else 0
        self.velocity_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=cfg.obs_feat_dim + cfg_extra,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=list(cfg.unet_down_dims),
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
        )

    # =================================================================
    # core flow forward / loss / sampling
    # =================================================================

    def _cfg_onehot(self, state: torch.Tensor, domain_onehot) -> torch.Tensor:
        """Resolve the CFG one-hot to a (B, num_domains) tensor on `state`'s device.

        Passing `None` returns the ∅ (all-zero) condition.
        """
        B = state.shape[0]
        if domain_onehot is None:
            return torch.zeros(B, self.cfg.cfg_num_domains,
                               device=state.device, dtype=state.dtype)
        return domain_onehot.to(device=state.device, dtype=state.dtype)

    def forward_velocity(self, x_t: torch.Tensor, t: torch.Tensor, state: torch.Tensor,
                         domain_onehot: "torch.Tensor | None" = None) -> torch.Tensor:
        """Predict velocity field v(x_t, t, state[, c]).

        x_t:   (B, Tp, action_dim);  t: (B,) in [0,1];  state: (B, state_dim)
        domain_onehot: optional (B, cfg_num_domains) one-hot when cfg_enabled.
                       `None` ⇒ ∅ (zeros) — the unconditional branch.
        Returns v_pred (B, Tp, action_dim).
        """
        if self.backbone == "unet1d":
            obs_cond = self.obs_enc(state)
            if self.cfg.cfg_enabled:
                obs_cond = torch.cat([obs_cond, self._cfg_onehot(state, domain_onehot)],
                                     dim=-1)
            # ConditionalUnet1D's SinusoidalPosEmb expects an integer-ish
            # range; rescale continuous t∈[0,1] up so the embedding spans
            # its full frequency band (mirrors the DDPM convention).
            t_scaled = t * self.cfg.time_scale
            return self.velocity_net(
                sample=x_t, timestep=t_scaled, global_cond=obs_cond,
            )
        # MLP backbone — flatten chunk to a single vector
        B = x_t.shape[0]
        x_flat = x_t.reshape(B, self.flat_action_dim)
        s_feat = self.state_proj(state)
        a_feat = self.action_proj(x_flat)
        t_feat = self.time_embed(t)
        parts = [s_feat, a_feat, t_feat]
        if self.cfg.cfg_enabled:
            parts.append(self._cfg_onehot(state, domain_onehot))
        h = torch.cat(parts, dim=-1)
        v_flat = self.net(h)
        return v_flat.reshape(B, self.Tp, self.action_dim)

    def loss(self, batch):
        """Rectified-flow MSE loss on velocity prediction (mean reduction).

        Expects:
            batch.obs["state"]              : (B, state_dim)
            batch.action["action_chunk"]    : (B, Tp, action_dim)
        """
        state = self._normalize_state(batch.obs["state"])
        a = self._normalize_action(batch.action["action_chunk"])  # (B, Tp, action_dim)
        B = a.shape[0]
        assert a.shape[1] == self.Tp, (
            f"action_chunk Tp={a.shape[1]} doesn't match policy pred_horizon={self.Tp} "
            "— check DatasetConfig.action_horizon"
        )
        assert a.shape[2] == self.action_dim

        t = torch.rand(B, device=a.device)
        z = torch.randn_like(a)                          # (B, Tp, D)
        t_view = t.view(B, 1, 1)
        x_t = (1.0 - t_view) * z + t_view * a            # (B, Tp, D)
        v_target = a - z                                 # (B, Tp, D)

        # CFG: build the one-hot condition with per-sample dropout to ∅, so
        # the same network learns both s(c) (conditional) and s(∅)
        # (unconditional). Disabled ⇒ pass None and skip the concat.
        if self.cfg.cfg_enabled:
            domain_id = batch.obs.get("domain_id")
            assert domain_id is not None, (
                "flow_policy.cfg_enabled=1 but batch has no `domain_id` — "
                "CFG requires CoTrainDataset (it tags samples 0=source / 1=target)"
            )
            one_hot = F.one_hot(
                domain_id.long(), num_classes=self.cfg.cfg_num_domains
            ).to(dtype=a.dtype)
            drop = torch.rand(B, device=a.device) < self.cfg.cfg_dropout_p
            one_hot = one_hot.masked_fill(drop.unsqueeze(-1), 0.0)
        else:
            one_hot = None

        v_pred = self.forward_velocity(x_t, t, state, domain_onehot=one_hot)
        # Mean over all elements — matches bc/diffusion_policy.py (F.mse_loss
        # default). Previously summed over (Tp*D) which scaled the loss by
        # ~Tp*D vs diffusion, effectively over-tuning lr.
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def _sample_full_chunk(self, state: torch.Tensor) -> torch.Tensor:
        """Run Euler ODE → return full predicted chunk (B, Tp, action_dim).

        With CFG enabled, each step costs two velocity forwards (conditional
        on `cfg_inference_domain_id` and unconditional ∅) and the velocities
        are interpolated as `(1+λ)·v_c − λ·v_∅`. `λ < 0` blends the
        unconditional ("averaged across domains") direction in, transferring
        less per-domain specificity.
        """
        B = state.shape[0]
        x = torch.randn(B, self.Tp, self.action_dim,
                        device=state.device, dtype=state.dtype)
        N = self.cfg.num_inference_steps
        dt = 1.0 / N

        if self.cfg.cfg_enabled:
            c_idx = torch.full((B,), int(self.cfg.cfg_inference_domain_id),
                               dtype=torch.long, device=state.device)
            c = F.one_hot(c_idx, num_classes=self.cfg.cfg_num_domains).to(dtype=state.dtype)
            lam = float(self.cfg.cfg_guidance_lambda)
        for i in range(N):
            t = torch.full((B,), i * dt, device=state.device, dtype=state.dtype)
            if self.cfg.cfg_enabled:
                v_c = self.forward_velocity(x, t, state, domain_onehot=c)
                v_null = self.forward_velocity(x, t, state, domain_onehot=None)
                v = (1.0 + lam) * v_c - lam * v_null
            else:
                v = self.forward_velocity(x, t, state)
            x = x + v * dt
        return x

    # =================================================================
    # eval-facing inference: act() and act_chunk()
    # =================================================================

    def act_chunk(self, obs, *, eval_mode=True):
        """Return a chunk of `action_horizon` (Ta) actions per batch element.

        Args:
            obs: {"state": (B, state_dim)} or {"state": (state_dim,)} for 1-env.
        Returns:
            (B, Ta, action_dim) on CPU. Caller (evaluator) is responsible for
            executing them in sequence before requesting another chunk.
        """
        assert eval_mode
        assert not self.training
        state = self._normalize_state(obs["state"])

        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True

        chunk_full = self._sample_full_chunk(state)         # (B, Tp, D), normalized
        chunk_exec = chunk_full[:, : self.Ta].contiguous()  # (B, Ta, D)
        chunk_exec = self._denormalize_action(chunk_exec)   # → raw action scale

        if unsqueezed:
            chunk_exec = chunk_exec.squeeze(0)              # (Ta, D)
        return chunk_exec.cpu()

    def act(self, obs, *, eval_mode=True, cpu=True):
        """Back-compat single-step action: returns first action of fresh chunk."""
        assert eval_mode
        assert not self.training
        state = self._normalize_state(obs["state"])

        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True

        chunk = self._sample_full_chunk(state)
        action = self._denormalize_action(chunk[:, 0])     # → raw action scale
        if unsqueezed:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

    # =================================================================
    # EMA: shadow params for stable eval
    # =================================================================

    def _init_ema(self) -> None:
        for name, param in self.named_parameters():
            self._ema_state[name] = param.data.detach().clone()
        self._ema_initialized = True

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update EMA shadow with current online params. Call after each optim.step()."""
        if not self.cfg.use_ema:
            return
        if not self._ema_initialized:
            self._init_ema()
            return
        d = self.cfg.ema_decay
        for name, param in self.named_parameters():
            self._ema_state[name].mul_(d).add_(param.data, alpha=1.0 - d)

    @torch.no_grad()
    def apply_ema(self) -> bool:
        """Swap online weights with EMA shadow. Returns True if EMA was applied."""
        if not self.cfg.use_ema or not self._ema_initialized:
            return False
        for name, param in self.named_parameters():
            self._ema_backup[name] = param.data.detach().clone()
            param.data.copy_(self._ema_state[name])
        return True

    @torch.no_grad()
    def restore_from_ema(self) -> None:
        """Restore online weights from backup. Mirror of apply_ema()."""
        if not self._ema_backup:
            return
        for name, param in self.named_parameters():
            if name in self._ema_backup:
                param.data.copy_(self._ema_backup[name])
        self._ema_backup.clear()

    # =================================================================
    # state_dict / load_state_dict: persist EMA shadow alongside params
    # =================================================================
    _EMA_PREFIX = "_ema_shadow__"

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars,
        )
        if self._ema_initialized:
            for name, t in self._ema_state.items():
                sd[f"{prefix}{self._EMA_PREFIX}{name}"] = t.detach().clone()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        # Split EMA keys out so super().load_state_dict isn't confused by them.
        state_dict = dict(state_dict)
        # Tolerate pre-normalization checkpoints: backfill any missing
        # normalization buffers with the current (identity) values so a strict
        # load doesn't fail. Identity buffers ⇒ normalization is a no-op, so
        # such checkpoints behave exactly as before this feature existed.
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
        # Install (and move to current param device) only after the base load
        # succeeded, so partial-failure states don't leak in.
        if new_ema:
            device = next(self.parameters()).device
            self._ema_state = {n: t.to(device) for n, t in new_ema.items()}
            self._ema_initialized = True
            self._ema_backup = {}
        return ret
