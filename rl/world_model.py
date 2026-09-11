"""State-based latent world model for the SimDist baseline (TD-MPC2 style).

SimDist (RSS 2026) distills a simulator into a latent world model and plans with
MPPI. The reference impl is built on TD-MPC2; for state-based control the latent
world model IS essentially TD-MPC2's: an encoder maps the observation to a
latent z, a latent dynamics model predicts z_{t+1}, and reward / value / policy
heads decode from z. We use MLPs (no Transformer; state obs needs no attention).

Heads (all MLPs):
  encoder : state(obs_dim) -> z(latent_dim)          (SimNorm'd)
  dynamics: (z, a)         -> z'                       (SimNorm'd)
  reward  : (z, a)         -> r (scalar OR num_bins two-hot logits)
  value   : z              -> V (scalar OR num_bins)   + EMA target copy
  policy  : z_pi           -> a in [-1,1] (tanh)       (proposal policy for MPPI)

TD-MPC2-aligned options (all flag-gated, default = legacy behavior so old ckpts
load byte-for-byte under strict=True):
  * discrete_heads : two-hot reward+value heads over `num_bins` (symlog space) +
    cross-entropy loss (math: symlog/two_hot/soft_ce). Far more robust than a
    scalar MSE head on sparse/peaky targets, and gives finer value resolution.
  * shared_encoder : the policy reuses the dynamics encoder (no separate enc_pi),
    so the planner can roll the policy DIRECTLY in latent (no decode bridge).
    Paired with detach_pi (default on) so the policy-BC gradient does NOT flow
    into the shared encoder (avoids the policy-vs-consistency fight, TD-MPC2 also
    detaches the latent for the policy update).
  * activation     : "relu" (legacy) or "mish" (TD-MPC2).
  * enc_lr_scale   : encoder trained at lr*enc_lr_scale (TD-MPC2: 0.3) — handled
    in the agent's optimizer, not here.

This module holds only the networks + forward helpers. The distillation /
finetune losses and the optimizers live in rl/simdist_agent.py.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# PIXEL arm config. Imported at module level (no cycle: pixel_obs_encoder only
# pulls from networks/) because pyrallis resolves dataclass type hints eagerly --
# a forward-reference string annotation fails to decode with
# "cannot create weak reference to 'str' object".
from rl.pixel_obs_encoder import PixelObsEncoderConfig


class SimNorm(nn.Module):
    """Simplicial normalization (TD-MPC2): split the latent into groups of
    `group_dim` and softmax each group. Bounds the latent (each group sums to 1)
    so the dynamics latent can't explode over a long rollout/training."""

    def __init__(self, group_dim: int = 8):
        super().__init__()
        self.group_dim = group_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.group_dim)
        x = torch.softmax(x, dim=-1)
        return x.reshape(*shp)


def _act(name: str) -> nn.Module:
    return nn.Mish(inplace=False) if name == "mish" else nn.ReLU()


def mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layer: int,
        layer_norm: int = 1, act_out: str = "none", act: str = "relu") -> nn.Sequential:
    """Generic MLP: num_layer hidden layers (Linear+[LN]+act) then a linear head.

    act:     hidden-layer activation, "relu" (legacy) or "mish" (TD-MPC2).
    act_out: 'none' (linear), 'tanh', or 'relu' on the output.
    """
    dims = [in_dim] + [hidden_dim] * num_layer
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(_act(act))
    layers.append(nn.Linear(dims[-1], out_dim))
    if act_out == "tanh":
        layers.append(nn.Tanh())
    elif act_out == "relu":
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ----------------------------------------------------- two-hot / symlog math
def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def two_hot(x: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Scalar [B] (or [B,1]) -> two-hot distribution [B, num_bins] in symlog space."""
    x = x.reshape(-1)
    x = torch.clamp(symlog(x), vmin, vmax)
    bin_size = (vmax - vmin) / (num_bins - 1)
    idx = ((x - vmin) / bin_size).floor().long()
    w = (x - vmin) / bin_size - idx.float()
    idx = idx.clamp(0, num_bins - 1)
    idx2 = (idx + 1).clamp(0, num_bins - 1)
    out = torch.zeros(x.shape[0], num_bins, device=x.device, dtype=x.dtype)
    out.scatter_(-1, idx.unsqueeze(-1), (1 - w).unsqueeze(-1))
    out.scatter_add_(-1, idx2.unsqueeze(-1), w.unsqueeze(-1))
    return out


def two_hot_inv(logits: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Two-hot logits [B, num_bins] -> scalar [B] (softmax expectation, symexp'd)."""
    bins = torch.linspace(vmin, vmax, num_bins, device=logits.device, dtype=logits.dtype)
    p = torch.softmax(logits, dim=-1)
    return symexp((p * bins).sum(-1))


def soft_ce(logits: torch.Tensor, target_scalar: torch.Tensor,
            num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Cross-entropy of two-hot(target) vs logits. Returns [B] (caller .mean())."""
    t = two_hot(target_scalar, num_bins, vmin, vmax)
    logp = torch.log_softmax(logits, dim=-1)
    return -(t * logp).sum(-1)


@dataclass
class WorldModelConfig:
    # encoder_type: "latent" (TD-MPC2 learned latent + SimNorm, default — old ckpts
    # load byte-for-byte) or "identity" (RAW-STATE arm: no learned encoder, the
    # "latent" z IS the z-scored raw state, dynamics is a residual next-state
    # predictor, decode is the analytic denorm inverse, the policy runs directly on
    # the normalized state). For state-based control the latent buys little (37-dim
    # state is already compact) while costing the whole §17 stability saga
    # (SimNorm / enc_pi / decoder bridge / policy-vs-consistency fight) — this arm
    # drops all of that and tests whether the latent helps at all.
    # "pixel" (NEW): images + prop -> rl/pixel_obs_encoder.py -> the same trunk
    # MLP -> SimNorm. Requires shared_encoder=1 and decoder_coef=0 (there is no
    # image decoder, and with shared_encoder the planner never needs decode()).
    encoder_type: str = "latent"
    # NeRF-style Fourier encoding of the raw state at the ENCODER input (latent
    # arm only). 0 = OFF (byte-for-byte unchanged; no input_pe module, no new
    # state_dict keys, old ckpts load strict). >0 expands state to
    # obs_dim*(1+2*L) via [x, sin(2^k*pi*x), cos] before the encoder MLP; z-score
    # stats live in the pe buffers (identity until set_input_fourier_stats).
    input_fourier_l: int = 0
    input_fourier_include_raw: int = 1
    input_fourier_normalize: int = 1
    latent_dim: int = 512
    mlp_hidden_dim: int = 512
    num_layer: int = 2
    layer_norm: int = 1
    simnorm_dim: int = 8        # SimNorm group size (latent_dim must be divisible by it)
    # PIXEL arm only (encoder_type=="pixel"). Default-constructed and unused
    # otherwise, so it adds no parameters and no state_dict keys to the state
    # arms -- old ckpts still load strict.
    pixel: PixelObsEncoderConfig = field(default_factory=PixelObsEncoderConfig)
    # ENCODER GRADIENT STARVATION mitigation (pixel arm; default 0 = legacy).
    # With shared_encoder=1, detach_pi=1 and decoder_coef=0 -- the pixel recipe --
    # the encoder's ONLY gradient source is the t=0 latent flowing into
    # consistency + reward + value. The dominant loss (policy_coef=10) is
    # detached and the consistency target is stop-grad, so a CONSTANT encoder is a
    # feasible consistency solution (SimNorm keeps it bounded); only reward and
    # value at t=0 penalise collapse. That is thin supervision for a
    # from-scratch ~21M-param conv on a sparse task.
    #
    # pi_grad_t0=1 routes the t=0 policy-BC term through the GRAD latent instead
    # of the detached cache, injecting the strong BC signal into the encoder. The
    # backward through those B frames is already paid for by the consistency
    # loss, so this costs ~0 extra time and memory. Steps t>=1 stay detached.
    # (The heavier alternative is detach_pi=0, which back-props BC through all H
    # steps -- several GB and a large step-time hit.)
    pi_grad_t0: int = 0

    # ★ AUXILIARY PRIVILEGED-STATE PREDICTION (docs/simdist.md §43).
    # The pixel arm's stage-2 BC is capped because the SimNorm latent is trained
    # by the VALUE objective: it keeps value structure and DISCARDS the mm-level
    # geometry a policy needs (findings.md §30). The state arm falls back to raw
    # 37-d state; pixels have no fallback. Measured: pixel policy_only 0.07-0.10
    # vs state 0.360, and NOT underfitting (BC loss went 3x BELOW state's with no
    # score change). Unfreezing the encoder in stage 2 recovers +0.14 but breaks
    # the frozen value head MPPI ranks with, so the fix has to be here in stage 1
    # where latent+value+reward train together.
    # This adds coef * MSE(aux_head(z_t0), z-scored privileged state) at t=0 ONLY,
    # i.e. through the GRAD latent, so it actually shapes the encoder (t>=1 latents
    # are detached and would train only the head). Source stage-1 runs in the
    # source sim where the privileged state is available for free -- the replay
    # already stores it beside the frames for the value teacher.
    # ★ AUGMENTATION (pixel arm; ALL default 0 = off, so every state arm and every
    # existing pixel ckpt is byte-for-byte unchanged). The reference recipe uses
    # four kinds; the `augment_fn` hook has existed in pixel_obs_encoder.forward
    # since the port but NO caller ever passed one, so we have trained with zero
    # augmentation on a replay we do 461 epochs over (docs/simdist.md §45).
    #
    # Applied ONCE to the whole [B,H+1,C,H,W] block in update_distill, with ONE
    # shift per (sample, camera) SHARED across all H+1 frames. Independent
    # per-frame shifts would inject apparent camera jitter into the transition the
    # dynamics head is learning, and would also desynchronise the network input
    # from the detached consistency target built from the same frame.
    # ★ HISTORY (rl/history.py). 0 = off, byte-for-byte legacy. >0 feeds the
    # encoder [prop_t..prop_{t-K+1}, a_{t-1}..a_{t-K}] alongside the most recent
    # image, which is the reference recipe's input contract (§44: the paper uses a
    # history encoder; our port had obs_stack/state_stack/prop_stack all = 1).
    # Requires the env to be built with prop_stack=K AND cond_action=K so the ACT
    # path can supply the same vector; the agent asserts that.
    # NOTE this does NOT touch MPPI: the encoder only ever runs on the real current
    # obs, and the planner rolls in latent space, so history changes encode() only.
    history_length: int = 0
    aug_shift_pad: int = 0        # RandomShiftsAug pad; repo's pixel BC/RL use 4
    aug_prop_noise: float = 0.0   # Gaussian sigma on the proprio vector
    # ★ §44's remaining two augmentation rows (added 2026-09-11 for §48).
    # Both default to 0 = OFF, so every state arm and every existing pixel ckpt
    # is byte-for-byte unchanged (verify with scripts/slurm/dedup_regress.slrm).
    # They are CONFIG fields, not modules, so unlike iron law #23's head/buffer
    # case they exist on all three `__init__` branches by construction -- but
    # read them with getattr() anyway, since old ckpt cfgs lack the attribute.
    #   aug_color_jitter: single strength s. brightness/contrast/saturation
    #     factors ~ U(1-s, 1+s); hue shift ~ U(-h, h) turns with h = min(0.5, s/3),
    #     so s=0.3 reproduces the conventional ColorJitter(0.3, 0.3, 0.3, 0.1).
    #   aug_blur_sigma_max: sigma ~ U(0, max] px of a separable Gaussian, in
    #     STORED-frame units (§46⑫: aug runs before PixelObsEncoder's resize_to).
    aug_color_jitter: float = 0.0
    aug_blur_sigma_max: float = 0.0
    aux_state_coef: float = 0.0
    aux_state_dim: int = 0        # set by the trainer from the replay's state dim
    horizon: int = 5            # H: rollout length for training + planning
    activation: str = "relu"    # "relu" (legacy) or "mish" (TD-MPC2)
    # loss weights
    consistency_coef: float = 20.0
    reward_coef: float = 1.0
    # Up-weight rare success steps (reward=1) in the SCALAR reward MSE (ignored
    # when discrete_heads=1, which is scale-robust by construction). 1.0 = legacy.
    reward_pos_weight: float = 1.0
    value_coef: float = 1.0
    policy_coef: float = 10.0
    decoder_coef: float = 1.0   # state-reconstruction (z -> state) for the MPPI proposal
    rho: float = 0.5            # per-step discount on the multi-step rollout loss
    # discrete (two-hot) reward + value heads (TD-MPC2). Default OFF (scalar MSE).
    discrete_heads: int = 0
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0
    # VALUE ENSEMBLE (MRS.Q-style search pessimism; CLAUDE.md §27/§29).
    # num_value_heads=1 (default) => the single legacy head; state_dict + math are
    # byte-for-byte unchanged and old ckpts load strictly. >1 adds N-1 EXTRA heads
    # (`value_extra`), each independently initialized and trained to the SAME
    # distillation target, so `value_reduce="min"` gives the pessimistic
    # min-over-ensemble terminal value the planner ranks candidates with.
    # MRS.Q (Chang et al. 2026) reports 2 heads insufficient, ~10 the sweet spot.
    num_value_heads: int = 1
    # How predict_value reduces across heads: "mean" (legacy/optimistic) or "min"
    # (pessimism). Identity when num_value_heads==1.
    value_reduce: str = "mean"
    # Per-head Bernoulli keep-prob on the value loss (bagging => ensemble diversity;
    # with a DETERMINISTIC distillation teacher and identical data, independent init
    # alone leaves the heads nearly identical => min == mean - const, which argmax is
    # INVARIANT to (§27's argmax-immune eps_bar) => zero planning effect). 1.0 = all
    # heads see all data (init diversity only).
    # MEASURED 2026-07-16 (§29): bagging=0.5 did NOT prevent collapse — per-batch
    # Bernoulli masking is not true bagging (no fixed per-sample subsets), so all
    # heads still see the same data distribution and converge to the same function
    # (disagree 0.0059 = 3.7% of the value's own spread => min == mean - 0.009 const
    # => exact no-op for argmax). Use value_prior_scale instead.
    value_bagging: float = 1.0
    # RANDOM PRIOR FUNCTIONS (Osband et al.): V_i(z) = f_i(z) + beta * p_i(z), with
    # p_i a FIXED, randomly-initialized, NEVER-trained net (one per head; kept in the
    # state_dict so eval reproduces). THE fix for the §29 collapse: in-distribution
    # f_i learns to cancel beta*p_i (so V_i still fits the teacher), but that
    # cancellation does NOT generalize off-manifold, so the members diverge exactly
    # where pessimism needs them to. The only clean diversity source under a
    # deterministic teacher + identical data. 0.0 = OFF (default, back-compat).
    value_prior_scale: float = 0.0
    # DYNAMICS ENSEMBLE (MOPO/PETS-style STATE-SPACE pessimism; CLAUDE.md §27).
    # The value ensemble (above / §29) is STRUCTURALLY BLIND to the binding error:
    # §27 (2026-07-16) measured Spearman(value-disagree, geometric error
    # ||ẑ_H − s_H||) = −0.30 — value heads live on the value manifold and cannot
    # see a dynamics error that stays ON that manifold ("gap 在 transition 不在
    # d(s)"). §27's verdict: the ONLY remaining pessimism direction is to penalize
    # the DYNAMICS ensemble's GEOMETRIC disagreement (spread of the members'
    # next-latent f^H), NOT value uncertainty. This is that arm.
    # num_dynamics_heads=1 (default) => the single legacy head; state_dict + math +
    # `next()` are byte-for-byte unchanged and old ckpts load strictly. >1 adds N-1
    # EXTRA dynamics heads (`dynamics_extra`), each independently initialized; the
    # planner rolls the ENSEMBLE MEAN and subtracts a penalty proportional to the
    # per-step cross-head std of the predicted next latent (see
    # PlannerConfig.dynamics_disagree_coef) so argmax dares NOT push into regions
    # the ensemble is unconstrained on. In-distribution the members agree (data
    # pins them); off-distribution (where MPPI's argmax wants to go) they diverge.
    num_dynamics_heads: int = 1
    # Per-head Bernoulli keep-prob on the consistency loss (bagging => diversity).
    # 1.0 = all heads see all data (init diversity only). Unlike the VALUE ensemble
    # (§29, which collapsed because value is blind to geometry), the dynamics heads
    # predict f^H directly, so init + bagging DO produce geometric disagreement
    # off-manifold. dynamics_prior_scale is the extra hedge if they still collapse.
    dynamics_bagging: float = 1.0
    # Random prior functions for the dynamics heads (same idea as value_prior_scale):
    # next_i = norm(MLP_i([z,a]) + beta * p_i([z,a])), p_i FIXED/untrained. Clean
    # off-manifold diversity source under a deterministic consistency target.
    # 0.0 = OFF (default, back-compat: no dynamics_prior params).
    dynamics_prior_scale: float = 0.0
    # shared encoder for the policy (no enc_pi). Default OFF (policy has its own
    # enc_pi). When ON, detach_pi controls whether the policy-BC gradient reaches
    # the shared encoder.
    shared_encoder: int = 0
    detach_pi: int = 1
    # optimization
    lr: float = 3e-4
    enc_lr_scale: float = 1.0   # encoder lr multiplier (TD-MPC2: 0.3); used in agent
    tau: float = 0.01           # EMA rate for the value target
    grad_clip: float = 20.0


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: WorldModelConfig,
                 *, image_shape=None, prop_dim: int = 0, cameras=()):
        """image_shape/prop_dim/cameras are PIXEL-arm only and keyword-only, so the
        six existing `WorldModel(obs_dim, action_dim, cfg)` call sites (train_dynamics,
        train_joint_dynamics, eval_flow_ug, ...) are untouched."""
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.raw_state = (cfg.encoder_type == "identity")
        self.pixel = (cfg.encoder_type == "pixel")
        h = cfg.mlp_hidden_dim
        nl = cfg.num_layer
        ln = cfg.layer_norm
        act = cfg.activation
        out_r = cfg.num_bins if cfg.discrete_heads else 1
        out_v = cfg.num_bins if cfg.discrete_heads else 1

        # ★ Declared before the branches: the raw-state (`identity`) arm returns
        # from its own branch and never reaches _build_aux_state at the end of the
        # latent path, and train_simdist.py reads `model.aux_state`
        # unconditionally. Assigning None registers no parameter and no buffer, so
        # state_dict is unchanged and strict loads of existing ckpts still pass;
        # the pixel and latent paths overwrite this via _build_aux_state.
        self.aux_state = None

        if self.raw_state:
            # RAW-STATE arm: z := z-scored raw state (dim = obs_dim). No learned
            # encoder, no SimNorm, no decoder net, no enc_pi. dynamics is a RESIDUAL
            # next-(normalized-)state predictor (z' = z + MLP([z,a])); decode is the
            # analytic denorm inverse. z-score stats live in buffers (identity until
            # set_normalization), so the ckpt self-normalizes at eval.
            z = obs_dim
            self.simnorm = None
            self.encoder = None
            self.decoder = None
            self.enc_pi = None
            self.obs_encoder = None
            self.register_buffer("norm_state_mean", torch.zeros(obs_dim))
            self.register_buffer("norm_state_std", torch.ones(obs_dim))
            self.dynamics = mlp(z + action_dim, h, z, nl, ln, act_out="none", act=act)
            self._build_dynamics_extra(z, h, nl, ln, act)
            self.reward = mlp(z + action_dim, h, out_r, nl, ln, act_out="none", act=act)
            self.value = mlp(z, h, out_v, nl, ln, act_out="none", act=act)
            self.policy = mlp(z, h, action_dim, nl, ln, act_out="tanh", act=act)
            self.value_target = mlp(z, h, out_v, nl, ln, act_out="none", act=act)
            self.value_target.load_state_dict(self.value.state_dict())
            for p in self.value_target.parameters():
                p.requires_grad_(False)
            self._build_value_extra(z, h, out_v, nl, ln, act)
            return

        z = cfg.latent_dim
        assert z % cfg.simnorm_dim == 0, "latent_dim must be divisible by simnorm_dim"
        self.simnorm = SimNorm(cfg.simnorm_dim)

        # ---- PIXEL arm: obs dict -> conv features -> the SAME trunk MLP -------
        if self.pixel:
            from rl.pixel_obs_encoder import PixelObsEncoder
            assert image_shape is not None and len(cameras) > 0, (
                "encoder_type='pixel' needs image_shape=(C,H,W) and cameras=[...]")
            # shared_encoder: with a dedicated enc_pi the planner needs decode(z)
            # to bridge back to an observation (simdist_planner.py:182), which is
            # impossible for images. decoder_coef: there is no image decoder, and
            # with shared_encoder decode() is unreachable at plan time.
            assert cfg.shared_encoder, "pixel arm requires shared_encoder=1"
            assert cfg.decoder_coef == 0, "pixel arm requires decoder_coef=0"
            assert cfg.input_fourier_l == 0, "input_fourier is a raw-state feature"
            self.obs_encoder = PixelObsEncoder(image_shape, prop_dim, cameras, cfg.pixel)
            self.input_pe = None
            self.encoder = mlp(self.obs_encoder.out_dim, h, z, nl, ln,
                               act_out="none", act=act)
            self.dynamics = mlp(z + action_dim, h, z, nl, ln, act_out="none", act=act)
            self._build_dynamics_extra(z, h, nl, ln, act)
            self.reward = mlp(z + action_dim, h, out_r, nl, ln, act_out="none", act=act)
            self.value = mlp(z, h, out_v, nl, ln, act_out="none", act=act)
            self.decoder = None          # no image decoder; decode() raises
            self.enc_pi = None           # shared encoder (asserted above)
            self.policy = mlp(z, h, action_dim, nl, ln, act_out="tanh", act=act)
            self.value_target = mlp(z, h, out_v, nl, ln, act_out="none", act=act)
            self.value_target.load_state_dict(self.value.state_dict())
            for p in self.value_target.parameters():
                p.requires_grad_(False)
            self._build_value_extra(z, h, out_v, nl, ln, act)
            # The pixel branch returns HERE, so it used to skip _build_aux_state
            # entirely and `self.aux_state` never existed -- train_simdist.py's
            # unconditional read then raised AttributeError on startup for every
            # pixel run. With aux_state_coef=0 this only sets the attribute to
            # None, so no state_dict key is added and strict loads are unaffected.
            self._build_aux_state(z, h, nl, ln, act)
            return

        self.obs_encoder = None
        # optional Fourier (PE) encoding of the raw state before the encoder MLP.
        if cfg.input_fourier_l > 0:
            from rl.fourier import FourierFeatures
            self.input_pe = FourierFeatures(
                obs_dim, cfg.input_fourier_l,
                include_raw=bool(cfg.input_fourier_include_raw),
                normalize=bool(cfg.input_fourier_normalize),
            )
            enc_in = self.input_pe.out_dim
        else:
            self.input_pe = None
            enc_in = obs_dim
        # DYNAMICS branch: encoder shared by dynamics/reward/value (+ decoder).
        self.encoder = mlp(enc_in, h, z, nl, ln, act_out="none", act=act)
        self.dynamics = mlp(z + action_dim, h, z, nl, ln, act_out="none", act=act)
        self._build_dynamics_extra(z, h, nl, ln, act)
        self.reward = mlp(z + action_dim, h, out_r, nl, ln, act_out="none", act=act)
        self.value = mlp(z, h, out_v, nl, ln, act_out="none", act=act)
        # decoder z -> state (lets the MPPI proposal convert a rolled latent back
        # to a state to query an EXTERNAL state-policy; not needed for the shared
        # policy which queries the latent directly).
        self.decoder = mlp(z, h, obs_dim, nl, ln, act_out="none", act=act)

        # POLICY branch.
        if cfg.shared_encoder:
            # reuse the dynamics encoder; policy queries the latent directly so the
            # planner can roll it without decode. enc_pi unused.
            self.enc_pi = None
        else:
            # dedicated encoder (legacy §17 stability fix): decoupled from dynamics.
            self.enc_pi = mlp(enc_in, h, z, nl, ln, act_out="none", act=act)
        self.policy = mlp(z, h, action_dim, nl, ln, act_out="tanh", act=act)

        # EMA target of the value head (TD bootstrap target)
        self.value_target = mlp(z, h, out_v, nl, ln, act_out="none", act=act)
        self.value_target.load_state_dict(self.value.state_dict())
        for p in self.value_target.parameters():
            p.requires_grad_(False)
        self._build_value_extra(z, h, out_v, nl, ln, act)
        self._build_aux_state(z, h, nl, ln, act)

    # -------------------------------------------- auxiliary privileged state
    def _build_aux_state(self, z, h, nl, ln, act):
        """Head predicting the z-scored PRIVILEGED state from the latent (§43).
        Built only when aux_state_coef > 0, so every existing checkpoint stays
        load_state_dict-compatible (strict load would otherwise reject them)."""
        self.aux_state = None
        c = getattr(self.cfg, "aux_state_coef", 0.0)
        d = int(getattr(self.cfg, "aux_state_dim", 0))
        if c > 0:
            assert d > 0, "aux_state_coef>0 requires aux_state_dim (the trainer sets it)"
            self.aux_state = mlp(z, h, d, nl, ln, act_out="none", act=act)
            # z-scoring stats for the target, filled by set_aux_state_stats().
            # Registered (not plain tensors) so they travel with the checkpoint --
            # a run resumed without them would silently regress to raw-scale MSE,
            # where the large-magnitude state dims dominate the 37-d target.
            self.register_buffer("aux_state_mean", torch.zeros(d))
            self.register_buffer("aux_state_std", torch.ones(d))

    def set_aux_state_stats(self, mean, std):
        """Per-dim mean/std of the privileged state, from the source replay."""
        if self.aux_state is None:
            return
        with torch.no_grad():
            self.aux_state_mean.copy_(mean.to(self.aux_state_mean.device))
            self.aux_state_std.copy_(std.clamp_min(1e-6).to(self.aux_state_std.device))

    def aux_state_loss(self, z, priv_state):
        """MSE on the z-scored privileged state. `z` must be the GRAD latent (t=0),
        otherwise this trains the head without shaping the encoder -- which is the
        entire point (§43 ②: the geometry is in the images but not in the latent)."""
        tgt = (priv_state - self.aux_state_mean) / self.aux_state_std
        return F.mse_loss(self.aux_state(z), tgt)

    # ------------------------------------------------------- value ensemble
    def _build_value_extra(self, z, h, out_v, nl, ln, act):
        """The N-1 EXTRA value heads (+ their EMA targets) and the optional random
        prior nets. Empty ModuleLists when num_value_heads==1 / value_prior_scale==0
        => NO parameters => state_dict identical to the legacy single-head model, so
        old ckpts load strictly and new N=1 runs are byte-for-byte unchanged."""
        n_extra = max(int(getattr(self.cfg, "num_value_heads", 1)) - 1, 0)
        self.value_extra = nn.ModuleList(
            [mlp(z, h, out_v, nl, ln, act_out="none", act=act) for _ in range(n_extra)])
        self.value_target_extra = nn.ModuleList(
            [mlp(z, h, out_v, nl, ln, act_out="none", act=act) for _ in range(n_extra)])
        for tgt, src in zip(self.value_target_extra, self.value_extra):
            tgt.load_state_dict(src.state_dict())
            for p in tgt.parameters():
                p.requires_grad_(False)
        # random prior functions: one FIXED untrained net per head (never optimized —
        # excluded from trainable_parameters — but IS in the state_dict so eval
        # reproduces the exact ensemble).
        beta = float(getattr(self.cfg, "value_prior_scale", 0.0))
        self.value_prior = nn.ModuleList(
            [mlp(z, h, out_v, nl, ln, act_out="none", act=act)
             for _ in range(n_extra + 1)] if beta > 0 else [])
        for p in self.value_prior.parameters():
            p.requires_grad_(False)

    def value_heads(self) -> list:
        return [self.value] + list(self.value_extra)

    def value_target_heads(self) -> list:
        return [self.value_target] + list(self.value_target_extra)

    def num_value_heads(self) -> int:
        return 1 + len(self.value_extra)

    def value_logits_all(self, z: torch.Tensor, target: bool = False) -> torch.Tensor:
        """-> [N, B, out] logits from every head (N=1 => the legacy head alone).
        With random priors: logits_i = f_i(z) + beta * p_i(z), the prior treated as a
        constant (no_grad) so gradients still reach f_i, which learns to cancel it
        in-distribution but not off it."""
        heads = self.value_target_heads() if target else self.value_heads()
        out = torch.stack([hd(z) for hd in heads], dim=0)
        if len(self.value_prior) > 0:
            with torch.no_grad():
                prior = torch.stack([p(z) for p in self.value_prior], dim=0)
            out = out + float(self.cfg.value_prior_scale) * prior
        return out

    def predict_value_all(self, z: torch.Tensor, target: bool = False) -> torch.Tensor:
        """-> [N, B] scalar value per head (discrete heads two_hot_inv'd)."""
        out = self.value_logits_all(z, target=target)          # [N,B,out]
        if self.cfg.discrete_heads:
            N, B, _ = out.shape
            return two_hot_inv(out.reshape(N * B, -1), self.cfg.num_bins,
                               self.cfg.vmin, self.cfg.vmax).view(N, B)
        return out.squeeze(-1)

    # ---------------------------------------------------- dynamics ensemble
    def _build_dynamics_extra(self, z, h, nl, ln, act):
        """The N-1 EXTRA dynamics heads and the optional random prior nets. Empty
        ModuleLists when num_dynamics_heads==1 / dynamics_prior_scale==0 => NO
        parameters => state_dict identical to the legacy single-head model, so old
        ckpts load strictly and new N=1 runs are byte-for-byte unchanged. Each head
        maps [z,a] -> z (same shape/arch as self.dynamics), independently init'd."""
        n_extra = max(int(getattr(self.cfg, "num_dynamics_heads", 1)) - 1, 0)
        self.dynamics_extra = nn.ModuleList(
            [mlp(z + self.action_dim, h, z, nl, ln, act_out="none", act=act)
             for _ in range(n_extra)])
        beta = float(getattr(self.cfg, "dynamics_prior_scale", 0.0))
        self.dynamics_prior = nn.ModuleList(
            [mlp(z + self.action_dim, h, z, nl, ln, act_out="none", act=act)
             for _ in range(n_extra + 1)] if beta > 0 else [])
        for p in self.dynamics_prior.parameters():
            p.requires_grad_(False)

    def dynamics_heads(self) -> list:
        return [self.dynamics] + list(self.dynamics_extra)

    def num_dynamics_heads(self) -> int:
        return 1 + len(self.dynamics_extra)

    def next_all(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Every head's predicted next latent -> [N, B, z]. Raw-state heads are
        residual (z + delta_i); latent heads are SimNorm(raw_i). With random priors:
        raw_i = MLP_i([z,a]) + beta * p_i([z,a]) (prior as a no_grad constant)."""
        x = torch.cat([z, a], dim=-1)
        raw = torch.stack([hd(x) for hd in self.dynamics_heads()], dim=0)  # [N,B,z]
        if len(self.dynamics_prior) > 0:
            with torch.no_grad():
                pri = torch.stack([p(x) for p in self.dynamics_prior], dim=0)
            raw = raw + float(self.cfg.dynamics_prior_scale) * pri
        if self.raw_state:
            return z.unsqueeze(0) + raw          # residual, broadcast over heads
        return self.simnorm(raw)                 # SimNorm each head (any leading dims)

    def dynamics_disagreement(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """GEOMETRIC disagreement [B]: cross-head std of the predicted next latent,
        averaged over latent dims. The MOPO/PETS state-space uncertainty the planner
        penalizes. 0 for a single head (nothing to disagree on)."""
        if self.num_dynamics_heads() == 1:
            return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return self.next_all(z, a).std(dim=0).mean(dim=-1)  # [B]

    def dynamics_parameters(self) -> list:
        """All trainable dynamics params (every ensemble head; priors excluded)."""
        p = list(self.dynamics.parameters())
        for hd in self.dynamics_extra:
            p += list(hd.parameters())
        return p

    # ----------------------------------------------------------- normalization
    def set_normalization(self, mean, std):
        """Set the raw-state z-score stats (raw-state arm only; no-op otherwise)."""
        if not self.raw_state:
            return
        with torch.no_grad():
            m = torch.as_tensor(mean, dtype=torch.float32, device=self.norm_state_mean.device)
            s = torch.as_tensor(std, dtype=torch.float32, device=self.norm_state_std.device)
            self.norm_state_mean.copy_(m)
            self.norm_state_std.copy_(s.clamp_min(1e-6))

    def set_input_fourier_stats(self, mean, std):
        """z-score stats for the ENCODER-input Fourier (PE) transform (latent arm
        with input_fourier_l>0). No-op otherwise."""
        pe = getattr(self, "input_pe", None)
        if pe is not None:
            pe.set_stats(mean, std)

    # ----------------------------------------------------------- forward bits
    def encode(self, state, augment_fn=None) -> torch.Tensor:
        """state: [B, obs_dim] tensor (state arms) OR an obs DICT
        {camera: uint8[B,C,H,W], 'prop': float[B,prop_dim]} (pixel arm)."""
        if self.pixel:
            return self.simnorm(self.encoder(self.obs_encoder(state, augment_fn=augment_fn)))
        if self.raw_state:
            return (state - self.norm_state_mean) / self.norm_state_std
        if getattr(self, "input_pe", None) is not None:
            state = self.input_pe(state)
        return self.simnorm(self.encoder(state))

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Latent dynamics z' (the point estimate the rollout advances on). Latent:
        SimNorm(MLP([z,a])) (direct). Raw-state: residual z + MLP([z,a]). With a
        dynamics ENSEMBLE (num_dynamics_heads>1) this is the ensemble MEAN; for a
        single head it is byte-for-byte the legacy expression."""
        if self.num_dynamics_heads() == 1 and len(self.dynamics_prior) == 0:
            d = self.dynamics(torch.cat([z, a], dim=-1))
            return z + d if self.raw_state else self.simnorm(d)
        return self.next_all(z, a).mean(dim=0)

    # raw logits (for the discrete cross-entropy loss)
    def reward_logits(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.reward(torch.cat([z, a], dim=-1))

    def value_logits(self, z: torch.Tensor, target: bool = False) -> torch.Tensor:
        return (self.value_target if target else self.value)(z)

    # scalar predictions (used by the planner + diagnostics; discrete -> two_hot_inv)
    def predict_reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        out = self.reward_logits(z, a)
        if self.cfg.discrete_heads:
            return two_hot_inv(out, self.cfg.num_bins, self.cfg.vmin, self.cfg.vmax)
        return out.squeeze(-1)

    def attach_value_residual(self, hidden: int = 256, scale: float = 1.0):
        """Add the "marginal value" head: V(z) = V_base(z).detach() + scale*mlp(z).

        Ported from scripts/square_value_iter.py --arm residual. The last layer is
        zero-initialised, so at attach time predict_value is BIT-IDENTICAL to the
        distilled value and every later change is an explicit departure rather
        than a free refit.

        Why it sits after the decode rather than inside the head: this model's
        value head is two-hot categorical (discrete_heads=1, 101 bins), so a
        scalar residual has no meaning in logit space. The residual is therefore
        applied to the DECODED scalar -- which is exactly the quantity MPPI ranks
        with (gamma^H * V(z_H)) -- and is trained by MSE on Monte-Carlo returns,
        not by the two-hot CE the non-residual value arms use. That loss
        difference is inherent to the parameterisation, not an oversight.

        Idempotent: calling twice keeps the first residual.
        """
        if getattr(self, "value_residual", None) is not None:
            return
        cfg = self.cfg
        act = getattr(cfg, "activation", "elu")
        ln = cfg.layer_norm
        # The value head's own input width -- correct for both the latent and the
        # raw-state model without re-deriving either.
        zdim = next(m for m in self.value.modules()
                    if isinstance(m, nn.Linear)).in_features
        net = mlp(zdim, hidden, 1, 2, ln, act_out="none", act=act)
        with torch.no_grad():
            last = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
            last.weight.zero_()
            if last.bias is not None:
                last.bias.zero_()
        self.value_residual = net
        self.value_residual_scale = float(scale)

    def predict_value(self, z: torch.Tensor, target: bool = False,
                      reduce: str | None = None) -> torch.Tensor:
        """Scalar terminal value [B]. With a value ensemble, `reduce` (default
        cfg.value_reduce) picks "min" (MRS.Q pessimism) or "mean". Single-head
        (default) short-circuits to the legacy path — bit-identical."""
        if self.num_value_heads() == 1 and len(self.value_prior) == 0:
            out = self.value_logits(z, target=target)
            if self.cfg.discrete_heads:
                base = two_hot_inv(out, self.cfg.num_bins, self.cfg.vmin, self.cfg.vmax)
            else:
                base = out.squeeze(-1)
            return self._add_value_residual(base, z)
        v = self.predict_value_all(z, target=target)           # [N, B]
        how = reduce or getattr(self.cfg, "value_reduce", "mean")
        if how == "min":
            base = v.min(0).values
        elif how == "mean":
            base = v.mean(0)
        else:
            raise ValueError(f"unknown value_reduce: {how}")
        return self._add_value_residual(base, z)

    def _add_value_residual(self, base: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """No-op unless attach_value_residual() was called. The BASE is detached:
        the residual is the only thing that learns, which is what makes the
        departure from the distilled value explicit and measurable."""
        net = getattr(self, "value_residual", None)
        if net is None:
            return base
        return base.detach() + self.value_residual_scale * net(z).squeeze(-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the state from a dynamics latent (for the MPPI proposal).
        Raw-state: the analytic denorm inverse of encode (exact, no params)."""
        if self.raw_state:
            return z * self.norm_state_std + self.norm_state_mean
        if self.pixel:
            # Unreachable at plan time with shared_encoder=1: _policy_rollout takes
            # the policy_from_latent branch (simdist_planner.py:176-179), and the
            # only other decode() sites are behind external reward_fn/value_fn
            # hooks. Raise loudly rather than inventing an image decoder.
            raise RuntimeError(
                "decode() is not available on the pixel arm (no image decoder). "
                "This means something requested a raw observation from a latent: "
                "check that shared_encoder=1 and that no external "
                "proposal_policy/reward_fn/value_fn is attached to the planner.")
        return self.decoder(z)

    def policy_action(self, state: torch.Tensor) -> torch.Tensor:
        """State-space policy: state -> action in [-1,1]. Raw-state: policy on the
        normalized state. Latent: shared encoder reuses the dynamics encoder,
        otherwise the dedicated enc_pi."""
        if self.raw_state:
            return self.policy(self.encode(state))
        if self.cfg.shared_encoder:
            z = self.encode(state)
        else:
            if getattr(self, "input_pe", None) is not None:
                state = self.input_pe(state)
            z = self.enc_pi(state)
        return self.policy(z)

    def policy_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Query the policy DIRECTLY on a dynamics latent (shared-encoder only),
        so the planner can roll the policy without a decode bridge."""
        assert self.cfg.shared_encoder, "policy_from_latent requires shared_encoder"
        return self.policy(z)

    # ------------------------------------------------------- param groups
    def encoder_parameters(self):
        """All params that map an observation to z. MUST include the conv stack on
        the pixel arm, or enc_lr_scale would silently leave it in the full-lr group
        (and any freeze-by-optimizer-exclusion would silently keep training it)."""
        p = [] if self.encoder is None else list(self.encoder.parameters())
        if getattr(self, "obs_encoder", None) is not None:
            p += list(self.obs_encoder.parameters())
        return p

    def policy_parameters(self):
        p = list(self.policy.parameters())
        if not self.cfg.shared_encoder and self.enc_pi is not None:
            p += list(self.enc_pi.parameters())
        return p

    def dynamics_policy_parameters(self):
        """Params trained in stage-2 finetune (dynamics head(s) + policy branch)."""
        return self.dynamics_parameters() + self.policy_parameters()

    def trainable_parameters(self):
        """Params trained in stage-1 distill (everything except the EMA targets and
        the FIXED random priors)."""
        skip = ("value_target.", "value_target_extra.", "value_prior.",
                "dynamics_prior.")
        return [p for n, p in self.named_parameters() if not n.startswith(skip)]
