"""SimDist agent: a latent world model + an MPPI planner (state-based baseline).

Wraps `rl/world_model.py::WorldModel` and `rl/simdist_planner.py::MppiPlanner`,
exposing the interface the rest of ibrl needs:
  * act(obs, eval_mode, stddev, cpu)        -> action via MPPI planning (eval +
                                               on-policy data collection)
  * update_distill(states, actions, rewards) -> stage-1 (all heads)
  * update_finetune(states, actions, rewards)-> stage-2 (dynamics + policy only)
  * state_dict / load_state_dict             -> world model only (planner has no
                                               params); standard nn.Module ckpt.

NOT a QAgent subclass: SimDist is model-based and structurally unlike the
model-free actor-critic family, but the public surface mirrors QAgent so the
eval framework (evaluate/eval.py, multi_process_eval.py) works unchanged.

Losses (multi-step latent rollout from the segment's first state, per-step
weighted by rho**t, TD-MPC2 style):
  consistency: || z_next_pred - sg(encode(s_{t+1})) ||^2
  reward:      || r_pred - r_t ||^2
  value:       dense || V(z_next_pred) - V_src(s_{t+1}) ||^2  (paper-aligned: the
               value head is evaluated on the PREDICTED next latent v_hat_{t+1});
               TD fallback || V(z_t) - sg(r_t + gamma*(1-done)*V_target(enc(s_{t+1}))) ||^2
  policy (BC): || pi(sg(z_t)) - a_t ||^2     (distill the data-collection action)
Stage-2 finetune keeps ONLY consistency (dynamics) + policy-BC, optimizing the
dynamics + policy heads (encoder/reward/value frozen by exclusion from the optim).
"""

from dataclasses import dataclass, field

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils import ibrl_utils as utils
from rl.world_model import WorldModel, WorldModelConfig, soft_ce
from rl.simdist_planner import MppiPlanner, PlannerConfig


@dataclass
class SimDistConfig:
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    # stage-2 finetune
    finetune_steps: int = 30000
    finetune_dynamics: int = 1
    finetune_policy: int = 1
    finetune_lr: float = 3e-4
    # execution: 1 = MPPI planning (SimDist); 0 = policy head directly (ablation /
    # diagnostic — the distilled proposal policy without planning).
    use_planner: int = 1


class SimDistAgent(nn.Module):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera,
                 cfg: SimDistConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.use_state = bool(use_state)
        self.action_dim = action_dim
        self.discount = cfg.planner.discount
        # PIXEL arm: obs_shape is (C,H,W) and the world model consumes an obs DICT.
        # `rl_camera` is the "a+b" string MainConfig builds; mirror its split.
        self.pixel = (cfg.world_model.encoder_type == "pixel")
        self.history_k = 0
        self.rl_cameras = [c for c in str(rl_camera).split("+") if c] if self.pixel else []

        if self.pixel:
            assert not use_state, "pixel arm expects use_state=0 (obs are images)"
            assert len(obs_shape) == 3, f"pixel arm needs (C,H,W), got {tuple(obs_shape)}"
            assert self.rl_cameras, "pixel arm needs rl_camera like 'agentview+robot0_eye_in_hand'"
            prop_dim = int(prop_shape[0]) if prop_shape is not None else 0
            self.history_k = int(getattr(cfg.world_model, "history_length", 0) or 0)
            if self.history_k > 0:
                from env.robosuite_wrapper import PROP_DIM as _PD
                from rl.history import history_dim
                assert prop_dim == _PD * self.history_k, (
                    f"history_length={self.history_k} needs the env built with "
                    f"prop_stack={self.history_k} (prop_shape would be "
                    f"{_PD * self.history_k}), but got {prop_dim}. Set prop_stack "
                    f"AND cond_action to {self.history_k} in the config.")
                prop_dim = history_dim(self.history_k, _PD, action_dim)
                print(f"[history] K={self.history_k} -> encoder prop_dim={prop_dim} "
                      f"({self.history_k}x{_PD} proprio + {self.history_k}x{action_dim} actions)")
            self.obs_dim = 0
            self.model = WorldModel(0, action_dim, cfg.world_model,
                                    image_shape=tuple(obs_shape), prop_dim=prop_dim,
                                    cameras=self.rl_cameras)
        else:
            assert use_state, "SimDistAgent state arm requires use_state=1"
            obs_dim = int(obs_shape[0])
            self.obs_dim = obs_dim
            self.model = WorldModel(obs_dim, action_dim, cfg.world_model)
        self.to(device)
        # planner holds a ref to the (now on-device) model; no params of its own
        self.planner = MppiPlanner(self.model, cfg.planner, action_dim, device=device)

        wm = cfg.world_model
        if wm.enc_lr_scale != 1.0 and not self.model.raw_state:
            # TD-MPC2: train the (shared) encoder at a smaller lr for stability.
            # (raw-state arm has no learned encoder -> single param group.)
            enc_params = self.model.encoder_parameters()
            enc_ids = {id(p) for p in enc_params}
            other = [p for p in self.model.trainable_parameters() if id(p) not in enc_ids]
            self.distill_opt = torch.optim.Adam([
                {"params": enc_params, "lr": wm.lr * wm.enc_lr_scale},
                {"params": other, "lr": wm.lr},
            ])
        else:
            self.distill_opt = torch.optim.Adam(
                self.model.trainable_parameters(), lr=wm.lr
            )
        ft_params = []
        if cfg.finetune_dynamics:
            ft_params += self.model.dynamics_parameters()  # all ensemble heads
        if cfg.finetune_policy:
            ft_params += self.model.policy_parameters()  # enc_pi + policy
        self.finetune_opt = (
            torch.optim.Adam(ft_params, lr=cfg.finetune_lr) if ft_params else None
        )
        # Lazily-built optimizers for the two-phase TARGET adaptation
        # (train_simdist_adapt.py): a policy-ONLY BC phase and a dynamics-ONLY
        # adaptation phase. Each scopes its optimizer to a single head so every
        # other parameter (INCLUDING the encoder) is frozen by exclusion.
        self._bc_opt = None
        self._dyn_opt = None
        self._dynval_opt = None
        self._dynres_opt = None
        self._dynvalenc_opt = None
        self._val_opt = None

    def set_normalization(self, mean, std):
        """Set the raw-state z-score stats (raw-state arm only; no-op otherwise)."""
        self.model.set_normalization(mean, std)

    def set_input_fourier_stats(self, mean, std):
        """z-score stats for the latent-arm encoder-input Fourier (PE) transform
        (no-op unless world_model.input_fourier_l>0)."""
        self.model.set_input_fourier_stats(mean, std)

    def set_planning_sources(self, value_fn=None, reward_fn=None, proposal_policy=None):
        """Plug EXTERNAL value / reward / proposal into the MPPI planner so it
        scores/seeds with them instead of the world model's own (here untrained)
        heads — the 'train only dynamics, borrow value+policy from the source agent
        + heuristic reward' mode (CLAUDE.md §17). Any arg left None keeps the
        planner's current source (model head by default)."""
        if value_fn is not None:
            self.planner.value_fn = value_fn
        if reward_fn is not None:
            self.planner.reward_fn = reward_fn
        if proposal_policy is not None:
            self.planner.proposal_policy = proposal_policy

    # --------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], *, eval_mode: bool = False,
            stddev: float = 0.0, cpu: bool = True) -> torch.Tensor:
        if self.pixel:
            return self._act_pixel(obs, eval_mode=eval_mode, stddev=stddev, cpu=cpu)
        state = obs["state"].to(self.device).float()
        single = state.dim() == 1
        if not self.cfg.use_planner:
            # policy head directly (no planning) — handles single + batched obs
            s2d = state.view(1, -1) if single else state
            action = self.model.policy_action(s2d)
            if not eval_mode and stddev > 0:
                action = (action + torch.randn_like(action) * stddev).clamp(-1, 1)
            action = action[0] if single else action
            return action.cpu() if cpu else action
        if single:
            # single obs (training collection / single-process eval)
            action = self.planner.plan(state, eval_mode=eval_mode, stddev=stddev)
            return action.cpu() if cpu else action
        # batched obs (multi-process eval sends a [B, obs_dim] stack): plan all B
        # in one set of GPU kernels instead of a per-row Python loop (the dominant
        # eval cost). Numerically equivalent to looping `plan` over the rows.
        actions = self.planner.plan_batch(state, eval_mode=eval_mode, stddev=stddev)
        return actions.cpu() if cpu else actions

    def _batch_pixel_obs(self, obs: dict) -> tuple[dict, bool]:
        """-> (batched obs dict on device, was_single). Returns a NEW dict; never
        mutates the caller's, which the collection loop reuses across steps
        (QAgent._maybe_unsqueeze_ mutates in place -- do not copy that here)."""
        single = obs[self.rl_cameras[0]].dim() == 3          # [C,H,W] vs [B,C,H,W]
        keys = list(self.rl_cameras)
        if self.model.obs_encoder.use_prop:
            keys.append("prop")
        out = {}
        for k in keys:
            v = obs[k].to(self.device)      # uint8 stays uint8; cast happens in the encoder
            out[k] = v.unsqueeze(0) if single else v
        if self.history_k > 0 and self.model.obs_encoder.use_prop:
            # ACT-side history. The env supplies obs["prop"] already stacked
            # (prop_stack=K, newest-first) and obs["past_action"] as the wrapper's
            # deque (OLDEST-first, [a_{t-K}..a_{t-1}]). `build_step` reverses the
            # actions so this matches the TRAINING layout exactly; both paths go
            # through rl/history.py precisely so they cannot drift apart.
            from env.robosuite_wrapper import PROP_DIM as _PD
            from rl.history import build_step
            assert "past_action" in obs, (
                "history_length>0 needs the env built with cond_action=K so it "
                "emits obs['past_action']")
            pa = obs["past_action"].to(self.device)
            if single:
                pa = pa.unsqueeze(0)
            out["prop"] = build_step(out["prop"], pa, self.history_k,
                                     _PD, self.action_dim)
        return out, single

    @torch.no_grad()
    def _act_pixel(self, obs, *, eval_mode, stddev, cpu):
        x, single = self._batch_pixel_obs(obs)
        if not self.cfg.use_planner:
            action = self.model.policy_action(x)
            if not eval_mode and stddev > 0:
                action = (action + torch.randn_like(action) * stddev).clamp(-1, 1)
        else:
            # ALWAYS route through plan_batch, even for a single obs. plan() does
            # `state.view(1,-1)` (simdist_planner.py:227), which is meaningless for
            # an image dict; plan_batch encodes then calls plan_batch_from_latent,
            # which is latent-native and needs no raw obs. Its docstring notes it is
            # the vectorized equivalent of looping plan() over the rows. The only
            # feature lost is warm_start, which plan_batch does not support and
            # which every eval/adapt config already sets to 0 (asserted in the
            # pixel config validation).
            action = self.planner.plan_batch(x, eval_mode=eval_mode, stddev=stddev)
        action = action[0] if single else action
        return action.cpu() if cpu else action

    @torch.no_grad()
    def plan_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Base MPPI action(s) starting from a LATENT z [B, z] — no encode/decode.
        Used by stage-1 latent Dyna to get the base action at an IMAGINED next
        latent z'=next(z,a) exactly (MPPI is latent-native; see
        SimdistPlanner.plan_batch_from_latent). Deterministic (eval_mode)."""
        assert self.cfg.use_planner, "plan_from_latent needs a planner (use_planner=1)"
        return self.planner.plan_batch_from_latent(z, eval_mode=True, stddev=0.0)

    @torch.no_grad()
    def plan_seq_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """The WHOLE committed base-action sequence [B, H, act] from a latent z,
        via ONE MPPI solve (MPPI internally rolls H steps in latent, so the full
        plan mean is free). Used by stage-1 K-step latent Dyna: one plan gives the
        base actions for the entire imagined rollout, instead of H serial plans."""
        assert self.cfg.use_planner, "plan_seq_from_latent needs a planner"
        return self.planner.plan_batch_from_latent(z, eval_mode=True, stddev=0.0,
                                                   return_full=True)

    # ----------------------------------------------------------- stage 1
    def _value_head_loss(self, z, v_tgt, disc, nb, vmin, vmax):
        """Distillation loss for the value head(s), averaged over the ensemble.

        Single-head (default) reduces EXACTLY to the legacy expression. With N>1
        every head regresses the SAME target; `value_bagging<1` gives each head an
        independent Bernoulli subset of the batch, which is what makes the members
        disagree — without it a deterministic teacher drives them to the same
        function and min-ensemble degenerates to (mean - const), which argmax is
        invariant to (§27) => the pessimism would be a no-op by construction."""
        model = self.model
        keep = float(getattr(model.cfg, "value_bagging", 1.0))
        losses = []
        if disc:
            logits = model.value_logits_all(z)                 # [N,B,bins]
            for i in range(logits.shape[0]):
                per = soft_ce(logits[i], v_tgt, nb, vmin, vmax)   # [B]
                losses.append(self._bag(per, keep))
        else:
            preds = model.predict_value_all(z)                 # [N,B]
            for i in range(preds.shape[0]):
                per = (preds[i] - v_tgt) ** 2                  # [B]
                losses.append(self._bag(per, keep))
        return torch.stack(losses).mean()

    @staticmethod
    def _bag(per_sample, keep: float):
        """Mean of a per-sample loss under an independent Bernoulli(keep) mask."""
        if keep >= 1.0:
            return per_sample.mean()
        m = (torch.rand_like(per_sample) < keep).float()
        return (per_sample * m).sum() / m.sum().clamp(min=1.0)

    def _dynamics_cons_loss(self, z, a, z_next_enc):
        """Consistency loss for the dynamics head(s) + the rolled next latent.

        Single-head (default) reduces EXACTLY to F.mse_loss(next(z,a), z_next_enc)
        and returns the legacy next latent — byte-for-byte. With N>1 EVERY head
        regresses the (stop-grad) target `z_next_enc` under its own Bernoulli
        bagging mask (=> off-manifold diversity), and the latent the rollout
        advances on is the ENSEMBLE MEAN (what next() / the planner use). Returns
        (loss_scalar, z_next_mean)."""
        model = self.model
        if model.num_dynamics_heads() == 1 and len(model.dynamics_prior) == 0:
            z_next = model.next(z, a)
            return F.mse_loss(z_next, z_next_enc), z_next
        preds = model.next_all(z, a)                       # [N, B, z]
        keep = float(getattr(model.cfg, "dynamics_bagging", 1.0))
        losses = []
        for i in range(preds.shape[0]):
            per = ((preds[i] - z_next_enc) ** 2).mean(-1)  # [B]
            losses.append(self._bag(per, keep))
        return torch.stack(losses).mean(), preds.mean(0)

    @staticmethod
    def _obs_at(obs, t):
        """Slice timestep t out of a batched sequence. Accepts a [B,H+1,...] tensor
        (state arms) or an obs DICT {key: [B,H+1,...]} (pixel arm), so every loss
        body indexes uniformly regardless of arm."""
        if isinstance(obs, dict):
            return {k: v[:, t] for k, v in obs.items()}
        return obs[:, t]

    def _augment_obs(self, obs):
        """Training-only image/proprio augmentation for the PIXEL arm (§44's row).

        ★ THE LOAD-BEARING INVARIANT: ONE augmentation draw per (sample, camera),
        SHARED across the H+1 frames of that sample. Every parameter -- the shift
        grid, the four jitter factors, the blur sigma -- is drawn once outside the
        `t` loop and reused for every timestep. Independent per-frame parameters
        would inject apparent camera jitter into the very transition the dynamics
        head is learning, AND desynchronise the network input from the detached
        consistency target built from the same frame.

        Order is geometric -> photometric -> blur, matching the reference's spirit:
        shift/crop, then colour jitter, then Gaussian blur.

        Everything runs in FLOAT and is cast back to the stored dtype at the end.
        The shift alone is lossless in uint8 (whole-pixel offsets land exactly on
        pixel centres), so the shift-only path is bit-identical to the pre-jitter
        implementation -- including the RNG stream, because a knob at 0 draws
        NOTHING. Jitter and blur are inherently float; the extra transient is one
        [B,C,H,W] slice (154 MB at 224), not the whole [B,H+1,C,H,W] block.

        No-op unless aug_* is configured AND obs is a pixel dict, so the state
        arms and every existing pixel run are untouched.
        """
        wm = self.cfg.world_model
        pad = int(getattr(wm, "aug_shift_pad", 0) or 0)
        sigma = float(getattr(wm, "aug_prop_noise", 0.0) or 0.0)
        jitter = float(getattr(wm, "aug_color_jitter", 0.0) or 0.0)
        blur = float(getattr(wm, "aug_blur_sigma_max", 0.0) or 0.0)
        if not isinstance(obs, dict) or (
                pad <= 0 and sigma <= 0.0 and jitter <= 0.0 and blur <= 0.0):
            return obs
        out = dict(obs)
        if pad > 0 or jitter > 0.0 or blur > 0.0:
            for k, v in obs.items():
                if k == "prop" or v.dim() != 5:
                    continue
                b, h1, c, hh, ww = v.shape
                # --- ONE draw per (sample, camera), reused for all H+1 frames ---
                grid = jit = kern = None
                if pad > 0:
                    assert hh == ww, f"shift aug needs square frames, got {hh}x{ww}"
                    grid = self._shift_grid(b, hh, pad, v.device)
                if jitter > 0.0:
                    jit = self._jitter_params(b, jitter, v.device)
                if blur > 0.0:
                    kern = self._blur_kernel(b, blur, v.device)
                aug = torch.empty_like(v)
                for t in range(h1):
                    x = v[:, t].float()
                    if grid is not None:
                        x = torch.nn.functional.pad(x, (pad,) * 4, "replicate")
                        x = torch.nn.functional.grid_sample(
                            x, grid, padding_mode="zeros", align_corners=False)
                        x = x.round_()      # whole-pixel shift => exact in uint8
                    if jit is not None:
                        x = self._apply_jitter(x, jit)
                    if kern is not None:
                        x = self._apply_blur(x, kern)
                    aug[:, t] = x.round_().clamp_(0, 255).to(v.dtype)
                out[k] = aug
        if sigma > 0.0 and "prop" in out:
            out["prop"] = out["prop"] + torch.randn_like(out["prop"]) * sigma
        return out

    @staticmethod
    def _shift_grid(n: int, h: int, pad: int, device):
        """RandomShiftsAug's grid, drawn ONCE so every timestep gets the same shift."""
        eps = 1.0 / (h + 2 * pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad,
                                device=device, dtype=torch.float32)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base = base.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2),
                              device=device, dtype=torch.float32)
        shift *= 2.0 / (h + 2 * pad)
        return base + shift

    # --------------------------------------------------------------- jitter
    # RGB -> YIQ and back, for the hue rotation. Registered as plain constants
    # (not buffers) so nothing enters any state_dict.
    _YIQ = ((0.299, 0.587, 0.114),
            (0.5959, -0.2746, -0.3213),
            (0.2115, -0.5227, 0.3112))

    @staticmethod
    def _jitter_params(n: int, s: float, device):
        """Four colour-jitter factors, ONE set per (sample, camera).

        brightness / contrast / saturation ~ U(1-s, 1+s) (clamped at 0), and a hue
        rotation of U(-h, h) turns with h = min(0.5, s/3) -- so s=0.3 reproduces
        the conventional ColorJitter(0.3, 0.3, 0.3, 0.1). Fixed application order
        (b -> c -> sat -> hue) so a given seed is reproducible; torchvision
        shuffles the order, which would make runs unrepeatable.
        """
        f = 1.0 + (torch.rand(3, n, device=device) * 2.0 - 1.0) * s
        f = f.clamp_min(0.0)
        hue_max = min(0.5, s / 3.0)
        hue = (torch.rand(n, device=device) * 2.0 - 1.0) * hue_max
        return f[0], f[1], f[2], hue

    def _apply_jitter(self, x, jit):
        """x: [B,C,H,W] float in 0..255 (NOT clamped -- the caller clamps once)."""
        bright, contrast, sat, hue = jit
        v = lambda t: t.view(-1, 1, 1, 1)
        x = x * v(bright)
        gray = self._to_gray(x) if x.shape[1] == 3 else x.mean(1, keepdim=True)
        m = gray.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - m) * v(contrast) + m
        if x.shape[1] == 3:
            gray = self._to_gray(x)
            x = (x - gray) * v(sat) + gray
            x = self._apply_hue(x, hue)
        return x

    @staticmethod
    def _to_gray(x):
        w = torch.tensor([0.299, 0.587, 0.114], device=x.device,
                         dtype=x.dtype).view(1, 3, 1, 1)
        return (x * w).sum(1, keepdim=True)

    def _apply_hue(self, x, hue):
        """Rotate the I/Q chroma plane by `hue` turns, per sample. [B,3,H,W]."""
        yiq = torch.tensor(self._YIQ, device=x.device, dtype=x.dtype)
        rgb = torch.linalg.inv(yiq)
        th = hue.to(x.dtype) * (2.0 * math.pi)
        cos, sin = th.cos(), th.sin()
        one, zero = torch.ones_like(cos), torch.zeros_like(cos)
        rot = torch.stack([
            torch.stack([one, zero, zero], dim=-1),
            torch.stack([zero, cos, -sin], dim=-1),
            torch.stack([zero, sin, cos], dim=-1)], dim=-2)          # [B,3,3]
        mat = rgb.unsqueeze(0) @ rot @ yiq.unsqueeze(0)               # [B,3,3]
        return torch.einsum("bij,bjhw->bihw", mat, x)

    # ----------------------------------------------------------------- blur
    @staticmethod
    def _blur_kernel(n: int, sigma_max: float, device):
        """A 1-D Gaussian per sample, sigma ~ U(0, sigma_max]. ONE per (sample, camera).

        The kernel WIDTH is a constant set by sigma_max so the whole batch is one
        grouped conv; a small sigma simply collapses the kernel toward a delta.
        sigma is in STORED-frame pixels (§46⑫: this runs before `resize_to`).
        """
        sig = (1.0 - torch.rand(n, device=device)) * sigma_max   # (0, sigma_max]
        r = max(1, int(math.ceil(2.0 * sigma_max)))
        d = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        w = torch.exp(-(d.view(1, -1) ** 2) / (2.0 * sig.view(-1, 1) ** 2))
        return w / w.sum(dim=1, keepdim=True)                   # [B, 2r+1]

    @staticmethod
    def _apply_blur(x, kern):
        """Separable Gaussian with a PER-SAMPLE kernel, via a grouped conv."""
        b, c, h, w = x.shape
        k = kern.shape[1]
        r = k // 2
        wt = kern.view(b, 1, 1, k).repeat_interleave(c, dim=0)  # [B*C,1,1,k]
        y = x.reshape(1, b * c, h, w)
        y = torch.nn.functional.pad(y, (r, r, 0, 0), mode="reflect")
        y = torch.nn.functional.conv2d(y, wt, groups=b * c)
        y = torch.nn.functional.pad(y, (0, 0, r, r), mode="reflect")
        y = torch.nn.functional.conv2d(y, wt.view(b * c, 1, k, 1), groups=b * c)
        return y.reshape(b, c, h, w)

    @staticmethod
    def _obs_batch(obs) -> int:
        """Batch size of a batched obs sequence, tensor or dict."""
        if isinstance(obs, dict):
            return next(iter(obs.values())).shape[0]
        return obs.shape[0]

    @staticmethod
    def _obs_flat(obs, hi):
        """Flatten timesteps 0..hi-1 into the batch dim: [B,H+1,...] -> [B*hi,...].
        Accepts a tensor or an obs dict, so the BC path is arm-agnostic."""
        if isinstance(obs, dict):
            return {k: v[:, :hi].reshape(-1, *v.shape[2:]) for k, v in obs.items()}
        return obs[:, :hi].reshape(-1, obs.shape[-1])

    @staticmethod
    def _obs_len(obs) -> int:
        """Sequence length (H+1) of a batched obs sequence, tensor or dict."""
        if isinstance(obs, dict):
            return next(iter(obs.values())).shape[1]
        return obs.shape[1]

    def update_distill(self, states, actions, rewards, value_targets=None,
                       policy_mask=None, policy_states=None, policy_actions=None,
                       policy_states_mask=None, priv_states=None) -> dict[str, float]:
        """states[B,H+1,obs], actions[B,H,act], rewards[B,H].

        policy_mask[B,H] (optional): the paper's per-step 1_e(a_{t+i}) indicator,
        aligned with `actions`. Restricts the policy-BC loss to uncorrupted-expert
        actions. The dynamics/reward/value/decoder heads ALWAYS train on the full
        batch (broad coverage); only the policy head's gradient is masked.
        None => full-batch policy BC (legacy).

        policy_states/policy_actions/policy_states_mask (optional, [B,H+1,obs] /
        [B,H,act] / [B,H]): a SEPARATE batch (e.g. pure-expert trajectories) that
        the policy-BC head trains on, decoupled from the model-head batch. The
        policy uses its own encoder (enc_pi) and does not depend on the rolled
        dynamics latent, so this is a clean split: dynamics/reward/value/decoder
        learn from `states` (exploration data) while only the policy clones
        `policy_actions`. None => policy trains on the main `states` (legacy).

        value_targets[B,H+1] (optional): DENSE per-state value labels from the
        source RL critic (V_src(s) = min-ensemble Q(s, pi_src(s))). When given,
        the value head regresses to them — this is SimDist's value distillation
        and is what makes MPPI work: under sparse reward + end_on_success the
        TD-bootstrap fallback (value_targets=None) collapses to V==0 (every
        non-terminal step is r=0,done=0 so V(z)=gamma*V(z') has fixed point 0),
        leaving MPPI returns flat = no planning signal. With dense V_src the
        return Sum gamma^t r + gamma^H V has a real gradient to climb.
        """
        cfg = self.cfg.world_model
        H = self._obs_len(states) - 1
        rho = cfg.rho

        # policy-BC may train on a SEPARATE (pure-expert) batch; default to the
        # main batch (legacy). The policy uses enc_pi and does not touch z, so
        # this only changes what the policy head clones.
        # TRAINING-ONLY augmentation. Must happen here, before the first encode:
        # every downstream use (grad latent, detached consistency target, decoder,
        # policy BC) then reads the SAME augmented frames. No-op when aug_* is 0.
        states = self._augment_obs(states)
        if policy_states is not None:
            policy_states = self._augment_obs(policy_states)

        if policy_states is None:
            p_states, p_actions, p_mask = states, actions, policy_mask
        else:
            p_states, p_actions, p_mask = policy_states, policy_actions, policy_states_mask

        z = self.model.encode(self._obs_at(states, 0))  # [B, z]

        # ---- encode each timestep EXACTLY ONCE -------------------------------
        # The original code called encode() up to 3H+2 times per gradient step:
        # once for z (t=0, with grad), once per step for the consistency target,
        # again per step for the decoder loss, and again per step for the policy
        # BC. Every one of those beyond the first is redundant -- the parameters
        # do not change until the single optimizer step at the end -- so they can
        # be served from one detached cache. For the state arm this is a modest
        # saving; for the PIXEL arm it is the difference between ~17 and ~6 conv
        # forwards of batch B per step, i.e. the dominant training cost.
        #
        # `encode(x).detach()` is numerically identical to `with no_grad:
        # encode(x)` (no_grad only suppresses graph recording), and every encoder
        # in this repo is per-sample deterministic -- MLP for state, GroupNorm
        # (never BatchNorm) + no dropout for the pixel encoders -- so the cache
        # is BIT-identical to the per-site calls it replaces. Verified across 8
        # config combinations by scripts/test_distill_dedup.py, which compares
        # post-step weight checksums (loss alone cannot see a detach regression).
        #
        # NOTE: deliberately NOT batched into one encode([B*(H+1), ...]) call.
        # That would change BLAS/cuDNN blocking and hence summation order,
        # costing exact identity for a ~1e-7 difference. Revisit only with a
        # measured win and a relaxed tolerance.
        z_det = [z.detach()]
        with torch.no_grad():
            for t in range(1, H + 1):
                z_det.append(self.model.encode(self._obs_at(states, t)))

        # The policy head may train on a SEPARATE batch, which then needs its own
        # cache. Only build it on the shared-encoder+detach path; the other paths
        # call policy_action(ps_t) with grad and cannot use a detached cache.
        zp_det = None
        if cfg.shared_encoder and cfg.detach_pi:
            if policy_states is None:
                zp_det = z_det                      # same batch -> reuse
            else:
                with torch.no_grad():
                    zp_det = [self.model.encode(self._obs_at(p_states, t))
                              for t in range(H)]

        cons_l = rew_l = val_l = pol_l = dec_l = 0.0
        wsum = 0.0
        for t in range(H):
            a_t = actions[:, t]
            r_t = rewards[:, t]
            w = rho ** t
            wsum += w

            z_next_enc = z_det[t + 1]

            cons_step, z_next_pred = self._dynamics_cons_loss(z, a_t, z_next_enc)
            cons_l = cons_l + w * cons_step
            disc = cfg.discrete_heads
            nb, vmin, vmax = cfg.num_bins, cfg.vmin, cfg.vmax
            # ---- reward ----
            if disc:
                rew_l = rew_l + w * soft_ce(
                    self.model.reward_logits(z, a_t), r_t, nb, vmin, vmax).mean()
            elif cfg.reward_pos_weight != 1.0:
                # up-weight the rare r=1 steps so the head doesn't collapse to 0
                rerr = (self.model.predict_reward(z, a_t) - r_t) ** 2
                rwt = torch.where(r_t > 0.5, cfg.reward_pos_weight, 1.0)
                rew_l = rew_l + w * (rerr * rwt).mean()
            else:
                rew_l = rew_l + w * F.mse_loss(self.model.predict_reward(z, a_t), r_t)
            # value: paper-aligned (v̂_{t+1} − V_src(s_{t+1})) — the value head is
            # evaluated on the PREDICTED next latent and regressed to the dense
            # source-critic value of the NEXT state. TD-bootstrap fallback (no
            # source agent): V(z_t) <- r_t + gamma*(1-done)*V_target(enc(s_{t+1})).
            if value_targets is None:
                with torch.no_grad():
                    done = (r_t > 0.5).float()
                    v_tgt = r_t + self.discount * (1 - done) * \
                        self.model.predict_value(z_next_enc, target=True, reduce="mean")
                val_l = val_l + w * self._value_head_loss(z, v_tgt, disc, nb, vmin, vmax)
            else:
                v_tgt = value_targets[:, t + 1]
                val_l = val_l + w * self._value_head_loss(
                    z_next_pred, v_tgt, disc, nb, vmin, vmax)
            # decoder: reconstruct the state from its OWN (detached) encoding, so
            # decode inverts the encoder; at plan time decode(rolled_z)≈state
            # (consistency makes z≈encode(s)). Detach the input so this only
            # trains the decoder, not the dynamics encoder. Raw-state arm: decode
            # is the analytic denorm inverse (no params, exact) -> skip (loss==0).
            if not self.model.raw_state and self.model.decoder is not None:
                dec_l = dec_l + w * F.mse_loss(
                    self.model.decode(z_det[t]), states[:, t])
            # policy BC. Three modes:
            #  - dedicated enc_pi (legacy): policy_action uses its own encoder, the
            #    dynamics z is never touched -> no detach needed.
            #  - shared encoder + detach_pi (default for shared): query the policy on
            #    encode(s).detach() so the BC gradient does NOT reshape the dynamics
            #    encoder (avoids the policy-vs-consistency fight; TD-MPC2 also
            #    detaches the latent for the policy update). Lets the planner roll
            #    the policy directly in latent (no decode bridge).
            #  - shared encoder, no detach: BC also shapes the shared encoder.
            ps_t = self._obs_at(p_states, t)
            if cfg.shared_encoder and cfg.detach_pi:
                if t == 0 and cfg.pi_grad_t0 and policy_states is None:
                    # Let the BC signal reach the encoder at t=0 only. `z` is still
                    # the grad latent here (it is reassigned to z_next_pred at the
                    # end of the loop body), and the backward through it is already
                    # paid for by the consistency loss -- so this is free. Requires
                    # policy_states is None, i.e. the policy shares the model batch.
                    pred_a = self.model.policy_from_latent(z)
                else:
                    pred_a = self.model.policy_from_latent(zp_det[t])
            else:
                pred_a = self.model.policy_action(ps_t)
            pa_t = p_actions[:, t]
            if p_mask is None:
                pol_step = F.mse_loss(pred_a, pa_t)
            else:
                # masked mean over step t: only 1_e(a_t)=1 (uncorrupted-expert)
                # samples contribute their BC error.
                m_t = p_mask[:, t]  # [B]
                per = ((pred_a - pa_t) ** 2).mean(dim=-1)  # [B]
                pol_step = (per * m_t).sum() / m_t.sum().clamp(min=1.0)
            pol_l = pol_l + w * pol_step

            z = z_next_pred  # roll predicted latent (TD-MPC2)

        cons_l, rew_l, val_l, pol_l, dec_l = (
            x / wsum for x in (cons_l, rew_l, val_l, pol_l, dec_l))
        loss = (cfg.consistency_coef * cons_l + cfg.reward_coef * rew_l
                + cfg.value_coef * val_l + cfg.policy_coef * pol_l
                + cfg.decoder_coef * dec_l)

        # ★ AUXILIARY PRIVILEGED-STATE term (§43). Applied at t=0 ONLY, on the
        # GRAD latent `z` -- the whole purpose is to shape the ENCODER, and every
        # t>=1 latent in z_det is detached, so applying it there would train the
        # head while leaving the encoder untouched. Costs no extra encode: the
        # t=0 graph is already retained.
        aux_l = None
        if getattr(cfg, "aux_state_coef", 0.0) > 0 and priv_states is not None:
            if self.model.aux_state is None:
                raise ValueError(
                    "aux_state_coef>0 but no aux head was built -- the ckpt/cfg pair "
                    "predates the head. Rebuild stage 1 with aux_state_coef set.")
            aux_l = self.model.aux_state_loss(z, priv_states[:, 0])
            loss = loss + cfg.aux_state_coef * aux_l

        self.distill_opt.zero_grad()
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(self.model.trainable_parameters(), cfg.grad_clip)
        self.distill_opt.step()
        # EMA target only matters for the TD-bootstrap fallback.
        if value_targets is None:
            for src, tgt in zip(self.model.value_heads(), self.model.value_target_heads()):
                utils.soft_update_params(src, tgt, cfg.tau)

        # monitor value-head spread: ~0 std means a collapsed value -> dead MPPI.
        with torch.no_grad():
            z0 = self.model.encode(self._obs_at(states, 0))
            v0 = self.model.predict_value(z0)
            if self.model.num_value_heads() > 1:
                # IN-DISTRIBUTION disagreement (real encoded states). With random
                # priors this SHOULD be small — the heads all fit the teacher here.
                v_dis = self.model.predict_value_all(z0).std(0).mean()
                # ROLLED disagreement: `z` is the H-step rolled PREDICTED latent, i.e.
                # exactly the kind of imagined terminal MPPI scores. THIS is the number
                # that decides whether pessimism can do anything: if it is ~0 (or ~=
                # the in-dist one), min is a constant shift => argmax-invariant => no-op.
                v_dis_roll = self.model.predict_value_all(z).std(0).mean()
            else:
                v_dis = v_dis_roll = torch.zeros(())
            # DYNAMICS ensemble geometric disagreement (the penalty signal). IN-dist
            # at (z0, a_0) vs ROLLED at the H-step imagined latent (z, a_last) — the
            # off-manifold region MPPI's argmax pushes into. For the penalty to bite,
            # rolled must be > in-dist (heads agree on data, diverge off it).
            if self.model.num_dynamics_heads() > 1:
                d_dis = self.model.dynamics_disagreement(z0, actions[:, 0]).mean()
                d_dis_roll = self.model.dynamics_disagreement(z, actions[:, -1]).mean()
            else:
                d_dis = d_dis_roll = torch.zeros(())
        out = {
            "distill/loss": loss.item(),
            "distill/consistency": cons_l.item(),
            "distill/reward": rew_l.item(),
            "distill/value": val_l.item(),
            "distill/policy": pol_l.item(),
            "distill/decoder": float(dec_l),  # float() handles raw-state (dec_l==0.0)
            **({"distill/aux_state": float(aux_l)} if aux_l is not None else {}),
            "distill/grad_norm": float(gnorm),
            "distill/value_pred_std": float(v0.std()),
            "distill/value_pred_mean": float(v0.mean()),
            "distill/value_disagree": float(v_dis),
            "distill/value_disagree_rolled": float(v_dis_roll),
            "distill/dyn_disagree": float(d_dis),
            "distill/dyn_disagree_rolled": float(d_dis_roll),
        }
        if p_mask is not None:
            out["distill/policy_expert_frac"] = float(p_mask.mean())
        return out

    # ----------------------------------------------------------- stage 2
    def update_finetune(self, states, actions, rewards) -> dict[str, float]:
        """Finetune dynamics + policy on TARGET demos (consistency + policy BC)."""
        assert self.finetune_opt is not None, "no finetune params enabled"
        cfg = self.cfg.world_model
        H = self._obs_len(states) - 1
        rho = cfg.rho

        z = self.model.encode(self._obs_at(states, 0))
        cons_l = pol_l = 0.0
        wsum = 0.0
        for t in range(H):
            a_t = actions[:, t]
            w = rho ** t
            wsum += w
            with torch.no_grad():
                z_next_enc = self.model.encode(self._obs_at(states, t + 1))
            if self.cfg.finetune_dynamics:
                cons_step, z_next_pred = self._dynamics_cons_loss(z, a_t, z_next_enc)
                cons_l = cons_l + w * cons_step
            else:
                z_next_pred = self.model.next(z, a_t)
            if self.cfg.finetune_policy:
                pol_l = pol_l + w * F.mse_loss(
                    self.model.policy_action(self._obs_at(states, t)), a_t)
            z = z_next_pred

        cons_l = cons_l / wsum if self.cfg.finetune_dynamics else torch.zeros((), device=self.device)
        pol_l = pol_l / wsum if self.cfg.finetune_policy else torch.zeros((), device=self.device)
        loss = cfg.consistency_coef * cons_l + cfg.policy_coef * pol_l

        self.finetune_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.dynamics_policy_parameters(), cfg.grad_clip)
        self.finetune_opt.step()

        return {
            "finetune/loss": float(loss.item()),
            "finetune/consistency": float(cons_l.item()) if self.cfg.finetune_dynamics else 0.0,
            "finetune/policy": float(pol_l.item()) if self.cfg.finetune_policy else 0.0,
        }

    # ------------------------------------------------ TARGET adaptation phases
    def _encode_frozen(self, obs, chunk_at_96: int = 2048) -> torch.Tensor:
        """encode(obs) under no_grad, in chunks sized by image resolution.

        A constant chunk is wrong for pixels: memory scales with H*W, so a value
        tuned at 96x96 is ~5.4x too large at 224. Holds chunk*H*W constant.
        Tensor (state-arm) input needs no chunking and takes the direct path, so
        the state arms are untouched.
        """
        if not isinstance(obs, dict):
            with torch.no_grad():
                return self.model.encode(obs)
        cam = next(k for k in obs if k != "prop")
        h, w = obs[cam].shape[-2:]
        n = obs[cam].shape[0]
        step = max(64, int(chunk_at_96 * (96 * 96) / max(h * w, 1)))
        with torch.no_grad():
            if n <= step:
                return self.model.encode(obs)
            return torch.cat([
                self.model.encode({k: v[i:i + step] for k, v in obs.items()})
                for i in range(0, n, step)])

    def update_bc_policy(self, states, actions, lr: float = 3e-4,
                         anchor_states=None, anchor_actions=None,
                         anchor_coef: float = 0.0, anchor_eps: float = 0.0,
                         anchor_sat: float = 0.0, l2init_coef: float = 0.0,
                         weight_decay: float = 0.0,
                         anchor_is_latent: bool = False,
                         train_encoder: bool = False) -> dict[str, float]:
        """Stage-2 TARGET BC: train ONLY the policy head, freezing everything else
        (INCLUDING the encoder). The policy is a pure state->action map
        (`policy_action(s) = policy(encode(s))`); the BC loss is a flat MSE over all
        (s_t, a_t) pairs in the sampled segments — no dynamics rollout needed. The
        encoder is frozen by exclusion from the optimizer (only `model.policy`).

        OPTIONAL SOURCE ANCHOR (`anchor_coef > 0`, default OFF ⇒ byte-identical to
        the legacy path). Adds `anchor_coef * MSE(pi_theta(s_src), pi_src(s_src))`
        over SOURCE states, where `anchor_actions` are pi_src's actions captured
        BEFORE stage-2 started (frozen targets). This makes "how far the fine-tune
        moves the policy away from pi_src" an explicit knob, decoupled from "how
        many BC steps" and "which demos": the demo term fits the TARGET actions
        where the demos live, the anchor pins the policy where only SOURCE data
        lives. Motivation: the stage-1 value head is a distilled V^{pi_src} fitted
        on the SOURCE distribution and is FROZEN through stages 2-3, so a policy
        that drifts far from pi_src may invalidate the terminal value MPPI ranks
        with. `anchor_coef` sweeps that trade-off directly."""
        if self._bc_opt is None:
            bc_params = list(self.model.policy.parameters())
            if train_encoder:
                # DIAGNOSTIC / pixel escape hatch. findings.md §30: the SimNorm
                # latent is trained by the VALUE objective, so it keeps value
                # structure and discards mm-level geometry -- and policy
                # improvement needs exactly that geometry ("to fix a
                # hair's-breadth insertion you must know which way the sub-mm
                # misalignment points"). The state arm can fall back to raw
                # state; the PIXEL arm cannot, so a frozen pixel latent can cap
                # stage-2 BC no matter how low the BC loss goes (measured:
                # bc_loss 0.0119 < state's 0.0226 yet policy_only 0.08 vs 0.360).
                # Letting BC reshape the encoder tests whether the information is
                # absent from the LATENT or absent from the IMAGES.
                bc_params += list(self.model.encoder_parameters())
            self._bc_opt = torch.optim.Adam(bc_params, lr=lr,
                                            weight_decay=weight_decay)
            if train_encoder and anchor_coef > 0 and anchor_is_latent:
                # A cached latent anchor pool is only exact while the encoder is
                # FROZEN. Training the encoder makes those latents stale and the
                # anchor would silently pin the policy to a dead latent space.
                raise ValueError(
                    "bc_train_encoder=1 is incompatible with a cached LATENT anchor "
                    "pool: the cached z's assume a frozen encoder. Use "
                    "bc_anchor_coef=0, or rebuild the anchor from frames.")
            # (F3) theta_src snapshot for the L2-to-init variant. Taken here, i.e.
            # BEFORE the first .step(), so `model.policy` is still exactly pi_src.
            self._bc_init = ([p.detach().clone()
                              for p in self.model.policy.parameters()]
                             if l2init_coef > 0 else None)
        H = self._obs_len(states) - 1
        s_flat = self._obs_flat(states, H)   # [B*H, obs] or dict of [B*H, ...]
        a_flat = actions.reshape(-1, actions.shape[-1])        # [B*H, act]
        if train_encoder or not isinstance(s_flat, dict):
            # STATE arms keep the exact legacy call. `policy_action` routes through
            # the DEDICATED enc_pi when shared_encoder=0, so substituting
            # `policy(encode(s))` there would silently use the DYNAMICS encoder --
            # a different network, i.e. a wrong policy, not just a slower one.
            # The pixel arm asserts shared_encoder=1 (world_model.py), where
            # policy_action IS policy(encode(s)), so the chunked path below is
            # exactly equivalent for it.
            pred = self.model.policy_action(s_flat)
        else:
            # ENCODER IS FROZEN here (excluded from self._bc_opt), so building an
            # autograd graph through it is pure waste -- and at 224 it is fatal:
            # bc_batch_size=256 x H=5 x 2 cameras is 2560 images, whose first conv
            # activations alone are ~10 GB, which OOMs an L40S before step 1.
            # Encoding under no_grad in resolution-sized chunks is EXACT, by the
            # same argument the cached latent anchor pool below already relies on:
            # this optimizer holds only model.policy.parameters(), so
            # d/d_theta_policy MSE(policy(z), a) is identical whether z carried a
            # graph or not. Forward values are unchanged, so the state arm's
            # numbers are unaffected -- it just stops paying for a discarded graph.
            pred = self.model.policy(self._encode_frozen(s_flat))
        loss = F.mse_loss(pred, a_flat)
        total, anchor_loss = loss, None
        if anchor_coef > 0 and anchor_states is not None:
            # anchor_is_latent: the pool is ALREADY ENCODED (pixel arm -- a frame
            # pool would be 40k x 2 x 27 KB re-encoded every BC step). Exact, not
            # an approximation: this optimizer holds only model.policy.parameters(),
            # so d/d_theta_policy MSE(policy(z), tgt) is identical whether z was
            # computed now or cached -- valid because the encoder is frozen here
            # AND deterministic (GroupNorm/LayerNorm, no dropout, no BatchNorm).
            a_pred = (self.model.policy(anchor_states) if anchor_is_latent
                      else self.model.policy_action(anchor_states))
            if anchor_sat > 0:
                # (F5) SATURATING anchor -- the mirror image of the hinge on the
                # g(d) axis. Quadratic for small deviations, capped for large ones,
                # so the demo term is free to win outright on the few states where
                # it wants to move a lot. This is exactly Hellinger^2 between the
                # two fixed-covariance Gaussians (1 - exp(-d^2/8 sigma^2)), i.e. a
                # real divergence, with anchor_sat playing the role of 8 sigma^2.
                d = a_pred - anchor_actions
                anchor_loss = (1.0 - torch.exp(-d.pow(2).sum(-1) / anchor_sat)).mean()
            elif anchor_eps > 0:
                # (F4) KL as a per-state HARD trust region instead of a penalty:
                # KL <= delta  <=>  ||pi_theta(s) - pi_src(s)||_2 <= eps, with
                # eps = sigma*sqrt(2*delta). Free inside the ball, quadratic
                # outside. sqrt(.+1e-8) because torch.norm's backward is NaN at 0
                # and the anchor residual does collapse toward 0 late in stage-2.
                d = a_pred - anchor_actions
                dist = torch.sqrt(d.pow(2).sum(-1) + 1e-8)
                anchor_loss = F.relu(dist - anchor_eps).pow(2).mean()
            else:
                anchor_loss = F.mse_loss(a_pred, anchor_actions)
            total = loss + anchor_coef * anchor_loss
        l2init_loss = None
        if l2init_coef > 0 and self._bc_init is not None:
            # (F3) same KL trust region but with the Fisher metric replaced by I:
            # proximity to pi_src in PARAMETER space rather than in function space
            # on the source state distribution.
            l2init_loss = sum(((p - p0) ** 2).sum() for p, p0
                              in zip(self.model.policy.parameters(), self._bc_init))
            total = total + l2init_coef * l2init_loss
        self._bc_opt.zero_grad()
        total.backward()
        gnorm = nn.utils.clip_grad_norm_(
            self.model.policy.parameters(), self.cfg.world_model.grad_clip)
        self._bc_opt.step()
        out = {"bc/policy_loss": float(loss.item()), "bc/grad_norm": float(gnorm)}
        if anchor_loss is not None:
            out["bc/anchor_loss"] = float(anchor_loss.item())
        if l2init_loss is not None:
            out["bc/l2init_loss"] = float(l2init_loss.item())
        return out

    def update_dynamics_only(self, states, actions, lr: float = 3e-4,
                             anchor_z=None, anchor_a=None, anchor_tgt=None,
                             anchor_coef: float = 0.0) -> dict[str, float]:
        """Stage-3 TARGET adapt: train ONLY the dynamics head, freezing everything
        else (INCLUDING the encoder). Multi-step latent-consistency loss
        ||next(z,a) - sg(encode(s'))||^2 rolled from the segment's first state
        (TD-MPC2 style); the next-state targets are produced by the FROZEN encoder,
        and only `model.dynamics` is in the optimizer.

        anchor_* (default OFF, byte-identical when anchor_coef<=0): the SOURCE
        anchor, the dynamics-side mirror of stage-2's bc_anchor. Adds
        `anchor_coef * MSE(next(z_src, a_src), next_stage1(z_src, a_src))` on
        SOURCE (z, a) pairs whose targets were snapshotted from the stage-1
        dynamics BEFORE adapt began. Motivation: stage-3 refits the dynamics on
        whatever narrow on-policy slice the current planner visits, and a
        134-run meta-analysis (CLAUDE.md corr(round-0, gain) = -0.72) shows every
        arm collapsing toward one ~0.7 attractor regardless of where it started
        -- consistent with the source structure being overwritten. The anchor
        pins the model on source (z, a) the target rollouts never cover, exactly
        as bc_anchor pins the policy.

        NOTE the anchor targets the ENSEMBLE MEAN (`next`), not each head, so
        head diversity -- which is where the dynens gain comes from -- stays
        free."""
        if self._dyn_opt is None:
            self._dyn_opt = torch.optim.Adam(self.model.dynamics_parameters(), lr=lr)
        cfg = self.cfg.world_model
        H = self._obs_len(states) - 1
        rho = cfg.rho
        with torch.no_grad():
            z = self.model.encode(self._obs_at(states, 0))  # frozen encoder
        cons_l = 0.0
        wsum = 0.0
        for t in range(H):
            with torch.no_grad():
                z_next_enc = self.model.encode(self._obs_at(states, t + 1))  # frozen target
            # trains ALL ensemble heads (per-head bagged); rolls the ensemble mean.
            cons_step, z_next_pred = self._dynamics_cons_loss(z, actions[:, t], z_next_enc)
            w = rho ** t
            wsum += w
            cons_l = cons_l + w * cons_step
            z = z_next_pred  # roll the predicted latent (multi-step, TD-MPC2)
        cons_l = cons_l / wsum
        out = {"adapt/consistency": float(cons_l.item())}
        total = cons_l
        if anchor_coef > 0 and anchor_z is not None:
            anc = F.mse_loss(self.model.next(anchor_z, anchor_a), anchor_tgt)
            total = total + anchor_coef * anc
            out["adapt/dyn_anchor"] = float(anc.item())
        self._dyn_opt.zero_grad()
        total.backward()
        gnorm = nn.utils.clip_grad_norm_(
            self.model.dynamics_parameters(), cfg.grad_clip)
        self._dyn_opt.step()
        out["adapt/grad_norm"] = float(gnorm)
        return out

    def reinit_head(self, name: str):
        """Head-ablation (train_simdist_headablate.py): re-initialize ONE head's
        weights so it trains from scratch on TARGET data while every OTHER module
        stays frozen at the loaded ckpt. name in {"dynamics", "value"}."""
        heads = ([self.model.dynamics] if name == "dynamics"
                 else self.model.value_heads())   # value: reinit EVERY ensemble member

        def _reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        for head in heads:
            head.apply(_reset)
        # drop the stale cached optimizer for this head (it referenced old params'
        # state; Adam moments must restart for the reinitialized weights).
        if name == "dynamics":
            self._dyn_opt = None
        else:
            for src, tgt in zip(self.model.value_heads(), self.model.value_target_heads()):
                tgt.load_state_dict(src.state_dict())
            self._val_opt = None
        print(f"[reinit] head '{name}' re-initialized from scratch")

    def update_value_only(self, states, value_targets, lr: float = 3e-4
                          ) -> dict[str, float]:
        """Head-ablation Exp2: train ONLY the value head (encoder/dynamics/reward/
        policy frozen). Per-STATE dense regression V(encode(s)) -> value_targets
        (the STORED target-expert critic Q from the diverse dataset's `values`
        field; two-hot CE if discrete_heads else MSE) on the FROZEN-encoder latent.
        Isolates whether a TARGET-distilled value (vs the frozen SOURCE-distilled
        one) unblocks MPPI on the target. states[B,obs], value_targets[B]."""
        vparams = [p for hd in self.model.value_heads() for p in hd.parameters()]
        if self._val_opt is None:
            self._val_opt = torch.optim.Adam(vparams, lr=lr)
        cfg = self.cfg.world_model
        with torch.no_grad():
            z = self.model.encode(states)  # frozen encoder
        v_tgt = value_targets.reshape(-1)
        val_l = self._value_head_loss(z, v_tgt, cfg.discrete_heads,
                                      cfg.num_bins, cfg.vmin, cfg.vmax)
        self._val_opt.zero_grad()
        val_l.backward()
        gnorm = nn.utils.clip_grad_norm_(vparams, cfg.grad_clip)
        self._val_opt.step()
        with torch.no_grad():
            # keep the EMA targets in sync (MPPI uses the online value, but stay tidy)
            for src, tgt in zip(self.model.value_heads(), self.model.value_target_heads()):
                tgt.load_state_dict(src.state_dict())
            v_pred = self.model.predict_value(z)
        return {
            "adapt/value_loss": float(val_l.item()),
            "adapt/value_pred_mean": float(v_pred.mean()),
            "adapt/value_target_mean": float(v_tgt.mean()),
            "adapt/grad_norm": float(gnorm),
        }

    def update_value_calib(self, states, actions, teacher, lr: float = 3e-4
                           ) -> dict[str, float]:
        """VALUE CALIBRATION onto the IMAGINED terminal latent (findings §27 #3,
        the online form of train_value_calibration.py): regress the value head at
        the latent the planner actually scores,
            z_H^img = f^H(encode(s_0), a_{0:H})            (dynamics, no grad)
        onto what the FROZEN stage-1 value says about the REAL terminal state,
            y = V_teacher(encode(s_H^real)),
        i.e. drive the value error eps = V(z_H^img) - V_teacher(z_H^real) to 0.
        Only the value head(s) are in the optimizer; encoder / dynamics / policy
        are untouched. `teacher` is a frozen copy of the model taken before adapt,
        so the label does not chase the head being trained."""
        vparams = [p for hd in self.model.value_heads() for p in hd.parameters()]
        if getattr(self, "_vcal_opt", None) is None:
            self._vcal_opt = torch.optim.Adam(vparams, lr=lr)
        cfg = self.cfg.world_model
        H = self._obs_len(states) - 1
        with torch.no_grad():
            z = self.model.encode(self._obs_at(states, 0))
            for t in range(H):
                z = self.model.next(z, actions[:, t])
            v_tgt = teacher.predict_value(
                teacher.encode(self._obs_at(states, H)), target=False).reshape(-1)
        val_l = self._value_head_loss(z, v_tgt, cfg.discrete_heads,
                                      cfg.num_bins, cfg.vmin, cfg.vmax)
        self._vcal_opt.zero_grad()
        val_l.backward()
        gnorm = nn.utils.clip_grad_norm_(vparams, cfg.grad_clip)
        self._vcal_opt.step()
        with torch.no_grad():
            for src, tgt in zip(self.model.value_heads(), self.model.value_target_heads()):
                tgt.load_state_dict(src.state_dict())
            eps = self.model.predict_value(z, target=False).reshape(-1) - v_tgt
        return {
            "vcalib/loss": float(val_l.item()),
            "vcalib/abs_eps": float(eps.abs().mean()),
            "vcalib/grad_norm": float(gnorm),
        }

    def update_dynamics_vaml(self, states, actions, lr: float = 3e-4) -> dict[str, float]:
        """与 update_dynamics_only 逐行相同, 唯一区别: loss 从 latent 一致性 MSE
        ||next(z,a) - sg(encode(s'))||^2 换成 VALUE-aware(VAML)
        ||V(next(z,a)) - sg(V(encode(s')))||^2 —— 只惩罚 dynamics 误差里"影响 value 评估"
        的部分, 放过 value 同构面上的偏差。value 头 predict_value 冻结(只 dynamics 在 optimizer)。
        用于 threading_simdist_adapt 的"只改 loss"对照。执行轨迹逐点(无候选集, 故是逐点
        value-aware 而非 argmax-aware 的集内方差版)。"""
        if self._dyn_opt is None:
            self._dyn_opt = torch.optim.Adam(self.model.dynamics.parameters(), lr=lr)
        cfg = self.cfg.world_model
        H = self._obs_len(states) - 1
        rho = cfg.rho
        with torch.no_grad():
            z = self.model.encode(self._obs_at(states, 0))  # frozen encoder
        vaml_l = 0.0
        wsum = 0.0
        for t in range(H):
            with torch.no_grad():
                z_next_enc = self.model.encode(self._obs_at(states, t + 1))  # frozen target
                v_true = self.model.predict_value(z_next_enc, target=False).detach()
            z_next_pred = self.model.next(z, actions[:, t])              # trains dynamics
            v_pred = self.model.predict_value(z_next_pred, target=False)  # value 头冻结, 但对 z 可导
            w = rho ** t
            wsum += w
            vaml_l = vaml_l + w * F.mse_loss(v_pred, v_true)
            z = z_next_pred  # roll the predicted latent
        vaml_l = vaml_l / wsum
        self._dyn_opt.zero_grad()
        vaml_l.backward()
        gnorm = nn.utils.clip_grad_norm_(
            self.model.dynamics.parameters(), cfg.grad_clip)
        self._dyn_opt.step()
        return {"adapt/consistency": float(vaml_l.item()), "adapt/grad_norm": float(gnorm)}

    def update_dynamics_residual_value(self, states, actions, value_targets,
                                       lr: float = 3e-4, value_coef: float = 1.0
                                       ) -> dict[str, float]:
        """Stage-3 adapt, dynamics + the MARGINAL (residual) value head.

        Same dynamics term as update_dynamics_value. The value term differs: the
        trainable parameter is the residual mlp of
        `V(z) = V_base(z).detach() + scale * mlp(z)` (world_model
        attach_value_residual), and it is fit by MSE on the Monte-Carlo
        return-to-go -- NOT the two-hot CE the categorical head uses, because a
        scalar residual has no logit-space counterpart. The distilled value head
        itself never moves, so the departure from V_src is exactly
        `scale * mlp(z)` and is directly readable.
        """
        m = self.model
        assert getattr(m, "value_residual", None) is not None, \
            "attach_value_residual() must be called before this update"
        if self._dynres_opt is None:
            params = list(m.dynamics.parameters()) + list(m.value_residual.parameters())
            self._dynres_opt = torch.optim.Adam(params, lr=lr)
        cfg = self.cfg.world_model
        B = states.shape[0]
        H = states.shape[1] - 1
        rho = cfg.rho
        with torch.no_grad():
            z = m.encode(states[:, 0])
        cons_l, wsum = 0.0, 0.0
        for t in range(H):
            with torch.no_grad():
                z_next_enc = m.encode(states[:, t + 1])
            z_next_pred = m.next(z, actions[:, t])
            w = rho ** t
            wsum += w
            cons_l = cons_l + w * F.mse_loss(z_next_pred, z_next_enc)
            z = z_next_pred
        cons_l = cons_l / wsum

        with torch.no_grad():
            z_all = m.encode(states.reshape(B * (H + 1), -1))      # frozen encoder
        v_tgt = value_targets.reshape(-1)
        v_pred = m.predict_value(z_all)                            # base.detach() + residual
        val_l = F.mse_loss(v_pred, v_tgt)

        loss = cfg.consistency_coef * cons_l + value_coef * val_l
        self._dynres_opt.zero_grad()
        loss.backward()
        params = list(m.dynamics.parameters()) + list(m.value_residual.parameters())
        gnorm = nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        self._dynres_opt.step()
        with torch.no_grad():
            delta = m.value_residual_scale * m.value_residual(z_all).squeeze(-1)
        return {
            "adapt/dyn_cons_loss": float(cons_l),
            "adapt/value_loss": float(val_l),
            "adapt/gnorm": float(gnorm),
            # how far the marginal value has moved off the distilled one; this is
            # the quantity the residual parameterisation exists to expose.
            "adapt/value_delta_absmean": float(delta.abs().mean()),
            "adapt/value_delta_std": float(delta.std()),
        }

    def update_dynamics_value(self, states, actions, value_targets,
                              lr: float = 3e-4, value_coef: float = 1.0
                              ) -> dict[str, float]:
        """Stage-3 TARGET adapt, DYNAMICS + VALUE heads (encoder/reward/policy frozen).

        dynamics: multi-step latent-consistency (as in update_dynamics_only).
        value:    V(encode(s_t)) -> value_targets[:,t] = MC discounted return-to-go
                  (two-hot CE if discrete_heads else MSE), on the FROZEN-encoder
                  latent of REAL states. Restores the on-target ranking signal MPPI
                  needs (its return is dominated by gamma^H * V(z_H)); the frozen
                  SOURCE value is flat on target. Only model.dynamics + model.value
                  train. value_targets[B, H+1] from sample(return_value=True)."""
        if self._dynval_opt is None:
            params = list(self.model.dynamics.parameters()) + list(self.model.value.parameters())
            self._dynval_opt = torch.optim.Adam(params, lr=lr)
        cfg = self.cfg.world_model
        B = self._obs_batch(states)
        H = self._obs_len(states) - 1
        rho = cfg.rho
        with torch.no_grad():
            z = self.model.encode(self._obs_at(states, 0))
        cons_l = 0.0
        wsum = 0.0
        for t in range(H):
            with torch.no_grad():
                z_next_enc = self.model.encode(self._obs_at(states, t + 1))
            z_next_pred = self.model.next(z, actions[:, t])
            w = rho ** t
            wsum += w
            cons_l = cons_l + w * F.mse_loss(z_next_pred, z_next_enc)
            z = z_next_pred
        cons_l = cons_l / wsum

        with torch.no_grad():
            z_all = self.model.encode(states.reshape(B * (H + 1), -1))  # frozen encoder
        v_tgt = value_targets.reshape(-1)
        if cfg.discrete_heads:
            val_l = soft_ce(self.model.value_logits(z_all), v_tgt,
                            cfg.num_bins, cfg.vmin, cfg.vmax).mean()
        else:
            val_l = F.mse_loss(self.model.predict_value(z_all), v_tgt)

        loss = cfg.consistency_coef * cons_l + value_coef * val_l
        self._dynval_opt.zero_grad()
        loss.backward()
        params = list(self.model.dynamics.parameters()) + list(self.model.value.parameters())
        gnorm = nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        self._dynval_opt.step()
        with torch.no_grad():
            v_pred = self.model.predict_value(z_all)
        return {
            "adapt/consistency": float(cons_l.item()),
            "adapt/value_loss": float(val_l.item()),
            "adapt/value_pred_mean": float(v_pred.mean()),
            "adapt/value_target_mean": float(v_tgt.mean()),
            "adapt/grad_norm": float(gnorm),
        }

    def update_dynamics_value_encoder(self, states, actions, value_targets,
                                      lr: float = 3e-4, value_coef: float = 1.0
                                      ) -> dict[str, float]:
        """Stage-3 TARGET adapt, DYNAMICS + VALUE + ENCODER (reward/policy frozen).

        Same losses as `update_dynamics_value`, but the encoder is now TRAINABLE:
        gradients flow through the online `encode(s)` in both the consistency loss
        (the t=0 source latent + the rolled prediction) and the value regression.
        The consistency TARGET `encode(s_{t+1})` is still stop-grad (TD-MPC2 style)
        so it doesn't chase a moving target. With shared_encoder=1 the policy reads
        the SAME encoder, so this also re-shapes the (frozen-weight) policy's input
        latent -- the whole point of the ablation. model.dynamics + model.value +
        model.encoder train. value_targets[B, H+1] from sample(return_value=True)."""
        if self._dynvalenc_opt is None:
            params = (list(self.model.dynamics.parameters())
                      + list(self.model.value.parameters())
                      + self.model.encoder_parameters())
            assert self.model.encoder_parameters(), \
                "encoder arm needs a learned encoder (encoder_type='latent')"
            self._dynvalenc_opt = torch.optim.Adam(params, lr=lr)
        cfg = self.cfg.world_model
        B = self._obs_batch(states)
        H = self._obs_len(states) - 1
        rho = cfg.rho
        z = self.model.encode(self._obs_at(states, 0))  # encoder TRAINS
        cons_l = 0.0
        wsum = 0.0
        for t in range(H):
            with torch.no_grad():
                z_next_enc = self.model.encode(self._obs_at(states, t + 1))  # stop-grad
            z_next_pred = self.model.next(z, actions[:, t])
            w = rho ** t
            wsum += w
            cons_l = cons_l + w * F.mse_loss(z_next_pred, z_next_enc)
            z = z_next_pred
        cons_l = cons_l / wsum

        z_all = self.model.encode(states.reshape(B * (H + 1), -1))  # encoder TRAINS
        v_tgt = value_targets.reshape(-1)
        if cfg.discrete_heads:
            val_l = soft_ce(self.model.value_logits(z_all), v_tgt,
                            cfg.num_bins, cfg.vmin, cfg.vmax).mean()
        else:
            val_l = F.mse_loss(self.model.predict_value(z_all), v_tgt)

        loss = cfg.consistency_coef * cons_l + value_coef * val_l
        self._dynvalenc_opt.zero_grad()
        loss.backward()
        params = (list(self.model.dynamics.parameters())
                  + list(self.model.value.parameters())
                  + self.model.encoder_parameters())
        gnorm = nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        self._dynvalenc_opt.step()
        with torch.no_grad():
            v_pred = self.model.predict_value(self.model.encode(
                states.reshape(B * (H + 1), -1)))
        return {
            "adapt/consistency": float(cons_l.item()),
            "adapt/value_loss": float(val_l.item()),
            "adapt/value_pred_mean": float(v_pred.mean()),
            "adapt/value_target_mean": float(v_tgt.mean()),
            "adapt/grad_norm": float(gnorm),
        }

    # ---------------------------------------------------------- eval hooks
    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)
