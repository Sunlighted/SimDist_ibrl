"""Flow BC policy (± Universal-Guidance steering by a frozen world model) as a
drop-in eval agent for the SimDist policy-replacement experiment (CLAUDE.md §24
follow-up).

Three arms (all share ONE trained flow policy from train_flow_ug.py):
  (A) use_ug=0            : the flow policy alone (no world model). act_chunk =
                           the flow's own Euler ODE sample.
  (B) use_ug=1, source WM : Universal Guidance — at each ODE step, form the
                           Tweedie denoised action estimate a_hat = x + (1-t)*v,
                           roll its first H actions through the FROZEN world-model
                           dynamics, score Σγ^h r̂ + γ^H V̂, and nudge x up that
                           gradient before the Euler step. WM = the source stage-1
                           `diversev2_h5` model.
  (B1) use_ug=1, target WM: same, but the dynamics head is the target-ADAPTED one
                           from `threading_simdist_adapt_seed1` (value/reward still
                           the source-distilled heads).

Universal Guidance (Bansal et al. 2023), forward variant: guidance is applied to
the single-point posterior x0_hat (the denoised action), so we differentiate the
world-model return w.r.t. x0_hat and step x along it. With `stop_grad_velocity=1`
(default, cheap) the flow velocity is treated as fixed when forming x0_hat, so no
gradient flows through the (heavy) UNet — only through the world model.

Exposes `act_chunk(obs, eval_mode=True)` so evaluate/{eval,multi_process_eval}.py
amortize the policy call over Ta env steps (see CLAUDE.md §4).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FlowUGConfig:
    use_ug: int = 0                 # 0 = flow-only (arm A); 1 = UG-steered (B/B1)
    steering_coeff: float = 0.5     # UG step size (on the unit-normalized grad)
    ug_num_steps: int = 0           # ODE steps for the UG sample; 0 = flow's own
    ug_rollout_H: int = 0           # world-model rollout depth for guidance; 0 = model.horizon
    use_reward: int = 1             # include the reward head in the guidance return
    discount: float = 0.99
    stop_grad_velocity: int = 1     # 1 = don't backprop through the flow velocity (cheap, standard)
    grad_normalize: int = 1         # per-sample unit-norm the guidance gradient
    guidance_start_step: int = 1    # skip UG on step 0 (x is pure noise; x0_hat meaningless)
    clamp_x: int = 1                # clamp the normalized action iterate to [-1,1]
    # ---- MPPI selection layer on top of flow(+UG) ----
    # Sample `mppi_candidates` chunk proposals from the flow (each UG-steered when
    # use_ug=1), OPTIONALLY add `mppi_random` Gaussian-noised copies (kept small —
    # "少随机、多 base"), roll each candidate's first H actions through the
    # target-adapted world model, and pick the best-by-return (or softmax-elite
    # mean). No CEM Gaussian iterations — the candidate pool IS the flow proposal
    # distribution, so the search stays on the flow manifold (anti model-exploit).
    mppi_enabled: int = 0
    mppi_candidates: int = 16       # # flow(+UG) chunk proposals per env
    mppi_random: int = 0            # # extra Gaussian-noised candidates (small!)
    mppi_random_std: float = 0.1    # noise std for the random candidates (normalized space)
    mppi_select: str = "best"       # "best" (argmax return) | "softmax" (elite-weighted mean)
    mppi_elites: int = 4            # # elites for softmax selection
    mppi_temperature: float = 10.0  # softmax temperature over elite returns


class FlowUGAgent(nn.Module):
    def __init__(self, flow, model=None, cfg: FlowUGConfig = FlowUGConfig(),
                 device: str = "cuda"):
        super().__init__()
        self.flow = flow                       # StateFlowPolicy (EMA applied at load)
        self.model = model                     # WorldModel (frozen) or None (arm A)
        self.cfg = cfg
        self.device = device
        if cfg.use_ug:
            assert model is not None, "use_ug=1 needs a world model"

    # --------------------------------------------------------------- guidance
    def _guidance_return(self, state_raw: torch.Tensor, a_raw: torch.Tensor,
                         H: int) -> torch.Tensor:
        """state_raw[B,obs], a_raw[B,Tp,act] (raw scale) -> return[B] =
        Σ_{h<H} γ^h r̂(z_h,a_h) + γ^H V̂(z_H). Differentiable w.r.t. a_raw."""
        m = self.model
        z = m.encode(state_raw)                       # [B, z]
        ret = 0.0
        for h in range(H):
            a_h = a_raw[:, h]
            if self.cfg.use_reward:
                ret = ret + (self.cfg.discount ** h) * m.predict_reward(z, a_h)
            z = m.next(z, a_h)
        ret = ret + (self.cfg.discount ** H) * m.predict_value(z)
        return ret

    def _sample_chunk_ug(self, state_raw: torch.Tensor) -> torch.Tensor:
        """UG-steered Euler ODE. state_raw[B,obs] -> normalized chunk [B,Tp,act]."""
        flow, cfg = self.flow, self.cfg
        state_n = flow._normalize_state(state_raw)
        B = state_n.shape[0]
        x = torch.randn(B, flow.Tp, flow.action_dim,
                        device=state_raw.device, dtype=state_n.dtype)
        N = cfg.ug_num_steps or flow.cfg.num_inference_steps
        dt = 1.0 / N
        H = cfg.ug_rollout_H or self.model.cfg.horizon
        H = min(H, flow.Tp)
        coeff = cfg.steering_coeff
        for i in range(N):
            t_val = i * dt
            t = torch.full((B,), t_val, device=state_raw.device, dtype=state_n.dtype)
            with torch.no_grad():
                v = flow.forward_velocity(x, t, state_n)   # velocity (fixed for guidance)
            if coeff > 0 and i >= cfg.guidance_start_step:
                with torch.enable_grad():
                    x_req = x.detach().requires_grad_(True)
                    if cfg.stop_grad_velocity:
                        a_hat = x_req + (1.0 - t_val) * v          # v const
                    else:
                        v_g = flow.forward_velocity(x_req, t, state_n)
                        a_hat = x_req + (1.0 - t_val) * v_g
                    a_raw = flow._denormalize_action(a_hat)
                    g = self._guidance_return(state_raw, a_raw, H)  # [B]
                    grad = torch.autograd.grad(g.sum(), x_req)[0]   # [B,Tp,act]
                with torch.no_grad():
                    if cfg.grad_normalize:
                        gn = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1)
                    else:
                        gn = grad
                    x = x + coeff * gn
            x = x + v * dt
            if cfg.clamp_x:
                x = x.clamp(-1.0, 1.0)
        return x.detach()

    @torch.no_grad()
    def _score_return(self, state_raw: torch.Tensor, a_raw: torch.Tensor,
                      H: int) -> torch.Tensor:
        """No-grad version of the guidance return, for ranking MPPI candidates."""
        return self._guidance_return(state_raw, a_raw, H)

    def _plan_flow_mppi(self, state_raw: torch.Tensor) -> torch.Tensor:
        """MPPI-select over flow(+UG) chunk proposals. state_raw[B,obs] ->
        selected normalized chunk [B,Tp,act]. Candidates are K flow samples (each
        UG-steered if use_ug), plus `mppi_random` Gaussian-noised copies; scored by
        the target-adapted world-model return; selected by best or softmax-elite."""
        cfg, flow = self.cfg, self.flow
        B = state_raw.shape[0]
        K = cfg.mppi_candidates
        H = cfg.ug_rollout_H or self.model.cfg.horizon
        H = min(H, flow.Tp)

        # K flow(+UG) proposals per env: tile state to [B*K] and sample once.
        st = state_raw.repeat_interleave(K, dim=0)                     # [B*K, obs]
        cand = self._sample_chunk_ug(st) if cfg.use_ug else \
            flow._sample_full_chunk(flow._normalize_state(st))          # [B*K, Tp, act] normalized
        cand = cand.view(B, K, flow.Tp, flow.action_dim)

        # optional small pool of Gaussian-noised candidates (normalized space)
        if cfg.mppi_random > 0:
            R = cfg.mppi_random
            base = cand[:, torch.randint(0, K, (R,), device=cand.device)]  # [B,R,Tp,act]
            noised = (base + torch.randn_like(base) * cfg.mppi_random_std).clamp(-1, 1)
            cand = torch.cat([cand, noised], dim=1)                     # [B, K+R, Tp, act]
        M = cand.shape[1]

        # score every candidate by the adapted-WM return (on raw actions)
        flat = cand.reshape(B * M, flow.Tp, flow.action_dim)
        a_raw = flow._denormalize_action(flat)
        st_m = state_raw.repeat_interleave(M, dim=0)                    # [B*M, obs]
        ret = self._score_return(st_m, a_raw, H).view(B, M)            # [B, M]

        if cfg.mppi_select == "softmax":
            k = min(cfg.mppi_elites, M)
            ev, ei = torch.topk(ret, k, dim=1)                         # [B,k]
            w = torch.softmax(cfg.mppi_temperature * (ev - ev.max(1, keepdim=True).values), dim=1)
            elite = torch.gather(
                cand, 1, ei[:, :, None, None].expand(B, k, flow.Tp, flow.action_dim))
            chunk = (w[:, :, None, None] * elite).sum(1)               # [B,Tp,act]
        else:  # best
            idx = ret.argmax(dim=1)                                    # [B]
            chunk = cand[torch.arange(B, device=cand.device), idx]     # [B,Tp,act]
        return chunk

    # --------------------------------------------------------------- act
    @torch.no_grad()
    def act_chunk(self, obs, *, eval_mode: bool = True):
        """Return (B, Ta, act) [or (Ta, act) for a single env] on CPU."""
        assert eval_mode
        if not self.cfg.use_ug and not self.cfg.mppi_enabled:
            return self.flow.act_chunk(obs, eval_mode=eval_mode)   # flow-only (arm A)

        state = obs["state"].to(self.device).float()
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
        # UG sampling needs grad internally; the outer @torch.no_grad is overridden
        # by torch.enable_grad() inside _sample_chunk_ug.
        if self.cfg.mppi_enabled:
            assert self.model is not None, "mppi_enabled needs a world model"
            chunk_full = self._plan_flow_mppi(state)               # [B,Tp,act] normalized
        else:
            chunk_full = self._sample_chunk_ug(state)              # [B,Tp,act] normalized
        chunk = chunk_full[:, : self.flow.Ta].contiguous()
        chunk = self.flow._denormalize_action(chunk)               # -> raw scale
        if single:
            chunk = chunk.squeeze(0)
        return chunk.cpu()

    @torch.no_grad()
    def act(self, obs, *, eval_mode: bool = True, stddev: float = 0.0, cpu: bool = True):
        """Single-action fallback (first action of a fresh chunk)."""
        chunk = self.act_chunk(obs, eval_mode=eval_mode)
        action = chunk[0] if chunk.dim() == 3 else chunk[0] if chunk.dim() == 2 else chunk
        # chunk is (B,Ta,act) or (Ta,act); take first action along the chunk axis
        if chunk.dim() == 3:
            action = chunk[:, 0]
        else:
            action = chunk[0]
        return action if cpu else action.to(self.device)

    def train(self, mode: bool = True):
        self.flow.train(mode)
        if self.model is not None:
            self.model.train(False)   # world model stays frozen/eval
        return self

    def eval(self):
        return self.train(False)


class OrigMppiFlowAgent(nn.Module):
    """The ORIGINAL SimDist MppiPlanner (Gaussian random candidates + CEM
    iterations), but with its base candidate SEQUENCES sourced from the FLOW
    policy (chunks) instead of the world model's MLP policy head. Re-plans every
    env step (exposes only `act`, no `act_chunk`), matching original SimDist.

    The planner's `base_seq_fn` hook is set to sample `num_base` flow chunks and
    use their first H actions as the base action sequences; the planner then adds
    `num_samples - num_base` Gaussian-random sequences and runs its CEM loop,
    scoring every sequence with the (target-adapted) world model's reward+value.
    """

    def __init__(self, flow, model, planner_cfg, device: str = "cuda"):
        super().__init__()
        from rl.simdist_planner import MppiPlanner
        self.flow = flow
        self.model = model
        self.device = device
        self.H = planner_cfg.horizon
        assert self.H <= flow.Tp, f"planner horizon {self.H} > flow Tp {flow.Tp}"
        self.planner = MppiPlanner(model, planner_cfg, flow.action_dim, device=device)
        self.planner.base_seq_fn = self._base_seq

    @torch.no_grad()
    def _sample_base(self, states: torch.Tensor, num: int) -> torch.Tensor:
        """states[B,obs] -> [B,H,num,act] raw base action sequences (flow chunks)."""
        flow = self.flow
        B = states.shape[0]
        st = states.repeat_interleave(num, dim=0)                 # [B*num, obs]
        chunk = flow._sample_full_chunk(flow._normalize_state(st)) # [B*num,Tp,act] normalized
        chunk = flow._denormalize_action(chunk)                    # raw
        seq = chunk[:, : self.H].view(B, num, self.H, flow.action_dim)
        return seq.permute(0, 2, 1, 3).contiguous()                # [B,H,num,act]

    @torch.no_grad()
    def _base_seq(self, states: torch.Tensor, num: int) -> torch.Tensor:
        """Planner hook. plan() passes a single [1,obs] state (return [H,num,act]);
        plan_batch() passes [B,obs] with B=num_proc>1 (return [B,H,num,act])."""
        seq = self._sample_base(states, num)                       # [B,H,num,act]
        return seq[0] if states.shape[0] == 1 else seq

    @torch.no_grad()
    def act(self, obs, *, eval_mode: bool = True, stddev: float = 0.0, cpu: bool = True):
        state = obs["state"].to(self.device).float()
        if state.dim() == 1:
            a = self.planner.plan(state, eval_mode=eval_mode, stddev=stddev)   # [act]
        else:
            a = self.planner.plan_batch(state, eval_mode=eval_mode, stddev=stddev)  # [B,act]
        return a.cpu() if cpu else a

    def train(self, mode: bool = True):
        self.flow.train(mode)
        self.model.train(False)
        return self

    def eval(self):
        return self.train(False)
