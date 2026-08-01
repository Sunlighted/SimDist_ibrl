"""Flow BC policy steered by a frozen Q-function via Universal Guidance.

Variant of rl/flow_ug_agent.py where the guidance signal is a Q-function
Q(s, a_0) evaluated on the FIRST action of the denoised chunk — NOT a world-model
rollout return. Motivation (2026-07-06 experiment): if the Q is a well-trained
TARGET expert critic (threading_ibrl_target_snapshots_rlprior_seed1), then
forward-UG ∇_a Q steers a SOURCE-trained flow toward high-Q (target-optimal)
actions at inference, testing (1) whether flow+UG is strong enough when Q is
good, and (2) which source flow (all-data vs expert-only) transfers better.

Q-guidance needs only the first action (Q(s,a) already encodes the full return),
so — unlike the H-step world-model rollout guidance — there is no dynamics model
and no horizon. With Ta=1 (re-query every step) the steered first action is what
gets executed.

Guidance (forward UG, Bansal et al. 2023): at each Euler ODE step form the
Tweedie denoised chunk â = x + (1-t)·v (v held fixed when stop_grad_velocity),
denormalize, and push x up ∇_x Q(s, â_raw[:,0]).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FlowQUGConfig:
    use_ug: int = 0                 # 0 = flow-only; 1 = Q-UG steered
    steering_coeff: float = 0.3     # UG step size on the unit-normalized grad
    ug_num_steps: int = 0           # ODE steps; 0 = flow's own num_inference_steps
    grad_normalize: int = 1         # per-sample unit-norm the guidance gradient
    guidance_start_step: int = 1    # skip UG on step 0 (x is pure noise)
    clamp_x: int = 1                # clamp the normalized action iterate to [-1,1]
    stop_grad_velocity: int = 1     # don't backprop through the flow velocity (cheap)


class FlowQUGAgent(nn.Module):
    def __init__(self, flow, q_agent=None, cfg: FlowQUGConfig = FlowQUGConfig(),
                 device: str = "cuda"):
        super().__init__()
        self.flow = flow                 # StateFlowPolicy (EMA applied at load)
        self.q_agent = q_agent           # frozen QAgent (state-based critic) or None
        self.cfg = cfg
        self.device = device
        if cfg.use_ug:
            assert q_agent is not None, "use_ug=1 needs a Q agent"

    # --------------------------------------------------------------- Q value
    def _q_value(self, state_raw: torch.Tensor, a0_raw: torch.Tensor) -> torch.Tensor:
        """min-ensemble Q(s, a_0) on the RAW state + first RAW action. [B]."""
        return self.q_agent.critic.forward_k(state_raw, a0_raw).min(-1)[0]

    def _sample_chunk_qug(self, state_raw: torch.Tensor) -> torch.Tensor:
        """Q-UG-steered Euler ODE. state_raw[B,obs] -> normalized chunk [B,Tp,act]."""
        flow, cfg = self.flow, self.cfg
        state_n = flow._normalize_state(state_raw)
        B = state_n.shape[0]
        x = torch.randn(B, flow.Tp, flow.action_dim,
                        device=state_raw.device, dtype=state_n.dtype)
        N = cfg.ug_num_steps or flow.cfg.num_inference_steps
        dt = 1.0 / N
        coeff = cfg.steering_coeff
        for i in range(N):
            t_val = i * dt
            t = torch.full((B,), t_val, device=state_raw.device, dtype=state_n.dtype)
            with torch.no_grad():
                v = flow.forward_velocity(x, t, state_n)
            if coeff > 0 and i >= cfg.guidance_start_step:
                with torch.enable_grad():
                    x_req = x.detach().requires_grad_(True)
                    if cfg.stop_grad_velocity:
                        a_hat = x_req + (1.0 - t_val) * v
                    else:
                        v_g = flow.forward_velocity(x_req, t, state_n)
                        a_hat = x_req + (1.0 - t_val) * v_g
                    a_raw = flow._denormalize_action(a_hat)        # [B,Tp,act]
                    q = self._q_value(state_raw, a_raw[:, 0])      # [B]
                    grad = torch.autograd.grad(q.sum(), x_req)[0]  # [B,Tp,act]
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

    # --------------------------------------------------------------- act
    @torch.no_grad()
    def act_chunk(self, obs, *, eval_mode: bool = True):
        assert eval_mode
        if not self.cfg.use_ug:
            return self.flow.act_chunk(obs, eval_mode=eval_mode)   # flow-only

        state = obs["state"].to(self.device).float()
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
        chunk_full = self._sample_chunk_qug(state)                 # [B,Tp,act] normalized
        chunk = chunk_full[:, : self.flow.Ta].contiguous()
        chunk = self.flow._denormalize_action(chunk)               # raw scale
        if single:
            chunk = chunk.squeeze(0)
        return chunk.cpu()

    @torch.no_grad()
    def act(self, obs, *, eval_mode: bool = True, stddev: float = 0.0, cpu: bool = True):
        chunk = self.act_chunk(obs, eval_mode=eval_mode)
        action = chunk[:, 0] if chunk.dim() == 3 else chunk[0]
        return action if cpu else action.to(self.device)

    def train(self, mode: bool = True):
        self.flow.train(mode)
        if self.q_agent is not None:
            self.q_agent.train(False)
        return self

    def eval(self):
        return self.train(False)
