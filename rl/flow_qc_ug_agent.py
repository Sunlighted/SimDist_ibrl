"""Flow-QC-UG: online model-free RL with a flow-policy actor + Q-chunking critic
+ Universal-Guidance policy improvement.

Motivation (see docs/flow_qc_ug.md). SimDist/MPPI on this task caps at ~81%
because it argmaxes candidate action chunks against a LEARNED DYNAMICS model and
exploits the model's errors; every diagnosis says the ceiling = policy quality
and the value is a liability under hard argmax. This agent is MODEL-FREE: the
policy is improved by Universal Guidance on a chunk-Q that is trained by TD on
TRUE env transitions (no dynamics model to exploit) — the same reason
IBRL/RLPD reach ~100%. It is generalized policy iteration with a flow actor:

  * policy evaluation : chunk-Q  Q(s, a_{0:h})  trained by chunk-TD (gamma^h),
                        min-ensemble + LayerNorm (RLPD), target clamped to [0,1]
                        (sparse 0/1 + end_on_success => true Q in [0,1]).
  * policy improvement: (a) collect with flow + UG (∇_a Q steers each ODE step
                        toward high-Q chunks, Bansal 2023 forward guidance) so
                        the rollout data is Q-improved, AND (b) advantage-
                        weighted flow-matching on the replay so the flow's own
                        proposal moves toward high-Q chunks (not just BC =
                        amortization, which caps at the teacher).

Q-chunk length h (TD backup + UG signal) is decoupled from the execution horizon
Ta (how many actions run before re-planning): h>1 gives coherent exploration +
fast value backup; Ta=1 (closed-loop) is safest on a dynamics gap, but on SOURCE
(no gap) Ta>1 is fine. Both are config knobs.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.rebrac_agent import EnsembleQ


@dataclass
class FlowQCUGConfig:
    # --- Q-chunking critic ---
    q_chunk_len: int = 8            # h: chunk length for Q + gamma^h TD backup (<= flow.Tp)
    critic_hidden_dim: int = 512
    critic_n_layer: int = 3
    critic_layer_norm: int = 1
    num_critics: int = 5            # RED-Q style min-ensemble
    gamma: float = 0.99
    tau: float = 0.005              # target soft-update rate
    clamp_q: int = 1                # clamp TD target to [0,1] (sparse + end_on_success)
    td_next_use_ug: int = 0         # next-action for TD: 0=flow-only (cheap), 1=flow+UG
    # --- execution ---
    exec_ta: int = 1                # actions executed per re-plan at rollout/eval (<= flow.Ta)
    # --- Universal Guidance (rollout action selection) ---
    use_ug: int = 1
    steering_coeff: float = 0.3     # UG step on the unit-normalized gradient
    grad_normalize: int = 1
    guidance_start_step: int = 1    # skip UG on step 0 (x is pure noise)
    stop_grad_velocity: int = 1     # don't backprop the flow velocity through UG (cheap)
    clamp_x: int = 1
    # --- actor (policy improvement) ---
    actor_awr: int = 1              # advantage-weighted flow-matching (0 = plain BC)
    awr_temp: float = 1.0           # temperature on exp(adv/temp)
    awr_adv_clip: float = 20.0      # clip adv/temp before exp
    awr_weight_clip: float = 20.0   # clip the exp weight
    # --- optim ---
    flow_lr: float = 1e-4
    critic_lr: float = 3e-4
    num_critic_update: int = 2      # critic updates per actor update (UTD-ish)


class FlowQCUGAgent(nn.Module):
    def __init__(self, flow, obs_dim: int, action_dim: int,
                 cfg: FlowQCUGConfig = FlowQCUGConfig(), device: str = "cuda"):
        super().__init__()
        self.flow = flow                         # StateFlowPolicy (online weights trained)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = device
        self.h = min(cfg.q_chunk_len, flow.Tp)   # Q-chunk length
        assert cfg.exec_ta <= flow.Ta, f"exec_ta {cfg.exec_ta} > flow.Ta {flow.Ta}"

        # chunk-Q over concat(state, flatten(a_{0:h}))
        self.critic = EnsembleQ(
            obs_dim, action_dim * self.h, cfg.critic_hidden_dim,
            cfg.critic_n_layer, bool(cfg.critic_layer_norm), cfg.num_critics)
        import copy
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.flow_opt = torch.optim.AdamW(self.flow.parameters(), lr=cfg.flow_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    # =============================================================== Q helpers
    def _chunk_q(self, state_raw, chunk_raw_h, target=False):
        """min-ensemble chunk-Q. state[B,obs], chunk[B,h,act] RAW -> [B]."""
        flat = chunk_raw_h.reshape(chunk_raw_h.shape[0], -1)
        net = self.critic_target if target else self.critic
        return net(state_raw, flat).min(0)[0]                # min over ensemble -> [B]

    # =============================================================== sampling
    @torch.no_grad()
    def _flow_chunk_raw(self, state_raw):
        """Flow-only sample (no UG). state[B,obs] RAW -> full chunk [B,Tp,act] RAW."""
        z = self.flow._sample_full_chunk(self.flow._normalize_state(state_raw))
        return self.flow._denormalize_action(z)

    def _ug_chunk_raw(self, state_raw):
        """flow + ∇_a Q(chunk) UG-steered Euler ODE. RAW -> full chunk [B,Tp,act] RAW."""
        flow, cfg = self.flow, self.cfg
        state_n = flow._normalize_state(state_raw)
        B = state_n.shape[0]
        x = torch.randn(B, flow.Tp, self.action_dim, device=state_raw.device, dtype=state_n.dtype)
        N = flow.cfg.num_inference_steps
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
                        a_hat = x_req + (1.0 - t_val) * flow.forward_velocity(x_req, t, state_n)
                    a_raw = flow._denormalize_action(a_hat)          # [B,Tp,act]
                    q = self._chunk_q(state_raw, a_raw[:, : self.h]) # [B]
                    grad = torch.autograd.grad(q.sum(), x_req)[0]
                with torch.no_grad():
                    if cfg.grad_normalize:
                        gn = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1)
                    else:
                        gn = grad
                    x = x + coeff * gn
            x = x + v * dt
            if cfg.clamp_x:
                x = x.clamp(-1.0, 1.0)
        return flow._denormalize_action(x.detach())

    # =============================================================== act
    @torch.no_grad()
    def act_chunk(self, obs, *, eval_mode: bool = True):
        state = obs["state"].to(self.device).float()
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
        if self.cfg.use_ug:
            chunk_full = self._ug_chunk_raw(state)               # [B,Tp,act] RAW
        else:
            chunk_full = self._flow_chunk_raw(state)
        chunk = chunk_full[:, : self.cfg.exec_ta].contiguous()
        if single:
            chunk = chunk.squeeze(0)
        return chunk.cpu()

    @torch.no_grad()
    def act(self, obs, *, eval_mode: bool = True, stddev: float = 0.0, cpu: bool = True):
        chunk = self.act_chunk(obs, eval_mode=eval_mode)
        action = chunk[0] if chunk.dim() == 2 else chunk[:, 0]
        if stddev > 0:
            action = (action + stddev * torch.randn_like(action)).clamp(-1.0, 1.0)
        return action if cpu else action.to(self.device)

    # =============================================================== updates
    def update_critic(self, states, actions, rewards):
        """chunk-TD. states[B,H+1,obs], actions[B,H,act], rewards[B,H] (H>=h)."""
        cfg, h, g = self.cfg, self.h, self.cfg.gamma
        s_t = states[:, 0]
        a_chunk = actions[:, :h]
        s_next = states[:, h]                                 # state after h steps
        # discounted h-step return
        disc = (g ** torch.arange(h, device=states.device)).view(1, h)
        R = (rewards[:, :h] * disc).sum(1)                    # [B]
        # sparse 0/1 + end_on_success: a success terminal (reward 1) inside the
        # window ends the episode -> don't bootstrap past it.
        boot_mask = (rewards[:, :h].sum(1) < 0.5).float()     # 1 if no success in window
        with torch.no_grad():
            if cfg.td_next_use_ug and cfg.use_ug:
                nxt = self._ug_chunk_raw(s_next)[:, :h]
            else:
                nxt = self._flow_chunk_raw(s_next)[:, :h]
            next_q = self._chunk_q(s_next, nxt, target=True)  # [B]
            y = R + (g ** h) * boot_mask * next_q
            if cfg.clamp_q:
                y = y.clamp(0.0, 1.0)
        q_all = self.critic(s_t, a_chunk.reshape(a_chunk.shape[0], -1))  # [E,B]
        loss = F.mse_loss(q_all, y.unsqueeze(0).expand_as(q_all))
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        return {"critic_loss": loss.item(), "td_target_mean": y.mean().item(),
                "q_mean": q_all.mean().item(), "boot_frac": boot_mask.mean().item()}

    def update_actor(self, states, actions):
        """Advantage-weighted flow-matching. states[B,H+1,obs], actions[B,H,act]."""
        cfg, flow, Tp, h = self.cfg, self.flow, self.flow.Tp, self.h
        s_raw = states[:, 0]
        a_raw = actions[:, :Tp]                               # data chunk (Tp long)
        s_n = flow._normalize_state(s_raw)
        a_n = flow._normalize_action(a_raw)
        B = a_n.shape[0]
        t = torch.rand(B, device=a_n.device)
        z = torch.randn_like(a_n)
        tv = t.view(B, 1, 1)
        x_t = (1.0 - tv) * z + tv * a_n
        v_tgt = a_n - z
        v_pred = flow.forward_velocity(x_t, t, s_n)
        per = ((v_pred - v_tgt) ** 2).mean(dim=[1, 2])        # [B] per-sample FM loss

        info = {}
        if cfg.actor_awr:
            with torch.no_grad():
                q_data = self._chunk_q(s_raw, a_raw[:, :h])   # Q(s, data chunk)
                base = self._flow_chunk_raw(s_raw)[:, :h]     # current policy chunk
                v_s = self._chunk_q(s_raw, base)              # V(s) ~ Q(s, pi(s))
                adv = q_data - v_s
                w = torch.exp((adv / cfg.awr_temp).clamp(-cfg.awr_adv_clip, cfg.awr_adv_clip))
                w = w.clamp(max=cfg.awr_weight_clip)
                w = w / w.mean().clamp_min(1e-8)              # normalize to mean 1
            loss = (w * per).mean()
            info.update(adv_mean=adv.mean().item(), w_max=w.max().item())
        else:
            loss = per.mean()

        self.flow_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.flow_opt.step()
        if hasattr(flow, "update_ema"):
            flow.update_ema()
        info.update(actor_loss=loss.item(), fm_loss=per.mean().item())
        return info

    def soft_update_target(self):
        tau = self.cfg.tau
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1 - tau).add_(tau * p)

    # =============================================================== misc
    def train(self, mode: bool = True):
        self.flow.train(mode)
        self.critic.train(mode)
        self.critic_target.train(False)
        return self

    def eval(self):
        return self.train(False)
