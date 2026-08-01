"""Residual off-policy RL on a frozen ADAPTED SimDist base (ResFiT-style).

A thin variant of `rl/residual_rl_agent.py::ResidualRLAgent` whose frozen base
policy is a TARGET-adapted `SimDistAgent` (a world model executed via MPPI, see
CLAUDE.md §17/§24) instead of a cheap deterministic BC MLP. Everything else — the
zero-init residual actor `a = clip(a_base + action_scale*tanh(res(s,a_base)),-1,1)`,
the LayerNorm REDQ critic ensemble, TD3 target smoothing, `Q∈[0,1]` clamp, z-score
state, RLPD online+offline mix — is inherited verbatim from `ResidualRLAgent`.

Two changes are forced by the SimDist base:

1. **`_base_action`** — a `SimDistAgent` has no `__call__`; query it via
   `base.act({"state": s}, eval_mode=True, stddev=0, cpu=False)` (runs MPPI,
   handles single + batched). Used only at `act()` / eval time.

2. **`update(batch, ...)`** consumes a **rela `Batch`** (n-step) whose base
   actions are **stored in `obs`/`next_obs`** under the key `base_action` — NOT
   recomputed. Re-planning MPPI 256×2 times per critic update is infeasible
   (num_samples×iters×H model forwards each), so — exactly as ResFiT stores
   `observation.base_action` in the replay — the trainer computes the base action
   ONCE per collected env step and stores it; the update just reads it. The
   Bellman target uses the n-step bootstrap `batch.bootstrap` (= gamma^n, already
   0 at a success terminal under end_on_success), so there is no separate
   gamma*(1-done) factor.
"""

import torch

from rl.residual_rl_agent import ResidualRLAgent, ResidualRLConfig


# The config is identical to the faithful ResFiT residual RL; base_policy_path is
# unused here (the SimDist base is loaded + attached by the trainer).
ResidualSimDistConfig = ResidualRLConfig


class ResidualSimDistAgent(ResidualRLAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # current residual bound (progressive clipping ramps this from 0 to
        # action_scale; the trainer sets it each step + before eval). Default =
        # full action_scale (no ramp). A plain attribute so it survives pickling
        # into the mp-eval workers.
        self.clip_scale = float(self.cfg.action_scale)
        # single-step discount for stage-1 latent Dyna bootstrap (trainer overrides
        # with its cfg.discount; default 0.99). Plain attr => survives mp pickling.
        self.cfg_discount = 0.99
        # frozen SGFT value source (§10). List-held so it is NOT a submodule (kept
        # out of state_dict; the trainer re-attaches by path). Empty unless
        # attach_value_source() is called (cfg.sgft_shaping gates the trainer).
        self._v_src: list = []

    def attach_value_source(self, agent):
        """Attach a frozen source QAgent whose value(obs) supplies V_src for the
        SGFT potential shaping in update(). agent.value expects {"state": s}."""
        self._v_src = [agent]

    def _phase_scale(self, s_raw: torch.Tensor):
        """Per-sample residual-scale multiplier m(s) ∈ [gate_min, gate_max] from the
        needle-ring distance. Contact (d small) → gate_max (big correction where the
        base fails); free space (d large) → gate_min (small, stay on base). Returns
        a [B,1] tensor, or None when phase_gate is off (⇒ identity)."""
        cfg = self.cfg
        if not cfg.phase_gate:
            return None
        needle = s_raw[:, cfg.phase_needle_lo:cfg.phase_needle_lo + 3]
        ring = s_raw[:, cfg.phase_ring_lo:cfg.phase_ring_lo + 3]
        d = (needle - ring).norm(dim=-1, keepdim=True)  # [B,1]
        gate = torch.sigmoid((cfg.phase_gate_mid - d) / cfg.phase_gate_width)
        return cfg.phase_gate_min + (cfg.phase_gate_max - cfg.phase_gate_min) * gate

    def _apply_gate(self, res: torch.Tensor, m, base_bound: float) -> torch.Tensor:
        """Scale the residual by the per-sample phase multiplier m and clamp to the
        (per-sample) bound m·base_bound. m=None ⇒ plain clamp to base_bound (identity
        to the pre-phase-gate path)."""
        if m is None:
            return res.clamp(-base_bound, base_bound)
        res = res * m
        bound = base_bound * m
        return torch.maximum(torch.minimum(res, bound), -bound)

    def _clamp_q_shift(self, q: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Clamp a shaped Q' to [q_min-shift, q_max-shift] (mirrors
        QAgent._clamp_q(q, shift)). shift = coef*V_src(x) tracks the per-state
        shaped range Q'(x)=Q(x)-coef*V_src(x)."""
        lo = self.cfg.q_target_min - shift
        hi = self.cfg.q_target_max - shift
        return torch.max(torch.min(q, hi), lo)

    # --------------------------------------------------------------------- base
    def attach_base_policy(self, module):
        """Attach the frozen SimDist base. When latent_features=1 the base's world
        model doubles as the residual's frozen featurizer (z = encode(s)), so wire
        its encoder in here and verify its width matches the nets we already built
        (the trainer constructs the agent before the base exists)."""
        super().attach_base_policy(module)
        if self.cfg.latent_features:
            wm = getattr(module, "model", None)
            assert wm is not None, "latent_features=1 needs a SimDist base with .model"
            assert not getattr(wm, "raw_state", False), (
                "latent_features=1 is meaningless with a raw-state WorldModel "
                "(encoder_type='identity': encode() is just a z-score)")
            with torch.no_grad():
                probe = wm.encode(torch.zeros(1, self.obs_dim, device=self.device))
            got = int(probe.shape[-1])
            assert got == self.feat_dim, (
                f"residual.latent_dim={self.feat_dim} but the base encoder emits "
                f"{got}-d latents — set --residual.latent_dim {got}")
            self.attach_encoder(wm.encode)

    @torch.no_grad()
    def _base_action(self, state_raw: torch.Tensor) -> torch.Tensor:
        """Deterministic MPPI base action a_base(s) in [-1,1] from RAW state.

        The base is a SimDistAgent (world model + MPPI planner); it exposes
        `act(obs, eval_mode, stddev, cpu)` rather than a plain forward. state_raw
        is [B, obs] (the trainer/act always pass a 2-D batch), so SimDistAgent.act
        routes to plan_batch and returns [B, act]."""
        assert self._base_policy, "base policy not attached"
        a = self._base_policy[0].act(
            {"state": state_raw}, eval_mode=True, stddev=0.0, cpu=False)
        if a.dim() == 1:  # SimDistAgent squeezes a single (dim==1) obs
            a = a.unsqueeze(0)
        return a.to(state_raw.device).float().clamp(-1.0, 1.0)

    # --------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], *, eval_mode: bool = False,
            stddev: float = 0.0, cpu: bool = True) -> torch.Tensor:
        """Combined action = clip(base + residual). If `obs` already carries a
        precomputed `base_action` (the collection loop reuses the plan it ran for
        the previous step's next_obs), use it instead of re-planning MPPI; else
        plan (the eval path, where the env obs has no base_action)."""
        state = obs["state"]
        single = state.dim() == 1
        s_raw = (state.view(1, -1) if single else state).to(self.device).float()
        if "base_action" in obs and obs["base_action"] is not None:
            ba = obs["base_action"]
            base_a = (ba.view(1, -1) if ba.dim() == 1 else ba).to(self.device).float()
        else:
            base_a = self._base_action(s_raw)
        res = self.actor(self._norm(s_raw), base_a)
        if (not eval_mode) and stddev > 0:
            res = res + torch.randn_like(res) * stddev
        # progressive-clipping bound (== action_scale when the ramp is off/complete);
        # applied in BOTH train and eval so eval reflects the current residual bound
        # (early ramp => residual ~0 => action == base => eval starts at the base score).
        # Phase gate (if on) scales the residual + bound per-sample by needle-ring dist.
        res = self._apply_gate(res, self._phase_scale(s_raw), self.clip_scale)
        action = (base_a + res).clamp(-1.0, 1.0)
        action = action[0] if single else action
        return action.cpu() if cpu else action

    # ------------------------------------------------------------------ update
    def update(self, batch, update_actor: bool) -> dict[str, float]:
        """One TD3 step on a MIXED (online + offline-demo) rela n-step batch.

        `batch` is a rl.replay.Batch: obs/next_obs each carry {state, base_action};
        action["action"] is the EXECUTED action; reward is the n-step discounted
        sum; bootstrap is gamma^n (0 at a success terminal within the window)."""
        cfg = self.cfg
        s_raw = batch.obs["state"].to(self.device).float()
        s2_raw = batch.next_obs["state"].to(self.device).float()
        base_a = batch.obs["base_action"].to(self.device).float()
        base_a2 = batch.next_obs["base_action"].to(self.device).float()
        s = self._norm(s_raw)
        s2 = self._norm(s2_raw)
        a = batch.action["action"].to(self.device).float()
        r = batch.reward.to(self.device).float().view(-1)
        disc = batch.bootstrap.to(self.device).float().view(-1)  # gamma^n (0 at terminal)
        metrics: dict[str, float] = {}

        # ---- critic: TD3 target with target-policy smoothing on the residual ----
        with torch.no_grad():
            res2 = self.actor_target(s2, base_a2)
            noise = (torch.randn_like(res2) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip)
            res2 = self._apply_gate(res2 + noise, self._phase_scale(s2_raw), self.clip_scale)
            next_action = (base_a2 + res2).clamp(-1.0, 1.0)
            q_next = self._min_q(self.critic_target(s2, next_action))  # [B]
            if cfg.sgft_shaping and self._v_src:
                # SGFT potential shaping. The n-step potential telescopes to
                #   r' = r + coef*(gamma^n*V_src(s_{t+n}) - V_src(s_t)),
                # disc = gamma^n (0 at a success terminal ⇒ V_src(s') drops out,
                # correct). The learned Q'(x)=Q(x)-coef*V_src(x) is mostly
                # negative, so clamp with a per-state shift = coef*V_src (bound
                # becomes [q_min-shift, q_max-shift]); clamp the bootstrap Q'(s2)
                # with V_src(s2) and the target y with V_src(s).
                src = self._v_src[0]
                v_s = src.value({"state": s_raw}).view(-1)
                v_s2 = src.value({"state": s2_raw}).view(-1)
                coef = cfg.sgft_coef
                r_eff = r + coef * (disc * v_s2 - v_s)
                if cfg.clamp_q_target:
                    q_next = self._clamp_q_shift(q_next, coef * v_s2)
                y = r_eff + disc * q_next
                if cfg.clamp_q_target:
                    y = self._clamp_q_shift(y, coef * v_s)
                metrics["sgft/v_source"] = v_s.mean().item()
                metrics["sgft/shaping"] = (coef * (disc * v_s2 - v_s)).mean().item()
            else:
                y = r + disc * q_next
                y = self._clamp_q(y)

        q = self.critic(s, a)  # [K, B]
        critic_loss = ((q - y.unsqueeze(0)) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()
        # ResFiT Algorithm 1 line 18: soft-update the critic target after EVERY
        # critic update (i.e. once per UTD iteration).
        self._soft_update_pair(self.critic, self.critic_target)

        metrics["residual/critic_loss"] = critic_loss.item()
        metrics["residual/q_mean"] = q.mean().item()
        metrics["residual/td_target"] = y.mean().item()
        metrics["data/batch_R"] = r.mean().item()

        # ---- residual actor (ResFiT Algorithm 1 lines 19-21) ----
        # `update_actor` is set by the trainer only on the last UTD iteration, so
        # the actor is updated once per env step (ResFiT actor_updates_per_iteration=1).
        if update_actor:
            res = self._apply_gate(self.actor(s, base_a), self._phase_scale(s_raw), self.clip_scale)
            a_pi = (base_a + res).clamp(-1.0, 1.0)
            # Line 20 policy improvement. "mean" = ResFiT-faithful ensemble mean
            # over all heads; "min" = pessimistic min over a random subset (default;
            # resists OOD Q-overestimation that otherwise collapses the residual on
            # this task — see ResidualRLConfig.actor_q_agg).
            q_all = self.critic(s, a_pi)  # [K, B]
            q_pi = q_all.mean(0) if cfg.actor_q_agg == "mean" else self._min_q(q_all)
            actor_loss = -q_pi.mean()
            if cfg.residual_l2_coef > 0:
                l2 = (res ** 2).sum(-1).mean()
                actor_loss = actor_loss + cfg.residual_l2_coef * l2
                metrics["residual/delta_l2"] = l2.item()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
            self.actor_opt.step()
            self._soft_update_pair(self.actor, self.actor_target)  # line 21
            metrics["residual/actor_loss"] = actor_loss.item()
            metrics["residual/delta_abs"] = res.abs().mean().item()

        return metrics

    @torch.no_grad()
    def _imagine_synthetic(self, s_raw: torch.Tensor, base_a: torch.Tensor,
                           exact_base: bool = False, reward_thresh: float = None,
                           horizon: int = 1):
        """Stage-1 Dyna: from a REAL start state, imagine a K-step latent rollout
        under the CURRENT combined policy and learn Q(z0,a0) via K-step value
        expansion — entirely in latent (no decode).

        base actions along the rollout: ONE MPPI solve on z0 returns the whole
        committed sequence [base_a_0..base_a_{K-1}] (MPPI rolls H steps in latent
        internally, so the full plan is free). The imagined path diverges from
        MPPI's own base-rollout only by the small residual (action_scale 0.1), so
        base_a_t is a good action at the (residual-perturbed) z_t. Falls back to
        base_a_0 for every step if exact_base=0 (cheap; but drifts for K>1 — only
        sensible at K=1).

        rollout:  z_0=encode(s); for t: a_t=clip(base_a_t+actor(z_t,base_a_t));
                  z_{t+1}=next(z_t,a_t); r_hat_t=1[V_src(z_{t+1})>c].
        target (anchor-terminated K-step expansion, mimics end_on_success):
            y = Σ_t γ^t·r_hat_t·(still-alive) + γ^K·(still-alive)·minQ_target(z_K,a_K)
        reward_thresh None => r_hat≡0 (pure K-step bootstrap y=γ^K·minQ).

        Returns (z_0, a_0, y, frac_anchor): the critic MSE is on the FIRST
        (z_0,a_0); the K-step path only shapes its target."""
        assert self._encoder, "stage-1 dyna needs the latent encoder (latent_features=1)"
        base = self._base_policy[0]
        wm = base.model
        K = max(1, int(horizon))
        g = self.cfg_discount

        z0 = self._encoder[0](s_raw)                         # [B, zdim]
        # whole base-action sequence for the imagined rollout (1 MPPI solve)
        if exact_base and K > 1:
            base_seq = base.plan_seq_from_latent(z0).clamp(-1.0, 1.0)  # [B, H, act]
            H = base_seq.shape[1]
        else:
            base_seq = None

        def base_at(t, z):
            if base_seq is not None:
                return base_seq[:, min(t, base_seq.shape[1] - 1)]
            if exact_base:
                return base.plan_from_latent(z).clamp(-1.0, 1.0)
            return base_a                                    # approx (K=1 regime)

        z = z0
        z0_out = a0_out = None
        alive = torch.ones(z0.shape[0], device=z0.device)    # 1 until an anchor fires
        acc = torch.zeros(z0.shape[0], device=z0.device)     # Σ γ^t r_hat_t (alive-gated)
        anchored = torch.zeros(z0.shape[0], device=z0.device)
        for t in range(K):
            ba = base_at(t, z)
            res = self.actor(z, ba).clamp(-self.clip_scale, self.clip_scale)
            a = (ba + res).clamp(-1.0, 1.0)
            if t == 0:
                z0_out, a0_out = z, a                        # critic trains on (z0,a0)
            z_next = wm.next(z, a)
            if reward_thresh is not None:
                v_next = wm.predict_value(z_next, target=False).view(-1)
                r_hat = (v_next > reward_thresh).float()
                # contribute γ^t only while still alive; then mark anchored
                acc = acc + alive * (g ** t) * r_hat
                anchored = torch.maximum(anchored, alive * r_hat)
                alive = alive * (1.0 - r_hat)
            z = z_next
        # bootstrap at z_K with the base action there, only where never anchored
        ba_K = base_at(K, z)
        res_K = self.actor_target(z, ba_K).clamp(-self.clip_scale, self.clip_scale)
        a_K = (ba_K + res_K).clamp(-1.0, 1.0)
        q_K = self._min_q(self.critic_target(z, a_K))        # [B]
        y = self._clamp_q(acc + alive * (g ** K) * q_K)
        frac_anchor = anchored.mean().item() if reward_thresh is not None else 0.0
        return z0_out, a0_out, y, frac_anchor

    def update_critic_synthetic(self, s_raw: torch.Tensor, base_a: torch.Tensor,
                                exact_base: bool = False,
                                reward_thresh: float = None, horizon: int = 1) -> dict:
        """One critic-only TD step on imagined latent transitions. Does NOT touch
        the actor (the actor learns only from real Q). Densifies value propagation
        in latent space; the real batch anchors the reward scale. reward_thresh
        None => r_hat=0 (pure bootstrap); float => r_hat=1[V_src(z')>thresh].
        horizon = K-step imagined rollout length (value expansion)."""
        z, a, y, frac_anchor = self._imagine_synthetic(
            s_raw.to(self.device).float(), base_a.to(self.device).float(),
            exact_base=exact_base, reward_thresh=reward_thresh, horizon=horizon)
        q = self.critic(z, a)                                # [K, B]
        loss = ((q - y.unsqueeze(0)) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()
        self._soft_update_pair(self.critic, self.critic_target)
        return {"dyna/critic_loss": loss.item(),
                "dyna/q_mean": q.mean().item(),
                "dyna/td_target": y.mean().item(),
                "dyna/frac_anchor": frac_anchor}

    def _soft_update_pair(self, src, tgt):
        tau = self.cfg.tau
        with torch.no_grad():
            for p, pt in zip(src.parameters(), tgt.parameters()):
                pt.mul_(1 - tau).add_(p, alpha=tau)
