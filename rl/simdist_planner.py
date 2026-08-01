"""MPPI / CEM planner over the latent world model (SimDist execution mechanism).

Ported from simdist/simdist/control/mppi.py to PyTorch, state-based, no command
conditioning. At each control step:
  1. encode the current state -> z0
  2. sample `num_samples` action sequences of length H ~ N(mean, std), plus a
     `mixture_coef` fraction seeded from the policy head's own rollout (the base
     proposal), and roll them through the latent dynamics
  3. score each sequence by return = sum_t gamma^t r_hat_t + gamma^H V_hat(z_H)
  4. take the top-k elites, reweight (mean, std) by exp(temperature * (R - max))
  5. iterate; return mean[0] as the action to execute.

Stateless by default (mean re-initialized to 0 every call) so `act()` carries no
cross-step state and is safe to pickle into multi-process eval workers. Optional
`warm_start` keeps a per-instance mean rolled forward (single-process only).
"""

from dataclasses import dataclass

import torch

from rl.world_model import WorldModel


@dataclass
class PlannerConfig:
    horizon: int = 5
    num_samples: int = 512
    num_elites: int = 64
    iterations: int = 6
    # LEAN ON THE POLICY PROPOSAL. On sparse Threading the value V_src is ~flat on
    # the expert success-manifold, so value-ranking alone can't distinguish
    # candidates; with the TD-MPC2 defaults (mixture 0.25, init_std 0.5) the 384
    # random trajectories diluted the 128 good policy proposals and MPPI scored
    # BELOW the bare policy (0.0 vs 0.10 source). Trusting the proposal (most
    # candidates from the policy rollout + little noise) lets MPPI REFINE it and
    # beat the policy (0.20 vs 0.10). Verified on a 30k ckpt 2026-06-05.
    mixture_coef: float = 0.8      # fraction of samples seeded from the policy head
    # Optional EXACT count of policy-seeded ("noised base policy") candidates.
    # >= 0 overrides mixture_coef (which truncates: int(num_samples*coef) can be
    # off-by-one, e.g. 100/350 -> 99). With it set, num_base = this value and
    # num_rand = num_samples - this, so a recipe can specify the two candidate
    # batch sizes as exact counts. -1 = use mixture_coef (legacy / default).
    num_base_override: int = -1
    temperature: float = 2.0
    momentum: float = 0.1
    init_std: float = 0.15
    min_std: float = 0.03
    discount: float = 0.99
    # Std of the Gaussian noise added to the policy-seeded ("noised base policy")
    # trajectories (SimDist Table III "Base policy action standard deviation").
    # The policy head is deterministic here, so without noise all num_base base
    # trajectories are IDENTICAL (a single proposal copied N times); this noise
    # spreads them into a cloud around the proposal. 0 = deterministic base
    # (legacy behavior).
    base_policy_std: float = 0.0
    # If 0, drop the reward term from the planning return (return = gamma^H * V(z_H)
    # only). On tasks where the reward head collapsed to ~0 (rare-success MSE),
    # the reward term is dead noise; ranking purely by the terminal value can be
    # cleaner. 1 = standard SimDist return (sum gamma^t r_hat + gamma^H V).
    use_reward: int = 1
    # DYNAMICS-ENSEMBLE geometric-disagreement term (MOPO/PETS state-space; §27/§31).
    # Requires world_model.num_dynamics_heads>1. The return gets
    # `ret -= coef * sum_t gamma^t * disagree_t`, disagree_t = cross-head std of the
    # predicted next latent (averaged over dims). SIGN sets the mode:
    #   coef > 0  => PESSIMISM (avoid unconstrained regions; §31 verdict: no benefit).
    #   coef < 0  => EXPLORATION BONUS (ret += |coef|*disagree => argmax SEEKS the
    #     model's most-uncertain regions). Used at DATA-COLLECTION time (Plan2Explore
    #     / MAX-style intrinsic reward) to fill dynamics-coverage holes faster —
    #     task-agnostic (no contact/task prior; the ensemble self-reports where it is
    #     uncertain, which on insertion is automatically the contact phase). Keep 0 at
    #     eval (pure exploit). 0 = OFF (ensemble mean still rolls; term inactive).
    dynamics_disagree_coef: float = 0.0
    # If 1, initialize the candidate-sampling MEAN from the DIRECT policy action
    # policy_action(state) (NOT the decode-bridge rollout), so the candidate cloud
    # is centered on the precise policy action and value-MPPI only REFINES it.
    # Decisive test of "does value add anything on top of the policy". H=1 only
    # (a single action; for H>1 the same action is broadcast over the horizon).
    seed_mean_from_policy: int = 0
    warm_start: int = 0            # keep mean across steps (single-process only)


def apply_original_recipe(p: PlannerConfig) -> PlannerConfig:
    """Mutate a PlannerConfig IN PLACE to the ORIGINAL SimDist MPPI recipe
    (simdist/config/control/mppi.yaml): a BROAD search — low policy mixture,
    large init_std, more iterations, soft temperature — instead of the
    ibrl-retuned "trust-the-BC-proposal" defaults (mixture 0.8 / init_std 0.15)
    which cap performance at the frozen policy. `horizon` / `discount` /
    `use_reward` are NOT touched (they are dynamics-/task-dependent: the dynamics
    head was distilled for a fixed rollout horizon, so planning deeper queries an
    untrained dynamics). Single source of truth shared by train_simdist_adapt.py
    and evaluate/eval.py."""
    p.num_samples = 350           # 100 noised-base + 250 random candidates
    p.num_base_override = 100     # exact base-policy candidate count
    p.mixture_coef = 100 / 350    # bookkeeping only (override takes precedence)
    p.init_std = 1.0
    p.min_std = 0.05
    p.iterations = 3
    p.temperature = 0.4
    p.momentum = 0.0
    p.base_policy_std = 0.1
    p.num_elites = 64
    p.warm_start = 0
    return p


class MppiPlanner:
    def __init__(self, model: WorldModel, cfg: PlannerConfig, action_dim: int,
                 device: str = "cuda"):
        self.model = model
        self.cfg = cfg
        self.action_dim = action_dim
        self.device = device
        H = cfg.horizon
        self.discounts = (cfg.discount ** torch.arange(H, device=device)).view(H, 1)
        self.final_discount = cfg.discount ** H
        self._prev_mean: torch.Tensor | None = None  # warm-start state
        # Optional EXTERNAL planning sources (callables operating on RAW/denormalized
        # state, NOT the latent). When set, they REPLACE the world model's own head
        # in scoring/seeding — this is the "train only dynamics, borrow value/reward/
        # policy from the source agent + a heuristic" mode (CLAUDE.md §17). Each is
        # None by default => fall back to the model's head (vanilla SimDist).
        #   proposal_policy : state[N,obs] -> action[N,act]   (MPPI candidate seed)
        #   value_fn        : state[N,obs] -> value[N]        (terminal value V(s_H))
        #   reward_fn       : state[N,obs], action[N,act] -> reward[N]  (per-step r̂)
        # value_fn/reward_fn receive decode(z) (in raw-state mode that's the exact
        # denormalized state, so e.g. value_fn = source_agent.value works directly).
        self.proposal_policy = None
        self.value_fn = None
        self.reward_fn = None
        # Optional EXTERNAL base-sequence source: replaces `_policy_rollout` (the
        # world model's own MLP policy head, closed-loop in latent) as the source
        # of the `num_base` "base" candidate action SEQUENCES. Set to a flow policy
        # sampler to run the original SimDist CEM/MPPI with a FLOW proposal instead
        # of the MLP head. Signatures (raw action space, [-1,1]-ish):
        #   single : base_seq_fn(state[1,obs], num) -> [H, num, act]
        #   batched: base_seq_fn(states[B,obs], num) -> [B, H, num, act]
        # None (default) => use the model's policy head (vanilla SimDist).
        self.base_seq_fn = None
        # Diagnostics: when record_origin=True, each plan/plan_batch call appends
        # a per-iteration list of (base_elite_frac, base_elite_weight) to
        # _origin_records — i.e. among the topk elites that produce the action,
        # what fraction (by count and by softmax-weight mass) came from the
        # policy-seeded "base" candidates vs the random candidates. Zero cost when
        # off. See scripts/diag_mppi_origin.py.
        self.record_origin = False
        self._origin_records: list = []

    def _proposal(self, state: torch.Tensor) -> torch.Tensor:
        if self.proposal_policy is not None:
            return self.proposal_policy(state)
        return self.model.policy_action(state)

    def _num_base(self) -> int:
        """Count of policy-seeded ("noised base") candidates: num_base_override
        if >=0 (exact count), else round(num_samples * mixture_coef)."""
        cfg = self.cfg
        if cfg.num_base_override >= 0:
            return min(cfg.num_base_override, cfg.num_samples)
        return max(int(cfg.num_samples * cfg.mixture_coef), 0)

    def reset(self):
        self._prev_mean = None

    @torch.no_grad()
    def _policy_rollout(self, z: torch.Tensor, num: int) -> torch.Tensor:
        """Roll the (own-encoder) policy H steps from z -> [H, num, act].

        The policy lives in its own encoder's space (not the dynamics latent), so
        each step decodes the rolled dynamics latent back to a state and queries
        the policy there: z -> decode -> state -> policy_action -> a -> next."""
        H = self.cfg.horizon
        z = z.expand(num, -1).clone()
        acts = []
        shared = getattr(self.model.cfg, "shared_encoder", 0) and self.proposal_policy is None
        for _ in range(H):
            if shared:
                # policy shares the dynamics encoder -> query the latent DIRECTLY,
                # no decode bridge (the whole point of shared_encoder).
                a = self.model.policy_from_latent(z)
            else:
                a = self._proposal(self.model.decode(z))
            acts.append(a)
            z = self.model.next(z, a)
        return torch.stack(acts, dim=0)  # [H, num, act]

    @torch.no_grad()
    def _estimate_returns(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """actions: [H, N, act]. Returns [N] = discounted reward sum + final value."""
        H, N, _ = actions.shape
        z = z0.expand(N, -1).clone()
        pen = (self.cfg.dynamics_disagree_coef != 0
               and self.model.num_dynamics_heads() > 1)
        rewards = []
        disagrees = [] if pen else None
        for t in range(H):
            a = actions[t]
            if self.cfg.use_reward:
                if self.reward_fn is not None:
                    rewards.append(self.reward_fn(self.model.decode(z), a))
                else:
                    rewards.append(self.model.predict_reward(z, a))
            if pen:
                disagrees.append(self.model.dynamics_disagreement(z, a))  # [N]
            z = self.model.next(z, a)
        if self.value_fn is not None:
            final_value = self.value_fn(self.model.decode(z)) * self.final_discount
        else:
            final_value = self.model.predict_value(z, target=False) * self.final_discount
        ret = final_value
        if self.cfg.use_reward:
            rewards = torch.stack(rewards, dim=0)            # [H, N]
            ret = ret + (rewards * self.discounts).sum(0)    # [N]
        if pen:
            dis = torch.stack(disagrees, dim=0)              # [H, N]
            ret = ret - self.cfg.dynamics_disagree_coef * (dis * self.discounts).sum(0)
        return ret

    @torch.no_grad()
    def plan(self, state: torch.Tensor, eval_mode: bool = True,
             stddev: float = 0.0, return_full: bool = False) -> torch.Tensor:
        """state: [obs_dim] (single). Returns the first action [act_dim], or the
        full committed plan mean [H, act_dim] when return_full (diagnostics only —
        the executed action is still mean[0])."""
        cfg = self.cfg
        H = cfg.horizon
        z0 = self.model.encode(state.view(1, -1))  # [1, z]

        # base proposal: flow base-seq hook if set, else the policy head
        num_base = self._num_base()
        if num_base > 0 and self.base_seq_fn is not None:
            base_actions = self.base_seq_fn(state.view(1, -1), num_base)  # [H, num_base, act]
        else:
            base_actions = (self._policy_rollout(z0, num_base)
                            if num_base > 0 else None)  # [H, num_base, act]
        # "Noised base policy actions": spread the (deterministic) proposal into a
        # cloud so the num_base trajectories are not identical (SimDist Table III).
        if base_actions is not None and cfg.base_policy_std > 0:
            base_actions = base_actions + torch.randn_like(base_actions) * cfg.base_policy_std

        # init (mean, std)
        if cfg.seed_mean_from_policy:
            # center the candidate cloud on the precise direct policy action
            a_pi = self._proposal(state.view(1, -1)).view(1, -1)  # [1, act]
            mean = a_pi.expand(H, -1).clone()
        elif cfg.warm_start and self._prev_mean is not None:
            mean = torch.roll(self._prev_mean, -1, dims=0)
            mean[-1] = 0.0
        else:
            mean = torch.zeros(H, self.action_dim, device=self.device)
        std = torch.full((H, self.action_dim), cfg.init_std, device=self.device)

        num_rand = cfg.num_samples - num_base
        origin_rec = [] if self.record_origin else None
        for _ in range(cfg.iterations):
            noise = torch.randn(H, num_rand, self.action_dim, device=self.device)
            rand_actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
            if base_actions is not None:
                actions = torch.cat([rand_actions, base_actions.clamp(-1, 1)], dim=1)
            else:
                actions = rand_actions
            returns = self._estimate_returns(z0, actions)  # [N]

            elite_val, elite_idx = torch.topk(returns, cfg.num_elites)
            elite_acts = actions[:, elite_idx]             # [H, K, act]

            # softmax reweight over elites
            score = torch.exp(cfg.temperature * (elite_val - elite_val.max()))
            score = score / (score.sum() + 1e-9)
            if origin_rec is not None:
                is_base = (elite_idx >= num_rand).float()  # base candidates are concat'd after num_rand
                origin_rec.append((is_base.mean().item(), (score * is_base).sum().item()))
            new_mean = (score.view(1, -1, 1) * elite_acts).sum(1)  # [H, act]
            new_std = torch.sqrt(
                (score.view(1, -1, 1) * (elite_acts - new_mean.unsqueeze(1)) ** 2).sum(1)
            )
            mean = cfg.momentum * mean + (1 - cfg.momentum) * new_mean
            std = new_std.clamp(min=cfg.min_std)

        if cfg.warm_start:
            self._prev_mean = mean.clone()
        if origin_rec is not None:
            self._origin_records.append(origin_rec)

        if return_full:
            return mean  # [H, act] committed plan (executed action is still mean[0])
        action = mean[0]
        if not eval_mode and stddev > 0:
            action = (action + torch.randn_like(action) * stddev).clamp(-1, 1)
        return action

    # ------------------------------------------------------- batched plan
    # The per-env `plan` above is fine for single-env training collection, but
    # multi-process eval stacks all B workers' obs and asks for B actions at once.
    # Looping `plan` per row serialized B MPPI solves on the GPU (the dominant
    # eval cost). `plan_batch` carries a leading B dim through the SAME MPPI math
    # so all B*num_samples candidates roll the dynamics in one set of kernels —
    # numerically equivalent to the per-row loop, just GPU-saturating.
    @torch.no_grad()
    def _policy_rollout_batch(self, z0: torch.Tensor, num: int) -> torch.Tensor:
        """z0: [B, z]. Roll the policy H steps for `num` copies per env.
        Returns [B, H, num, act]."""
        H = self.cfg.horizon
        B, zdim = z0.shape
        z = z0[:, None, :].expand(B, num, zdim).reshape(B * num, zdim).clone()
        shared = getattr(self.model.cfg, "shared_encoder", 0) and self.proposal_policy is None
        acts = []
        for _ in range(H):
            a = (self.model.policy_from_latent(z) if shared
                 else self._proposal(self.model.decode(z)))   # [B*num, act]
            acts.append(a.view(B, num, -1))
            z = self.model.next(z, a)
        return torch.stack(acts, dim=1)  # [B, H, num, act]

    @torch.no_grad()
    def _estimate_returns_batch(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z0: [B, z], actions: [B, H, N, act]. Returns [B, N]."""
        B, H, N, _ = actions.shape
        zdim = z0.shape[1]
        z = z0[:, None, :].expand(B, N, zdim).reshape(B * N, zdim).clone()
        pen = (self.cfg.dynamics_disagree_coef != 0
               and self.model.num_dynamics_heads() > 1)
        rewards = []
        disagrees = [] if pen else None
        for t in range(H):
            a = actions[:, t].reshape(B * N, -1)
            if self.cfg.use_reward:
                if self.reward_fn is not None:
                    r = self.reward_fn(self.model.decode(z), a)
                else:
                    r = self.model.predict_reward(z, a)
                rewards.append(r.view(B, N))
            if pen:
                disagrees.append(self.model.dynamics_disagreement(z, a).view(B, N))
            z = self.model.next(z, a)
        if self.value_fn is not None:
            final_value = self.value_fn(self.model.decode(z)).view(B, N) * self.final_discount
        else:
            final_value = self.model.predict_value(z, target=False).view(B, N) * self.final_discount
        ret = final_value
        if self.cfg.use_reward:
            rewards = torch.stack(rewards, dim=1)                       # [B, H, N]
            ret = ret + (rewards * self.discounts.view(1, H, 1)).sum(1)  # [B, N]
        if pen:
            dis = torch.stack(disagrees, dim=1)                         # [B, H, N]
            ret = ret - self.cfg.dynamics_disagree_coef * (dis * self.discounts.view(1, H, 1)).sum(1)
        return ret

    @torch.no_grad()
    def plan_batch(self, states: torch.Tensor, eval_mode: bool = True,
                   stddev: float = 0.0) -> torch.Tensor:
        """states: [B, obs_dim]. Returns first actions [B, act_dim]. Vectorized
        equivalent of looping `plan` over the B rows (warm_start NOT supported here
        — all eval/adapt configs use warm_start=0)."""
        return self.plan_batch_from_latent(
            self.model.encode(states), eval_mode=eval_mode, stddev=stddev,
            states=states)

    @torch.no_grad()
    def plan_batch_from_latent(self, z0: torch.Tensor, eval_mode: bool = True,
                               stddev: float = 0.0,
                               states: torch.Tensor = None,
                               return_full: bool = False) -> torch.Tensor:
        """Same MPPI as `plan_batch` but STARTING FROM A LATENT z0 [B, z] — no
        encode, no decode. This is what makes stage-1 latent Dyna exact: an
        IMAGINED next latent z'=next(z,a) can be planned on DIRECTLY to get the
        base's action there, because MPPI is latent-native (encode is only the
        entry adapter). `states` is only needed by the raw-state proposal paths
        (base_seq_fn / seed_mean_from_policy); with shared_encoder the policy
        rolls from latent so states may be None."""
        cfg = self.cfg
        H = cfg.horizon
        B = z0.shape[0]
        A = self.action_dim

        num_base = self._num_base()
        if num_base > 0 and self.base_seq_fn is not None:
            assert states is not None, "base_seq_fn needs raw states"
            base_actions = self.base_seq_fn(states, num_base)  # [B, H, num_base, act]
        else:
            base_actions = (self._policy_rollout_batch(z0, num_base)
                            if num_base > 0 else None)  # [B, H, num_base, act]
        if base_actions is not None and cfg.base_policy_std > 0:
            base_actions = base_actions + torch.randn_like(base_actions) * cfg.base_policy_std

        if cfg.seed_mean_from_policy:
            assert states is not None, "seed_mean_from_policy needs raw states"
            a_pi = self._proposal(states).view(B, 1, A)   # [B, 1, act]
            mean = a_pi.expand(B, H, A).clone()
        else:
            mean = torch.zeros(B, H, A, device=self.device)
        std = torch.full((B, H, A), cfg.init_std, device=self.device)

        num_rand = cfg.num_samples - num_base
        origin_rec = [] if self.record_origin else None
        for _ in range(cfg.iterations):
            noise = torch.randn(B, H, num_rand, A, device=self.device)
            rand_actions = (mean[:, :, None, :] + std[:, :, None, :] * noise).clamp(-1, 1)
            if base_actions is not None:
                actions = torch.cat([rand_actions, base_actions.clamp(-1, 1)], dim=2)
            else:
                actions = rand_actions                    # [B, H, N, act]
            returns = self._estimate_returns_batch(z0, actions)  # [B, N]

            elite_val, elite_idx = torch.topk(returns, cfg.num_elites, dim=1)  # [B, K]
            K = elite_idx.shape[1]
            idx = elite_idx[:, None, :, None].expand(B, H, K, A)
            elite_acts = torch.gather(actions, 2, idx)    # [B, H, K, act]

            score = torch.exp(cfg.temperature
                              * (elite_val - elite_val.max(dim=1, keepdim=True).values))
            score = score / (score.sum(dim=1, keepdim=True) + 1e-9)  # [B, K]
            if origin_rec is not None:
                is_base = (elite_idx >= num_rand).float()  # [B, K]
                origin_rec.append(((is_base.mean()).item(),
                                   (score * is_base).sum(dim=1).mean().item()))
            sc = score[:, None, :, None]                  # [B, 1, K, 1]
            new_mean = (sc * elite_acts).sum(2)           # [B, H, act]
            new_std = torch.sqrt((sc * (elite_acts - new_mean[:, :, None, :]) ** 2).sum(2))
            mean = cfg.momentum * mean + (1 - cfg.momentum) * new_mean
            std = new_std.clamp(min=cfg.min_std)
        if origin_rec is not None:
            self._origin_records.append(origin_rec)

        if return_full:
            return mean  # [B, H, act] — the whole committed base-action sequence
        action = mean[:, 0]  # [B, act]
        if not eval_mode and stddev > 0:
            action = (action + torch.randn_like(action) * stddev).clamp(-1, 1)
        return action
