from typing import Optional
from dataclasses import dataclass, field
import copy
from contextlib import contextmanager

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from networks.encoder import VitEncoder, VitEncoderConfig
from networks.encoder import ResNetEncoder, ResNetEncoderConfig, DrQEncoder
from networks.encoder import ResNet96Encoder, ResNet96EncoderConfig
from rl.critic import Critic, CriticConfig
from rl.actor import Actor, ActorConfig
from rl.actor import FcActor, FcActorConfig
from rl.critic import MultiFcQ, MultiFcQConfig


@dataclass
class QAgentConfig:
    device: str = "cuda"
    lr: float = 1e-4
    critic_target_tau: float = 0.01
    stddev_clip: float = 0.3
    # encoder
    use_prop: int = 0
    enc_type: str = "vit"
    vit: VitEncoderConfig = field(default_factory=lambda: VitEncoderConfig())
    resnet: ResNetEncoderConfig = field(default_factory=lambda: ResNetEncoderConfig())
    resnet96: ResNet96EncoderConfig = field(default_factory=lambda: ResNet96EncoderConfig())
    # critic & actor
    critic: CriticConfig = field(default_factory=lambda: CriticConfig())
    actor: ActorConfig = field(default_factory=lambda: ActorConfig())
    state_critic: MultiFcQConfig = field(default_factory=lambda: MultiFcQConfig())
    state_actor: FcActorConfig = field(default_factory=lambda: FcActorConfig())
    # algo
    act_method: str = "rl"  # "rl/ibrl/ibrl_soft"
    bootstrap_method: str = ""  # "rl/ibrl/ibrl_soft"
    # ibrl
    ibrl_eps_greedy: float = 1
    soft_ibrl_beta: float = 10
    # bc loss regularization
    bc_loss_coef: float = 0.1
    bc_loss_dynamic: int = 0  # dynamically scale bc loss weight
    # SGFT: potential-based reward shaping from a frozen source-domain value
    # function. When `sgft_shaping`, the critic's TD target reward is replaced
    # by `r + sgft_coef * (discount * V_src(s') - V_src(s))`, where V_src is the
    # source agent's value (set via set_sgft_value_source). Potential-based, so
    # the optimal target policy is unchanged — it only guides exploration.
    sgft_shaping: int = 0
    sgft_coef: float = 1.0
    # sgft_potential=1 (default): true potential-based shaping
    #   r' = r + coef*(gamma^n V_src(s') - V_src(s)). Policy-invariant (Ng 1999):
    #   argmax_a Q'(s,a) = argmax_a Q(s,a), so it CANNOT steer exploration in a
    #   DDPG/IBRL actor — only reshapes the critic target (equiv. to initializing
    #   Q<-V_src, Wiewiora 2003). Useful only when the bottleneck is value
    #   PROPAGATION, not exploration.
    # sgft_potential=0: guidance shaping r' = r + coef*V_src(s'). Deliberately
    #   NOT potential-based: Q'(s,a) now DEPENDS on V_src, so argmax IS pulled
    #   toward high-V states -> it can steer exploration. Biased (optimal policy
    #   changes), but with end_on_success + a well-ranked V_src (Spearman~0.96)
    #   the bias is likely benign. No q-clamp shift applies (Q' has no fixed
    #   [q_min,q_max] offset), so clamp_q_target is ignored in this mode.
    sgft_potential: int = 1
    # Clamp the critic's Bellman TD target to [q_target_min, q_target_max].
    # Default OFF (byte-identical to every existing run/config). Unbiased for a
    # sparse 0/1 reward + end_on_success (true Q in [0,1]): sparse reward +
    # pessimistic min-ensemble + an OOD bootstrap (e.g. SGFT's per-step V_src
    # shaping under a wide/OOD hand-init sigma) otherwise drives Q into runaway
    # divergence with no correcting signal — the exact failure mode fixed the
    # same way in GatedGapAgent/ResidualAgent/OracleMapAgent (§16/20/21).
    clamp_q_target: int = 0
    q_target_min: float = 0.0
    q_target_max: float = 1.0

    def __post_init__(self):
        if self.bootstrap_method == "":
            self.bootstrap_method = self.act_method


class QAgent(nn.Module):
    def __init__(
        self, use_state, obs_shape, prop_shape, action_dim, rl_camera: str, cfg: QAgentConfig
    ):
        super().__init__()
        self.use_state = use_state
        self.rl_camera = rl_camera
        self.cfg = cfg

        if use_state:
            self.critic = MultiFcQ(obs_shape, action_dim, cfg.state_critic)
            self.actor = FcActor(obs_shape, action_dim, cfg.state_actor)
        else:
            self.encoder = self._build_encoders(obs_shape)
            repr_dim = self.encoder.repr_dim
            patch_repr_dim = self.encoder.patch_repr_dim
            print("encoder output dim: ", repr_dim)
            print("patch output dim: ", patch_repr_dim)

            assert len(prop_shape) == 1
            prop_dim = prop_shape[0] if cfg.use_prop else 0

            # create critics & actor
            self.critic = Critic(
                repr_dim=repr_dim,
                patch_repr_dim=patch_repr_dim,
                prop_dim=prop_dim,
                action_dim=action_dim,
                cfg=self.cfg.critic,
            )
            self.actor = Actor(repr_dim, patch_repr_dim, prop_dim, action_dim, cfg.actor)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        if not self.use_state:
            print(common_utils.wrap_ruler(f"encoder weights"))
            print(self.encoder)
            common_utils.count_parameters(self.encoder)

        print(common_utils.wrap_ruler("critic weights"))
        print(self.critic)
        common_utils.count_parameters(self.critic)

        print(common_utils.wrap_ruler("actor weights"))
        print(self.actor)
        common_utils.count_parameters(self.actor)

        # optimizers
        if not self.use_state:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.lr)

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)

        # data augmentation
        self.aug = common_utils.RandomShiftsAug(pad=4)

        self.bc_policies: list[nn.Module] = []
        # SGFT value source (frozen source-domain QAgent). Held in a list so it
        # is NOT registered as a submodule (would otherwise bloat/duplicate this
        # agent's state_dict). Empty unless set_sgft_value_source() is called.
        self.sgft_value_source: list["QAgent"] = []
        # IBRL prior-preference margin (subtracted from the RL actor's Q in the
        # max-Q action selection, so the actor only overrides the BC prior when it
        # beats it by > margin). 0.0 = standard IBRL (no change). A trainer can set
        # this to a large value early and anneal it to 0, to keep eval >= the prior
        # while the critic calibrates (see train_ibrl_simdist prior_margin_*). Plain
        # attribute (default 0) so all existing runs are byte-for-byte unchanged.
        self.prior_margin: float = 0.0
        # to log rl vs bc during evaluation
        self.stats: Optional[common_utils.MultiCounter] = None

        self.critic_target.train(False)
        self.train(True)
        self.to(self.cfg.device)

    def set_state_fourier_stats(self, mean, std):
        """Push per-dim z-score stats into every FourierFeatures module in the
        agent (critic/actor + their target copies), so the PE input transform
        aligns to unit scale. No-op when Fourier encoding is off (no such
        modules). Stats persist in the ckpt so standalone eval self-normalizes."""
        from rl.fourier import FourierFeatures
        n = 0
        for m in self.modules():
            if isinstance(m, FourierFeatures):
                m.set_stats(mean, std)
                n += 1
        return n

    def _build_encoders(self, obs_shape):
        if self.cfg.enc_type == "vit":
            return VitEncoder(obs_shape, self.cfg.vit).to(self.cfg.device)
        elif self.cfg.enc_type == "resnet":
            return ResNetEncoder(obs_shape, self.cfg.resnet).to(self.cfg.device)
        elif self.cfg.enc_type == "resnet96":
            return ResNet96Encoder(obs_shape, self.cfg.resnet96).to(self.cfg.device)
        elif self.cfg.enc_type == "drq":
            return DrQEncoder(obs_shape).to(self.cfg.device)
        else:
            assert False, f"Unknown encoder type {self.cfg.enc_type}."

    def add_bc_policy(self, bc_policy):
        bc_policy.train(False)
        self.bc_policies.append(bc_policy)

    def set_sgft_value_source(self, agent: "QAgent"):
        """Register a frozen source-domain agent whose value function V_src is
        used for SGFT potential-based reward shaping (see update_critic)."""
        agent.train(False)
        for p in agent.parameters():
            p.requires_grad_(False)
        self.sgft_value_source = [agent]

    @torch.no_grad()
    def value(self, obs: dict[str, torch.Tensor], use_ibrl: bool = False) -> torch.Tensor:
        """V(s) = min-ensemble Q(s, pi(s)).

        `pi` is the deterministic RL actor by default — this is the SGFT potential
        V_src (§10) and what H2O+ / the gap methods consume, so the default must
        not change.

        `use_ibrl=True` instead evaluates the agent's IBRL HYBRID policy: the max-Q
        pick over {RL actor, prior}, i.e. the policy an ibrl agent ACTUALLY executes.
        For an agent trained with act_method="ibrl", V(s, pi_rl) understates its own
        value wherever the prior is the better action, and — more importantly for
        planning — the two Vs are not a monotone rescaling of each other (the prior
        lifts V only where it wins), so they induce DIFFERENT rankings. Requires the
        prior to be attached via add_bc_policy().

        Expects a BATCHED obs dict. Builds its own feature (for pixel obs) from the
        raw camera frames so it never reuses a caller's precomputed `feat` (which
        may belong to a different encoder).
        """
        obs = {k: v for k, v in obs.items() if k != "feat"}
        if not self.use_state:
            obs["feat"] = self._encode(obs, augment=False)

        def _min_q(action: torch.Tensor) -> torch.Tensor:
            if isinstance(self.critic, Critic):
                return torch.min(*self.critic.forward(obs["feat"], obs["prop"], action))
            return self.critic.forward_k(obs["state"], action).min(-1)[0]

        rl_action = self._act_default(
            obs=obs, eval_mode=True, stddev=0.0, clip=None, use_target=False
        )
        q = _min_q(rl_action)
        if use_ibrl:
            assert len(self.bc_policies) == 1, (
                "value(use_ibrl=True) needs the IBRL prior attached (add_bc_policy)"
            )
            # max-Q over {RL actor, prior}, SELECTED AND EVALUATED WITH THE SAME
            # (online) critic. We deliberately do NOT reuse `_act_ibrl`: it argmaxes
            # with `critic_target` (correct there — it builds the bootstrap), while V
            # reads the online `critic`. Those two are EMA-lagged apart, so the action
            # _act_ibrl picks can score BELOW the RL action under the online critic —
            # which would break the defining property V_hybrid >= V_rl (measured: it
            # held for only ~60% of states). Selecting and scoring with one critic
            # restores it (and prior_margin is a training-time knob, not part of V).
            bc_action = self.bc_policies[0].act(obs, cpu=False)
            q = torch.maximum(q, _min_q(bc_action))
        return q

    def set_stats(self, stats):
        self.stats = stats

    def train(self, training=True):
        self.training = training
        if not self.use_state:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

        assert not self.critic_target.training
        for bc_policy in self.bc_policies:
            assert not bc_policy.training

    @contextmanager
    def override_act_method(self, override_method: str):
        original_method = self.cfg.act_method
        assert original_method != override_method

        self.cfg.act_method = override_method
        yield

        self.cfg.act_method = original_method
        return

    def _encode(self, obs: dict[str, torch.Tensor], augment: bool) -> torch.Tensor:
        """This function encodes the observation into feature tensor."""
        data = obs[self.rl_camera].float()
        if augment:
            data = self.aug(data)
        return self.encoder.forward(data, flatten=False)

    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False
        if self.use_state:
            if obs["state"].dim() == 1:
                should_unsqueeze = True
        else:
            if obs[self.rl_camera].dim() == 3:
                should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
        return should_unsqueeze

    def act(
        self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True
    ) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor"""
        assert not self.training
        assert not self.actor.training
        unsqueezed = self._maybe_unsqueeze_(obs)

        if not self.use_state:
            assert "feat" not in obs
            obs["feat"] = self._encode(obs, augment=False)

        if self.cfg.act_method == "rl":
            action = self._act_default(
                obs=obs,
                eval_mode=eval_mode,
                stddev=stddev,
                clip=None,
                use_target=False,
            )
        elif self.cfg.act_method == "ibrl":
            action = self._act_ibrl(
                obs=obs,
                eval_mode=eval_mode,
                stddev=stddev,
                clip=None,
                eps_greedy=self.cfg.ibrl_eps_greedy,
                use_target=False,
            )
        elif self.cfg.act_method == "ibrl_soft":
            action = self._act_ibrl_soft(
                obs=obs,
                eval_mode=eval_mode,
                stddev=stddev,
                clip=None,
                use_target=False,
            )
        else:
            assert False, f"unknown act method {self.cfg.act_method}"

        if unsqueezed:
            action = action.squeeze(0)

        action = action.detach()
        if cpu:
            action = action.cpu()
        return action

    def _act_default(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        use_target: bool,
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        dist = actor.forward(obs, stddev)

        if eval_mode:
            assert not self.training

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=clip)

        return action

    def _act_ibrl(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        eps_greedy: float,
        use_target: bool,
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        if eval_mode:
            assert not actor.training

        assert len(self.bc_policies) == 1
        bc_policy = self.bc_policies[0]
        bc_action = bc_policy.act(obs, cpu=False)

        rl_dist: utils.TruncatedNormal = actor(obs, stddev)
        if eval_mode:
            rl_action = rl_dist.mean
        else:
            rl_action = rl_dist.sample(clip)

        rl_bc_actions = torch.stack([rl_action, bc_action], dim=1)
        bsize, num_action, _ = rl_bc_actions.size()

        # get q(a)
        # feat -> [batch, num_patch, patch_dim] -> [batch, num_action(2), num_patch, patch_dim]
        flat_actions = rl_bc_actions.flatten(0, 1)
        if isinstance(self.critic_target, Critic):
            flat_qfeats = obs["feat"].unsqueeze(1).repeat(1, num_action, 1, 1).flatten(0, 1)
            flat_props = obs["prop"].unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            q1, q2 = self.critic_target.forward(flat_qfeats, flat_props, flat_actions)
            qa: torch.Tensor = torch.min(q1, q2).view(bsize, num_action)
        else:
            state = obs["state"]
            flat_state = state.unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            qa: torch.Tensor = self.critic_target.forward_k(flat_state, flat_actions)
            qa = qa.min(-1)[0].view(bsize, num_action)

        # Prior-preference margin: penalize the RL actor's Q (column 0 =
        # rl_action; column 1 = bc_action) so the actor only wins the argmax when it
        # beats the BC prior by > prior_margin. Default 0.0 => no change.
        if self.prior_margin != 0.0:
            qa = qa.clone()
            qa[:, 0] = qa[:, 0] - self.prior_margin

        # best_action_idx: [batch]
        greedy_action_idx: torch.Tensor = qa.argmax(1)
        greedy_action = rl_bc_actions[range(bsize), greedy_action_idx]
        # actions: [batch, action_dim]

        if eval_mode or eps_greedy == 1:
            action = greedy_action
            selected_action_idx = greedy_action_idx
        else:
            eps = torch.rand((bsize, 1), device=qa.device)
            use_greedy = (eps < eps_greedy).float()
            rand_action_idx = torch.randint(0, num_action, (bsize,))
            rand_action = rl_bc_actions[range(bsize), rand_action_idx]
            assert rand_action.size() == greedy_action.size()
            action = rand_action * (1 - use_greedy) + greedy_action * use_greedy
            selected_action_idx = rand_action_idx * (1 - use_greedy) + greedy_action_idx * use_greedy

            if self.stats is not None:
                self.stats["actor/greedy"].append(use_greedy.sum(), bsize)

        if self.stats is not None:
            use_bc = (selected_action_idx >= 1).float()
            if use_target:
                use_bc = use_bc.mean().item()
                self.stats["actor/bootstrap_bc"].append(use_bc)
            else:
                use_bc = use_bc.sum().item()
                if eval_mode:
                    self.stats["actor/bc_eval"].append(use_bc, bsize)
                else:
                    assert bsize == 1, f"bsize should be 1, but got {bsize}"
                    rl_action_norm = rl_action.squeeze()[:6].norm().item()
                    bc_action_norm = bc_action.squeeze()[:6].norm().item()
                    self.stats["actor/anorm_rl"].append(rl_action_norm)
                    self.stats["actor/anorm_bc"].append(bc_action_norm)
                    self.stats["actor/bc_train"].append(use_bc, bsize)

        return action

    def _act_ibrl_soft(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        use_target: bool,
    ):
        actor = self.actor_target if use_target else self.actor
        if eval_mode:
            assert not actor.training

        assert len(self.bc_policies) == 1
        bc_policy = self.bc_policies[0]
        bc_action = bc_policy.act(obs, cpu=False)

        rl_dist: utils.TruncatedNormal = actor(obs, stddev)
        if eval_mode:
            rl_action = rl_dist.mean
        else:
            rl_action = rl_dist.sample(clip)

        rl_bc_actions = torch.stack([rl_action, bc_action], dim=1)

        # cat along the num action dim
        # actions: [bsize, n_rl_actions + n_bc_actions * n_bc, action_dim]
        bsize, num_action, _ = rl_bc_actions.size()

        flat_actions = rl_bc_actions.flatten(0, 1)
        if isinstance(self.critic_target, Critic):
            flat_qfeats = obs["feat"].unsqueeze(1).repeat(1, num_action, 1, 1).flatten(0, 1)
            flat_props = obs["prop"].unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            q1, q2 = self.critic_target.forward(flat_qfeats, flat_props, flat_actions)
            qa: torch.Tensor = torch.min(q1, q2).view(bsize, num_action)
        else:
            state = obs["state"]
            flat_state = state.unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            qa: torch.Tensor = self.critic_target.forward_k(flat_state, flat_actions)
            qa = qa.min(-1)[0].view(bsize, num_action)

        # decide which action to take
        p_center = torch.nn.functional.softmax(qa * self.cfg.soft_ibrl_beta, dim=1)
        center_idx = p_center.multinomial(1)
        if (not use_target) and self.stats is not None and (not eval_mode):
            assert bsize == 1
            self.stats["actor/p_max"].append(p_center.max().item())

        # center_idx: [batchsize, 1]
        action = rl_action * (1 - center_idx) + bc_action * center_idx

        if self.stats is not None:
            use_bc = center_idx.sum().item()
            if use_target:
                # must be called from update_critic
                self.stats["actor/bootstrap_bc"].append(use_bc, bsize)
            else:
                if eval_mode:
                    self.stats["actor/bc_eval"].append(use_bc, bsize)
                else:
                    assert bsize == 1
                    self.stats["actor/bc_act"].append(use_bc, bsize)
                    self.stats["actor/qrl-qbc"].append((qa[0][0] - qa[0][1]).item())

        return action

    def _clamp_q(self, q: torch.Tensor, shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project a Q estimate onto its known true range; no-op when off.

        Unshifted (`shift=None`): [q_target_min, q_target_max]. Unbiased for a
        sparse 0/1 reward + end_on_success (true Q in [0,1]).

        SGFT shaping changes the range. The shaped MDP's value is
            Q'(s,a) = Q(s,a) - coef * Phi(s),
        so with Phi = V_src measured in [0.45, 1.02] on target states, Q' legally
        lives in [q_min - coef*Phi(s), q_max - coef*Phi(s)] ~= [-1.0, +0.55] — i.e.
        MOSTLY NEGATIVE. Clamping that to the unshifted [0,1] clips the whole
        legitimate negative range and pins the critic at the floor (observed:
        critic_qt collapsing 0.51 -> 0.04 with score ~0). Pass `coef * Phi` as
        `shift` so the bound tracks the shaped range PER-STATE.
        """
        if not self.cfg.clamp_q_target:
            return q
        if shift is None:
            return q.clamp(self.cfg.q_target_min, self.cfg.q_target_max)
        return torch.max(torch.min(q, self.cfg.q_target_max - shift),
                         self.cfg.q_target_min - shift)

    def update_critic(
        self,
        obs: dict[str, torch.Tensor],
        reply: dict[str, torch.Tensor],
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        stddev: float,
    ):
        with torch.no_grad():
            # use train mode as we use actor dropout
            assert self.actor_target.training

            if self.cfg.bootstrap_method == "rl":
                next_action = self._act_default(
                    obs=next_obs,
                    eval_mode=False,
                    stddev=stddev,
                    clip=self.cfg.stddev_clip,
                    use_target=True,
                )
            elif self.cfg.bootstrap_method == "ibrl":
                next_action = self._act_ibrl(
                    obs=next_obs,
                    eval_mode=False,
                    stddev=stddev,
                    clip=self.cfg.stddev_clip,
                    eps_greedy=1.0,
                    use_target=True,
                )
            elif self.cfg.bootstrap_method == "ibrl_soft":
                next_action = self._act_ibrl_soft(
                    obs=next_obs,
                    eval_mode=False,
                    stddev=stddev,
                    clip=self.cfg.stddev_clip,
                    use_target=True,
                )
            else:
                assert False, f"unknown bootstrap method {self.cfg.bootstrap_method}"

            if isinstance(self.critic_target, Critic):
                target_q1, target_q2 = self.critic_target.forward(
                    next_obs["feat"], next_obs["prop"], next_action
                )
                target_q = torch.min(target_q1, target_q2)
            else:
                target_q = self.critic_target.forward_k(next_obs["state"], next_action).min(-1)[0]

            # SGFT potential-based reward shaping. Telescopes correctly over the
            # n-step transition because `discount` here is the batch bootstrap
            # (gamma^n, or 0 at terminal where V_src(s') must drop out):
            #   r' = r + coef * (gamma^n * V_src(s_{t+n}) - V_src(s_t))
            sgft_reward = reward
            sgft_shaping_mean = 0.0
            sgft_v_mean = 0.0
            if self.cfg.sgft_shaping and self.sgft_value_source:
                src = self.sgft_value_source[0]
                v_s = src.value(obs)
                v_sp = src.value(next_obs)
                if self.cfg.sgft_potential:
                    shaping = self.cfg.sgft_coef * (discount * v_sp - v_s)
                    # Shaped MDP: learned Q'(x) = Q(x) - coef*Phi(x), so each
                    # clamp tracks the potential AT ITS OWN STATE — Phi(s') for
                    # the bootstrap Q'(s'), Phi(s) for the target y ~= Q'(s,a).
                    shift_sp = self.cfg.sgft_coef * v_sp
                    shift_s = self.cfg.sgft_coef * v_s
                    do_clamp = True
                else:
                    # Guidance (non-potential): r' = r + coef*V_src(s'). Breaks
                    # invariance on purpose so V steers the actor. Q' has no fixed
                    # [q_min,q_max] range (it absorbs an accumulated V term), so
                    # skip the clamp entirely here.
                    shaping = self.cfg.sgft_coef * v_sp
                    shift_sp = shift_s = None
                    do_clamp = False
                sgft_reward = reward + shaping
                sgft_shaping_mean = shaping.mean().item()
                sgft_v_mean = v_s.mean().item()
            else:
                shift_sp = shift_s = None
                do_clamp = True

            if do_clamp:
                target_q = self._clamp_q(target_q, shift=shift_sp)
            target_q = (sgft_reward + (discount * target_q)).detach()
            if do_clamp:
                target_q = self._clamp_q(target_q, shift=shift_s)

        action = reply["action"]
        loss_fn = nn.functional.mse_loss
        if isinstance(self.critic, Critic):
            q1, q2 = self.critic.forward(obs["feat"], obs["prop"], action)
            critic_loss = loss_fn(q1, target_q) + loss_fn(q2, target_q)
        else:
            qs: torch.Tensor = self.critic(obs["state"], action)
            critic_loss = nn.functional.mse_loss(
                qs, target_q.unsqueeze(1).repeat(1, qs.size(1)), reduction="none"
            )
            critic_loss = critic_loss.sum(1).mean(0)

        metrics = {}
        metrics["train/critic_qt"] = target_q.mean().item()
        metrics["train/critic_loss"] = critic_loss.item()
        if self.cfg.sgft_shaping and self.sgft_value_source:
            metrics["sgft/shaping"] = sgft_shaping_mean
            metrics["sgft/v_source"] = sgft_v_mean

        if not self.use_state:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward(retain_graph=True)

        if not self.use_state:
            self.encoder_opt.step()
        self.critic_opt.step()
        return metrics

    def _compute_actor_loss(self, obs: dict[str, torch.Tensor], stddev: float):
        if not self.use_state:
            assert "feat" in obs, "safety check"

        action: torch.Tensor = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=stddev,
            clip=self.cfg.stddev_clip,
            use_target=False,
        )

        if isinstance(self.critic, Critic):
            q = torch.min(*self.critic.forward(obs["feat"], obs["prop"], action))
        else:
            q: torch.Tensor = self.critic(obs["state"], action).min(-1)[0]
        actor_loss = -q.mean()
        return actor_loss

    def _compute_actor_bc_loss(self, batch, *, backprop_encoder):
        obs: dict[str, torch.Tensor] = batch.obs

        if not self.use_state:
            assert "feat" not in obs, "safety check"
            obs["feat"] = self._encode(obs, augment=True)

        if not backprop_encoder and not self.use_state:
            obs["feat"] = obs["feat"].detach()

        pred_action = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=0,
            clip=None,
            use_target=False,
        )
        action: torch.Tensor = batch.action["action"]
        loss = nn.functional.mse_loss(pred_action, action, reduction="none")
        loss = loss.sum(1).mean(0)
        return loss

    def update_actor(self, obs: dict[str, torch.Tensor], stddev: float):
        metrics = {}
        actor_loss = self._compute_actor_loss(obs, stddev)
        metrics["train/actor_loss"] = actor_loss.item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return metrics

    def update_actor_rft(
        self,
        obs: dict[str, torch.Tensor],
        stddev: float,
        bc_batch,
        ref_agent: "QAgent",
    ):
        metrics = {}
        actor_loss = self._compute_actor_loss(obs, stddev)
        metrics["train/actor_loss"] = actor_loss.item()
        bc_loss = self._compute_actor_bc_loss(bc_batch, backprop_encoder=False)
        assert actor_loss.size() == bc_loss.size()

        ratio = 1
        if self.cfg.bc_loss_dynamic:
            with torch.no_grad(), utils.eval_mode(self, ref_agent):
                assert ref_agent.cfg.act_method == "rl"

                # temporarily change to rl since we want to regularize actor not hybrid
                act_method = self.cfg.act_method
                self.cfg.act_method = "rl"

                ref_bc_obs = bc_batch.obs.copy()  # shallow copy
                ref_action = ref_agent.act(ref_bc_obs, eval_mode=True, cpu=False)

                # we first get the ref_action and then pop the feature
                # then we get the curr_action so that the obs["feat"] is the current feature
                # which can be used for computing q-values
                bc_obs = bc_batch.obs
                curr_action = self.act(bc_obs, eval_mode=True, cpu=False)

                if isinstance(self.critic, Critic):
                    curr_q = torch.min(*self.critic(bc_obs["feat"], bc_obs["prop"], curr_action))
                    ref_q = torch.min(*self.critic(bc_obs["feat"], bc_obs["prop"], ref_action))
                else:
                    curr_q = self.critic.forward_k(bc_obs["state"], curr_action).min(-1)[0]
                    ref_q = self.critic.forward_k(bc_obs["state"], ref_action).min(-1)[0]

                ratio = (ref_q > curr_q).float().mean().item()

                # recover to original act_method
                self.cfg.act_method = act_method

        loss = actor_loss + (self.cfg.bc_loss_coef * ratio * bc_loss).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()

        metrics["rft/bc_loss"] = bc_loss.mean().item()
        metrics["rft/ratio"] = ratio
        return metrics

    def update(
        self,
        batch,
        stddev,
        update_actor,
        bc_batch=None,
        ref_agent: Optional["QAgent"] = None,
    ):
        obs: dict[str, torch.Tensor] = batch.obs
        reward: torch.Tensor = batch.reward
        discount: torch.Tensor = batch.bootstrap
        next_obs: dict[str, torch.Tensor] = batch.next_obs

        if not self.use_state:
            obs["feat"] = self._encode(obs, augment=True)
            with torch.no_grad():
                next_obs["feat"] = self._encode(next_obs, augment=True)

        metrics = {}
        metrics["data/batch_R"] = reward.mean().item()
        critic_metric = self.update_critic(
            obs=obs,
            reply=batch.action,
            reward=reward,
            discount=discount,
            next_obs=next_obs,
            stddev=stddev,
        )
        utils.soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)
        metrics.update(critic_metric)

        if not update_actor:
            return metrics

        # NOTE: actor loss does not backprop into the encoder
        if not self.use_state:
            obs["feat"] = obs["feat"].detach()

        if bc_batch is None:
            actor_metric = self.update_actor(obs, stddev)
        else:
            assert ref_agent is not None
            actor_metric = self.update_actor_rft(obs, stddev, bc_batch, ref_agent)

        utils.soft_update_params(self.actor, self.actor_target, self.cfg.critic_target_tau)
        metrics.update(actor_metric)

        return metrics

    def pretrain_actor_with_bc(self, batch):
        """pretrain actor and encoder with bc"""
        loss = self._compute_actor_bc_loss(batch, backprop_encoder=True)

        if not self.use_state:
            self.encoder_opt.zero_grad(set_to_none=True)

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()

        if not self.use_state:
            self.encoder_opt.step()
        self.actor_opt.step()

        return {"pretrain/loss": loss.item()}
