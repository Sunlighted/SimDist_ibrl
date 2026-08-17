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
        assert use_state, "SimDistAgent is state-only"
        self.cfg = cfg
        self.device = device
        obs_dim = int(obs_shape[0])
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = cfg.planner.discount

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

    def update_distill(self, states, actions, rewards, value_targets=None,
                       policy_mask=None, policy_states=None, policy_actions=None,
                       policy_states_mask=None) -> dict[str, float]:
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
        H = states.shape[1] - 1
        rho = cfg.rho

        # policy-BC may train on a SEPARATE (pure-expert) batch; default to the
        # main batch (legacy). The policy uses enc_pi and does not touch z, so
        # this only changes what the policy head clones.
        if policy_states is None:
            p_states, p_actions, p_mask = states, actions, policy_mask
        else:
            p_states, p_actions, p_mask = policy_states, policy_actions, policy_states_mask

        z = self.model.encode(states[:, 0])  # [B, z]
        cons_l = rew_l = val_l = pol_l = dec_l = 0.0
        wsum = 0.0
        for t in range(H):
            a_t = actions[:, t]
            r_t = rewards[:, t]
            w = rho ** t
            wsum += w

            with torch.no_grad():
                z_next_enc = self.model.encode(states[:, t + 1])

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
            if not self.model.raw_state:
                dec_l = dec_l + w * F.mse_loss(
                    self.model.decode(self.model.encode(states[:, t]).detach()), states[:, t])
            # policy BC. Three modes:
            #  - dedicated enc_pi (legacy): policy_action uses its own encoder, the
            #    dynamics z is never touched -> no detach needed.
            #  - shared encoder + detach_pi (default for shared): query the policy on
            #    encode(s).detach() so the BC gradient does NOT reshape the dynamics
            #    encoder (avoids the policy-vs-consistency fight; TD-MPC2 also
            #    detaches the latent for the policy update). Lets the planner roll
            #    the policy directly in latent (no decode bridge).
            #  - shared encoder, no detach: BC also shapes the shared encoder.
            ps_t = p_states[:, t]
            if cfg.shared_encoder and cfg.detach_pi:
                pred_a = self.model.policy_from_latent(self.model.encode(ps_t).detach())
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
            z0 = self.model.encode(states[:, 0])
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
        H = states.shape[1] - 1
        rho = cfg.rho

        z = self.model.encode(states[:, 0])
        cons_l = pol_l = 0.0
        wsum = 0.0
        for t in range(H):
            a_t = actions[:, t]
            w = rho ** t
            wsum += w
            with torch.no_grad():
                z_next_enc = self.model.encode(states[:, t + 1])
            if self.cfg.finetune_dynamics:
                cons_step, z_next_pred = self._dynamics_cons_loss(z, a_t, z_next_enc)
                cons_l = cons_l + w * cons_step
            else:
                z_next_pred = self.model.next(z, a_t)
            if self.cfg.finetune_policy:
                pol_l = pol_l + w * F.mse_loss(self.model.policy_action(states[:, t]), a_t)
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
    def update_bc_policy(self, states, actions, lr: float = 3e-4,
                         anchor_states=None, anchor_actions=None,
                         anchor_coef: float = 0.0) -> dict[str, float]:
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
            self._bc_opt = torch.optim.Adam(self.model.policy.parameters(), lr=lr)
        H = states.shape[1] - 1
        s_flat = states[:, :H].reshape(-1, states.shape[-1])  # [B*H, obs]
        a_flat = actions.reshape(-1, actions.shape[-1])        # [B*H, act]
        pred = self.model.policy_action(s_flat)
        loss = F.mse_loss(pred, a_flat)
        total, anchor_loss = loss, None
        if anchor_coef > 0 and anchor_states is not None:
            anchor_loss = F.mse_loss(self.model.policy_action(anchor_states),
                                     anchor_actions)
            total = loss + anchor_coef * anchor_loss
        self._bc_opt.zero_grad()
        total.backward()
        gnorm = nn.utils.clip_grad_norm_(
            self.model.policy.parameters(), self.cfg.world_model.grad_clip)
        self._bc_opt.step()
        out = {"bc/policy_loss": float(loss.item()), "bc/grad_norm": float(gnorm)}
        if anchor_loss is not None:
            out["bc/anchor_loss"] = float(anchor_loss.item())
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
        H = states.shape[1] - 1
        rho = cfg.rho
        with torch.no_grad():
            z = self.model.encode(states[:, 0])  # frozen encoder
        cons_l = 0.0
        wsum = 0.0
        for t in range(H):
            with torch.no_grad():
                z_next_enc = self.model.encode(states[:, t + 1])  # frozen encoder target
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
        H = states.shape[1] - 1
        rho = cfg.rho
        with torch.no_grad():
            z = self.model.encode(states[:, 0])  # frozen encoder
        vaml_l = 0.0
        wsum = 0.0
        for t in range(H):
            with torch.no_grad():
                z_next_enc = self.model.encode(states[:, t + 1])         # frozen encoder target
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
        B = states.shape[0]
        H = states.shape[1] - 1
        rho = cfg.rho
        with torch.no_grad():
            z = self.model.encode(states[:, 0])
        cons_l = 0.0
        wsum = 0.0
        for t in range(H):
            with torch.no_grad():
                z_next_enc = self.model.encode(states[:, t + 1])
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
        B = states.shape[0]
        H = states.shape[1] - 1
        rho = cfg.rho
        z = self.model.encode(states[:, 0])  # encoder TRAINS
        cons_l = 0.0
        wsum = 0.0
        for t in range(H):
            with torch.no_grad():
                z_next_enc = self.model.encode(states[:, t + 1])  # stop-grad target
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
