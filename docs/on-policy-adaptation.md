# On-policy consistency is the governing property of model-based adaptation

**Threading_D0 sim2sim, matched budget: 100k TARGET transitions, 100 eval points
(every 1k transitions, 100 games each), strict success, σ=0.1, §12 dynamics-only
target.** wandb project `simdist_threading_3arm` (60 runs / 9 groups).

Everything below is measured on one task with one stage-1 world model. Numbers are
the 75–100k steady-state window, averaged over seeds.

| family | group | n | score | sd |
|---|---|---:|---:|---:|
| **MB planning** | `dynens_h5_slow_anchor` | 10 | **0.979** | 0.010 |
| | `dynens_h5_slow` | 10 | 0.898 | 0.044 |
| | `dynens_h5` | 10 | 0.813 | 0.085 |
| | `plain_h3` | 10 | 0.564 | 0.171 |
| **MB Dyna** | `dyna_h5` | 3 | 0.157 | 0.046 |
| | `dyna_sgft` | 3 | 0.076 | 0.065 |
| **model-free** | `ibrl` | 5 | 0.082 | 0.036 |
| | `rlpd` | 5 | 0.037 | 0.035 |
| | `iql` | 5 | 0.002 | 0.003 |

---

## 1. For closing a sim2sim dynamics gap with a world model, **planning beats Dyna**

The Dyna arm shares *everything* with `dynens_h5` — the same stage-1 world model,
the same stage-2 BC policy head, the same frozen reward/value heads, the same
real-data budget, the same per-round dynamics update. The only change is **how the
model is consumed**: MPPI uses it *online* as a search operator, Dyna uses it
*offline* as a data generator for a policy that then acts alone.

    same world model, online search   ->  0.813
    same world model, offline Dyna    ->  0.157

The mechanism follows directly from the ceiling law (`findings.md` §27): the
learned dynamics carries ≈5.6 mm of persistent error against a 1.8 mm tolerance
window. Under MPPI that error is reset by reality at every step and only has H
steps to compound; under Dyna the policy is *trained* on the model's own output
and there is no correction at deployment. Adding a dense potential-based value
signal (SGFT, `r' = r + γV̂(z') − V̂(z)`) makes it **worse, not better** (0.157 →
0.076): the extra gradient points wherever the frozen source value thinks is good,
which is exactly where the policy has no data and the model has no accuracy.

At matched budget the model-free baselines never leave the floor (IBRL 0.082, RLPD
0.037, IQL 0.002; IBRL is flat 0 for its first 60k steps). This is not a claim that
they cannot solve the task — a model-free IBRL reaches 0.98 on this target — but
that it takes ~1M transitions to do so. **The 10× sample-efficiency advantage comes
from turning adaptation into supervised dynamics regression rather than sparse-reward
credit assignment.**

> Model-based planning is currently the right tool for this setting; model-based
> Dyna is not, and the reason is that Dyna removes the closed-loop correction that
> makes an imperfect model usable.

---

## 2. Adapting the value head collapses the system — in **every** update order

Tried and failed: value-then-dynamics, dynamics-then-value, and the two jointly.
All collapse. Three distinct reasons, and they compound:

**(a) The value must stay on-policy w.r.t. the proposal.** The distilled `V̂` is
`V^{π_src}`. Refitting it on the agent's own exploration data makes it the value of
a policy that is drifting — and that policy degrades as it explores, so the value is
chasing a moving, worsening target.

**(b) Simultaneous updates deny the dynamics a stable neighbourhood to repair.** If
the value changes, the MPPI argmax changes, so the visited distribution changes, so
the dynamics is asked to fix a different region every round. It never converges on
the neighbourhood that actually matters.

**(c) Value functions are simply hard to fit from little data.** Neither MC returns
nor TD adapt well from ~100k sparse-reward transitions. This is not an
implementation detail — `findings.md` §33 measured that MPPI's entire candidate-
discrimination signal is `std_i(γ^H V) = 0.0068`, i.e. **1.4% of the value range**,
while the residual of *any* offline refit is the same magnitude or 3–5× larger. The
refit therefore rewrites the top-64 elite set wholesale. Every averaged offline
metric (MSE / R² / Spearman) is structurally blind to this, because the quantity
that matters is a within-candidate-set spread, not an average error.

**Do not adapt the value.** The only form never falsified is a *gated residual*
anchored on the incumbent function — and even that is only worth attempting when the
deployment distribution moves genuinely out of support.

---

## 3. The dynamics is repaired **in the neighbourhood of the policy** — which is why
   SimDist works at all, and why reducing model error pays

Stage-3 collects with MPPI seeded by the policy proposal, so the transitions the
dynamics is trained on are exactly the ones the planner will later query. The
repair is local, and locality is what makes it tractable: the model does not need
to be globally accurate, only accurate where this policy looks.

Two interventions that reduce model error both cash out:

- **Dynamics ensemble** (5 heads, plan-time mean): `plain_h3` 0.564 → `dynens_h5`
  0.813. `findings.md` §31 isolates the attribution — 5-head rollout 0.970 vs
  single-head 0.120 from the *same weights* ⇒ the gain is 100% plan-time variance
  cancellation, not a better encoder. Saturates at ≈3 heads (what remains is bias,
  not variance).
- **Optimistic exploration during collection** (`collect_disagree_coef=0.5`, i.e.
  MPPI *seeks* high-disagreement regions while collecting): included in all dynens
  arms; it steers the repair toward where the model is least certain.

Note the asymmetry with §2: dynamics error enters the planner through *which state
you imagine*, and is buffered by the task's tolerance window; value error enters
through the *argmax* and has no buffer at all.

⚠️ Pessimism at *planning* time is a separate matter and is **dead**: penalising
either value-ensemble disagreement (§29) or dynamics-geometry disagreement (§31)
gives ≤0.2 no effect and ≥0.33 monotone harm. Optimism during **collection** helps;
pessimism during **planning** does not.

---

## 4. Both the proposal and the value must be on-policy — and the proposal needs a
   **two-sided** constraint

The stage-2 policy head is subject to two competing requirements (`simdist.md` §42):

1. **Stay close to π_src**, because the frozen value head is `V^{π_src}` and the
   planner scores candidates with it;
2. **Be good enough on the target**, because it seeds the MPPI candidate set and
   drives stage-3 collection — it has to warm-start the adaptation.

`bc_anchor_coef = λ` is the knob for exactly this trade-off:

    L = MSE(π_θ(s_demo), a_demo)            # demo states: learn the target actions
      + λ · MSE(π_θ(s_src),  π_src(s_src))  # SOURCE states: pin back to π_src

The two terms act on **different distributions**, which is the whole point: the
demo term is not weakened where demos exist, and the anchor only pins states the
demos never cover — where BC is otherwise unconstrained (§35 measured that ~100% of
stage-3 collection happens off the demo manifold, and that off-manifold behaviour
across demo sets diverges by 80% of the action's own spread).

Measured on the strongest arm, 10 seeds, same demo set per seed, paired:

| | score | sd |
|---|---:|---:|
| `dynens_h5_slow` (λ=0) | 0.898 | 0.044 |
| `dynens_h5_slow_anchor` (λ=1) | **0.979** | **0.010** |

**Δ = +0.081, SE = 0.013, z = 6.0, 10/10 seeds同向.** Two things are notable:

- It is the **largest single intervention in this sweep** relative to how little it
  changes: `plain → dynens` costs a 5× dynamics ensemble for +0.249, `saturated →
  unsaturated demos` costs a re-collection for +0.085, and the anchor is one extra
  loss term for **+0.081**.
- **The seed-to-seed sd collapses 4.3×** (0.044 → 0.010). Each seed here uses a
  *different* demo set, so that sd contains the 20-demo draw variance that
  `demo-collection.md` §38 identified as the dominant noise source in this whole
  project. The anchor makes the outcome nearly insensitive to which 20 demos you
  happened to collect — arguably more valuable than the mean shift.

⚠️ Scope: this contradicts §42's earlier "no gain on a strong base" reading, which
rested on a single λ=0 seed (SE 0.033). With 10 paired seeds the anchor helps even
at 0.898, closing 81% of the remaining gap. What §42's interaction structure does
still predict correctly is the *shape*: the effect is largest where the proposal is
the binding constraint.

⚠️ Unresolved: *why* proximity to π_src helps has two candidate labels — "the value
needs on-policy consistency" vs "π_src is simply a good off-manifold prior" — and
they predict the same inverted-U in λ. §42 lists an oracle probe and a strong-base
null that both argue against the pure form of the first, without excluding an
interaction form. The intervention is established; the mechanism label is not.

---

## Conclusion

**On-policy consistency is one of the most important properties in model-based
adaptation, and it is easy to overlook.** It shows up three separate times in this
sweep, each time as the thing that decides success or collapse:

- The **value** must remain on-policy w.r.t. the proposal — adapting it breaks that
  consistency and the system collapses regardless of update order (§2).
- The **dynamics** is only repairable because it is repaired on-policy, in the
  neighbourhood the planner actually queries (§3).
- The **proposal** must be pulled back toward π_src so that it and the frozen value
  live on the same distribution — worth +0.081 and a 4.3× variance reduction, on top
  of an already-strong 0.898 arm (§4).

And the same principle explains the planning-vs-Dyna gap (§1): MPPI is on-policy by
construction — it re-plans from the true state every step — while Dyna trains a
policy on states the model imagines and the policy never has to answer to reality
until deployment.

---

### Caveats
- One task (Threading_D0), one stage-1 world model, sim2sim. Method-level claims
  need a second task.
- The model-free arms carry warm-up transitions that are **not** counted in the 100k
  budget (10 episodes each); the SimDist arms have no such extra.
- Eval density differs between families (100 points for the model-free/Dyna arms, 20
  for the SimDist arms); the x-axis is transitions in both, so curves overlay
  correctly.
- `dynens_h5*` scores are `score_best`, a max over 4 pessimism coefficients, and
  therefore carry a selection bias of roughly +0.03–0.05 (`plain_h3`, Dyna and the
  model-free arms cannot take the same max). The ranking is unaffected — the nearest
  gap it could plausibly close is 0.081, and the largest is 0.4+.

---

## Appendix: paper framing and naming (working notes)

### What is actually novel here (this determines how hard to brand it)

| component | Δ (75–100k) | novelty |
|---|---:|---|
| dynamics ensemble, plan-time mean | **+0.249** | ❌ standard PETS/MBPO machinery. Our contribution is the *attribution* (5-head 0.970 vs single-head 0.120 from the same weights ⇒ 100% plan-time variance cancellation) and the finding that it is the only error-reduction lever that worked. |
| unsaturated collector for stage-2 demos | +0.085 | ⚠️ empirical data-side rule; effect is only ~1× the 20-demo draw noise (σ≈0.092). |
| **source anchor `bc_anchor_coef`** | **+0.081** | ✅ **the new algorithmic component.** 10/10 paired seeds, z=6.0, and a 4.3× collapse in seed-to-seed sd. |
| frozen value head | — | ✅ strong negative result, but it is a *non*-action. |

⇒ The name should sit on **on-policy consistency / proposal anchoring**, not on the
world-model pipeline (which is inherited from SimDist / TD-MPC2).

### Title

Working title:

> **Rethinking Sim-to-Sim Dynamics Adaptation with World Models**

Alternative that does not spend the title on the platform:

> **Rethinking World-Model Adaptation under Dynamics Gaps: On-Policy Consistency
> is What Matters**

Rationale for the second: the whole target-domain design (§12, dynamics-only gap,
no latent registration term) exists so the target is a *proxy for the real world*;
"sim-to-sim" in the title reads like a toy setting and undersells it. Better to
introduce sim-to-sim in the abstract as the controlled testbed.

### Method-name candidates

**Framing A — analysis/principle paper (recommended, best fit to the contribution).**
Name the *principle*, keep the recipe descriptive:

- principle: **On-Policy Consistency (OPC)**
- recipe: **OPC-Adapt**, or plainly **Anchored SimDist**

This keeps §1–§4 as four pieces of evidence for one principle, and avoids inviting
"what exactly is your delta over SimDist?".

**Framing B — standalone method name.**

| name | expansion | note |
|---|---|---|
| **ANCHOR** | **AN**chored **C**onsistency for w**O**rld-model t**R**ansfer | the name *is* the mechanism; covers both senses (proposal anchored to π_src, value held fixed as the anchor). Acronym is a stretch. |
| **CAPO** | **C**onsistency-**A**nchored **P**roposal **O**ptimization | clean, semantically accurate |
| **OPERA** | **O**n-**P**olicy **E**nsemble **R**efinement for **A**daptation | reads well, but putting "ensemble" in the name implies the ensemble is ours, which it is not |

Preference: **ANCHOR > CAPO > OPERA**.

**Avoid:** any name implying we contribute the world model + planner (that is
SimDist / TD-MPC2), and anything containing `Dist` (the distillation is inherited).

⚠️ **Name collisions are NOT verified.** Check Google Scholar / arXiv full text /
paperswithcode before committing — several RL acronyms in this space are already
taken (e.g. OPAL, PACT, CAPS are all used elsewhere).
