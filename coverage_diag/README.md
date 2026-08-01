# Coverage diagnostic — can "source-online + target-demo-anchored" work for Threading?

**Question (go/no-go).** `pi_phi` only interacts in SOURCE; the 20 target demos are
the only window onto TARGET dynamics. The method can only "anchor" the dynamics
where `pi_phi` VISITS states the demos also cover. So: **are the high-dynamics-gap
transitions `pi_phi` visits NEAR the target-demo support, or is there a high-gap
coverage desert** (source physics wrong + no demo + `pi_demo` OOD)?

## Method
- Roll out the SOURCE RL actor (= `pi_phi` init, `act_method="rl"`, stddev=0.1) in
  the SOURCE env. 40 episodes, 7203 transitions.
- Per transition: GT dynamics gap = `||s'_tgt - s'_src||` via counterfactual replay
  of `(s,a)` in the TARGET sim (`cf_utils.counterfactual_step`, qacc_warmstart=0).
- Per state: distance to nearest target-demo state in per-dim-standardized,
  quaternion-canonicalized 37-d space, **robust z (per-dim clip ±5)** so a few
  near-constant demo dims (tripod pose, demo std ~0.002) don't dominate the L2.
- Headline is **threshold-free + relative**: is the demo-distance of HIGH-gap
  transitions larger than LOW-gap? (`coverage_diag/coverage_final.png`)

## Result
| set | Spearman(gap, demo-dist) | median dist high/low gap | coverage high/low |
|---|---:|---:|---:|
| all (7203) | **-0.28** | 6.85 / 7.99 (0.86×) | 0.54 / 0.49 |
| contact-phase (1292, needle-ring<0.15) | **-0.03** | 6.44 / 7.08 (0.91×) | 0.63 / 0.54 |

- Gap concentrates in the contact phase (Panel D) — as expected, it's a contact gap.
- **No high-gap coverage desert.** High-gap transitions are NOT farther from the
  demos than low-gap ones — if anything slightly closer. Coverage-vs-gap-percentile
  (Panel C) is flat-to-rising, not falling.

## Verdict: the coverage precondition PASSES for this gap → proceed.
Because the gap is a contact-phase phenomenon and the (successful) target demos
necessarily pass through the contact phase, the states where source physics is most
wrong are exactly where demos exist. The "you can never anchor where it matters"
failure mode does **not** occur here.

## Honest caveats
- **Absolute coverage is loose.** Typical visited state is ~6–8 robust-z from the
  nearest demo vs demo intra-spacing ~0.5–0.9 — `pi_phi` (with exploration / while
  still mediocre) roams several demo-tube-widths outside the 20-demo support. So the
  anchoring is **coarse (same phase, not same fine pose)**: the demo term anchors
  Q/dynamics at a coarse scale; `pi_demo` will EXTRAPOLATE for fine corrections
  (e.g. recovery from a near-miss). This bounds, but does not negate, the method.
- Single policy (source RL actor) measured; the `pi_demo` rollout bracket was
  skipped (a wrapper/`eval_mode` bug — TODO if a second view is wanted).
- Raw (unclipped) z-distances have a heavy tail (p90≈108) from near-constant demo
  dims; this is a z-scoring artifact, not real structure — the relative result is
  invariant to it (see `finalize.py` clip ablation).

## Files
- `coverage_diag.py` — rollout + counterfactual gap + nearest-demo distance + npz.
- `finalize.py` — robust-z re-analysis + final figure (run on the npz, no env).
- `coverage_data.npz` — states, gaps, needle_ring, demo_dist per transition.
- `coverage_final.png` — the figure. (`coverage.png`/`coverage_v2.png` = earlier raw-z.)
