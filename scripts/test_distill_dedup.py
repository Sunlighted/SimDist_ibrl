"""Regression harness for the `update_distill` encode-once refactor.

The refactor replaces up to 3*H+2 separate `encode()` calls per gradient step
with 2 (one grad-mode call on t=0, one no_grad call on t=1..H), which is the
single biggest cost lever for the pixel arm (the encoder goes from ~17 forwards
of batch B to 2). It touches the most load-bearing function in the repo, so it
must be proven BIT-IDENTICAL on the state arm before anything depends on it --
otherwise every state result in docs/findings.md is silently at risk.

Usage:
    # BEFORE refactoring, capture the reference:
    python scripts/test_distill_dedup.py --save /tmp/distill_ref.json
    # AFTER refactoring, compare:
    python scripts/test_distill_dedup.py --check /tmp/distill_ref.json

Covers the config combinations that change which encode calls happen:
  * shared_encoder x detach_pi  (whether the policy reuses the dynamics latent)
  * decoder on/off              (decoder_coef>0 adds an encode per step)
  * value_targets given or None (None takes the TD-bootstrap branch, which is
                                 the only consumer of z_next_enc's value head)
  * separate policy_states      (p_states is NOT `states`, so the cache for the
                                 policy path must come from its own encode)
  * policy_mask                 (masked-mean BC path)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.simdist_agent import SimDistAgent, SimDistConfig  # noqa: E402
from rl.world_model import WorldModelConfig  # noqa: E402

OBS_DIM, ACT_DIM, B, H = 37, 7, 8, 3


def build(shared_encoder, detach_pi, decoder_coef, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    wm = WorldModelConfig()
    wm.latent_dim = 64
    wm.mlp_hidden_dim = 64
    wm.num_layer = 2
    wm.horizon = H
    wm.shared_encoder = int(shared_encoder)
    wm.detach_pi = int(detach_pi)
    wm.decoder_coef = float(decoder_coef)
    cfg = SimDistConfig()
    cfg.world_model = wm
    agent = SimDistAgent(True, (OBS_DIM,), (9,), ACT_DIM, "", cfg, device="cpu")
    return agent


def batch(seed=0, sep_policy=False, mask=False, with_values=True):
    g = torch.Generator().manual_seed(seed)
    out = dict(
        states=torch.randn(B, H + 1, OBS_DIM, generator=g),
        actions=torch.randn(B, H, ACT_DIM, generator=g).clamp(-1, 1),
        rewards=(torch.rand(B, H, generator=g) > 0.8).float(),
    )
    if with_values:
        out["value_targets"] = torch.randn(B, H + 1, generator=g) * 0.1
    if sep_policy:
        out["policy_states"] = torch.randn(B, H + 1, OBS_DIM, generator=g)
        out["policy_actions"] = torch.randn(B, H, ACT_DIM, generator=g).clamp(-1, 1)
        if mask:
            out["policy_states_mask"] = (torch.rand(B, H, generator=g) > 0.3).float()
    elif mask:
        out["policy_mask"] = (torch.rand(B, H, generator=g) > 0.3).float()
    return out


CASES = [
    # name,                       shared, detach, dec,  sep,   mask,  values
    ("shared+detach",                  1,      1, 1.0, False, False, True),
    ("shared+detach,no_dec",           1,      1, 0.0, False, False, True),
    ("shared,no_detach",               1,      0, 1.0, False, False, True),
    ("enc_pi (legacy)",                0,      1, 1.0, False, False, True),
    ("shared+detach,tdboot",           1,      1, 1.0, False, False, False),
    ("shared+detach,sep_policy",       1,      1, 1.0, True,  False, True),
    ("shared+detach,mask",             1,      1, 1.0, False, True,  True),
    ("shared+detach,sep+mask",         1,      1, 1.0, True,  True,  True),
]


def run():
    results = {}
    for name, shared, detach, dec, sep, mask, vals in CASES:
        agent = build(shared, detach, dec, seed=0)
        # Freeze the optimizer out of the picture: we want the LOSS values for a
        # fixed set of weights, not the post-step weights.
        b = batch(seed=1, sep_policy=sep, mask=mask, with_values=vals)
        metrics = agent.update_distill(**b)
        results[name] = {k: float(v) for k, v in sorted(metrics.items())}

        # ---- per-parameter-group POST-STEP checksums -------------------------
        # Loss values alone are NOT a sufficient test: with shared_encoder=1 the
        # detach_pi=0 and detach_pi=1 paths produce the SAME forward value and
        # differ only in gradient flow (verified: both give loss 9.27062321). So
        # compare weights AFTER update_distill's internal backward+step, which
        # is sensitive to gradients reaching the wrong module -- precisely the
        # regression the refactor could introduce.
        for tag, mod in (("encoder", agent.model.encoder),
                         ("enc_pi", agent.model.enc_pi),
                         ("dynamics", agent.model.dynamics),
                         ("policy", agent.model.policy),
                         ("value", agent.model.value),
                         ("reward", agent.model.reward),
                         ("decoder", agent.model.decoder)):
            if mod is None:
                continue
            s = sum(p.detach().double().sum().item() for p in mod.parameters())
            results[name][f"__w_{tag}"] = float(s)

        # the latents themselves -- the quantity the refactor actually moves
        with torch.no_grad():
            z_all = torch.stack([agent.model.encode(b["states"][:, t])
                                 for t in range(H + 1)], dim=1)
        results[name]["__z_checksum"] = float(z_all.double().sum())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--check", type=str, default=None)
    ap.add_argument("--atol", type=float, default=0.0,
                    help="0.0 = require EXACT equality (the refactor claims bit-identity)")
    args = ap.parse_args()

    res = run()

    if args.save:
        with open(args.save, "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
        print(f"saved reference for {len(res)} cases -> {args.save}")
        for n, m in res.items():
            print(f"  {n:28s} loss={m.get('distill/loss', float('nan')):.8f}")
        return 0

    if args.check:
        ref = json.load(open(args.check))
        bad = []
        for name in sorted(set(ref) | set(res)):
            if name not in ref or name not in res:
                bad.append(f"{name}: present in only one run")
                continue
            for k in sorted(set(ref[name]) | set(res[name])):
                a, b_ = ref[name].get(k), res[name].get(k)
                if a is None or b_ is None:
                    bad.append(f"{name}.{k}: missing on one side")
                elif a != b_ and abs(a - b_) > args.atol:
                    bad.append(f"{name}.{k}: {a!r} -> {b_!r}  (delta {b_ - a:+.3e})")
        print(f"compared {len(res)} cases")
        if bad:
            print("MISMATCH:")
            for x in bad:
                print("  - " + x)
            return 2
        print("IDENTICAL -- refactor preserves state-arm numerics exactly")
        return 0

    for n, m in res.items():
        print(f"{n:28s} {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
