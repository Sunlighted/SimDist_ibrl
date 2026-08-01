"""Is the raw-state dynamics TARGET adapt actually improving the model?

`train_dynamics_adapt.py` only ever logs the MPPI score and the in-distribution
`adapt/consistency` train loss — neither answers "did adapting on target rollouts
reduce target prediction error?" (§27#5: an in-distribution mean-type metric going
down does NOT imply the planner got better; var-finetune improved held-out 8x and
lost 27.5 points).

This measures the thing directly, with zero env interaction: HELD-OUT target
open-loop rollout error (feed the REAL executed actions), needle-position mm MAE,
bucketed by phase, for:
  * PRE  = the source raw dyn  (threading_dyn_diverse_v2_seed1/best.pt)
  * POST = the target-adapted raw dyn (threading_dyn_adapt_*/model0.pt)
against the yardsticks §27 established (1.8 mm tolerance window; the actual needle
displacement over the same H = the signal the model must track).

Held-out data = the 30k SimDist-adapt target rollouts, which the raw adapt never
trained on (it collected its own MPPI rollouts), so this is a genuine held-out
TARGET set for both models.

Run:
  python scripts/diag_raw_dyn_adapt.py
"""
import argparse
import dataclasses
import json
import sys

import numpy as np
import torch

from rl.world_model import WorldModel, WorldModelConfig

NEEDLE = slice(9, 12)
RING = slice(23, 26)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_eps(path):
    d = np.load(path, allow_pickle=True)
    out = []
    for e in d["episodes"]:
        st = np.asarray(e["states"], dtype=np.float32)
        ac = np.asarray(e["actions"], dtype=np.float32)
        if st.shape[0] >= 2 and ac.shape[0] == st.shape[0] - 1:
            out.append((st, ac))
    return out


def build_raw_model(ckpt_path, dyn_cfg_path, device):
    """Rebuild the raw-state WorldModel (encoder_type='identity') and load either a
    bare WorldModel state_dict (train_dynamics.py) or a SimDistAgent one ('model.'
    prefixed, from train_dynamics_adapt.py)."""
    dc = json.load(open(dyn_cfg_path))
    cfg = WorldModelConfig(
        encoder_type="identity",
        mlp_hidden_dim=dc["hidden_dim"],
        num_layer=dc["num_layer"],
        horizon=dc["horizon"],
        activation=dc.get("activation", "relu"),
    )
    fields = {f.name for f in dataclasses.fields(WorldModelConfig)}
    if "layer_norm" in dc and "layer_norm" in fields:
        cfg.layer_norm = dc["layer_norm"]
    m = WorldModel(dc["obs_dim"], dc["action_dim"], cfg).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    log(f"  loaded {ckpt_path}  (missing={len(missing)} unexpected={len(unexpected)})")
    assert m.raw_state, "expected a raw-state (identity-encoder) world model"
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def rollout(m, s0, actions, H):
    z = m.encode(s0)                       # raw arm: z-score
    for k in range(H):
        z = m.next(z, actions[:, k])
    return m.decode(z)                     # raw arm: exact denorm inverse


def starts(eps, H):
    S0, AC, SH = [], [], []
    for st, ac in eps:
        T = ac.shape[0]
        for t in range(0, T - H + 1):
            S0.append(st[t]); AC.append(ac[t:t + H]); SH.append(st[t + H])
    return np.asarray(S0), np.asarray(AC), np.asarray(SH)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default="exps/mimicgen/threading_simdist_adapt_30k_seed1/rollouts.npz")
    p.add_argument("--dyn_cfg", default="exps/mimicgen/threading_dyn_diverse_v2_seed1/dyn_cfg.json")
    p.add_argument("--pre", default="exps/mimicgen/threading_dyn_diverse_v2_seed1/best.pt")
    p.add_argument("--post", default="exps/mimicgen/threading_dyn_adapt_goodpolicy_seed1/model0.pt")
    p.add_argument("--horizons", default="1,3,5")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    Hs = [int(x) for x in args.horizons.split(",")]
    dev = args.device

    eps = load_eps(args.rollouts)
    log(f"held-out TARGET episodes: {len(eps)}")
    log("[PRE ] source raw dyn"); pre = build_raw_model(args.pre, args.dyn_cfg, dev)
    log("[POST] target-adapted raw dyn"); post = build_raw_model(args.post, args.dyn_cfg, dev)

    S0, _, _ = starts(eps, min(Hs))
    d0 = np.linalg.norm(S0[:, NEEDLE] - S0[:, RING], axis=1)
    q33, q66 = np.quantile(d0, [0.33, 0.66])
    buckets = [("insertion", -np.inf, q33), ("approach", q33, q66), ("free", q66, np.inf)]
    log(f"phase buckets (needle-ring BODY dist tertiles): "
        f"insertion<{q33:.3f} approach {q33:.3f}-{q66:.3f} free>{q66:.3f}")

    print("\n=== held-out TARGET needle-position MAE (mm) — raw dyn, PRE vs POST adapt ===")
    print(f"{'H':>2} {'phase':>10} {'n':>6} {'actual_move':>12} "
          f"{'PRE(source)':>12} {'POST(adapt)':>12} {'delta':>8}")
    for H in Hs:
        S0, AC, SH = starts(eps, H)
        s0 = torch.as_tensor(S0, device=dev)
        ac = torch.as_tensor(AC, device=dev)
        sH = torch.as_tensor(SH, device=dev)
        d0 = np.linalg.norm(S0[:, NEEDLE] - S0[:, RING], axis=1)
        p_pre = rollout(pre, s0, ac, H)
        p_post = rollout(post, s0, ac, H)

        def mm(pred, mk):
            e = torch.norm(pred[:, NEEDLE] - sH[:, NEEDLE], dim=1) * 1000.0
            return e[mk].mean().item()

        for name, lo, hi in buckets:
            mk = torch.as_tensor((d0 >= lo) & (d0 < hi), device=dev)
            n = int(mk.sum())
            move = (torch.norm(s0[:, NEEDLE] - sH[:, NEEDLE], dim=1) * 1000.0)[mk].mean().item()
            a, b = mm(p_pre, mk), mm(p_post, mk)
            print(f"{H:>2} {name:>10} {n:>6} {move:>12.2f} {a:>12.2f} {b:>12.2f} "
                  f"{b-a:>+8.2f}")
        print("-" * 68)
    print("yardstick: target tolerance window = 1.8 mm/side (ring_size 0.85).")
    print("delta < 0 => the target adapt genuinely improved held-out TARGET prediction.")


if __name__ == "__main__":
    main()
