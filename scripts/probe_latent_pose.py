"""Does the FROZEN stage-1 latent retain the object pose a policy needs?

WHY: docs/simdist.md §43 claims pixel stage-2 BC is capped because the SimNorm
latent -- trained by the value objective -- discards mm-level geometry
(findings.md §30). That is an INFERENCE from behaviour. This measures it
directly, and can refute it.

METHOD: freeze the stage-1 encoder, fit a probe from z -> a slice of the
privileged 37-d state, and report held-out R^2. The state layout is
    [0:3) eef_pos | [3:7) eef_quat | [7:9) gripper_qpos | [9:37) object
so `object` (needle + tripod pose) is dims 9:37 -- exactly the geometry the
insertion needs and the thing the pixel encoder must EXTRACT from images.

CONTROLS, so a low number cannot be misread:
  * prop slice [0:9): fed straight INTO the encoder, so it must probe well.
    If it does NOT, the probe/training is broken, not the latent.
  * linear vs 2-layer MLP probe: linear-low + MLP-high means the information is
    present but nonlinearly encoded (fine for a policy head, which is an MLP).
  * held-out EPISODES, not held-out frames -- frames inside one episode are
    near-duplicates and would inflate R^2 (§27#5's distribution-shift trap).
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn

import train_simdist
from env.robosuite_wrapper import PROP_KEYS  # noqa: F401  (documents the layout)

PROP_DIM = 9


def r2(pred, tgt):
    ss_res = ((pred - tgt) ** 2).sum(0)
    ss_tot = ((tgt - tgt.mean(0)) ** 2).sum(0) + 1e-12
    return (1.0 - ss_res / ss_tot).clamp(min=-1.0)


def fit_linear_exact(z_tr, y_tr, z_te, y_te, ridge=1e-3):
    """Closed-form ridge least squares -- NO optimisation error.

    The gradient-descent linear probe recovered `prop` at only R^2 0.557 on the
    STATE arm, where prop is literally a SUBSET of the 37-d encoder input and a
    faithful probe must approach 1.0. That gap could be SimNorm destroying
    information OR the probe being under-fitted, and those imply opposite
    conclusions -- so solve it exactly and let the control adjudicate.
    """
    X = torch.cat([z_tr, torch.ones(len(z_tr), 1, device=z_tr.device)], 1).double()
    Xt = torch.cat([z_te, torch.ones(len(z_te), 1, device=z_te.device)], 1).double()
    Y = y_tr.double()
    A = X.T @ X + ridge * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(A, X.T @ Y)
    return r2((Xt @ W).float(), y_te)


def fit_probe(z_tr, y_tr, z_te, y_te, hidden, steps=20000, lr=3e-4):
    d_in, d_out = z_tr.shape[1], y_tr.shape[1]
    net = (nn.Linear(d_in, d_out) if hidden == 0 else
           nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mu, sd = y_tr.mean(0), y_tr.std(0).clamp_min(1e-6)
    for _ in range(steps):
        idx = torch.randint(0, len(z_tr), (512,), device=z_tr.device)
        loss = nn.functional.mse_loss(net(z_tr[idx]), (y_tr[idx] - mu) / sd)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(z_te) * sd + mu
    return r2(pred, y_te)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--frames", type=int, default=20000)
    args = p.parse_args()

    agent, _, _ = train_simdist.load_model(args.ckpt, "cuda")
    model = agent.model
    model.eval()
    # world_model sets .pixel from encoder_type, and the camera list lives on the
    # obs_encoder (rl/pixel_obs_encoder.py:101), NOT on the world model. Reading a
    # non-existent model.cameras silently yielded () and made the pixel arm run as
    # if it were the state arm -- an IndexError one frame later.
    is_pixel = bool(getattr(model, "pixel", False)) and model.obs_encoder is not None
    cams = tuple(model.obs_encoder.cameras) if is_pixel else ()
    if is_pixel and not cams:
        raise SystemExit("FATAL: pixel model reports no cameras -- refusing to "
                         "silently probe it as a state model")
    print(f"ckpt={args.ckpt}\n  pixel={is_pixel}  cameras={cams or '(state arm)'}")

    import h5py
    rng = np.random.RandomState(0)
    with h5py.File(args.replay, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        keys = keys[:args.episodes]
        # HELD-OUT BY EPISODE (see docstring)
        n_te = max(1, len(keys) // 5)
        te_keys = set(rng.choice(len(keys), n_te, replace=False).tolist())
        split = {"train": [], "test": []}
        for i, k in enumerate(keys):
            split["test" if i in te_keys else "train"].append(k)

        def load(ks):
            S, F = [], {c: [] for c in cams}
            for k in ks:
                S.append(np.asarray(f[f"data/{k}"]["obs"]["state"]))
                for c in cams:
                    F[c].append(np.asarray(f[f"data/{k}"]["obs"][f"{c}_image"]))
            S = np.concatenate(S)
            F = {c: np.concatenate(v) for c, v in F.items()}
            if len(S) > args.frames:
                sel = rng.choice(len(S), args.frames, replace=False)
                S = S[sel]; F = {c: v[sel] for c, v in F.items()}
            return S, F

        S_tr, F_tr = load(split["train"])
        S_te, F_te = load(split["test"])

    def encode(S, F):
        zs = []
        with torch.no_grad():
            for i in range(0, len(S), 1024):
                if cams:
                    ob = {c: torch.as_tensor(v[i:i + 1024], device="cuda") for c, v in F.items()}
                    ob["prop"] = torch.as_tensor(S[i:i + 1024, :PROP_DIM],
                                                 dtype=torch.float32, device="cuda")
                else:
                    ob = torch.as_tensor(S[i:i + 1024], dtype=torch.float32, device="cuda")
                zs.append(model.encode(ob))
        return torch.cat(zs)

    z_tr, z_te = encode(S_tr, F_tr), encode(S_te, F_te)
    y_tr = torch.as_tensor(S_tr, dtype=torch.float32, device="cuda")
    y_te = torch.as_tensor(S_te, dtype=torch.float32, device="cuda")
    print(f"  train {len(z_tr)} frames / {len(split['train'])} eps | "
          f"test {len(z_te)} frames / {len(split['test'])} eps | z dim {z_tr.shape[1]}")

    slices = {"prop [0:9) CONTROL, fed into encoder": slice(0, PROP_DIM),
              "object [9:37) needle+tripod POSE": slice(PROP_DIM, y_tr.shape[1])}
    for name, sl in slices.items():
        print(f"\n  {name}")
        for lab, fn in (("linear*", lambda: fit_linear_exact(z_tr, y_tr[:, sl], z_te, y_te[:, sl])),
                        ("MLP-512", lambda: fit_probe(z_tr, y_tr[:, sl], z_te, y_te[:, sl], 512))):
            v = fn()
            print(f"     {lab:<8} held-out R^2: mean {v.mean():.3f}  "
                  f"min {v.min():.3f}  max {v.max():.3f}  "
                  f"frac<0.5: {(v < 0.5).float().mean():.2f}")
        print("     (* closed-form ridge: no optimisation error)")


if __name__ == "__main__":
    main()
