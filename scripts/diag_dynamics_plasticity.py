"""Plasticity probe: is adapt_seed1's dynamics plateau a PLASTICITY loss (warm-start
rigidity) or a physical ceiling? Freeze the encoder (as in adapt), and on held-out
TARGET data compare the 1-step latent dynamics error of:
  (0) adapt_seed1 dynamics, NO further training        (the continue-adapt plateau)
  (a) continue: start from adapt_seed1, train N steps   (more of the same)
  (b) reset:    reinit the LAST dynamics layer, train N  (the user's proposal)
  (c) fresh:    reinit the WHOLE dynamics head, train N   (plasticity ceiling)
All share the SAME frozen encoder, SAME train/held-out target split, SAME steps.
Verdict: (b)/(c) << (a)  => plasticity was limiting (reset helps).
         (b)~(c)~(a)     => physical ceiling (contact chaos), reset can't help.
Latent target = sg(encode(s_{t+1})); loss = ||next(z_t,a_t) - z_true||^2 (the exact
consistency objective). Reports held-out MSE and MSE / latent-move (relative).
"""
import os, sys, copy
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, pyrallis, h5py

WM_CKPT = "exps/mimicgen/threading_simdist_adapt_seed1/latest.pt"
WM_CFG_DIR = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1"
TGT_DATA = "release/data/mimicgen/threading_d0_target_diverse_v2/dataset.hdf5"
N_TRAIN_TRANS = 40000
N_HELD = 10000
STEPS = 20000
BATCH = 256
LR = 3e-4
EVAL_EVERY = 2000


def build_model():
    from train_simdist import MainConfig
    from rl.world_model import WorldModel
    bc = pyrallis.load(MainConfig, open(os.path.join(WM_CFG_DIR, "cfg.yaml")))
    f = h5py.File(TGT_DATA, "r"); k0 = sorted(f["data"].keys())[0]
    od = f["data"][k0]["obs"]["state"].shape[1]; ad = f["data"][k0]["actions"].shape[1]; f.close()
    m = WorldModel(od, ad, bc.simdist.world_model).to("cuda")
    sd = torch.load(WM_CKPT)
    m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
    return m, od, ad


def load_transitions(n_needed):
    """(s_t, a_t, s_{t+1}) from target trajectories."""
    f = h5py.File(TGT_DATA, "r"); g = f["data"]
    ids = sorted((k for k in g if k.startswith("demo_")), key=lambda k: int(k.split("_")[1]))
    S, A, SN = [], [], []
    tot = 0
    for k in ids:
        s = np.array(g[k]["obs"]["state"], np.float32)
        a = np.array(g[k]["actions"], np.float32)
        T = min(len(s) - 1, len(a))
        if T <= 0:
            continue
        S.append(s[:T]); A.append(a[:T]); SN.append(s[1:T + 1])
        tot += T
        if tot >= n_needed:
            break
    f.close()
    S = np.concatenate(S)[:n_needed]; A = np.concatenate(A)[:n_needed]; SN = np.concatenate(SN)[:n_needed]
    return (torch.tensor(S).cuda(), torch.tensor(A).cuda(), torch.tensor(SN).cuda())


@torch.no_grad()
def held_out_err(m, S, A, SN):
    z = m.encode(S); zt = m.encode(SN)
    zp = m.next(z, A)
    mse = ((zp - zt) ** 2).sum(-1).mean().item()
    move = ((zt - z) ** 2).sum(-1).mean().item()
    return mse, mse / max(move, 1e-9)


def reinit_module(mod):
    for lyr in mod.modules():
        if isinstance(lyr, (nn.Linear, nn.LayerNorm)):
            lyr.reset_parameters()


def train_dyn(m, S, A, SN, hS, hA, hSN, tag):
    for p in m.encoder.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(m.dynamics.parameters(), lr=LR)
    N = len(S)
    print(f"\n[{tag}] held-out BEFORE: mse={held_out_err(m, hS, hA, hSN)[0]:.5f}", flush=True)
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, N, (BATCH,), device="cuda")
        with torch.no_grad():
            z = m.encode(S[idx]); zt = m.encode(SN[idx])
        zp = m.next(z, A[idx])
        loss = ((zp - zt) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0:
            mse, rel = held_out_err(m, hS, hA, hSN)
            print(f"  [{tag}] step {step:5d}: held-out mse={mse:.5f}  rel(mse/move)={rel:.3f}", flush=True)
    mse, rel = held_out_err(m, hS, hA, hSN)
    print(f"[{tag}] FINAL held-out mse={mse:.5f}  rel={rel:.3f}", flush=True)
    return mse, rel


def main():
    torch.manual_seed(0); np.random.seed(0)
    m0, od, ad = build_model()
    S, A, SN = load_transitions(N_TRAIN_TRANS + N_HELD)
    # split
    perm = torch.randperm(len(S), device="cuda")
    tr, he = perm[:-N_HELD], perm[-N_HELD:]
    Str, Atr, SNtr = S[tr], A[tr], SN[tr]
    hS, hA, hSN = S[he], A[he], SN[he]
    print(f"=== dynamics plasticity | adapt_seed1 dyn | train {len(tr)} / held-out {len(he)} target trans ===")
    print(f"    dynamics head layers: {[type(l).__name__ for l in m0.dynamics]}")

    # (0) plateau baseline (no training)
    mse0, rel0 = held_out_err(m0, hS, hA, hSN)
    print(f"\n(0) adapt_seed1 dynamics, NO training  : held-out mse={mse0:.5f}  rel={rel0:.3f}   <- the plateau")

    # (a) continue
    ma = copy.deepcopy(m0)
    a = train_dyn(ma, Str, Atr, SNtr, hS, hA, hSN, "a: continue")

    # (b) reset LAST dynamics layer
    mb = copy.deepcopy(m0)
    mb.dynamics[-1].reset_parameters()
    print(f"\n(b) reset last layer = dynamics[{len(mb.dynamics)-1}] ({type(mb.dynamics[-1]).__name__})")
    b = train_dyn(mb, Str, Atr, SNtr, hS, hA, hSN, "b: reset-last")

    # (c) fresh whole dynamics
    mc = copy.deepcopy(m0)
    reinit_module(mc.dynamics)
    c = train_dyn(mc, Str, Atr, SNtr, hS, hA, hSN, "c: fresh-all")

    print("\n===== PLASTICITY VERDICT (held-out 1-step latent MSE, lower=better) =====")
    print(f"  (0) plateau (no train)   : {mse0:.5f}")
    print(f"  (a) continue-adapt       : {a[0]:.5f}")
    print(f"  (b) reset last layer     : {b[0]:.5f}")
    print(f"  (c) fresh whole head     : {c[0]:.5f}")
    if min(b[0], c[0]) < 0.85 * a[0]:
        print("  => (b)/(c) clearly BELOW (a): PLASTICITY was limiting — reset helps.")
    else:
        print("  => (b)/(c) ~ (a): PHYSICAL CEILING (contact chaos) — reset can't help.")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
