"""SimDist stage-2: freeze the stage-1 world model (encoder/dynamics/value/
reward/decoder) and train ONLY the policy head, by BC, on pure expert demos.

Requires a stage-1 ckpt trained with shared_encoder=1 + detach_pi=1: the policy
queries the (frozen) dynamics latent directly, so this is a clean BC on a STATIC
latent space. Periodically evals policy-only (use_planner=0); saves the best
checkpoint (by SOURCE success) as model0.pt in the output dir, plus a cfg.yaml
so evaluate/eval.py --mode simdist can load it.

Run:
  python scripts/train_policy_head.py \
      --stage1_weight exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt \
      --out_dir exps/mimicgen/threading_simdist_tdmpc2_policy_seed1
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import shutil
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as ev
from env.domain_shift import DomainShiftConfig
from evaluate import run_eval_mp
import common_utils


def load_expert(path, max_traj=-1):
    S, A = [], []
    with h5py.File(path, "r") as f:
        keys = list(f["data"].keys())
        if max_traj and max_traj > 0:
            keys = keys[:max_traj]
        for k in keys:
            g = f["data"][k]
            n = min(len(g["actions"]), len(g["obs"]["state"]))
            S.append(np.asarray(g["obs"]["state"][:n], np.float32))
            A.append(np.clip(np.asarray(g["actions"][:n], np.float32), -1, 1))
    return np.concatenate(S), np.concatenate(A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_weight", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--expert_path",
                    default="release/data/mimicgen/threading_d0_source_sigma0.1_n5000/dataset.hdf5")
    ap.add_argument("--max_traj", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--num_eval", type=int, default=50)
    ap.add_argument("--num_proc", type=int, default=10)
    ap.add_argument("--joint_sigma", type=float, default=0.1)
    ap.add_argument("--select_on", choices=["source", "target"], default="source")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # copy stage-1 cfg.yaml so evaluate/eval.py --mode simdist can rebuild the arch
    src_cfg = os.path.join(os.path.dirname(args.stage1_weight), "cfg.yaml")
    shutil.copy(src_cfg, os.path.join(args.out_dir, "cfg.yaml"))

    agent, ws = ev.build(args.stage1_weight, use_planner=0)
    m = agent.model
    assert m.cfg.shared_encoder, "stage-2 expects a shared_encoder stage-1 ckpt"

    # freeze everything except the policy head
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.policy.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(m.policy.parameters(), lr=args.lr)

    S, A = load_expert(args.expert_path, args.max_traj)
    print(f"expert: {len(S)} (s,a) pairs from {args.expert_path}")
    S = torch.tensor(S).cuda(); A = torch.tensor(A).cuda()
    N = len(S)

    # eval env params: source (mode none) + target (the cfg's shift)
    tgt_params = dict(ws.eval_env_params)
    src_params = dict(ws.eval_env_params)
    src_params["domain_shift"] = DomainShiftConfig(mode="none")
    if args.joint_sigma is not None:
        for pr in (tgt_params, src_params):
            pr["hand_init_perturb_joint_sigma"] = args.joint_sigma

    saver = common_utils.TopkSaver(save_dir=args.out_dir, topk=1)
    agent.cfg.use_planner = 0  # policy-only eval + selection

    def evaluate(seed):
        sc_src = float(np.mean(run_eval_mp(env_params=src_params, agent=agent,
                       num_proc=args.num_proc, num_game=args.num_eval, seed=seed, verbose=False)))
        sc_tgt = float(np.mean(run_eval_mp(env_params=tgt_params, agent=agent,
                       num_proc=args.num_proc, num_game=args.num_eval, seed=seed, verbose=False)))
        return sc_src, sc_tgt

    best = -1.0
    for it in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch,), device="cuda")
        with torch.no_grad():
            z = m.encode(S[idx])            # frozen encoder -> static latent
        pred = m.policy_from_latent(z)       # only the policy head has grad
        loss = F.mse_loss(pred, A[idx])
        opt.zero_grad(); loss.backward(); opt.step()

        if it % args.eval_every == 0:
            m.eval()
            sc_src, sc_tgt = evaluate(seed=it)
            m.train()
            sel = sc_src if args.select_on == "source" else sc_tgt
            saved = saver.save(agent.state_dict(), sel, save_latest=True)
            best = max(best, sel)
            print(f"step {it}: bc_loss={loss.item():.4f}  source={sc_src:.3f}  "
                  f"target={sc_tgt:.3f}  [select={args.select_on} best={best:.3f}{'  *saved' if saved else ''}]")
            sys.stdout.flush()

    print(f"done. best {args.select_on} success = {best:.3f}  (model0.pt in {args.out_dir})")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
