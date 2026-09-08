"""Evaluate a chunked BC policy in SINGLE-STEP (closed-loop) mode.

Why: `evaluate/eval.py` auto-detects `act_chunk` and runs an open-loop chunk of
Ta actions per query -- that is how the BC score in train_bc's log is produced.
But IBRL queries the prior once per step (`rl/q_agent.py:_act_ibrl` calls
`bc_policy.act(obs)`), so the number that matters for "how strong is the IBRL
prior" is the single-step one. This script hides `act_chunk` behind a shim so
the same harness runs the policy step-by-step.

    python scripts/eval_bc_singlestep.py --weight exps/.../model0.pt --num_game 100 --mp 10
"""

import argparse
import os

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

import train_bc  # noqa: E402
from evaluate.multi_process_eval import run_eval as mp_run_eval  # noqa: E402
from evaluate.eval import run_eval  # noqa: E402


class SingleStepShim(torch.nn.Module):
    """Exposes only `act`, so the eval harness never sees `act_chunk`."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def act(self, obs, eval_mode=True, cpu=True):
        return self.policy.act(obs, eval_mode=eval_mode, cpu=cpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--num_game", type=int, default=100)
    ap.add_argument("--mp", type=int, default=0, help="num eval processes (0 = single proc)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunked", type=int, default=0, help="1 = normal chunked eval (control arm)")
    args = ap.parse_args()

    policy, _, env_params = train_bc.load_model(args.weight, "cuda")
    agent = policy if args.chunked else SingleStepShim(policy)
    agent.train(False)

    print(f"[EVAL ENV] {env_params}")
    print(f"mode = {'CHUNKED (act_chunk)' if args.chunked else 'SINGLE-STEP (act)'}")

    if args.mp > 0:
        scores = mp_run_eval(env_params, agent, args.num_game, args.mp, args.seed)
    else:
        scores = run_eval(env_params, agent, args.num_game, args.seed)
    scores = np.array(scores)
    n = len(scores)
    sr = float((scores > 0).mean())
    print(f"\nsuccess rate: {sr:.3f}  ({int((scores>0).sum())}/{n}), "
          f"stderr {np.sqrt(sr * (1 - sr) / n):.3f}")


if __name__ == "__main__":
    main()
