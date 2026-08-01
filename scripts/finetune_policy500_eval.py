"""Take the adapted world model from threading_simdist_adapt_ibrlmppi_400k_seed1
(dynamics/encoder/value adapted on TARGET rollouts; policy = 20-demo BC), give it
a BETTER base policy by re-BC-ing ONLY the policy head on 500 TARGET demos
(encoder frozen), then compare policy-only vs narrow-search MPPI (the same
original_mppi=0 recipe the 400k run used).

4 evals on TARGET (sigma 0.1), each num_game games:
  OLD policy (20-demo BC)   : policy-only, MPPI
  NEW policy (500-demo BC)  : policy-only, MPPI
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as ev
from rl.sequence_replay import SequenceReplayBuffer, load_demos_to_sequence_buffer
from evaluate import run_eval_mp
import common_utils

STAGE1 = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt"
ADAPTED = "exps/mimicgen/threading_simdist_adapt_ibrlmppi_400k_seed1/model0.pt"
DEMOS = os.environ.get("DEMOS",
                       "release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5")
N_DEMO = int(os.environ.get("N_DEMO", "500"))
BC_STEPS = int(os.environ.get("BC_STEPS", "3000"))
BC_BATCH = 256
BC_LR = 3e-4
NUM_GAME = int(os.environ.get("NUM_GAME", "200"))
SEED = 1
SIGMA = 0.1
LABEL = os.environ.get("LABEL", f"{N_DEMO}-demo")
SAVE_DIR = os.environ.get(
    "SAVE_DIR", f"exps/mimicgen/threading_simdist_adapt_ibrlmppi_400k_bc{N_DEMO}_seed1")


def evaluate(agent, ws, use_planner, tag):
    agent.cfg.use_planner = use_planner
    params = dict(ws.eval_env_params)
    params["hand_init_perturb_joint_sigma"] = SIGMA
    scores = run_eval_mp(env_params=params, agent=agent, num_proc=10,
                         num_game=NUM_GAME, seed=SEED, verbose=False)
    m = float(np.mean(scores))
    se = float(np.std(scores) / np.sqrt(len(scores)))
    print(f"  [{tag:28s}] use_planner={use_planner}  score = {m:.3f} +/- {se:.3f}  (n={NUM_GAME})",
          flush=True)
    return m, se


def main():
    common_utils.set_all_seeds(SEED)
    # build from stage-1 MainConfig (has the narrow trust-proposal planner block),
    # then load the ADAPTED world model + 20-demo policy on top
    agent, ws = ev.build(STAGE1, use_planner=0)
    agent.load_state_dict(torch.load(ADAPTED))
    agent = agent.to("cuda")
    p = agent.planner.cfg
    print(f"[planner] narrow recipe: samples={p.num_samples} mixture={p.mixture_coef} "
          f"init_std={p.init_std} iters={p.iterations} use_reward={p.use_reward} H={p.horizon}\n")

    print("===== OLD base policy (20-demo BC, from the adapt run) =====")
    old_po = evaluate(agent, ws, 0, "OLD policy-only")
    old_mppi = evaluate(agent, ws, 1, "OLD MPPI (narrow)")

    print(f"\n===== re-BC policy head on {N_DEMO} TARGET demos (encoder frozen) =====")
    H = ws.agent.cfg.world_model.horizon
    demo_replay = SequenceReplayBuffer(max_episodes=10000, device="cpu")
    load_demos_to_sequence_buffer(demo_replay, DEMOS, num_data=N_DEMO,
                                  reward_scale=ws.cfg.env_reward_scale)
    assert demo_replay.can_sample(H)
    agent.train()
    for step in range(BC_STEPS):
        states, actions, _ = demo_replay.sample(BC_BATCH, H, "cuda")
        m = agent.update_bc_policy(states, actions, lr=BC_LR)
        if (step + 1) % 500 == 0:
            print(f"  bc step {step+1}/{BC_STEPS}  policy_loss={m['bc/policy_loss']:.5f}", flush=True)
    agent.eval()

    # save the new ckpt (adapted WM + 500-demo policy) + stage-1 cfg for reuse
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
    shutil.copy(os.path.join(os.path.dirname(STAGE1), "cfg.yaml"),
                os.path.join(SAVE_DIR, "cfg.yaml"))
    print(f"saved {SAVE_DIR}/model.pt")

    print(f"\n===== NEW base policy ({LABEL} BC) =====")
    new_po = evaluate(agent, ws, 0, "NEW policy-only")
    new_mppi = evaluate(agent, ws, 1, "NEW MPPI (narrow)")

    print("\n================= SUMMARY (TARGET, sigma=0.1) =================")
    print(f"{'':16s}{'policy-only':>16s}{'MPPI(narrow)':>16s}")
    print(f"{'OLD (20-demo)':16s}{old_po[0]:>10.3f}±{old_po[1]:.3f}{old_mppi[0]:>10.3f}±{old_mppi[1]:.3f}")
    print(f"{('NEW '+LABEL):16s}{new_po[0]:>10.3f}±{new_po[1]:.3f}{new_mppi[0]:>10.3f}±{new_mppi[1]:.3f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
