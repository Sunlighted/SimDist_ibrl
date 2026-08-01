"""Eval the D1 flow-BC prior under the IBRL recovery_d1 run's EXACT eval env
(Threading_D0 + wide eval_init_dist + sigma=0.1 + strict_success + mode=none),
so it is apples-to-apples with threading_recovery_d1_sigma0.1_seed1's score.
"""
import os, sys
import pyrallis
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_rl import MainConfig as RLConfig
import train_bc
from env.robosuite_wrapper import PixelRobosuite
from evaluate import run_eval_mp

IBRL_DIR = "exps/mimicgen/threading_recovery_d1_sigma0.1_seed1"
BC_CKPT = "exps/mimicgen/threading_d1_flow_bc_seed1/model0.pt"
NUM_GAME = 200
SEED = 1

rl_cfg = pyrallis.load(RLConfig, open(os.path.join(IBRL_DIR, "cfg.yaml")))

env_params = dict(
    env_name=rl_cfg.task_name,
    robots=rl_cfg.robots,
    episode_length=rl_cfg.episode_length,
    reward_shaping=False,
    image_size=rl_cfg.image_size,
    rl_image_size=rl_cfg.rl_image_size,
    camera_names=rl_cfg.rl_cameras,
    rl_cameras=rl_cfg.rl_cameras,
    use_state=rl_cfg.use_state,
    obs_stack=rl_cfg.obs_stack,
    state_stack=rl_cfg.state_stack,
    prop_stack=rl_cfg.prop_stack,
    domain_shift=rl_cfg.eval_domain_shift,
    strict_success=rl_cfg.strict_success,
    hand_init_perturb=rl_cfg.eval_hand_init_perturb,
    hand_init_perturb_joint_sigma=rl_cfg.eval_hand_init_perturb_joint_sigma,
    init_distribution=rl_cfg.eval_init_dist,
)

print("=== matched eval env ===")
print("task:", env_params["env_name"], "| strict:", rl_cfg.strict_success.enabled,
      "| sigma:", rl_cfg.eval_hand_init_perturb_joint_sigma,
      "| init_dist.enabled:", rl_cfg.eval_init_dist.enabled,
      "| shift mode:", rl_cfg.eval_domain_shift.mode)

# build a probe env (cuda) just to read obs/action shape, then load the BC policy
probe = PixelRobosuite(**{**env_params, "device": "cuda"})
policy = train_bc._load_model(BC_CKPT, probe, "cuda")
policy.eval()
del probe

scores = run_eval_mp(env_params, policy, num_game=NUM_GAME, num_proc=10,
                     seed=SEED, verbose=False)
sr = sum(scores) / len(scores)
print(f"\n=== D1 flow-BC prior under IBRL's eval env: success = {sr:.3f} "
      f"({int(sum(scores))}/{len(scores)}) ===")
