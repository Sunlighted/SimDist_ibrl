"""MimicGen demo generation for PegInsertion_D0 (real2sim peg task), sim only.

MimicGen needs a handful of SOURCE demos carrying raw sim states; here they come
from the scripted expert (scripts/smoke_peg_insertion.py), so no human / real
teleop data is involved. Pipeline:

  1. source   : scripted expert -> robomimic-format hdf5 (states + model_file +
                env_args), then mimicgen's prepare_src_dataset (adds datagen_info)
  2. generate : mimicgen generate_dataset with a 2-subtask spec
                (grasp relative to `peg`, insert relative to `board`)
  3. visualize: scripts/render_mimicgen_demos.py

    python scripts/gen_peg_mimicgen.py source   --num 10
    python scripts/gen_peg_mimicgen.py generate --num 50 --name peg_try1

Why generate at all when the scripted expert exists: MimicGen replays the source
segments OPEN-LOOP after an SE(3) transform, so it is expected to be WORSE than
the closed-loop expert at the ~1.26 mm/side clearance -- measure the generation
success rate before scaling up.
"""

import argparse
import json
import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: F401,E402  (import order: torch before mujoco per repo habit)
import h5py  # noqa: E402
import robosuite  # noqa: E402
from robosuite import load_controller_config  # noqa: E402

import mimicgen  # noqa: F401,E402
import env.peg_insertion_mg  # noqa: F401,E402  (registers PegInsertion_D0 + MG_PegInsertion)
from scripts.smoke_peg_insertion import scripted_expert_action  # noqa: E402

ENV_NAME = "PegInsertion_D0"
SRC_DIR = os.path.join(REPO, "release/data/mimicgen/source")
SRC_PATH = os.path.join(SRC_DIR, "peg_insertion_scripted.hdf5")
GEN_ROOT = os.path.join(REPO, "release/data/mimicgen/generated/peg_insertion")
TEMPLATE = os.path.join(
    os.path.dirname(mimicgen.__file__), "exps/templates/robosuite/square.json")


def env_kwargs(sigma):
    """Everything robomimic needs to rebuild the env; stored as env_args."""
    ctrl = load_controller_config(default_controller="OSC_POSE")
    return dict(
        robots=["Panda"],   # list, as in official mimicgen datasets (train_bc cfg parses list[str])
        controller_configs=ctrl,
        # sigma=0.1 joint-space init noise = the project's demo/RL setting
        initialization_noise={"magnitude": float(sigma), "type": "gaussian"},
        control_freq=20,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
    )


def collect_source(args):
    kw = env_kwargs(args.sigma)
    env = robosuite.make(env_name=ENV_NAME, **kw)
    os.makedirs(SRC_DIR, exist_ok=True)
    accepted, seed = [], args.seed
    while len(accepted) < args.num:
        np.random.seed(seed)
        env.reset()
        xml = env.sim.model.get_xml()
        ctl, states, actions, succ_t = {}, [], [], None
        for t in range(args.max_len):
            states.append(np.array(env.sim.get_state().flatten()))
            a = scripted_expert_action(env, ctl)
            actions.append(a)
            env.step(a)
            if succ_t is None and env._check_success():
                succ_t = t
            # a few extra steps past first success so the insert segment ends settled
            if succ_t is not None and t >= succ_t + args.post_success_steps:
                break
        print(f"seed {seed}: success={succ_t is not None} len={len(actions)}")
        if succ_t is not None:
            accepted.append(dict(states=np.stack(states), actions=np.stack(actions),
                                 xml=xml, seed=seed))
        seed += 1

    with h5py.File(SRC_PATH, "w") as f:
        grp = f.create_group("data")
        env_args = dict(env_name=ENV_NAME, type=1, env_kwargs=kw)
        grp.attrs["env_args"] = json.dumps(env_args)
        total = 0
        for i, ep in enumerate(accepted):
            g = grp.create_group(f"demo_{i}")
            g.create_dataset("states", data=ep["states"])
            g.create_dataset("actions", data=ep["actions"].astype(np.float64))
            g.attrs["model_file"] = ep["xml"]
            g.attrs["num_samples"] = len(ep["actions"])
            g.attrs["seed"] = ep["seed"]
            total += len(ep["actions"])
        grp.attrs["total"] = total
    print(f"wrote {len(accepted)} source demos -> {SRC_PATH}")

    # adds data/demo_i/datagen_info (eef/target poses, object poses, subtask signals)
    sys.argv = ["prepare_src_dataset", "--dataset", SRC_PATH,
                "--env_interface", "MG_PegInsertion", "--env_interface_type", "robosuite"]
    runpy.run_module("mimicgen.scripts.prepare_src_dataset", run_name="__main__")


def generate(args):
    with open(TEMPLATE) as f:
        cfg = json.load(f)
    # `name` selects a REGISTERED mimicgen config class; square's is the same
    # 2-subtask layout, and every field it defines is overridden below
    cfg["name"] = "square"
    exp = cfg["experiment"]
    exp["name"] = args.name
    exp["source"]["dataset_path"] = SRC_PATH
    exp["generation"]["path"] = GEN_ROOT
    # guarantee=1: keep trying until `num` SUCCESSES (num_trials then counts successes)
    exp["generation"]["guarantee"] = bool(args.guarantee)
    exp["generation"]["keep_failed"] = True     # keep failures so they can be viewed too
    exp["generation"]["num_trials"] = args.num
    exp["task"]["name"] = ENV_NAME
    # board reset jitter for the GENERATED scenes (source demos stay fixed-board;
    # subtask 2 is transformed by the board pose, so a moved board is handled).
    # It also lands in the output's env_args, so the converter / BC eval inherit it.
    exp["task"]["env_meta_update_kwargs"] = (
        dict(env_kwargs=dict(board_pos_noise=args.board_pos_noise,
                             board_yaw_noise=args.board_yaw_noise))
        if (args.board_pos_noise or args.board_yaw_noise) else {})
    exp["max_num_failures"] = max(50, 20 * args.num)
    exp["render_video"] = False                 # stock video uses agentview (clips the board)
    exp["seed"] = args.seed
    cfg["obs"]["collect_obs"] = True
    cfg["obs"]["camera_names"] = []             # state-only
    spec = cfg["task"]["task_spec"]
    spec["subtask_1"]["object_ref"] = "peg"
    spec["subtask_2"]["object_ref"] = "board"
    spec["subtask_1"]["action_noise"] = args.grasp_noise
    spec["subtask_2"]["action_noise"] = args.insert_noise

    os.makedirs(GEN_ROOT, exist_ok=True)
    cfg_path = os.path.join(GEN_ROOT, f"{args.name}_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)
    sys.argv = ["generate_dataset", "--config", cfg_path, "--auto-remove-exp"]
    runpy.run_module("mimicgen.scripts.generate_dataset", run_name="__main__")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("source")
    s.add_argument("--num", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--sigma", type=float, default=0.1)
    s.add_argument("--max_len", type=int, default=300)
    s.add_argument("--post_success_steps", type=int, default=5)
    g = sub.add_parser("generate")
    g.add_argument("--num", type=int, default=50, help="number of generation trials")
    g.add_argument("--name", type=str, default="peg_try1")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--guarantee", type=int, default=0)
    g.add_argument("--board_pos_noise", type=float, default=0.0)  # m, uniform +- per axis
    g.add_argument("--board_yaw_noise", type=float, default=0.0)  # deg, uniform +-
    g.add_argument("--grasp_noise", type=float, default=0.05)   # template default
    g.add_argument("--insert_noise", type=float, default=0.05)  # template default
    args = p.parse_args()
    collect_source(args) if args.cmd == "source" else generate(args)


if __name__ == "__main__":
    main()
