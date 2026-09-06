#!/usr/bin/env python3
"""Build the train_rl.py configs for the Square/Coffee model-free `ibrl_m001` arms.

Why a generated yaml and not CLI flags: the target domain is defined by
`{train,eval}_domain_shift` and `{train,eval}_init_dist`, which are nested
dataclasses with ~30 fields each. Passing those on the command line is not
practical, and pyrallis accepts `--config_path` plus CLI overrides for the few
per-seed fields.

Why the env is copied from the STAGE-1 checkpoint and not from the target expert:
the dyna arms take their eval env from the stage-1 cfg (train_simdist_adapt.py:398,
`self.eval_env = self.ws.eval_env`). Two numbers are only comparable if that
environment matches (CLAUDE.md rule 19), so the model-free arm must be built from
the same source of truth the arms it is compared against use.

⚠️ That is not a formality on Coffee: the stage-1 checkpoints and
coffee_ibrl_target_G5x_seed1 disagree on FOUR shift knobs (friction_scale
0.5 vs 0.2, lid_damping_scale 50 vs 5, mass_scale 3 vs 8,
pod_holder_friction_scale 2 vs 5) -- i.e. they are different Coffee targets. This
script follows the stage-1 side. See the printed diff.

    python3 scripts/make_ibrl_m001_cfg.py            # writes both yamls
"""
import os, sys, yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (name, stage-1 cfg = source of truth for the env, template = a real train_rl.py
#  cfg so every field train_rl.py expects is present, expert cfg for the diff)
SPECS = [
    # Threading had NO cfg at first: the arms passed bare CLI flags and inherited
    # train_rl.py's dataclass defaults, which means domain_shift.mode="none" --
    # i.e. they trained AND evaluated in the SOURCE domain while the dyna arms
    # they were meant to be compared against ran in the target (mode="fixed",
    # friction 0.5 / gripper_kp 0.12 / mass 3.0 / ring_friction 2.0 /
    # ring_size 0.85). Same class of bug as act_method defaulting to "rl".
    dict(name="threading_ibrl_m001",
         stage1="exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/cfg.yaml",
         template="exps/mimicgen/threading_ibrl_exp2_nosimdist_seed1/cfg.yaml"),
    dict(name="square_rotw1_ibrl_m001",
         stage1="exps/mimicgen/square_simdist_rotw1_dynens_h5_seed1/cfg.yaml",
         template="exps/mimicgen/square_ibrl_target_B_rotw1_sig01_seed1/cfg.yaml"),
    # ⚠️ coffee_s1g5x_dynens, NOT coffee_simdist_diversev1_h5_seed1: the latter is
    # the C50-era stage-1 on a different target. See docs/infra.md 13.
    dict(name="coffee_g5x_ibrl_m001",
         stage1="exps/mimicgen/coffee_s1g5x_dynens/cfg.yaml",
         template="exps/mimicgen/coffee_ibrl_target_G5x_seed1/cfg.yaml"),
]

# Fields the arm needs at a fixed value. Everything else is inherited from the
# template, which is a per-task tuned train_rl.py run (CLAUDE.md rule 9: these
# were NOT ported from Threading).
BUDGET = {
    "num_train_step": 100_000,      # matched to the dyna arms' 100k transitions
    # 30 for all three tasks. The templates carried 60 (Square/Coffee target
    # experts) while Threading's CLI said 30; with episode lengths of 400 / ~251
    # / ~168 that is 24k / 15k / 5k transitions of target interaction spent on
    # pure-prior rollouts -- up to 24% of the budget, differing per task. Now it
    # is one number, and count_warmup_transitions bills it to global_step so the
    # 100k means the same thing here as it does for the dyna arms.
    "num_warm_up_episode": 30,
    "count_warmup_transitions": 1,
    "num_eval_episode": 100,        # the dyna arms' eval size
    "log_per_step": 1000,           # the dyna arms log every 1000
    "preload_num_data": 20,         # == the dyna arms' --target_num_data 20
    "freeze_bc_replay": 1,
    "prior_margin_init": 0.01,      # CONSTANT margin: init == final
    "prior_margin_final": 0.01,
    "prior_margin_steps": 50_000,
    "mp_eval": 1,
    "use_state": 1,
    # per-seed, overridden on the command line by the job script:
    # σ=0.1 is the standing eval/train perturbation for all three tasks (the dyna
    # arms pass it explicitly on the CLI). Do NOT inherit it from stage-1: the
    # Threading stage-1 cfg has it as None, which silently produced a Threading
    # ibrl arm running at robosuite's default init noise while the dyna arms ran
    # at 0.1. Square/Coffee only escaped because their stage-1 happens to say 0.1.
    "train_hand_init_perturb_joint_sigma": 0.1,
    "eval_hand_init_perturb_joint_sigma": 0.1,
    "seed": 1,
    "save_dir": "exps/PLACEHOLDER",
    "bc_policy": "PLACEHOLDER",
    # The template is a target-expert run that used the source RL agent as its
    # prior; train_rl.py asserts bc_policy and sgft_source_agent are mutually
    # exclusive (train_rl.py:326). This arm's prior is the target BC policy.
    "sgft_source_agent": "",
    "preload_datapath": "PLACEHOLDER",
}
# Copied verbatim from the stage-1 cfg: this IS the target definition.
FROM_STAGE1 = ["task_name", "episode_length", "strict_success"]


def main() -> int:
    os.chdir(REPO)
    for spec in SPECS:
        for k in ("stage1", "template"):
            if not os.path.isfile(spec[k]):
                print(f"MISSING {spec[k]}", file=sys.stderr)
                return 1
        s1 = yaml.safe_load(open(spec["stage1"]))
        cfg = yaml.safe_load(open(spec["template"]))

        # the env: train and eval both in the target domain, both from stage-1
        for dst in ("train_domain_shift", "eval_domain_shift"):
            cfg[dst] = s1["eval_domain_shift"]
        # Threading's stage-1 cfg has no init-distribution block at all (the
        # nut_z_rot/needle_* windows are a Square/Coffee thing), so copy it only
        # when the stage-1 actually defines one; otherwise the template's stays.
        if "eval_init_dist" in s1:
            for dst in ("train_init_dist", "eval_init_dist"):
                cfg[dst] = s1["eval_init_dist"]
        else:
            print(f"   note: {spec['stage1']} has no eval_init_dist -- "
                  f"keeping the template's init distribution")
        for k in FROM_STAGE1:
            if k in s1:
                cfg[k] = s1[k]
        cfg.update(BUDGET)

        out = f"release/cfgs/mimicgen/{spec['name']}.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)

        # report where stage-1 and the target expert disagree -- silence here
        # would hide that Coffee has two incompatible "targets" in the tree.
        tmpl = yaml.safe_load(open(spec["template"]))
        a, b = s1["eval_domain_shift"], tmpl["eval_domain_shift"]
        diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b)
                if a.get(k) != b.get(k) and not (a.get(k) in (1.0, 0.0, None) and k not in b)}
        print(f"{out}")
        print(f"   env from  {spec['stage1']}")
        print(f"   rest from {spec['template']}")
        if diff:
            print("   ⚠️ stage-1 vs target-expert shift disagreement (stage-1 wins):")
            for k, (x, y) in sorted(diff.items()):
                print(f"       {k}: stage1={x}  expert={y}")
        else:
            print("   stage-1 and target expert agree on eval_domain_shift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
