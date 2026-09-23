"""Write the DR-baseline configs: train under wide randomisation that EXCLUDES
the target, evaluate on the target.

Base = each task's own SOURCE IBRL recipe (verified to match the checkpoint the
stage-1 distillation used), so the only difference from the source expert is
that physics is resampled every reset. No target demonstrations, no target
fine-tuned prior, no target warm-start -- a DR baseline that saw target data
would not be a DR baseline.

Target definitions come from each task's stage-1 cfg, which is the only source
of truth for what "target" means (CLAUDE.md rule 19b).

Every range is asserted to (a) contain 1.0, so the source domain stays in
distribution, and (b) exclude the target value, so DR is not an oracle. The
assert is the point of this script: a DR baseline whose range accidentally
covers the target measures nothing.

    python scripts/make_dr_cfgs.py            # -> release/cfgs/mimicgen/*_dr_wide.yaml
"""
import os
import yaml

CFG = "release/cfgs/mimicgen"
TASKS = {
    "threading": dict(
        base=f"{CFG}/threading_d0_state_ibrl_source_strict.yaml",
        stage1="exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/cfg.yaml",
        steps=800_000,
        dr={"friction": (0.7, 1.4), "mass": (0.5, 2.0), "gripper_kp": (0.5, 1.5),
            "ring_size": (0.9, 1.15), "ring_friction": (0.5, 1.5),
            "solref": (0.7, 1.5), "solimp": (0.85, 1.15),
            "arm_mass": (0.9, 1.1), "damping": (0.7, 1.3)},
    ),
    "coffee": dict(
        base=f"{CFG}/coffee_d0_state_ibrl_source_sig01.yaml",
        stage1="exps/mimicgen/coffee_s1g5x_dynens/cfg.yaml",
        steps=800_000,
        dr={"friction": (0.5, 2.0), "mass": (0.3, 3.0), "gripper_kp": (0.6, 1.5),
            "lid_damping": (0.2, 2.0), "pod_holder_friction": (0.3, 2.5),
            "solref": (0.7, 1.5), "solimp": (0.85, 1.15),
            "arm_mass": (0.9, 1.1), "damping": (0.7, 1.3)},
    ),
    "square": dict(
        base=f"{CFG}/square_d0_state_ibrl_source_rotw1_sig01_snap.yaml",
        stage1="exps/mimicgen/square_simdist_rotw1_dynens_h5_seed1/cfg.yaml",
        steps=800_000,
        dr={"gripper_kp": (0.6, 1.5), "peg_friction": (0.3, 1.5),
            "peg_size": (0.85, 1.1), "friction": (0.7, 1.4), "mass": (0.5, 2.0),
            "solref": (0.7, 1.5), "solimp": (0.85, 1.15),
            "arm_mass": (0.9, 1.1), "damping": (0.7, 1.3)},
    ),
    "tpa": dict(
        base=f"{CFG}/tpa_d0_state_ibrl_source.yaml",
        # TPA keeps its source recipe's 1.5M steps: its episodes are 1000 long,
        # so 800k steps is 2.5x less experience than 800k on the other tasks and
        # its source expert needed 1.5M to learn the task at all.
        stage1="exps/mimicgen/tpa_simdist_plain_diversev1_h5_seed1/cfg.yaml",
        steps=1_500_000,
        dr={"gripper_kp": (0.6, 1.5), "base_friction": (0.3, 1.5),
            "base_hole_size": (1.0, 1.15), "friction": (0.7, 1.4),
            "mass": (0.5, 2.0), "solref": (0.7, 1.5), "solimp": (0.85, 1.15),
            "arm_mass": (0.9, 1.1), "damping": (0.7, 1.3)},
    ),
}
# DR knob -> the fixed-mode field it randomises
FIELD = {k: f"{k}_scale" for k in
         ("friction", "mass", "solref", "solimp", "gripper_kp", "ring_size",
          "ring_friction", "peg_size", "peg_friction", "base_friction",
          "lid_damping", "pod_holder_friction")}
FIELD["base_hole_size"] = "base_hole_size_scale"
FIELD["arm_mass"] = None            # nuisance dims, no fixed-mode counterpart
FIELD["damping"] = None


def main():
    for task, spec in TASKS.items():
        base = yaml.safe_load(open(spec["base"]))
        tgt = yaml.safe_load(open(spec["stage1"]))["eval_domain_shift"]

        for knob, (lo, hi) in spec["dr"].items():
            assert lo <= 1.0 <= hi, f"{task}/{knob}: range {lo,hi} excludes source (1.0)"
            f = FIELD[knob]
            if f is None:
                continue
            tv = tgt.get(f, 1.0)
            if tv == 1.0:
                continue            # not part of this target; nothing to exclude
            assert not (lo <= tv <= hi), (
                f"{task}/{knob}: DR range {lo,hi} COVERS the target value {tv} "
                f"-- that makes this an oracle DR baseline, not a DR baseline")

        out = dict(base)
        out["num_train_step"] = spec["steps"]
        out["train_domain_shift"] = {
            "mode": "dr", "dr_seed": 1, "mask_hole_offset_in_obs": False,
            **{f"dr_{k}_range": list(v) for k, v in spec["dr"].items()},
        }
        out["eval_domain_shift"] = tgt
        out["strict_success"] = yaml.safe_load(open(spec["stage1"])).get("strict_success")
        # sigma is forced to 0.1 for ALL four rather than inherited: Threading's
        # and TPA's stage-1 cfgs leave it unset (their adapt runs pass it on the
        # command line), so inheriting would evaluate two of the four arms
        # without hand perturbation and make the row incomparable.
        out["eval_hand_init_perturb_joint_sigma"] = 0.1
        out["use_wb"] = 1

        path = f"{CFG}/{task}_dr_wide.yaml"
        with open(path, "w") as fh:
            yaml.safe_dump(out, fh, sort_keys=False, default_flow_style=False)
        covered = {FIELD[k]: tgt.get(FIELD[k]) for k in spec["dr"]
                   if FIELD[k] and tgt.get(FIELD[k], 1.0) != 1.0}
        print(f"{task:10s} -> {path}")
        print(f"{'':10s}    steps={spec['steps']}  strict="
              f"{(out['strict_success'] or {}).get('enabled')}  "
              f"sigma={out['eval_hand_init_perturb_joint_sigma']}")
        print(f"{'':10s}    target knobs randomised around (all excluded): {covered}")


if __name__ == "__main__":
    main()
