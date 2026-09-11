#!/bin/bash
# THEN_SBATCH hook for the §48 ladder, dynens stage-1.
# A thin wrapper exists so the THEN_SBATCH value carries NO '=' and no spaces:
# it has to survive being re-forwarded through --export at every chunk hop
# (infra §7z ①/②), and the arch must not depend on an env var propagating.
export ARCH=dynens
exec bash "$(dirname "$0")/launch_aug_ladder_followon.sh"
