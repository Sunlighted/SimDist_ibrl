"""Custom Threading reset-distribution variants for MimicGen data generation.

Importing this module registers the env(s) with robosuite (robosuite's EnvMeta
metaclass auto-registers any MujocoEnv subclass on definition) WITHOUT editing
the vendored mimicgen submodule. MimicGen's generate_dataset.py can then target
`Threading_Enlarged` (pass --task_name Threading_Enlarged, or set it in the
config) as long as this module is imported in the generating process.

Threading_Enlarged = the DATA-GENERATION reset distribution (deliberately a
superset of the RL "enlarged" env so the generated BC prior over-covers it):
  * needle box x[-0.25, 0.2] y[-0.05, 0.28]  (vs D1's x[-0.2,0.05] y[0.15,0.25];
    vs the RL enlarged env's x[-0.2,0.1] y[-0.05,0.25] — this is larger on both
    axes so demos cover beyond what the RL policy is trained on).
  * tripod + rotations = MimicGen D1.
Same objects/state space as D0/D1 (37-dim), so the generated demos convert and
train exactly like the D1 dataset.
"""

import numpy as np
from mimicgen.envs.robosuite.threading import Threading_D1


class Threading_Enlarged(Threading_D1):
    """D1 tripod/rotations + an enlarged needle placement box (data gen)."""

    def _get_initial_placement_bounds(self):
        return dict(
            needle=dict(
                x=(-0.25, 0.2),
                y=(-0.05, 0.28),
                # D1 near-full needle rotation
                z_rot=(-7.0 * np.pi / 6.0, np.pi / 6.0),
                reference=self.table_offset,
            ),
            tripod=dict(
                x=(-0.1, 0.15),
                y=(-0.2, -0.1),
                z_rot=(np.pi / 6.0, 5.0 * np.pi / 6.0),
                reference=self.table_offset,
            ),
        )
