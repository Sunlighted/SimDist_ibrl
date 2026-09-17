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


# ---------------------------------------------------------------------------
# REAL-CELL variant: the D0 task laid out on the MEASURED bench geometry.
# ---------------------------------------------------------------------------
import os as _os

from mimicgen.envs.robosuite.threading import Threading_D0

REAL_CELL_YAML = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "release/cfgs/sim2real/threading_real_cell.yaml")


class Threading_RealCell(Threading_D0):
    """Threading_D0 with the table height and workspace placement read from the
    sim2real cell YAML -- the SAME file train_rl.py's `sim2real_cell` consumes, so
    the demos, the BC prior and the RL env cannot drift apart.

    Why a registered subclass instead of the PixelRobosuite `sim2real` hook:
    MimicGen (generation) and robomimic (tools/dataset_states_to_obs.py) build
    envs from the env_name stored in the hdf5, never through our wrapper. A
    geometry that only exists inside the wrapper would be silently absent from
    both -- the converted demos would be replayed on the OLD table.

    Only the MATCHED geometry is applied here (table height, workspace offset).
    Randomisation (dr_*) stays in the wrapper at RL time: demos are generated at
    the nominal cell, which is what a BC prior should imitate.

    Same objects, same 37-dim state layout as D0, so the output converts and
    trains exactly like the core D0 dataset.
    """

    def _cell(self):
        from env.sim2real import load_sim2real
        cfg = load_sim2real(REAL_CELL_YAML)
        assert cfg.enabled and cfg.table_above_base is not None, (
            f"{REAL_CELL_YAML}: Threading_RealCell needs enabled=1 and table_above_base")
        return cfg

    def _load_model(self):
        cfg = self._cell()
        # Robot base world z comes from the mount (0.912 for Panda on its default
        # column) and does NOT depend on table_offset -- verified: the base stays
        # at z=0.912 after the lift. So the table can be placed relative to it
        # before the arena is built.
        base_z = 0.912
        self.table_offset = [0.0, 0.0, base_z + float(cfg.table_above_base)]
        import numpy as np
        self.table_offset = np.array(self.table_offset)
        self._cell_offset = tuple(float(v) for v in cfg.workspace_offset)
        self._assumed_base_z = base_z
        super()._load_model()

    def _reset_internal(self):
        super()._reset_internal()
        # Guard the base-height assumption in _load_model against the BUILT sim:
        # if the mount ever changes, the table would sit at the wrong height
        # relative to the robot with no other symptom.
        self.sim.forward()
        got = float(self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_base")][2])
        assert abs(got - self._assumed_base_z) < 1e-6, (
            f"robot base z is {got}, Threading_RealCell assumes {self._assumed_base_z}")

    def _get_initial_placement_bounds(self):
        b = super()._get_initial_placement_bounds()
        dx, dy = getattr(self, "_cell_offset", (0.0, 0.0))
        for k in b:
            b[k] = dict(b[k])
            b[k]["x"] = (b[k]["x"][0] + dx, b[k]["x"][1] + dx)
            b[k]["y"] = (b[k]["y"][0] + dy, b[k]["y"][1] + dy)
        return b
