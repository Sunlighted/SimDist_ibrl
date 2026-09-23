"""Square (NutAssemblySquare) reset-distribution variants for MimicGen.

Importing this module registers the env(s) with robosuite (EnvMeta auto-registers
any MujocoEnv subclass on definition) WITHOUT editing the vendored mimicgen
submodule.

-------------------------------------------------------------------------------
Square_RealCell = Square_D0 laid out on the MEASURED bench (docs/infra.md §14),
the same arena Threading_RealCell and PegInsertion_D0 use.

WHY THIS NEEDS ITS OWN CLASS RATHER THAN THE WRAPPER'S sim2real HOOK -- three
reasons, all of which would fail SILENTLY if the cell were only applied in the
wrapper (the first two apply to Threading too, the third is Square-specific):

  1. MimicGen (generation) and robomimic (tools/dataset_states_to_obs.py) build
     envs from the env_name stored in the hdf5, never through our wrapper.

  2. `apply_workspace_offset` patches `_get_initial_placement_bounds`, which
     Threading re-reads on every hard reset. SQUARE DOES NOT: Square_D0.__init__
     builds a SequentialCompositeSampler from the bounds ONCE and hands it to
     NutAssemblySquare, whose `_load_model` only rebuilds a sampler when there
     is none (`if self.placement_initializer is None`). So the wrapper's offset
     patch is a NO-OP on Square -- the nut would stay where D0 put it. Here the
     bounds are overridden before __init__ consumes them, which is the only
     point at which Square reads them.

  3. Square's target is not a placed object: peg1/peg2 live in `pegs_arena.xml`
     at FIXED WORLD coordinates (0.23, +-0.1, 0.85), and neither TableArena nor
     PegsArena moves them with `table_offset`. Lifting the table from 0.820 to
     0.933 m therefore BURIES THE PEG: it is 0.1 m half-height about z=0.85, so
     the 0.130 m that stuck out of the stock table becomes 0.017 m -- the task
     is not merely harder, it is geometrically impossible, and nothing raises.
     This class re-places both pegs on the new table surface and applies the
     same workspace offset to them as to the nut, so the nut<->peg geometry
     (i.e. the task) is unchanged and only its position on the bench moves.

Only the MATCHED geometry is applied here (table size/height, base xy, workspace
offset, reset pose). Randomisation (dr_*) stays in the wrapper at RL time: demos
are generated at the nominal cell, which is what a BC prior should imitate.
"""

import os as _os

import numpy as np
from mimicgen.envs.robosuite.nut_assembly import Square_D0

REAL_CELL_YAML = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "release/cfgs/sim2real/square_real_cell.yaml")

# Stock robosuite constants this class re-derives from. Asserted against the
# built sim in _reset_internal, so a robosuite bump cannot move them silently.
_STOCK_TABLE_TOP_Z = 0.82      # NutAssembly.table_offset[2]
_STOCK_PEG_Z = 0.85            # pegs_arena.xml body/@pos, i.e. table top + 0.03
_ASSUMED_BASE_Z = 0.912        # Panda on its default mount; independent of the table


class Square_RealCell(Square_D0):
    """Square_D0 on the measured real-cell arena (release/cfgs/sim2real/square_real_cell.yaml)."""

    # pegs_arena.xml ships TWO pegs -- a square one (peg1, the target) and a
    # round one (peg2) for the RoundNut half of the full NutAssembly task.
    # Square_D0 sets single_object_mode=2, which teleports RoundNut to
    # (10,10,10), but NOT its peg: peg2 stays on the table as a distractor with
    # no counterpart on the real bench.
    #
    # OFF (2026-09-21, user's call on the bench layout): the real cell gets ONE
    # 3D-printed square nut and ONE square peg, so peg2 is parked off-table the
    # same way RoundNut is. It is a switch rather than a deletion because it is
    # the one thing here that makes the scene differ from stock Square_D0, and a
    # future comparison against the public Square datasets would need it back.
    # ⚠️ Flipping it changes the merged task XML, i.e. what MimicGen stores in
    # `model_file` -- it is baked into every generated demo, not a render-time
    # option. Regenerate, do not mix.
    ROUND_PEG = False

    _cell_cache = None

    def _cell(self):
        # Cached per instance: _get_initial_placement_bounds is called from
        # __init__ (before any attribute exists) and again per hard reset.
        if self._cell_cache is None:
            from env.sim2real import load_sim2real
            cfg = load_sim2real(REAL_CELL_YAML)
            assert cfg.enabled and cfg.table_above_base is not None, (
                f"{REAL_CELL_YAML}: Square_RealCell needs enabled=1 and table_above_base")
            self._cell_cache = cfg
        return self._cell_cache

    def _table_top_z(self):
        return _ASSUMED_BASE_Z + float(self._cell().table_above_base)

    # ---------------------------------------------------------------- layout
    def _get_initial_placement_bounds(self):
        """D0's bounds, translated onto the measured bench.

        Called from Square_D0.__init__ BEFORE NutAssemblySquare.__init__, so
        `self.table_offset` does not exist yet -- the reference is computed from
        the cell, not read off the env.
        """
        b = super()._get_initial_placement_bounds()
        cfg = self._cell()
        dx, dy = (float(v) for v in cfg.workspace_offset)
        out = {}
        for k, v in b.items():
            v = dict(v)
            v["x"] = (v["x"][0] + dx, v["x"][1] + dx)
            v["y"] = (v["y"][0] + dy, v["y"][1] + dy)
            v["reference"] = np.array((0.0, 0.0, self._table_top_z()))
            out[k] = v
        return out

    def _load_model(self):
        cfg = self._cell()
        self.table_offset = np.array([0.0, 0.0, self._table_top_z()])
        # Measured arena (docs/real2sim.md §2e). The size must be set BEFORE
        # super() builds the arena; the base xy AFTER, because NutAssembly
        # derives it from table_full_size[0] inside super().
        if cfg.table_full_size is not None:
            assert cfg.base_xy is not None, "table_full_size needs base_xy"
            self.table_full_size = tuple(float(v) for v in cfg.table_full_size)
        # Everything the wrapper's sim2real hook would install is baked in here;
        # mark it so a wrapper built on this env does not stack a second
        # workspace offset / arena patch on top.
        self._s2r_offset_applied = True
        self._s2r_arena_applied = True
        if cfg.init_qpos is not None:
            # Via robot_configs, NOT robots[0].init_qpos: _load_robots
            # RE-INSTANTIATES every Robot from robot_configs on each hard reset
            # (docs/infra.md §14b). robosuite adds the joint init noise on top.
            self.robot_configs[0]["initial_qpos"] = np.array(cfg.init_qpos, dtype=np.float64)
        super()._load_model()
        if cfg.base_xy is not None:
            bx, by = (float(v) for v in cfg.base_xy)
            self.robots[0].robot_model.set_base_xpos((bx, by, 0.0))
        self._place_pegs(cfg)

    def _place_pegs(self, cfg):
        """Re-place the arena's two pegs on the new table, and NAME their geoms.

        The pegs are plain worldbody children of pegs_arena.xml with hardcoded
        world coordinates, so they do not follow `table_offset` (see the module
        docstring). Edited on the merged task XML after super()._load_model(),
        because that XML -- not the live mjModel -- is what MimicGen stores in
        the hdf5 `model_file` and what robomimic replays from.

        Their geoms ship UNNAMED, which makes them invisible to everything that
        works by geom name: the cell's `object_rgb` recolouring, the occlusion
        report in scripts/preview_sim2real.py, and any per-geom domain shift.
        Naming them here is what lets the peg be treated as a task object.
        """
        dx, dy = (float(v) for v in cfg.workspace_offset)
        dz = self._table_top_z() - _STOCK_TABLE_TOP_Z
        for name in ("peg1", "peg2"):
            body = self.model.worldbody.find(f".//body[@name='{name}']")
            assert body is not None, f"{name} not found in the merged task XML"
            pos = np.fromstring(body.get("pos"), sep=" ")
            assert abs(pos[2] - _STOCK_PEG_Z) < 1e-9, (
                f"{name} sits at z={pos[2]}, this class assumes the stock "
                f"{_STOCK_PEG_Z} (pegs_arena.xml changed?)")
            if name == "peg2" and not self.ROUND_PEG:
                # Same disposal robosuite uses for the unused nut: far enough
                # that it is out of every camera and cannot be collided with.
                body.set("pos", "10 10 10")
                continue
            body.set("pos", " ".join(f"{v!r}" for v in
                                     (pos[0] + dx, pos[1] + dy, pos[2] + dz)))
            for i, g in enumerate(body.findall("./geom")):
                if g.get("name") is None:
                    # group 0 = collision, 1 = visual; match the *_vis suffix
                    # convention preview_sim2real.py looks for.
                    g.set("name", f"{name}_{'vis' if g.get('group') == '1' else 'col'}{i}")

    def _reset_internal(self):
        super()._reset_internal()
        # Guard the base-height assumption in _load_model against the BUILT sim:
        # if the mount ever changes, the table (and hence the pegs) would sit at
        # the wrong height relative to the robot with no other symptom.
        self.sim.forward()
        got = float(self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_base")][2])
        assert abs(got - _ASSUMED_BASE_Z) < 1e-6, (
            f"robot base z is {got}, Square_RealCell assumes {_ASSUMED_BASE_Z}")


# ---------------------------------------------------------------------------
# rotw1: the 120-180 deg nut orientation window this project's Square line runs
# on (docs/tasks.md §32/§41 -- source expert 0.98 there vs 0.89 over the full
# 360 deg with the same single-step MLP prior).
# ---------------------------------------------------------------------------
class Square_RealCell_RotW1(Square_RealCell):
    """Square_RealCell restricted to the rotw1 nut-orientation window.

    A REGISTERED CLASS rather than the `init_dist.nut_z_rot` knob, because the
    knob lives in PixelRobosuite (env/domain_shift.py :: apply_init_distribution)
    and MimicGen generation never goes through that wrapper. Defining the window
    in both places is how the demos and the RL env drift apart -- and silently,
    since neither side errors. Here the class name carries the whole task
    definition: demo generation, the BC prior and train_rl.py all name it, and
    `init_dist` stays off.

    ⚠️ The window applies to the SQUARE nut only. RoundNut keeps D0's bounds; it
    is parked off-table anyway (single_object_mode=2).
    """

    # 2*pi/3 .. pi. Matches release/cfgs/mimicgen/square_d0_state_ibrl_source_rotw1_sig01.yaml
    # byte for byte, so the sim2sim rotw1 results remain the reference line.
    NUT_Z_ROT = (2.0943951023931953, 3.141592653589793)

    def _get_initial_placement_bounds(self):
        b = super()._get_initial_placement_bounds()
        b["nut"] = dict(b["nut"])
        b["nut"]["z_rot"] = tuple(float(v) for v in self.NUT_Z_ROT)
        return b
