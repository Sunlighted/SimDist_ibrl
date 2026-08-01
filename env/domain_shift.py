"""Domain shifts for MimicGen Threading_D0.

Mirrors /home/Chenyang/sgft/tdmpc2/envs/domain_shift.py (peg-insert-side) but
adapted to robosuite v1.4.1 + the Threading_D0 needle/tripod scene.

Two mechanisms:
  - "fixed": single target shift (apply_contact_shift + apply_hole_offset)
  - "dr":    per-reset random sampling (DRSampler) — DR baseline

Both rely on resolve_bindings() to find: needle root body, tripod root body +
freejoint, Panda gripper actuators, and arm bodies. The contact-side mutation
walks geoms via body_geomadr / body_geomnum and writes geom_friction,
geom_solref[:,0], geom_solimp[:,0:2] in place; mass/inertia/com are written
on the body itself; gripper kp via actuator_gainprm[:,0] and
actuator_biasprm[:,1] (Panda's gripper is mjACT_POSITION with kp=1000).

Because robosuite uses hard_reset=True (sim is rebuilt from MJCF on every
reset), the caller MUST snapshot nominal once and re-apply (or restore+
re-sample for DR) after every env.reset() — see PixelRobosuite.reset() in
env/robosuite_wrapper.py.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DomainShiftConfig:
    mode: str = "none"  # "none" | "fixed" | "dr"

    # ---- fixed-shift target (used when mode == "fixed") ----
    friction_scale: float = 1.0
    mass_scale: float = 1.0
    solref_scale: float = 1.0
    solimp_scale: float = 1.0
    com_offset: tuple = (0.0, 0.0, 0.0)
    gripper_kp_scale: float = 1.0
    hole_offset: tuple = (0.0, 0.0, 0.0)
    mask_hole_offset_in_obs: bool = True

    # ---- ring/hole precision shifts (target-only; make insertion harder) ----
    # ring_size_scale < 1.0 shrinks the tripod hole opening (nominal 16x16 mm)
    # by growing every ring box geom inward; ring_friction_scale > 1.0 raises
    # the (deliberately low, 0.3) ring sliding friction so the needle catches
    # on the rim. Both demand a cleaner-aligned insertion without changing the
    # 12 mm success radius (which is a fixed object attribute, not read from
    # the model). See apply_ring_shift() for the geometry + safe ranges.
    ring_size_scale: float = 1.0
    ring_friction_scale: float = 1.0
    # ---- TRAP knobs: make the tripod TIP OVER (target-only) ----------------
    # Motivation (CLAUDE.md §33 follow-up): every shift so far only makes the
    # task uniformly HARDER, which leaves the state ORDERING intact -- that is
    # why a frozen source value head keeps working and only dynamics needs to
    # adapt. To test the one regime where value adapt should be NECESSARY, we
    # need a shift that changes REACHABILITY: a state from which success is
    # impossible in target but fine in source. A tipped-over tripod is exactly
    # that (insertion becomes physically impossible, and no demo ever rights
    # it), and it is observable (tripod_quat is in obs["object-state"][17:21],
    # i.e. state[26:30]), which is required -- an unobservable trap could not
    # be fixed by ANY value function.
    #   tripod_mass_scale  < 1 : lighter tripod, easier to knock over
    #   tripod_com_height_scale: scales body_ipos[2] RELATIVE to the body
    #       origin. Nominal ipos_z = -0.028 (com 2.8 cm BELOW the origin =
    #       bottom-heavy = very stable). Scale < 1 raises the com toward the
    #       origin; NEGATIVE values put it ABOVE (top-heavy => tips easily).
    # Both default to 1.0 => byte-for-byte unchanged.
    tripod_mass_scale: float = 1.0
    tripod_com_height_scale: float = 1.0
    # --- Square_D0 (NutAssemblySquare) target knobs (analogous to ring_*) ---
    # peg_size_scale > 1 GROWS the square peg's x/y half-size, shrinking the
    # nut-over-peg clearance (nominal 6.75 mm/side; nut hole 45.5 mm, peg
    # 32 mm). The success check is nut-body-pos vs the FIXED peg-body-pos with a
    # 3 cm tolerance and is unaffected by peg geom size, so growing the peg adds
    # a real insertion obstacle without moving the success threshold (same trick
    # as ring_size_scale). Feasibility floor: peg must stay < 45.5 mm, i.e.
    # peg_size_scale < ~1.42 (half-size < 0.02275). Equivalent to "shrinking the
    # square nut hole" but cleaner (peg is a single fixed geom).
    #   1.00 -> 32.0 mm peg (6.75 mm/side, nominal)
    #   1.25 -> 40.0 mm peg (2.75 mm/side)
    #   1.30 -> 41.6 mm peg (1.95 mm/side, ~Threading-tight)
    #   >=1.42 -> peg >= nut hole -> impossible
    peg_size_scale: float = 1.0
    # peg_friction_scale scales the square peg's sliding friction (nominal 1.0).
    # Grippier peg -> a sloppy approach catches/jams instead of sliding down.
    peg_friction_scale: float = 1.0

    # ---- DR ranges (used when mode == "dr"); centered on 1.0, narrow ----
    # Per sgft design: target shift values (mass=3, fric=0.3, ...) deliberately
    # fall outside these ranges so DR isn't an "oracle" baseline.
    dr_friction_range: tuple = (0.7, 1.3)
    dr_mass_range: tuple = (0.7, 1.5)
    dr_solref_range: tuple = (0.7, 1.5)
    dr_solimp_range: tuple = (0.85, 1.15)
    dr_arm_mass_range: tuple = (0.9, 1.1)
    dr_damping_range: tuple = (0.7, 1.3)
    # Oracle DR extras (sgft's "blind gap" dims). Default ranges are
    # (1.0, 1.0) / (0.0, 0.0) → no-op, so existing DR configs are
    # unaffected. Set to bracket the target values for an oracle baseline.
    dr_gripper_kp_range: tuple = (1.0, 1.0)
    # Half-range for the per-axis hole offset (meters). Sampler draws
    # uniform(low, high) independently for x and y; z is always 0.
    dr_hole_offset_range: tuple = (0.0, 0.0)
    # Ring precision dims (also "blind gap" by default → no-op (1.0, 1.0)).
    dr_ring_size_range: tuple = (1.0, 1.0)
    dr_ring_friction_range: tuple = (1.0, 1.0)
    dr_seed: int = 0


@dataclass
class InitDistConfig:
    """Override the Threading needle/tripod reset placement bounds.

    Threading's reset distribution is defined by Threading._get_initial_placement_bounds()
    (x/y/z_rot uniform ranges, offset from table center). D0 fixes the tripod
    and uses a narrow needle box; D1/D2 widen both. apply_init_distribution()
    monkeypatches that method on the env instance — because robosuite rebuilds
    the model (and re-reads the bounds) on every hard reset, the override
    persists without per-reset re-application.

    Defaults below = the MimicGen **D1** bounds. The recovery-training config
    enlarges `needle_x` / `needle_y` on top of D1 (kept disjoint from the
    tripod region so resets stay collision-free). All angles in RADIANS.

    Geometry guardrails (table is 0.8x0.8 -> x,y in [-0.4, 0.4] in these
    coords; needle long-axis half-extent ~0.08 m, tripod footprint ~0.05 m):
    keep the needle box clear of the tripod box by >= ~0.15 m and inside the
    table with margin, else `ensure_valid_placement` thrashes / resets stall.
    """
    enabled: bool = False
    # needle (D1 defaults)
    needle_x: tuple = (-0.2, 0.05)
    needle_y: tuple = (0.15, 0.25)
    needle_z_rot: tuple = (-7.0 * np.pi / 6.0, np.pi / 6.0)
    # tripod / hole (D1 defaults)
    tripod_x: tuple = (-0.1, 0.15)
    tripod_y: tuple = (-0.2, -0.1)
    tripod_z_rot: tuple = (np.pi / 6.0, 5.0 * np.pi / 6.0)


def apply_init_distribution(env, cfg: "InitDistConfig") -> None:
    """Monkeypatch env._get_initial_placement_bounds to use cfg's ranges.

    Threading-only. Called once at PixelRobosuite.__init__ (after robosuite.make);
    persists across hard resets because _load_model re-invokes the (now patched)
    bound method each reset. Rebuilds the placement_initializer immediately so the
    very first reset already uses the new bounds.
    """
    import types

    if not getattr(cfg, "enabled", False):
        return
    if not hasattr(env, "_get_initial_placement_bounds"):
        raise NotImplementedError(
            "apply_init_distribution only supports Threading-style envs "
            "(needle/tripod placement bounds)."
        )

    needle = dict(x=tuple(cfg.needle_x), y=tuple(cfg.needle_y),
                  z_rot=tuple(cfg.needle_z_rot))
    tripod = dict(x=tuple(cfg.tripod_x), y=tuple(cfg.tripod_y),
                  z_rot=tuple(cfg.tripod_z_rot))

    def _custom_bounds(self_env, _needle=needle, _tripod=tripod):
        ref = self_env.table_offset
        return dict(
            needle=dict(x=_needle["x"], y=_needle["y"], z_rot=_needle["z_rot"],
                        reference=ref),
            tripod=dict(x=_tripod["x"], y=_tripod["y"], z_rot=_tripod["z_rot"],
                        reference=ref),
        )

    env._get_initial_placement_bounds = types.MethodType(_custom_bounds, env)
    # Rebuild the sampler now so the first reset (which may not hard-reset) is
    # already on the new bounds; subsequent hard resets re-call it via _load_model.
    env._get_placement_initializer()


# Nominal half-width of the Threading_D0 ring opening, in meters. The ring is a
# rectangular frame of box geoms (unit half-size 2 mm) with a 16x16 mm inner
# opening, so the half-opening is 8 mm. apply_ring_shift() grows each ring geom
# inward by (1 - ring_size_scale) * this to shrink the opening to
# ring_size_scale * 16 mm. (Confirmed against ring_tripod.py:54-115.)
NOMINAL_HOLE_HALF = 0.008


# ---------------------------------------------------------------------------
# binding resolution
# ---------------------------------------------------------------------------


def resolve_threading_bindings(env) -> dict:
    """Look up MuJoCo ids for everything the shift functions touch.

    `env` is the raw robosuite env (e.g. an instance of Threading_D0), NOT the
    PixelRobosuite wrapper.
    """
    if env.__class__.__name__ != "Threading_D0":
        raise NotImplementedError(
            f"resolve_threading_bindings only supports Threading_D0, got "
            f"{env.__class__.__name__}. Add a registry for other tasks if needed."
        )

    model = env.sim.model

    # Object (needle) — what gets mass/friction shifts. Tripod is NOT here
    # (it's the static target; treated separately via hole offset).
    needle_bid = model.body_name2id(env.needle.root_body)
    object_body_ids = [needle_bid]

    # All geoms under each object body, for per-geom contact shifts.
    object_geom_ids = []
    for bid in object_body_ids:
        g0 = int(model.body_geomadr[bid])
        ng = int(model.body_geomnum[bid])
        object_geom_ids.extend(range(g0, g0 + ng))

    # Needle freejoint — used by the strict-success validator to read the
    # needle's linear/angular velocity (optional speed gate).
    needle_joint = env.needle.joints[0]  # "needle_obj_joint0", a freejoint

    # Tripod (hole) — gets per-reset position offset via its freejoint qpos.
    tripod_bid = model.body_name2id(env.tripod.root_body)
    tripod_joint = env.tripod.joints[0]  # "tripod_obj_joint0", a freejoint

    # Ring geoms (the rectangular frame forming the hole) — get size/friction
    # shifts. The success check (threading.py:_check_success) averages these
    # same geoms' positions, so we only resize/refriction them, never move
    # them. Each ring element has a collision geom ("tripod_obj_ring_N") AND a
    # visual-only twin ("..._vis", contype/conaffinity == 0). We keep ONLY the
    # collision geoms: friction is meaningless on non-colliding geoms, and
    # resizing the visual twin would shrink the *rendered* hole (a stray
    # visual cue in pixel mode) without changing the contact geometry. Names
    # contain "ring" (legs are "tripod_*", posts "post_*"), so the substring
    # filter is unambiguous; we still scope to the tripod body.
    ring_geom_ids = []
    tg0 = int(model.body_geomadr[tripod_bid])
    tng = int(model.body_geomnum[tripod_bid])
    for g in range(tg0, tg0 + tng):
        gname = model.geom_id2name(g)
        collides = model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
        if gname is not None and "ring" in gname and collides:
            ring_geom_ids.append(g)

    # Panda gripper actuators (kp lives at gainprm[:,0]).
    gripper_act_ids = [
        model.actuator_name2id(a) for a in env.robots[0].gripper.actuators
    ]

    # Arm bodies for the DR nuisance dim — match the robot's naming prefix
    # but exclude the gripper prefix.
    arm_prefix = env.robots[0].robot_model.naming_prefix  # "robot0_"
    grip_prefix = env.robots[0].gripper.naming_prefix  # "gripper0_"
    arm_body_ids = []
    for bid in range(model.nbody):
        name = model.body_id2name(bid)
        if name is None:
            continue
        if name.startswith(arm_prefix) and not name.startswith(grip_prefix):
            arm_body_ids.append(bid)

    # Find slices inside obs["object-state"] for tripod_pos and
    # tripod_to_robot0_eef_pos, so we can mask the hole offset out of the
    # concatenated state vector. Order is determined by env.active_observables.
    # We rebuild the offset table by querying the observable specs.
    obj_state_slices = _compute_object_state_slices(env)

    return {
        "object_body_ids": object_body_ids,
        "object_geom_ids": object_geom_ids,
        "needle_joint": needle_joint,
        "tripod_bid": tripod_bid,
        "tripod_joint": tripod_joint,
        "ring_geom_ids": ring_geom_ids,
        "gripper_act_ids": gripper_act_ids,
        "arm_body_ids": arm_body_ids,
        "obj_state_slices": obj_state_slices,
    }


def resolve_square_bindings(env) -> dict:
    """MuJoCo ids for the Square_D0 (NutAssemblySquare) contact shift.

    `env` is the raw robosuite env. Returns a bindings dict whose keys
    `object_body_ids` / `object_geom_ids` / `gripper_act_ids` are consumed by
    apply_contact_shift() AS-IS (nut mass via friction_scale/mass_scale, gripper
    kp via gripper_kp_scale), plus `peg_geom_ids` for apply_square_peg_shift()
    (peg size + peg friction). No tripod/ring/hole_offset keys -> the Threading-
    only shift paths (apply_hole_offset / apply_ring_shift / obs masking) are
    skipped by the wrapper (it branches on binding contents).
    """
    if not env.__class__.__name__.startswith("Square"):
        raise NotImplementedError(
            f"resolve_square_bindings supports Square_D0*, got "
            f"{env.__class__.__name__}."
        )
    m = env.sim.model

    # Square nut: root body holds the mass; collect its geoms (collision +
    # visual, by name prefix) for the friction shift (mirrors Threading, where
    # friction on visual geoms is a harmless no-op).
    nut_bid = m.body_name2id("SquareNut_main")
    object_body_ids = [nut_bid]
    object_geom_ids = [
        g for g in range(m.ngeom)
        if (m.geom_id2name(g) or "").startswith("SquareNut")
    ]

    # peg1 = the square peg the square nut goes onto (peg2 is the round peg).
    # Keep only its COLLISION geom(s); the visual twin (contype/conaffinity==0)
    # must not be resized (would shrink only the rendered peg) or re-frictioned.
    peg_bid = m.body_name2id("peg1")
    peg_geom_ids = [
        g for g in range(m.ngeom)
        if m.geom_bodyid[g] == peg_bid
        and (m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0)
    ]

    gripper_act_ids = [
        m.actuator_name2id(a) for a in env.robots[0].gripper.actuators
    ]

    # Arm bodies for the (currently unused for Square) DR nuisance dim.
    arm_prefix = env.robots[0].robot_model.naming_prefix
    grip_prefix = env.robots[0].gripper.naming_prefix
    arm_body_ids = [
        bid for bid in range(m.nbody)
        if (m.body_id2name(bid) or "").startswith(arm_prefix)
        and not (m.body_id2name(bid) or "").startswith(grip_prefix)
    ]

    return {
        "object_body_ids": object_body_ids,
        "object_geom_ids": object_geom_ids,
        "peg_geom_ids": peg_geom_ids,
        "gripper_act_ids": gripper_act_ids,
        "arm_body_ids": arm_body_ids,
    }


def resolve_bindings(env) -> dict:
    """Task-dispatched binding resolver. Threading_D0 -> threading bindings;
    Square_D0* -> square bindings. Raises for anything else."""
    cls = env.__class__.__name__
    if cls == "Threading_D0":
        return resolve_threading_bindings(env)
    if cls.startswith("Square"):
        return resolve_square_bindings(env)
    raise NotImplementedError(
        f"domain_shift bindings not implemented for {cls}. "
        f"Add a resolver (see resolve_threading_bindings / resolve_square_bindings)."
    )


def _compute_object_state_slices(env) -> dict:
    """Walk env._observables in declaration order, build per-observable byte
    slices into the concatenated `object-state` vector.

    Returns dict: obs_name -> (start, end) into obs["object-state"]. Only
    keeps entries we care about for hole masking.
    """
    # object-state in robosuite is the concat of all observables with modality
    # "object" or matching the env's object-state grouping. For Threading the
    # 8 keys appear in this order: needle_pos, needle_quat, needle_to_eef_pos,
    # needle_to_eef_quat, tripod_pos, tripod_quat, tripod_to_eef_pos,
    # tripod_to_eef_quat -> 3+4+3+4+3+4+3+4 = 28. We compute it generically by
    # iterating the observables that are tagged as "object".
    slices = {}
    cursor = 0
    obs_cache = env._get_observations(force_update=True)
    # Order is the env's observable insertion order; use env.active_observables
    # if available, else fall back to ordered keys with `object-state` excluded.
    observable_names = []
    try:
        # robosuite v1.4.1: env._observables is a dict; each has .modality.
        for name, obsv in env._observables.items():
            if getattr(obsv, "modality", None) == "object":
                observable_names.append(name)
    except Exception:
        # Fallback: hardcode Threading's ordering.
        observable_names = [
            "needle_pos", "needle_quat",
            "needle_to_robot0_eef_pos", "needle_to_robot0_eef_quat",
            "tripod_pos", "tripod_quat",
            "tripod_to_robot0_eef_pos", "tripod_to_robot0_eef_quat",
        ]

    for name in observable_names:
        if name not in obs_cache:
            continue
        n = int(np.asarray(obs_cache[name]).size)
        slices[name] = (cursor, cursor + n)
        cursor += n

    return slices


def snapshot_nominal(env) -> dict:
    """Copy nominal values of all model fields the shift functions will mutate.

    Called once per PixelRobosuite construction, after robosuite.make(). Used
    to restore nominal before re-applying on every reset (because hard_reset
    rebuilds the sim).
    """
    m = env.sim.model
    return {
        "body_mass": np.array(m.body_mass, copy=True),
        "body_inertia": np.array(m.body_inertia, copy=True),
        "body_ipos": np.array(m.body_ipos, copy=True),
        "geom_friction": np.array(m.geom_friction, copy=True),
        "geom_solref": np.array(m.geom_solref, copy=True),
        "geom_solimp": np.array(m.geom_solimp, copy=True),
        "geom_size": np.array(m.geom_size, copy=True),
        "dof_damping": np.array(m.dof_damping, copy=True),
        "actuator_gainprm": np.array(m.actuator_gainprm, copy=True),
        "actuator_biasprm": np.array(m.actuator_biasprm, copy=True),
    }


def restore_nominal(env, nominal: dict) -> None:
    """Write nominal arrays back. Use BEFORE re-applying shift on every reset
    so that scales compose with the original values, not the previously-
    shifted ones. With hard_reset=True robosuite rebuilds the sim so this is
    usually redundant, but cheap insurance against subtle differences."""
    m = env.sim.model
    m.body_mass[:] = nominal["body_mass"]
    m.body_inertia[:] = nominal["body_inertia"]
    m.body_ipos[:] = nominal["body_ipos"]
    m.geom_friction[:] = nominal["geom_friction"]
    m.geom_solref[:] = nominal["geom_solref"]
    m.geom_solimp[:] = nominal["geom_solimp"]
    m.geom_size[:] = nominal["geom_size"]
    m.dof_damping[:] = nominal["dof_damping"]
    m.actuator_gainprm[:] = nominal["actuator_gainprm"]
    m.actuator_biasprm[:] = nominal["actuator_biasprm"]


# ---------------------------------------------------------------------------
# the actual shifts
# ---------------------------------------------------------------------------


def apply_contact_shift(
    env,
    bindings: dict,
    *,
    friction_scale: float = 1.0,
    mass_scale: float = 1.0,
    solref_scale: float = 1.0,
    solimp_scale: float = 1.0,
    com_offset=(0.0, 0.0, 0.0),
    gripper_kp_scale: float = 1.0,
) -> None:
    """In-place mutation of contact-phase model fields.

    Mirrors sgft/tdmpc2/envs/domain_shift.py:apply_contact_shift exactly:
    object bodies get mass/inertia/com shifts; their geoms get friction,
    solref timeconst, solimp dmin/dmax; gripper actuators get kp scaled
    on both gainprm[0] and biasprm[1].
    """
    m = env.sim.model
    com_offset = np.asarray(com_offset, dtype=np.float64)

    for bid in bindings["object_body_ids"]:
        if mass_scale != 1.0:
            m.body_mass[bid] *= mass_scale
            m.body_inertia[bid] *= mass_scale
        if np.any(com_offset != 0):
            m.body_ipos[bid] += com_offset

    for g in bindings["object_geom_ids"]:
        if friction_scale != 1.0:
            m.geom_friction[g] *= friction_scale
        if solref_scale != 1.0:
            m.geom_solref[g, 0] *= solref_scale
        if solimp_scale != 1.0:
            m.geom_solimp[g, 0] = np.clip(
                m.geom_solimp[g, 0] * solimp_scale, 0.0, 0.999
            )
            m.geom_solimp[g, 1] = np.clip(
                m.geom_solimp[g, 1] * solimp_scale, 0.0, 0.999
            )

    if gripper_kp_scale != 1.0:
        for aid in bindings["gripper_act_ids"]:
            m.actuator_gainprm[aid, 0] *= gripper_kp_scale
            if m.actuator_biasprm.shape[1] > 1:
                m.actuator_biasprm[aid, 1] *= gripper_kp_scale


def apply_hole_offset(env, bindings: dict, offset) -> None:
    """Shift the tripod's freejoint qpos by `offset` after env.reset().

    Cannot use model.body_pos because Threading's _reset_internal explicitly
    sets the tripod via freejoint qpos (threading.py:357-365), so any
    body_pos write gets clobbered. The freejoint qpos is the actual reset
    target.

    Caller is responsible for sim.forward() afterwards so data.body_xpos
    catches up.
    """
    offset = np.asarray(offset, dtype=np.float64)
    if not np.any(offset != 0):
        return
    tj = bindings["tripod_joint"]
    qp = env.sim.data.get_joint_qpos(tj).copy()
    qp[:3] += offset
    env.sim.data.set_joint_qpos(tj, qp)


def apply_ring_shift(
    env,
    bindings: dict,
    *,
    ring_size_scale: float = 1.0,
    ring_friction_scale: float = 1.0,
) -> None:
    """Shrink the tripod hole and/or raise the ring's friction (target-only).

    Both knobs make the threading insertion require cleaner alignment, WITHOUT
    touching the 12 mm success radius (that radius is `tripod.ring_size[1]`, a
    Python attribute fixed at object construction, not re-read from the model).

    ring_size_scale (shrink the opening):
        The ring is a rectangular frame of box geoms; the nominal inner opening
        is 16x16 mm. Growing every ring geom's y and z half-size by
        `delta = NOMINAL_HOLE_HALF * (1 - ring_size_scale)` moves each inner
        face toward the center by `delta`, so the opening becomes
        `ring_size_scale * 16 mm` in both axes. geom centers (geom_pos /
        geom_xpos) are untouched, so the success check's averaged ring center
        is unchanged — only the physical aperture the needle must thread
        shrinks. The needle cross-section is 10x10 mm (needle.py), so the
        opening must stay > 10 mm to remain feasible:
            scale 1.00 -> 16.0 mm (nominal, 3.0 mm/side clearance)
            scale 0.85 -> 13.6 mm (1.8 mm/side)   <- suggested target
            scale 0.80 -> 12.8 mm (1.4 mm/side)
            scale 0.75 -> 12.0 mm (1.0 mm/side, very tight)
            scale<=0.625 -> <=10 mm (needle no longer fits -> impossible)
        Safe "harder" range ~= [0.78, 0.95]; the snapshot/restore path lets
        this compose from the nominal geom_size on every reset.

    ring_friction_scale (grippier rim):
        Scales the ring geoms' friction (nominal sliding 0.3, deliberately low
        "to ensure easy insertion"). MuJoCo combines contact friction as the
        element-wise max of the two geoms, and the needle's sliding friction is
        already ~0.09-0.3, so the contact friction is ~= 0.3 * ring_friction_scale
        in the relevant regime. Suggested target 3.0 (-> 0.9 sliding); useful
        range ~= [1.0, 5.0]. Much above that the needle can stick/jam.

    Assumes restore_nominal() ran first so the scale composes from nominal.
    Caller is responsible for sim.forward() afterwards.
    """
    m = env.sim.model
    ring_geom_ids = bindings.get("ring_geom_ids", [])

    if ring_size_scale != 1.0:
        delta = NOMINAL_HOLE_HALF * (1.0 - ring_size_scale)
        for g in ring_geom_ids:
            # Grow the cross-section facing the hole (y, z); leave x (the
            # thickness the needle passes through) alone. Symmetric growth
            # keeps the geom centered, so ring_pos is unchanged.
            m.geom_size[g, 1] += delta
            m.geom_size[g, 2] += delta

    if ring_friction_scale != 1.0:
        for g in ring_geom_ids:
            m.geom_friction[g] *= ring_friction_scale


def apply_tripod_trap_shift(
    env,
    bindings: dict,
    *,
    tripod_mass_scale: float = 1.0,
    tripod_com_height_scale: float = 1.0,
) -> None:
    """Make the tripod easy to TIP OVER (Threading target-only, default off).

    Unlike every other shift here (which uniformly raise difficulty and leave
    the state ordering intact), this one changes REACHABILITY: once the tripod
    is over, insertion is physically impossible and nothing in the demos rights
    it. That makes it the test case for "when is adapting the value head
    actually necessary" -- see CLAUDE.md §33.

    tripod_mass_scale: scales the tripod root body mass (nominal 18.09 g) AND
        its inertia by the same factor, so the density interpretation stays
        consistent (a lighter body of the same shape).
    tripod_com_height_scale: scales body_ipos[2] (nominal -0.028 m, i.e. the
        com sits 2.8 cm BELOW the body origin -> bottom-heavy -> stable).
        <1 raises the com; NEGATIVE flips it above the origin (top-heavy).
        The x/y com components are left alone.

    NOTE: mass/inertia/ipos are all in `snapshot_nominal`, so these scales
    always compose from nominal on every reset (hard_reset rebuilds the sim).
    """
    if tripod_mass_scale == 1.0 and tripod_com_height_scale == 1.0:
        return
    m = env.sim.model
    tb = bindings["tripod_bid"]
    if tripod_mass_scale != 1.0:
        m.body_mass[tb] *= tripod_mass_scale
        m.body_inertia[tb] *= tripod_mass_scale
    if tripod_com_height_scale != 1.0:
        m.body_ipos[tb, 2] *= tripod_com_height_scale


def apply_square_peg_shift(
    env,
    bindings: dict,
    *,
    peg_size_scale: float = 1.0,
    peg_friction_scale: float = 1.0,
) -> None:
    """Grow the square peg and/or raise its friction (Square_D0 target-only).

    peg_size_scale (>1 grows the peg's x/y half-size): shrinks the nut-over-peg
    clearance without moving the success threshold (success is nut-body-pos vs
    the FIXED peg-body-pos within 3 cm; peg geom size is irrelevant to it). The
    peg height (half-size[2]) is left alone. Feasibility floor: peg must stay
    inside the 45.5 mm nut hole -> scale < ~1.42.

    peg_friction_scale: scales the peg collision geom's friction (nominal 1.0).
    Contact mu = max(nut, peg) in MuJoCo; the nut sliding mu is ~0.95, so this
    grips the insertion once above ~1.0.

    Assumes restore_nominal() ran first so scales compose from nominal. Caller
    forwards the sim afterward.
    """
    m = env.sim.model
    peg_geom_ids = bindings.get("peg_geom_ids", [])
    for g in peg_geom_ids:
        if peg_size_scale != 1.0:
            m.geom_size[g, 0] *= peg_size_scale
            m.geom_size[g, 1] *= peg_size_scale
        if peg_friction_scale != 1.0:
            m.geom_friction[g] *= peg_friction_scale


# ---------------------------------------------------------------------------
# DR sampler
# ---------------------------------------------------------------------------


class DRSampler:
    """Per-reset random sampling of contact-phase shifts + nuisance dims.

    Stateful: holds an RNG seeded from cfg.dr_seed so samples are
    reproducible across instances given the same seed. Records the last
    sampled dict on self.last_sample for logging.

    By default does NOT randomize hole_offset, com_offset, or gripper_kp —
    those are sgft's "unknown gap" dims a blind DR would miss. Set
    `dr_hole_offset_range` / `dr_gripper_kp_range` to non-default values to
    include them (oracle DR baseline).
    """

    def __init__(self, cfg: DomainShiftConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.dr_seed) + 7919)
        self.last_sample: Optional[dict] = None

    def sample_and_apply(self, env, bindings: dict) -> None:
        c = self.cfg
        fric = float(self.rng.uniform(*c.dr_friction_range))
        mass = float(self.rng.uniform(*c.dr_mass_range))
        solref = float(self.rng.uniform(*c.dr_solref_range))
        solimp = float(self.rng.uniform(*c.dr_solimp_range))
        arm_m = float(self.rng.uniform(*c.dr_arm_mass_range))
        damp = float(self.rng.uniform(*c.dr_damping_range))
        # Oracle-only extras (no-op when range is degenerate).
        gripper_kp = float(self.rng.uniform(*c.dr_gripper_kp_range))
        hole_x = float(self.rng.uniform(*c.dr_hole_offset_range))
        hole_y = float(self.rng.uniform(*c.dr_hole_offset_range))
        ring_size = float(self.rng.uniform(*c.dr_ring_size_range))
        ring_fric = float(self.rng.uniform(*c.dr_ring_friction_range))

        # Object-side contact shift (includes gripper_kp for oracle DR).
        apply_contact_shift(
            env, bindings,
            friction_scale=fric, mass_scale=mass,
            solref_scale=solref, solimp_scale=solimp,
            gripper_kp_scale=gripper_kp,
        )

        # Hole offset (oracle DR only — default range is (0,0) → no-op).
        if hole_x != 0.0 or hole_y != 0.0:
            apply_hole_offset(env, bindings, (hole_x, hole_y, 0.0))

        # Ring size/friction (oracle DR only — default ranges (1,1) → no-op).
        if ring_size != 1.0 or ring_fric != 1.0:
            apply_ring_shift(
                env, bindings,
                ring_size_scale=ring_size, ring_friction_scale=ring_fric,
            )

        # Nuisance: arm/hand link mass (uniform scalar across arm bodies).
        m = env.sim.model
        if arm_m != 1.0:
            for bid in bindings["arm_body_ids"]:
                m.body_mass[bid] *= arm_m
                m.body_inertia[bid] *= arm_m

        # Nuisance: joint damping (global scalar; object freejoints have
        # damping=0 so the multiplier is a no-op for them).
        if damp != 1.0:
            m.dof_damping[:] *= damp

        self.last_sample = dict(
            friction=fric, mass=mass, solref=solref, solimp=solimp,
            arm_mass=arm_m, damping=damp,
            gripper_kp=gripper_kp, hole_offset=(hole_x, hole_y, 0.0),
            ring_size=ring_size, ring_friction=ring_fric,
        )


# ---------------------------------------------------------------------------
# obs masking helper
# ---------------------------------------------------------------------------


def mask_hole_offset_from_obs(obs: dict, bindings: dict, offset) -> None:
    """Subtract `offset` from the tripod position channels in `obs` so the
    policy sees the nominal hole position (sgft "obs-blind" design).

    Mutates `obs` in place. Handles:
      - obs["tripod_pos"]: world-frame tripod position
      - obs["tripod_to_robot0_eef_pos"]: tripod_xpos - eef_xpos (so shifted
        by +offset; subtract)
      - obs["object-state"]: concatenated state; subtract at the slices
        precomputed in bindings["obj_state_slices"]
    """
    offset = np.asarray(offset, dtype=np.float64)
    if not np.any(offset != 0):
        return

    if "tripod_pos" in obs:
        obs["tripod_pos"] = np.asarray(obs["tripod_pos"]) - offset
    if "tripod_to_robot0_eef_pos" in obs:
        obs["tripod_to_robot0_eef_pos"] = (
            np.asarray(obs["tripod_to_robot0_eef_pos"]) - offset
        )

    slices = bindings.get("obj_state_slices", {})
    if "object-state" in obs and slices:
        obj_state = np.asarray(obs["object-state"]).copy()
        for key in ("tripod_pos", "tripod_to_robot0_eef_pos"):
            if key in slices:
                s, e = slices[key]
                obj_state[s:e] -= offset
        obs["object-state"] = obj_state
