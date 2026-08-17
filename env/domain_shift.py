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
    # needle_size_scale > 1.0 THICKENS the needle shaft (nominal 10x10 mm
    # cross-section) instead of shrinking the ring. It is the OTHER way to
    # close the same insertion clearance: the aperture the shaft must thread
    # is (ring_opening - needle_cross_section), and this knob attacks it from
    # the needle side. The needle geom's CENTER is untouched, so
    # geom_xpos["needle_obj_needle"] -- the quantity the 12 mm success check
    # reads -- is unchanged (exactly the ring_size_scale trick, mirrored).
    # Mass/inertia are deliberately NOT rescaled: this is a pure geometry knob
    # (use mass_scale for the inertial axis) so the two stay separable.
    #   1.00 -> 10.0 mm shaft (3.0 mm/side vs a nominal 16 mm ring)
    #   1.20 -> 12.0 mm shaft (2.0 mm/side vs 16 mm; 0.8 mm/side vs a 13.6 mm ring)
    #   1.40 -> 14.0 mm shaft (1.0 mm/side vs 16 mm)
    #   >=1.60 -> shaft >= 16 mm ring opening -> impossible at ring_size 1.0
    # See apply_needle_shift() for the feasibility table.
    needle_size_scale: float = 1.0
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
    # nut_size_scale < 1 SHRINKS the HELD square nut's inner hole (nominal
    # 45.5x45.5 mm) by thickening the 4 frame bars inward -- the direct analog
    # of Threading's ring_size_scale (shrink the opening the held object must
    # thread), but on the manipulated NUT instead of thickening the fixed peg.
    # Geometry: grow each frame bar's hole-facing (thin in-plane) half-size by
    # delta = NUT_HOLE_HALF*(1-scale); symmetric growth keeps the nut body
    # center fixed, so the success check (nut-body-pos vs fixed peg within 3 cm)
    # is unchanged -- only the physical clearance over the peg shrinks. Peg is
    # 32 mm, so the hole must stay > 32 mm: scale floor ~= 0.70 (32/45.5).
    #   1.00 -> 45.5 mm hole (6.75 mm/side clearance, nominal)
    #   0.85 -> 38.7 mm hole (3.35 mm/side; ~ peg_size 1.20's 3.55 mm/side)
    #   0.80 -> 36.4 mm hole (2.2 mm/side)
    #   <=0.70 -> hole <= peg -> impossible
    nut_size_scale: float = 1.0

    # ---- Coffee_D0-only knobs (see resolve_coffee_bindings / apply_coffee_shift) ----
    # pod_holder_size_scale < 1.0 shrinks the pod holder's INNER opening (the
    # 64 `coffee_machine_pod_holder_cup_body_hc_*` boxes forming the cup wall)
    # by thickening each wall box radially. Direct analog of Threading's
    # ring_size_scale. Symmetric growth keeps each box center -- and thus the
    # holder body center -- fixed, and the success check reads Python-side
    # attrs (`env.pod_holder_size` / `env.pod_size`, set from the MujocoObject
    # spec, NOT re-read from sim.model), so `r_diff` stays 5.2 mm no matter
    # what we do here. VERIFIED: 1.5x wall thickening + _setup_references()
    # left r_diff at 0.0052 exactly.
    #
    # !! MEASURED, and it kills the obvious design: the nominal PHYSICAL
    # clearance is only 0.41 mm/side -- cup inner radius 24.97 mm vs the pod
    # mesh's true max radius 24.56 mm (the pod is a truncated cone: 20.71 mm at
    # the bottom, 24.56 mm at the top rim). NOTE 24.56 != `env.pod_size[0]` =
    # 24.30 mm; the latter is the MujocoObject spec's bounding half-size and is
    # 0.26 mm SMALLER than the actual mesh. The 5.2 mm figure people quote is
    # the SUCCESS TOLERANCE (`r_diff`, built from the holder's OUTER half-size
    # 29.5 mm), not a physical clearance -- do not confuse the two.
    #
    # ⛔ EMPIRICALLY THIS KNOB IS A NO-OP (2026-08-08): the 99.8% source expert
    # scored 100% at EVERY setting from 0.995 down to 0.976 (50 games each).
    # Reason: the available travel before the wall reaches the pod is 0.41 mm,
    # and sub-0.2 mm interference is simply absorbed by MuJoCo's contact
    # softness (solimp dmin 0.9) -- the pod is pushed through. Success does
    # genuinely require ~30 mm of real insertion (a rim-seated pod's center sits
    # 30 mm above the band the closed lid allows), so this is not a loose-
    # criterion artifact; the knob just cannot generate enough interference.
    # Use gripper_kp_scale (stage 1) / lid_damping_scale (stage 2) instead.
    # Floor 0.93 = wall thickened >1.7 mm, unambiguously impossible.
    pod_holder_size_scale: float = 1.0
    # Scales the cup wall's sliding friction (nominal 1.0). Contact mu =
    # max(pod, cup); the pod is also 1.0, so >1 grips the insertion.
    pod_holder_friction_scale: float = 1.0
    # ---- lid knobs: Coffee's SECOND stage (Threading/Square have no analog) ----
    # The lid hinge `coffee_machine_lid_main_joint0` is nominally near-free:
    # dof_damping 0.005, frictionloss 0, stiffness 0, range [0, 2.094 rad].
    # Reset always forces it open to 120 deg; success needs it closed < 15 deg.
    # lid_damping_scale scales dof_damping (viscous, resists fast closing).
    lid_damping_scale: float = 1.0
    # lid_frictionloss is ABSOLUTE (N*m), not a scale -- nominal is exactly 0,
    # so a multiplicative knob would be a no-op. Adds dry (Coulomb) friction to
    # the hinge: the lid stops dead unless pushed hard enough. This is the
    # cleanest "second-stage gets harder" lever.
    lid_frictionloss: float = 0.0

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
    """Override an env's reset placement distribution (Threading or Square).

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

    **Square** uses a different mechanism (see _apply_square_init_distribution):
    only `nut_z_rot` applies, and the default is Square_D0's own full circle, so
    enabling this config on Square with no override is a no-op. Square_D0's nut
    z_rot is (0, 2pi) = 360 deg wide, vs Threading_D0's 60 deg -- narrowing it is
    the point of the knob.
    """
    enabled: bool = False
    # ---- Threading: needle (D1 defaults) ----
    needle_x: tuple = (-0.2, 0.05)
    needle_y: tuple = (0.15, 0.25)
    needle_z_rot: tuple = (-7.0 * np.pi / 6.0, np.pi / 6.0)
    # ---- Threading: tripod / hole (D1 defaults) ----
    tripod_x: tuple = (-0.1, 0.15)
    tripod_y: tuple = (-0.2, -0.1)
    tripod_z_rot: tuple = (np.pi / 6.0, 5.0 * np.pi / 6.0)
    # ---- Square: SquareNut z-rotation window (default = Square_D0 as-is) ----
    nut_z_rot: tuple = (0.0, 2.0 * np.pi)


def apply_init_distribution(env, cfg: "InitDistConfig") -> None:
    """Task-dispatched reset-distribution override. No-op unless cfg.enabled.

    Called once at PixelRobosuite.__init__ (after robosuite.make). Threading and
    Square need DIFFERENT mechanisms (monkeypatch vs live-sampler mutation) --
    see each helper for why -- so dispatch on the env class, mirroring
    resolve_bindings(). Square must be checked first: Square_D0 also defines
    _get_initial_placement_bounds, but patching it is a no-op there.
    """
    if not getattr(cfg, "enabled", False):
        return
    cls = env.__class__.__name__
    if cls.startswith("Square"):
        _apply_square_init_distribution(env, cfg)
        return
    if cls.startswith("Threading"):
        _apply_threading_init_distribution(env, cfg)
        return
    raise NotImplementedError(
        f"apply_init_distribution not implemented for {cls}. "
        f"Add a branch (see _apply_threading_init_distribution / "
        f"_apply_square_init_distribution)."
    )


def _apply_square_init_distribution(env, cfg: "InitDistConfig") -> None:
    """Narrow the SquareNut z-rotation window by mutating the live sampler.

    Why not the Threading monkeypatch: Square_D0.__init__ reads
    _get_initial_placement_bounds ONCE and hands the built SequentialCompositeSampler
    to NutAssemblySquare as `placement_initializer`. _load_model only rebuilds it
    when it is None (nut_assembly.py:406), and _reset_internal just calls
    .sample() on the existing object (nut_assembly.py:598) -- so patching the
    bounds method after construction would never be read again. Mutating the
    sampler's `rotation` attr instead persists across hard resets;
    SequentialCompositeSampler.reset() only clears sampled-object state
    (placement_samplers.py:390) and does not touch it.

    Only the SquareNut is touched. RoundNut is deliberately left alone (Square_D0
    parks it off-table at y=(-1.1,-1.0)), as is the peg (Square_D0 has no peg
    sampler at all -- it is fixed; D1/D2 add one).
    """
    samplers = getattr(getattr(env, "placement_initializer", None), "samplers", None)
    if samplers is None or "SquareNutSampler" not in samplers:
        raise NotImplementedError(
            f"{env.__class__.__name__} has no 'SquareNutSampler' in its "
            f"placement_initializer; cannot apply nut_z_rot."
        )
    nut_sampler = samplers["SquareNutSampler"]
    if nut_sampler.rotation_axis != "z":
        raise NotImplementedError(
            f"SquareNutSampler rotates about '{nut_sampler.rotation_axis}', "
            f"expected 'z'."
        )
    nut_sampler.rotation = tuple(cfg.nut_z_rot)


def _apply_threading_init_distribution(env, cfg: "InitDistConfig") -> None:
    """Monkeypatch env._get_initial_placement_bounds to use cfg's ranges.

    Persists across hard resets because _load_model re-invokes the (now patched)
    bound method each reset. Rebuilds the placement_initializer immediately so the
    very first reset already uses the new bounds.
    """
    import types

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

    # The SHAFT geoms only (the thin part that threads the ring), for
    # apply_needle_shift(). The needle body carries 4 geoms:
    # needle_obj_needle{,_vis} (shaft, 5x60x5 mm half-size) and
    # needle_obj_handle{,_vis} (20 mm cube). We keep BOTH the collision geom
    # and its visual twin -- unlike the ring case, a thicker needle is a real,
    # legitimately-visible property of the object (not a hidden gap we must
    # avoid leaking into pixels), so the render should match the physics.
    _ng0 = int(model.body_geomadr[needle_bid])
    _nng = int(model.body_geomnum[needle_bid])
    needle_geom_ids = [
        g for g in range(_ng0, _ng0 + _nng)
        if (model.geom_id2name(g) or "").endswith(("_needle", "_needle_vis"))
    ]

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
        "needle_geom_ids": needle_geom_ids,
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

    # The 4 frame bars forming the nut hole (for nut_size_scale). SquareNut
    # collision geoms whose in-plane center is inside the hole region; the
    # handle geom (at x~=0.054) is excluded by the 0.045 m position gate.
    nut_frame_geom_ids = [
        g for g in object_geom_ids
        if (m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0)
        and max(abs(float(m.geom_pos[g][0])), abs(float(m.geom_pos[g][1]))) < 0.045
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

    # Free joint of the square nut, mirroring Threading's `needle_joint`. Only
    # consumed by the OPTIONAL strict-success speed gate (`max_nut_speed`); the
    # shift paths ignore it.
    nut_joint = "SquareNut_joint0"

    return {
        "object_body_ids": object_body_ids,
        "object_geom_ids": object_geom_ids,
        "peg_geom_ids": peg_geom_ids,
        "nut_frame_geom_ids": nut_frame_geom_ids,
        "gripper_act_ids": gripper_act_ids,
        "arm_body_ids": arm_body_ids,
        "nut_joint": nut_joint,
    }


def resolve_coffee_bindings(env) -> dict:
    """MuJoCo ids for the Coffee_D0 contact shift.

    `env` is the raw robosuite env. `object_body_ids` / `object_geom_ids` /
    `gripper_act_ids` are consumed by apply_contact_shift() AS-IS (pod mass via
    mass_scale, pod friction via friction_scale, gripper kp via
    gripper_kp_scale), plus Coffee-specific keys for apply_coffee_shift():
    `pod_holder_geom_ids` (cup wall: opening size + friction) and `lid_dof_ids`
    (hinge damping / frictionloss). No ring/peg/hole keys -> the Threading- and
    Square-only shift paths are skipped by the wrapper (it branches on binding
    contents).

    NOTE `CoffeePreparation_D0` is a Coffee SUBCLASS with extra objects (mug,
    drawer) and a different success check; it is deliberately NOT accepted here
    until its own knobs are designed.
    """
    cls = env.__class__.__name__
    if not cls.startswith("Coffee_D"):
        raise NotImplementedError(
            f"resolve_coffee_bindings supports Coffee_D0/D1/D2, got {cls}."
        )
    m = env.sim.model

    # The coffee pod: the only manipulated free body. Its root body carries the
    # mass; both geoms (collision `coffee_pod_g0` + `_visual` twin) take the
    # friction shift -- friction on a non-colliding visual geom is a harmless
    # no-op, same convention as Threading/Square.
    pod_bid = env.obj_body_id["coffee_pod"]
    object_body_ids = [pod_bid]
    object_geom_ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == pod_bid]

    # Cup wall = 64 thin boxes `coffee_machine_pod_holder_cup_body_hc_{i}`
    # arranged on a circle of local radius 27.22 mm, each with radial half-size
    # 2.247 mm -> inner opening radius 24.97 mm. Take them from the env (it
    # already resolved these for its contact checks) so we stay in sync with
    # upstream naming; fall back to a name scan if that attr is absent.
    pod_holder_geom_ids = list(getattr(env, "pod_holder_geom_ids", []))
    if not pod_holder_geom_ids:
        pod_holder_geom_ids = [
            g for g in range(m.ngeom)
            if (m.geom_id2name(g) or "").startswith(
                "coffee_machine_pod_holder_cup_body_hc_"
            )
        ]

    # Lid hinge dof (single hinge joint -> single dof address).
    lid_jid = m.joint_name2id("coffee_machine_lid_main_joint0")
    lid_dof_ids = [int(m.jnt_dofadr[lid_jid])]

    gripper_act_ids = [
        m.actuator_name2id(a) for a in env.robots[0].gripper.actuators
    ]

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
        "pod_holder_geom_ids": pod_holder_geom_ids,
        "lid_dof_ids": lid_dof_ids,
        "gripper_act_ids": gripper_act_ids,
        "arm_body_ids": arm_body_ids,
    }


def resolve_bindings(env) -> dict:
    """Task-dispatched binding resolver. Threading_D0 -> threading bindings;
    Square_D0* -> square bindings; Coffee_D0* -> coffee bindings. Raises for
    anything else."""
    cls = env.__class__.__name__
    if cls == "Threading_D0":
        return resolve_threading_bindings(env)
    if cls.startswith("Square"):
        return resolve_square_bindings(env)
    if cls.startswith("Coffee_D"):
        return resolve_coffee_bindings(env)
    raise NotImplementedError(
        f"domain_shift bindings not implemented for {cls}. "
        f"Add a resolver (see resolve_threading_bindings / resolve_square_bindings "
        f"/ resolve_coffee_bindings)."
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
        # Coffee's lid_frictionloss knob mutates this; nominal is all-zero for
        # Threading/Square so restoring it there is a no-op.
        "dof_frictionloss": np.array(m.dof_frictionloss, copy=True),
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
    # .get() so snapshots taken by older code (no dof_frictionloss key) still load.
    if "dof_frictionloss" in nominal:
        m.dof_frictionloss[:] = nominal["dof_frictionloss"]
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


def apply_needle_shift(
    env,
    bindings: dict,
    *,
    needle_size_scale: float = 1.0,
) -> None:
    """THICKEN the needle shaft (Threading target-only, default off).

    The mirror image of `ring_size_scale`: instead of shrinking the ring hole,
    grow the thing that must pass through it. What the insertion actually cares
    about is the CLEARANCE `ring_opening - needle_cross_section`, so both knobs
    move the same physical quantity from opposite sides -- which makes this a
    clean second, independent way to set the "tolerance window" of CLAUDE.md
    §27 ("ceiling = model error / tolerance window").

    Geometry: the shaft geom `needle_obj_needle` is a box with half-size
    (0.005, 0.06, 0.005) -- 10x10 mm cross-section in its LOCAL x/z, 120 mm
    long in y. The ring's aperture (16x16 mm) lies in the tripod's local y/z
    with the hole axis along x, so the shaft threads it cross-section-first;
    both openings are square, so orientation about the shaft axis is moot.
    We scale ONLY the x/z half-sizes; the length (y) and the geom CENTER
    (geom_pos) are untouched, so `geom_xpos["needle_obj_needle"]` -- the exact
    quantity `threading.py:_check_success` compares against the ring center
    with a 12 mm tolerance -- is unchanged. The success criterion therefore
    does not move; only the physical aperture the shaft must clear does.

    Clearance per side vs the ring opening (2*(ring/2 - shaft/2)/2):
        needle 1.00 -> 10.0 mm : 3.0 mm/side @ ring 16.0 | 1.8 mm/side @ 13.6
        needle 1.10 -> 11.0 mm : 2.5 mm/side @ ring 16.0 | 1.3 mm/side @ 13.6
        needle 1.20 -> 12.0 mm : 2.0 mm/side @ ring 16.0 | 0.8 mm/side @ 13.6
        needle 1.40 -> 14.0 mm : 1.0 mm/side @ ring 16.0 | impossible @ 13.6
        needle >=1.60 -> >=16.0 mm : impossible even at a nominal ring
    Safe "harder" range at ring_size 1.0 ~= [1.05, 1.40].

    Mass and inertia are deliberately NOT rescaled with the volume: this stays
    a pure geometry knob so it is separable from `mass_scale` (the inertial
    axis). The shaft's own friction is likewise untouched.

    Assumes restore_nominal() ran first so the scale composes from nominal
    (`geom_size` is in the snapshot). Caller does sim.forward() afterwards.
    """
    if needle_size_scale == 1.0:
        return
    m = env.sim.model
    for g in bindings.get("needle_geom_ids", []):
        # x and z are the cross-section that threads the hole; y is the length.
        m.geom_size[g, 0] *= needle_size_scale
        m.geom_size[g, 2] *= needle_size_scale


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


# Nominal SquareNut inner-hole half-width (45.5 mm hole -> 0.02275 m half).
NOMINAL_NUT_HOLE_HALF = 0.02275


def apply_square_peg_shift(
    env,
    bindings: dict,
    *,
    peg_size_scale: float = 1.0,
    peg_friction_scale: float = 1.0,
    nut_size_scale: float = 1.0,
) -> None:
    """Grow the square peg / raise its friction / shrink the nut hole
    (Square_D0 target-only).

    peg_size_scale (>1 grows the peg's x/y half-size): shrinks the nut-over-peg
    clearance without moving the success threshold (success is nut-body-pos vs
    the FIXED peg-body-pos within 3 cm; peg geom size is irrelevant to it). The
    peg height (half-size[2]) is left alone. Feasibility floor: peg must stay
    inside the 45.5 mm nut hole -> scale < ~1.42.

    peg_friction_scale: scales the peg collision geom's friction (nominal 1.0).
    Contact mu = max(nut, peg) in MuJoCo; the nut sliding mu is ~0.95, so this
    grips the insertion once above ~1.0.

    nut_size_scale (<1 shrinks the HELD nut's hole): the analog of Threading's
    ring_size_scale, on the manipulated nut. Each of the 4 frame bars grows its
    hole-facing (thinner in-plane) half-size by delta = NOMINAL_NUT_HOLE_HALF*
    (1-scale); symmetric growth keeps the nut body center fixed (success
    threshold unchanged), only the physical hole over the peg shrinks. Floor
    ~0.70 (hole must stay > 32 mm peg).

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

    if nut_size_scale != 1.0:
        delta = NOMINAL_NUT_HOLE_HALF * (1.0 - nut_size_scale)
        for g in bindings.get("nut_frame_geom_ids", []):
            # grow the thinner in-plane half-dim (the hole-facing wall) inward;
            # symmetric so the geom center (and nut body center) is unchanged.
            ax = 0 if m.geom_size[g, 0] <= m.geom_size[g, 1] else 1
            m.geom_size[g, ax] += delta


# Cup wall inner opening radius (m): the 64 `..._cup_body_hc_*` boxes sit on a
# circle of local radius 0.02722 with radial half-size 0.00224729, so the pod
# passes through 0.02722 - 0.00224729 = 0.024973. The pod's own radius is
# 0.0243 -> nominal PHYSICAL clearance 0.67 mm/side. (Measured off the built
# model, not read from XML.)
NOMINAL_POD_HOLDER_INNER_R = 0.024973


def apply_coffee_shift(
    env,
    bindings: dict,
    *,
    pod_holder_size_scale: float = 1.0,
    pod_holder_friction_scale: float = 1.0,
    lid_damping_scale: float = 1.0,
    lid_frictionloss: float = 0.0,
) -> None:
    """Shrink the pod holder opening / grip its wall / stiffen the lid hinge
    (Coffee_D0 target-only).

    pod_holder_size_scale (<1 shrinks the cup's inner opening): thickens each
    of the 64 cup-wall boxes RADIALLY by
    `delta = NOMINAL_POD_HOLDER_INNER_R * (1 - scale)`. Growth is symmetric
    about each box center, so the boxes -- and the holder body -- do not move,
    and the success check is untouched (it compares body_xpos against the
    Python-side `r_diff = env.pod_holder_size[0] - env.pod_size[0]` = 5.2 mm,
    which is built from the MujocoObject spec and never re-read from
    sim.model). Feasibility floor: opening must stay > the 24.30 mm pod, i.e.
    scale > 0.973. See the DomainShiftConfig comment for the mm table.

    The radial axis is index 0 for these boxes (half-size
    [0.00225 radial, 0.00145 tangential, 0.028 height]); asserted below rather
    than inferred, because getting it wrong would silently thicken the wall
    tangentially (closing the gaps between boxes) instead of shrinking the bore.

    pod_holder_friction_scale: scales the cup wall's sliding friction (nominal
    1.0). MuJoCo max-combines, and the pod is also 1.0, so >1 raises contact mu.

    lid_damping_scale / lid_frictionloss: the SECOND-stage knob. Nominal hinge
    is near-free (damping 0.005, frictionloss 0), and reset always forces the
    lid open to 120 deg while success needs < 15 deg. Damping is viscous
    (resists fast closing); frictionloss is dry/Coulomb (absolute N*m -- the
    lid simply stops unless pushed hard enough) and is the sharper of the two.

    Assumes restore_nominal() ran first so scales compose from nominal. Caller
    forwards the sim afterward.
    """
    m = env.sim.model

    holder_geoms = bindings.get("pod_holder_geom_ids", [])
    if pod_holder_size_scale != 1.0 and holder_geoms:
        if pod_holder_size_scale <= 0.93:
            raise ValueError(
                f"pod_holder_size_scale={pod_holder_size_scale} thickens the cup "
                f"wall by >1.7 mm, far past the pod's own radius -- insertion is "
                f"physically impossible (floor 0.93)."
            )
        delta = NOMINAL_POD_HOLDER_INNER_R * (1.0 - pod_holder_size_scale)
        for g in holder_geoms:
            # radial axis is 0 for the cup boxes; verify rather than infer.
            assert m.geom_size[g, 0] > m.geom_size[g, 1], (
                f"cup box {m.geom_id2name(g)} half-size {m.geom_size[g]} is not "
                f"radial-major; apply_coffee_shift's axis assumption is wrong."
            )
            m.geom_size[g, 0] += delta

    if pod_holder_friction_scale != 1.0:
        for g in holder_geoms:
            m.geom_friction[g] *= pod_holder_friction_scale

    lid_dofs = bindings.get("lid_dof_ids", [])
    for dof in lid_dofs:
        if lid_damping_scale != 1.0:
            m.dof_damping[dof] *= lid_damping_scale
        if lid_frictionloss != 0.0:
            m.dof_frictionloss[dof] = lid_frictionloss


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
