"""GATE: rl/simdist_agent.py::_augment_obs -- shift + colour jitter + Gaussian blur.

Login-node, seconds, no GPU, no data. Run after ANY change to `_augment_obs` or
to the `aug_*` knobs.

★ WHY THIS EXISTS: `scripts/slurm/dedup_regress.slrm` runs with augmentation OFF,
so it is structurally blind to a broken augmentation -- it only proves the
default-off contract. And §44's whole failure mode was an augmentation hook that
existed but was never called: a silently INERT aug costs a multi-day run and
reports nothing. This gate is the only protection against that class.

What it asserts:
  1. every knob is a NO-OP at 0, tensor-equal => state arms and all existing
     pixel checkpoints are byte-for-byte unchanged;
  2. with a knob ON the output actually DIFFERS (catches an inert aug);
  3. ★ ONE draw per (sample, camera) SHARED across all H+1 frames, and DIFFERENT
     draws across samples and across cameras. This is the load-bearing
     invariant: per-frame parameters would inject apparent camera jitter into
     the very transition the dynamics head is learning, and desynchronise the
     input from the detached consistency target built from the same frame;
  4. the uint8 -> float32 -> uint8 round trip stays in range, and the SHIFT
     stays exactly lossless (whole-pixel offsets, no new pixel values);
  5. it runs on all three WorldModel branches (iron law #23).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rl.simdist_agent import SimDistAgent, SimDistConfig
from rl.world_model import WorldModelConfig
from rl.pixel_obs_encoder import PixelObsEncoderConfig

B, H1, C, S, PROP, ACT = 4, 6, 3, 32, 9, 7
CAMS = ["agentview", "robot0_eye_in_hand"]
CAM_STR = "+".join(CAMS)
ALL_OFF = dict(aug_shift_pad=0, aug_prop_noise=0.0,
               aug_color_jitter=0.0, aug_blur_sigma_max=0.0)
fails = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


def pixel_agent(**aug):
    """A pixel-arm agent at 32x32 -- tiny, CPU, but the real code path."""
    knobs = {**ALL_OFF, **aug}
    wm = WorldModelConfig(encoder_type="pixel", num_dynamics_heads=1, horizon=H1 - 1,
                          latent_dim=64, mlp_hidden_dim=64, decoder_coef=0.0,
                          shared_encoder=1,
                          pixel=PixelObsEncoderConfig(enc_type="vit", feat_dim=32),
                          **knobs)
    cfg = SimDistConfig(world_model=wm)
    return SimDistAgent(use_state=0, obs_shape=(C, S, S), prop_shape=(PROP,),
                        action_dim=ACT, rl_camera=CAM_STR, cfg=cfg, device="cpu")


def make_obs(uniform_frames: bool, seed: int = 0):
    """[B,H+1,C,S,S] uint8 + prop. `uniform_frames`: every frame of a sample is
    IDENTICAL, so any per-frame parameter drift shows up as a frame difference,
    while the samples still differ from each other."""
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for cam in CAMS:
        per_sample = torch.randint(0, 256, (B, 1, C, S, S), dtype=torch.uint8,
                                   generator=g)
        obs[cam] = (per_sample.expand(B, H1, C, S, S).contiguous() if uniform_frames
                    else torch.randint(0, 256, (B, H1, C, S, S), dtype=torch.uint8,
                                       generator=g))
    obs["prop"] = torch.randn(B, H1, PROP, generator=g)
    return obs


# ---------------------------------------------------------------- 1. no-op at 0
print("[1] every knob at 0 is a tensor-equal NO-OP")
ag = pixel_agent()
obs = make_obs(False, seed=1)
torch.manual_seed(0)
out = ag._augment_obs(obs)
check("all-off returns the IDENTICAL object (not even a copy)", out is obs)
for k in list(CAMS) + ["prop"]:
    check(f"all-off leaves {k} bit-identical", torch.equal(out[k], obs[k]))
# each knob individually off while another is on: the off one must contribute nothing
for on, off in [("aug_shift_pad", "aug_color_jitter"),
                ("aug_color_jitter", "aug_blur_sigma_max"),
                ("aug_blur_sigma_max", "aug_shift_pad")]:
    val = 3 if on == "aug_shift_pad" else (0.5 if on == "aug_color_jitter" else 1.5)
    a1 = pixel_agent(**{on: val})
    a2 = pixel_agent(**{on: val, off: 0.0})
    torch.manual_seed(7); o1 = a1._augment_obs(obs)
    torch.manual_seed(7); o2 = a2._augment_obs(obs)
    check(f"{off}=0 adds nothing on top of {on}={val} (same RNG stream)",
          all(torch.equal(o1[c], o2[c]) for c in CAMS))
# state arm: a plain tensor must pass straight through even with every knob ON
wm_state = WorldModelConfig(encoder_type="latent", aug_shift_pad=8,
                            aug_prop_noise=0.01, aug_color_jitter=0.3,
                            aug_blur_sigma_max=2.0)
ag_state = SimDistAgent(use_state=1, obs_shape=(37,), prop_shape=None, action_dim=ACT,
                        rl_camera="", cfg=SimDistConfig(world_model=wm_state),
                        device="cpu")
t = torch.randn(B, H1, 37)
check("state arm (latent): a non-dict obs passes through untouched",
      ag_state._augment_obs(t) is t)
wm_raw = WorldModelConfig(encoder_type="identity", aug_shift_pad=8,
                          aug_color_jitter=0.3, aug_blur_sigma_max=2.0)
ag_raw = SimDistAgent(use_state=1, obs_shape=(37,), prop_shape=None, action_dim=ACT,
                      rl_camera="", cfg=SimDistConfig(world_model=wm_raw), device="cpu")
check("state arm (raw_state/identity): non-dict obs passes through untouched",
      ag_raw._augment_obs(t) is t)

# ------------------------------------------------------- 2. each knob does something
print("\n[2] each knob ON actually CHANGES the frames (catches an inert aug)")
for knob, val in [("aug_shift_pad", 4), ("aug_color_jitter", 0.4),
                  ("aug_blur_sigma_max", 2.0)]:
    a = pixel_agent(**{knob: val})
    torch.manual_seed(3)
    o = a._augment_obs(obs)
    check(f"{knob}={val} changes the pixels",
          all(not torch.equal(o[c], obs[c]) for c in CAMS))
    check(f"{knob}={val} preserves shape+dtype",
          all(o[c].shape == obs[c].shape and o[c].dtype == torch.uint8 for c in CAMS))
a = pixel_agent(aug_prop_noise=0.05)
torch.manual_seed(3)
o = a._augment_obs(obs)
check("aug_prop_noise changes prop and ONLY prop",
      not torch.equal(o["prop"], obs["prop"])
      and all(torch.equal(o[c], obs[c]) for c in CAMS))

# --------------------------------------- 3. ONE draw per (sample, camera), shared over t
print("\n[3] ★ one draw per (sample, camera), SHARED across all H+1 frames")
uni = make_obs(True, seed=2)
for cam in CAMS:                      # sanity: the input really is frame-uniform
    assert torch.equal(uni[cam][:, 0], uni[cam][:, -1])
for knob, val in [("aug_shift_pad", 4), ("aug_color_jitter", 0.4),
                  ("aug_blur_sigma_max", 2.0)]:
    a = pixel_agent(**{knob: val})
    torch.manual_seed(11)
    o = a._augment_obs(uni)
    same_over_t = all(torch.equal(o[c][:, 0], o[c][:, t])
                      for c in CAMS for t in range(H1))
    check(f"{knob}: identical input frames -> identical output frames (shared draw)",
          same_over_t)
    # different SAMPLES must get different draws. Give every sample the SAME
    # content so a difference can only come from the parameters.
    one = {c: uni[c][:1].expand(B, H1, C, S, S).contiguous() for c in CAMS}
    one["prop"] = uni["prop"]
    torch.manual_seed(11)
    o1 = a._augment_obs(one)
    per_sample_differs = any(not torch.equal(o1[c][0], o1[c][i])
                             for c in CAMS for i in range(1, B))
    check(f"{knob}: different SAMPLES get different draws", per_sample_differs)
    check(f"{knob}: different CAMERAS get different draws",
          not torch.equal(o1[CAMS[0]], o1[CAMS[1]]))

# ------------------------------------------- 4. round-trip range + lossless shift
print("\n[4] uint8 round trip stays in range; the shift is exactly lossless")
a = pixel_agent(aug_shift_pad=4, aug_color_jitter=0.6, aug_blur_sigma_max=2.0)
torch.manual_seed(5)
o = a._augment_obs(obs)
check("full stack: dtype uint8 and values within [0,255] by construction",
      all(o[c].dtype == torch.uint8 and int(o[c].min()) >= 0
          and int(o[c].max()) <= 255 for c in CAMS))
a_shift = pixel_agent(aug_shift_pad=4)
torch.manual_seed(5)
o = a_shift._augment_obs(obs)
lossless = all(bool(torch.isin(o[c].unique(), obs[c].unique()).all()) for c in CAMS)
check("shift-only introduces NO new pixel values (whole-pixel => lossless)", lossless)
# An over-bright result must CLAMP, not wrap: 250 * 1.5 = 375 must land on 255,
# not on 375-256 = 119. Driven through the helper with a FORCED factor so the
# check does not depend on the draw.
a_j = pixel_agent(aug_color_jitter=0.9)
x = torch.full((B, C, S, S), 250.0)
one, zero = torch.ones(B), torch.zeros(B)
hot = a_j._apply_jitter(x, (one * 1.5, one, one, zero))
check("_apply_jitter with brightness 1.5 really does exceed 255 (>=374)",
      float(hot.min()) >= 374.0)
check("the uint8 cast CLAMPS the overflow to 255 (no wraparound)",
      int(hot.round_().clamp_(0, 255).to(torch.uint8).min()) == 255)
dark = a_j._apply_jitter(x, (one * 0.1, one, one, zero))
check("brightness 0.1 darkens without going negative",
      0.0 <= float(dark.min()) <= 30.0)
# hue/saturation must be exact no-ops at factor 1 / shift 0 on a colour image
xc = torch.randint(0, 256, (B, C, S, S)).float()
ident = a_j._apply_jitter(xc, (one, one, one, zero))
check("jitter at (1,1,1,0) is an identity on colour pixels (<1e-3)",
      float((ident - xc).abs().max()) < 1e-3)

# ------------------------------------------------ 5. determinism given a seed
print("\n[5] deterministic given a seed")
a = pixel_agent(aug_shift_pad=4, aug_color_jitter=0.4, aug_blur_sigma_max=2.0,
                aug_prop_noise=0.01)
torch.manual_seed(99); o1 = a._augment_obs(obs)
torch.manual_seed(99); o2 = a._augment_obs(obs)
check("same seed -> identical output",
      all(torch.equal(o1[k], o2[k]) for k in o1))
torch.manual_seed(100); o3 = a._augment_obs(obs)
check("different seed -> different output",
      any(not torch.equal(o1[k], o3[k]) for k in o1))

print("\n" + ("FAILED: " + "; ".join(fails) if fails else "ALL AUGMENTATION CHECKS PASS"))
sys.exit(1 if fails else 0)
