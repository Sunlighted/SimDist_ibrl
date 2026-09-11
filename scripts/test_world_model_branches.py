"""Construct ALL THREE WorldModel arms on CPU and check their attribute surface.

Iron law #23: `_build` has three mutually exclusive branches that each `return`
(raw_state / pixel / latent), so anything appended to the latent path simply does
not exist on the other two. That is a RUNTIME failure on one arm only, and the
dedup gate does not see it because the gate is state-only.

This runs in seconds on a login node with no GPU and no data. Run it after ANY
change to rl/world_model.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rl.world_model import WorldModel, WorldModelConfig
from rl.pixel_obs_encoder import PixelObsEncoderConfig

OBS, ACT, PROP = 37, 7, 9
CAMS = ["agentview", "robot0_eye_in_hand"]

# Every attribute train_simdist.py / train_simdist_adapt.py read unconditionally.
REQUIRED = ["aux_state", "encoder", "dynamics", "reward", "value", "value_target",
            "policy", "decoder", "enc_pi", "obs_encoder"]


def build(name, **kw):
    cfg = WorldModelConfig(**kw.pop("cfg"))
    m = WorldModel(OBS, ACT, cfg, **kw)
    missing = [a for a in REQUIRED if not hasattr(m, a)]
    heads = m.num_dynamics_heads()
    print(f"{name:14s} heads={heads}  params={sum(p.numel() for p in m.parameters()):,}"
          f"  missing={missing or 'none'}")
    assert not missing, f"{name} arm is missing {missing} (iron law #23)"
    return m


print("constructing every WorldModel branch on CPU...")
build("latent", cfg=dict(encoder_type="latent", num_dynamics_heads=5, horizon=5))
build("raw_state", cfg=dict(encoder_type="identity", num_dynamics_heads=5, horizon=5))
m = build("pixel@96", cfg=dict(encoder_type="pixel", num_dynamics_heads=5, horizon=5,
                               decoder_coef=0.0, shared_encoder=1,
                               pixel=PixelObsEncoderConfig(enc_type="vit")),
          image_shape=(3, 96, 96), cameras=CAMS, prop_dim=PROP)
m224 = build("pixel@224/r18",
             cfg=dict(encoder_type="pixel", num_dynamics_heads=5, horizon=5,
                      decoder_coef=0.0, shared_encoder=1,
                      pixel=PixelObsEncoderConfig(enc_type="resnet18_pretrained")),
             image_shape=(3, 224, 224), cameras=CAMS, prop_dim=PROP)

# aux_state_coef>0 must actually BUILD the head on the pixel arm (it used to be
# silently ignored there), and must add state_dict keys ONLY when on.
off = build("pixel aux=0", cfg=dict(encoder_type="pixel", num_dynamics_heads=5, horizon=5,
                                    decoder_coef=0.0, shared_encoder=1,
                                    aux_state_coef=0.0, aux_state_dim=OBS,
                                    pixel=PixelObsEncoderConfig(enc_type="vit")),
            image_shape=(3, 96, 96), cameras=CAMS, prop_dim=PROP)
on = build("pixel aux=1", cfg=dict(encoder_type="pixel", num_dynamics_heads=5, horizon=5,
                                   decoder_coef=0.0, shared_encoder=1,
                                   aux_state_coef=1.0, aux_state_dim=OBS,
                                   pixel=PixelObsEncoderConfig(enc_type="vit")),
           image_shape=(3, 96, 96), cameras=CAMS, prop_dim=PROP)
assert off.aux_state is None and on.aux_state is not None, "aux_state_coef ignored on pixel"
extra = set(on.state_dict()) - set(off.state_dict())
assert extra and all("aux_state" in k for k in extra), extra
assert not (set(off.state_dict()) - set(on.state_dict())), "aux=0 added keys aux=1 lacks"
print(f"aux_state_coef=0 adds NO state_dict keys; =1 adds {len(extra)} (all aux_state): OK")

# a real forward at 224 through the ImageNet encoder
obs = {c: torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8) for c in CAMS}
obs["prop"] = torch.randn(2, PROP)
with torch.no_grad():
    z = m224.encode(obs)
assert z.shape == (2, m224.cfg.latent_dim), z.shape
print(f"pixel@224 resnet18 encode -> {tuple(z.shape)}: OK")

# BN frozen => per-sample deterministic (update_distill serves t>0 from a
# detached cache; scripts/test_distill_dedup.py is bit-exact)
m224.train()
with torch.no_grad():
    z_batch = m224.encode(obs)
    z_single = m224.encode({k: (v[:1] if v.dim() > 1 else v[:1]) for k, v in obs.items()})
d = (z_batch[:1] - z_single).abs().max().item()
assert d < 1e-5, f"encoder is batch-coupled in train mode (max diff {d}) -- freeze_bn broken"
print(f"resnet18 per-sample deterministic in train() mode (max diff {d:.2e}): OK")
print("\nALL WORLD-MODEL BRANCH CHECKS PASSED")
