"""Multi-camera image encoder for the PIXEL arm of the SimDist world model.

Maps an observation DICT -> a flat feature vector, which `WorldModel`'s existing
trunk MLP then projects to `latent_dim` and SimNorm-normalises. Keeping the
contract at "obs dict in, flat features out" means the planner, all five loss
heads, and every stage-2/3 entry point are untouched: the only thing that
changes is what `WorldModel.encode` accepts.

Design notes (see the plan's Phase 2.1):

* ONE ResNet96 per camera, not a shared trunk. A wrist view and a 3rd-person
  view have completely different statistics; this also matches how QAgent and
  bc/multiview_encoder.py already do it. The cost of not batching both views
  into one kernel is a second fixed ~2.3 ms launch, which is noise.
* `prop` (9-dim eef_pos/quat/gripper_qpos) is concatenated in by default. It is
  produced unconditionally by the env wrapper, it is legitimately available on a
  real robot, and 9 of the 37 state dims ARE prop -- so withholding it would
  force the encoder to regress EE pose from a wrist camera for no scientific
  gain. The 28 object-state dims (needle/tripod pose) still have to come from
  pixels, which is the actual test.
* GroupNorm only (inherited from ResNet96) and no dropout, so `encode` is
  per-sample deterministic. `update_distill` depends on that: it serves every
  timestep beyond t=0 from a detached cache, which is only bit-identical to
  per-site encoding if the encoder has no batch-coupled op. Do NOT swap in a
  BatchNorm backbone without revisiting scripts/test_distill_dedup.py.
* Augmentation is OFF by default here and applied by the caller, because the
  same frame is used as both a network input and (detached) a consistency
  target; augmenting those independently would inject fake camera jitter into
  the transition the dynamics head is trying to learn.

Why not reuse bc/multiview_encoder.py: it hardcodes the 224-oriented
`ResNetEncoder` rather than the 96-native `ResNet96Encoder` we have timings for,
carries obs_horizon logic we do not need, and folds prop into its own repr_dim.
Four BC trainers depend on it, so widening it there has a blast radius this
module avoids. The per-camera-encoder + Linear/LayerNorm/ReLU compress + concat
structure below is copied from it.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from networks.encoder import (ResNet18PretrainedEncoder,
                              ResNet18PretrainedEncoderConfig,
                              ResNet96Encoder, ResNet96EncoderConfig,
                              VitEncoder, VitEncoderConfig)


@dataclass
class PixelObsEncoderConfig:
    # "resnet96" or "vit". MEASURED at our working point (B=512 frames, 7 encode
    # calls/step): ViT embed2/depth1 is 5.9x CHEAPER than ResNet96 (41.5 vs 241.2
    # ms of encoder time per step) at a third of the parameters. It is also what
    # the upstream ibrl pixel configs used for IBRL. ResNet96 gives a finer patch
    # grid (144 vs 121) and a larger repr (36864 vs 15488), which may matter for a
    # ~1.8 mm tolerance window -- so this is a speed/spatial-detail knob, not a
    # free win. DrQEncoder is cheaper still but internally downsizes to 84x84,
    # discarding resolution this task cannot spare.
    # "resnet18_pretrained" is the REFERENCE recipe's encoder and the only one
    # here that carries an ImageNet prior -- which is the whole reason to move to
    # 224 (the prior is largely wasted at 96). It is also the only option that
    # works above 96: min_vit hardcodes 96-based patch counts, and ResNet96 is
    # sized for 96. Default stays "vit" so this is inert.
    enc_type: str = "vit"
    resnet96: ResNet96EncoderConfig = field(default_factory=ResNet96EncoderConfig)
    resnet18: ResNet18PretrainedEncoderConfig = field(
        default_factory=ResNet18PretrainedEncoderConfig)
    vit: VitEncoderConfig = field(
        default_factory=lambda: VitEncoderConfig(embed_style="embed2", depth=1))
    # per-camera compressed feature width (before concat)
    feat_dim: int = 256
    use_prop: int = 1
    # spatial pooling of the conv map before the compress Linear:
    #   "flatten" -- keep every cell (repr_dim 36864 for 96x96 -> 9.4M params/cam,
    #                what bc/multiview_encoder.py does; maximum spatial detail)
    #   "spatial_softmax" -- keypoint coordinates (tiny, robomimic-standard)
    # Mean pooling is deliberately NOT offered: it destroys the spatial precision
    # a ~1.8 mm-tolerance insertion task depends on.
    pool: str = "flatten"
    num_keypoints: int = 32
    # Downsample the STORED frame to this size before encoding. 0 = off (use the
    # frame as stored). Exists so the 224 replay can ALSO feed a 96-native arm
    # (vit / resnet96) without generating a second 340 GB dataset: it is the
    # data-scale control for the 224 run, which otherwise changes resolution,
    # backbone, augmentation and data volume all at once (iron law #9).
    # Bilinear + antialias, matching the wrapper's own resize path.
    resize_to: int = 0


class SpatialSoftmax(nn.Module):
    """Per-channel spatial expectation -> 2 coords per keypoint channel."""

    def __init__(self, in_ch: int, num_keypoints: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, num_keypoints, 1)
        self.num_keypoints = num_keypoints

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        logits = self.proj(feat).flatten(2)                       # [B, K, H*W]
        attn = torch.softmax(logits, dim=-1).view(b, -1, h, w)
        ys = torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype)
        ey = (attn.sum(-1) * ys).sum(-1)                           # [B, K]
        ex = (attn.sum(-2) * xs).sum(-1)                           # [B, K]
        return torch.cat([ex, ey], dim=-1)                         # [B, 2K]


class PixelObsEncoder(nn.Module):
    def __init__(self, image_shape, prop_dim: int, cameras, cfg: PixelObsEncoderConfig):
        super().__init__()
        assert len(image_shape) == 3, f"expected (C,H,W), got {tuple(image_shape)}"
        assert cfg.enc_type in ("resnet96", "vit", "resnet18_pretrained"), \
            f"unsupported enc_type {cfg.enc_type!r}"
        assert len(cameras) > 0, "pixel encoder needs at least one camera"
        self.cfg = cfg
        self.cameras = list(cameras)
        self.use_prop = bool(cfg.use_prop) and prop_dim > 0
        self.prop_dim = int(prop_dim)

        if cfg.resize_to:
            image_shape = (image_shape[0], cfg.resize_to, cfg.resize_to)

        def _mk():
            if cfg.enc_type == "resnet18_pretrained":
                return ResNet18PretrainedEncoder(list(image_shape), cfg.resnet18)
            if cfg.enc_type == "vit":
                # MinVit hardcodes its patch counts for 96x96
                # (networks/min_vit.py:11,31), so changing rl_image_size would
                # silently corrupt pos_embed rather than error.
                return VitEncoder(list(image_shape), cfg.vit)
            return ResNet96Encoder(list(image_shape), cfg.resnet96)

        self.encoders = nn.ModuleList([_mk() for _ in self.cameras])
        enc0 = self.encoders[0]

        if cfg.pool == "flatten":
            in_dim = enc0.repr_dim
            self.pools = nn.ModuleList([nn.Identity() for _ in self.cameras])
        elif cfg.pool == "spatial_softmax":
            in_dim = 2 * cfg.num_keypoints
            self.pools = nn.ModuleList(
                [SpatialSoftmax(enc0.patch_repr_dim, cfg.num_keypoints)
                 for _ in self.cameras])
        else:
            raise ValueError(f"unknown pool {cfg.pool!r}")
        self._pool = cfg.pool

        self.compress = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, cfg.feat_dim),
                          nn.LayerNorm(cfg.feat_dim),
                          nn.ReLU())
            for _ in self.cameras])

        self.out_dim = cfg.feat_dim * len(self.cameras) + (self.prop_dim if self.use_prop else 0)

    def forward(self, obs: dict, augment_fn=None) -> torch.Tensor:
        """obs: {camera: uint8[B,C,H,W] (or [C,H,W]), 'prop': float[B,prop_dim]}."""
        feats = []
        for i, cam in enumerate(self.cameras):
            x = obs[cam]
            if x.dim() == 3:
                x = x.unsqueeze(0)
            # ResNet96Encoder asserts raw 0-255 input and normalises internally,
            # so cast but never pre-scale (networks/encoder.py:33).
            x = x.float()
            if self.cfg.resize_to and x.shape[-1] != self.cfg.resize_to:
                x = torch.nn.functional.interpolate(
                    x, size=(self.cfg.resize_to, self.cfg.resize_to),
                    mode="bilinear", align_corners=False, antialias=True)
            if augment_fn is not None:
                x = augment_fn(x)
            if self._pool == "flatten":
                h = self.encoders[i](x, flatten=True)
            else:
                # [B, num_patch, patch_dim] -> [B, patch_dim, s, s] for pooling
                p = self.encoders[i](x, flatten=False)
                b, n, c = p.shape
                s = int(round(n ** 0.5))
                assert s * s == n, f"non-square patch grid: {n}"
                h = self.pools[i](p.transpose(1, 2).reshape(b, c, s, s))
            feats.append(self.compress[i](h))
        out = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
        if self.use_prop:
            prop = obs["prop"]
            if prop.dim() == 1:
                prop = prop.unsqueeze(0)
            out = torch.cat([out, prop.float()], dim=-1)
        return out
