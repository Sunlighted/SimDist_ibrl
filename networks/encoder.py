from dataclasses import dataclass, field
import torch
from torch import nn

import common_utils
from networks.resnet import ResNet
from networks.resnet_rl import ResNet96
from networks.min_vit import MinVit


@dataclass
class ResNet96EncoderConfig:
    use_1x1: int = 0
    shallow: int = 0


class ResNet96Encoder(nn.Module):
    def __init__(self, obs_shape: list[int], cfg: ResNet96EncoderConfig):
        super().__init__()

        self.resnet = ResNet96(obs_shape[0], cfg.use_1x1, cfg.shallow)
        self.repr_dim, self.num_patch, self.patch_repr_dim = self._get_repr_dim(obs_shape)

    def _get_repr_dim(self, obs_shape: list[int]):
        x = torch.rand(1, *obs_shape)
        y = self.resnet.forward(x).flatten(2, 3)
        repr_dim = y.flatten().size(0)
        _, patch_repr_dim, num_patch = y.size()
        return repr_dim, num_patch, patch_repr_dim

    def forward(self, obs, flatten=True):
        # assert not flatten
        assert obs.max() > 5

        obs = obs / 255.0 - 0.5
        h: torch.Tensor = self.resnet(obs)

        if flatten:
            # mostly for bc
            h = h.flatten(1, -1)
        else:
            # convert to [batch, num_patch, dim] just to be consistent with RL
            h = h.flatten(2, 3).transpose(1, 2)
        return h


@dataclass
class VitEncoderConfig:
    patch_size: int = 8
    depth: int = 3
    embed_dim: int = 128
    num_heads: int = 4
    act_layer = nn.GELU
    stride: int = -1
    embed_style: str = "embed1"
    embed_norm: int = 0


class VitEncoder(nn.Module):
    def __init__(self, obs_shape: list[int], cfg: VitEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.vit = MinVit(
            embed_style=cfg.embed_style,
            embed_dim=cfg.embed_dim,
            embed_norm=cfg.embed_norm,
            num_head=cfg.num_heads,
            depth=cfg.depth,
        )

        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = self.cfg.embed_dim
        self.repr_dim = self.cfg.embed_dim * self.vit.num_patches

    def forward(self, obs, flatten=True) -> torch.Tensor:
        assert obs.max() > 5
        obs = obs / 255.0 - 0.5
        feats: torch.Tensor = self.vit.forward(obs)
        if flatten:
            feats = feats.flatten(1, 2)
        return feats


@dataclass
class ResNetEncoderConfig:
    stem: str = "default"
    downsample: str = "default"
    norm_layer: str = "gnn"
    shallow: int = 0


class ResNetEncoder(nn.Module):
    def __init__(self, obs_shape, cfg: ResNetEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        layers = [1, 1, 1, 1] if self.cfg.shallow else [2, 2, 2, 2]
        self.nets = ResNet(
            stem=self.cfg.stem,
            downsample=self.cfg.downsample,
            norm_layer=self.cfg.norm_layer,
            layers=layers,
            in_channels=obs_shape[0],
        )
        self.repr_dim, self.num_patch, self.patch_repr_dim = self._get_repr_dim(obs_shape)

    def _get_repr_dim(self, obs_shape: list[int]):
        x = torch.rand(1, *obs_shape)
        y = self.nets.forward(x).flatten(2, 3)
        repr_dim = y.flatten().size(0)
        _, patch_repr_dim, num_patch = y.size()
        return repr_dim, num_patch, patch_repr_dim

    def forward(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5
        h = self.nets(obs)
        if flatten:
            h = h.flatten(1, -1)
        else:
            # convert to [batch, num_patch, dim] just to be consistent with RL
            h = h.flatten(2, 3).transpose(1, 2)
        return h


class DrQEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.patch_repr_dim = 32
        self.num_patch = 35 * 35
        self.transform = common_utils.ibrl_utils.get_rescale_transform(84)

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(common_utils.ibrl_utils.orth_weight_init)

    def forward(self, obs, flatten=True):
        if obs.size(-1) != 84:
            obs = self.transform(obs)

        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        if flatten:
            h = h.flatten(1, -1)
        else:
            # convert to [batch, num_patch, dim] just to be consistent with RL
            h = h.flatten(2, 3).transpose(1, 2)
        return h


if __name__ == "__main__":
    import rich.traceback
    import pyrallis

    @dataclass
    class MainConfig:
        net_type: str
        obs_shape: list[int] = field(default_factory=lambda: [3, 96, 96])
        resnet: ResNetEncoderConfig = field(default_factory=lambda: ResNetEncoderConfig())
        vit: VitEncoderConfig = field(default_factory=lambda: VitEncoderConfig())

    rich.traceback.install()
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    if cfg.net_type == "resnet":
        enc = ResNetEncoder(cfg.obs_shape, cfg.resnet)
    elif cfg.net_type == "vit":
        enc = VitEncoder(cfg.obs_shape, cfg.vit)
    else:
        assert False

    print(enc)
    x = torch.rand(1, *cfg.obs_shape) * 255
    print(common_utils.count_parameters(enc))
    print("output size:", enc(x, flatten=False).size())
    print("repr dim:", enc.repr_dim, ", real dim:", enc(x, flatten=True).size())


@dataclass
class ResNet18PretrainedEncoderConfig:
    # ImageNet-pretrained torchvision ResNet-18, the reference recipe's image
    # encoder. NOT interchangeable with ResNet96Encoder:
    #  * it wants ImageNet mean/std, not ResNet96's `x/255 - 0.5`; the
    #    normalisation is applied HERE so callers keep passing raw 0-255 uint8.
    #  * it carries BatchNorm, which is batch-coupled. `WorldModel.update_distill`
    #    serves every t>0 from a DETACHED cache and scripts/test_distill_dedup.py
    #    is bit-exact, so a batch-coupled encoder would break the regression gate.
    #    `freeze_bn=1` (default) keeps every BN in eval mode with frozen running
    #    stats, which restores per-sample determinism AND is the standard way to
    #    finetune a pretrained backbone on a small domain.
    pretrained: int = 1
    freeze_bn: int = 1
    # 0 = train the whole trunk. >0 = freeze the stem + this many residual stages
    # (torchvision layer1..layer4), for the low-data regime.
    freeze_stages: int = 0


class ResNet18PretrainedEncoder(nn.Module):
    """torchvision resnet18 trunk (conv1..layer4), no avgpool/fc.

    Output grid is H/32 x W/32: 7x7 at 224 (repr_dim 25088), 3x3 at 96.
    Weights must be fetched on a LOGIN node -- compute nodes have no egress.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, obs_shape: list[int], cfg: ResNet18PretrainedEncoderConfig):
        super().__init__()
        import torchvision
        from torchvision.models import resnet18, ResNet18_Weights

        self.cfg = cfg
        weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        net = resnet18(weights=weights)
        assert obs_shape[0] == 3, f"resnet18 expects 3-channel input, got {obs_shape}"
        # drop avgpool + fc: we keep the spatial map, like every other encoder here
        self.nets = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                  net.layer1, net.layer2, net.layer3, net.layer4)

        if cfg.freeze_stages > 0:
            # stem is index 0..3, then layer1..layer4 at 4..7
            for i in range(min(4 + cfg.freeze_stages, len(self.nets))):
                for p in self.nets[i].parameters():
                    p.requires_grad_(False)

        self.register_buffer(
            "_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer(
            "_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1) * 255.0)

        self.repr_dim, self.num_patch, self.patch_repr_dim = self._get_repr_dim(obs_shape)

    def _get_repr_dim(self, obs_shape: list[int]):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.rand(1, *obs_shape) * 255.0
            y = self.forward(x, flatten=False)
        self.train(was_training)
        _, num_patch, patch_repr_dim = y.size()
        return num_patch * patch_repr_dim, num_patch, patch_repr_dim

    def train(self, mode: bool = True):
        """Keep BN in eval mode even inside a training loop (see config docstring)."""
        super().train(mode)
        if self.cfg.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, obs, flatten=True):
        # Callers pass RAW 0-255 floats (ResNet96Encoder asserts this); the
        # ImageNet normalisation that torchvision expects happens here.
        obs = (obs - self._mean) / self._std
        h: torch.Tensor = self.nets(obs)
        if flatten:
            h = h.flatten(1, -1)
        else:
            h = h.flatten(2, 3).transpose(1, 2)   # [B, num_patch, dim]
        return h
