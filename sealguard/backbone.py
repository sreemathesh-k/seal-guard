"""Feature extractors. Each one maps a list of BGR pocket crops to (N, C, H, W).

  classic : OpenCV colour/texture features. No PyTorch needed. Used for the
            self-test and as the "rules + tiny model" fallback in the deck.
  torch   : pretrained ResNet18 (layer1 + layer2), FP32, for the laptop.
  pte     : the same ResNet18 exported to ExecuTorch (.pte), for the Pi 5.

Enrol and inspect with the SAME backend: INT8 features differ slightly from FP32.
"""
import cv2
import numpy as np

GRID = 32  # feature map is GRID x GRID for a 128 px crop


def to_nchw(crops):
    """BGR uint8 crops -> RGB float32 NCHW in 0..1 (the network normalises internally)."""
    x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


# ---------------------------------------------------------------- classic
def _classic_one(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    L = lab[..., 0]
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2))  # sqrt tames heavy tails
    mu = cv2.GaussianBlur(L, (0, 0), 2)
    std = np.maximum(cv2.GaussianBlur(L * L, (0, 0), 2) - mu * mu, 0) ** 0.25
    chans = [cv2.GaussianBlur(lab[..., i], (0, 0), 1.5) for i in range(3)] + [grad, std]
    return np.stack([cv2.resize(c, (GRID, GRID), interpolation=cv2.INTER_AREA) for c in chans])


class ClassicBackbone:
    name = "classic"

    def __call__(self, crops):
        return np.stack([_classic_one(c) for c in crops])


# ---------------------------------------------------------------- torch
def build_resnet18_features(pretrained=True):
    """ResNet18 trunk returning layer1 and upsampled layer2 features (192 channels, stride 4)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    class ResNet18Features(nn.Module):
        def __init__(self):
            super().__init__()
            w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            m = torchvision.models.resnet18(weights=w)
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
            self.layer1, self.layer2 = m.layer1, m.layer2
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mean) / self.std
            f1 = self.layer1(self.stem(x))
            f2 = F.interpolate(self.layer2(f1), size=f1.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([f1, f2], dim=1)

    return ResNet18Features().eval()


def build_patchcore_features(arch="wide_resnet50_2", out_dim=None, pretrained=True):
    """PatchCore features: layer2 + layer3, 3x3 local averaging, at layer2 resolution (stride 8).

    wide_resnet50_2 -> 1536 channels (most accurate, heavy for the Pi)
    resnet18        ->  384 channels (fast)
    out_dim averages neighbouring channels down to that many, to save memory.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    class PatchCoreFeatures(nn.Module):
        def __init__(self):
            super().__init__()
            if arch == "wide_resnet50_2":
                w = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
                m = torchvision.models.wide_resnet50_2(weights=w)
            elif arch == "resnet18":
                w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
                m = torchvision.models.resnet18(weights=w)
            else:
                raise ValueError(f"unknown arch {arch!r}")
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
            self.layer1, self.layer2, self.layer3 = m.layer1, m.layer2, m.layer3
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mean) / self.std
            f2 = self.layer2(self.layer1(self.stem(x)))
            f3 = self.layer3(f2)
            f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
            f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            f = torch.cat([f2, f3], dim=1)
            if out_dim and f.shape[1] > out_dim:
                n, c, h, w = f.shape
                f = F.adaptive_avg_pool1d(f.permute(0, 2, 3, 1).reshape(-1, 1, c), out_dim)
                f = f.reshape(n, h, w, out_dim).permute(0, 3, 1, 2)
            return f

    return PatchCoreFeatures().eval()


class TorchBackbone:
    name = "torch"

    def __init__(self, pretrained=True, batch=32, model=None):
        import torch
        self.torch, self.batch = torch, batch
        self.model = model if model is not None else build_resnet18_features(pretrained)

    def __call__(self, crops):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(crops), self.batch):
                x = self.torch.from_numpy(to_nchw(crops[i:i + self.batch]))
                out.append(self.model(x).numpy())
        return np.concatenate(out)


# ---------------------------------------------------------------- ExecuTorch
class PteBackbone:
    """Runs the exported backbone .pte (batch size 1) through the ExecuTorch runtime."""
    name = "pte"

    def __init__(self, pte_path):
        import torch
        from executorch.runtime import Runtime
        self.torch = torch
        self.method = Runtime.get().load_program(str(pte_path)).load_method("forward")

    def __call__(self, crops):
        x = to_nchw(crops)
        outs = [self.method.execute([self.torch.from_numpy(x[i:i + 1])])[0] for i in range(len(x))]
        return np.concatenate([np.asarray(o) for o in outs])


def make_backbone(name, pte_path=None):
    if name == "classic":
        return ClassicBackbone()
    if name == "torch":
        return TorchBackbone()
    if name == "pte":
        return PteBackbone(pte_path)
    raise ValueError(f"unknown backend {name!r}")
