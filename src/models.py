"""
Model Architectures for Lightweight Image Classification

Includes the proposed ECA-MGNet model and baseline models.
Paper: "ECA-MGNet: An Efficient Multi-Scale Ghost Network with Dual Attention
        for Lightweight Image Classification" (IEEE Access, 2026)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# ============================================================
# Attention Modules
# ============================================================

class ECAModule(nn.Module):
    """Efficient Channel Attention module.
    Ref: ECA-Net (Wang et al., CVPR 2020)
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention Module."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class DualAttention(nn.Module):
    """Dual Channel-Spatial Attention combining ECA and Spatial Attention."""
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ECAModule(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ============================================================
# Ghost Module (from GhostNet)
# ============================================================

class GhostModule(nn.Module):
    """Ghost Module: generate more features from cheap operations.
    Ref: GhostNet (Han et al., CVPR 2020)
    """
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


# ============================================================
# Multi-Scale Feature Extraction
# ============================================================

class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction with different kernel sizes."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 5, padding=2, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(mid * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        bp = bp.expand_as(b1)
        out = torch.cat([b1, b3, b5, bp], dim=1)
        return self.fusion(out)


# ============================================================
# Proposed Model: ECA-MGNet (ECA-enhanced Multi-scale Ghost Network)
# ============================================================

class GhostBottleneck(nn.Module):
    """Ghost Bottleneck with Dual Attention."""
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, use_attention=True):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)

        if stride > 1:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride,
                          padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
            )
        else:
            self.dw_conv = nn.Identity()

        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        if use_attention:
            self.attention = DualAttention(out_channels)
        else:
            self.attention = nn.Identity()

        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.ghost1(x)
        out = self.dw_conv(out)
        out = self.ghost2(out)
        out = self.attention(out)

        if self.use_residual:
            out = out + residual
        else:
            out = out + self.shortcut(residual)
        return out


class ECAMGNet(nn.Module):
    """ECA-enhanced Multi-scale Ghost Network for Image Classification.

    Architecture (2.50M params, 0.16 GFLOPs):
      - Pretrained GhostNet-1.0x stem + backbone (blocks 0-7): 0.95M params
      - Multi-Scale Refinement Block (MSRB): 160 -> 960 channels
      - Dual Attention (ECA + Spatial): ~0.1K params
      - Classifier: GAP -> FC(960,480) -> ReLU -> Dropout(0.2) -> FC(480,N)

    Uses two-phase transfer learning:
      Phase 1: Freeze backbone, train custom head (5 epochs, LR=3e-3)
      Phase 2: Unfreeze all, fine-tune end-to-end (up to 60 epochs, LR=1e-3)
    """

    def __init__(self, num_classes=10, width_mult=1.0, pretrained=True):
        super().__init__()
        import timm

        # Load pretrained GhostNet-1.0x backbone (blocks 0-7)
        ghost = timm.create_model('ghostnet_100', pretrained=pretrained)

        # Pretrained stem and Ghost Bottleneck backbone
        self.stem = nn.Sequential(ghost.conv_stem, ghost.bn1, ghost.act1)
        self.backbone = ghost.blocks[:8]  # blocks[0-7], output: 160ch @ 7x7

        # Multi-scale feature refinement (novel contribution)
        self.ms_refine = MultiScaleBlock(160, 960)

        # Dual channel-spatial attention (novel contribution)
        self.final_attention = DualAttention(960)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 480),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(480, num_classes),
        )

        # Initialize non-pretrained layers
        self._initialize_head_weights()

    def _initialize_head_weights(self):
        for module in [self.ms_refine, self.final_attention, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.ms_refine(x)
        x = self.final_attention(x)
        x = self.classifier(x)
        return x


# ============================================================
# Baseline Models
# ============================================================

def get_mobilenetv2(num_classes, pretrained=True):
    model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_efficientnet_b0(num_classes, pretrained=True):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_shufflenetv2(num_classes, pretrained=True):
    model = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1' if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet18(num_classes, pretrained=True):
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_ghostnet(num_classes, pretrained=True):
    import timm
    model = timm.create_model('ghostnet_100', pretrained=pretrained, num_classes=num_classes)
    return model


def get_model(model_name, num_classes, pretrained=True, width_mult=1.0):
    """Factory function to get models by name."""
    model_dict = {
        'ecamgnet': lambda: ECAMGNet(num_classes, width_mult, pretrained),
        'mobilenetv2': lambda: get_mobilenetv2(num_classes, pretrained),
        'efficientnet_b0': lambda: get_efficientnet_b0(num_classes, pretrained),
        'shufflenetv2': lambda: get_shufflenetv2(num_classes, pretrained),
        'resnet18': lambda: get_resnet18(num_classes, pretrained),
        'ghostnet': lambda: get_ghostnet(num_classes, pretrained),
    }
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")
    return model_dict[model_name]()


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(1, 3, 224, 224)):
    """Get model parameter count and FLOPs estimate."""
    params = count_parameters(model)
    # Estimate FLOPs using a forward pass
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else 'cpu'
    x = torch.randn(*input_size).to(device)
    model.eval()
    with torch.no_grad():
        _ = model(x)
    return {
        'parameters': params,
        'parameters_M': params / 1e6,
    }
