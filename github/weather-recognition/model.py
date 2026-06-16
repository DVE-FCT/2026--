import torch
from torch import nn
import torchvision.models as models

from config import Common, Train


# ============================================================
# 2D SE Attention
# ============================================================
class SE2D(nn.Module):
    """Squeeze-and-Excitation for [B, C, H, W] feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"SE2D expects [B,C,H,W], but got shape={tuple(x.shape)}"
            )
        b, c, _, _ = x.shape
        scale = self.avg_pool(x).flatten(1)
        scale = self.mlp(scale).view(b, c, 1, 1)
        return x * scale


class SwinFeatureMapBackbone(nn.Module):
    """Return the final normalized Swin feature map in NCHW format."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.features = net.features
        self.norm = net.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchvision Swin features: [B, H, W, C]
        x = self.features(x)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def build_backbone(name: str, use_se: bool = False):
    """
    Build a pretrained backbone.

    When use_se=True, the backbone must return a 4D feature map so that
    SE2D can operate before global average pooling.
    """
    if name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        dim = 2048
        if use_se:
            # Remove avgpool and fc. Output: [B, 2048, 7, 7] for 224x224.
            net = nn.Sequential(*list(base.children())[:-2])
            return net, dim, "feature_map"
        base.fc = nn.Identity()
        return base, dim, "vector"

    if name == "resnet101":
        base = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        dim = 2048
        if use_se:
            net = nn.Sequential(*list(base.children())[:-2])
            return net, dim, "feature_map"
        base.fc = nn.Identity()
        return base, dim, "vector"

    if name == "swin_t":
        base = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        dim = base.head.in_features
        if use_se:
            return SwinFeatureMapBackbone(base), dim, "feature_map"
        base.head = nn.Identity()
        return base, dim, "vector"

    if name == "convnext_tiny":
        base = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )
        dim = base.classifier[2].in_features
        if use_se:
            return base.features, dim, "feature_map"
        base.classifier[2] = nn.Identity()
        return base, dim, "vector"

    if name == "convnext_small":
        base = models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.DEFAULT
        )
        dim = base.classifier[2].in_features
        if use_se:
            return base.features, dim, "feature_map"
        base.classifier[2] = nn.Identity()
        return base, dim, "vector"

    if name == "efficientnet_b3":
        base = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT
        )
        dim = base.classifier[1].in_features
        if use_se:
            return base.features, dim, "feature_map"
        base.classifier[1] = nn.Identity()
        return base, dim, "vector"

    if name == "inception_v3":
        if use_se:
            raise ValueError(
                "The current implementation does not support final SE2D "
                "for inception_v3. Set use_se_attention=False."
            )
        base = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            aux_logits=True,
        )
        dim = base.fc.in_features
        base.fc = nn.Identity()
        return base, dim, "inception"

    raise ValueError(f"Unsupported backbone: {name}")


class WeatherModel(nn.Module):
    """
    Shared backbone with:
      - fine head: 8 weather classes
      - optional coarse head: ambiguous vs distinctive
      - optional SupCon projection head

    Compatibility:
      model(x) returns only fine logits, so existing test/inference code
      can continue to use the model as a normal 8-class classifier.

      model(x, return_aux=True) returns:
          fine_logits, coarse_logits

      model(x, return_features=True) returns:
          fine_logits, coarse_logits_or_None, features, projection_or_None
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        model_type: str = "vector",
        use_se: bool = False,
        use_coarse_head: bool = False,
        num_coarse_classes: int = 2,
        supcon_dim: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.model_type = model_type
        self.use_se = use_se
        self.use_coarse_head = use_coarse_head

        self.se_block = SE2D(feature_dim) if use_se else None
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fine_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        self.coarse_classifier = (
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_coarse_classes),
            )
            if use_coarse_head
            else None
        )

        self.projection_head = (
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, supcon_dim),
            )
            if supcon_dim > 0
            else None
        )

        self._init_new_heads()

    def _init_new_heads(self) -> None:
        heads = [self.fine_classifier]
        if self.coarse_classifier is not None:
            heads.append(self.coarse_classifier)
        if self.projection_head is not None:
            heads.append(self.projection_head)

        for head in heads:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "inception":
            output = self.backbone(x)
            # During training torchvision Inception may return InceptionOutputs.
            features = output.logits if hasattr(output, "logits") else output
        else:
            features = self.backbone(x)

        if self.use_se:
            if features.dim() != 4:
                raise RuntimeError(
                    "SE attention is enabled, but the backbone did not return "
                    f"a 4D feature map. Got {tuple(features.shape)}."
                )
            features = self.se_block(features)

        if features.dim() == 4:
            features = self.gap(features).flatten(1)
        elif features.dim() != 2:
            raise RuntimeError(
                f"Backbone output must be 2D or 4D, got {tuple(features.shape)}."
            )

        if features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Expected feature_dim={self.feature_dim}, "
                f"but got features={tuple(features.shape)}."
            )
        return features

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_features: bool = False,
    ):
        features = self.extract_features(x)
        fine_logits = self.fine_classifier(features)
        coarse_logits = (
            self.coarse_classifier(features)
            if self.coarse_classifier is not None
            else None
        )

        if return_features:
            projection = (
                self.projection_head(features)
                if self.projection_head is not None
                else None
            )
            return fine_logits, coarse_logits, features, projection

        if return_aux:
            return fine_logits, coarse_logits

        # Keep old test/predict code compatible.
        return fine_logits


def build_model():
    backbone_name = getattr(Train, "backbone", "resnet50")
    use_se = getattr(Train, "use_se_attention", False)
    use_coarse_head = getattr(Train, "use_coarse_head", False)
    use_supcon = getattr(Train, "supcon_enabled", False)

    backbone, feature_dim, model_type = build_backbone(
        backbone_name,
        use_se=use_se,
    )

    model = WeatherModel(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=len(Common.labels),
        dropout=getattr(Train, "dropout", 0.2),
        model_type=model_type,
        use_se=use_se,
        use_coarse_head=use_coarse_head,
        num_coarse_classes=getattr(Train, "num_coarse_classes", 2),
        supcon_dim=(
            getattr(Train, "supcon_dim", 128)
            if use_supcon
            else 0
        ),
    )
    return model, backbone_name, feature_dim
