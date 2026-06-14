import torch
from torch import nn
import torchvision.models as models
from config import Common, Train


def build_backbone(name):
    """根据配置创建 backbone，返回 (net, feature_dim)。feature_dim 自动从预训练模型读取"""
    if name == "resnet50":
        net = models.resnet50(weights="DEFAULT")
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feature_dim

    if name == "resnet101":
        net = models.resnet101(weights="DEFAULT")
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feature_dim

    if name == "convnext_tiny":
        net = models.convnext_tiny(weights="DEFAULT")
        feature_dim = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
        return net, feature_dim

    if name == "convnext_small":
        net = models.convnext_small(weights="DEFAULT")
        feature_dim = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
        return net, feature_dim

    if name == "inception_v3":
        net = models.inception_v3(weights="DEFAULT", aux_logits=True)  # 预训练权重要求
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feature_dim, "inception"  # 特殊标记

    if name == "efficientnet_b3":
        net = models.efficientnet_b3(weights="DEFAULT")
        feature_dim = net.classifier[1].in_features
        net.classifier[1] = nn.Identity()
        return net, feature_dim

    raise ValueError(f"不支持的 backbone: {name}")


class WeatherModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes, dropout=0.2, model_type=None,
                 supcon_dim=128):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        self.model_type = model_type
        self.supcon_dim = supcon_dim
        if supcon_dim > 0:
            self.projection_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, supcon_dim)
            )
        else:
            self.projection_head = None

    def forward(self, x, return_features=False):
        if self.model_type == "inception":
            if self.training:
                features, _ = self.backbone(x)
            else:
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        logits = self.classifier(features)
        if return_features:
            z = self.projection_head(features) if self.projection_head is not None else features
            return logits, features, z
        return logits


def build_model():
    """创建模型实例，返回 (model, backbone_name, feature_dim)"""
    backbone_name = getattr(Train, "backbone", "resnet50")
    result = build_backbone(backbone_name)
    backbone_net, feature_dim = result[0], result[1]
    model_type = result[2] if len(result) > 2 else None
    model = WeatherModel(
        backbone=backbone_net,
        feature_dim=feature_dim,
        num_classes=len(Common.labels),
        dropout=0.2,
        model_type=model_type,
        supcon_dim=getattr(Train, "supcon_dim", 128),
    )
    return model, backbone_name, feature_dim


model, BACKBONE_NAME, FEATURE_DIM = build_model()
