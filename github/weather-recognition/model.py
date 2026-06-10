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
    def __init__(self, backbone, feature_dim, num_classes, dropout=0.2, model_type=None):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        self.model_type = model_type

    def forward(self, x):
        if self.model_type == "inception":
            # InceptionV3 train 模式返回 (main, aux), eval 返回 main
            if self.training:
                x, _ = self.backbone(x)
            else:
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        x = self.classifier(x)
        return x


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
    )
    return model, backbone_name, feature_dim


model, BACKBONE_NAME, FEATURE_DIM = build_model()
