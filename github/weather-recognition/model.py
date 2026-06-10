import torch
from torch import nn
import torchvision.models as models
from config import Common, Train


def build_backbone(name):
    """根据配置创建 backbone，返回 (net, feature_dim)"""
    if name == "resnet50":
        net = models.resnet50(weights="DEFAULT")
        net.fc = nn.Identity()
        dim = 2048
    elif name == "convnext_tiny":
        net = models.convnext_tiny(weights="DEFAULT")
        net.classifier[2] = nn.Identity()
        dim = 768
    else:
        raise ValueError(f"不支持的 backbone: {name}")
    return net, dim


class WeatherModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def build_model():
    """创建模型实例，返回 (model, backbone_name, feature_dim)"""
    backbone_name = getattr(Train, "backbone", "resnet50")
    backbone_net, feature_dim = build_backbone(backbone_name)
    model = WeatherModel(
        backbone=backbone_net,
        feature_dim=feature_dim,
        num_classes=len(Common.labels),
        dropout=0.2,
    )
    return model, backbone_name, feature_dim


model, BACKBONE_NAME, FEATURE_DIM = build_model()
