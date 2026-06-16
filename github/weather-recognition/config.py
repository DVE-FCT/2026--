import time
import random
import numpy as np
import torch


# ============================================================
# 可复现性
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Keep the original performance-oriented setup.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class Common:
    """通用配置."""

    basePath = (
        "C:/Users/Lenovo/OneDrive/Desktop/caip_code/data/"
        "RSCM/classification/weather_classification/"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imageSize = (224, 224)
    labels = [
        "cloudy",
        "haze",
        "rainy",
        "shine",
        "snow",
        "sunny",
        "sunrise",
        "thunder",
    ]


class Train:
    """训练配置."""

    batch_size = 128
    num_workers = 4
    lr = 0.001
    epochs = 100
    dropout = 0.2

    logDir = "./log/" + time.strftime(
        "%Y-%m-%d-%H-%M-%S",
        time.gmtime(),
    )
    modelDir = "./model/"

    # ========================================================
    # 模型架构
    # ========================================================
    # First dual-head ablation: keep the M40 ResNet50 backbone.
    backbone = "resnet50"
    use_se_attention = False

    @classmethod
    def get_image_size(cls):
        return {
            "resnet50": 224,
            "resnet101": 224,
            "swin_t": 224,
            "convnext_tiny": 224,
            "convnext_small": 224,
            "inception_v3": 299,
            "efficientnet_b3": 300,
        }.get(cls.backbone, 224)

    # ========================================================
    # 粗细双头分类
    # ========================================================
    use_coarse_head = True
    num_coarse_classes = 2

    # 0=模糊(cloudy/haze/rainy/sunny) 1=清晰(shine/snow/sunrise/thunder).
    coarse_ambiguous_labels = [
        "cloudy",
        "haze",
        "rainy",
        "sunny",
    ]

    # 第一轮干净消融：仅 coarse CE.
    coarse_lambda = 0.1

    # 确认 coarse CE 有效后再开启.
    use_consistency_loss = False
    consistency_lambda = 0.01
    consistency_start_epoch = 5

    # 困难类卷积专家 (AmbiguousExpert)
    use_expert_head = True             # 是否启用困难4类卷积专家
    expert_lambda = 0.2                # 专家损失权重

    # ========================================================
    # Focal Loss + 动态 alpha
    # ========================================================
    focal_loss_gamma = 1.0
    focal_loss_alpha_eps = 1e-3
    focal_loss_alpha_source = "uniform_then_train_f1"

    dynamic_alpha_enabled = True
    dynamic_alpha_interval = 1
    dynamic_alpha_warmup = 0
    dynamic_alpha_learned = False
    dynamic_alpha_from_train = True

    # 原版 M22 公式: difficulty^1.0.
    # power=0.5 是独立消融实验 and must not be called original M22.
    alpha_power = 1.0

    # 梯度学习 alpha 参数(保留兼容).
    dynamic_alpha_grad_mode = "f1_balance"
    dynamic_alpha_lr = 0.5
    dynamic_alpha_momentum = 0.1
    dynamic_alpha_min = 0.5
    dynamic_alpha_max = 2.0

    # Legacy fixed-alpha values are kept only for fixed-alpha ablations.
    # 主实验不使用测试集推导的 alpha.
    focal_loss_per_class_acc = {
        "cloudy": 0.6267,
        "haze": 0.8400,
        "rainy": 0.8067,
        "shine": 1.0000,
        "snow": 0.8533,
        "sunny": 0.7867,
        "sunrise": 0.9630,
        "thunder": 0.9800,
    }

    # ========================================================
    # 监督对比学习
    # ========================================================
    supcon_enabled = False
    supcon_lambda = 0.01
    supcon_temperature = 0.2
    supcon_dim = 128

    # ========================================================
    # 优化器/学习率调度
    # ========================================================
    optimizer = "Adam"
    weight_decay = 0.0

    lr_scheduler = "CosineAnnealing"
    lr_min = 1e-6

    # ========================================================
    # 早停: 监控验证集 Macro F1
    # ========================================================
    early_stop_enabled = True
    early_stop_patience = 15
    early_stop_min_delta = 0.005

    # ========================================================
    # 数据配置
    # ========================================================
    data_augmentation_enabled = True
    stratified_split_enabled = True

    # ========================================================
    # 输出配置
    # ========================================================
    save_misclassified_images = False
    misclassified_per_pair_limit = 30
    save_last_model = False
