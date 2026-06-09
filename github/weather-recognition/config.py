import time

import torch
# 开启 cuDNN benchmark，自动寻找最快卷积算法（输入尺寸固定时有效）
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
# 项目配置文件

class Common:
    '''
    通用配置
    '''
    basePath = "C:/Users/Lenovo/OneDrive/Desktop/caip_code/data/RSCM/classification/weather_classification/"  # 图片文件基本路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 设备配置
    imageSize = (224,224) # 图片大小
    labels = ["cloudy","haze","rainy","shine","snow","sunny","sunrise","thunder"] # 标签名称/文件夹名称


class Train:
    '''     
    训练相关配置
    '''
    batch_size = 128
    num_workers = 4  # 数据加载进程数（Windows 多 worker 共享内存限制）
    lr = 0.001
    epochs = 100
    logDir = "./log/" + time.strftime('%Y-%m-%d-%H-%M-%S',time.gmtime()) # 日志存放位置
    modelDir = "./model/" # 模型存放位置

    # 早停机制配置
    early_stop_patience = 10   # 验证准确率无有效上升的最大 epochs 数
    early_stop_min_delta = 0.005  # 被认为"有效上升"的最小阈值（提升至少 0.5%）
    early_stop_enabled = True    # 是否启用早停

    # 学习率调度配置
    lr_scheduler = "CosineAnnealing"  # 学习率调度策略："CosineAnnealing" 或 None
    lr_min = 1e-6                     # CosineAnnealing 最低学习率

    # Focal Loss 配置
    focal_loss_gamma = 1.0      # 聚焦参数，γ 越大越关注困难样本
    focal_loss_alpha_source = "model_18_test"  # alpha 权重来源："model_18_test" 或 "manual"
    # 基于 model_18（数据清洗后）测试集每类准确率计算 alpha
    focal_loss_per_class_acc = {
        "cloudy":  0.6267,
        "haze":    0.8400,
        "rainy":   0.8067,
        "shine":   1.0000,
        "snow":    0.8533,
        "sunny":   0.7867,
        "sunrise": 0.9630,
        "thunder": 0.9800,
    }
    focal_loss_alpha_eps = 0.01  # 计算 alpha 时的平滑项，避免除零

    # 动态 alpha 配置（Model 31: F1 梯度 + clamp 对齐 Model22 自然范围）
    dynamic_alpha_enabled = True     # 是否启用动态 alpha
    dynamic_alpha_interval = 3       # 更新间隔（epoch）
    dynamic_alpha_warmup = 0         # warmup 期：0=从 epoch 1 开始调整
    dynamic_alpha_learned = True     # True: 梯度学习, False: 公式计算
    dynamic_alpha_grad_mode = "f1_balance"  # "f1_balance"=低F1类获高alpha, "val_loss"=已废弃
    dynamic_alpha_lr = 0.5           # alpha 学习率
    dynamic_alpha_momentum = 0.1     # alpha 动量（低惯性）
    dynamic_alpha_min = 0.5          # alpha 下限（匹配 Model22 自然 range）
    dynamic_alpha_max = 2.0          # alpha 上限（匹配 Model22 自然 range）

    # 数据增强与分层采样控制
    data_augmentation_enabled = True   # 是否启用数据增强（RandomResizedCrop+Flip+ColorJitter）
    stratified_split_enabled = True    # 是否启用分层划分（70/15/15），False 则随机划分

    # 错分图片保存配置
    save_misclassified_images = False    # 是否保存错分图片
    misclassified_per_pair_limit = 30   # 每个错分类别对最多保存的图片数量

    # 模型保存配置
    save_last_model = False              # 是否保存训练结束时的最后一个模型



