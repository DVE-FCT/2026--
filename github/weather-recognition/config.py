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
    num_workers = 0  # 对于Windows用户，这里应设置为0，否则会出现多线程错误
    lr = 0.001
    epochs = 100
    logDir = "./log/" + time.strftime('%Y-%m-%d-%H-%M-%S',time.gmtime()) # 日志存放位置
    modelDir = "./model/" # 模型存放位置

    # 早停机制配置
    early_stop_patience = 15   # 验证准确率无有效上升的最大 epochs 数
    early_stop_min_delta = 0.005  # 被认为"有效上升"的最小阈值（提升至少 0.5%）
    early_stop_enabled = True    # 是否启用早停

    # 学习率调度配置
    lr_scheduler = "CosineAnnealing"  # 学习率调度策略："CosineAnnealing" 或 None
    lr_min = 1e-6                     # CosineAnnealing 最低学习率

    # Focal Loss 配置
    focal_loss_gamma = 1.0      # 聚焦参数，γ 越大越关注困难样本
    focal_loss_alpha_source = "model_10_test"  # alpha 权重来源："model_10_test" 或 "manual"
    # 基于 model_10 测试集每类准确率计算 alpha（准确率越低权重越高）
    focal_loss_per_class_acc = {
        "cloudy":  0.5867,
        "haze":    0.8133,
        "rainy":   0.8333,
        "shine":   0.9487,
        "snow":    0.8533,
        "sunny":   0.7200,
        "sunrise": 0.9630,
        "thunder": 0.9667,
    }
    focal_loss_alpha_eps = 0.01  # 计算 alpha 时的平滑项，避免除零

    # 数据增强与分层采样控制
    data_augmentation_enabled = True   # 是否启用数据增强（RandomResizedCrop+Flip+ColorJitter）
    stratified_split_enabled = True    # 是否启用分层划分（70/15/15），False 则随机划分



