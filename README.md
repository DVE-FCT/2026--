# 2026 CAIP 天气图片分类

本项目为 **2026 睿抗机器人开发者大赛（CAIP）智能算法赛项** 的天气分类任务实现，基于 PyTorch 构建多类别天气图像识别模型。

## 任务简介

对包含多种拍摄视角及场景的天气场景图片进行分类，共 **8 个类别**：

`cloudy` | `haze` | `rainy` | `shine` | `snow` | `sunny` | `sunrise` | `thunder`

## 项目结构

```
.
├── github/weather-recognition/    # 核心代码目录
│   ├── config.py                  # 训练参数与路径配置
│   ├── data_loader.py             # 自定义数据集与 DataLoader
│   ├── model.py                   # ResNet-50 分类模型
│   ├── train.py                   # 训练脚本
│   └── pridect.py                 # 预测/推理脚本
├── data/                          # 数据集存放目录（Git 忽略）
├── 记录.md                         # 调研与实验记录
├── 赛题.txt                        # 赛题说明
└── CLAUDE.md                      # 项目开发指南
```

## 环境依赖

- Python 3.10+
- PyTorch + TorchVision
- Pillow、Matplotlib、TensorBoard、NumPy

> 本项目使用 Conda 管理环境，无 `requirements.txt`，请根据代码 import 手动安装依赖。

## 快速开始

### 训练

```bash
cd github/weather-recognition
python train.py
```

- 训练日志保存在 `./log/<timestamp>/`，可用 TensorBoard 查看
- 模型保存于 `./model/`，当验证准确率超过阈值（默认 0.75）时自动保存

### 预测

```bash
cd github/weather-recognition
python pridect.py
```

> 预测脚本中的图片路径和模型路径为硬编码，使用前请根据实际路径修改。

### 查看训练日志

```bash
tensorboard --logdir=github/weather-recognition/log
```

## 数据集

项目涉及三个公开数据集：

| 数据集 | 规模 | 类别数 | 说明 |
|--------|------|--------|------|
| **RSCM** | 60,000 张 | 6 类 | 规模最大、分布均衡，主要训练数据来源 |
| **MWD** | 1,125 张 | 4 类 | 轻量级验证数据集，补充 `shine` 和 `sunrise` |
| **WEAPD** | 6,862 张 | 11 类 | 类别最全，尚未提取使用 |

数据集存放于 `data/` 目录下，**不纳入 Git 版本控制**。

## 模型架构

- **Backbone**：ResNet-50（加载 ImageNet 预训练权重）
- **分类头**：ReLU → Dropout(0.1) → Linear(1000, 8) → Softmax
- **优化器**：Adam（lr=0.001）
- **损失函数**：CrossEntropyLoss
- **图像尺寸**：224×224

## 参考

- 基础代码参考自 [mengxianglong123/weather-recognition](https://github.com/mengxianglong123/weather-recognition)
- 数据集来源：Heywhale、Mendeley Data、Harvard Dataverse
