# Model 1 / Model 9 / Model 10 / Model 13 对比分析报告

> 日期：2026-05-18
> 目的：对比基线模型、结构修复模型、增强训练模型、以及数据增强消融实验

---

## 1. 模型概述

| 模型 | 数据增强 | 分层划分 | 最佳 epoch | 总准确率 | Macro F1 |
|------|---------|---------|-----------|----------|----------|
| **Model 1** | 无 | 无（random） | — | 73.39% | — |
| **Model 9** | 无 | 无（random） | 18 | **81.15%** | **0.8386** |
| **Model 10** | 有 | 有（70/15/15） | 35 | 81.07% | 0.8342 |
| **Model 13** | 无 | 有（70/15/15） | 85 | 76.13% | 0.7951 |

**Model 13 = Model 10（取消数据增强，保留分层划分）— 数据增强消融实验**

---

## 2. 训练配置参数说明

所有可通过 `config.py` 配置的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `early_stop_patience` | 15 | 验证准确率无有效上升的最大 epochs 数 |
| `early_stop_min_delta` | 0.005 | 被认为"有效上升"的最小阈值（提升至少 0.5%） |
| `early_stop_enabled` | True | 是否启用早停 |
| `data_augmentation_enabled` | True | 是否启用数据增强（RandomResizedCrop+Flip+ColorJitter） |
| `stratified_split_enabled` | True | 是否启用分层划分（70/15/15），False 则随机划分 |

**数据增强参数说明：**
- `RandomResizedCrop(224, scale=(0.7, 1.0))` — 随机裁剪到 224x224，保留原图 70%~100% 区域
- `RandomHorizontalFlip()` — 随机水平翻转
- `ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03)` — 颜色抖动

**分层划分逻辑：**
- 每类按 70/15/15 比例分配到训练/验证/测试集
- 保证测试集中各类别分布均衡

---

## 3. 总体指标对比

| 指标 | Model 1 | Model 9 | Model 10 | Model 13 | 13 vs 10 |
|------|---------|---------|----------|----------|----------|
| 总准确率 | 73.39% | **81.15%** | 81.07% | 76.13% | **-4.94%** |
| Macro F1 | — | **0.8386** | 0.8342 | 0.7951 | **-0.0391** |
| Weighted F1 | — | **0.8121** | 0.8108 | 0.7611 | **-0.0497** |
| 最佳 epoch | — | 18 | 35 | 85 | +50 |
| 过拟合程度 | 严重 | 严重 | 轻微 | 中等 | 加重 |

**关键发现：**
- 数据增强对准确率提升 **+4.94%**（Model 10 vs Model 13）
- 无数据增强时（Model 13）收敛更慢（epoch 85 vs 35），且过拟合更明显
- Model 10 = Model 13（无数据增强）+ 数据增强，验证了数据增强的正则化效果

---

## 4. 逐类 F1 对比

| 类别 | Model 9 F1 | Model 10 F1 | Model 13 F1 | 13 vs 10 |
|------|-----------|-----------|-----------|----------|
| cloudy | 0.6520 | 0.5966 | 0.5155 | **-0.0811** |
| haze | **0.7922** | 0.8026 | 0.7350 | **-0.0676** |
| rainy | **0.7750** | 0.7837 | 0.6667 | **-0.1170** |
| shine | 0.9512 | 0.9250 | **0.9744** | **+0.0494** |
| snow | 0.8938 | **0.8982** | 0.8523 | **-0.0459** |
| sunny | **0.7336** | 0.7322 | 0.7181 | -0.0141 |
| sunrise | 0.9524 | **0.9720** | 0.9391 | -0.0329 |
| thunder | 0.9583 | 0.9635 | **0.9595** | -0.0040 |

**分析：**
- 数据增强对 cloudy/rainy/haze/snow 影响最大，关闭后 F1 显著下降
- shine 是唯一在无数据增强时提升的类别（+4.94%）
- rainy 受数据增强影响最严重（-11.70%），说明数据增强让 rainy 类更难被识别

---

## 5. 训练稳定性分析

### 5.1 收敛速度对比

| 模型 | 最佳 epoch | 收敛稳定性 | 过拟合程度 |
|------|-----------|-----------|-----------|
| Model 9 | 18 | 差（剧烈波动） | 严重（24.18%） |
| Model 10 | 35 | **好（平稳）** | 轻微（21.23%） |
| Model 13 | 85 | 中等 | 中等（~23%） |

### 5.2 数据增强消融总结（Model 10 vs Model 13）

| 对比项 | Model 10（有增强） | Model 13（无增强） | 结论 |
|--------|-------------------|-------------------|------|
| 准确率 | 81.07% | 76.13% | 数据增强 +4.94% |
| Macro F1 | 0.8342 | 0.7951 | 数据增强 +0.0391 |
| 最佳 epoch | 35 | 85 | 数据增强加速收敛 |
| 过拟合 | 21.23% | ~23% | 数据增强减少过拟合 |

---

## 6. 关键发现

### 6.1 结构修复效果（Model 1 → Model 9）

| 改进项 | 影响 |
|--------|------|
| ResNet50 特征层（2048维）替换 ImageNet 输出（1000维） | 准确率 +7.76%，Macro F1 = 0.8386 |
| 移除 Softmax，让 CrossEntropyLoss 内部处理 | 损失计算正常化，收敛更稳定 |

**结论**：架构修复是性能提升的主要来源。

### 6.2 数据增强效果（Model 13 → Model 10）

| 改进项 | 影响 |
|--------|------|
| RandomResizedCrop + HorizontalFlip + ColorJitter | 准确率 +4.94%，Macro F1 +0.0391 |
| 分层划分（每类 70/15/15） | 测试集分布更均衡，结果可靠 |
| 最佳 epoch 从 85 提前到 35 | 数据增强加速收敛 |

### 6.3 主要短板

- **cloudy**：无数据增强时 F1 仅 0.5155，是最大短板
- **rainy**：数据增强对 rainy 影响最大（+11.70%），但本身仍是困难类

---

## 7. 结论与建议

### 7.1 模型推荐

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **最佳准确率** | **Model 10** | 数据增强 + 分层划分，准确率 81.07%，训练稳定 |
| **无数据增强场景** | **Model 13** | 仅用分层划分，准确率 76.13%，适合数据增强不适用的情况 |

### 7.2 后续优化方向

1. **cloudy 专项优化**（F1 最低，数据增强对其帮助有限）
2. **分层采样 + 无数据增强 + CrossEntropyLoss**（对比 Model 9 的无分层版本）
   - 或使用 class-weighted loss 补偿

2. **学习率调度**
   - Model 10 适合使用 CosineAnnealing（收敛更稳定）
   - 参考 feature 分支实验（CosineAnnealing 可提升 +2% Macro F1）

3. **早停配合**
   - Model 10 最佳 epoch=35，后续_epochs_no_improve 可触发早停
   - 建议 patience 调小到 10~15 避免过度训练

---

## 8. 各模型完整输出文件

- [Model 1 指标](github/weather-recognition/model/model_1/test_result_model_1.txt)
- [Model 9 训练日志](github/weather-recognition/model/model_9/training_log_model_9.txt)
- [Model 9 测试结果](github/weather-recognition/model/model_9/test_result_model_9.txt)
- [Model 10 训练日志](github/weather-recognition/model/model_10/training_log_model_10.txt)
- [Model 10 测试结果](github/weather-recognition/model/model_10/test_result_model_10.txt)
- [Model 13 训练日志](github/weather-recognition/model/model_13/training_log_model_13.txt)
- [Model 13 测试结果](github/weather-recognition/model/model_13/test_result_model_13.txt)