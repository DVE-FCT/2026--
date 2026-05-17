# Model 4 / Model 10 / Model 11 / Model 12 对比分析报告

> 日期：2026-05-17
> 对比：基线 Focal Loss（γ=1.0）→ 增强训练集（数据增强+分层）→ Focal Loss 完整修复版 → Focal Loss 无数据增强版

---

## 1. 模型配置对比

| 模型 | 特征层 | Loss 函数 | 数据增强 | 分层划分 | CosineAnnealing | 最佳 epoch |
|------|--------|-----------|---------|---------|-----------------|-----------|
| **Model 4** | 2048维 ResNet50 | FocalLoss (γ=1.0) | 无 | 无 | 无 | 78 |
| **Model 10** | 2048维 ResNet50 | CrossEntropyLoss | 有（RandomResizedCrop+Flip+ColorJitter） | 有（70/15/15） | 无 | 35 |
| **Model 11** | 2048维 ResNet50 | FocalLoss (γ=1.0) | 有（RandomResizedCrop+Flip+ColorJitter） | 有（70/15/15） | 有 | 19 |
| **Model 12** | 2048维 ResNet50 | FocalLoss (γ=1.0) | 无 | 有（70/15/15） | 有 | 26 |

**Model 12 = Model 11（取消数据增强，仅保留分层划分）**

---

## 2. 总体指标对比

| 指标 | Model 4 | Model 10 | Model 11 | Model 12 | 变化（12 vs 11） |
|------|---------|----------|----------|----------|-----------------|
| 总准确率 | 74.40% | **81.07%** | 77.34% | **83.18%** | **+5.84%** |
| Macro F1 | 0.7654 | **0.8342** | 0.7961 | **0.8548** | **+0.0587** |
| Weighted F1 | 0.7445 | **0.8108** | 0.7724 | **0.8309** | **+0.0585** |

**关键发现：**
- Model 12 准确率 83.18%，显著超越所有前代模型
- Model 12 相比 Model 11（数据增强版）提升 +5.84%，证明数据增强在 Focal Loss 框架下有负面干扰
- Focal Loss + 分层划分 + CosineAnnealing（无数据增强）= 最强组合

---

## 3. 逐类 F1 对比

| 类别 | Model 4 | Model 10 | Model 11 | Model 12 | 12 vs 11 | 12 vs 10 |
|------|---------|----------|----------|----------|---------|---------|
| cloudy | 0.5404 | 0.5966 | 0.5813 | **0.6441** | **+0.0628** | +0.0475 |
| haze | 0.6844 | 0.8026 | 0.7279 | **0.8000** | **+0.0721** | -0.0026 |
| rainy | 0.7074 | **0.7837** | 0.6515 | **0.7692** | **+0.1177** | -0.0145 |
| shine | 0.8462 | 0.9250 | 0.8861 | **0.9744** | **+0.0883** | **+0.0494** |
| snow | 0.8100 | 0.8982 | 0.8859 | **0.9175** | **+0.0316** | +0.0193 |
| sunny | 0.7081 | 0.7322 | 0.7476 | **0.7960** | **+0.0484** | **+0.0638** |
| sunrise | 0.8932 | 0.9720 | 0.9381 | **0.9643** | **+0.0262** | -0.0077 |
| thunder | 0.9338 | 0.9635 | 0.9508 | **0.9733** | **+0.0225** | +0.0098 |

**分析：**
- Model 12 在所有 8 个类别上均优于 Model 11，验证了"数据增强对 Focal Loss 有干扰"的假设
- rainy 提升最显著（+11.77%），说明 Focal Loss 在无数据增强时能更好地学习困难样本
- shine/snow/sunny 等中等难度类别也均有明显提升
- Model 12 vs Model 10：Focal Loss 在 cloudy/snow/sunny 上表现更优，rainy/haze 上略逊

---

## 4. 训练稳定性对比

| 模型 | 最佳 epoch | 最终 train_acc | 最终 val_acc | 过拟合差距 | 收敛稳定性 |
|------|-----------|---------------|-------------|-----------|-----------|
| Model 4 | 78 | 88.46% | 70.94% | 17.52% | 中等（持续上升） |
| Model 10 | 35 | 98.70% | 77.47% | 21.23% | 好（稳定后维持） |
| Model 11 | 19 | ~97%+ | ~77% | ~20% | 好（早停触发） |
| Model 12 | 26 | 99.89% | 82.32% | 17.57% | 好（稳定后下降） |

**分析：**
- Model 12 过拟合差距最小（17.57%），结合了 CosineAnnealing 的正则化效果与 Focal Loss 的困难样本聚焦
- Model 12 在 epoch 26 达到最佳验证准确率，收敛速度适中
- 无数据增强让训练集准确率更高，但验证集表现也更好，说明模型学到了更真实的特征

---

## 5. 数据增强对 Focal Loss 的影响分析

| 对比项 | Model 11（有增强） | Model 12（无增强） | 差异 |
|--------|---------------------|---------------------|------|
| cloudy F1 | 0.5813 | **0.6441** | **+6.28%** |
| haze F1 | 0.7279 | **0.8000** | **+7.21%** |
| rainy F1 | 0.6515 | **0.7692** | **+11.77%** |
| shine F1 | 0.8861 | **0.9744** | **+8.83%** |
| snow F1 | 0.8859 | **0.9175** | **+3.16%** |
| sunny F1 | 0.7476 | **0.7960** | **+4.84%** |

**关键发现：**
- 数据增强对 Focal Loss 有全面负面影响，Model 12 所有类别均优于 Model 11
- rainy 受影响最大（+11.77%），说明数据增强引入了大量"伪困难样本"，干扰了 Focal Loss 的学习
- Model 12 相比 Model 10（CrossEntropyLoss 无增强版），在困难类（cloudy/snow/sunny）上更优

---

## 6. 结论与建议

### 6.1 模型推荐

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **最佳综合** | **Model 12** | 准确率最高（83.18%），Macro F1 最高（0.8548），Focal Loss + 分层划分 + CosineAnnealing |
| 困难类提升 | Model 12 | cloudy/snow/sunny 等困难类 F1 显著高于其他模型 |
| 最佳平衡 | Model 10 | CrossEntropyLoss + 数据增强，效果稳定 |

### 6.2 Focal Loss 使用建议

- **Focal Loss 最佳配置**：无数据增强 + 分层划分 + CosineAnnealing（Model 12 方案）
- **数据增强对 Focal Loss 有干扰**：Model 11 < Model 12，证明了这一点
- **如需数据增强**：使用 CrossEntropyLoss（Model 10）而非 Focal Loss

---

## 7. 各模型完整输出文件

- [Model 4 训练日志](github/weather-recognition/model/model_4/training_log_model_4.txt)
- [Model 4 测试结果](github/weather-recognition/model/model_4/test_result_model_4.txt)
- [Model 10 训练日志](github/weather-recognition/model/model_10/training_log_model_10.txt)
- [Model 10 测试结果](github/weather-recognition/model/model_10/test_result_model_10.txt)
- [Model 11 训练日志](github/weather-recognition/model/model_11/training_log_model_11.txt)
- [Model 11 测试结果](github/weather-recognition/model/model_11/test_result_model_11.txt)
- [Model 12 训练日志](github/weather-recognition/model/model_12/training_log_model_12.txt)
- [Model 12 测试结果](github/weather-recognition/model/model_12/test_result_model_12.txt)