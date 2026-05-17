# Model 4 / Model 10 / Model 11 对比分析报告

> 日期：2026-05-17
> 对比：基线 Focal Loss（γ=1.0）→ 增强训练集（数据增强+分层）→ Focal Loss 完整修复版

---

## 1. 模型配置对比

| 模型 | 特征层 | Loss 函数 | 数据增强 | 分层划分 | CosineAnnealing | 最佳 epoch |
|------|--------|-----------|---------|---------|-----------------|-----------|
| **Model 4** | 2048维 ResNet50 | FocalLoss (γ=1.0) | 无 | 无 | 无 | 78 |
| **Model 10** | 2048维 ResNet50 | CrossEntropyLoss | 有（RandomResizedCrop+Flip+ColorJitter） | 有（70/15/15） | 无 | 35 |
| **Model 11** | 2048维 ResNet50 | FocalLoss (γ=1.0) | 有（RandomResizedCrop+Flip+ColorJitter） | 有（70/15/15） | 有 | 19 |

**Model 11 = Model 4（Focal Loss）+ Model 10（数据增强+分层）+ CosineAnnealing**

---

## 2. 总体指标对比

| 指标 | Model 4 | Model 10 | Model 11 | 变化（11 vs 4） | 变化（11 vs 10） |
|------|---------|----------|----------|----------------|-----------------|
| 总准确率 | 74.40% | **81.07%** | 77.34% | **+2.94%** | -3.73% |
| Macro F1 | 0.7654 | **0.8342** | 0.7961 | **+0.0307** | -0.0381 |
| Weighted F1 | 0.7445 | **0.8108** | 0.7724 | **+0.0279** | -0.0384 |

**关键发现：**
- Model 11 在 Model 4 基础上提升了 +2.94% 准确率和 +0.03 Macro F1
- 但 Model 11 相比 Model 10（无 Focal Loss）反而下降了 -3.73%
- Focal Loss + 数据增强的组合效果不如纯 CrossEntropyLoss + 数据增强

---

## 3. 逐类 F1 对比

| 类别 | Model 4 | Model 10 | Model 11 | 11 vs 4 | 11 vs 10 |
|------|---------|----------|----------|---------|---------|
| cloudy | 0.5404 | **0.5966** | 0.5813 | **+0.0409** | -0.0153 |
| haze | 0.6844 | **0.8026** | 0.7279 | **+0.0435** | -0.0747 |
| rainy | 0.7074 | **0.7837** | 0.6515 | -0.0559 | **-0.1322** |
| shine | 0.8462 | **0.9250** | 0.8861 | **+0.0399** | -0.0389 |
| snow | 0.8100 | **0.8982** | 0.8859 | **+0.0759** | -0.0123 |
| sunny | 0.7081 | 0.7322 | **0.7476** | **+0.0395** | +0.0154 |
| sunrise | 0.8932 | **0.9720** | 0.9381 | **+0.0449** | -0.0339 |
| thunder | **0.9338** | 0.9635 | 0.9508 | +0.0170 | -0.0127 |

**分析：**
- Model 11 在 cloudy/haze/shine/snow/sunny/sunrise 上优于 Model 4
- 但 rainy 下降 5.6%（0.7074 → 0.6515），haze 下降 7.5%（0.8026 → 0.7279）
- Focal Loss 在数据增强环境下对困难类（rainy/haze）反而有负面影响

---

## 4. 训练稳定性对比

| 模型 | 最佳 epoch | 最终 train_acc | 最终 val_acc | 过拟合差距 | 收敛稳定性 |
|------|-----------|---------------|-------------|-----------|-----------|
| Model 4 | 78 | 88.46% | 70.94% | 17.52% | 中等（持续上升） |
| Model 10 | 35 | 98.70% | 77.47% | 21.23% | 好（稳定后维持） |
| Model 11 | 19 | ~97%+ | ~77% | ~20% | 好（早停触发） |

**分析：**
- Model 11 收敛最快（epoch 19 触发早停），训练效率最高
- Model 10 收敛最稳定，Model 4 需要 78 epoch 才达峰值
- 过拟合程度：Model 10 > Model 11 > Model 4（CosineAnnealing 有效减少过拟合）

---

## 5. Focal Loss 在数据增强环境下的表现

| 对比项 | Model 4（无增强） | Model 11（有增强） | 说明 |
|--------|------------------|---------------------|------|
| cloudy F1 | 0.5404 | **0.5813**（+7.6%） | Focal Loss 有效提升困难类 |
| haze F1 | 0.6844 | **0.7279**（+6.4%） | Focal Loss 有效提升 haze |
| rainy F1 | 0.7074 | **0.6515**（-7.9%） | Focal Loss 在增强环境下反而下降 |
| sunny F1 | 0.7081 | **0.7476**（+5.6%） | Focal Loss 有效提升 sunny |

**关键发现：**
- Focal Loss 在无数据增强时能有效帮助困难类（cloudy/haze/sunny）
- 但在数据增强环境下，rainy/haze 等类反而下降明显
- 原因推测：数据增强改变了样本分布，Focal Loss 的类别权重与增强后的分布不匹配

---

## 6. 结论与建议

### 6.1 模型推荐

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **最佳综合** | **Model 10** | 准确率最高（81.07%），Macro F1 最高（0.8342），训练稳定 |
| 快速收敛 | Model 11 | epoch 19 即触发早停，训练效率最高 |
| 困难类提升 | Model 4 / Model 11 | 在特定类别（cloudy/haze）上优于 Model 10 |

### 6.2 Focal Loss 使用建议

- **无数据增强时**：Focal Loss γ=1.0 能有效提升困难类（推荐 Model 4 方案）
- **有数据增强时**：CrossEntropyLoss 表现更好（推荐 Model 10 方案）
- **如需结合两者**：建议重新调优 Focal Loss 的 alpha 权重（在数据增强后重新计算）

### 6.3 后续优化方向

1. **Model 10 为基线**：在 Model 10 基础上添加 CosineAnnealing（参考 Model 11 的 LR 调度）
2. **重调 alpha 权重**：基于 Model 10 的每类准确率重新计算 Focal Loss alpha
3. **rainy 专项优化**：Model 11 中 rainy 下降严重，需针对性增强

---

## 7. 各模型完整输出文件

- [Model 4 训练日志](github/weather-recognition/model/model_4/training_log_model_4.txt)
- [Model 4 测试结果](github/weather-recognition/model/model_4/test_result_model_4.txt)
- [Model 10 训练日志](github/weather-recognition/model/model_10/training_log_model_10.txt)
- [Model 10 测试结果](github/weather-recognition/model/model_10/test_result_model_10.txt)
- [Model 11 训练日志](github/weather-recognition/model/model_11/training_log_model_11.txt)
- [Model 11 测试结果](github/weather-recognition/model/model_11/test_result_model_11.txt)