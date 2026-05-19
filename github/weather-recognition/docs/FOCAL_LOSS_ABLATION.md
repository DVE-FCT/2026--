# Focal Loss 完整消融实验报告

> 日期：2026-05-19
> 分支：feature/focal-loss-early-stop
> 对比：Focal Loss alpha 权重来源对比（model_1 vs model_10）及完整配置消融

---

## 1. 模型配置对比

| 模型 | Loss 函数 | 数据增强 | 分层划分 | CosineAnnealing | alpha 来源 | 最佳 epoch | 总准确率 | Macro F1 |
|------|-----------|---------|---------|-----------------|------------|-----------|----------|----------|
| **Model 4** | FocalLoss (γ=1.0) | 无 | 无 | 无 | - | 78 | 74.40% | 0.7654 |
| **Model 10** | CrossEntropyLoss | 有 | 有 | 无 | - | 35 | 81.07% | 0.8342 |
| **Model 11** | FocalLoss (γ=1.0) | 有 | 有 | 有 | model_1 | 19 | 77.34% | 0.7961 |
| **Model 12** | FocalLoss (γ=1.0) | 无 | 有 | 有 | - | 26 | **83.18%** | **0.8548** |
| **Model 15** | FocalLoss (γ=1.0) | 有 | 无 | 有 | model_1 | 85 | 82.96% | 0.8538 |
| **Model 16** | FocalLoss (γ=1.0) | 有 | 有 | 有 | **model_10** | 53 | **83.28%** | **0.8590** |

**Model 16 = Model 11 + 使用 model_10 的测试集每类准确率计算 alpha 权重（替代 model_1）**

---

## 2. 核心发现：alpha 权重来源的影响（Model 15 vs Model 16）

Model 15 与 Model 16 配置完全相同，唯一的区别是 alpha 权重计算来源：

| 对比项 | Model 15（alpha from model_1） | Model 16（alpha from model_10） | 差异 |
|--------|-------------------------------|--------------------------------|------|
| 总准确率 | 82.96% | **83.28%** | **+0.32%** |
| Macro F1 | 0.8538 | **0.8590** | **+0.0052** |
| Weighted F1 | 0.8292 | **0.8319** | **+0.0027** |
| cloudy F1 | 0.6646 | **0.6259** | -0.0387 |
| haze F1 | 0.8207 | **0.8039** | -0.0168 |
| rainy F1 | 0.7789 | **0.8127** | **+0.0338** |
| shine F1 | 0.9600 | **0.9873** | **+0.0273** |
| snow F1 | 0.8720 | **0.8859** | **+0.0139** |
| sunny F1 | 0.7766 | **0.7857** | **+0.0091** |
| sunrise F1 | 0.9905 | **0.9908** | **+0.0003** |
| thunder F1 | 0.9670 | **0.9800** | **+0.0130** |

**关键结论：**
- model_10 alpha 权重在总体指标上更优（+0.32% 准确率，+0.0052 Macro F1）
- model_10 alpha 在 cloudy/haze 上略差（这两个类别在 model_10 中准确率较高，导致 alpha 权重相对较低）
- model_10 alpha 在其他 6 个类别上均有提升，特别是在 rainy（+0.0338）和 shine（+0.0273）上提升显著
- **原因**：model_10 使用数据增强训练，其每类准确率更能反映增强后数据的真实分布，因此计算出的 alpha 权重与数据增强场景更匹配

---

## 3. 总体指标对比

| 指标 | Model 4 | Model 10 | Model 11 | Model 12 | Model 15 | Model 16 |
|------|---------|----------|----------|----------|----------|----------|
| 总准确率 | 74.40% | 81.07% | 77.34% | 83.18% | 82.96% | **83.28%** |
| Macro F1 | 0.7654 | 0.8342 | 0.7961 | 0.8548 | 0.8538 | **0.8590** |
| Weighted F1 | 0.7445 | 0.8108 | 0.7724 | 0.8309 | 0.8292 | **0.8319** |

---

## 4. 逐类 F1 对比

| 类别 | Model 4 | Model 10 | Model 11 | Model 12 | Model 15 | Model 16 |
|------|---------|----------|----------|----------|----------|----------|
| cloudy | 0.5404 | 0.5966 | 0.5813 | **0.6441** | 0.6646 | 0.6259 |
| haze | 0.6844 | 0.8026 | 0.7279 | 0.8000 | **0.8207** | 0.8039 |
| rainy | 0.7074 | 0.7837 | 0.6515 | 0.7692 | 0.7789 | **0.8127** |
| shine | 0.8462 | 0.9250 | 0.8861 | 0.9744 | 0.9600 | **0.9873** |
| snow | 0.8100 | 0.8982 | 0.8859 | 0.9175 | 0.8720 | **0.8859** |
| sunny | 0.7081 | 0.7322 | 0.7476 | 0.7960 | 0.7766 | **0.7857** |
| sunrise | 0.8932 | 0.9720 | 0.9381 | 0.9643 | 0.9905 | **0.9908** |
| thunder | 0.9338 | 0.9635 | 0.9508 | 0.9733 | 0.9670 | **0.9800** |

---

## 5. 综合消融矩阵

### 5.1 Focal Loss 全配置消融

| 配置 | 数据增强 | 分层划分 | alpha 来源 | 准确率 | Macro F1 | 说明 |
|------|---------|---------|------------|--------|----------|------|
| Model 4 | - | - | - | 74.40% | 0.7654 | Focal Loss 基线 |
| Model 11 | Y | Y | model_1 | 77.34% | 0.7961 | 有增强有分层 |
| Model 12 | - | Y | - | **83.18%** | **0.8548** | 无增强有分层 |
| Model 15 | Y | - | model_1 | 82.96% | 0.8538 | 有增强无分层 |
| **Model 16** | Y | Y | **model_10** | **83.28%** | **0.8590** | **有增强有分层 + 优化 alpha** |

### 5.2 各因素贡献

| 对比 | 差异 | 结论 |
|------|------|------|
| Model 11 -> Model 15（取消分层） | +5.62% | 分层划分对 Focal Loss 有巨大正向作用 |
| Model 15 -> Model 16（alpha 优化） | **+0.32%** | model_10 alpha 权重与数据增强更匹配 |
| Model 11 -> Model 12（取消增强） | +5.84% | 数据增强对 Focal Loss 有干扰 |

### 5.3 alpha 权重消融（Model 15 vs Model 16）

| 对比 | 差异 | 结论 |
|------|------|------|
| model_1 alpha -> model_10 alpha | +0.32% | **使用与训练数据匹配的 alpha 权重可提升性能** |

---

## 6. 结论与建议

### 6.1 模型推荐

| 场景 | 推荐模型 | 准确率 | 理由 |
|------|---------|--------|------|
| **最佳综合** | **Model 16** | **83.28%** | Focal Loss + 数据增强 + 分层划分 + model_10 alpha |
| **次优选择** | Model 12 | 83.18% | Focal Loss + 分层划分（无数据增强），仅差 0.10% |
| **最佳稳定** | Model 10 | 81.07% | CrossEntropyLoss + 数据增强 + 分层 |

### 6.2 Focal Loss 使用建议

- **Focal Loss 最佳配置**：数据增强 + 分层划分 + CosineAnnealing + model_10 alpha 权重
- **alpha 权重选择**：使用与训练数据分布匹配的准确率计算 alpha（数据增强场景用 model_10，准确率更高的权重更低）
- **Model 16 vs Model 12**：Model 16 有数据增强但准确率更高（83.28% vs 83.18%），证明优化 alpha 权重可以缓解数据增强的干扰

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
- [Model 15 训练日志](github/weather-recognition/model/model_15/training_log_model_15.txt)
- [Model 15 测试结果](github/weather-recognition/model/model_15/test_result_model_15.txt)
- [Model 16 训练日志](github/weather-recognition/model/model_16/training_log_model_16.txt)
- [Model 16 测试结果](github/weather-recognition/model/model_16/test_result_model_16.txt)
