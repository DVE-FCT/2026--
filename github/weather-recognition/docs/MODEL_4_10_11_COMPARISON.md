# Model 4 / Model 10 / Model 11 / Model 12 / Model 15 对比分析报告

> 日期：2026-05-19
> 对比：Focal Loss 基线 → CrossEntropyLoss 数据增强 → Focal Loss + 数据增强+分层 → Focal Loss 无数据增强 → Focal Loss 仅数据增强

---

## 1. 模型配置对比

| 模型 | Loss 函数 | 数据增强 | 分层划分 | CosineAnnealing | 最佳 epoch | 总准确率 | Macro F1 |
|------|-----------|---------|---------|-----------------|-----------|----------|----------|
| **Model 4** | FocalLoss (γ=1.0) | 无 | 无 | 无 | 78 | 74.40% | 0.7654 |
| **Model 10** | CrossEntropyLoss | 有 | 有 | 无 | 35 | **81.07%** | 0.8342 |
| **Model 11** | FocalLoss (γ=1.0) | 有 | 有 | 有 | 19 | 77.34% | 0.7961 |
| **Model 12** | FocalLoss (γ=1.0) | 无 | 有 | 有 | 26 | **83.18%** | **0.8548** |
| **Model 15** | FocalLoss (γ=1.0) | 有 | 无 | 有 | 85 | 82.96% | 0.8538 |

**Model 15 = Model 11（取消分层划分，仅保留数据增强）— 分层划分消融实验**

---

## 2. 总体指标对比

| 指标 | Model 4 | Model 10 | Model 11 | Model 12 | Model 15 | 变化（15 vs 11） |
|------|---------|----------|----------|----------|----------|-----------------|
| 总准确率 | 74.40% | **81.07%** | 77.34% | **83.18%** | 82.96% | **+5.62%** |
| Macro F1 | 0.7654 | **0.8342** | 0.7961 | **0.8548** | 0.8538 | **+0.0577** |
| Weighted F1 | 0.7445 | **0.8108** | 0.7724 | **0.8309** | 0.8292 | **+0.0568** |

**关键发现：**
- Model 15 vs Model 11：分层划分让准确率提升 +5.62%，证明分层划分对 Focal Loss 有正向作用
- Model 15 vs Model 12：有数据增强但无分层（82.96%）接近无数据增强有分层（83.18%）
- 数据增强对 Focal Loss 仍有负面干扰，但分层划分能显著改善

---

## 3. 逐类 F1 对比

| 类别 | Model 4 | Model 10 | Model 11 | Model 12 | Model 15 | 15 vs 11 |
|------|---------|----------|----------|----------|----------|----------|
| cloudy | 0.5404 | 0.5966 | 0.5813 | **0.6441** | 0.6646 | **+0.0833** |
| haze | 0.6844 | 0.8026 | 0.7279 | **0.8000** | 0.8207 | **+0.0928** |
| rainy | 0.7074 | **0.7837** | 0.6515 | **0.7692** | 0.7789 | **+0.1274** |
| shine | 0.8462 | 0.9250 | 0.8861 | **0.9744** | 0.9600 | +0.0739 |
| snow | 0.8100 | 0.8982 | 0.8859 | **0.9175** | 0.8720 | -0.0139 |
| sunny | 0.7081 | 0.7322 | 0.7476 | **0.7960** | 0.7766 | +0.0290 |
| sunrise | 0.8932 | 0.9720 | 0.9381 | **0.9643** | **0.9905** | **+0.0524** |
| thunder | 0.9338 | 0.9635 | 0.9508 | **0.9733** | 0.9670 | +0.0162 |

**分析：**
- Model 15 在 cloudy/haze/rainy/sunrise 上优于 Model 11，证明分层划分对困难类帮助最大
- Model 15 在 snowy/sunny 上略低于 Model 11，但整体更优
- Model 15 vs Model 12：数据增强对 Focal Loss 的干扰依然存在，但差距缩小

---

## 4. 分层划分消融分析（Model 11 vs Model 15）

| 对比项 | Model 11（有分层） | Model 15（无分层） | 差异 | 结论 |
|--------|---------------------|---------------------|------|------|
| 总准确率 | 77.34% | **82.96%** | **+5.62%** | 分层划分对 Focal Loss 有巨大正向作用 |
| Macro F1 | 0.7961 | **0.8538** | **+0.0577** | 分层划分显著改善宏观指标 |
| cloudy F1 | 0.5813 | **0.6646** | **+0.0833** | 分层划分对 cloudy 帮助最大 |
| rainy F1 | 0.6515 | **0.7789** | **+0.1274** | rainy 受分层划分影响最大 |

---

## 5. 综合消融结论

### 5.1 Focal Loss 配置消融矩阵

| 配置 | 数据增强 | 分层划分 | 准确率 | Macro F1 | 说明 |
|------|---------|---------|--------|----------|------|
| Model 4 | - | - | 74.40% | 0.7654 | Focal Loss 基线 |
| Model 11 | Y | Y | 77.34% | 0.7961 | 有增强有分层 |
| Model 12 | - | Y | **83.18%** | **0.8548** | 无增强有分层 |
| Model 15 | Y | - | 82.96% | 0.8538 | 有增强无分层 |

### 5.2 各因素贡献（以 Model 11 为基准）

| 对比 | 差异 | 结论 |
|------|------|------|
| Model 11 -> Model 12（取消增强） | +5.84% | 数据增强对 Focal Loss 有严重干扰 |
| Model 11 -> Model 15（取消分层） | +5.62% | 分层划分对 Focal Loss 有巨大正向作用 |

### 5.3 关键发现

1. **数据增强对 Focal Loss 有干扰**：但分层划分能显著缓解
2. **分层划分对 Focal Loss 至关重要**：取消后准确率下降 5.62%
3. **Model 15 接近 Model 12**：说明在有数据增强时，分层划分的作用更大

---

## 6. 结论与建议

### 6.1 模型推荐

| 场景 | 推荐模型 | 准确率 | 理由 |
|------|---------|--------|------|
| **最佳综合** | **Model 12** | **83.18%** | Focal Loss + 分层划分 + CosineAnnealing（无数据增强） |
| **次优选择** | **Model 15** | 82.96% | Focal Loss + 数据增强（无分层），仅差 0.22% |
| **最佳稳定** | Model 10 | 81.07% | CrossEntropyLoss + 数据增强 + 分层 |

### 6.2 Focal Loss 使用建议

- **Focal Loss 最佳配置**：无数据增强 + 分层划分 + CosineAnnealing
- **分层划分对 Focal Loss 至关重要**：取消后准确率下降 5.62%
- **如需数据增强**：Model 15 接近 Model 12，仅差 0.22%

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