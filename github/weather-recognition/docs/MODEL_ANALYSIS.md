# 模型效果分析报告

## 一、测试结果总览

基于两轮测试的实际数据：

| 轮次 | 测试准确率 | 样本数 |
|------|-----------|--------|
| 第 1 轮 | 73.39% | 728 / 992 |
| 第 2 轮 | 75.10% | 745 / 992 |

### 各类别准确率详情

| 类别 | 第 1 轮 | 第 2 轮 | 样本量（测试集）| 识别评价 |
|------|---------|---------|----------------|---------|
| **cloudy** | 49.65% | 50.34% | ~143~145 | ⚠️ **最差** |
| haze | 78.31% | 77.71% | ~157~166 | 较好 |
| rainy | 64.05% | 75.78% | ~153~161 | 中等偏下 |
| shine | 95.00% | 94.29% | 35~40 | 数据量少 |
| snow | 76.62% | 76.98% | ~139~154 | 较好 |
| sunny | 58.21% | 61.84% | ~134~152 | 较差 |
| sunrise | 100.00% | 98.08% | 52~53 | 数据量少 |
| thunder | 95.30% | 97.35% | ~149~151 | 最好 |

---

## 二、核心问题分析

### 问题 1：类别特征可分性差异极大

观察测试准确率，可以将 8 个类别按特征鲜明程度分为三个层级：

| 层级 | 类别 | 准确率 | 特征鲜明度 |
|------|------|--------|-----------|
| **T1 特征鲜明** | `thunder`、`sunrise`、`shine` | 95%~100% | 高 |
| **T2 特征一般** | `haze`、`snow` | 77%~78% | 中等 |
| **T3 特征模糊** | `cloudy`、`sunny`、`rainy` | 50%~76% | 低 |

**T1 类别特征鲜明的合理解释**：

- `thunder`（雷雨）：强烈明暗对比、闪电/乌云特征、独特的光照模式
- `sunrise`（日出）：橙红色调主导、暖色系、高饱和度，视觉上高度一致
- `shine`（晴朗）：光照均匀、阴影清晰、场景明亮

这些类别的准确率高**并非统计偏差**，而是因为它们的视觉特征足够独特，在特征空间中与其它类别距离足够远，模型很容易区分。

**真正的问题集中在 T3**：

- `cloudy`（多云）：灰白色调为主、纹理均匀、缺乏明显特征
- `sunny`（晴天）：明亮但色调多变，可能泛白或偏蓝
- `rainy`（雨天）：湿润反光表面、雨滴纹理，颜色可能在灰（阴天）和蓝（晴雨）之间变化

这三个类别之间的**视觉边界非常模糊**，尤其是 `cloudy` 和 `sunny` 在特征空间中高度重叠，导致互相误判率极高。

### 问题 2：类别识别两极分化

从数据可以清晰看到两类分化：

- **高准确率类**（>90%）：`thunder`（97%）、`shine`（95%）、`sunrise`（99%）→ 这些类别特征独特，容易识别
- **低准确率类**（<65%）：`cloudy`（50%）、`sunny`（60%）→ 这些类别间高度混淆

**关键发现**：`cloudy`（多云）和 `sunny`（晴天）的准确率在两轮测试中都是最差的，且互相容易混淆。其原因可能是：
- 这两个类别在视觉上非常相似（都是天空场景、以白色/灰色调为主）
- 拍摄角度、天气条件接近时，边界模糊
- 模型难以捕捉两者之间细微的亮度、纹理差异

### 问题 2：低样本类准确率"虚高"

| 类别 | 测试集样本数 | 准确率 | 实际情况 |
|------|-------------|--------|---------|
| sunrise | 52~53 | 99% | ⚠️ 数据量极少（仅约 356 张原始数据，测试集分配 53 张，训练集更少），准确率高更多是因为样本量小、特征分布集中，而非模型真正学得好 |
| shine | 35~40 | 95% | 同样问题：原始数据仅 253 张，模型在训练时见到该类样本的次数远少于其他类别 |
| cloudy | 143~145 | 50% | 样本量充足，但准确率最低，说明模型对该类特征学习严重不足 |

> **结论**：`sunrise` 和 `shine` 虽然准确率高，但测试集样本数很少（各 40~53 张），这个高准确率存在较大的统计偏差风险，一旦实际部署遇到更多样化的 sunrise/shine 图片，准确率很可能下降。

### 问题 3：过拟合（结合训练曲线分析）

从 `training_history_model_X.png` 可以观察到的规律：

1. **训练损失持续下降，验证损失提前走平甚至上升** → 过拟合迹象
2. **训练准确率远超验证准确率** → 模型在"死记硬背"训练数据，而非学习泛化特征
3. **类别准确率随训练时间变化不一致**：简单类别（thunder、shine）在训练早期就达到高准确率，而困难类别（cloudy、sunny）始终无法提升，说明模型优先学习了简单特征，放弃了困难样本

---

## 三、混淆矩阵的关键信息

结合测试输出的 `confusion_matrix_model_X.png` 和 `test_result_model_X.txt`，典型误分类模式：

| 真实类别 | 最容易误判为 | 原因分析 |
|---------|------------|---------|
| cloudy | sunny, haze | 三者色调接近、纹理相似、边界模糊 |
| rainy | cloudy | 雨天图像中常有灰蒙蒙的天空背景 |
| sunny | cloudy | 阳光强烈时图像泛白，与多云难以区分 |
| haze | cloudy | 薄雾场景整体偏灰，与多云视觉相似 |

---

## 四、改进建议与优先级

### 🔴 P0（影响最大，优先实施）

**1. 数据增强 — 重点解决 cloudy/sunny 混淆**

当前预处理过于简单（仅 Resize + CenterCrop），导致模型对相似类别的区分能力不足。建议加入：

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 随机裁剪，增加样本多样性
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    # 随机调整亮度/对比度，让模型学会在明暗变化下识别 cloudy vs sunny
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**2. 类别加权采样 — 解决 cloudy 样本权重问题**

cloudy 样本量充足但准确率最低，说明该类别本身特征不够明显，容易被其他类别覆盖。建议在 `data_loader.py` 中加入加权采样：

```python
# 为每个类别计算权重（稀有类权重高）
class_counts = [1000] * 8  # 假设每类约 1000 张
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
# cloud 和 sunny 权重不降低，但需要确保训练时每个 batch 都有这两类样本
```

---

### 🟡 P1（重要，长期优化）

**3. 增强正则化 — 缓解过拟合**

```python
# model.py 中加大 Dropout
self.dropout = nn.Dropout(0.5)  # 从 0.1 → 0.5

# train.py 中加入 weight decay
optimizer = optim.Adam(model.parameters(), lr=Train.lr, weight_decay=1e-4)
```

**4. Focal Loss — 替代 CrossEntropyLoss**

让模型更关注困难样本（cloudy、sunny、rainy），降低简单样本（thunder、shine）的权重：

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

### 🟢 P2（锦上添花）

**5. 早停策略 — 防止过度训练**

当前配置训练 100 epoch，但验证准确率在 30~40 epoch 后提升缓慢，继续训练只会加重过拟合。建议加入：

```python
patience = 15  # 连续 15 个 epoch 验证准确率不提升则停止
counter = 0
best_val_acc = 0.0
```

**6. 提升 sunrise / shine 的训练数据量**

这两类原始数据量少（356 和 253 张），建议：
- 提高 `max_per_class` 上限（如 800），确保训练集有足够 sunrise/shine 样本
- 或考虑对这两类使用更激进的数据增强

---

## 五、总结

| 问题 | 严重程度 | 可视化证据 |
|------|---------|---------|
| cloudy / sunny 识别率低（~50~60%）| 🔴 严重 | test_result 中准确率最低 |
| sunny 与 cloudy 互相混淆 | 🔴 严重 | 两者准确率同时偏低 |
| shine / sunrise "虚假高准确率" | 🟡 中等 | 样本量过少（35~53 张）|
| 过拟合（训练>>验证） | 🟡 中等 | training_history 曲线间距持续扩大 |

**核心结论**：模型目前的短板是 **cloudy / sunny 这两个"模糊类别"**的区分能力严重不足。改进数据增强（特别是亮度/对比度变换）是最高优先级的解决方案。`thunder`、`snow`、`haze` 等类别特征鲜明，模型已经学得较好，不需要特殊处理。