# 数据集说明

## 数据来源

本项目的数据集由 **RSCM** 和 **MWD** 两个公开数据集整合而成，均采用文件夹分类结构（每类一个子目录）：

| 数据集 | 来源 | 说明 |
|--------|------|------|
| **RSCM** | 深圳大学林迪等人，IEEE TIP 2017 | 主要数据来源，提供 6 类 |
| **MWD** | University of South Africa，Mendeley Data | 补充 `shine` 和 `sunrise` 类别 |

合并后的 8 类数据统一存放在：

```
data/RSCM/classification/weather_classification/
├── cloudy/     # 多云
├── haze/       # 薄雾
├── rainy/      # 雨天
├── shine/      # 晴朗（来自 MWD）
├── snow/       # 大雪
├── sunny/      # 晴天
├── sunrise/    # 日出（来自 MWD）
└── thunder/    # 雷雨
```

## 各类别图片数量

| 类别 | 总数量 | 原始来源 |
|------|--------|----------|
| cloudy  | 10,300 | RSCM + MWD |
| haze    | 10,000 | RSCM |
| rainy   | 10,213 | RSCM + MWD |
| shine   | 253    | MWD（数量最少）|
| snow    | 10,000 | RSCM |
| sunny   | 10,000 | RSCM |
| sunrise | 356    | MWD（数量最少）|
| thunder | 10,000 | RSCM |
| **合计** | **61,122** | |

## 数据采样策略

由于 `shine`（253 张）和 `sunrise`（356 张）类别数量远少于其他类别（各约 10,000 张），且为了控制训练规模，代码中设置了 **每类最多采样 1000 张**：

```python
# data_loader.py
class WeatherDataSet(Dataset):
    def __init__(self, max_per_class=1000):
        # ...
        if len(image_files) > max_per_class:
            image_files = random.sample(image_files, max_per_class)
```

实际使用的训练数据量：

| 数据集 | 样本数 |
|--------|--------|
| 训练集（70%）| 4,626 |
| 验证集（15%）| 991 |
| 测试集（15%）| 992 |
| **总计** | **6,609** |

## 数据划分比例

```
训练集 70%  | 验证集 15%  | 测试集 15%
```

使用 `torch.utils.data.random_split` 进行随机划分。

## 图片预处理

所有图片统一经过以下预处理流程（与 ImageNet 预训练模型对齐）：

```python
transforms.Compose([
    transforms.Resize(256),           # 短边缩放到 256，保持比例
    transforms.CenterCrop(224),       # 中心裁剪 224×224
    transforms.ToTensor(),            # 转为 (C, H, W) 张量
    transforms.Normalize(             # ImageNet 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## 类别不均衡问题

`shine`（253 张）和 `sunrise`（356 张）数量明显偏少，存在一定的类别不均衡。在 1000 张/类的采样限制下：

| 类别 | 可用数量 | 采样数量 |
|------|----------|----------|
| shine | 253 | 253（全部使用）|
| sunrise | 356 | 356（全部使用）|
| 其他 6 类 | 各 10,000+ | 各 1,000 |

建议后续考虑：
- 使用数据增强（随机翻转、旋转、颜色抖动）
- 或使用加权采样 / Focal Loss 缓解不均衡问题