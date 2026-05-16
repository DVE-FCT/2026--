# 训练加速优化说明

## 问题描述

训练速度极慢，Epoch 1 每个 batch 耗时约 **14.72 秒**，单轮训练接近 9 分钟，严重影响迭代效率。

## 瓶颈诊断

通过拆解测试定位瓶颈：

| 测试项 | 耗时 | 说明 |
|--------|------|------|
| 纯模型 forward (原始) | **3.57 s/batch** | GPU 计算极慢，ResNet-50 在 batch=128 时不应超过 0.5s |
| 数据加载 | 0.625 s/batch | 不是主要瓶颈 |

**结论**：GPU 卷积计算效率低下是主要瓶颈，而非数据 I/O。

---

## 优化手段与代码对照

### 1. 开启 cuDNN Benchmark

**文件**：`config.py`

**代码**：
```python
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
```

**原理**：当输入尺寸固定时，cuDNN 会自动搜索最快的卷积算法，避免每次使用通用慢算法。

**效果**：forward 从 3.57s 降至 **2.38s**（提速 33%）。

---

### 2. AMP 混合精度训练 (Automatic Mixed Precision)

**文件**：`train.py`、`test.py`

**代码**（以 train.py 为例）：
```python
from torch.amp import GradScaler, autocast

scaler = GradScaler('cuda')

def train(epoch):
    for data, label in pbar:
        optimizer.zero_grad()
        with autocast('cuda'):
            output = model(data)
            loss = criterion(output, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**原理**：
- `autocast('cuda')` 自动将部分运算切换为 FP16（半精度），充分利用 NVIDIA Tensor Core
- `GradScaler` 对梯度进行缩放，避免 FP16 下梯度下溢
- 显存占用降低，计算吞吐量大幅提升

**效果**：forward 从 2.38s 降至 **0.124s**（提速 19 倍）。

---

### 3. 非阻塞数据传输

**文件**：`train.py`、`test.py`

**代码**：
```python
data, label = data.to(Common.device, non_blocking=True), label.to(Common.device, non_blocking=True)
```

**原理**：`non_blocking=True` 让 CPU→GPU 的内存拷贝与 GPU 计算并行，减少等待时间。

---

### 4. pin_memory 加速

**文件**：`data_loader.py`

**代码**：
```python
trainLoader = DataLoader(..., pin_memory=True)
valLoader   = DataLoader(..., pin_memory=True)
testLoader  = DataLoader(..., pin_memory=True)
```

**原理**：`pin_memory=True` 将数据预分配到固定内存页，配合 `non_blocking=True` 可大幅加速 Host→Device 传输。

---

## 综合效果

| 阶段 | 每 batch 耗时 | 相对原始 |
|------|---------------|----------|
| 优化前（用户实测） | **14.72 s** | 1x |
| 优化后（实测） | **1.01 s** | **~14.6x** |

单轮训练时间从约 9 分钟降至 **~40 秒**，40 个 epoch 总训练时间从 6 小时缩短至约 **25 分钟**。

---

## 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `config.py` | 添加 `torch.backends.cudnn.benchmark = True` |
| `data_loader.py` | `DataLoader` 增加 `pin_memory=True` |
| `train.py` | 引入 `GradScaler` + `autocast('cuda')`；数据迁移加 `non_blocking=True` |
| `test.py` | 推理同样启用 `autocast('cuda')` 和 `non_blocking=True` |

---

## 环境要求

- PyTorch >= 1.6（本环境为 2.11.0）
- NVIDIA GPU 支持 FP16（如 GTX 10 系、RTX 20/30/40 系）
- CUDA 驱动版本 >= 11.0（本环境为 12.8）
