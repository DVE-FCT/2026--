# 训练部分
import time
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from config import Common, Train
from model import model as weatherModel
from data_loader import trainLoader, valLoader
from torch import optim

# ============================================================
# Focal Loss — 解决类别不平衡与难分样本
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    γ（gamma）: 聚焦参数，γ 越大越关注困难样本。默认 2.0
    α（alpha）: 类别权重，可传入 8 维 tensor 指定每类权重。默认每类权重 1.0
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_weight = self.alpha.to(targets.device).gather(0, targets)
            loss = alpha_weight * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================
# 自动创建编号文件夹 model_1, model_2, ...
# ============================================================
MODEL_ROOT = "./model"

def get_next_model_index():
    """获取下一个可用的 model_X 文件夹编号"""
    if not os.path.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT, exist_ok=True)
        return 1
    indices = []
    for name in os.listdir(MODEL_ROOT):
        m = re.match(r'^model_(\d+)$', name)
        if m and os.path.isdir(os.path.join(MODEL_ROOT, name)):
            indices.append(int(m.group(1)))
    return max(indices) + 1 if indices else 1

def create_model_dir():
    """创建本次训练的输出目录"""
    idx = get_next_model_index()
    run_dir = os.path.join(MODEL_ROOT, f"model_{idx}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "log"), exist_ok=True)
    return run_dir, idx

# ============================================================
# 模型初始化
# ============================================================
run_dir, run_idx = create_model_dir()
Train.modelDir = run_dir + "/"
Train.logDir = os.path.join(run_dir, "log", time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()))

# 文件名后缀，与文件夹编号匹配
SF = f"_model_{run_idx}"  # 例: _model_1

print(f"\n{'='*60}")
print(f"训练输出目录: {run_dir}")
print(f"{'='*60}\n")

model = weatherModel
model.to(Common.device)

# ---- Focal Loss alpha 权重计算（基于 config 中的每类准确率）----
per_class_acc = Train.focal_loss_per_class_acc
eps = Train.focal_loss_alpha_eps
alpha_list = [1.0 / (per_class_acc[c] + eps) for c in Common.labels]
alpha_tensor = torch.tensor(alpha_list, dtype=torch.float)
alpha_tensor = alpha_tensor / alpha_tensor.sum() * len(Common.labels)
print(f"[Focal Loss] alpha 权重（基于每类准确率）: ")
for c, a in zip(Common.labels, alpha_tensor.tolist()):
    print(f"  {c:>8s}: {a:.4f}  (准确率 {per_class_acc[c]:.4f})")

criterion = FocalLoss(gamma=Train.focal_loss_gamma, alpha=alpha_tensor)
optimizer = optim.Adam(model.parameters(), lr=Train.lr)

# ---- 学习率调度：CosineAnnealing ----
if Train.lr_scheduler == "CosineAnnealing":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Train.epochs,
        eta_min=Train.lr_min if hasattr(Train, 'lr_min') else 1e-6
    )
    print(f"[学习率调度] 策略: CosineAnnealingLR, 初始={Train.lr}, 最终={Train.lr_min}")
else:
    scheduler = None
    print(f"[学习率调度] 策略: 固定学习率 ({Train.lr})")

scaler = GradScaler('cuda')
writer = SummaryWriter(log_dir=Train.logDir, flush_secs=500)

history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [],   "val_acc": [],
    "lr": []
}


def train(epoch):
    model.train()
    epochLoss = 0
    correctNum = 0
    pbar = tqdm(trainLoader, desc=f"Epoch {epoch}/{Train.epochs} [Train]", ncols=100)
    for data, label in pbar:
        data, label = data.to(Common.device, non_blocking=True), label.to(Common.device, non_blocking=True)
        batchCorrectNum = 0
        optimizer.zero_grad()
        with autocast('cuda'):
            output = model(data)
            label_idx = torch.argmax(label, dim=1)
            loss = criterion(output, label_idx)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epochLoss += loss.item() * data.size(0)
        labels = label_idx
        outputs = torch.argmax(output, dim=1)
        for i in range(len(labels)):
            if labels[i] == outputs[i]:
                correctNum += 1
                batchCorrectNum += 1
        batchAcc = batchCorrectNum / data.size(0)
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batchAcc:.4f}", "lr": f"{current_lr:.2e}"})

    epochLoss = epochLoss / len(trainLoader.dataset)
    epochAcc = correctNum / len(trainLoader.dataset)
    print(f"Epoch:{epoch}\t Train Loss:{epochLoss:.4f} \t Train Acc:{epochAcc:.4f}")
    writer.add_scalar("train_loss", epochLoss, epoch)
    writer.add_scalar("train_acc", epochAcc, epoch)
    history["train_loss"].append(epochLoss)
    history["train_acc"].append(epochAcc)
    history["lr"].append(optimizer.param_groups[0]['lr'])
    return epochAcc


def val(epoch):
    model.eval()
    epochLoss = 0
    correctNum = 0
    pbar = tqdm(valLoader, desc=f"Epoch {epoch}/{Train.epochs} [Val  ]", ncols=100)
    with torch.no_grad():
        for data, label in pbar:
            data, label = data.to(Common.device, non_blocking=True), label.to(Common.device, non_blocking=True)
            batchCorrectNum = 0
            with autocast('cuda'):
                output = model(data)
                label_idx = torch.argmax(label, dim=1)
                loss = criterion(output, label_idx)
            epochLoss += loss.item() * data.size(0)
            labels = label_idx
            outputs = torch.argmax(output, dim=1)
            for i in range(len(labels)):
                if labels[i] == outputs[i]:
                    correctNum += 1
                    batchCorrectNum += 1
            batchAcc = batchCorrectNum / data.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batchAcc:.4f}", "lr": f"{current_lr:.2e}"})

        epochLoss = epochLoss / len(valLoader.dataset)
        epochAcc = correctNum / len(valLoader.dataset)
        print(f"Epoch:{epoch}\t Val   Loss:{epochLoss:.4f} \t Val   Acc:{epochAcc:.4f}")
        writer.add_scalar("val_loss", epochLoss, epoch)
        writer.add_scalar("val_acc", epochAcc, epoch)
    history["val_loss"].append(epochLoss)
    history["val_acc"].append(epochAcc)
    return epochAcc


def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    axes[0].plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], 'b-', label='Train Acc')
    axes[1].plot(epochs, history["val_acc"], 'r-', label='Val Acc')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"训练曲线已保存至: {save_path}")
    plt.close()


def save_training_log(run_dir, run_idx, history, bestEpoch, bestAcc, epochs):
    """保存训练信息到 txt 文件"""
    log_path = os.path.join(run_dir, f"training_log{SF}.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*50}\n")
        f.write(f"  训练记录 - model_{run_idx}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"模型目录: {run_dir}\n\n")
        f.write(f"--- 训练配置 ---\n")
        f.write(f"  epochs         : {epochs}\n")
        f.write(f"  batch_size     : {Train.batch_size}\n")
        f.write(f"  learning_rate   : {Train.lr}\n")
        f.write(f"  device          : {Common.device}\n")
        f.write(f"  loss_function   : FocalLoss (gamma={Train.focal_loss_gamma})\n")
        f.write(f"  lr_scheduler     : {Train.lr_scheduler if hasattr(Train, 'lr_scheduler') else 'None'}\n")
        if hasattr(Train, 'lr_scheduler') and Train.lr_scheduler == "CosineAnnealing":
            f.write(f"  lr_min          : {Train.lr_min}\n")
        f.write(f"  early_stop      : {'启用' if Train.early_stop_enabled else '禁用'}\n")
        if Train.early_stop_enabled:
            f.write(f"  early_stop_patience: {Train.early_stop_patience}\n")
            f.write(f"  early_stop_min_delta: {Train.early_stop_min_delta}\n")
        f.write(f"  alpha来源        : {Train.focal_loss_alpha_source}\n\n")
        f.write(f"--- 训练结果 ---\n")
        f.write(f"  最佳 epoch     : {bestEpoch}/{epochs}\n")
        f.write(f"  最佳验证准确率  : {bestAcc:.4f}\n\n")
        f.write(f"--- 指标变化 ---\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<12}\n")
        for i in range(len(history["train_loss"])):
            lr_val = history["lr"][i] if i < len(history["lr"]) else Train.lr
            f.write(f"{i+1:<8} {history['train_loss'][i]:<12.4f} "
                    f"{history['train_acc'][i]:<12.4f} "
                    f"{history['val_loss'][i]:<12.4f} "
                    f"{history['val_acc'][i]:<12.4f} "
                    f"{lr_val:<12.6f}\n")
        f.write(f"\n--- 完整路径 ---\n")
        f.write(f"  best.pt            : {run_dir}/best{SF}.pt\n")
        f.write(f"  last.pt            : {run_dir}/last{SF}.pt\n")
        f.write(f"  训练曲线          : {run_dir}/training_history{SF}.png\n")
        f.write(f"  tensorboard log    : {Train.logDir}\n")
    print(f"训练日志已保存至: {log_path}")


if __name__ == '__main__':
    bestAcc = 0.0
    bestEpoch = 0
    epochs_no_improve = 0

    for epoch in range(1, Train.epochs + 1):
        trainAcc = train(epoch)
        valAcc = val(epoch)

        # 更新学习率（CosineAnnealing）
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("lr", current_lr, epoch)

        if valAcc > bestAcc + Train.early_stop_min_delta:
            bestAcc = valAcc
            bestEpoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, f"best{SF}.pt"))
            print(f">>> 新的最佳模型! Epoch:{epoch} ValAcc:{valAcc:.4f} 已保存")
        else:
            epochs_no_improve += 1

        if Train.early_stop_enabled and epochs_no_improve >= Train.early_stop_patience:
            recent_train_accs = history["train_acc"][-5:] if len(history["train_acc"]) >= 5 else history["train_acc"]
            train_acc_trend = all(recent_train_accs[i] < recent_train_accs[i+1]
                                  for i in range(len(recent_train_accs)-1))
            prev_train_acc = history["train_acc"][-2] if len(history["train_acc"]) >= 2 else 0

            if train_acc_trend and trainAcc > prev_train_acc:
                print(f"\n早停触发: 验证准确率连续 {epochs_no_improve} 个 epoch 无有效上升，")
                print(f"         且训练准确率持续上升（过拟合）。")
                print(f"         当前 trainAcc={trainAcc:.4f}, valAcc={valAcc:.4f}")
                break

    torch.save(model.state_dict(), os.path.join(run_dir, f"last{SF}.pt"))
    print(f"\n训练结束。最佳模型在 Epoch {bestEpoch}, ValAcc={bestAcc:.4f}")

    plot_history(history, save_path=os.path.join(run_dir, f"training_history{SF}.png"))
    save_training_log(run_dir, run_idx, history, bestEpoch, bestAcc, Train.epochs)

    writer.close()