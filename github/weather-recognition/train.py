# 训练部分
import time
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from scipy.stats import spearmanr

# 设置中文字体
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
try:
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.sans-serif'] = ['SimHei']
except Exception:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from config import Common, Train
from model import model as weatherModel
from torch import optim

MODEL_ROOT = "./model"

# 类别颜色与标记（统一视觉风格）
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD", "#E09F2B"]
CLASS_MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h"]
CLASS_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        pt = torch.exp(log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
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


def update_alpha_tensor(alpha_list):
    """将 alpha list 转为归一化的 tensor"""
    t = torch.tensor(alpha_list, dtype=torch.float)
    return t / t.sum() * len(Common.labels)


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


def train_epoch(epoch, model, train_loader, criterion, optimizer, scaler, writer, history):
    """训练一个 epoch"""
    model.train()
    epoch_loss = 0
    correct_num = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Train.epochs} [Train]", ncols=100)
    for data, label in pbar:
        data, label = data.to(Common.device, non_blocking=True), label.to(Common.device, non_blocking=True)
        label_idx = torch.argmax(label, dim=1)
        batch_correct = 0
        optimizer.zero_grad()
        with autocast('cuda'):
            output = model(data)
            loss = criterion(output, label_idx)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item() * data.size(0)
        outputs = torch.argmax(output, dim=1)
        for i in range(len(label_idx)):
            if label_idx[i] == outputs[i]:
                correct_num += 1
                batch_correct += 1
        batch_acc = batch_correct / data.size(0)
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}", "lr": f"{current_lr:.2e}"})

    epoch_loss = epoch_loss / len(train_loader.dataset)
    epoch_acc = correct_num / len(train_loader.dataset)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch:{epoch}\t Train Loss:{epoch_loss:.4f} \t Train Acc:{epoch_acc:.4f}\t LR:{current_lr:.2e}")
    writer.add_scalar("train_loss", epoch_loss, epoch)
    writer.add_scalar("train_acc", epoch_acc, epoch)
    writer.add_scalar("lr", current_lr, epoch)
    history["train_loss"].append(epoch_loss)
    history["train_acc"].append(epoch_acc)
    history["lr"].append(current_lr)
    return epoch_acc


def val_epoch(epoch, model, val_loader, criterion, optimizer, writer, history):
    """验证一个 epoch"""
    model.eval()
    epoch_loss = 0
    correct_num = 0
    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{Train.epochs} [Val  ]", ncols=100)
    with torch.no_grad():
        for data, label in pbar:
            data, label = data.to(Common.device, non_blocking=True), label.to(Common.device, non_blocking=True)
            label_idx = torch.argmax(label, dim=1)
            batch_correct = 0
            with autocast('cuda'):
                output = model(data)
                loss = criterion(output, label_idx)
            epoch_loss += loss.item() * data.size(0)
            outputs = torch.argmax(output, dim=1)
            for i in range(len(label_idx)):
                if label_idx[i] == outputs[i]:
                    correct_num += 1
                    batch_correct += 1
            batch_acc = batch_correct / data.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}", "lr": f"{current_lr:.2e}"})

        epoch_loss = epoch_loss / len(val_loader.dataset)
        epoch_acc = correct_num / len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch:{epoch}\t Val   Loss:{epoch_loss:.4f} \t Val   Acc:{epoch_acc:.4f}\t LR:{current_lr:.2e}")
        writer.add_scalar("val_loss", epoch_loss, epoch)
        writer.add_scalar("val_acc", epoch_acc, epoch)
    history["val_loss"].append(epoch_loss)
    history["val_acc"].append(epoch_acc)
    return epoch_acc


def compute_alpha_gradient(model, val_loader, current_alpha, gamma, device):
    """计算 val_loss 对 alpha 的梯度 — alpha 从验证集学习，不参与训练反传"""
    alpha_param = torch.tensor(current_alpha, device=device, dtype=torch.float, requires_grad=True)

    model.eval()
    total_samples = 0

    for data, label in val_loader:
        data = data.to(device)
        label_idx = torch.argmax(label, dim=1).to(device)

        with autocast('cuda'):
            output = model(data)
            log_probs = F.log_softmax(output, dim=1)
            ce = F.nll_loss(log_probs, label_idx, reduction='none')
            pt = torch.exp(log_probs.gather(1, label_idx.unsqueeze(1)).squeeze(1))
            focal_each = (1 - pt) ** gamma * ce
            alpha_w = alpha_param.gather(0, label_idx)
            loss = (focal_each * alpha_w).mean()

        total_samples += data.size(0)
        loss.backward()  # 累积梯度到 alpha_param

    if alpha_param.grad is not None:
        grad = alpha_param.grad.cpu().numpy() / max(total_samples, 1)
    else:
        grad = np.zeros_like(current_alpha)

    return grad


def compute_per_class_val_metrics(model, val_loader):
    """计算验证集每类的 acc 和 F1，返回 (per_class_acc, per_class_f1)"""
    model.eval()
    n = len(Common.labels)
    conf_matrix = np.zeros((n, n), dtype=np.int64)

    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(Common.device)
            label_idx = torch.argmax(label, dim=1)
            with autocast('cuda'):
                output = model(data)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            labels_np = label_idx.cpu().numpy()
            for p, l in zip(preds, labels_np):
                conf_matrix[l, p] += 1

    per_class_acc = {}
    per_class_f1 = {}
    for i, c in enumerate(Common.labels):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        total = conf_matrix[i, :].sum()
        per_class_acc[c] = tp / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_f1[c] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return per_class_acc, per_class_f1


def make_alpha_target(difficulty, gamma=1.0, min_w=0.5, max_w=2.0, eps=1e-6):
    """将难度映射为 alpha：归一化 → clamp → 再次归一化（均值保持 1）"""
    difficulty = np.maximum(difficulty, eps)
    target = difficulty ** gamma
    target = target / (target.mean() + eps)     # 归一化到均值 1
    target = np.clip(target, min_w, max_w)      # 截断极端值
    target = target / (target.mean() + eps)     # 截断后重新归一化
    return target.tolist()


def plot_alpha_spearman(alpha_history, spearman_history, alpha_source, save_path):
    """绘制 alpha 权重演化图和 Spearman ρ 检验图（1×2）"""
    update_epochs = sorted(alpha_history.keys())
    labels = Common.labels
    n_classes = len(labels)

    # 初始 alpha
    initial_alphas = alpha_history[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)

    # ---- 左图：Spearman ρ 随 epoch 变化 ----
    if spearman_history:
        spearman_epochs = sorted(spearman_history.keys())
        rho_values = [spearman_history[e] for e in spearman_epochs]
        ax1.plot(spearman_epochs, rho_values, 'o-', color='#2C7BB6', linewidth=2,
                 markersize=8, markerfacecolor='white', markeredgewidth=2,
                 label='Spearman $\\rho$')
        ax1.axhline(y=1.0, color='#D7191C', linestyle='--', linewidth=1, alpha=0.5, label='$\\rho=1$ (理想)')
        ax1.fill_between(spearman_epochs, 0, rho_values, alpha=0.15, color='#2C7BB6')
        ax1.set_ylim(-0.1, 1.15)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spearman $\\rho$', fontsize=12, fontweight='bold')
    ax1.set_title('Alpha-Difficulty Spearman $\\rho$', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ---- 右图：Alpha 权重演化 ----
    for i, c in enumerate(labels):
        values = [alpha_history[e][i] for e in update_epochs]
        color = CLASS_COLORS[i]
        marker = CLASS_MARKERS[i]
        ls = CLASS_LINESTYLES[i]
        ax2.plot(update_epochs, values, color=color, linestyle=ls, linewidth=1.8,
                 marker=marker, markersize=6, markerfacecolor='white', markeredgewidth=1.5,
                 label=f'{c}', markevery=max(1, len(update_epochs) // 10))

    ax2.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, label='$\\alpha=1$ (uniform)')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Alpha Weight', fontsize=12, fontweight='bold')
    ax2.set_title(f'Focal Loss Alpha Evolution ({alpha_source})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 图例放在右侧框外，避免遮挡曲线
    ax2.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5),
               framealpha=0.9, edgecolor='gray', ncol=1, borderpad=0.6, labelspacing=0.3)

    fig.suptitle('Focal Loss Alpha 动态调整与 Spearman 检验', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, top=0.92)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Alpha/Spearman 演化图已保存至: {save_path}")
    plt.close()


def plot_history(history, save_path):
    """绘制训练曲线"""
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


def save_training_log(run_dir, run_idx, sf, history, best_epoch, best_acc, epochs,
                      alpha_history=None, spearman_history=None):
    """保存训练信息到 txt 文件"""
    log_path = os.path.join(run_dir, f"training_log{sf}.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*50}\n")
        f.write(f"  训练记录 - model_{run_idx}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"模型目录: {run_dir}\n\n")
        f.write(f"--- 训练配置 ---\n")
        f.write(f"  epochs              : {epochs}\n")
        f.write(f"  batch_size          : {Train.batch_size}\n")
        f.write(f"  learning_rate        : {Train.lr}\n")
        f.write(f"  device               : {Common.device}\n")
        f.write(f"  loss_function        : FocalLoss (gamma={Train.focal_loss_gamma})\n")
        f.write(f"  alpha来源             : {Train.focal_loss_alpha_source}\n")
        f.write(f"  num_workers          : {Train.num_workers if hasattr(Train, 'num_workers') else 0}\n")
        if hasattr(Train, 'dynamic_alpha_enabled') and Train.dynamic_alpha_enabled:
            interval = Train.dynamic_alpha_interval if hasattr(Train, 'dynamic_alpha_interval') else 5
            warmup = Train.dynamic_alpha_warmup if hasattr(Train, 'dynamic_alpha_warmup') else 5
            learned = hasattr(Train, 'dynamic_alpha_learned') and Train.dynamic_alpha_learned
            d_min = Train.dynamic_alpha_min if hasattr(Train, 'dynamic_alpha_min') else 0.1
            d_max = Train.dynamic_alpha_max if hasattr(Train, 'dynamic_alpha_max') else 4.0
            f.write(f"  alpha动态更新          : 启用\n")
            f.write(f"    更新间隔              : 每 {interval} epoch\n")
            f.write(f"    warmup               : {warmup} epoch\n")
            if learned:
                lr_a = Train.dynamic_alpha_lr if hasattr(Train, 'dynamic_alpha_lr') else 0.01
                mom = Train.dynamic_alpha_momentum if hasattr(Train, 'dynamic_alpha_momentum') else 0.9
                f.write(f"    模式                  : 梯度学习（alpha -= lr × ∇val_loss）\n")
                f.write(f"    alpha lr             : {lr_a}\n")
                f.write(f"    alpha momentum       : {mom}\n")
            else:
                f.write(f"    模式                  : 公式计算（difficulty → normalize → clamp → EMA）\n")
            f.write(f"    clamp                : [{d_min}, {d_max}]\n")
        else:
            f.write(f"  alpha动态更新          : 禁用\n")
        f.write(f"  lr_scheduler         : {Train.lr_scheduler if hasattr(Train, 'lr_scheduler') else 'None'}\n")
        if hasattr(Train, 'lr_scheduler') and Train.lr_scheduler == "CosineAnnealing":
            f.write(f"  lr_min               : {Train.lr_min}\n")
        f.write(f"  early_stop           : {'启用' if Train.early_stop_enabled else '禁用'}\n")
        if Train.early_stop_enabled:
            f.write(f"  early_stop_patience   : {Train.early_stop_patience}\n")
            f.write(f"  early_stop_min_delta  : {Train.early_stop_min_delta}\n")
        f.write(f"  data_augmentation    : {'启用' if Train.data_augmentation_enabled else '禁用'}\n")
        f.write(f"  stratified_split     : {'启用' if Train.stratified_split_enabled else '禁用'}\n")
        f.write(f"  save_misclassified   : {'启用' if getattr(Train, 'save_misclassified_images', False) else '禁用'}\n\n")
        f.write(f"--- 训练结果 ---\n")
        f.write(f"  最佳 epoch           : {best_epoch}/{epochs}\n")
        f.write(f"  最佳验证准确率         : {best_acc:.4f}\n\n")

        # Alpha 动态更新记录
        if alpha_history:
            learned = hasattr(Train, 'dynamic_alpha_learned') and Train.dynamic_alpha_learned
            mode_label = "梯度学习" if learned else "公式计算"
            f.write(f"--- Alpha 动态更新记录（{mode_label}） ---\n")
            update_epochs = sorted(alpha_history.keys())
            f.write(f"{'Epoch':<8} " + " ".join([f"{c:>8}" for c in Common.labels]) + "   Spearman ρ\n")
            f.write("-" * (16 + 9 * 8 + 12) + "\n")
            for e in update_epochs:
                alphas = alpha_history[e]
                rho_str = f"  {spearman_history.get(e, 0):.4f}" if spearman_history else ""
                f.write(f"{e:<8} " + " ".join([f"{v:>8.4f}" for v in alphas]) + f"{rho_str}\n")

        f.write(f"\n--- 指标变化 ---\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<12}\n")
        for i in range(len(history["train_loss"])):
            lr_val = history["lr"][i] if i < len(history["lr"]) else Train.lr
            f.write(f"{i+1:<8} {history['train_loss'][i]:<12.4f} "
                    f"{history['train_acc'][i]:<12.4f} "
                    f"{history['val_loss'][i]:<12.4f} "
                    f"{history['val_acc'][i]:<12.4f} "
                    f"{lr_val:<12.6f}\n")
        f.write(f"\n--- 完整路径 ---\n")
        f.write(f"  best.pt                 : {run_dir}/best{sf}.pt\n")
        if getattr(Train, 'save_last_model', True):
            f.write(f"  last.pt                 : {run_dir}/last{sf}.pt\n")
        f.write(f"  训练曲线                : {run_dir}/training_history{sf}.png\n")
        if alpha_history:
            f.write(f"  Alpha/Spearman 演化图    : {run_dir}/alpha_spearman{sf}.png\n")
        f.write(f"  tensorboard log         : {Train.logDir}\n")
    print(f"训练日志已保存至: {log_path}")


def main():
    """主训练流程 — 仅在主进程执行，避免 Windows multiprocessing spawn 重复创建文件夹"""
    from data_loader import trainLoader, valLoader

    run_dir, run_idx = create_model_dir()
    Train.modelDir = run_dir + "/"
    Train.logDir = os.path.join(run_dir, "log", time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()))
    sf = f"_model_{run_idx}"

    print(f"\n{'='*60}")
    print(f"训练输出目录: {run_dir}")
    print(f"{'='*60}\n")

    model = weatherModel
    model.to(Common.device)

    # ---- 初始 alpha 权重 ----
    dynamic_alpha = hasattr(Train, 'dynamic_alpha_enabled') and Train.dynamic_alpha_enabled
    eps = Train.focal_loss_alpha_eps
    alpha_source_label = Train.focal_loss_alpha_source
    alpha_update_interval = Train.dynamic_alpha_interval if hasattr(Train, 'dynamic_alpha_interval') else 5
    warmup_epochs = Train.dynamic_alpha_warmup if hasattr(Train, 'dynamic_alpha_warmup') else 5

    if dynamic_alpha:
        learned = hasattr(Train, 'dynamic_alpha_learned') and Train.dynamic_alpha_learned
        if learned:
            lr_a = Train.dynamic_alpha_lr if hasattr(Train, 'dynamic_alpha_lr') else 0.01
            mom = Train.dynamic_alpha_momentum if hasattr(Train, 'dynamic_alpha_momentum') else 0.9
            print(f"[Focal Loss] 动态 alpha 梯度学习模式")
            print(f"  alpha = alpha - lr × ∇(val_loss)")
            print(f"  lr={lr_a}, momentum={mom}, warmup={warmup_epochs}ep")
            print(f"  clamp: [{Train.dynamic_alpha_min if hasattr(Train, 'dynamic_alpha_min') else 0.1}, {Train.dynamic_alpha_max if hasattr(Train, 'dynamic_alpha_max') else 4.0}]")
        else:
            print(f"[Focal Loss] 动态 alpha 公式计算模式")
            print(f"  target: difficulty → normalize → clamp → renormalize")
            print(f"  EMA β=0.8, warmup={warmup_epochs}ep, 每 {alpha_update_interval} ep 更新")
        per_class_acc_init = Train.focal_loss_per_class_acc
        initial_alphas = [1.0 / (per_class_acc_init[c] + eps) for c in Common.labels]
        print(f"[Focal Loss] 初始 alpha 来源: {alpha_source_label}，之后动态调整")
        for c, a in zip(Common.labels, initial_alphas):
            print(f"  {c:>8s}: {a:.4f}  (准确率 {per_class_acc_init[c]:.4f})")
    else:
        per_class_acc_init = Train.focal_loss_per_class_acc
        initial_alphas = [1.0 / (per_class_acc_init[c] + eps) for c in Common.labels]
        print(f"[Focal Loss] alpha 权重来源: {alpha_source_label}")
        for c, a in zip(Common.labels, initial_alphas):
            print(f"  {c:>8s}: {a:.4f}  (准确率 {per_class_acc_init[c]:.4f})")

    alpha_tensor = update_alpha_tensor(initial_alphas)
    criterion = FocalLoss(gamma=Train.focal_loss_gamma, alpha=alpha_tensor)

    # ---- 记录 alpha 和 Spearman 历史 ----
    alpha_history = {}  # {epoch: [alpha_0, alpha_1, ...]}
    spearman_history = {}  # {epoch: rho}
    alpha_history[0] = initial_alphas.copy()
    _ema_alphas = np.array(initial_alphas, dtype=np.float32)  # alpha 状态
    alpha_velocity = np.zeros(len(Common.labels), dtype=np.float32)  # 动量

    optimizer = optim.Adam(model.parameters(), lr=Train.lr)

    # ---- 学习率调度 ----
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

    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, Train.epochs + 1):
        train_acc = train_epoch(epoch, model, trainLoader, criterion, optimizer, scaler, writer, history)
        val_acc = val_epoch(epoch, model, valLoader, criterion, optimizer, writer, history)

        # 更新学习率（CosineAnnealing）
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("lr", current_lr, epoch)

        # ---- 动态更新 alpha (Model 27: 梯度学习 — alpha 从验证集 loss 梯度优化) ----
        if dynamic_alpha and epoch >= warmup_epochs and epoch % alpha_update_interval == 0:
            learned = hasattr(Train, 'dynamic_alpha_learned') and Train.dynamic_alpha_learned

            # 计算 per-class 指标（用于日志）
            per_class_acc, per_class_f1 = compute_per_class_val_metrics(model, valLoader)
            difficulty = np.array([1.0 - per_class_f1[c] for c in Common.labels])
            d_min = Train.dynamic_alpha_min if hasattr(Train, 'dynamic_alpha_min') else 0.1
            d_max = Train.dynamic_alpha_max if hasattr(Train, 'dynamic_alpha_max') else 4.0

            if learned:
                # === 梯度学习模式 ===
                # 1. 计算 d(val_loss)/d(alpha)
                alpha_lr = Train.dynamic_alpha_lr if hasattr(Train, 'dynamic_alpha_lr') else 0.01
                alpha_momentum = Train.dynamic_alpha_momentum if hasattr(Train, 'dynamic_alpha_momentum') else 0.9
                grad = compute_alpha_gradient(model, valLoader, _ema_alphas,
                                              gamma=Train.focal_loss_gamma, device=Common.device)

                # 2. SGD with momentum
                alpha_velocity = alpha_momentum * alpha_velocity + (1 - alpha_momentum) * grad
                _ema_alphas = _ema_alphas - alpha_lr * alpha_velocity

                # 3. clamp + normalize
                _ema_alphas = np.clip(_ema_alphas, d_min, d_max)
                _ema_alphas = _ema_alphas / _ema_alphas.mean()

                alpha_history[epoch] = _ema_alphas.tolist()
                criterion.alpha = update_alpha_tensor(_ema_alphas.tolist()).to(Common.device)

                # Spearman: 学出来的 alpha 是否对准困难类
                rho, _ = spearmanr(_ema_alphas, difficulty)
                spearman_history[epoch] = rho

                print(f"\n  [Alpha 学习] epoch {epoch} (梯度下降, lr={alpha_lr}, momentum={alpha_momentum}):")
                print(f"  {'类别':<10} {'F1':>8} {'难度':>8} {'梯度':>10} {'alpha':>8}")
                for i, c in enumerate(Common.labels):
                    print(f"  {c:<10} {per_class_f1[c]:>8.4f} {difficulty[i]:>8.4f} {grad[i]:>10.6f} {_ema_alphas[i]:>8.4f}")
                print(f"  Spearman ρ = {rho:.4f}  (学习到的 alpha 与当前难度的排序一致性)\n")
            else:
                # === 公式计算模式（回退） ===
                target = np.array(make_alpha_target(difficulty, gamma=1.0, min_w=d_min, max_w=d_max))
                ema_beta = 0.8
                _ema_alphas = _ema_alphas * ema_beta + target * (1.0 - ema_beta)
                _ema_alphas = _ema_alphas / _ema_alphas.mean()

                alpha_history[epoch] = _ema_alphas.tolist()
                criterion.alpha = update_alpha_tensor(_ema_alphas.tolist()).to(Common.device)

                rho, _ = spearmanr(_ema_alphas, difficulty)
                spearman_history[epoch] = rho

                print(f"\n  [Alpha 更新] epoch {epoch} (公式计算, EMA β=0.8):")
                print(f"  {'类别':<10} {'F1':>8} {'难度':>8} {'alpha':>8}")
                for i, c in enumerate(Common.labels):
                    print(f"  {c:<10} {per_class_f1[c]:>8.4f} {difficulty[i]:>8.4f} {_ema_alphas[i]:>8.4f}")
                print(f"  Spearman ρ = {rho:.4f}\n")

        if val_acc > best_acc + Train.early_stop_min_delta:
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, f"best{sf}.pt"))
            print(f">>> 新的最佳模型! Epoch:{epoch} ValAcc:{val_acc:.4f} 已保存")
        else:
            epochs_no_improve += 1

        if Train.early_stop_enabled and epochs_no_improve >= Train.early_stop_patience:
            recent_train_accs = history["train_acc"][-5:] if len(history["train_acc"]) >= 5 else history["train_acc"]
            train_acc_trend = all(recent_train_accs[i] <= recent_train_accs[i+1] + 0.002
                                  for i in range(len(recent_train_accs)-1))

            if train_acc_trend or train_acc > 0.95:
                print(f"\n早停触发: 验证准确率连续 {epochs_no_improve} 个 epoch 无有效上升，")
                print(f"         且训练准确率持续上升（过拟合）或已饱和（>0.95）。")
                print(f"         当前 trainAcc={train_acc:.4f}, valAcc={val_acc:.4f}")
                break

    # 保存 final alpha（如果还没记录）
    if epoch not in alpha_history:
        final_alphas = criterion.alpha.cpu().tolist() if criterion.alpha is not None else [1.0] * 8
        alpha_history[epoch] = final_alphas

    if getattr(Train, 'save_last_model', True):
        torch.save(model.state_dict(), os.path.join(run_dir, f"last{sf}.pt"))
    print(f"\n训练结束。最佳模型在 Epoch {best_epoch}, ValAcc={best_acc:.4f}")

    # 绘制
    plot_history(history, save_path=os.path.join(run_dir, f"training_history{sf}.png"))
    if dynamic_alpha and len(spearman_history) > 0:
        plot_alpha_spearman(alpha_history, spearman_history, alpha_source_label,
                            save_path=os.path.join(run_dir, f"alpha_spearman{sf}.png"))

    save_training_log(run_dir, run_idx, sf, history, best_epoch, best_acc, Train.epochs,
                      alpha_history=alpha_history if dynamic_alpha else None,
                      spearman_history=spearman_history if dynamic_alpha else None)

    writer.close()


if __name__ == '__main__':
    main()
