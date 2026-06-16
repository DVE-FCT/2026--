"""
天气分类训练脚本（精简版）

保留功能：
1. Fine 8 类 Focal Loss 分类；
2. Coarse 2 类辅助分类；
3. Fine/Coarse KL 一致性约束；
4. 公式动态 alpha；
5. 验证 Macro F1 保存模型与早停；
6. TensorBoard、训练曲线和 alpha 演化日志。

已移除：
1. 已验证效果较差且当前关闭的 SupCon；
2. 已废弃的 val_loss 梯度 alpha 分支；
3. 大量只为格式换行产生的冗余代码。
"""

import json
import os
import re
import time
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Common, SEED, Train
from model import build_model
MODEL_ROOT = './model'
AMP_ENABLED = Common.device.type == 'cuda'
CLASS_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD', '#E09F2B']
CLASS_MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
CLASS_LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']
FONT_PATH = 'C:\\Windows\\Fonts\\simhei.ttf'
try:
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.sans-serif'] = ['SimHei']
except Exception:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    """多分类 Focal Loss。"""

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('alpha', alpha.detach().float().clone() if alpha is not None else None)

    def set_alpha(self, alpha):
        self.alpha = alpha.detach().float().clone() if alpha is not None else None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        pt = torch.exp(log_probs.gather(1, targets[:, None]).squeeze(1))
        loss = (1.0 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            loss = loss * self.alpha.to(targets.device).gather(0, targets)
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()

# ============================================================
# 通用工具
# ============================================================
def amp_context():
    return autocast(device_type=Common.device.type, enabled=AMP_ENABLED)

def normalize_alpha(values):
    """将 alpha 归一化到均值为 1。"""
    alpha = torch.as_tensor(values, dtype=torch.float32)
    return alpha / alpha.sum().clamp_min(1e-12) * len(Common.labels)

def build_coarse_info():
    """构建 fine→coarse 映射，以及两组 fine 类别索引。"""
    ambiguous_names = set(getattr(Train, 'coarse_ambiguous_labels', ['cloudy', 'haze', 'rainy', 'sunny']))
    unknown = ambiguous_names.difference(Common.labels)
    if unknown:
        raise ValueError('coarse_ambiguous_labels 中存在未知类别: ' + ', '.join(sorted(unknown)))
    mapping = [0 if name in ambiguous_names else 1 for name in Common.labels]
    ambiguous = [i for i, value in enumerate(mapping) if value == 0]
    distinctive = [i for i, value in enumerate(mapping) if value == 1]
    if not ambiguous or not distinctive:
        raise ValueError('coarse 两组都必须至少包含一个类别。')
    device = Common.device
    return (torch.tensor(mapping, dtype=torch.long, device=device), torch.tensor(ambiguous, dtype=torch.long, device=device), torch.tensor(distinctive, dtype=torch.long, device=device))

def safe_spearman(alpha, difficulty):
    if np.std(alpha) < 1e-12 or np.std(difficulty) < 1e-12:
        return 0.0
    rho, _ = spearmanr(alpha, difficulty)
    return float(rho) if np.isfinite(rho) else 0.0

def get_next_model_index():
    os.makedirs(MODEL_ROOT, exist_ok=True)
    indices = []
    for name in os.listdir(MODEL_ROOT):
        match = re.fullmatch('model_(\\d+)', name)
        if match and os.path.isdir(os.path.join(MODEL_ROOT, name)):
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 1

def create_model_dir():
    index = get_next_model_index()
    run_dir = os.path.join(MODEL_ROOT, f'model_{index}')
    os.makedirs(os.path.join(run_dir, 'log'), exist_ok=True)
    return (run_dir, index)

def save_config_snapshot(run_dir):
    """保存本次实验配置。"""
    snapshot = {}
    for name in dir(Train):
        if name.startswith('_'):
            continue
        value = getattr(Train, name)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            snapshot[name] = value
        elif isinstance(value, (list, tuple, dict)):
            snapshot[name] = value
    snapshot.update(seed=SEED, device=str(Common.device), labels=Common.labels)
    path = os.path.join(run_dir, 'config_snapshot.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)

def confusion_to_metrics(confusion):
    """
    返回：
        per_class_acc：实际等价于每类 Recall，保留旧名称用于兼容；
        per_class_f1；
        overall_acc；
        macro_f1。
    """
    per_class_acc = {}
    per_class_f1 = {}
    for i, class_name in enumerate(Common.labels):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        support = int(confusion[i, :].sum())
        recall = tp / support if support else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_acc[class_name] = recall
        per_class_f1[class_name] = f1
    total = int(confusion.sum())
    overall_acc = float(np.trace(confusion) / total) if total else 0.0
    macro_f1 = float(np.mean(list(per_class_f1.values())))
    return (per_class_acc, per_class_f1, overall_acc, macro_f1)

# ============================================================
# 单轮训练
# ============================================================
def train_epoch(epoch, model, loader, fine_criterion, coarse_criterion, coarse_map, ambiguous_indices, distinctive_indices, optimizer, scaler, writer, history):
    model.train()
    use_coarse = bool(getattr(Train, 'use_coarse_head', False) and getattr(model, 'use_coarse_head', False))
    use_consistency = bool(use_coarse and getattr(Train, 'use_consistency_loss', False) and (epoch >= getattr(Train, 'consistency_start_epoch', 5)))
    coarse_weight = getattr(Train, 'coarse_lambda', 0.1)
    consistency_weight = getattr(Train, 'consistency_lambda', 0.01)
    sums = {'total': 0.0, 'fine': 0.0, 'coarse': 0.0, 'kl': 0.0}
    fine_correct = 0
    coarse_correct = 0
    seen = 0
    progress = tqdm(loader, desc=f'Epoch {epoch}/{Train.epochs} [训练]', ncols=125)
    for images, one_hot_labels in progress:
        images = images.to(Common.device, non_blocking=True)
        one_hot_labels = one_hot_labels.to(Common.device, non_blocking=True)
        fine_labels = one_hot_labels.argmax(dim=1)
        batch_size = images.size(0)
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            if use_coarse:
                fine_logits, coarse_logits = model(images, return_aux=True)
            else:
                fine_logits = model(images)
                coarse_logits = None
            fine_loss = fine_criterion(fine_logits, fine_labels)
            total_loss = fine_loss
            coarse_loss = fine_loss.detach() * 0.0
            coarse_labels = None
            if use_coarse:
                if coarse_logits is None:
                    raise RuntimeError('已启用 coarse head，但模型没有返回 coarse_logits。')
                coarse_labels = coarse_map[fine_labels]
                coarse_loss = coarse_criterion(coarse_logits, coarse_labels)
                total_loss = total_loss + coarse_weight * coarse_loss
            kl_loss = fine_loss.detach() * 0.0
            if use_consistency:
                fine_prob = torch.softmax(fine_logits.float(), dim=1)
                coarse_from_fine = torch.stack([fine_prob.index_select(1, ambiguous_indices).sum(dim=1), fine_prob.index_select(1, distinctive_indices).sum(dim=1)], dim=1).detach()
                kl_loss = F.kl_div(F.log_softmax(coarse_logits.float(), dim=1), coarse_from_fine, reduction='batchmean')
                total_loss = total_loss + consistency_weight * kl_loss
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        fine_predictions = fine_logits.argmax(dim=1)
        fine_correct += int((fine_predictions == fine_labels).sum().item())
        if use_coarse:
            coarse_predictions = coarse_logits.argmax(dim=1)
            coarse_correct += int((coarse_predictions == coarse_labels).sum().item())
        seen += batch_size
        sums['total'] += total_loss.item() * batch_size
        sums['fine'] += fine_loss.item() * batch_size
        sums['coarse'] += coarse_loss.item() * batch_size
        sums['kl'] += kl_loss.item() * batch_size
        postfix = {'loss': f'{total_loss.item():.4f}', 'fine': f'{fine_loss.item():.4f}', 'acc': f'{(fine_predictions == fine_labels).float().mean().item():.4f}'}
        if use_coarse:
            postfix['coarse'] = f'{coarse_loss.item():.4f}'
        if use_consistency:
            postfix['kl'] = f'{kl_loss.item():.4f}'
        progress.set_postfix(postfix)
    count = max(seen, 1)
    metrics = {'total_loss': sums['total'] / count, 'fine_loss': sums['fine'] / count, 'coarse_loss': sums['coarse'] / count, 'kl_loss': sums['kl'] / count, 'accuracy': fine_correct / count, 'coarse_accuracy': coarse_correct / count if use_coarse else 0.0, 'lr': optimizer.param_groups[0]['lr']}
    print(f"Epoch:{epoch}  Total:{metrics['total_loss']:.4f}  Fine:{metrics['fine_loss']:.4f}  Coarse:{metrics['coarse_loss']:.4f}  KL:{metrics['kl_loss']:.4f}  FineAcc:{metrics['accuracy']:.4f}  CoarseAcc:{metrics['coarse_accuracy']:.4f}  LR:{metrics['lr']:.2e}")
    history['train_total_loss'].append(metrics['total_loss'])
    history['train_fine_loss'].append(metrics['fine_loss'])
    history['train_coarse_loss'].append(metrics['coarse_loss'])
    history['train_kl_loss'].append(metrics['kl_loss'])
    history['train_acc'].append(metrics['accuracy'])
    history['train_coarse_acc'].append(metrics['coarse_accuracy'])
    history['lr'].append(metrics['lr'])
    for key, value in metrics.items():
        writer.add_scalar(f'train/{key}', value, epoch)
    return metrics

# ============================================================
# 验证与通用评估
# ============================================================
@torch.no_grad()
def evaluate_loader(model, loader, criterion=None, coarse_map=None, description=None):
    model.eval()
    confusion = np.zeros((len(Common.labels), len(Common.labels)), dtype=np.int64)
    use_coarse = bool(coarse_map is not None and getattr(Train, 'use_coarse_head', False) and getattr(model, 'use_coarse_head', False))
    loss_sum = 0.0
    coarse_correct = 0
    seen = 0
    iterator = tqdm(loader, desc=description, ncols=125) if description else loader
    for images, one_hot_labels in iterator:
        images = images.to(Common.device, non_blocking=True)
        one_hot_labels = one_hot_labels.to(Common.device, non_blocking=True)
        fine_labels = one_hot_labels.argmax(dim=1)
        with amp_context():
            if use_coarse:
                fine_logits, coarse_logits = model(images, return_aux=True)
            else:
                fine_logits = model(images)
                coarse_logits = None
            batch_loss = criterion(fine_logits, fine_labels) if criterion is not None else None
        predictions = fine_logits.argmax(dim=1)
        batch_size = images.size(0)
        seen += batch_size
        labels_np = fine_labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        np.add.at(confusion, (labels_np, predictions_np), 1)
        if use_coarse:
            coarse_labels = coarse_map[fine_labels]
            coarse_correct += int((coarse_logits.argmax(dim=1) == coarse_labels).sum().item())
        if batch_loss is not None:
            loss_sum += batch_loss.item() * batch_size
        if description:
            iterator.set_postfix({'loss': f'{batch_loss.item():.4f}' if batch_loss is not None else '-', 'acc': f'{(predictions == fine_labels).float().mean().item():.4f}'})
    per_acc, per_f1, overall_acc, macro_f1 = confusion_to_metrics(confusion)
    return {'loss': loss_sum / max(seen, 1) if criterion is not None else None, 'accuracy': overall_acc, 'macro_f1': macro_f1, 'coarse_accuracy': coarse_correct / max(seen, 1) if use_coarse else 0.0, 'per_class_acc': per_acc, 'per_class_f1': per_f1, 'confusion': confusion}

def validate_epoch(epoch, model, loader, criterion, coarse_map, writer, history):
    metrics = evaluate_loader(model, loader, criterion=criterion, coarse_map=coarse_map, description=f'Epoch {epoch}/{Train.epochs} [验证]')
    print(f"Epoch:{epoch}  ValLoss:{metrics['loss']:.4f}  FineAcc:{metrics['accuracy']:.4f}  CoarseAcc:{metrics['coarse_accuracy']:.4f}  MacroF1:{metrics['macro_f1']:.4f}")
    history['val_loss'].append(metrics['loss'])
    history['val_acc'].append(metrics['accuracy'])
    history['val_coarse_acc'].append(metrics['coarse_accuracy'])
    history['val_macro_f1'].append(metrics['macro_f1'])
    writer.add_scalar('val/loss', metrics['loss'], epoch)
    writer.add_scalar('val/accuracy', metrics['accuracy'], epoch)
    writer.add_scalar('val/coarse_accuracy', metrics['coarse_accuracy'], epoch)
    writer.add_scalar('val/macro_f1', metrics['macro_f1'], epoch)
    return metrics

# ============================================================
# 动态 alpha 更新
# ============================================================
def update_dynamic_alpha(epoch, model, criterion, train_eval_loader, val_metrics, coarse_map):
    """
    alpha = (difficulty + eps)^power / mean(...)

    difficulty 默认来自无增强训练评估集的每类 F1；
    也可以通过 dynamic_alpha_from_train=False 改为验证集 F1。
    """
    use_train = getattr(Train, 'dynamic_alpha_from_train', True)
    if use_train:
        source_metrics = evaluate_loader(model, train_eval_loader, coarse_map=coarse_map)
        source_name = 'train_eval'
    else:
        source_metrics = val_metrics
        source_name = 'val'
    source_f1 = source_metrics['per_class_f1']
    val_f1 = val_metrics['per_class_f1']
    difficulty = np.asarray([1.0 - source_f1[name] for name in Common.labels], dtype=np.float64)
    val_difficulty = np.asarray([1.0 - val_f1[name] for name in Common.labels], dtype=np.float64)
    power = getattr(Train, 'alpha_power', 1.0)
    epsilon = getattr(Train, 'focal_loss_alpha_eps', 0.001)
    new_alpha = np.power(difficulty + epsilon, power)
    new_alpha = new_alpha / (new_alpha.mean() + 1e-08)
    criterion.set_alpha(normalize_alpha(new_alpha).to(Common.device))
    rho = safe_spearman(new_alpha, val_difficulty)
    print(f'\n[动态 Alpha] epoch={epoch}  来源={source_name}  power={power}')
    print(f"{'类别':<12}{'来源F1':>12}{'Alpha':>12}{'验证F1':>12}{'验证难度':>12}")
    for i, name in enumerate(Common.labels):
        print(f'{name:<12}{source_f1[name]:>12.4f}{new_alpha[i]:>12.4f}{val_f1[name]:>12.4f}{val_difficulty[i]:>12.4f}')
    print(f'Spearman rho={rho:.4f}\n')
    return (new_alpha.astype(np.float32), rho)

# ============================================================
# 绘图
# ============================================================
def plot_history(history, save_path):
    epochs = range(1, len(history['train_total_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    axes[0].plot(epochs, history['train_total_loss'], label='Train Total Loss')
    axes[0].plot(epochs, history['train_fine_loss'], label='Train Fine Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Fine Loss')
    axes[0].set(title='Loss Curves', xlabel='Epoch', ylabel='Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, history['train_acc'], label='Train Fine Accuracy')
    axes[1].plot(epochs, history['val_acc'], label='Val Fine Accuracy')
    if getattr(Train, 'use_coarse_head', False):
        axes[1].plot(epochs, history['val_coarse_acc'], label='Val Coarse Accuracy')
    axes[1].set(title='Accuracy Curves', xlabel='Epoch', ylabel='Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(epochs, history['val_macro_f1'], label='Val Macro F1')
    axes[2].set(title='Validation Macro F1', xlabel='Epoch', ylabel='Macro F1')
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'训练曲线已保存至: {save_path}')

def plot_alpha_spearman(alpha_history, spearman_history, alpha_source, save_path):
    epochs = sorted(alpha_history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
    if spearman_history:
        rho_epochs = sorted(spearman_history)
        ax1.plot(rho_epochs, [spearman_history[e] for e in rho_epochs], 'o-', label='Spearman rho')
        ax1.axhline(1.0, linestyle='--', label='rho=1')
        ax1.set_ylim(-1.05, 1.05)
    ax1.set(title='Alpha-Difficulty Correlation', xlabel='Epoch', ylabel='Spearman rho')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    for i, name in enumerate(Common.labels):
        ax2.plot(epochs, [alpha_history[e][i] for e in epochs], color=CLASS_COLORS[i % len(CLASS_COLORS)], marker=CLASS_MARKERS[i % len(CLASS_MARKERS)], linestyle=CLASS_LINESTYLES[i % len(CLASS_LINESTYLES)], linewidth=1.7, markersize=5, label=name, markevery=max(1, len(epochs) // 10))
    ax2.axhline(1.0, linestyle='--', label='Uniform alpha')
    ax2.set(title=f'Focal Alpha Evolution ({alpha_source})', xlabel='Epoch', ylabel='Alpha Weight')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Alpha 演化图已保存至: {save_path}')

# ============================================================
# 日志保存
# ============================================================
def save_training_log(run_dir, run_index, suffix, history, best_epoch, best_macro_f1, best_acc_at_best_f1, max_val_acc, alpha_history=None, spearman_history=None):
    alpha_history = alpha_history or {}
    spearman_history = spearman_history or {}
    log_path = os.path.join(run_dir, f'training_log{suffix}.txt')
    lines = ['=' * 80, f'训练记录 - model_{run_index}', '=' * 80, '', '【核心配置】', f"主干网络              : {getattr(Train, 'backbone', 'resnet50')}", f"使用 SE 注意力        : {getattr(Train, 'use_se_attention', False)}", f"使用 coarse head      : {getattr(Train, 'use_coarse_head', False)}", f"coarse 损失权重       : {getattr(Train, 'coarse_lambda', 0.0)}", f"使用 KL 一致性损失    : {getattr(Train, 'use_consistency_loss', False)}", f"KL 损失权重           : {getattr(Train, 'consistency_lambda', 0.0)}", f"KL 开始 Epoch         : {getattr(Train, 'consistency_start_epoch', 0)}", f"Focal gamma           : {getattr(Train, 'focal_loss_gamma', 1.0)}", f"Alpha 初始来源        : {getattr(Train, 'focal_loss_alpha_source', 'uniform')}", f"Alpha power           : {getattr(Train, 'alpha_power', 1.0)}", f"动态 Alpha            : {getattr(Train, 'dynamic_alpha_enabled', False)}", f"动态 Alpha 更新间隔   : {getattr(Train, 'dynamic_alpha_interval', 1)}", f'Batch Size            : {Train.batch_size}', f'初始学习率            : {Train.lr}', f"学习率调度            : {getattr(Train, 'lr_scheduler', None)}", f'最大 Epoch            : {Train.epochs}', f'随机种子              : {SEED}', f'运行设备              : {Common.device}', '', '【最佳验证结果】', f'最佳 Epoch                    : {best_epoch}', f'最佳验证 Macro F1             : {best_macro_f1:.6f}', f'Best F1 对应验证准确率        : {best_acc_at_best_f1:.6f}', f'训练过程最高验证准确率        : {max_val_acc:.6f}', '', '【每轮指标】', 'Epoch\tTrainTotal\tTrainFine\tTrainCoarse\tTrainKL\tTrainAcc\tTrainCoarseAcc\tValLoss\tValAcc\tValCoarseAcc\tValMacroF1\tLR']
    for i in range(len(history['train_total_loss'])):
        lines.append(f"{i + 1}\t{history['train_total_loss'][i]:.6f}\t{history['train_fine_loss'][i]:.6f}\t{history['train_coarse_loss'][i]:.6f}\t{history['train_kl_loss'][i]:.6f}\t{history['train_acc'][i]:.6f}\t{history['train_coarse_acc'][i]:.6f}\t{history['val_loss'][i]:.6f}\t{history['val_acc'][i]:.6f}\t{history['val_coarse_acc'][i]:.6f}\t{history['val_macro_f1'][i]:.6f}\t{history['lr'][i]:.8f}")
    if alpha_history:
        lines.extend(['', '【动态 Alpha 历史】', 'Epoch\t' + '\t'.join(Common.labels) + '\tSpearman'])
        for epoch in sorted(alpha_history):
            values = alpha_history[epoch]
            if len(values) != len(Common.labels):
                raise ValueError(f'epoch={epoch} 的 alpha 数量与类别数不一致。')
            lines.append(f'{epoch}\t' + '\t'.join((f'{value:.6f}' for value in values)) + f'\t{spearman_history.get(epoch, 0.0):.6f}')
    with open(log_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))
    print(f'训练日志已保存至: {log_path}')

# ============================================================
# 主训练流程
# ============================================================
def main():
    from data_loader import trainEvalLoader, trainLoader, valLoader
    if getattr(Train, 'supcon_enabled', False):
        raise ValueError('该精简版已移除 SupCon。请将 supcon_enabled 设置为 False。')
    if getattr(Train, 'dynamic_alpha_learned', False):
        raise ValueError('该精简版只保留公式动态 alpha，请将 dynamic_alpha_learned 设置为 False。')
    run_dir, run_index = create_model_dir()
    suffix = f'_model_{run_index}'
    Train.modelDir = run_dir + '/'
    Train.logDir = os.path.join(run_dir, 'log', time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()))
    save_config_snapshot(run_dir)
    print('\n' + '=' * 60)
    print(f'训练输出目录: {run_dir}')
    print('=' * 60 + '\n')
    model, backbone_name, feature_dim = build_model()
    model.to(Common.device)
    parameter_count = sum((parameter.numel() for parameter in model.parameters())) / 1000000.0
    print(f"[模型] backbone={backbone_name}, feature_dim={feature_dim}, parameters={parameter_count:.2f}M, coarse_head={getattr(model, 'use_coarse_head', False)}")
    initial_alpha = np.ones(len(Common.labels), dtype=np.float32)
    fine_criterion = FocalLoss(gamma=Train.focal_loss_gamma, alpha=normalize_alpha(initial_alpha)).to(Common.device)
    coarse_criterion = nn.CrossEntropyLoss()
    coarse_map, ambiguous_indices, distinctive_indices = build_coarse_info()
    optimizer_name = getattr(Train, 'optimizer', 'Adam').lower()
    optimizer_class = optim.AdamW if optimizer_name == 'adamw' else optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=Train.lr, weight_decay=getattr(Train, 'weight_decay', 0.0))
    scheduler = None
    if getattr(Train, 'lr_scheduler', None) == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Train.epochs, eta_min=getattr(Train, 'lr_min', 1e-06))
    scaler = GradScaler('cuda', enabled=AMP_ENABLED)
    writer = SummaryWriter(log_dir=Train.logDir, flush_secs=500)
    history = {'train_total_loss': [], 'train_fine_loss': [], 'train_coarse_loss': [], 'train_kl_loss': [], 'train_acc': [], 'train_coarse_acc': [], 'val_loss': [], 'val_acc': [], 'val_coarse_acc': [], 'val_macro_f1': [], 'lr': []}
    dynamic_alpha = getattr(Train, 'dynamic_alpha_enabled', False)
    alpha_interval = getattr(Train, 'dynamic_alpha_interval', 1)
    alpha_warmup = getattr(Train, 'dynamic_alpha_warmup', 0)
    alpha_history = {0: initial_alpha.tolist()}
    spearman_history = {}
    best_macro_f1 = -float('inf')
    best_acc_at_best_f1 = 0.0
    max_val_acc = 0.0
    best_epoch = 0
    patience_reference_f1 = -float('inf')
    epochs_no_improve = 0
    try:
        for epoch in range(1, Train.epochs + 1):
            train_metrics = train_epoch(epoch, model, trainLoader, fine_criterion, coarse_criterion, coarse_map, ambiguous_indices, distinctive_indices, optimizer, scaler, writer, history)
            val_metrics = validate_epoch(epoch, model, valLoader, fine_criterion, coarse_map, writer, history)
            val_acc = val_metrics['accuracy']
            val_macro_f1 = val_metrics['macro_f1']
            max_val_acc = max(max_val_acc, val_acc)
            if val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                best_acc_at_best_f1 = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(run_dir, f'best{suffix}.pt'))
                print(f'>>> 新的最佳模型：Epoch={epoch}, ValMacroF1={val_macro_f1:.4f}, ValAcc={val_acc:.4f}')
            if dynamic_alpha and epoch >= alpha_warmup and (epoch % alpha_interval == 0):
                new_alpha, rho = update_dynamic_alpha(epoch, model, fine_criterion, trainEvalLoader, val_metrics, coarse_map)
                alpha_history[epoch] = new_alpha.tolist()
                spearman_history[epoch] = rho
            if scheduler is not None:
                scheduler.step()
            min_delta = getattr(Train, 'early_stop_min_delta', 0.005)
            if patience_reference_f1 == -float('inf') or val_macro_f1 > patience_reference_f1 + min_delta:
                patience_reference_f1 = val_macro_f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if getattr(Train, 'early_stop_enabled', True) and epochs_no_improve >= getattr(Train, 'early_stop_patience', 15):
                print(f'\n早停触发：验证 Macro F1 连续 {epochs_no_improve} 个 epoch 没有超过 min_delta={min_delta} 的有效提升。')
                print(f"TrainAcc={train_metrics['accuracy']:.4f}, ValAcc={val_acc:.4f}, ValMacroF1={val_macro_f1:.4f}")
                break
        if getattr(Train, 'save_last_model', False):
            torch.save(model.state_dict(), os.path.join(run_dir, f'last{suffix}.pt'))
        print(f'\n训练结束。最佳 Epoch={best_epoch}, ValMacroF1={best_macro_f1:.4f}, Acc@BestF1={best_acc_at_best_f1:.4f}, MaxValAcc={max_val_acc:.4f}')
        plot_history(history, os.path.join(run_dir, f'training_history{suffix}.png'))
        if dynamic_alpha and spearman_history:
            plot_alpha_spearman(alpha_history, spearman_history, getattr(Train, 'focal_loss_alpha_source', 'uniform_then_dynamic'), os.path.join(run_dir, f'alpha_spearman{suffix}.png'))
        save_training_log(run_dir, run_index, suffix, history, best_epoch, best_macro_f1, best_acc_at_best_f1, max_val_acc, alpha_history if dynamic_alpha else None, spearman_history if dynamic_alpha else None)
    finally:
        writer.close()
if __name__ == '__main__':
    main()
