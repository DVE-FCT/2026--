# 测试部分
import os
import re
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.amp import autocast
from tqdm import tqdm
from PIL import Image
from config import Common, Train
from data_loader import testLoader, test_dataset


# ============================================================
# 自动定位最新的 model_X 文件夹
# ============================================================
MODEL_ROOT = "./model"

def get_latest_model_dir():
    if not os.path.exists(MODEL_ROOT):
        raise FileNotFoundError(f"未找到模型目录: {MODEL_ROOT}")
    indices = []
    for name in os.listdir(MODEL_ROOT):
        m = re.match(r'^model_(\d+)$', name)
        if m and os.path.isdir(os.path.join(MODEL_ROOT, name)):
            indices.append(int(m.group(1)))
    if not indices:
        raise FileNotFoundError(f"未找到任何 model_X 文件夹: {MODEL_ROOT}")
    latest_idx = max(indices)
    return os.path.join(MODEL_ROOT, f"model_{latest_idx}"), latest_idx


def save_misclassified_images(test_dataset, all_labels, all_preds, run_dir):
    """
    保存所有错分图片用于人工分析。
    每个错分类别对保存最多 Train.misclassified_per_pair_limit 张图片。
    """
    labels_list = Common.labels
    per_pair_limit = Train.misclassified_per_pair_limit if hasattr(Train, 'misclassified_per_pair_limit') else 30

    test_samples = test_dataset.samples

    misclassified_by_pair = {}
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        if pred != label:
            true_label = labels_list[label]
            pred_label = labels_list[pred]
            pair = (true_label, pred_label)

            if pair not in misclassified_by_pair:
                misclassified_by_pair[pair] = []
            image_path, _ = test_samples[i]
            misclassified_by_pair[pair].append({'image_path': image_path})

    misclassified_dir = os.path.join(run_dir, "misclassified_images")
    os.makedirs(misclassified_dir, exist_ok=True)

    saved_count = {}
    for pair, samples in misclassified_by_pair.items():
        true_label, pred_label = pair
        pair_dir = os.path.join(misclassified_dir, f"{true_label}_to_{pred_label}")
        os.makedirs(pair_dir, exist_ok=True)

        selected = samples[:per_pair_limit]
        pair_saved = 0
        for idx, sample in enumerate(selected):
            image_path = sample['image_path']
            original_filename = os.path.basename(image_path)
            save_name = f"{idx:03d}_{original_filename}"
            save_path = os.path.join(pair_dir, save_name)
            try:
                img = Image.open(image_path)
                img.save(save_path)
                img.close()
                pair_saved += 1
            except Exception as e:
                print(f"  保存图片失败: {image_path} -> {e}")

        saved_count[pair] = pair_saved
        print(f"  {true_label} -> {pred_label}: 保存 {pair_saved} 张（总共 {len(samples)} 个错分）")

    # 汇总报告
    summary_path = os.path.join(misclassified_dir, "misclassification_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("错分图片保存汇总\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"每对保存上限：{per_pair_limit} 张\n\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'类别对':<30} {'错分数':<10} {'已保存':<10}\n")
        f.write("-" * 50 + "\n")
        for pair in sorted(misclassified_by_pair.keys(), key=lambda p: len(misclassified_by_pair[p]), reverse=True):
            true_label, pred_label = pair
            total = len(misclassified_by_pair[pair])
            saved = saved_count.get(pair, 0)
            f.write(f"{true_label} -> {pred_label:<20} {total:<10} {saved:<10}\n")

    print(f"\n错分图片已保存至: {misclassified_dir}")
    print(f"汇总报告: {summary_path}")
    return misclassified_dir


def test():
    from model import model as weatherModel

    run_dir, run_idx = get_latest_model_dir()
    SF = f"_model_{run_idx}"
    model_path = os.path.join(run_dir, f"best{SF}.pt")

    model = weatherModel
    model.to(Common.device)
    model.load_state_dict(torch.load(model_path, map_location=Common.device))
    model.eval()
    print(f"已加载模型: {model_path}")
    print(f"测试输出目录: {run_dir}")

    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    testLoss = 0
    correctNum = 0

    pbar = tqdm(testLoader, desc="[Test]", ncols=100)
    with torch.no_grad():
        for data, label in pbar:
            data, label = data.to(Common.device, non_blocking=True), label.to(Common.device, non_blocking=True)
            with autocast('cuda'):
                output = model(data)
                loss = criterion(output, label)
            testLoss += loss.item() * data.size(0)

            preds = torch.argmax(output, dim=1).cpu().numpy()
            labels = torch.argmax(label, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

            batchCorrectNum = sum(p == l for p, l in zip(preds, labels))
            correctNum += batchCorrectNum
            batchAcc = batchCorrectNum / data.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batchAcc:.4f}"})

    avgLoss = testLoss / len(testLoader.dataset)
    avgAcc = correctNum / len(testLoader.dataset)
    labels_list = Common.labels

    print()
    print('========== Test Result ==========')
    print(f"Test Loss: {avgLoss:.4f}")
    print(f"Test Acc : {avgAcc:.4f} ({correctNum}/{len(testLoader.dataset)})")
    print('=================================')

    # ========== 各类别指标（Precision/Recall/F1）==========
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(labels_list))), average=None, zero_division=0
    )

    print()
    print('========== Per-Class Metrics ==========')
    print(f"{'类别':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print('-' * 52)
    for i, name in enumerate(labels_list):
        print(f"{name:<10} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10d}")
    print('-' * 52)

    # 总体 Macro 和 Weighted F1
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / support.sum()
    print(f"{'Macro F1':<10} {macro_f1:>36.4f}")
    print(f"{'Weighted F1':<10} {weighted_f1:>34.4f}")
    print('=========================================')

    # 打印 sklearn 完整报告
    print()
    print('========== Classification Report ==========')
    print(classification_report(all_labels, all_preds, target_names=labels_list, zero_division=0))

    # ========== 混淆矩阵数值（先计算，再写入 txt）==========
    cm = confusion_matrix(all_labels, all_preds)

    # ========== 保存完整测试结果到 txt ==========
    result_path = os.path.join(run_dir, f"test_result{SF}.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"测试时间:\n")
        f.write(f"模型路径: {model_path}\n\n")
        f.write("========== Test Result ==========\n")
        f.write(f"Test Loss: {avgLoss:.4f}\n")
        f.write(f"Test Acc : {avgAcc:.4f} ({correctNum}/{len(testLoader.dataset)})\n")
        f.write("=================================\n\n")

        f.write("========== Per-Class Accuracy ==========\n")
        for i, name in enumerate(labels_list):
            class_total = sum(1 for l in all_labels if l == i)
            class_correct = sum(1 for l, p in zip(all_labels, all_preds) if l == i and p == i)
            if class_total > 0:
                class_acc = class_correct / class_total
                f.write(f"  {name:10s}: {class_correct:4d}/{class_total:4d} = {class_acc:.4f}\n")
            else:
                f.write(f"  {name:10s}: 0/0 = N/A\n")
        f.write('=========================================\n\n')

        f.write("========== Per-Class Metrics (P/R/F1) ==========\n")
        f.write(f"{'类别':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        f.write('-' * 52 + '\n')
        for i, name in enumerate(labels_list):
            f.write(f"{name:<10} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10d}\n")
        f.write('-' * 52 + '\n')
        f.write(f"{'Macro F1':<10} {macro_f1:>36.4f}\n")
        f.write(f"{'Weighted F1':<10} {weighted_f1:>34.4f}\n")
        f.write('=========================================\n\n')

        f.write("========== Classification Report ==========\n")
        f.write(classification_report(all_labels, all_preds, target_names=labels_list, zero_division=0))

        f.write("\n========== Confusion Matrix ==========\n")
        col_labels = "True\\Pred"
        f.write(f"{col_labels:<12}")
        for name in labels_list:
            f.write(f"{name:<12}")
        f.write("\n")
        f.write("\n")
        for i, row in enumerate(cm):
            f.write(f"{labels_list[i]:<12}")
            for val in row:
                f.write(f"{val:<12}")
            f.write("\n")
        f.write("\n矩阵数值说明：行=真实类别，列=预测类别，对角线=正确数\n")
    print(f"测试结果已保存至: {result_path}")

    # ========== 绘制混淆矩阵 ==========
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_list)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Test Confusion Matrix (Acc={avgAcc:.4f})')

    cm_path = os.path.join(run_dir, f"confusion_matrix{SF}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    print(f"混淆矩阵已保存至: {cm_path}")
    plt.close()

    # ========== 保存错分图片用于人工分析 ==========
    if getattr(Train, 'save_misclassified_images', False):
        save_misclassified_images(test_dataset, all_labels, all_preds, run_dir)


if __name__ == '__main__':
    test()