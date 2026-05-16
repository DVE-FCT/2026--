# 测试部分
import os
import re
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.amp import autocast
from tqdm import tqdm
from config import Common
from model import model as weatherModel
from data_loader import testLoader


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


run_dir, run_idx = get_latest_model_dir()
SF = f"_model_{run_idx}"  # 文件名后缀，与文件夹编号匹配

model_path = os.path.join(run_dir, f"best{SF}.pt")

model = weatherModel
model.to(Common.device)
model.load_state_dict(torch.load(model_path, map_location=Common.device))
model.eval()
print(f"已加载模型: {model_path}")
print(f"测试输出目录: {run_dir}")

criterion = nn.CrossEntropyLoss()


def test():
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

    print()
    print('========== Test Result ==========')
    print(f"Test Loss: {avgLoss:.4f}")
    print(f"Test Acc : {avgAcc:.4f} ({correctNum}/{len(testLoader.dataset)})")
    print('=================================')

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    labels_list = Common.labels

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_list)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Test Confusion Matrix (Acc={avgAcc:.4f})')

    cm_path = os.path.join(run_dir, f"confusion_matrix{SF}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    print(f"混淆矩阵已保存至: {cm_path}")
    plt.close()

    # 保存测试结果到 txt
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
        f.write('=========================================\n')
    print(f"测试结果已保存至: {result_path}")

    print()
    print("========== Per-Class Accuracy ==========")
    for i, name in enumerate(labels_list):
        class_total = sum(1 for l in all_labels if l == i)
        class_correct = sum(1 for l, p in zip(all_labels, all_preds) if l == i and p == i)
        if class_total > 0:
            class_acc = class_correct / class_total
            print(f"  {name:10s}: {class_correct:4d}/{class_total:4d} = {class_acc:.4f}")
        else:
            print(f"  {name:10s}: 0/0 = N/A")
    print('=========================================')


if __name__ == '__main__':
    test()