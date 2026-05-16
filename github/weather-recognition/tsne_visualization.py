# t-SNE 可视化分析
# 功能：对训练集特征做 t-SNE 降维，直观展示各类别在特征空间中的分布

import os
import re
import random
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from sklearn.manifold import TSNE
from config import Common
from model import model as weatherModel

# ============================================================
# 设置中文字体与样式
# ============================================================
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"

def get_chinese_font():
    try:
        fm.fontManager.addfont(FONT_PATH)
        return fm.FontProperties(fname=FONT_PATH)
    except Exception:
        return None

CJK_FONT = get_chinese_font()
if CJK_FONT:
    print(f"[字体] 使用中文字体: {FONT_PATH}")
else:
    print("[字体] 未找到中文字体，将使用默认字体")

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 类别中文名映射
LABEL_NAMES_CN = {
    "cloudy":  "多云",
    "haze":    "薄雾",
    "rainy":   "雨天",
    "shine":   "晴朗",
    "snow":    "大雪",
    "sunny":   "晴天",
    "sunrise": "日出",
    "thunder": "雷雨"
}

# 各类别颜色与标记（8 个类别，确保视觉区分度）
CLASS_STYLE = {
    "cloudy":  {"color": "#4C72B0", "marker": "o", "alpha": 0.6},
    "haze":    {"color": "#DD8452", "marker": "s", "alpha": 0.6},
    "rainy":   {"color": "#55A868", "marker": "^", "alpha": 0.6},
    "shine":   {"color": "#C44E52", "marker": "D", "alpha": 0.6},
    "snow":    {"color": "#8172B3", "marker": "v", "alpha": 0.6},
    "sunny":   {"color": "#CCB974", "marker": "p", "alpha": 0.6},
    "sunrise": {"color": "#64B5CD", "marker": "*", "alpha": 0.7},
    "thunder": {"color": "#E09F2B", "marker": "h", "alpha": 0.6}
}


# ============================================================
# 工具函数：获取最新 model_X 目录
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
        raise FileNotFoundError(f"未找到任何 model_X 文件夹")
    latest_idx = max(indices)
    return os.path.join(MODEL_ROOT, f"model_{latest_idx}"), latest_idx


# ============================================================
# 获取特征与标签（指定模型）
# ============================================================
def get_feature_labels_from_loader(dataLoader, model, device, max_samples_per_class=200):
    """
    从 DataLoader 中提取特征向量和标签
    仅取每类最多 max_samples_per_class 条（防止 t-SNE 计算量过大）
    """
    model.eval()
    features = []
    labels = []

    class_count = {c: 0 for c in Common.labels}

    with torch.no_grad():
        for data, label in dataLoader:
            data = data.to(device)
            feat = model.net(data)  # 取 ResNet-50 backbone 特征（2048 维）
            feat = feat.cpu().numpy()

            label_indices = torch.argmax(label, dim=1).numpy()

            for f, l in zip(feat, label_indices):
                class_name = Common.labels[l]
                if class_count[class_name] < max_samples_per_class:
                    features.append(f)
                    labels.append(l)
                    class_count[class_name] += 1

            if all(c >= max_samples_per_class for c in class_count.values()):
                break

    return np.array(features), np.array(labels)


def get_backbone_features(dataLoader, model, device, max_samples_per_class=200):
    """取 backbone 特征（2048-dim）"""
    return get_feature_labels_from_loader(dataLoader, model, device, max_samples_per_class)


def plot_tsne(features, labels, title, save_path, n_samples_limit=None):
    """绘制 t-SNE 可视化"""
    if n_samples_limit and len(features) > n_samples_limit:
        indices = np.random.choice(len(features), n_samples_limit, replace=False)
        features = features[indices]
        labels = labels[indices]

    print(f"[{title}] 正在进行 t-SNE 降维，样本数: {len(features)}...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca')
    coords = tsne.fit_transform(features)
    print(f"[{title}] t-SNE 完成!")

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    for class_idx, class_name in enumerate(Common.labels):
        mask = (labels == class_idx)
        x = coords[mask, 0]
        y = coords[mask, 1]
        if len(x) == 0:
            continue
        style = CLASS_STYLE.get(class_name, {"color": "gray", "marker": "o", "alpha": 0.5})
        label_cn = LABEL_NAMES_CN.get(class_name, class_name)
        ax.scatter(
            x, y,
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            s=80,
            edgecolors='white',
            linewidths=0.5,
            label=f"{label_cn} ({class_name})"
        )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15, fontproperties=CJK_FONT)
    ax.set_xlabel("t-SNE 维度 1", fontsize=12, fontproperties=CJK_FONT)
    ax.set_ylabel("t-SNE 维度 2", fontsize=12, fontproperties=CJK_FONT)

    legend = ax.legend(
        loc='lower right',
        fontsize=10,
        framealpha=0.9,
        edgecolor='gray',
        fancybox=True,
        shadow=True,
        ncol=1,
        markerscale=1.2,
        borderpad=0.8,
        labelspacing=0.5
    )
    if CJK_FONT:
        legend.get_title().set_fontproperties(CJK_FONT)
        for text in legend.get_texts():
            text.set_fontproperties(CJK_FONT)

    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"已保存: {save_path}")
    plt.close()
    return coords


def plot_tsne_advanced(features, labels, title, save_path, n_samples_limit=None):
    """高级 t-SNE 绘图"""
    if n_samples_limit and len(features) > n_samples_limit:
        indices = np.random.choice(len(features), n_samples_limit, replace=False)
        features = features[indices]
        labels = labels[indices]

    print(f"[{title}] 正在进行 t-SNE 降维，样本数: {len(features)}...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca')
    coords = tsne.fit_transform(features)
    print(f"[{title}] t-SNE 完成!")

    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)

    for class_idx, class_name in enumerate(Common.labels):
        mask = (labels == class_idx)
        x = coords[mask, 0]
        y = coords[mask, 1]
        if len(x) == 0:
            continue
        style = CLASS_STYLE.get(class_name, {"color": "gray", "marker": "o", "alpha": 0.5})
        label_cn = LABEL_NAMES_CN.get(class_name, class_name)
        ax.scatter(
            x, y,
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            s=60,
            edgecolors='white',
            linewidths=0.4,
            label=f"{label_cn} (n={mask.sum()})"
        )

        cx, cy = np.median(x), np.median(y)
        ax.annotate(
            label_cn,
            (cx, cy),
            fontsize=11,
            fontweight='bold',
            color=style["color"],
            ha='center', va='center',
            fontproperties=CJK_FONT,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=style["color"])
        )

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20, fontproperties=CJK_FONT)
    ax.set_xlabel("t-SNE 维度 1", fontsize=13, fontproperties=CJK_FONT)
    ax.set_ylabel("t-SNE 维度 2", fontsize=13, fontproperties=CJK_FONT)
    ax.grid(True, alpha=0.25, linestyle='--')

    legend = ax.legend(
        loc='center left',
        bbox_to_anchor=(1.03, 0.5),
        fontsize=10,
        framealpha=0.9,
        edgecolor='gray',
        fancybox=True,
        shadow=False,
        ncol=1,
        markerscale=1.2,
        borderpad=0.6,
        labelspacing=0.5,
        title="类别",
        title_fontsize=11
    )
    if CJK_FONT:
        legend.get_title().set_fontproperties(CJK_FONT)
        for text in legend.get_texts():
            text.set_fontproperties(CJK_FONT)

    plt.tight_layout()
    plt.subplots_adjust(right=0.80)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"已保存: {save_path}")
    plt.close()


def get_raw_pixel_features(max_samples_per_class=200, target_size=64):
    """
    直接读取原始图片文件，转为像素向量（resize 到 target_size 后 flatten）
    用于在像素空间做 t-SNE 对比
    """
    from PIL import Image

    samples = []  # (image_path, label_index)
    for d in os.listdir(Common.basePath):
        dir_path = os.path.join(Common.basePath, d)
        if not os.path.isdir(dir_path):
            continue
        categoryIndex = Common.labels.index(d)
        image_files = [f for f in os.listdir(dir_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        if len(image_files) > 1000:
            image_files = random.sample(image_files, 1000)
        for imagePath in image_files:
            samples.append((os.path.join(dir_path, imagePath), categoryIndex))

    # 打乱保证随机性（与 WeatherDataSet 的 split 逻辑一致）
    random.shuffle(samples)

    # 70/15/15 分割（对应 train/val/test）
    total = len(samples)
    train_len = int(total * 0.7)
    val_len   = int(total * 0.15)

    all_samples = samples  # 使用全部样本做 t-SNE（保证各类别都有）

    features = []
    labels = []
    class_count = {c: 0 for c in Common.labels}

    for imagePath, label_idx in all_samples:
        class_name = Common.labels[label_idx]
        if class_count[class_name] >= max_samples_per_class:
            continue

        img = Image.open(imagePath).convert('RGB')
        img = img.resize((target_size, target_size))
        arr = np.array(img, dtype=np.float32).flatten() / 255.0
        img.close()

        features.append(arr)
        labels.append(label_idx)
        class_count[class_name] += 1

    return np.array(features), np.array(labels)


# ============================================================
# 主程序：原始像素 t-SNE vs 训练后模型特征 t-SNE 对比
# ============================================================
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data_loader import trainLoader, valLoader

    random.seed(42)
    np.random.seed(42)

    # 自动找最新模型
    run_dir, run_idx = get_latest_model_dir()
    SF = f"_model_{run_idx}"

    print(f"\n{'='*60}")
    print(f"t-SNE 可视化 - model_{run_idx}")
    print(f"{'='*60}\n")

    # ===== 1. 提取原始图片像素特征（不使用任何模型）=====
    print("[原始像素] 正在读取图片并提取像素向量...")
    raw_features, raw_labels = get_raw_pixel_features(max_samples_per_class=300, target_size=64)
    print(f"[原始像素] 样本数: {len(raw_features)}, 特征维度: {raw_features.shape[1]}")

    # ===== 2. 加载训练好的模型，提取 backbone 特征=====
    trained_model = weatherModel
    trained_model.to(Common.device)
    trained_model_path = os.path.join(run_dir, f"best{SF}.pt")
    trained_model.load_state_dict(torch.load(trained_model_path, map_location=Common.device))
    trained_model.eval()
    print(f"[训练模型] 已加载: {trained_model_path}")

    print("\n[训练模型] 正在提取训练集特征...")
    train_feat_trained, train_lbl = get_feature_labels_from_loader(
        trainLoader, trained_model, Common.device, max_samples_per_class=300)

    print("[训练模型] 正在提取验证集特征...")
    val_feat_trained, _ = get_feature_labels_from_loader(
        valLoader, trained_model, Common.device, max_samples_per_class=150)

    model_features = np.vstack([train_feat_trained, val_feat_trained])
    model_labels   = np.concatenate([train_lbl, train_lbl])  # 标签从 train 提取时已生成
    model_labels   = np.concatenate([train_lbl, raw_labels[:len(val_feat_trained)]])  # 用 raw 的标签对应

    # 重新提取验证集标签（与原始像素对齐）
    print("[训练模型] 重新提取验证集标签...")
    _, val_labels = get_feature_labels_from_loader(valLoader, trained_model, Common.device, max_samples_per_class=150)
    model_features = np.vstack([train_feat_trained, val_feat_trained])
    model_labels   = np.concatenate([train_lbl, val_labels])

    print(f"[训练模型] 样本数: {len(model_features)}, 特征维度: {model_features.shape[1]}")
    print(f"样本分布: {np.bincount(model_labels)}")

    # ===== 3. t-SNE 降维 =====
    print("\n[原始像素] 正在进行 t-SNE 降维...")
    tsne_raw = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca')
    coords_raw = tsne_raw.fit_transform(raw_features)
    print("[原始像素] t-SNE 完成!")

    print("[训练模型] 正在进行 t-SNE 降维...")
    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca')
    coords_model = tsne_model.fit_transform(model_features)
    print("[训练模型] t-SNE 完成!")

    # 样本分布（原始像素标签）
    print(f"原始像素样本分布: {np.bincount(raw_labels)}")

    # ===== 4. 绘制并排对比图 =====
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=150)

    for ax, coords, title in [
        (axes[0], coords_raw,  "原始图片像素空间（64×64 resize）"),
        (axes[1], coords_model, "训练后模型特征空间（ResNet-50 2048-dim）")
    ]:
        current_labels = raw_labels if len(coords) == len(raw_labels) else model_labels
        for class_idx, class_name in enumerate(Common.labels):
            mask = (current_labels == class_idx)
            x = coords[mask, 0]
            y = coords[mask, 1]
            if len(x) == 0:
                continue
            style = CLASS_STYLE.get(class_name, {"color": "gray", "marker": "o", "alpha": 0.5})
            label_cn = LABEL_NAMES_CN.get(class_name, class_name)
            ax.scatter(
                x, y,
                c=style["color"], marker=style["marker"], alpha=style["alpha"],
                s=60, edgecolors='white', linewidths=0.4,
                label=f"{label_cn} (n={mask.sum()})"
            )
            cx, cy = np.median(x), np.median(y)
            ax.annotate(
                label_cn, (cx, cy),
                fontsize=10, fontweight='bold', color=style["color"],
                ha='center', va='center',
                fontproperties=CJK_FONT,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=style["color"])
            )
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15, fontproperties=CJK_FONT)
        ax.set_xlabel("t-SNE 维度 1", fontsize=12, fontproperties=CJK_FONT)
        ax.set_ylabel("t-SNE 维度 2", fontsize=12, fontproperties=CJK_FONT)
        ax.grid(True, alpha=0.25, linestyle='--')

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_legend,
        loc='center right', bbox_to_anchor=(1.02, 0.5),
        fontsize=10, framealpha=0.9, edgecolor='gray',
        title="天气类别", title_fontsize=11,
        markerscale=1.2, borderpad=0.6, labelspacing=0.5
    )
    if fig.legends and CJK_FONT:
        for lg in fig.legends:
            lg.get_title().set_fontproperties(CJK_FONT)
            for text in lg.get_texts():
                text.set_fontproperties(CJK_FONT)
    fig.suptitle("天气图片特征空间 t-SNE 对比（8 类天气分类）", fontsize=18, fontweight='bold', y=1.02, fontproperties=CJK_FONT)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    comparison_path = os.path.join(run_dir, f"tsne_comparison{SF}.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n对比图已保存: {comparison_path}")
    plt.close()

    # ===== 5. 单独绘制模型特征的高级版 t-SNE =====
    plot_tsne_advanced(
        model_features, model_labels,
        title="训练后 ResNet-50 特征空间 t-SNE（8 类天气分类）",
        save_path=os.path.join(run_dir, f"tsne_features_advanced{SF}.png"),
        n_samples_limit=None
    )

    # ===== 6. 单独绘制原始像素 t-SNE =====
    plot_tsne_advanced(
        raw_features, raw_labels,
        title="原始图片像素空间 t-SNE（64×64 resize，训练前）",
        save_path=os.path.join(run_dir, f"tsne_raw_pixels{SF}.png"),
        n_samples_limit=None
    )

    print("\n" + "="*60)
    print("t-SNE 可视化完成！")
    print(f"输出文件:")
    print(f"  - {os.path.join(run_dir, f'tsne_comparison{SF}.png')}       ← 左右对比图")
    print(f"  - {os.path.join(run_dir, f'tsne_features_advanced{SF}.png')} ← 训练后模型特征")
    print(f"  - {os.path.join(run_dir, f'tsne_raw_pixels{SF}.png')}        ← 原始像素空间")
    print("="*60)
