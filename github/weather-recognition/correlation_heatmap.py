# 斯皮尔曼相关性热力图分析
# 功能：基于原始像素特征计算各类别间的相关性，展示像素空间中类别混淆程度

import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from config import Common


# ============================================================
# 设置中文字体
# ============================================================
import matplotlib.font_manager as fm

# 构建字体缓存，强制 matplotlib 加载中文字体
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"

def get_chinese_font():
    """获取中文字体 FontProperties"""
    try:
        # 确保字体被 matplotlib 加载
        fm.fontManager.addfont(FONT_PATH)
        prop = fm.FontProperties(fname=FONT_PATH)
        return prop
    except Exception:
        return None

chinese_prop = get_chinese_font()
if chinese_prop:
    print(f"[字体] 使用中文字体: {FONT_PATH}")
else:
    print("[字体] 未找到中文字体，将使用默认字体")

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.style.use('seaborn-v0_8-whitegrid')

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


# ============================================================
# 计算类别像素均值特征
# ============================================================
def compute_class_mean_pixels(target_size=64, max_per_class=500):
    """读取每类图片，resize 后 flatten，得到每类的均值像素向量"""
    from PIL import Image

    class_samples = {c: [] for c in Common.labels}

    for d in os.listdir(Common.basePath):
        dir_path = os.path.join(Common.basePath, d)
        if not os.path.isdir(dir_path):
            continue
        if d not in Common.labels:
            continue

        image_files = [f for f in os.listdir(dir_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        if len(image_files) > max_per_class:
            image_files = random.sample(image_files, max_per_class)

        for imagePath in image_files:
            img = Image.open(os.path.join(dir_path, imagePath)).convert('RGB')
            img = img.resize((target_size, target_size))
            arr = np.array(img, dtype=np.float32).flatten() / 255.0
            img.close()
            class_samples[d].append(arr)

    mean_pixels = {}
    for class_name, samples in class_samples.items():
        if samples:
            mean_pixels[class_name] = np.mean(samples, axis=0)
    return mean_pixels


# ============================================================
# 计算类别间相关性矩阵
# ============================================================
def compute_spearman_matrix(mean_pixels, class_names):
    n = len(class_names)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vi = mean_pixels[class_names[i]]
            vj = mean_pixels[class_names[j]]
            r, _ = spearmanr(vi, vj)
            corr[i, j] = r
    return corr


def compute_pearson_matrix(mean_pixels, class_names):
    n = len(class_names)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vi = mean_pixels[class_names[i]]
            vj = mean_pixels[class_names[j]]
            r, _ = pearsonr(vi, vj)
            corr[i, j] = r
    return corr


# ============================================================
# 绘制热力图（通用函数，使用 prop 控制字体）
# ============================================================
def draw_heatmap(ax, corr, class_names, title):
    """在指定 axes 上绘制单张热力图，使用中文字体"""
    n = len(class_names)
    labels = [LABEL_NAMES_CN.get(c, c) for c in class_names]

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            color = 'white' if abs(val) > 0.65 else 'black'
            ax.text(j, i, f'{val:.3f}',
                    ha='center', va='center',
                    color=color, fontsize=12, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    # 手动设置刻度标签，确保中文字符正确渲染
    ax.set_xticklabels(labels, fontproperties=chinese_prop, rotation=35, ha='right')
    ax.set_yticklabels(labels, fontproperties=chinese_prop)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=12, fontproperties=chinese_prop)
    ax.set_xlabel('天气类别', fontsize=12, fontproperties=chinese_prop)
    ax.set_ylabel('天气类别', fontsize=12, fontproperties=chinese_prop)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('相关系数', fontsize=11, fontproperties=chinese_prop)
    cbar.ax.tick_params(labelsize=9)


# ============================================================
# 绘制双图并排（斯皮尔曼 + 皮尔逊）
# ============================================================
def plot_correlation_dual(mean_pixels, class_names, save_path):
    """左右并排绘制斯皮尔曼和皮尔逊相关性热力图"""
    spearman_corr = compute_spearman_matrix(mean_pixels, class_names)
    pearson_corr  = compute_pearson_matrix(mean_pixels, class_names)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=150)

    draw_heatmap(axes[0], spearman_corr, class_names, "斯皮尔曼相关系数 (Spearman ρ)")
    draw_heatmap(axes[1], pearson_corr,  class_names, "皮尔逊相关系数 (Pearson r)")

    fig.suptitle('8 类天气图片像素空间相关性分析', fontsize=18, fontweight='bold', y=1.02, fontproperties=chinese_prop)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"相关性对比图已保存: {save_path}")
    plt.close()


# ============================================================
# 绘制单张热力图
# ============================================================
def plot_single_correlation(mean_pixels, class_names, method='spearman',
                            save_path=None, cmap='RdBu_r'):
    """绘制单张相关性热力图"""
    if method == 'spearman':
        corr = compute_spearman_matrix(mean_pixels, class_names)
        title = '斯皮尔曼相关性热力图 (Spearman ρ)'
    else:
        corr = compute_pearson_matrix(mean_pixels, class_names)
        title = '皮尔逊相关性热力图 (Pearson r)'

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    draw_heatmap(ax, corr, class_names, f'{title}\n基于 64×64 像素均值向量')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"已保存: {save_path}")
    plt.close()
    return corr


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    MODEL_ROOT = "./model"
    indices = []
    for name in os.listdir(MODEL_ROOT):
        m = re.match(r'^model_(\d+)$', name)
        if m and os.path.isdir(os.path.join(MODEL_ROOT, name)):
            indices.append(int(m.group(1)))
    latest_idx = max(indices) if indices else 0
    SF = f"_model_{latest_idx}"
    run_dir = os.path.join(MODEL_ROOT, f"model_{latest_idx}")

    print(f"\n{'='*60}")
    print(f"相关性热力图分析 - model_{latest_idx}")
    print(f"{'='*60}\n")

    print("[1/3] 正在读取图片并计算像素均值特征...")
    mean_pixels = compute_class_mean_pixels(target_size=64, max_per_class=500)
    class_names = [c for c in Common.labels if c in mean_pixels]
    print(f"      类别数: {len(class_names)}, 特征维度: {mean_pixels[class_names[0]].shape[0] if class_names else 0}")

    print("\n[2/3] 正在计算相关性矩阵...")
    spearman_corr = compute_spearman_matrix(mean_pixels, class_names)
    pearson_corr  = compute_pearson_matrix(mean_pixels, class_names)

    labels_cn = [LABEL_NAMES_CN.get(c, c) for c in class_names]
    header = f"{'':>10}" + "".join([f"{c:>10}" for c in labels_cn])
    print(f"\n斯皮尔曼相关系数矩阵 (Spearman ρ):")
    print(header)
    for i, cn in enumerate(labels_cn):
        row = f"{cn:>10}" + "".join([f"{spearman_corr[i,j]:>10.4f}" for j in range(len(class_names))])
        print(row)

    print(f"\n皮尔逊相关系数矩阵 (Pearson r):")
    print(header)
    for i, cn in enumerate(labels_cn):
        row = f"{cn:>10}" + "".join([f"{pearson_corr[i,j]:>10.4f}" for j in range(len(class_names))])
        print(row)

    print("\n[3/3] 正在绘制热力图...")

    plot_correlation_dual(
        mean_pixels, class_names,
        save_path=os.path.join(run_dir, f"correlation_dual{SF}.png")
    )

    plot_single_correlation(
        mean_pixels, class_names, method='spearman',
        save_path=os.path.join(run_dir, f"correlation_spearman{SF}.png"),
        cmap='RdBu_r'
    )

    plot_single_correlation(
        mean_pixels, class_names, method='pearson',
        save_path=os.path.join(run_dir, f"correlation_pearson{SF}.png"),
        cmap='RdBu_r'
    )

    print("\n" + "="*60)
    print("相关性热力图分析完成！")
    print(f"输出文件:")
    print(f"  - {os.path.join(run_dir, f'correlation_dual{SF}.png')}       ← 斯皮尔曼+皮尔逊对比")
    print(f"  - {os.path.join(run_dir, f'correlation_spearman{SF}.png')} ← 斯皮尔曼单独")
    print(f"  - {os.path.join(run_dir, f'correlation_pearson{SF}.png')}  ← 皮尔逊单独")
    print("="*60)