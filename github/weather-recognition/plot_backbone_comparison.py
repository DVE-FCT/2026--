"""
Backbone 对比气泡图: 横轴=推理速度(ms/张) 纵轴=参数量(M) 气泡=Macro F1
"""
import sys, os, time, importlib
sys.path.insert(0, ".")
import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
try:
    fm.fontManager.addfont(FONT_PATH)
    CJK_FONT = fm.FontProperties(fname=FONT_PATH)
except Exception:
    CJK_FONT = None
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


def benchmark_one(backbone_name, model_dir):
    """为指定 backbone 构建模型并测试推理速度"""
    import torch
    from config import Train, Common
    from data_loader import test_dataset
    from torch.utils.data import DataLoader

    # 切换 backbone 并重新加载模型模块
    Train.backbone = backbone_name
    import model as _m
    importlib.reload(_m)
    model = _m.model
    model.to('cuda')
    num = model_dir.split('_')[-1]
    ckpt = f"{model_dir}/best_model_{num}.pt"
    model.load_state_dict(
        torch.load(ckpt, map_location='cuda', weights_only=True),
        strict=False)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    for data, _ in loader:
        data = data.to('cuda')
        with torch.no_grad():
            _ = model(data)
        break
    torch.cuda.synchronize()

    times = []
    n = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to('cuda')
            t0 = time.time()
            _ = model(data)
            torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000 / 32)
            n += 1
            if n >= 10:
                break
    return np.mean(times), params_m


BACKBONES = [
    ("ResNet101",   "resnet101",       "./model/model_38", None,   "#E09F2B"),  # TBD
    ("ResNet50",    "resnet50",        "./model/model_22", 0.9394, "#4C72B0"),
    ("ConvNeXt-T",  "convnext_tiny",   "./model/model_34", 0.9286, "#DD8452"),
    ("ConvNeXt-S",  "convnext_small",  "./model/model_35", 0.9031, "#55A868"),
    ("InceptionV3", "inception_v3",    "./model/model_36", 0.8816, "#C44E52"),
    ("Efficient-B3","efficientnet_b3", "./model/model_37", 0.9016, "#8172B3"),
]

results = []
for name, bb, path, f1, color in BACKBONES:
    try:
        ms_img, params_m = benchmark_one(bb, path)
        # 若 F1 未指定，从 test_result 自动读取
        if f1 is None:
            import re
            tf = path + "/test_result_" + path.split("/")[-1] + ".txt"
            with open(tf, encoding='utf-8') as fh:
                for line in fh:
                    m = re.search(r'Macro F1\s+([\d.]+)', line)
                    if m:
                        f1 = float(m.group(1))
                        break
        results.append((name, bb, params_m, ms_img, f1, color))
        print(f"{name:12s} | {params_m:.1f}M | {ms_img:.2f}ms | F1={f1:.4f}")
    except Exception as e:
        print(f"{name}: SKIP ({e})")

results = [r for r in results if r[4] is not None]  # 过滤未训练的
results.sort(key=lambda x: x[4], reverse=True)

# 气泡大小 = 相对 F1 比例映射到 300-800 范围
f1_vals = [r[4] for r in results]
f1_min, f1_max = min(f1_vals), max(f1_vals)
MIN_SIZE, MAX_SIZE = 200, 700

def bubble_size(f1):
    if f1_max == f1_min:
        return MAX_SIZE
    return MIN_SIZE + (f1 - f1_min) / (f1_max - f1_min) * (MAX_SIZE - MIN_SIZE)

# ---- 绘图 ----
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
for name, bb, params_m, ms_img, f1, color in results:
    sz = bubble_size(f1)
    ax.scatter(ms_img, params_m, s=sz, alpha=0.7, color=color,
               edgecolors='white', linewidths=1.5, zorder=3)
    ax.annotate(f"{name}\n{f1*100:.1f}% {ms_img:.1f}ms",
                xy=(ms_img, params_m), xytext=(8, 8), textcoords='offset points',
                fontsize=9, fontweight='bold', color=color, alpha=0.9,
                fontproperties=CJK_FONT)

for name, bb, params_m, ms_img, f1, color in results:
    sz = bubble_size(f1)
    ax.scatter([], [], s=sz, alpha=0.7, color=color,
               edgecolors='white', linewidths=1.5,
               label=f"{name}  F1={f1:.4f}  {ms_img:.1f}ms  {params_m:.1f}M")

legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                   fontsize=10, framealpha=0.9, edgecolor='gray',
                   borderpad=0.8, labelspacing=0.8, markerscale=0.8)
if CJK_FONT:
    for text in legend.get_texts():
        text.set_fontproperties(CJK_FONT)

ax.set_xlabel("Inference Latency (ms / image)", fontsize=13, fontweight='bold')
ax.set_ylabel("Parameters (M)", fontsize=13, fontweight='bold')
ax.set_title("Backbone Comparison  |  Bubble = Macro F1  |  Top-Left = Best",
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.subplots_adjust(right=0.72)
sp = "./model/backbone_comparison_bubble_v2.png"
plt.savefig(sp, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {sp}")
plt.close()
