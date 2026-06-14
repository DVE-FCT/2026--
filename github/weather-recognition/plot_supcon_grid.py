"""
SupCon 网格搜索 3D 图 + 2D 投影热力图
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import griddata

# 中文字体
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
try:
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.sans-serif'] = ['SimHei']
except Exception:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

# 读取数据（第二轮全量数据）
df = pd.read_csv("./model/supcon_grid_search.csv")
# df 包含全部 37 组 (M41-M77)

lam_vals = sorted(df['lambda'].unique())
tau_vals = sorted(df['tau'].unique())
L, T = np.meshgrid(lam_vals, tau_vals)

# 插值网格
pts = np.column_stack([df['lambda'], df['tau']])
f1_grid = griddata(pts, df['Macro_F1'], (L, T), method='cubic')
acc_grid = griddata(pts, df['Acc'], (L, T), method='cubic')

# M40 baseline
M40_F1 = 0.9434

fig = plt.figure(figsize=(18, 8))

# ---- 3D 图 ----
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(L, T, f1_grid, cmap='RdYlGn', alpha=0.85, edgecolor='none')
ax.scatter(df['lambda'], df['tau'], df['Macro_F1'], c=df['Macro_F1'],
           cmap='RdYlGn', s=15, alpha=0.8, edgecolors='#333', linewidths=0.3, depthshade=True)
# M40 baseline plane
ax.plot_surface(L, T, np.full_like(L, M40_F1), color='green', alpha=0.12)
ax.set_xlabel('λ', fontsize=11)
ax.set_ylabel('τ', fontsize=11)
ax.set_zlabel('Macro F1', fontsize=11)
ax.set_title(f'SupCon 网格搜索 3D (M40={M40_F1:.4f})', fontsize=13, fontweight='bold')
ax.view_init(25, -60)
fig.colorbar(surf, ax=ax, shrink=0.5, label='Macro F1')

# ---- 2D 投影 + 等高线 ----
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(L, T, f1_grid, levels=15, cmap='RdYlGn')
ax2.scatter(df['lambda'], df['tau'], c=df['Macro_F1'], s=40, edgecolors='#333',
            linewidths=0.8, cmap='RdYlGn', zorder=3)
# 在每个点上标 F1 值
for _, row in df.iterrows():
    ax2.annotate(f"{row['Macro_F1']:.3f}", (row['lambda'], row['tau']),
                 fontsize=7, ha='center', va='bottom', alpha=0.7)

# M40 等高线（0.94 远高于所有 SupCon 结果）
ax2.contour(L, T, f1_grid, levels=[0.85, 0.86, 0.87],
            colors='gray', linestyles='--', linewidths=1, alpha=0.5)

ax2.set_xlabel('λ', fontsize=12, fontweight='bold')
ax2.set_ylabel('τ', fontsize=12, fontweight='bold')
ax2.set_title(f'SupCon F1 热力图 (M40 基线={M40_F1:.4f}, 全高 {M40_F1-f1_grid.min():.4f})',
              fontsize=13, fontweight='bold')
fig.colorbar(contour, ax=ax2, shrink=0.8, label='Macro F1')

plt.tight_layout()
sp = "./model/supcon_grid_3d_heatmap.png"
plt.savefig(sp, dpi=800, bbox_inches='tight', facecolor='white')
print(f"Saved: {sp}")
plt.close()
