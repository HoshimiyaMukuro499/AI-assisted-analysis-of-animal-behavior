"""
绘制6只果蝇在T型二元选择迷宫中的运动轨迹
数据来自DLC（DeepLabCut）追踪，包含head/body/tail三个部位
位置计算：((头+尾)/2 + 腹)/2，若部位不全则使用腹部代替
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，无需GUI
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import os

# ===== 设置中文字体 =====
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ===== 参数设置 =====
LIKELIHOOD_THRESHOLD = 0.6  # 置信度阈值，低于此值认为该部位缺失

# 获取文件列表
data_dir = Path(__file__).parent  # 脚本所在目录（可移植）
files = sorted([f for f in data_dir.glob("*.h5")])
print(f"共找到 {len(files)} 个文件")

# ===== 读取所有数据 =====
all_data = {}  # {filename: {frame_idx: [], positions: [], timestamps: []}}

for fpath in files:
    fname = fpath.stem  # 不含扩展名的文件名

    with h5py.File(fpath, 'r') as f:
        table = f['df_with_missing']['table']
        data = table[:]
        n_frames = len(data)

    # 提取数据
    positions = []  # 每帧的 (x, y)
    frame_indices = []

    for row in data:
        frame_idx = row[0]
        vals = row[1]  # shape (9,): [head_x, head_y, head_lik, body_x, body_y, body_lik, tail_x, tail_y, tail_lik]

        head_x, head_y, head_lik = vals[0], vals[1], vals[2]
        body_x, body_y, body_lik = vals[3], vals[4], vals[5]
        tail_x, tail_y, tail_lik = vals[6], vals[7], vals[8]

        # 判断三个部位是否都有有效数据
        head_valid = head_lik > LIKELIHOOD_THRESHOLD
        body_valid = body_lik > LIKELIHOOD_THRESHOLD
        tail_valid = tail_lik > LIKELIHOOD_THRESHOLD

        if head_valid and body_valid and tail_valid:
            # 所有部位有效：((head+tail)/2 + body)/2
            centroid_x = ((head_x + tail_x) / 2 + body_x) / 2
            centroid_y = ((head_y + tail_y) / 2 + body_y) / 2
        elif body_valid:
            # 部位不全，使用腹部
            centroid_x = body_x
            centroid_y = body_y
        else:
            # 连腹部的置信度都不够，跳过这一帧
            continue

        positions.append([centroid_x, centroid_y])
        frame_indices.append(frame_idx)

    positions = np.array(positions)
    frame_indices = np.array(frame_indices)

    all_data[fname] = {
        'positions': positions,
        'frame_indices': frame_indices,
        'n_frames': n_frames,
        'n_valid': len(positions)
    }

    # 检查缺失情况
    n_missing = n_frames - len(positions)
    print(f"\n{fname}:")
    print(f"  总帧数: {n_frames}, 有效帧: {len(positions)}, 缺失帧: {n_missing} ({100*n_missing/n_frames:.1f}%)")

    # 检查各部位逐帧的缺失情况
    with h5py.File(fpath, 'r') as f:
        raw = f['df_with_missing']['table'][:]
    head_ok = np.sum(raw['values_block_0'][:, 2] > LIKELIHOOD_THRESHOLD)
    body_ok = np.sum(raw['values_block_0'][:, 5] > LIKELIHOOD_THRESHOLD)
    tail_ok = np.sum(raw['values_block_0'][:, 8] > LIKELIHOOD_THRESHOLD)
    print(f"  head有效帧: {head_ok}/{n_frames}, body有效帧: {body_ok}/{n_frames}, tail有效帧: {tail_ok}/{n_frames}")

# ===== 计算统一坐标范围 =====
all_x = np.concatenate([d['positions'][:, 0] for d in all_data.values()])
all_y = np.concatenate([d['positions'][:, 1] for d in all_data.values()])

x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()

# 添加5%的边距
x_margin = (x_max - x_min) * 0.05
y_margin = (y_max - y_min) * 0.05

x_lim = (x_min - x_margin, x_max + x_margin)
y_lim = (y_min - y_margin, y_max + y_margin)

print(f"\n===== 统一坐标范围 =====")
print(f"X: [{x_lim[0]:.1f}, {x_lim[1]:.1f}]")
print(f"Y: [{y_lim[0]:.1f}, {y_lim[1]:.1f}]")

# ===== 绘图 =====
n_files = len(all_data)
n_cols = 4
n_rows = 3  # 12个文件用4x3网格

# 科研配色：使用感知均匀的配色方案
# 主轨迹使用 viridis 色图按时间着色
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 16))
axes = axes.flatten()

# 自动缩写文件名用于标题
def shorten_name(fname):
    """从长文件名中提取有意义的短名"""
    # 移除固定后缀
    s = fname.replace('DLC_Resnet50_T_trapJul8shuffle1_snapshot_best-30', '')
    s = s.replace('_Resnet50_T_trapJul8shuffle1_snapshot_best-30', '')
    # 统一缩写
    s = s.replace('ETOH', 'EtOH')
    s = s.replace('air_air', 'air-air')
    s = s.replace('airw_air', 'airw-air')
    s = s.replace('5et_air', '5Et-air')
    s = s.replace('AIR-air', 'AIR-air')
    return s

short_names = {fname: shorten_name(fname) for fname in all_data.keys()}

for idx, (fname, dat) in enumerate(all_data.items()):
    ax = axes[idx]
    positions = dat['positions']
    frames = dat['frame_indices']

    # 标准化时间用于着色 (0到1)
    time_norm = (frames - frames.min()) / (frames.max() - frames.min())

    # 绘制轨迹，按时间着色
    # 使用散点图以颜色表示时间
    # 为减少点数（性能），如果点数太多则采样
    n_pts = len(positions)
    if n_pts > 5000:
        step = n_pts // 5000
        plot_pos = positions[::step]
        plot_time = time_norm[::step]
    else:
        plot_pos = positions
        plot_time = time_norm

    # 使用 viridis 色图：紫色=早期，黄色=晚期
    scatter = ax.scatter(
        plot_pos[:, 0], plot_pos[:, 1],
        c=plot_time, cmap='viridis',
        s=3, alpha=0.6, linewidths=0, rasterized=True
    )

    # 标记起点和终点
    ax.scatter(positions[0, 0], positions[0, 1],
               c='#0072B2', s=100, marker='o',
               edgecolors='white', linewidths=1.5,
               zorder=5, label=f'Start (t={frames[0]})')
    ax.scatter(positions[-1, 0], positions[-1, 1],
               c='#D55E00', s=100, marker='s',
               edgecolors='white', linewidths=1.5,
               zorder=5, label=f'End (t={frames[-1]})')

    # 设置标签
    short_name = short_names.get(fname, fname[:30])
    ax.set_title(f'{short_name}\n({dat["n_valid"]}/{dat["n_frames"]} frames)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=9)
    ax.set_ylabel('Y (pixels)', fontsize=9)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(labelsize=8)

# 删除多余的子图
for idx in range(n_files, n_rows * n_cols):
    axes[idx].set_visible(False)

# 添加总标题
fig.suptitle('Drosophila Movement Trajectories in T-Maze\n'
             f'(Position = ((Head+Tail)/2 + Body)/2, Likelihood Threshold = {LIKELIHOOD_THRESHOLD})\n'
             'Color: Viridis colormap (Purple=Early → Yellow=Late)',
             fontsize=14, fontweight='bold', y=1.01)

# 添加统一的 colorbar
cbar_ax = fig.add_axes([0.94, 0.08, 0.012, 0.82])
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label('Normalized Time', fontsize=10)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 0.93, 0.96])

# 保存图片
output_path = data_dir / 'trajectories.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n===== 图片已保存至: {output_path} =====")

# 同时保存一份高分辨率版本
output_path_hd = data_dir / 'trajectories_hd.png'
fig.savefig(output_path_hd, dpi=600, bbox_inches='tight', facecolor='white')
print(f"高清版已保存至: {output_path_hd}")

print("完成！")
