"""
T型迷宫果蝇朝向分析 — 全T型范围 / 三区段 / 2×2条件空间
========================================================
朝向定义: body→head, tail→head, tail→body 三个方向向量的圆均值
2×2 条件空间:
  - Y 裁剪: 裁剪(Y∈[195,395]) / 不裁剪(全Y)
  - 犹豫区: 包含 / 排除(X∉[500,700])
对每个条件×分组, 分析朝向偏左/偏右比例、角度分布、位置-朝向联合分布
"""

import matplotlib
matplotlib.use('Agg')
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import re

# ===== 字体与样式 =====
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ===== 参数 =====
LIKELIHOOD_THRESHOLD = 0.6
DATA_DIR = Path(__file__).parent  # 脚本所在目录（可移植）
BAR_Y_MIN = 195
BAR_Y_MAX = 395
LEFT_MAX_X = 500
HESITATION_MAX_X = 700

# ----- 自动版本号 -----
existing_dirs = list(DATA_DIR.glob("orientation_analysis_v*"))
versions = []
for d in existing_dirs:
    m = re.search(r'orientation_analysis_v(\d+)$', d.name)
    if m:
        versions.append(int(m.group(1)))
AUTO_VERSION = max(versions) + 1 if versions else 1
OUTPUT_DIR = DATA_DIR / f"orientation_analysis_v{AUTO_VERSION}"
OUTPUT_DIR.mkdir(exist_ok=True)

# ===== 文件分组 (同 v5) =====
FILE_GROUPS = {
    '10% EtOH':   ['10%ETOH-air-WTDLC'],
    '5% EtOH':    ['5%ETOH-air-WTDLC', '5et_air_WT_2DLC'],
    '15% EtOH':   ['15%ETOH-air-WTDLC'],
    'Air (Ctrl)': ['AIR-air-WT2DLC', 'AIR-air-WTDLC',
                   'air_air_WT_2DLC', 'air_air_WT_4DLC',
                   'air_air_WTDLC'],
}
GROUP_ORDER = ['10% EtOH', '5% EtOH', '15% EtOH', 'Air (Ctrl)']
GROUP_COLORS = ['#2166AC', '#4DAF4A', '#FF7F00', '#999999']

# ===== 条件定义 =====
CONDITIONS = [
    {'key': 'uncropped_with_hes',  'label': 'No Y-crop\n+ Hesitation',
     'y_crop': False, 'exclude_hes': False},
    {'key': 'uncropped_no_hes',    'label': 'No Y-crop\n- Hesitation',
     'y_crop': False, 'exclude_hes': True},
    {'key': 'cropped_with_hes',    'label': 'Y-crop [195-395]\n+ Hesitation',
     'y_crop': True,  'exclude_hes': False},
    {'key': 'cropped_no_hes',      'label': 'Y-crop [195-395]\n- Hesitation',
     'y_crop': True,  'exclude_hes': True},
]

# ===== 辅助: 圆统计 (手动实现, 避免 scipy 依赖) =====
def circ_mean(angles):
    """圆均值, angles in radians [-pi, pi]"""
    if len(angles) == 0:
        return np.nan
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

def circ_r(angles):
    """平均合向量长度 R ∈ [0,1]"""
    if len(angles) == 0:
        return np.nan
    s = np.sum(np.sin(angles))
    c = np.sum(np.cos(angles))
    return np.sqrt(s**2 + c**2) / len(angles)

def circ_std(angles):
    """圆标准差 = sqrt(-2 * ln(R))"""
    R = circ_r(angles)
    if R <= 0 or np.isnan(R):
        return np.nan
    return np.sqrt(-2 * np.log(R))

def rayleigh_test(angles):
    """Rayleigh 均匀性检验, 返回 (Z, p_value)"""
    n = len(angles)
    if n < 2:
        return np.nan, np.nan
    R = circ_r(angles)
    Z = n * R**2
    # p-value 近似: p = exp(-Z) * (1 + (2Z - Z^2)/(4n) - ...)
    # 简化: p ≈ exp(-Z)  对 n > 50 足够; 对 n <= 50 用更精确的近似
    if n <= 50:
        p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n))
    else:
        p = np.exp(-Z)
    # 修正: 确保 p in [0,1]
    p = max(0.0, min(1.0, p))
    return Z, p

# ===== 数据加载 =====
def load_orientation_data(filepath):
    """
    从 H5 加载数据, 计算每帧的:
      - 位置 (cx, cy)
      - 朝向角度 (radians, -pi..pi, 0=右, pi/2=下, -pi/2=上)
      - 是否在横杠区域
      - 是否在犹豫区
    返回结构化 dict 列表
    """
    with h5py.File(filepath, 'r') as f:
        raw = f['df_with_missing']['table'][:]

    frames = []
    for row in raw:
        vals = row[1]
        hx, hy, hl = vals[0], vals[1], vals[2]
        bx, by, bl = vals[3], vals[4], vals[5]
        tx, ty, tl = vals[6], vals[7], vals[8]

        # --- 位置计算 (同 v5) ---
        head_ok = hl > LIKELIHOOD_THRESHOLD
        body_ok = bl > LIKELIHOOD_THRESHOLD
        tail_ok = tl > LIKELIHOOD_THRESHOLD

        if head_ok and body_ok and tail_ok:
            cx = ((hx + tx) / 2 + bx) / 2
            cy = ((hy + ty) / 2 + by) / 2
        elif body_ok:
            cx, cy = bx, by
        else:
            continue  # 位置不可用, 跳过

        # --- 朝向计算 ---
        vecs = []
        if head_ok and body_ok:
            vecs.append((hx - bx, hy - by))   # body → head
        if head_ok and tail_ok:
            vecs.append((hx - tx, hy - ty))   # tail → head
        if body_ok and tail_ok:
            vecs.append((bx - tx, by - ty))   # tail → body

        if len(vecs) == 0:
            orient_angle = np.nan
        else:
            angles = [np.arctan2(vy, vx) for vx, vy in vecs]
            orient_angle = circ_mean(angles)

        # --- 区域判定 ---
        in_bar = (BAR_Y_MIN <= cy <= BAR_Y_MAX)
        in_hesitation = (LEFT_MAX_X <= cx <= HESITATION_MAX_X)

        frames.append({
            'x': cx,
            'y': cy,
            'angle': orient_angle,
            'in_bar': in_bar,
            'in_hesitation': in_hesitation,
        })

    return frames


def filter_frames(frames, y_crop, exclude_hes):
    """根据条件过滤帧"""
    result = []
    for f in frames:
        if np.isnan(f['angle']):
            continue
        if y_crop and not f['in_bar']:
            continue
        if exclude_hes and f['in_hesitation']:
            continue
        result.append(f)
    return result


# ===== 加载所有数据 =====
print("=" * 70)
print(f"T型迷宫果蝇朝向分析 v{AUTO_VERSION}")
print(f"朝向 = 圆均值(body→head, tail→head, tail→body 中有数据者)")
print(f"2x2 条件空间: Y裁剪/不裁剪 × 含犹豫区/不含犹豫区")
print("=" * 70)

files = sorted(DATA_DIR.glob('*.h5'))
print(f"\n共 {len(files)} 个H5文件\n")

file_to_group = {}
for fpath in files:
    fname = fpath.stem
    for group_name, prefixes in FILE_GROUPS.items():
        for prefix in prefixes:
            if fname.startswith(prefix):
                file_to_group[fpath] = group_name
                break
        if fpath in file_to_group:
            break
    if fpath not in file_to_group:
        print(f"[!] 未匹配: {fname}")

# 加载
grouped_data = {}
for group_name in GROUP_ORDER:
    group_files = [fp for fp, gn in file_to_group.items() if gn == group_name]
    if not group_files:
        continue

    file_data = []
    for fp in group_files:
        frames = load_orientation_data(fp)
        n_total = len(frames)
        n_orient = sum(1 for f in frames if not np.isnan(f['angle']))
        file_data.append({
            'name': fp.stem,
            'frames': frames,
            'n_total': n_total,
            'n_orient': n_orient,
        })
        print(f"  {fp.stem[:50]}: {n_total:,} frames, "
              f"{n_orient:,} orient-valid ({100*n_orient/max(n_total,1):.1f}%)")

    grouped_data[group_name] = {'file_data': file_data}

# ===== 统计计算 =====
cond_stats = {}   # cond_stats[(cond_key, group_name)] = {...}
file_cond_stats = {}  # file_cond_stats[(cond_key, group_name, file_name)] = {...}

for cond in CONDITIONS:
    ck = cond['key']
    for group_name in GROUP_ORDER:
        if group_name not in grouped_data:
            continue

        all_angles = []
        per_file = []

        for fd in grouped_data[group_name]['file_data']:
            filtered = filter_frames(fd['frames'], cond['y_crop'], cond['exclude_hes'])
            angles = [f['angle'] for f in filtered]

            n_orient = len(angles)
            n_left = sum(1 for a in angles if np.cos(a) < 0)
            n_right = n_orient - n_left
            prop_right = n_right / n_orient if n_orient > 0 else np.nan
            cmean = circ_mean(angles)
            R = circ_r(angles)
            cstd = circ_std(angles)
            Z, p_rayl = rayleigh_test(angles)

            per_file.append({
                'name': fd['name'],
                'n_orient': n_orient,
                'n_left': n_left,
                'n_right': n_right,
                'prop_right': prop_right,
                'circ_mean': cmean,
                'R': R,
                'circ_std': cstd,
                'rayleigh_Z': Z,
                'rayleigh_p': p_rayl,
            })

            all_angles.extend(angles)

        # 组级统计 (聚合所有帧)
        n_tot = len(all_angles)
        n_l = sum(1 for a in all_angles if np.cos(a) < 0)
        n_r = n_tot - n_l
        prop_r = n_r / n_tot if n_tot > 0 else np.nan
        cmean_all = circ_mean(all_angles)
        R_all = circ_r(all_angles)
        cstd_all = circ_std(all_angles)
        Z_all, p_all = rayleigh_test(all_angles)

        # 个体级均值和 SEM
        props = [pf['prop_right'] for pf in per_file if not np.isnan(pf['prop_right'])]
        mean_prop = np.mean(props) if props else np.nan
        sem_prop = np.std(props, ddof=1)/np.sqrt(len(props)) if len(props) > 1 else 0.0

        cond_stats[(ck, group_name)] = {
            'n_orient': n_tot,
            'n_left': n_l,
            'n_right': n_r,
            'prop_right': prop_r,
            'mean_prop_right': mean_prop,
            'sem_prop_right': sem_prop,
            'circ_mean': cmean_all,
            'R': R_all,
            'circ_std': cstd_all,
            'rayleigh_Z': Z_all,
            'rayleigh_p': p_all,
            'n_files': len(per_file),
        }

        for pf in per_file:
            file_cond_stats[(ck, group_name, pf['name'])] = pf

# ===== 打印摘要 =====
print(f"\n{'='*70}")
print("朝向左/右偏好摘要 (Proportion Facing Right)")
print(f"{'条件':<32} {'10% EtOH':>10} {'5% EtOH':>10} {'15% EtOH':>10} {'Air(Ctrl)':>10}")
print(f"{'-'*32} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for cond in CONDITIONS:
    ck = cond['key']
    short_label = cond['label'].replace('\n', ' ')
    vals = []
    for gn in GROUP_ORDER:
        s = cond_stats.get((ck, gn), None)
        if s and not np.isnan(s['prop_right']):
            sem = s.get('sem_prop_right', 0)
            vals.append(f"{s['prop_right']:.3f}+/-{sem:.2f}")
        else:
            vals.append('N/A')
    print(f"{short_label:<32} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

# ===================================================================
# 图1: 极坐标朝向直方图 (4行×4列)
# ===================================================================
print(f"\n>>> 图1: 极坐标朝向直方图 ...")

fig1, axes1 = plt.subplots(4, 4, figsize=(22, 20),
                           subplot_kw={'projection': 'polar'})
n_bins = 24
bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

for ci, cond in enumerate(CONDITIONS):
    ck = cond['key']
    for gi, group_name in enumerate(GROUP_ORDER):
        ax = axes1[ci][gi]
        key = (ck, group_name)

        if group_name not in grouped_data or key not in cond_stats:
            ax.set_title(f'{group_name}\n(no data)', fontsize=7, pad=15)
            ax.set_visible(True)
            continue

        # 收集该条件+分组的所有朝向角度
        all_angles = []
        for fd in grouped_data[group_name]['file_data']:
            filtered = filter_frames(fd['frames'], cond['y_crop'], cond['exclude_hes'])
            all_angles.extend([f['angle'] for f in filtered])

        if len(all_angles) == 0:
            ax.set_title(f'{group_name}\n(no frames)', fontsize=7, pad=15)
            continue

        counts, _ = np.histogram(all_angles, bins=bin_edges)
        width = 2 * np.pi / n_bins

        # 左右半圆着色
        for bi in range(n_bins):
            mid = bin_centers[bi]
            color = '#D62728' if np.cos(mid) < 0 else '#1F77B4'  # red=left, blue=right
            ax.bar(bin_centers[bi], counts[bi], width=width,
                   color=color, alpha=0.75, edgecolor='white', linewidth=0.3)

        # 标注
        s = cond_stats[key]
        ax.set_title(f'{group_name}\nR:{s["R"]:.2f} | PropR:{s["prop_right"]:.3f}',
                     fontsize=7, fontweight='bold', pad=15)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)

# 行标题 (条件)
row_labels = [c['label'] for c in CONDITIONS]
for ci, label in enumerate(row_labels):
    axes1[ci][0].set_ylabel(label, fontsize=8, fontweight='bold',
                            labelpad=30, rotation=0, ha='right', va='center')

fig1.suptitle(f'Step 1: Orientation Polar Histograms (24 bins) — v{AUTO_VERSION}\n'
              'Blue=Right-facing (cos>0) | Red=Left-facing (cos<0)\n'
              'Rows: 4 conditions | Columns: 4 ethanol groups',
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig1.savefig(OUTPUT_DIR / 'step1_polar_orientation.png', dpi=300,
             bbox_inches='tight', facecolor='white')
print("  [OK] step1_polar_orientation.png")

# ===================================================================
# 图2: 左/右朝向偏好柱状图
# ===================================================================
print(f"\n>>> 图2: 左/右朝向偏好柱状图 ...")

fig2, ax2 = plt.subplots(figsize=(16, 8))
n_groups = len(GROUP_ORDER)
n_conds = len(CONDITIONS)
bar_width = 0.18
x_pos = np.arange(n_groups)

cond_colors = ['#1B3A5C', '#4A90D9', '#C44536', '#E8927C']
cond_hatches = ['', '//', '', '//']

for ci, cond in enumerate(CONDITIONS):
    ck = cond['key']
    offsets = (ci - n_conds/2 + 0.5) * bar_width
    props = []
    errs = []
    for gn in GROUP_ORDER:
        s = cond_stats.get((ck, gn), None)
        if s and not np.isnan(s['prop_right']):
            props.append(s['prop_right'])
            errs.append(s.get('sem_prop_right', 0))
        else:
            props.append(0)
            errs.append(0)

    bars = ax2.bar(x_pos + offsets, props, bar_width,
                   color=cond_colors[ci], edgecolor='white', linewidth=0.6,
                   hatch=cond_hatches[ci], alpha=0.9,
                   yerr=errs, capsize=4, error_kw={'linewidth': 1},
                   label=cond['label'].replace('\n', ' '))

    # 数值标注
    for bi, (xi, p) in enumerate(zip(x_pos + offsets, props)):
        if p > 0.02:
            ax2.text(xi, p + max(errs[bi], 0.01) + 0.015, f'{p:.3f}',
                     ha='center', fontsize=7, fontweight='bold', rotation=90)

ax2.axhline(y=0.5, color='gray', linewidth=1, linestyle=':', alpha=0.7, zorder=0)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(GROUP_ORDER, fontsize=11)
ax2.set_ylabel('Proportion Facing Right', fontsize=12)
ax2.set_title(f'Step 2: Left/Right Facing Bias — v{AUTO_VERSION}\n'
              'Bar groups = 4 conditions | Higher = more right-facing',
              fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.08)
ax2.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.8)
ax2.grid(axis='y', alpha=0.2)

# 标注单文件组
for gi, gn in enumerate(GROUP_ORDER):
    nf = len(grouped_data.get(gn, {}).get('file_data', []))
    if nf == 1:
        ax2.text(gi, 0.01, '(n=1)', ha='center', fontsize=7, color='gray')

plt.tight_layout()
fig2.savefig(OUTPUT_DIR / 'step2_facing_bias.png', dpi=300,
             bbox_inches='tight', facecolor='white')
print("  [OK] step2_facing_bias.png")

# ===================================================================
# 图3 & 图4: 位置-朝向 2D 直方图
# ===================================================================
def plot_position_angle_figure(y_crop_flag, fig_id, fig_title_suffix):
    """绘制位置-朝向 2D 直方图: 2行(hes)×4列(组)"""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    hes_modes = [False, True]  # [含犹豫, 不含犹豫]
    hes_labels = ['With Hesitation', 'Excluding Hesitation']

    for hi, (exclude_hes, hes_label) in enumerate(zip(hes_modes, hes_labels)):
        for gi, group_name in enumerate(GROUP_ORDER):
            ax = axes[hi][gi]
            if group_name not in grouped_data:
                ax.set_visible(False)
                continue

            all_angles = []
            all_x = []
            for fd in grouped_data[group_name]['file_data']:
                filtered = filter_frames(fd['frames'], y_crop_flag, exclude_hes)
                all_angles.extend([f['angle'] for f in filtered])
                all_x.extend([f['x'] for f in filtered])

            if len(all_angles) == 0:
                ax.text(600, 0, 'No data', ha='center', va='center',
                        fontsize=11, color='gray', transform=ax.transData)
                ax.set_xlim(0, 1200)
                ax.set_ylim(-np.pi, np.pi)
                ax.set_title(f'{group_name}\n{hes_label}', fontsize=8)
                continue

            h2d, xedges, yedges = np.histogram2d(
                all_x, all_angles,
                bins=[40, 36],
                range=[[0, 1200], [-np.pi, np.pi]]
            )
            h2d_masked = np.ma.masked_where(h2d == 0, h2d)

            im = ax.pcolormesh(xedges, yedges, h2d_masked.T,
                               cmap='inferno',
                               norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(h2d.max(), 10)),
                               rasterized=True)

            # 区段分界线
            ax.axvline(x=LEFT_MAX_X, color='#FFD700', linewidth=1.5, linestyle='-', alpha=0.7)
            ax.axvline(x=HESITATION_MAX_X, color='#FF4500', linewidth=1.5, linestyle='-', alpha=0.7)

            # 0 参考线 (朝向正右)
            ax.axhline(y=0, color='white', linewidth=0.8, linestyle=':', alpha=0.4)

            ax.set_xlim(0, 1200)
            ax.set_ylim(-np.pi, np.pi)
            ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_yticklabels(['-π\n(left)', '-π/2\n(up)', '0\n(right)', 'π/2\n(down)', 'π\n(left)'],
                               fontsize=6)

            n_frames = len(all_angles)
            n_r = sum(1 for a in all_angles if np.cos(a) > 0)
            ax.set_title(f'{group_name}\n{hes_label} | N={n_frames:,} PropR={n_r/max(n_frames,1):.2f}',
                         fontsize=8, fontweight='bold')

            if gi == 3:
                plt.colorbar(im, ax=ax, label='Frames (log)', shrink=0.7)

    ylabel_text = 'Y-crop [195-395]' if y_crop_flag else 'Full Y range'
    fig.suptitle(f'Step {fig_id}: Position-Orientation 2D Histogram ({ylabel_text}) — v{AUTO_VERSION}\n'
                 f'Top row: with hesitation | Bottom row: excluding hesitation\n'
                 f'Columns: 4 ethanol groups | Y-axis: orientation angle (-π to π)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = f'step{fig_id}_position_angle_{"cropped" if y_crop_flag else "uncropped"}.png'
    fig.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] {fname}")

plot_position_angle_figure(False, 3, 'Full Y')
plot_position_angle_figure(True, 4, 'Y-cropped')

# ===================================================================
# 统计报告
# ===================================================================
print(f"\n>>> 统计报告 ...")
with open(OUTPUT_DIR / 'statistics_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 90 + "\n")
    f.write(f"T型迷宫果蝇朝向分析统计报告 v{AUTO_VERSION}\n")
    f.write(f"日期: 2026/07/09\n")
    f.write(f"朝向定义: 圆均值(body→head, tail→head, tail→body 中有数据者)\n")
    f.write(f"左/右分类: cos(angle)>0 = facing right, cos(angle)<0 = facing left\n")
    f.write("=" * 90 + "\n\n")

    f.write("[数据加载摘要]\n")
    f.write("-" * 80 + "\n")
    for gn in GROUP_ORDER:
        if gn not in grouped_data:
            continue
        for fd in grouped_data[gn]['file_data']:
            f.write(f"  {gn:<16} {fd['name'][:45]:<45} "
                    f"total={fd['n_total']:,}  orient-valid={fd['n_orient']:,} "
                    f"({100*fd['n_orient']/max(fd['n_total'],1):.1f}%)\n")

    f.write(f"\n\n[组级统计: 4条件 × 4分组]\n")
    f.write("=" * 90 + "\n")
    header = (f"{'条件':<28} {'分组':<12} {'N_frames':>9} {'N_left':>7} {'N_right':>7} "
              f"{'PropR':>7} {'PropR_SEM':>9} {'CircMean':>8} {'R':>6} {'CircStd':>8} {'Rayl_Z':>8} {'Rayl_p':>8}")
    f.write(header + "\n")
    f.write("-" * 90 + "\n")

    for cond in CONDITIONS:
        ck = cond['key']
        for gn in GROUP_ORDER:
            s = cond_stats.get((ck, gn), None)
            if s is None:
                continue
            mean_str = f"{np.degrees(s['circ_mean']):>7.1f}°" if not np.isnan(s['circ_mean']) else "    N/A"
            f.write(f"{cond['label'].replace(chr(10),' '):<28} {gn:<12} "
                    f"{s['n_orient']:>9,} {s['n_left']:>7,} {s['n_right']:>7,} "
                    f"{s['prop_right']:>7.3f} {s['sem_prop_right']:>9.4f} "
                    f"{mean_str} {s['R']:>6.3f} {s['circ_std']:>8.4f} "
                    f"{s['rayleigh_Z']:>8.2f} {s['rayleigh_p']:>8.4f}\n")
        f.write("\n")

    f.write(f"\n\n[各文件逐条件统计]\n")
    f.write("=" * 90 + "\n")
    for gn in GROUP_ORDER:
        if gn not in grouped_data:
            continue
        for fd in grouped_data[gn]['file_data']:
            fname = fd['name']
            f.write(f"\n  文件: {fname}  [分组: {gn}]\n")
            f.write(f"  {'条件':<30} {'N':>7} {'N_L':>7} {'N_R':>7} {'PropR':>7} "
                    f"{'CMean':>7} {'R':>6} {'CStd':>7} {'Rayl_p':>8}\n")
            f.write(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*8}\n")
            for cond in CONDITIONS:
                ck = cond['key']
                pf = file_cond_stats.get((ck, gn, fname), None)
                if pf is None:
                    continue
                mean_str = f"{np.degrees(pf['circ_mean']):>6.1f}°" if not np.isnan(pf['circ_mean']) else "   N/A"
                f.write(f"  {cond['label'].replace(chr(10),' '):<30} "
                        f"{pf['n_orient']:>7,} {pf['n_left']:>7,} {pf['n_right']:>7,} "
                        f"{pf['prop_right']:>7.3f} {mean_str} {pf['R']:>6.3f} "
                        f"{pf['circ_std']:>7.4f} {pf['rayleigh_p']:>8.4f}\n")

print("  [OK] statistics_report.txt")

# ===== Summary =====
print(f"\n{'='*70}")
print(f"v{AUTO_VERSION} orientation analysis complete!")
print(f"Output files in {OUTPUT_DIR.name}/:")
for fpath in sorted(OUTPUT_DIR.iterdir()):
    size_kb = fpath.stat().st_size / 1024
    print(f"  {fpath.name} ({size_kb:.0f} KB)")
print(f"{'='*70}")
