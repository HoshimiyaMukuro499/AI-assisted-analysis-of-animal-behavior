"""
T型迷宫行为分析 — 全T型范围 / 三区段分割 / 无Y裁剪
=====================================================
- 左臂: X < 500  |  犹豫区: 500 <= X <= 700  |  右臂: X > 700
- 包含全部Y范围（竖杆 + 横杠），不再裁剪
- Ratio (1): 左:犹豫:右 三者比例
- Ratio (2): 左/(左+右) — 排除犹豫区
- 版本: 自动递增 (扫描 analysis_output_v* 目录取 max+1)
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
DATA_DIR = Path(r"D:\文件\大一下\AI Eth 实验\Ttrap")

# ----- 自动版本号：扫描已有目录，递增 -----
existing_dirs = list(DATA_DIR.glob("analysis_output_v*"))
versions = []
for d in existing_dirs:
    m = re.search(r'analysis_output_v(\d+)$', d.name)
    if m:
        versions.append(int(m.group(1)))
AUTO_VERSION = max(versions) + 1 if versions else 1
OUTPUT_DIR = DATA_DIR / f"analysis_output_v{AUTO_VERSION}"
OUTPUT_DIR.mkdir(exist_ok=True)

# ===== 三区段定义（全T型范围）=====
LEFT_MAX_X = 500
HESITATION_MAX_X = 700

# ===== 文件分组 =====
FILE_GROUPS = {
    '10% EtOH':   ['10%ETOH-air-WTDLC'],
    '5% EtOH':    ['5%ETOH-air-WTDLC', '5et_air_WT_2DLC', '5%ETOH-air-WT -3DLC'],
    '15% EtOH':   ['15%ETOH-air-WTDLC'],
    '20% EtOH':   ['20%EtOH_air_WT'],
    '25% EtOH':   ['25%EtOH_air_WT'],
    'Air (Ctrl)': ['AIR-air-WT2DLC', 'AIR-air-WTDLC',
                   'air_air_WT_2DLC', 'air_air_WT_4DLC',
                   'air_air_WTDLC'],
}

GROUP_ORDER = ['Air (Ctrl)', '5% EtOH', '10% EtOH', '15% EtOH', '20% EtOH', '25% EtOH']


# ===== X坐标偏移（部分视频画面偏移的校正）=====
X_OFFSETS = {
    '20%EtOH_air_WT': -100,     # 20% EtOH 视频左移 100 像素校正
    '25%EtOH_air_WT': -100,     # 25% EtOH 视频左移 100 像素校正
    '5%ETOH-air-WT -3DLC': -100,  # 新增 5% EtOH 视频左移 100 像素校正
}


def load_and_calculate_positions(filepath):
    # 根据文件前缀确定 X 偏移量
    fname = filepath.stem
    x_offset = 0
    for prefix, offset in X_OFFSETS.items():
        if fname.startswith(prefix):
            x_offset = offset
            break

    with h5py.File(filepath, 'r') as f:
        raw = f['df_with_missing']['table'][:]

    positions = []
    for row in raw:
        vals = row[1]
        hx, hy, hl = vals[0], vals[1], vals[2]
        bx, by, bl = vals[3], vals[4], vals[5]
        tx, ty, tl = vals[6], vals[7], vals[8]

        head_ok = hl > LIKELIHOOD_THRESHOLD
        body_ok = bl > LIKELIHOOD_THRESHOLD
        tail_ok = tl > LIKELIHOOD_THRESHOLD

        if head_ok and body_ok and tail_ok:
            cx = ((hx + tx) / 2 + bx) / 2 + x_offset
            cy = ((hy + ty) / 2 + by) / 2
        elif body_ok:
            cx = bx + x_offset
            cy = by
        else:
            continue
        positions.append([cx, cy])

    return np.array(positions)


def classify_zone(x):
    """三区段分类（全T型，无Y限制）"""
    if x < LEFT_MAX_X:
        return 'left'
    elif x <= HESITATION_MAX_X:
        return 'hesitation'
    else:
        return 'right'


# ===== 加载并分组 =====
print("=" * 70)
print(f"T型迷宫行为分析 v{AUTO_VERSION} — 全T型 / 三区段 / 无Y裁剪")
print(f"左臂: X < {LEFT_MAX_X}  |  犹豫区: [{LEFT_MAX_X}, {HESITATION_MAX_X}]  |  右臂: X > {HESITATION_MAX_X}")
print(f"范围: 全部Y (含竖杆+横杠)")
print("=" * 70)

files = sorted(DATA_DIR.glob('*.h5'))
print(f"\n共 {len(files)} 个H5文件")

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

# 按组聚合
grouped_data = {}
for group_name in GROUP_ORDER:
    group_files = [fp for fp, gn in file_to_group.items() if gn == group_name]
    if not group_files:
        continue

    file_data = []
    for fp in group_files:
        pos = load_and_calculate_positions(fp)
        file_data.append({
            'name': fp.stem,
            'n_frames': len(pos),
            'positions': pos,
        })

    grouped_data[group_name] = {'file_data': file_data}

    total = sum(fd['n_frames'] for fd in file_data)
    print(f"\n[分组] {group_name} ({len(group_files)} files, {total:,} frames)")
    for fd in file_data:
        left_n = sum(1 for x, _ in fd['positions'] if classify_zone(x) == 'left')
        hesi_n = sum(1 for x, _ in fd['positions'] if classify_zone(x) == 'hesitation')
        right_n = sum(1 for x, _ in fd['positions'] if classify_zone(x) == 'right')
        total_n = fd['n_frames']
        print(f"    {fd['name'][:55]}: {total_n:,} frm | "
              f"L:{left_n:,} ({100*left_n/total_n:.0f}%) | H:{hesi_n:,} ({100*hesi_n/total_n:.0f}%) | R:{right_n:,} ({100*right_n/total_n:.0f}%)")

# ===== 统计 =====
print(f"\n{'='*70}")
print("三区段统计（全T型，无Y裁剪）")
print(f"{'分组':<16} {'左臂':>9} {'犹豫区':>9} {'右臂':>9} {'总计':>9} | {'左:犹豫:右 比例':>20} | {'左/(左+右)':>10}")
print(f"{'-'*16} {'-'*9} {'-'*9} {'-'*9} {'-'*9} | {'-'*20} | {'-'*10}")

group_stats = {}
all_file_stats = []

for group_name in GROUP_ORDER:
    if group_name not in grouped_data:
        continue

    left_total = 0
    hesi_total = 0
    right_total = 0

    for fd in grouped_data[group_name]['file_data']:
        left_f = sum(1 for x, _ in fd['positions'] if classify_zone(x) == 'left')
        hesi_f = sum(1 for x, _ in fd['positions'] if classify_zone(x) == 'hesitation')
        right_f = sum(1 for x, _ in fd['positions'] if classify_zone(x) == 'right')

        left_total += left_f
        hesi_total += hesi_f
        right_total += right_f

        arms_lr = left_f + right_f
        ratio_lr = left_f / arms_lr if arms_lr > 0 else 0
        all_file_stats.append({
            'group': group_name,
            'name': fd['name'],
            'left': left_f, 'hesitation': hesi_f, 'right': right_f,
            'total': fd['n_frames'],
            'ratio_left_vs_right': ratio_lr,
        })

    grand_total = left_total + hesi_total + right_total
    arms_lr_total = left_total + right_total
    ratio_lr = left_total / arms_lr_total if arms_lr_total > 0 else 0

    if grand_total > 0:
        pct_left = left_total / grand_total
        pct_hesi = hesi_total / grand_total
        pct_right = right_total / grand_total
    else:
        pct_left = pct_hesi = pct_right = 0

    group_stats[group_name] = {
        'left': left_total, 'hesitation': hesi_total, 'right': right_total,
        'total': grand_total,
        'ratio_left_vs_right': ratio_lr,
        'pct_left': pct_left, 'pct_hesi': pct_hesi, 'pct_right': pct_right,
    }

    ratio_str = f"{pct_left:.1%} : {pct_hesi:.1%} : {pct_right:.1%}"
    print(f"{group_name:<16} {left_total:>9,} {hesi_total:>9,} {right_total:>9,} {grand_total:>9,} | {ratio_str:>20} | {ratio_lr:>10.4f}")

# ===== SEM =====
for group_name in GROUP_ORDER:
    if group_name not in grouped_data:
        continue
    fd_list = grouped_data[group_name]['file_data']
    if len(fd_list) > 1:
        ratios_lr = []
        ratios_l = []
        ratios_h = []
        ratios_r = []
        for fd in fd_list:
            l = sum(1 for x, _ in fd['positions'] if classify_zone(x) == 'left')
            h = sum(1 for x, _ in fd['positions'] if classify_zone(x) == 'hesitation')
            r = sum(1 for x, _ in fd['positions'] if classify_zone(x) == 'right')
            t = l + h + r
            lr = l + r
            ratios_lr.append(l / lr if lr > 0 else 0)
            if t > 0:
                ratios_l.append(l / t)
                ratios_h.append(h / t)
                ratios_r.append(r / t)

        group_stats[group_name]['mean_ratio_lr'] = np.mean(ratios_lr)
        group_stats[group_name]['sem_ratio_lr'] = np.std(ratios_lr, ddof=1) / np.sqrt(len(ratios_lr))
        group_stats[group_name]['mean_pct_l'] = np.mean(ratios_l)
        group_stats[group_name]['sem_pct_l'] = np.std(ratios_l, ddof=1) / np.sqrt(len(ratios_l))
        group_stats[group_name]['mean_pct_h'] = np.mean(ratios_h)
        group_stats[group_name]['sem_pct_h'] = np.std(ratios_h, ddof=1) / np.sqrt(len(ratios_h))
        group_stats[group_name]['mean_pct_r'] = np.mean(ratios_r)
        group_stats[group_name]['sem_pct_r'] = np.std(ratios_r, ddof=1) / np.sqrt(len(ratios_r))
    else:
        group_stats[group_name]['mean_ratio_lr'] = group_stats[group_name]['ratio_left_vs_right']
        group_stats[group_name]['sem_ratio_lr'] = 0
        group_stats[group_name]['mean_pct_l'] = group_stats[group_name]['pct_left']
        group_stats[group_name]['mean_pct_h'] = group_stats[group_name]['pct_hesi']
        group_stats[group_name]['mean_pct_r'] = group_stats[group_name]['pct_right']
        group_stats[group_name]['sem_pct_l'] = 0
        group_stats[group_name]['sem_pct_h'] = 0
        group_stats[group_name]['sem_pct_r'] = 0

# ===================================================================
# 图1: 全T型三区段热图
# ===================================================================
print(f"\n>>> 图1: 全T型三区段热图 ...")

fig1, axes1 = plt.subplots(2, 3, figsize=(22, 14))
axes1 = axes1.flatten()

for idx, group_name in enumerate(GROUP_ORDER):
    ax = axes1[idx]
    if group_name not in grouped_data:
        ax.set_visible(False)
        continue

    all_pos = np.vstack([fd['positions'] for fd in grouped_data[group_name]['file_data']])

    h, xedges, yedges = np.histogram2d(
        all_pos[:, 0], all_pos[:, 1],
        bins=[80, 60],
        range=[[0, 1200], [0, 1000]]
    )
    h_masked = np.ma.masked_where(h == 0, h)

    im = ax.pcolormesh(xedges, yedges, h_masked.T,
                       cmap='inferno',
                       norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(h.max(), 100)),
                       rasterized=True)

    # 三区段分界线
    ax.axvline(x=LEFT_MAX_X, color='#FFD700', linewidth=2, linestyle='-', alpha=0.9,
               label=f'X={LEFT_MAX_X}')
    ax.axvline(x=HESITATION_MAX_X, color='#FF4500', linewidth=2, linestyle='-', alpha=0.9,
               label=f'X={HESITATION_MAX_X}')
    ax.axvspan(LEFT_MAX_X, HESITATION_MAX_X, alpha=0.10, color='#FFD700')

    # 标注
    ax.text(LEFT_MAX_X / 2, 60, 'LEFT\nX<500', color='white',
            fontsize=11, fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#377EB8', alpha=0.7))
    ax.text(600, 60, 'HESITATION\n500-700', color='black',
            fontsize=9, fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#FFD700', alpha=0.7))
    ax.text(950, 60, 'RIGHT\nX>700', color='white',
            fontsize=11, fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#E41A1C', alpha=0.7))

    stats = group_stats[group_name]
    ax.set_title(f'{group_name}  ({stats["total"]:,} total frames)\n'
                 f'L:{stats["pct_left"]:.1%} | H:{stats["pct_hesi"]:.1%} | R:{stats["pct_right"]:.1%}  '
                 f'|  L/(L+R):{stats["ratio_left_vs_right"]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=9)
    ax.set_ylabel('Y (pixels)', fontsize=9)
    ax.set_xlim(0, 1200)
    ax.set_ylim(950, 0)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.8)
    plt.colorbar(im, ax=ax, label='Frame count (log)', shrink=0.82)

# 隐藏多余子图
for extra_idx in range(len(GROUP_ORDER), len(axes1)):
    axes1[extra_idx].set_visible(False)

fig1.suptitle(f'Step 1: Three-Zone Heatmaps (Full T-Maze, No Y-Filter) — v{AUTO_VERSION}\n'
              f'Left: X<{LEFT_MAX_X} | Hesitation: [{LEFT_MAX_X},{HESITATION_MAX_X}] | Right: X>{HESITATION_MAX_X}',
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig1.savefig(OUTPUT_DIR / 'step1_heatmaps.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  [OK] step1_heatmaps.png")

# ===================================================================
# 图2: 轨迹+三区段分割线 (审查用)
# ===================================================================
print(f"\n>>> 图2: 轨迹+三区段分割 (审查用) ...")

fig2, axes2 = plt.subplots(2, 3, figsize=(22, 14))
axes2 = axes2.flatten()
colors_file = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

for idx, group_name in enumerate(GROUP_ORDER):
    ax = axes2[idx]
    if group_name not in grouped_data:
        ax.set_visible(False)
        continue

    for fi, fd in enumerate(grouped_data[group_name]['file_data']):
        color = colors_file[fi % len(colors_file)]
        pos = fd['positions']
        n_pts = len(pos)
        if n_pts > 3000:
            step = n_pts // 3000
            plot_pos = pos[::step]
        else:
            plot_pos = pos

        ax.scatter(plot_pos[:, 0], plot_pos[:, 1],
                   c=color, s=1.2, alpha=0.35, linewidths=0, rasterized=True,
                   label=fd['name'][:38] if len(grouped_data[group_name]['file_data']) <= 7 else '_nolegend_')

    # 三区段分界线
    ax.axvline(x=LEFT_MAX_X, color='#FFD700', linewidth=2.5, linestyle='-', alpha=0.9, zorder=10)
    ax.axvline(x=HESITATION_MAX_X, color='#FF4500', linewidth=2.5, linestyle='-', alpha=0.9, zorder=10)

    # 半透明区段覆盖
    ax.fill_between([0, LEFT_MAX_X], 0, 1000, alpha=0.05, color='#377EB8', zorder=0)
    ax.fill_between([LEFT_MAX_X, HESITATION_MAX_X], 0, 1000, alpha=0.08, color='#FFD700', zorder=0)
    ax.fill_between([HESITATION_MAX_X, 1200], 0, 1000, alpha=0.05, color='#E41A1C', zorder=0)

    # 标注
    label_y_top = 40
    ax.text(LEFT_MAX_X / 2, label_y_top, 'LEFT\nX<500', color='#377EB8',
            fontsize=10, fontweight='bold', ha='center', va='top')
    ax.text(600, label_y_top, 'HESITATION\n500-700', color='#B8860B',
            fontsize=8, fontweight='bold', ha='center', va='top')
    ax.text(950, label_y_top, 'RIGHT\nX>700', color='#E41A1C',
            fontsize=10, fontweight='bold', ha='center', va='top')

    stats = group_stats[group_name]
    ax.set_title(f'{group_name}\n'
                 f'L:{stats["left"]:,} | H:{stats["hesitation"]:,} | R:{stats["right"]:,} | '
                 f'L/(L+R):{stats["ratio_left_vs_right"]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=9)
    ax.set_ylabel('Y (pixels)', fontsize=9)
    ax.set_xlim(0, 1200)
    ax.set_ylim(950, -10)

    if len(grouped_data[group_name]['file_data']) <= 7:
        ax.legend(fontsize=5, loc='lower right', markerscale=4, framealpha=0.7)

# 隐藏多余子图
for extra_idx in range(len(GROUP_ORDER), len(axes2)):
    axes2[extra_idx].set_visible(False)

fig2.suptitle(f'Step 2: Full T-Maze Trajectories with Three-Zone Segmentation (for Review) — v{AUTO_VERSION}\n'
              f'Blue=Left(X<{LEFT_MAX_X}) | Yellow=Hesitation([{LEFT_MAX_X},{HESITATION_MAX_X}]) | '
              f'Red=Right(X>{HESITATION_MAX_X})',
              fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / 'step2_trajectories_review.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  [OK] step2_trajectories_review.png")

# ===================================================================
# 图3: Ratio (1) 三者比例 + Ratio (2) 左/(左+右)  — 组级
# ===================================================================
print(f"\n>>> 图3: 组级偏好统计 ...")

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))

# 3a: Ratio (1) 三者比例堆叠柱状图
ax3a = axes3[0]
group_names = [gn for gn in GROUP_ORDER if gn in group_stats]
left_pcts = [group_stats[gn]['pct_left'] * 100 for gn in group_names]
hesi_pcts = [group_stats[gn]['pct_hesi'] * 100 for gn in group_names]
right_pcts = [group_stats[gn]['pct_right'] * 100 for gn in group_names]

ax3a.bar(group_names, left_pcts, color='#377EB8', label='Left (X<500)',
         edgecolor='white', linewidth=0.8)
ax3a.bar(group_names, hesi_pcts, bottom=left_pcts, color='#FFD700',
         label='Hesitation (500-700)', edgecolor='white', linewidth=0.8)
ax3a.bar(group_names, right_pcts, bottom=[l + h for l, h in zip(left_pcts, hesi_pcts)],
         color='#E41A1C', label='Right (X>700)', edgecolor='white', linewidth=0.8)

for i, (gn, lp, hp, rp) in enumerate(zip(group_names, left_pcts, hesi_pcts, right_pcts)):
    if lp > 8:
        ax3a.text(i, lp / 2, f'{lp:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    if hp > 6:
        ax3a.text(i, lp + hp / 2, f'{hp:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    if rp > 8:
        ax3a.text(i, lp + hp + rp / 2, f'{rp:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax3a.text(i, 102, f'n={group_stats[gn]["total"]:,}', ha='center', va='bottom', fontsize=8, color='gray')

ax3a.set_ylabel('% of Total Frames', fontsize=11)
ax3a.set_title('Ratio (1): Three-Zone Distribution\n(Full T-Maze, No Y-Filter)', fontsize=12, fontweight='bold')
ax3a.set_ylim(0, 116)
ax3a.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1))

# 3b: Ratio (2) 左/(左+右) — 排除犹豫区
ax3b = axes3[1]
x_pos = np.arange(len(group_names))
ratios_lr = [group_stats[gn]['mean_ratio_lr'] for gn in group_names]
errors_lr = [group_stats[gn]['sem_ratio_lr'] for gn in group_names]

bar_colors_lr = []
for gn in group_names:
    r = group_stats[gn]['mean_ratio_lr']
    if r > 0.55:
        bar_colors_lr.append('#377EB8')
    elif r < 0.45:
        bar_colors_lr.append('#E41A1C')
    else:
        bar_colors_lr.append('#999999')

ax3b.bar(x_pos, ratios_lr, color=bar_colors_lr, edgecolor='white', linewidth=0.8,
         yerr=errors_lr, capsize=6, error_kw={'linewidth': 1.5})
ax3b.set_xticks(x_pos)
ax3b.set_xticklabels(group_names, fontsize=10)
ax3b.set_ylabel('Left / (Left + Right)', fontsize=11)
ax3b.set_title('Ratio (2): Left/(Left+Right)\n(Hesitation Zone Excluded)', fontsize=12, fontweight='bold')
ax3b.set_ylim(0, 1.12)
ax3b.axhline(y=0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

for i, (gn, ratio, err) in enumerate(zip(group_names, ratios_lr, errors_lr)):
    ax3b.text(i, ratio + max(err, 0.01) + 0.03, f'{ratio:.3f}', ha='center', fontsize=12, fontweight='bold')
    n_files = len(grouped_data[gn]['file_data'])
    ax3b.text(i, 0.03, f'{n_files} file(s)', ha='center', fontsize=8, color='gray')

from matplotlib.patches import Patch
legend_el = [
    Patch(facecolor='#377EB8', label='Left-biased (>55%)'),
    Patch(facecolor='#999999', label='Neutral (45-55%)'),
    Patch(facecolor='#E41A1C', label='Right-biased (<45%)'),
]
ax3b.legend(handles=legend_el, fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1))

fig3.suptitle(f'Step 3: Group-Level Preference (Full T-Maze, Three Zones) — v{AUTO_VERSION}',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig3.savefig(OUTPUT_DIR / 'step3_group_preference.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  [OK] step3_group_preference.png")

# ===================================================================
# 图4: 各文件个体统计
# ===================================================================
print(f"\n>>> 图4: 个体统计 ...")

all_file_stats.sort(key=lambda x: (GROUP_ORDER.index(x['group']) if x['group'] in GROUP_ORDER else 99, x['name']))

fig4, axes4 = plt.subplots(1, 2, figsize=(18, 9))

for s in all_file_stats:
    short = s['name'].replace('DLC_Resnet50_T_trapJul8shuffle1_snapshot_best-30', '')
    short = short.replace('_Resnet50_T_trapJul8shuffle1_snapshot_best-30', '')
    s['short_name'] = short

y_labels = [f"{s['short_name'][:38]}\n[{s['group']}]" for s in all_file_stats]
y_pos = np.arange(len(all_file_stats))

# 4a: Ratio (1) — 三者比例
ax4a = axes4[0]
left_each_pct = [s['left'] / s['total'] if s['total'] > 0 else 0 for s in all_file_stats]
hesi_each_pct = [s['hesitation'] / s['total'] if s['total'] > 0 else 0 for s in all_file_stats]
right_each_pct = [s['right'] / s['total'] if s['total'] > 0 else 0 for s in all_file_stats]

ax4a.barh(y_pos, left_each_pct, color='#377EB8', label='Left (X<500)', edgecolor='none')
ax4a.barh(y_pos, hesi_each_pct, left=left_each_pct, color='#FFD700', label='Hesitation (500-700)', edgecolor='none')
ax4a.barh(y_pos, right_each_pct, left=[l + h for l, h in zip(left_each_pct, hesi_each_pct)],
          color='#E41A1C', label='Right (X>700)', edgecolor='none')

prev_group = None
for i, s in enumerate(all_file_stats):
    if s['group'] != prev_group:
        if prev_group is not None:
            ax4a.axhline(y=i - 0.5, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
        prev_group = s['group']
    # 标注左/(左+右)
    ax4a.text(1.02, i, f'L/(L+R)=\n{s["ratio_left_vs_right"]:.3f}',
              va='center', fontsize=6.5, color='#333333')

ax4a.set_yticks(y_pos)
ax4a.set_yticklabels(y_labels, fontsize=6)
ax4a.set_xlabel('Proportion of Total Frames', fontsize=10)
ax4a.set_title('Ratio (1): Per-File Three-Zone Distribution', fontsize=12, fontweight='bold')
ax4a.set_xlim(0, 1.15)
ax4a.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1))

# 4b: Ratio (2) — 左/(左+右)
ax4b = axes4[1]
ratios_per_file = [s['ratio_left_vs_right'] for s in all_file_stats]
colors_per_file = []
for s in all_file_stats:
    if s['ratio_left_vs_right'] > 0.55:
        colors_per_file.append('#377EB8')
    elif s['ratio_left_vs_right'] < 0.45:
        colors_per_file.append('#E41A1C')
    else:
        colors_per_file.append('#999999')

ax4b.barh(y_pos, ratios_per_file, color=colors_per_file, edgecolor='white', linewidth=0.5)

prev_group = None
for i, s in enumerate(all_file_stats):
    if s['group'] != prev_group:
        if prev_group is not None:
            ax4b.axhline(y=i - 0.5, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
        prev_group = s['group']
    ax4b.text(s['ratio_left_vs_right'] + 0.015, i, f"{s['ratio_left_vs_right']:.3f}",
              va='center', fontsize=9, fontweight='bold')
    ax4b.text(0.015, i, f"{s['total']:,} fr",
              va='center', fontsize=5.5, color='white', fontweight='bold')

ax4b.set_yticks(y_pos)
ax4b.set_yticklabels(y_labels, fontsize=6)
ax4b.set_xlabel('Left / (Left + Right)', fontsize=10)
ax4b.set_title('Ratio (2): Per-File Left/(Left+Right)\n(Hesitation Zone Excluded)', fontsize=12, fontweight='bold')
ax4b.axvline(x=0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
ax4b.set_xlim(0, 1.12)
ax4b.legend(handles=legend_el, fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1))

fig4.suptitle(f'Step 3: Per-File Preference (Full T-Maze, Three Zones) — v{AUTO_VERSION}',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig4.savefig(OUTPUT_DIR / 'step3_per_file_preference.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  [OK] step3_per_file_preference.png")

# ===================================================================
# 统计报告
# ===================================================================
print(f"\n>>> 统计报告 ...")
with open(OUTPUT_DIR / 'statistics_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write(f"T型迷宫行为分析统计报告 v{AUTO_VERSION}\n")
    f.write(f"日期: 2026/07/09\n")
    f.write(f"范围: 全T型（含竖杆+横杠，无Y裁剪）\n")
    f.write(f"左臂: X < {LEFT_MAX_X}  |  犹豫区: [{LEFT_MAX_X}, {HESITATION_MAX_X}]  |  右臂: X > {HESITATION_MAX_X}\n")
    f.write(f"位置计算: ((Head+Tail)/2 + Body)/2，部位不全则用Body\n")
    f.write(f"Ratio (1) = 三者比例 (Left : Hesitation : Right)\n")
    f.write(f"Ratio (2) = Left / (Left + Right)  -- 排除犹豫区\n")
    f.write("=" * 80 + "\n\n")

    f.write("[文件分组]\n")
    f.write("-" * 80 + "\n")
    for gn in GROUP_ORDER:
        if gn in grouped_data:
            fds = grouped_data[gn]['file_data']
            f.write(f"{gn} ({len(fds)} files):\n")
            for fd in fds:
                f.write(f"  - {fd['name']}\n")
    f.write("\n")

    f.write("[Ratio (1) 三者比例]\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'分组':<16} {'左臂':>9} {'犹豫区':>9} {'右臂':>9} {'总计':>9} | {'左%':>6} {'犹豫%':>6} {'右%':>6}\n")
    f.write("-" * 80 + "\n")
    for gn in GROUP_ORDER:
        if gn not in group_stats:
            continue
        s = group_stats[gn]
        f.write(f"{gn:<16} {s['left']:>9,} {s['hesitation']:>9,} {s['right']:>9,} {s['total']:>9,} | "
                f"{s['pct_left']:>5.1%} {s['pct_hesi']:>5.1%} {s['pct_right']:>5.1%}\n")

    f.write(f"\n{'分组':<16} {'左%+/-SEM':>14} {'犹豫%+/-SEM':>14} {'右%+/-SEM':>14}\n")
    f.write("-" * 65 + "\n")
    for gn in GROUP_ORDER:
        if gn not in group_stats:
            continue
        s = group_stats[gn]
        f.write(f"{gn:<16} {s.get('mean_pct_l',s['pct_left'])*100:>6.1f}% +/-{s.get('sem_pct_l',0)*100:>5.1f}% "
                f"{s.get('mean_pct_h',s['pct_hesi'])*100:>6.1f}% +/-{s.get('sem_pct_h',0)*100:>5.1f}% "
                f"{s.get('mean_pct_r',s['pct_right'])*100:>6.1f}% +/-{s.get('sem_pct_r',0)*100:>5.1f}%\n")

    f.write(f"\n\n[Ratio (2) 左/(左+右) -- 排除犹豫区]\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'分组':<16} {'左臂':>9} {'右臂':>9} {'左+右':>9} {'左/(左+右)':>12} {'SEM':>8}\n")
    f.write("-" * 80 + "\n")
    for gn in GROUP_ORDER:
        if gn not in group_stats:
            continue
        s = group_stats[gn]
        lr_total = s['left'] + s['right']
        sem = s.get('sem_ratio_lr', 0)
        f.write(f"{gn:<16} {s['left']:>9,} {s['right']:>9,} {lr_total:>9,} {s['ratio_left_vs_right']:>12.4f} {sem:>8.4f}\n")

    f.write(f"\n\n[各文件单独统计]\n")
    f.write("-" * 85 + "\n")
    f.write(f"{'文件':<47} {'分组':<12} {'左臂':>7} {'犹豫':>7} {'右臂':>7} | {'左%':>6} {'犹豫%':>6} {'右%':>6} | {'L/(L+R)':>8}\n")
    f.write("-" * 85 + "\n")
    for s in all_file_stats:
        t = s['total']
        lpct = s['left'] / t if t > 0 else 0
        hpct = s['hesitation'] / t if t > 0 else 0
        rpct = s['right'] / t if t > 0 else 0
        f.write(f"{s['name']:<47} {s['group']:<12} {s['left']:>7,} {s['hesitation']:>7,} {s['right']:>7,} | "
                f"{lpct:>5.1%} {hpct:>5.1%} {rpct:>5.1%} | "
                f"{s['ratio_left_vs_right']:>8.4f}\n")

print("  [OK] statistics_report.txt")

# ===== Summary =====
print(f"\n{'='*70}")
print(f"v{AUTO_VERSION} analysis complete! Output files in {OUTPUT_DIR.name}/:")
for fpath in sorted(OUTPUT_DIR.iterdir()):
    size_kb = fpath.stat().st_size / 1024
    print(f"  {fpath.name} ({size_kb:.0f} KB)")
print(f"{'='*70}")
