# T型迷宫果蝇行为分析项目 (T-Maze Drosophila Behavior Analysis)

## 项目概述

本项目利用 **DeepLabCut (DLC)** 无标记姿态估计技术，追踪果蝇（*Drosophila melanogaster*）在 T 型迷宫中的运动轨迹，研究**乙醇（EtOH）暴露**对果蝇左/右臂选择偏好的影响。

### 实验设计

- **实验装置**: T 型迷宫（竖杆 + 水平横杠，横杠两端为左臂和右臂）
- **追踪部位**: 头部 (head)、腹部 (body)、尾部 (tail) — 每帧输出 (x, y, likelihood)
- **实验分组**: 4 个处理条件
  | 分组 | 乙醇浓度 | 说明 |
  |------|---------|------|
  | 10% EtOH | 10% | 低浓度乙醇 |
  | 5% EtOH | 5% | 中间浓度乙醇 |
  | 15% EtOH | 15% | 高浓度乙醇 |
  | Air (Ctrl) | 0% | 空气对照组 |

### 核心假设

乙醇暴露会改变果蝇在 T 型迷宫中的左/右臂选择行为。通过比较不同乙醇浓度组与对照组在左右臂的停留时间比例，判断乙醇对行为的影响。

---

## 数据格式

### H5 文件（DLC 输出）

每个 `.h5` 文件对应一只果蝇在一次实验中的追踪数据：

- **存储路径**: `df_with_missing/table`
- **每行格式**: `[frame_index, [head_x, head_y, head_lik, body_x, body_y, body_lik, tail_x, tail_y, tail_lik]]`
- **likelihood**: DLC 置信度 (0~1)，< 0.6 视为该部位追踪失败

### 文件命名规则

```
<条件>-<个体ID>DLC_Resnet50_T_trapJul8shuffle1_snapshot_best-30.h5
```

示例: `10%ETOH-air-WTDLC_Resnet50_T_trapJul8shuffle1_snapshot_best-30.h5`
- `10%ETOH-air`: 处理条件
- `WTDLC`: 果蝇个体标识
- 后缀: DLC 模型信息（固定不变）

### 位置计算方法

所有分析脚本统一使用以下方法计算果蝇质心位置：

```python
if head_ok and body_ok and tail_ok:
    # 三部位均有效: ((头+尾)/2 + 腹)/2
    cx = ((head_x + tail_x) / 2 + body_x) / 2
    cy = ((head_y + tail_y) / 2 + body_y) / 2
elif body_ok:
    # 仅腹部有效: 使用腹部位置
    cx, cy = body_x, body_y
else:
    # 全部无效: 跳过该帧
```

---

## 分析脚本版本演进

### 版本总览

| 版本 | 脚本文件 | 输出目录 | 分析方法 |
|------|---------|---------|---------|
| v1 | `maze_analysis.py` | `analysis_output/` | 两区段 (左/右), Y 裁剪, X=570 分界 |
| v2 | `maze_analysis_v2.py` | `analysis_output_v2/` | 两区段 (左/右), Y 裁剪, X=600 分界, 排除竖杆 |
| v3 | `maze_analysis_v3.py` | `analysis_output_v3/` | 三区段 (左/犹豫/右), Y 裁剪, X=500/700 |
| v4 | `maze_analysis_v4.py` | `analysis_output_v4/` | 三区段 (左/犹豫/右), **无 Y 裁剪**, X=500/700 |
| **v5** | **`maze_analysis_v5.py`** | **`analysis_output_v5/`** | 三区段 (左/犹豫/右), 无 Y 裁剪, X=500/700, **最新数据 + 自动递增版本号** |

### 各版本详解

#### v1 — 初始两区段分析
- **分界**: 单条分割线 X=570
- **Y 范围**: 仅分析横杠区域 (Y ∈ [195, 395])
- **输出**: 位置热图、轨迹审查图、左臂偏好统计
- **局限**: 未考虑 T 型交汇处的"犹豫"行为

#### v2 — 修正方向 + 排除竖杆
- **改进**: 确认图像方向为头朝上（低 Y = 上方），调整坐标解读
- **分界**: 单条分割线 X=600
- **Y 范围**: 仅分析横杠区域 (Y ∈ [195, 395])
- **新增**: 竖杆 vs 横杠过滤可视化

#### v3 — 三区段分割
- **重大改进**: 引入三区段模型
  - **左臂**: X < 500
  - **犹豫区**: 500 ≤ X ≤ 700（T 型交汇处）
  - **右臂**: X > 700
- **Y 范围**: 仍裁剪至横杠区域 (Y ∈ [195, 395])
- **双比率输出**:
  - Ratio (1): 左:犹豫:右 三者比例
  - Ratio (2): 左/(左+右) — 排除犹豫区
- **包含文件**: 12 个原始文件（含 `air_air_WT_3DLC`, `airw_air_WT_1DLC`）

#### v4 — 全 T 型范围（无 Y 裁剪）
- **改进**: 移除 Y 轴裁剪，分析竖杆+横杠的全部区域
- **分类**: 纯粹基于 X 坐标，无 Y 轴限制
- **剔除**: 移除 2 个低质量对照文件 (`air_air_WT_3DLC`, `airw_air_WT_1DLC`)
- **当前推荐的方法论标准**

#### v5 — 更新数据库 + 自动版本管理 (2026/07/09)
- **与 v4 相同的方法论**（全 T 型、三区段、无 Y 裁剪）
- **文件分组更新**: 5% EtOH 组中 `5et_air_WTDLC` 替换为 `5et_air_WT_2DLC`
- **当前文件数**: 9 个 H5 文件
- **🆕 自动递增版本号**: 每次运行自动扫描已有 `analysis_output_v*` 目录，输出到 `analysis_output_v{N+1}`，历史结果永不覆盖

### 辅助脚本

| 脚本 | 用途 |
|------|------|
| `plot_trajectories.py` | 绘制所有果蝇的轨迹图（按时间着色），输出 300/600 DPI 版本 |
| **`maze_orientation_analysis.py`** | **朝向分析**：计算果蝇逐帧朝向，分析左/右朝向偏好 |

---

## 朝向分析 (Orientation Analysis)

### 朝向定义

果蝇朝向 = 以下方向向量中有数据者的**圆均值 (circular mean)**：

| 向量 | 公式 | 条件 |
|------|------|------|
| body → head | (head - body) | head 和 body likelihood > 0.6 |
| tail → head | (head - tail) | head 和 tail likelihood > 0.6 |
| tail → body | (body - tail) | body 和 tail likelihood > 0.6 |

- 角度 = `atan2(y, x)`，范围 [-π, π]
- **0° = 朝右**（指向右臂），**±180° = 朝左**（指向左臂）
- cos(angle) > 0 → facing right；cos(angle) < 0 → facing left

### 2×2 分析条件

| 条件 | Y 范围 | X 区域 |
|------|--------|--------|
| No Y-crop + Hesitation | 全部 Y | 全部 X |
| No Y-crop − Hesitation | 全部 Y | 排除犹豫区 X∉[500,700] |
| Y-crop [195-395] + Hesitation | Y∈[195,395] | 全部 X |
| Y-crop [195-395] − Hesitation | Y∈[195,395] | 排除犹豫区 X∉[500,700] |

### 运行朝向分析

```bash
python maze_orientation_analysis.py
```

输出目录自动递增：`orientation_analysis_v1/` → `v2/` → ...，与位置分析目录独立。

### 输出文件

| 文件 | 内容 |
|------|------|
| `step1_polar_orientation.png` | 4×4 极坐标朝向直方图（4条件 × 4分组） |
| `step2_facing_bias.png` | 左/右朝向偏好柱状图（分组比较） |
| `step3_position_angle_uncropped.png` | 位置-朝向 2D 直方图（全 Y 范围） |
| `step4_position_angle_cropped.png` | 位置-朝向 2D 直方图（Y 裁剪区域） |
| `statistics_report.txt` | 完整统计报告（圆统计量、Rayleigh 检验） |

### v1 核心发现 (2026/07/09)

**所有组均表现出强烈的左朝向偏好**（80-99% 朝向左侧）：

| 条件 | 10% EtOH | 5% EtOH | 15% EtOH | Air (Ctrl) |
|------|----------|---------|----------|------------|
| No Y-crop + Hesitation | **0.024** | **0.135** | **0.079** | **0.110** |
| No Y-crop − Hesitation | **0.011** | **0.166** | **0.099** | **0.083** |
| Y-crop + Hesitation | **0.025** | **0.129** | **0.086** | **0.082** |
| Y-crop − Hesitation | **0.011** | **0.165** | **0.099** | **0.083** |

*表中数值 = 朝右帧占比 (PropR)，越低表示越偏左*

关键结论：
- **压倒性左偏**：无论乙醇浓度高低，果蝇在 T 型迷宫中几乎始终面朝左臂方向
- **Y 裁剪与犹豫区过滤影响极小**：4 个条件组内差异仅 ~0.01-0.03
- **5% EtOH 组个体差异最大**：`5%ETOH-air-WTDLC` PropR=0.206（有 20% 朝右），`5et_air_WT_2DLC` PropR=0.081（仅 8% 朝右）
- **Air 组内存在异常个体**：`air_air_WTDLC` 朝向接近随机分布 (R=0.158, PropR=0.435)，与其他个体截然不同

---

## 迷宫区域定义（v3-v5 通用）

```
        LEFT ARM              HESITATION             RIGHT ARM
        X < 500              500 ≤ X ≤ 700           X > 700
    ┌──────────────┬─────────────────────┬──────────────┐
    │              │                     │              │
    │   左臂       │     犹豫区          │    右臂      │
    │              │   (T型交汇处)        │              │
    └──────────────┴─────────────────────┴──────────────┘
                          ↑ 竖杆 (stem)
                          │ 通往迷宫入口
```

---

## 项目文件结构

```
Ttrap/
├── README.md                          # 本文件 — 项目文档
│
├── *.h5                               # DLC追踪数据 (13个文件, 6组)
│   ├── 10%ETOH-air-WTDLC_...h5        #   10%乙醇组 (1只)
│   ├── 5%ETOH-air-WTDLC_...h5         #   5%乙醇组 (3只)
│   ├── 5et_air_WT_2DLC_...h5          #
│   ├── 5%ETOH-air-WT -3DLC_...h5      #
│   ├── 15%ETOH-air-WTDLC_...h5        #   15%乙醇组 (1只)
│   ├── 20%EtOH_air_WT_1DLC_...h5      #   20%乙醇组 (2只, X-100校正)
│   ├── 20%EtOH_air_WT_2DLC_...h5      #
│   ├── 25%EtOH_air_WTDLC_...h5        #   25%乙醇组 (1只, X-100校正)
│   ├── AIR-air-WT2DLC_...h5           #   空气对照组 (5只)
│   ├── AIR-air-WTDLC_...h5            #
│   ├── air_air_WT_2DLC_...h5          #
│   ├── air_air_WT_4DLC_...h5          #
│   └── air_air_WTDLC_...h5            #
│
├── maze_analysis.py                   # v1 分析脚本
├── maze_analysis_v2.py                # v2 分析脚本
├── maze_analysis_v3.py                # v3 分析脚本
├── maze_analysis_v4.py                # v4 分析脚本
├── maze_analysis_v5.py                # v5 分析脚本 (位置分析，自动递增输出版本)
├── maze_orientation_analysis.py       # 朝向分析脚本 (自动递增输出版本)
├── plot_trajectories.py               # 轨迹绘制工具
│
├── analysis_output/                   # v1 位置分析输出
├── analysis_output_v2/                # v2 位置分析输出
├── analysis_output_v3/                # v3 位置分析输出
├── analysis_output_v4/                # v4 位置分析输出
├── analysis_output_v5/                # v5 位置分析输出 (当前最新)
├── analysis_output_vN/                # (自动递增...)
├── orientation_analysis_v1/           # v1 朝向分析输出 (当前最新)
├── orientation_analysis_vN/           # (自动递增...)
│   ├── step2_trajectories_review.png  #   轨迹+分割线审查图
│   ├── step3_group_preference.png     #   组级偏好统计
│   ├── step3_per_file_preference.png  #   个体偏好统计
│   └── statistics_report.txt          #   统计报告文本
│
├── trajectories.png                   # 全局轨迹图 (300 DPI)
├── trajectories_hd.png                # 全局轨迹图 (600 DPI)
└── .claude/                           # Claude Code 配置
    └── settings.local.json
```

---

## 如何重新运行分析

### 前置条件

```bash
pip install h5py numpy matplotlib
```

### 运行位置分析（当前推荐）

```bash
cd "D:\文件\大一下\AI Eth 实验\Ttrap"
python maze_analysis_v5.py
```

### 运行朝向分析

```bash
python maze_orientation_analysis.py
```

**版本号自动管理**：两个脚本均会扫描已有输出目录，自动取最大版本号 +1。无需手动修改 `OUTPUT_DIR`。

### 更新数据库后重新分析

1. 将新的 `.h5` 文件放入项目根目录，或替换旧的 `.h5` 文件
2. 编辑 `maze_analysis_v5.py` 中的 `FILE_GROUPS` 字典，更新文件前缀映射（如有新增/删除文件）
3. 直接运行脚本——输出目录版本号自动递增，无需手动修改

### 生成轨迹总览图

```bash
python plot_trajectories.py
```

---

## v9 最新结果摘要 (2026/07/09)

### L/(L+R) 比率（排除犹豫区，全 T 型无 Y 裁剪）

| 分组 | 文件数 | 总帧数 | L/(L+R) | SEM | 偏好 |
|------|--------|--------|---------|-----|------|
| Air (Ctrl) | 5 | 57,582 | **0.5555** | 0.0577 | 轻微左偏 |
| 10% EtOH | 1 | 7,649 | **0.5848** | — | 左偏 |
| 5% EtOH | 3 | 52,378 | **0.4990** | 0.1852 | 个体差异极大 |
| 15% EtOH | 1 | 3,518 | **0.7427** | — | 强左偏 |
| 20% EtOH | 2 | 15,991 | **0.3952** | 0.0507 | 右偏 |
| 25% EtOH | 1 | 11,031 | **0.2378** | — | 强右偏 |

### 关键发现

- **剂量-反应非线性**: 低浓度乙醇 (10-15%) 增强左偏，高浓度 (20-25%) 翻转为右偏，25% 右偏最强
- **5% EtOH 组个体差异最大**: 三只果蝇覆盖全谱——强左偏 (0.76)、中性 (0.56)、强右偏 (0.12)，均值恰好接近 0.50
- **对照组**稳定轻微左偏 (0.5555)，5 只个体 SEM 仅 0.0577
- **20% 和 25% EtOH 视频有偏移**: 已左移 100 像素校正；5% 组新增个体同样校正
- **样本量限制**: 10%、15%、25% 组各仅 1 只个体

---

## 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `LIKELIHOOD_THRESHOLD` | 0.6 | DLC 置信度阈值 |
| `LEFT_MAX_X` | 500 | 左臂 X 坐标上限 |
| `HESITATION_MAX_X` | 700 | 犹豫区 X 坐标上限 |
| `X range` | [0, 1200] | 热图 X 轴范围 |
| `Y range` | [0, 1000] | 热图 Y 轴范围 |
| `DPI` | 300 | 输出图片分辨率 |
| `Font` | Microsoft YaHei | 图表中文字体 |

---

*最后更新: 2026/07/09 — v9 位置分析（6组×13只果蝇）+ v1 朝向分析*
