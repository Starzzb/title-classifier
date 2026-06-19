# 视频/图片标题批量标准化重命名工具

## 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [快速开始](#快速开始)
- [CLI命令详解](#cli命令详解)
- [GUI使用说明](#gui使用说明)
- [YOLO视觉分析](#yolo视觉分析)
- [帧数处理流程](#帧数处理流程)
- [OpenVINO CPU加速](#openvino-cpu加速)
- [运动检测前置过滤](#运动检测前置过滤)
- [硬件视频解码](#硬件视频解码)
- [多线程并行处理](#多线程并行处理)
- [音频字幕集成](#音频字幕集成)
- [VAD分段策略](#vad分段策略)
- [AI Provider 配置](#ai-provider-配置)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [更新日志](#更新日志)

---

## 项目简介

本工具是一套**三阶段媒体文件批量重命名系统**，专为本地视频/图片库管理设计。通过智能分词、AI 清洗和视觉识别，将杂乱的文件名转换为**标准化、可检索、语义清晰**的格式，同时保证操作的安全性与可逆性。

**适用场景**：
- 整理下载目录中的大量视频/图片文件
- 为媒体库建立统一命名规范
- 为后续 AI 分类/标签系统准备数据
- 批量清理历史遗留的混乱文件名

---

## 设计理念

>没想好😵‍💫
---

## 核心功能

| 功能模块 | 特性描述 |
|---------|---------|
| **YOLO 视觉分析** | 集成 YOLO11/YOLOv8，支持检测、姿态估计、实例分割 |
| **视频全面分析** | 每2秒采样，智能选择10帧代表性帧给VLM |
| **姿态分析** | 17个关键点，识别跪姿、站立、坐姿等动作 |
| **智能帧选择** | 基于姿态变化、置信度、关键点可见性选择最佳帧 |
| **关键词提取** | 聚焦穿着、姿势、行为，水印博主名最优先 |
| **SRT字幕生成** | 视觉描述写入SRT开头，支持音频字幕追加 |
| **VAD语音分段** | Silero VAD 三层策略：微合并→语义打包→静音过滤 |
| **音频字幕** | 基于VAD分段的智能音频转录，支持API拒绝自动重试 |
| **字幕后处理** | 拆分长字幕、过滤无效内容、格式化时间戳 |
| **VLM字幕上下文** | 视觉识别时自动匹配帧对应的字幕时间段 |
| **智能扫描** | 递归遍历指定目录，支持 10+ 种视频格式和 7 种图片格式 |
| **无意义检测** | 程序自动识别 hash、Tg 来源、IMG_xxx 等无意义标题 |
| **CLIP 预分类** | 使用 OpenCLIP 进行本地预分类，支持多标签输出 |
| **智能压缩** | 自动压缩图片/视频帧，避免 API 传输过大文件 |
| **安全预览** | 三阶段设计，先生成待审表，人工确认后再执行，零风险操作 |
| **冲突避让** | 自动检测文件名冲突，智能追加序号（`_1`, `_2`），防止覆盖 |

---

## 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.10+ | 推荐 3.12 |
| 操作系统 | Windows / macOS / Linux | 全平台兼容 |
| 磁盘空间 | >= 500MB | 包含模型文件和依赖 |
| 权限要求 | 目标目录读写权限 | 必需 |
| ffmpeg | 全局 PATH | 用于视频帧提取和音频处理 |

---

## 安装步骤

### 1. 安装 uv（Python 包管理器）

**Windows（PowerShell）**：
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux**：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 克隆项目

```bash
git clone https://github.com/Starzzb/title-classifier.git
cd title-classifier
```

### 3. 创建虚拟环境并安装依赖

```powershell
uv venv --python 3.12
uv sync
```

### 4. 下载模型文件

#### 一键下载所有模型（推荐）

```powershell
uv run python scripts/download_models.py
```

这会自动下载 YOLO 模型和 CLIP 模型到 `models/` 目录。Silero VAD 模型会在首次使用时自动下载。

#### 手动下载单个模型

**YOLO 模型**（用于姿态检测，推荐）：

```powershell
uv run python scripts/download_yolo_models.py
```

| 模型 | 大小 | 功能 | 说明 |
|------|------|------|------|
| `yolo11m-pose.pt` | 40MB | 姿态估计 | medium，**默认推荐**，精度+5-8% |
| `yolov8n.pt` | 6MB | 人体检测 | nano，最快 |
| `yolov8s-pose.pt` | 22MB | 姿态估计 | small，旧版默认 |
| `yolov8n-seg.pt` | 7MB | 实例分割 | nano，最快 |

> **模型选择**：默认使用 `yolo11m-pose`（medium），精度比 yolov8s 提升 ~5-8% mAP，CPU 推理 ~150ms/帧。
> 如需更高速度，可改回 `yolov8n-pose`（6MB，~50ms/帧）。
> 修改 `src/title_classifier/detectors/yolo.py` 中的 `YOLO_MODELS` 配置即可。

#### 切换更高精度模型

pose 模型支持多种精度等级，按需选择：

**YOLO11 系列（推荐）**：

| 模型 | 大小 | CPU推理 | 精度 | 适用场景 |
|------|------|---------|------|----------|
| `yolo11n-pose.pt` | 6MB | ~80ms | 低 | 批量处理、速度优先 |
| `yolo11s-pose.pt` | 20MB | ~120ms | 中 | 均衡选择 |
| `yolo11m-pose.pt` | 40MB | ~150ms | 高 | **默认推荐** |
| `yolo11l-pose.pt` | 85MB | ~300ms | 很高 | 高精度需求 |

**YOLOv8 系列（旧版）**：

| 模型 | 大小 | CPU推理 | 精度 | 适用场景 |
|------|------|---------|------|----------|
| `yolov8n-pose.pt` | 6.5MB | ~50ms | 低 | 批量处理、速度优先 |
| `yolov8s-pose.pt` | 22MB | ~100ms | 中 | 旧版默认 |
| `yolov8m-pose.pt` | 52MB | ~200ms | 高 | 精度优先 |

**切换步骤**：

```powershell
# 1. 下载模型（以 yolo11m-pose 为例）
uv run python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"
mv yolo11m-pose.pt models/yolo/

# 2. 修改配置
# 编辑 src/title_classifier/detectors/yolo.py 第 19 行：
#   "pose": YOLO_MODEL_DIR / "yolov8s-pose.pt",
# 改为：
#   "pose": YOLO_MODEL_DIR / "yolo11m-pose.pt",
```

> **注意**：detect 和 segment 模型也支持同样切换，下载对应文件并修改 `YOLO_MODELS` 配置即可。

**CLIP 模型**（可选，用于图像预分类）：

```powershell
uv run python scripts/download_clip.py
```

**Silero VAD 模型**（音频分段）：

无需手动下载，`silero-vad` 包会在首次使用时自动下载模型。

### 5. 配置 AI API

在项目根目录创建 `.env` 文件：

```env
# gcli API（用于视觉识别，推荐）
GCLI_API_KEY=your_gcli_key_here

# 小米 MiMo API（用于视觉识别和音频字幕）
MIMO_API_KEY=your_mimo_key_here

# 智谱 API（用于文本优化）
ZHIPU_API_KEY=your_zhipu_key_here
```

---

## 快速开始

### CLI 命令方式

```powershell
# 步骤1：扫描目录并生成待审表
uv run title-classifier scan -d "F:\Videos"

# 步骤2：（可选）AI 优化标题
uv run title-classifier refine -p gcli

# 步骤3：（可选）音频识别生成字幕
uv run title-classifier audio -p mimo

# 步骤4：视觉识别提取关键词（使用YOLO全面分析）
uv run title-classifier vision --use-yolo --yolo-model pose -p gcli

# 步骤5：预览重命名结果
uv run title-classifier rename --dry-run

# 步骤6：执行重命名
uv run title-classifier rename
```

### GUI 方式

```powershell
# 启动图形界面
uv run title-classifier gui
```

---

## CLI命令详解

### 全局参数

| 参数 | 说明 |
|------|------|
| `-v, --verbose` | 详细输出（控制台显示 DEBUG 级别日志） |
| `--log` | 日志文件路径（不指定则自动输出到 `logs/<日期>/` 目录） |

**日志系统：**
- 控制台默认输出 INFO 级别，`--verbose` 时输出 DEBUG 级别
- 日志文件始终记录 DEBUG 级别，按天分目录存储在 `logs/<日期>/` 下
- 文件名格式：`HHMMSS.log`

**耗时追踪（视觉识别）：**
```
处理完成: 总耗时=12.35s | YOLO=4.20s(38帧推理,12帧跳过) | CLIP=1.52s | 帧选择=0.008s | VLM=5.71s
```

**模型详情日志（DEBUG 级别）：**
```
[YOLO] detect: 1人, 置信度=0.923, 耗时=0.085s
[YOLO] pose: 关键点=15/17, 姿态=['站立'], 耗时=0.120s
[YOLO] segment: mask=0.318, 色彩变化=45.2, 耗时=0.095s
[CLIP] 分类完成: 耗时=0.235s | 穿着=黑色丝袜(0.723), 动作=坐姿(0.851), 发型=长发(0.912)
[CLIP] 差异度计算: 50帧, 编码=1.20s, 计算=0.015s, 总计=1.22s | 分数: min=0.012, max=0.823, avg=0.156
Vision API调用成功: provider=siliconflow, model=Qwen/Qwen3.6-35B-A3B, 耗时=5.71s, tokens=1234+567=1801, 响应长度=256
```

### scan 命令 - 扫描目录

```powershell
uv run title-classifier scan -d "F:\Videos" [选项]
```

| 参数 | 说明 |
|------|------|
| `-d, --dir` | 目标目录或单个媒体文件路径（必需） |
| `-o, --output` | 输出文件路径 |
| `--output-dir` | 输出目录（默认 data/output） |
| `-a, --append` | 追加模式 |
| `--exclude-dir` | 排除的目录 |
| `--force` | 强制重新分类（全量CSV + 更新数据库） |
| `--sync-db` | 仅同步数据库（不生成CSV） |

**扫描模式说明：**

| 模式 | CSV | 数据库 |
|------|-----|--------|
| 普通扫描 | 生成（仅未分类文件） | 不动 |
| `--force` | 生成（全量，重新判断 needs_vision） | 全量更新 |
| `--sync-db` | 不生成 | 全量更新（含已分类文件的路径和元数据） |

```powershell
# 普通扫描：只处理未分类文件
uv run title-classifier scan -d "F:\Videos"

# 强制重分类：全量CSV + 更新数据库
uv run title-classifier scan -d "F:\Videos" --force

# 同步数据库：将所有文件信息写入数据库（不生成CSV）
# 同步完成后自动检查缺少视觉描述的记录，生成待处理 CSV
uv run title-classifier scan -d "F:\Videos" --sync-db

# 同步整个磁盘（排除系统目录）
uv run title-classifier scan -d "E:\" --sync-db --exclude-dir "$RECYCLE.BIN" "System Volume Information"
```

### refine 命令 - AI优化标题

```powershell
uv run title-classifier refine [选项]
```

| 参数 | 说明 |
|------|------|
| `-c, --csv` | CSV文件路径 |
| `-p, --provider` | AI Provider（gcli/zhipu/ollama） |

### audio 命令 - 音频识别

```powershell
uv run title-classifier audio [选项]
```

| 参数 | 说明 |
|------|------|
| `-c, --csv` | CSV文件路径 |
| `-p, --provider` | AI Provider（默认 mimo） |
| `--all` | 处理所有未识别文件 |

音频识别使用 Silero VAD 进行语音活动检测，通过三层策略（微合并→语义打包→静音过滤）生成最优分段，然后调用 MiMo API 进行语音转录。详见 [VAD分段策略](#vad分段策略)。

### vision 命令 - 视觉识别

```powershell
uv run title-classifier vision [选项]
```

| 参数 | 说明 |
|------|------|
| `-c, --csv` | CSV文件路径 |
| `-p, --provider` | AI Provider |
| `--use-yolo` | 使用YOLO检测 |
| `--yolo-model` | YOLO模型类型（detect/pose/segment，可多选） |
| `--yolo-conf` | YOLO置信度阈值（默认0.4） |
| `--use-clip` | 使用CLIP预分类 |
| `--vlm-frames` | VLM帧数（默认10） |
| `--analysis-step` | 采样间隔秒数（默认2.0） |
| `--device` | 推理设备（auto/cuda/cpu，默认auto） |
| `--concurrent` | 并发处理视频数（默认1，推荐3） |
| `--retry-failed` | 重试之前失败的行（vision_failed=true） |
| `--all` | 处理所有未识别文件 |

### rename 命令 - 执行重命名

```powershell
uv run title-classifier rename [选项]
```

| 参数 | 说明 |
|------|------|
| `-c, --csv` | CSV文件路径 |
| `--dry-run` | 模拟运行 |

### gui 命令 - 启动图形界面

```powershell
uv run title-classifier gui
```

---

## GUI使用说明

### 启动 GUI

```powershell
uv run title-classifier gui
```

### 功能标签页

| 标签页 | 功能 |
|--------|------|
| **Stage1 扫描** | 扫描目录或单个文件，生成待审表 |
| **Stage1b AI优化** | AI优化标题，支持预览编辑 |
| **Stage1c 音频识别** | VAD语音分段 + MiMo API语音转录 |
| **Stage1c 视觉识别** | YOLO检测 + VLM识别 |
| **Stage2 重命名** | 执行重命名操作 |

### Stage1 扫描

支持选择**目录**或**单个媒体文件**：
- 选择目录：递归扫描所有子目录中的媒体文件
- 选择文件：只处理选中的单个媒体文件

**选项：**
- **追加模式**：新扫描结果追加到现有CSV文件
- **强制重新分类**：即使文件已有分类标签也重新处理（全量CSV + 更新数据库）
- **同步数据库**：仅将文件信息同步到数据库（不生成CSV）

"开始扫描"按钮根据勾选的选项执行对应模式。

### Stage1b 预览编辑功能

**布局：** 顶部操作栏按功能分组（AI操作 / 编辑 / 过滤），底部状态栏包含确认写入和撤销按钮。

**表格列：** 原始标题 / 视觉 / 音频 / AI优化结果 / 最终文件名预览

**操作：**
- **AI操作组**：优化选中、优化全部、填入原标题
- **编辑组**：编辑、删除
- **过滤组**：多条件下拉过滤（needs_vision / audio_recognized / 已修改）
- **右键菜单**：编辑标题、采用原标题、重置、状态切换、批量操作
- **撤销**：支持撤销最近 10 步操作
- **最终文件名预览**：实时显示 `[关键词]_原标题` 的实际效果
    - 音频已识别：切换 TRUE/FALSE（即时写入CSV）
  - **其他**：
    - 删除：从预览中移除
- **确认写入**：只写入已修改行的 final_name（自动加中括号）
- **修改行高亮**：浅蓝色背景，一眼区分修改/未修改

### Stage1c 音频识别

音频识别标签页配置：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| 音量阈值 | 0.01 | 静音检测阈值（RMS能量） |
| 跳过静音 | 勾选 | 跳过静音片段节省API调用 |
| VAD语音检测 | 勾选 | 使用Silero VAD（推荐） |
| VAD最小时长 | 250ms | 低于此时长的语音段忽略 |
| VAD最小静音 | 100ms | 用于合并相邻语音段 |
| 字幕后处理 | 勾选 | 拆分长字幕、过滤无效内容 |
| 最长字幕时长 | 10秒 | 单个字幕的最大时长 |
| 最大字符数 | 100 | 单个字幕的最大字符数 |

**VAD分段参数**（在 `config/default.toml` 中配置）：

```toml
[audio.vad]
# 第一层：微合并
merge_gap = 1.5          # 间隙小于此值的相邻语音段合并
min_keep_duration = 1.0  # 合并后仍不足此值的跳过

# 第二层：语义打包
max_chunk = 25.0         # 模型时长上限
long_gap = 3.0           # 间隙超过此值强制封口

# 第三层：静音过滤
min_duration = 1.0       # 最小块时长
min_speech_ratio = 0.3   # 最小语音占比
```

### Stage1c 视觉识别

- **检测器选择**：YOLO（默认）
- **YOLO模型多选**：detect、pose、segment
- **分析参数**：采样间隔、VLM帧数
- **选项**：处理所有文件、启用调试模式

如果视频已有音频字幕（SRT文件），视觉识别会自动读取字幕内容作为VLM上下文，并将每帧对应的时间戳与字幕匹配，帮助VLM理解画面与语音的关联。

### Stage2 批量操作

- **一键确认所有记录**：将所有记录设置为"已确认"
- **一键清空final_name**：清空所有final_name

---

## YOLO视觉分析

### 功能说明

YOLO11/YOLOv8 是多功能视觉分析模型，支持：

1. **目标检测**（detect）：检测人体位置和边界框
2. **姿态估计**（pose）：17个关键点，识别动作姿态
3. **实例分割**（segment）：像素级人体区域分割

### 两种分析模式

#### 1. 基础模式（默认）

使用单个YOLO模型（pose）进行分析：

```powershell
uv run title-classifier vision --use-yolo -p gcli
```

**送入VLM的图片逻辑**：
- 每2秒采样一帧（可配置，上限50帧）
- 使用YOLO Pose模型分析每帧姿态
- **分区段选择**：将采样帧等分为 vlm_frames 个区段（默认10段），每段内按评分选最优帧，保证全视频均匀覆盖
- 区段内评分：置信度40% + 关键点可见性30% + 姿态变化30%；无人体帧取段内中间帧
- 将选中帧的图片传给VLM
- 同时传入姿态分析结果作为上下文

#### 2. 全面分析模式

使用三个YOLO模型（detect、pose、segment）进行全面分析：

```powershell
uv run title-classifier vision --use-yolo --comprehensive -p gcli
```

**送入VLM的图片逻辑**：
- 每2秒采样一帧（可配置，上限50帧）
- **三个模型并行分析每帧**：
  - **detect模型**：检测人体位置和边界框
  - **pose模型**：分析人体姿态（17个关键点）
  - **segment模型**：实例分割，提供人体区域掩码和穿着分析
- **投票决策**：至少两个模型检测到人体才认为有人体
- **动态权重**：根据每个模型的置信度自动调整权重
- **分区段选择**：将采样帧等分为 max_frames 个区段（默认10段），每段内按置信度+关键点+姿态变化加权选最优帧，保证全视频均匀覆盖
- 将选中帧的图片传给VLM
- 同时传入三个模型的详细分析结果作为上下文（分别标注来源）

### 帧数处理流程

视频分析分为三个阶段，每阶段的帧数关系如下：

```
┌─────────────────────────────────────────────────────────────┐
│                      采样阶段                                │
│  视频时长 108秒，采样间隔 2秒                                 │
│  → np.arange(0, 108, 2) = 54个采样点                         │
│  → 超过 max_sample_frames(50) 限制                           │
│  → np.linspace(0, 108, 50) = 50个采样点（均匀分布）           │
│                                                              │
│  采样帧数 = min(视频时长/间隔, max_sample_frames)             │
│  默认上限: 50帧                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    YOLO 推理阶段                             │
│  对每帧进行运动检测：                                         │
│  - 静止帧 → 复用上一帧结果（跳过YOLO推理）                    │
│  - 运动帧 → 执行 YOLO 推理                                   │
│                                                              │
│  YOLO 推理帧数 = 采样帧数 - 运动检测跳过数                    │
│  示例：50帧采样，30帧静止 → YOLO 推理 20帧                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    VLM 帧选择阶段                            │
│  从 timeline（包含所有帧）中智能选择代表性帧：                │
│  - 将 timeline 等分为 vlm_frames 个区段                      │
│  - 每区段按置信度+关键点评分选最优帧                          │
│  - 无人体帧取区段中间帧                                       │
│                                                              │
│  VLM 发送帧数 = vlm_frames（默认 10帧）                      │
└─────────────────────────────────────────────────────────────┘
```

### 帧数配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--analysis-step` | 2.0 | 采样间隔（秒），越小越细致 |
| `--max-sample-frames` | 50 | 采样帧数上限，超过后均匀分布 |
| `--vlm-frames` | 10 | 发送给 VLM 的帧数 |
| `--motion-threshold` | 5.0 | 运动检测阈值（%） |

### 配置示例

```powershell
# 更细致的分析（更多采样帧）
uv run title-classifier vision --use-yolo --max-sample-frames 80 --vlm-frames 15 -p gcli

# 快速分析（更少采样帧）
uv run title-classifier vision --use-yolo --max-sample-frames 30 --vlm-frames 5 -p gcli
```

### 送入VLM的上下文格式

#### 基础模式上下文示例
```
【视频全面分析结果】
- 视频时长: 108.0秒
- 人体出现比例: 85.2%
- 主要姿态: 跪姿/蹲姿, 弯腰/前倾
- 姿态分布: 跪姿/蹲姿(45次), 弯腰/前倾(32次), 站立/正常姿态(18次)
- 姿态变化次数: 8
  * 5.2s: 站立/正常姿态 -> 弯腰/前倾
  * 12.8s: 弯腰/前倾 -> 跪姿/蹲姿
- 人体出现时间段:
  * 2.0s - 106.0s
- 平均置信度: 0.87
- 平均可见关键点: 12.3/17

【各帧详细分析（图片序号对应下方描述）】
- 图1@2.0s: 人体检测, 姿态=站立/正常姿态, 置信度=0.92, 关键点=15/17
- 图2@12.8s: 人体检测, 姿态=弯腰/前倾, 置信度=0.88, 关键点=14/17
...
```

#### 全面分析模式上下文示例
```
【视频全面分析结果】
- 视频时长: 108.0秒
- 人体出现比例: 85.2%
- 主要姿态: 跪姿/蹲姿, 弯腰/前倾
- 使用模型: detect, pose, segment
- 平均投票数: 2.8/3
- 姿态分布: 跪姿/蹲姿(45次), 弯腰/前倾(32次), 站立/正常姿态(18次)
- 姿态变化次数: 8
  * 5.2s: 站立/正常姿态 -> 弯腰/前倾
  * 12.8s: 弯腰/前倾 -> 跪姿/蹲姿
- 人体出现时间段:
  * 2.0s - 106.0s
- 平均置信度: 0.87
- 平均可见关键点: 12.3/17
- 穿着色彩变化: 45.2

【各帧详细分析（图片序号对应下方描述）】
- 图1@2.0s: [检测]置信度=0.92 [姿态]站立/正常姿态, 关键点=15/17 [分割]置信度=0.89 穿着色彩变化=42.3 投票=3/3
- 图2@12.8s: [检测]置信度=0.88 [姿态]弯腰/前倾, 关键点=14/17 [分割]置信度=0.85 穿着色彩变化=48.7 投票=3/3
...
```

**字幕上下文去重**：多个帧落入同一字幕时间段时，合并为一条输出，避免重复发送相同字幕内容：
```
- 图1,2,3,4,5,6,7,8@4.0s-30.0s: [00:00:01 --> 00:00:31] 字幕内容...
- 图9@48.0s: [00:00:33 --> 00:00:59] 字幕内容...
```

### 使用示例

```powershell
# 基础模式：使用YOLO姿态估计
uv run title-classifier vision --use-yolo -p gcli

# 全面分析模式：使用三个YOLO模型
uv run title-classifier vision --use-yolo --comprehensive -p gcli

# 自定义采样间隔
uv run title-classifier vision --use-yolo --analysis-step 1.0 -p gcli

# 增加VLM帧数
uv run title-classifier vision --use-yolo --vlm-frames 15 -p gcli
```

### 输出示例

**关键词格式**（水印博主名最优先）：
```
标签1，标签2……
```

**final_name格式**：
```
[标签_标签1]_原文件名.mp4
```

---

## OpenVINO CPU加速

### 为什么需要 OpenVINO？

默认情况下，YOLO 使用 PyTorch 进行推理。在 CPU 上，PyTorch 的性能较差：
- Python 层开销大
- 未针对 CPU 指令集优化
- 内存碎片严重

**OpenVINO** 是 Intel 开源的推理引擎，专门针对 CPU 优化：
- 利用 AVX2/AVX-512/VNNI 等指令集
- 模型图优化和算子融合
- 内存布局优化

### 性能对比

| 后端 | 推理速度 | 精度 | 适用场景 |
|------|---------|------|---------|
| PyTorch (FP32) | 1x | 最高 | GPU 或调试 |
| OpenVINO (FP16) | 2-3x | 极小损失 | **CPU 推荐** |
| OpenVINO (INT8) | 3-5x | 较小损失 | 极致性能 |

### 使用方法

#### 自动模式（推荐）

首次运行时自动导出 OpenVINO FP16 模型，后续直接加载缓存：

```powershell
# 自动检测 OpenVINO，CPU 时默认使用
uv run title-classifier vision --use-yolo -p gcli
```

#### 指定后端

```powershell
# 强制使用 OpenVINO（默认）
uv run title-classifier vision --use-yolo --backend openvino -p gcli

# 强制使用 PyTorch
uv run title-classifier vision --use-yolo --backend pytorch -p gcli

# 自动检测后端
uv run title-classifier vision --use-yolo --backend auto -p gcli
```

#### 配置文件

在 `config/default.toml` 中配置：

```toml
[yolo]
model_type = "pose"      # detect / pose / segment
confidence = 0.5         # 置信度阈值（0.1-0.9）
backend = "openvino"     # auto / openvino / pytorch

[yolo.openvino]
precision = "FP16"  # FP16 / INT8
cache_dir = "models/yolo/openvino"
```

### 模型缓存

首次运行后，OpenVINO 模型会缓存到 `models/yolo/openvino/` 目录：

```
models/yolo/openvino/
├── detect_openvino_model/
│   ├── yolov8n.xml    # 模型定义
│   ├── yolov8n.bin    # 模型权重
│   └── metadata.yaml
├── pose_openvino_model/
│   └── ...
└── segment_openvino_model/
    └── ...
```

删除缓存目录后，下次运行会重新导出。

### INT8 量化（高级）

INT8 量化需要校准数据（200-500 张图片），可以进一步提升性能：

```bash
# 从视频中抽取校准帧
python scripts/convert_yolo_openvino.py --model pose --calibrate-dir test/ --num-frames 500

# 转换所有模型
python scripts/convert_yolo_openvino.py --model all --calibrate-dir test/ --num-frames 500
```

转换完成后，在配置中切换精度：

```toml
[yolo.openvino]
precision = "INT8"
```

> **注意**：INT8 量化会导致轻微精度损失，建议先用 FP16 验证效果。

---

## 运动检测前置过滤

### 为什么需要运动检测？

视频中很多画面是静止的（如监控、对话场景），对这些帧进行 YOLO 推理是浪费。

**运动检测**可以在 YOLO 推理前判断画面是否有变化：
- 静止帧 → 复用上一帧结果，跳过 YOLO 推理
- 运动帧 → 正常执行 YOLO 推理

### 性能提升

| 场景 | 静止帧比例 | 推理次数减少 |
|------|-----------|-------------|
| 监控视频 | 60-80% | 60-80% |
| 对话场景 | 30-50% | 30-50% |
| 动作片 | 5-15% | 5-15% |

### 使用方法

#### 默认启用

运动检测默认启用，无需额外配置：

```powershell
uv run title-classifier vision --use-yolo -p gcli
```

#### 禁用运动检测

```powershell
# 禁用运动检测（每帧都执行 YOLO 推理）
uv run title-classifier vision --use-yolo --no-motion-detection -p gcli
```

#### 调整阈值

```powershell
# 降低阈值（更敏感，更多帧会被判定为运动）
uv run title-classifier vision --use-yolo --motion-threshold 2.0 -p gcli

# 提高阈值（更不敏感，更多帧会被跳过）
uv run title-classifier vision --use-yolo --motion-threshold 10.0 -p gcli
```

#### 配置文件

在 `config/default.toml` 中配置：

```toml
[vision]
motion_detection = true
motion_threshold = 5.0       # 变化像素比例阈值（%）
motion_min_interval = 5.0    # 最小强制推理间隔（秒），需大于采样间隔
```

### 运动检测原理

1. **帧差法**：计算相邻帧的像素差异
2. **高斯模糊**：降噪，减少压缩伪影影响
3. **二值化**：超过阈值的像素标记为变化
4. **变化比例**：计算变化像素占总像素的百分比
5. **判断**：变化比例 > 阈值 → 有运动

### 防止误判

- **motion_min_interval**：即使画面静止，也每 N 秒强制执行一次 YOLO 推理
- 防止视频跳转后复用旧结果
- 默认值：5.0 秒（需大于采样间隔才能触发跳帧）

### 输出统计

运动检测会在日志中输出统计信息：

```
运动检测统计: 跳过 45 帧静止画面，节省 45 次YOLO推理
```

视频摘要中也会包含运动检测信息：

```
- 运动检测: 跳过45帧静止画面 (YOLO推理5次)
```

---

## 硬件视频解码

### 为什么需要硬件解码？

视频解码是 CPU 密集型操作。使用硬件解码可以：
- **降低 CPU 占用**：将解码工作从 CPU 转移到 GPU/专用解码器
- **提升解码速度**：硬件解码通常比软解快 2-5 倍
- **释放 CPU 资源**：让 CPU 专注于 YOLO 推理

### 支持的硬件解码器

| 平台 | 解码器 | 说明 |
|------|--------|------|
| Windows | d3d11va | Direct3D 11 Video Acceleration（推荐） |
| Windows | dxva2 | DirectX Video Acceleration 2 |
| Windows | qsv | Intel Quick Sync Video |
| Linux | vaapi | Video Acceleration API |
| Linux | qsv | Intel Quick Sync Video |
| macOS | videotoolbox | Apple VideoToolbox |

### 使用方法

#### 自动模式（推荐）

系统会自动检测可用的硬件解码器，失败时回退到软解：

```powershell
# 自动检测硬件解码器
uv run title-classifier vision --use-yolo -p gcli
```

#### 配置文件

在 `config/default.toml` 中配置：

```toml
[vision.decode]
hw_accel = "auto"      # auto / d3d11va / qsv / vaapi / none
batch_extract = true   # 批量提取帧
```

- `auto`：自动检测（默认）
- `none`：禁用硬件解码，强制使用软解
- 其他值：指定解码器名称

### 批量帧提取

默认启用批量帧提取，单次 ffmpeg 调用提取多个帧：

| 模式 | ffmpeg 调用次数 | 启动开销 |
|------|----------------|---------|
| 逐帧提取 | N 次 | N x ~100ms |
| 批量提取 | 1 次 | ~100ms |

对于 50 帧的视频，批量提取可减少约 5 秒的启动开销。

---

## 多线程并行处理

### 为什么多线程对 YOLO 推理有效？

Python 的全局解释器锁（GIL）通常被认为是多线程的瓶颈，特别是对于 CPU 密集型任务。但 **YOLO 推理实际上可以充分利用多线程并行**，原因是：

#### 1. PyTorch/OpenVINO 底层释放 GIL

YOLO 推理的核心计算（矩阵运算、卷积等）是由 C++ 实现的，执行时会释放 Python GIL：

```
Python 代码 → 受 GIL 限制
    ↓
PyTorch C++ 扩展 → 释放 GIL，真正并行
    ↓
OpenVINO 推理引擎 → 释放 GIL，真正并行
```

#### 2. 混合 I/O 和计算

视频处理流程中混合了多种操作：

| 操作类型 | 是否受 GIL 限制 | 示例 |
|---------|----------------|------|
| I/O 操作 | ❌ 释放 GIL | ffmpeg 帧提取、文件读写、API 调用 |
| C++ 扩展 | ❌ 释放 GIL | NumPy、OpenCV、PyTorch、OpenVINO |
| Python 代码 | ✅ 受 GIL 限制 | 数据处理、逻辑判断 |

#### 3. 实测验证

```python
# 测试 YOLO 推理的多线程加速比
单线程: 0.02s
多线程: 0.01s
加速比: 4.89x  # 接近 CPU 核心数
```

### 使用方法

```powershell
# 默认 4 线程并发处理
uv run title-classifier vision --use-yolo -p gcli

# 自定义并发数（匹配 CPU 核心数）
uv run title-classifier vision --use-yolo --concurrent 8 -p gcli
```

### 最佳实践

| CPU 核心数 | 推荐并发数 | 说明 |
|-----------|-----------|------|
| 4 核 | 3-4 | 留 1 核给系统 |
| 8 核 | 6-7 | 留 1-2 核给系统 |
| 16 核 | 12-14 | 留 2-4 核给系统 |

### 内存需求

每个并发线程共享同一个 YOLO 模型实例，内存占用约为：

- **OpenVINO FP16**: ~200MB（3 个模型）
- **PyTorch FP32**: ~500MB（3 个模型）

4 线程并发时，总内存占用约为 1-2GB。

---

## 音频字幕集成

### 功能说明

音频字幕功能使用 Silero VAD 进行语音活动检测，通过三层策略生成最优分段，然后调用 MiMo API 进行语音转录，生成 SRT 字幕文件。

### 核心特性

- **VAD语音检测**：基于深度学习的 Silero VAD，能区分人声与噪音/音乐
- **三层分段策略**：微合并→语义打包→静音过滤，消除碎片化
- **API拒绝自动重试**：被拒绝的长段自动按10秒切片重试
- **字幕后处理**：拆分长字幕、过滤无效内容（时间戳列表、拒绝响应等）
- **日志系统**：所有处理日志同时显示在GUI运行日志框和控制台

### 处理流程

```
视频文件
  ↓ ffmpeg 提取音频 (16kHz, 单声道, float32)
  ↓
Silero VAD 检测语音段
  ↓ 第一层：微合并（间隙 < 1.5秒合并）
  ↓ 第二层：语义打包（长停顿 > 3秒断开，最大块 25秒）
  ↓ 第三层：静音过滤（时长 < 1秒跳过，语音占比 < 30%跳过）
  ↓
逐段调用 MiMo API 语音转录
  ↓ 被拒绝的段按10秒切片自动重试
  ↓
字幕后处理（可选）
  ↓ 拆分长字幕、过滤无效内容
  ↓
生成 SRT 字幕文件
```

### 使用方法

**CLI方式**：
```powershell
# 独立运行音频识别
uv run title-classifier audio -p mimo

# 处理所有未识别文件
uv run title-classifier audio -p mimo --all
```

**GUI方式**：
1. 打开 "Stage1c 音频识别" 标签页
2. 配置参数（推荐使用默认值）
3. 点击 "音频识别" 按钮

### SRT文件格式

```srt
1
00:00:01,500 --> 00:00:05,200
你好世界，欢迎来到这个视频

2
00:00:05,200 --> 00:00:10,800
今天我们来聊一聊有趣的话题

3
00:00:10,800 --> 00:00:15,000
希望大家喜欢这个内容
```

### SRT文件命名

SRT文件名与原视频文件同名：
```
原视频：my_video.mp4
字幕：  my_video.srt
```

### 配置参考

完整配置项见 `config/default.toml`：

```toml
[audio]
skip_silence = true
volume_threshold = 0.01

[audio.vad]
enabled = true
min_speech_ms = 250
min_silence_ms = 350
merge_gap = 1.5
min_keep_duration = 1.0
max_chunk = 25.0
long_gap = 3.0
min_duration = 1.0
min_speech_ratio = 0.3

[audio.postprocess]
enabled = true
max_subtitle_duration = 10
max_subtitle_chars = 100
filter_invalid = true
format_text = true
```

---

## VAD分段策略

### 概述

VAD（Voice Activity Detection）分段使用 Silero VAD 模型检测语音活动，通过三层策略将音频切分为最优的语音块，适配多模态模型的输入窗口。

### 第一层：微合并

将 VAD 检测到的细粒度语音段进行初步合并：

```
VAD原始: [1.0-1.5] [1.7-2.3] [2.5-3.1] [5.0-6.2]
           ↑间隙0.2s↑间隙0.2s↑间隙1.9s↑
           ↓ 合并（间隙 < 0.8秒）
合并后:  [1.0-3.1] [5.0-6.2]
```

- 合并阈值：`merge_gap`（默认0.8秒）
- 最小保留时长：`min_keep_duration`（默认1.0秒）

### 第二层：语义打包

将微合并后的语音段打包成适合模型的块：

```
微合并后: [1.0-8.0] [10.5-12.0] [15.0-18.0]
           ↑7.0秒    ↑1.5秒      ↑3.0秒
           ↓ 语义打包
打包后:  [1.0-12.0] [15.0-18.0]
         ↑长停顿断开（间隙 > 2秒）
```

- 最大块时长：`max_chunk`（默认25秒）
- 长停顿阈值：`long_gap`（默认2秒）

### 第三层：静音过滤

过滤掉不值得发送给模型的块：

| 过滤规则 | 条件 | 说明 |
|----------|------|------|
| 时长过短 | < `min_duration`(1秒) | 大概率是咳嗽/气声 |
| 语音占比低 | < `min_speech_ratio`(40%) | 环境噪音为主 |

### API拒绝自动重试

当某个语音块被API拒绝转录时（通常是时长过长导致），自动按10秒切片重试：

```
[18/32] 处理区块: 268.60s-288.50s (19.90秒)
[18/32] API拒绝转录: The request was rejected...
[18/32] 重试子块1: 268.60s-278.60s (10.00秒)
[18/32] 子块1识别成功: 268.60s-278.60s
[18/32] 重试子块2: 278.60s-288.50s (9.90秒)
[18/32] 子块2仍失败，跳过
```

- 按10秒为单位切片
- 每片独立提取音频并调用API
- 成功的子块写入SRT，失败的子块跳过
- 只拆分一次，不递归重试

---

## SQLite 数据库

### 功能说明

v7.5.0 新增 SQLite 数据库存储媒体元数据，支持标签索引、搜索、改动历史记录。数据库与视频文件分离，便于迁移和云端查看。

### 数据库位置

```
data/
├── media.db          # SQLite 数据库
├── covers/           # VLM 图片库（按视频 ID 分目录）
│   ├── 1/
│   │   ├── frame_000.jpg
│   │   └── ...
│   └── ...
└── output/           # CSV 输出
```

### CLI 命令

```powershell
# 初始化数据库
title-classifier db init

# 从 CSV 导入数据
title-classifier db import --csv "data/output/love/title_review.csv"

# 导入所有 CSV
title-classifier db import --all

# 列出记录
title-classifier db list --limit 20

# 搜索
title-classifier db search --query "关键词"
title-classifier db search --tag "tag_name"

# 查看单条记录
title-classifier db show <media_id>

# 查看改动历史
title-classifier db history <media_id>

# 统计信息
title-classifier db stats
```

### 数据库表结构

| 表名 | 说明 |
|------|------|
| `media_files` | 媒体文件主表（路径、标题、描述、状态等） |
| `tags` | 标签表（从 vision_keywords 自动提取） |
| `media_tags` | 媒体-标签关联表 |
| `change_log` | 改动记录表（跟踪所有修改） |
| `vlm_frames` | VLM 帧表（记录 VLM 图片路径） |

### 实时同步

GUI 所有操作自动同步到数据库：
- Stage1 扫描 → 写入新文件
- Stage1b AI优化 → 更新 final_name/keywords
- Stage1c 音频识别 → 更新 audio_recognized/srt_path
- Stage1c 视觉识别 → 更新 description/keywords/flags
- Stage2 重命名 → 更新 current_path

### 去重策略

- **Level 1**: `original_title` 精确匹配 + `file_size` ±1% 区间验证
- **Level 2**: `file_size` ±1% + `duration` ±1s 区间匹配（兜底）
- 匹配到已有记录时更新路径，保留旧数据（描述、关键词、标签）
- 抛弃路径匹配，避免外置硬盘盘符变化导致重复记录

### 视频元数据收集

扫描阶段自动收集：
- `file_size`：`os.path.getsize()`（毫秒级）
- `duration`：ffprobe → cv2 备用（秒级）
- `resolution`：ffprobe → cv2 备用（如 1920x1080）

视觉识别阶段补充：
- `file_size`：如果扫描时未收集
- `duration`：如果扫描时未收集

---

## AI Provider 配置

### 内置 Provider

| Provider | 默认模型 | 环境变量 | 支持阶段 |
|----------|---------|---------|----------|
| `ollama` | qwen2.5:7b-instruct-q4_K_M | - | 1b, 1c |
| `zhipu` | GLM-4.7-Flash | ZHIPU_API_KEY | 1b, 1c |
| `gcli` | gemini-3-flash-preview | GCLI_API_KEY | 1b, 1c |
| `mimo` | mimo-v2.5 | MIMO_API_KEY | 1c, audio |

### 自定义 Provider

创建 `config/providers.json` 文件添加自定义 Provider：

```json
{
  "my_provider": {
    "name": "我的Provider",
    "type": "multi",
    "url": "https://api.example.com/v1/chat/completions",
    "env_key": "MY_API_KEY",
    "default_model": "my-model",
    "requires_api_key": true,
    "supports_1b": true,
    "supports_1c": true,
    "supports_audio": false,
    "description": "自定义Provider描述"
  }
}
```

---

## CPU 多线程推理

### 为什么不用 CUDA

本项目默认使用 **CPU 版 PyTorch**（~250MB），不再推荐 CUDA 版（~2.4GB）。原因：

| 问题 | 说明 |
|------|------|
| **uv lock 冲突** | `uv run` 会自动同步 lockfile，将 CUDA 版覆盖回 CPU 版，每次需加 `--no-sync` |
| **依赖膨胀** | CUDA 版 PyTorch ~4.4GB，CPU 版 ~250MB，差 18 倍 |
| **并发限制** | CUDA 不支持多线程并发推理，必须串行执行（`_gpu_lock`） |
| **多任务死锁** | 不能同时运行两个 CUDA 模式的视觉识别任务 |
| **收益有限** | YOLO 单帧推理 200ms→20ms，但 VLM 调用（云端 API）才是瓶颈 |

**CPU 多核并行的优势**：20 核 CPU 可同时处理 4 个视频（`--concurrent 4`），总吞吐量反而更高。

### 使用方式

**CLI：**
```bash
# 默认 CPU 模式
uv run title-classifier vision --use-yolo -p gcli

# 指定并发数（推荐 4）
uv run title-classifier vision --use-yolo --concurrent 4 -p gcli
```

**GUI：**
```bash
uv run title-classifier gui
```

视觉识别标签页的"推理设备"下拉框选择 cpu（默认）。

### 并发策略

| 模式 | YOLO推理 | VLM调用 | 并发数 |
|------|----------|---------|--------|
| cpu（默认） | CPU多核并行 | 并发 | min(cores-1, 4) |

- **CPU模式**：利用多核CPU并行推理，多个视频同时处理
- **并发数**：默认 `min(cores-1, 4)`，GUI 中可手动调整

### 多任务并行

可以同时运行多个实例处理不同 CSV：

```bash
# 终端1：GUI（处理CSV1）
uv run title-classifier gui

# 终端2：CLI（处理CSV2）
uv run title-classifier vision -c "data/output/Movies/title_review.csv" --use-yolo --concurrent 4 -p gcli
```

### VLM失败处理

当VLM API调用失败时（超时、限流、空响应），系统会：

1. **重试一次**：同样帧数，同样参数
2. **仍失败则标记**：CSV中 `vision_failed` 列设为 `true`
3. **跳过失败行**：默认视觉识别会跳过 `vision_failed=true` 的行

**CLI重试失败行：**
```bash
uv run --no-sync title-classifier vision -c "data/output/Download/title_review.csv" --use-yolo --retry-failed -p gcli
```

**GUI重试失败行：**
- 视觉识别标签页的"重试失败行"按钮
- 只处理 `vision_failed=true` 的行
- 成功后自动清除失败标记

### 设备检测逻辑

- `auto`（默认）：使用 CPU
- `cpu`：强制 CPU（推荐）
- `cuda`：强制 GPU，CUDA 不可用时回退到 CPU（需要手动安装 CUDA 版 PyTorch）

---

## 项目结构

```
title-classifier/
├── src/
│   └── title_classifier/
│       ├── __init__.py              # 版本信息
│       ├── __main__.py              # CLI入口
│       │
│       ├── core/
│       │   ├── scanner.py           # 文件扫描
│       │   ├── refiner.py           # AI优化
│       │   ├── vision.py            # 视觉识别（VLM + YOLO + 字幕上下文）
│       │   ├── renamer.py           # 重命名
│       │   ├── db_schema.sql        # SQLite 表结构定义
│       │   └── db_store.py          # SQLite 数据库访问层
│       │
│       ├── detectors/
│       │   ├── base.py              # 检测器基类
│       │   ├── yolo.py              # YOLO检测
│       │   └── clip.py              # CLIP分类
│       │
│       ├── providers/
│       │   ├── __init__.py          # Provider管理
│       │   └── base.py              # Provider基类
│       │
│       ├── utils/
│       │   ├── video.py             # 视频工具（get_video_info, 帧提取）
│       │   ├── image.py             # 图片工具
│       │   ├── audio.py             # 音频处理（VAD分段 + API调用）
│       │   ├── atomic_csv.py        # 原子化CSV读写（崩溃安全）
│       │   ├── config.py            # 配置加载（TOML）
│       │   ├── file_resolve.py      # 文件路径解析（Stage2重命名后回退查找）
│       │   ├── muxer.py             # 字幕封装（SRT嵌入视频）
│       │   ├── subtitle_postprocessor.py  # 字幕后处理
│       │   ├── prompt_loader.py     # 提示词加载
│       │   └── stats.py             # 标签统计
│       │
│       └── gui/
│           ├── app.py               # 图形界面
│           └── debug_window.py      # 调试窗口
│
├── config/
│   ├── default.toml                 # 默认配置
│   ├── providers.json               # 自定义Provider配置
│   └── providers.example.json       # Provider配置示例
│
├── models/
│   ├── clip/                        # CLIP模型（自动下载）
│   └── yolo/                        # YOLO模型（自动下载）
│
├── scripts/
│   ├── README.md                  # 脚本说明文档
│   ├── output/                    # 脚本产生的日志/报告
│   │
│   ├── full_workflow.py           # 完整工作流（扫描→音频→视觉→封装→确认→重命名）
│   ├── workflow_common.py         # 工作流公共模块
│   ├── workflow_scan_audio.py     # 扫描+音频识别
│   ├── workflow_vision.py         # 扫描+视觉识别
│   ├── workflow_reclassify.py     # 强制重分类
│   ├── workflow_rename.py         # 确认+重命名
│   ├── workflow_mux.py            # 字幕封装
│   │
│   ├── fix_bracket_only.py        # 修CSV final_name：[标题]→标题
│   ├── fix_bracket_filenames.py   # 修磁盘文件名+CSV：[标题].mp4→标题.mp4
│   ├── fix_csv_paths.py           # 修CSV original_path：去掉[关键词]_前缀
│   ├── fix_all_csvs.py            # 批量对所有CSV执行fix_bracket_only
│   │
│   ├── download_models.py         # 一键下载所有模型
│   ├── download_clip.py           # 下载CLIP模型
│   ├── download_yolo_models.py    # 下载YOLO模型
│   ├── import_csv.py              # CSV导入到数据库
│   └── test_prompts.py            # 测试prompt效果
│
├── tests/                           # 测试文件
├── test/                            # 测试数据
├── data/
│   ├── media.db                     # SQLite 数据库
│   ├── covers/                      # VLM 图片库
│   │   └── <video_id>/              # 按视频 ID 分目录
│   │       ├── frame_000.jpg        # 发送给 VLM 的帧
│   │       └── ...
│   ├── output/                      # 输出目录
│   │   └── <目录名>/                # 每个目标目录独立子目录
│   │       ├── title_review.csv     # 待审表
│   │       ├── subtitles/           # SRT字幕目录
│   │       └── workflow.log         # 工作流日志
│   └── debug/                       # 调试数据
│
├── pyproject.toml
└── .env
```

---

## 常见问题

### Q1：YOLO模型下载失败

```powershell
# 使用国内镜像下载
uv run python scripts/download_yolo_models.py
```

### Q2：CSV 在 Excel 中打开乱码

1. 用 VS Code 打开 CSV
2. 右下角点击编码 -> "通过编码保存"
3. 选择 "UTF-8 with BOM"



### Q4：SRT文件名格式不对

SRT文件名使用 final_name 格式，确保先运行 vision 命令生成 final_name。

### Q5：如何撤销重命名？

```powershell
# 执行前备份目录
robocopy "F:\Videos" "F:\Videos_Backup" /E /NFL /NDL /NJH /NJS
```

### Q6：音频识别被API拒绝

长音频段可能被API拒绝转录。当前版本已内置自动重试机制：被拒绝的段会按10秒切片重试，成功的子块写入SRT，失败的跳过。

### Q7：VAD切分太碎/太粗

调整 `config/default.toml` 中的参数：
- 切分太碎：增大 `merge_gap`（如1.0秒）
- 切分太粗：减小 `max_chunk`（如20秒）或 `long_gap`（如1.5秒）

### Q8：字幕后处理如何禁用

在 GUI 的 "Stage1c 音频识别" 标签页取消勾选 "字幕后处理"，或在 `config/default.toml` 中设置：

```toml
[audio.postprocess]
enabled = false
```

### Q9：如何并行处理多个目录？

使用工作流脚本，每个目录自动分配独立的 CSV 和日志：

```powershell
# 窗口1
python scripts/full_workflow.py "D:/aria2/love"

# 窗口2（同时运行）
python scripts/full_workflow.py "D:/aria2/anime"
```

每个目录的输出在 `data/output/<目录名>/` 下，互不干扰。

### Q10：CSV 写入时崩溃会丢数据吗？

不会。v7.4.0 起所有 CSV 写入使用原子化操作：先写入临时文件，再用 `os.replace()` 原子替换原文件。即使崩溃，原文件仍完好。

### Q11：数据库文件在哪里？如何备份？

数据库文件在 `data/media.db`，VLM 图片在 `data/covers/`。备份时复制这两个位置即可。数据库使用 WAL 模式，支持并发读取。

### Q12：如何查看数据库中的数据？

```powershell
# 使用 CLI 命令
title-classifier db list
title-classifier db search --query "关键词"
title-classifier db stats

# 或使用数据库浏览器打开 data/media.db
```

### Q13：强制重分类后，视觉识别会产生嵌套中括号吗？

**不会**。代码有防嵌套保护。

**流程说明**：
```
1. 原始文件: [旧关键词]_原始标题.mp4
2. --force 扫描后 CSV:
   original_title = "原始标题.mp4"  ← 干净标题（已剥离中括号）
3. 视觉识别 generate_final_name():
   # 安全兜底：如果 original_title 仍带有 [kw]_ 前缀，剥离
   clean_title = re.sub(r"^\[[^\]]*\]_?", "", original_title)
   return f"[{新关键词}]_{clean_title}"
4. 重命名后: [新关键词]_原始标题.mp4
```

**关键代码** (`vision.py:1299-1303`):
```python
# 安全兜底：如果 original_title 仍带有 [kw]_ 前缀，剥离
clean_title = re.sub(r"^\[[^\]]*\]_?", "", original_title, count=1)
prefix = "_".join(kw_list)
return f"[{prefix}]_{clean_title}"
```

### Q14：CLIP 模型的作用是什么？

CLIP 模型作为**本地零样本多维分类器**：
- **三维分类**：服装、动作、发型
- **预分类过滤**：置信度高时直接使用本地结果，节省 API 调用
- **动态学习**：支持从云端 VLM 结果中吸收新标签

### Q15：如何搜索更好的 CLIP 模型？

**搜索关键词**：`NSFW CLIP model`、`cosplay CLIP fine-tuned`

**推荐平台**：HuggingFace、CivitAI、GitHub

### Q16：无意义标题检测规则？

| 规则 | 示例 |
|------|------|
| IMG/VID/DCIM 前缀 | `IMG_7940`、`VID_20240115` |
| T ***且无中文 | `T****@ciyuanb@-merged-...` |
| 纯 hex/hash | `25bdc148` |
| 纯数字/日期 | `20240115` |
| 短标题 | `视频`、`video (1)` |
| #tag 标题 | `#树木`（**不会**被标记） |

### Q17：如何提高处理速度？

1. 减少 VLM 帧数：`--vlm-frames 5`
2. 减少采样帧数：`--max-sample-frames 30`
3. 启用运动检测（默认开启）
4. 使用 OpenVINO 后端（默认开启）
5. 排除不需要的目录：`--exclude-dir temp`

### Q18：图片模式没有被处理？

**可能原因**：
1. 图片的 `needs_vision` 不是 `true`
2. 图片压缩失败
3. VLM API 调用失败

**解决方法**：
```powershell
# 处理所有未处理的文件
uv run title-classifier vision --all -p gcli
```

---

## 更新日志

### v8.1.0（当前版本）

**修复：YOLO 帧提取画面比例变形**

- ffmpeg 缩放滤镜修复：只缩放长边，保持原始画面比例
- 1280x720 视频 max_size=400: 400x400(变形) → 400x226(原比例)
- 修复正方形视频(1080x1080)不缩放的问题（gt → gte）
- 关键点检测精度显著提升

**修复：OpenVINO pose/segment 模型失效**

- ultralytics 加载 OpenVINO 模型时未指定 task 参数
- pose 和 segment 模型被当作 detect 模型加载，完全不输出关键点
- 修复: `YOLO(path)` → `YOLO(path, task=model_type)` 三处
- 已清理旧的 OpenVINO 缓存模型

**优化：姿态分析**

- 姿态规则人体归一化：用躯干长度替代硬编码像素阈值
- 新增姿态类别：手臂抬起、躺卧、双腿张开、弓背/驼背
- 双侧关键点平均：不再只用左侧关键点
- 最小躯干阈值：防止躯干过小时所有姿态同时触发
- 时序平滑：持续 ≥2 帧才算真正姿态变化

**优化：YOLO 置信度**

- 默认置信度 0.5 → 0.4
- GUI 新增 YOLO 置信度输入框（带范围校验 0.1-0.9）
- CLI `--yolo-conf` 默认值同步更新

**优化：VLM 提示词**

- 加强 system prompt：增加 file management tool 上下文
- 加强 system_header：明确 user-owned media 场景
- 加强 retry prompts：增加 pre-authorized 声明
- SSL 错误处理：SSL/EOF 类错误增加额外等待时间

**新增：CLIP 详细置信度输出**

- 新增 `clip_detail` 字段，存储每个维度 top-5 标签及置信度
- JSON 格式: `{"clothing": [{"label": "...", "confidence": 0.82}], ...}`

**新增：数据库去重改为原始文件名匹配**

- 新增 `find_match(original_title, file_size, duration)` 方法
- Level 1: original_title 精确匹配 + file_size ±1% 区间验证
- Level 2: file_size ±1% + duration ±1s 区间匹配
- 抛弃路径匹配，避免外置硬盘盘符变化导致重复记录
- 匹配到已有记录时更新路径，保留旧数据

**新增：扫描阶段收集视频元数据**

- 新增 `get_video_info()` 一次性获取 duration + resolution
- scanner 自动收集 file_size / duration / resolution
- 视觉识别时补充 file_size 和 duration 到数据库

**新增：调试窗口 VLM 帧预览**

- VLM 缩略图可点击，在主画布中预览大图
- 显示帧序号和时间戳（与 prompt 中图1@20.0s 对应）
- 独立 photo 引用列表，防止切换帧时缩略图消失

**新增：GUI 进度显示**

- 日志工具栏添加进度状态标签
- 解析 CLI 输出中的 [1/50] 格式实时更新

**修复：VLM 返回为空未标记失败**

- VLM 返回为空或关键词为空时，vision_failed 未标记为 true
- 修复后重试失败行功能正常工作

**修复：运动检测兼容灰度帧**

- 某些视频帧是单通道灰度图，cv2.cvtColor 崩溃
- 添加通道数检查，已灰度的帧直接使用

**修复：debug 窗口 crash**

- self.summary 未初始化就被使用导致 AttributeError

**优化：字幕封装 overwrite 模式备份机制**

- 覆写前先备份原文件为 .bak
- 替换成功后删除备份
- 替换失败时自动从备份恢复

**优化：运动检测默认值**

- motion_min_interval 2.0 → 5.0
- 使运动检测在默认采样间隔(2.0s)下生效

### v8.0.0

**新增：OpenVINO CPU 加速**

- 新增 OpenVINO 推理后端，CPU 推理速度提升 2-3x
- 首次运行自动导出 FP16 模型到 `models/yolo/openvino/`
- 后续运行直接加载缓存，无需重复导出
- 自动检测 OpenVINO 可用性，不可用时回退到 PyTorch
- 新增 `--backend` 参数：auto / openvino / pytorch

**新增：运动检测前置过滤**

- 新增帧差法运动检测，跳过静止画面的 YOLO 推理
- 静止帧复用上一帧结果，保持时间线完整
- 新增 `--motion-threshold` 参数：变化像素比例阈值（默认 5%）
- 新增 `--no-motion-detection` 参数：禁用运动检测
- 新增 `motion_min_interval` 配置：最小强制推理间隔（默认 2 秒）
- 监控等静态场景可减少 60-80% 无效推理

**新增：硬件视频解码**

- 新增 ffmpeg 硬件解码支持，自动检测最优解码器
- Windows: d3d11va / dxva2 / qsv
- Linux: vaapi / qsv
- macOS: videotoolbox
- 硬件解码失败时自动回退到软解
- 新增批量帧提取，单次 ffmpeg 调用提取多个帧，减少启动开销

**新增：INT8 量化校准脚本**

- 新增 `scripts/convert_yolo_openvino.py`：独立的 INT8 校准工具
- 从视频中抽取校准帧，生成 INT8 量化的 OpenVINO 模型
- 不影响主业务流程，可后续单独运行

**文档：多线程并行处理原理**

- 说明为什么多线程对 YOLO 推理有效（PyTorch/OpenVINO 底层释放 GIL）
- 解释混合 I/O 和计算的并行机制
- 提供最佳并发数和内存需求参考

**配置扩展**

- `[yolo]` 新增 `backend = "auto"`
- `[yolo.openvino]` 新增 `precision = "FP16"` 和 `cache_dir`
- `[vision]` 新增 `motion_detection`、`motion_threshold`、`motion_min_interval`
- `[vision]` 新增 `max_sample_frames = 50`（采样帧数上限）
- `[vision.decode]` 新增 `hw_accel` 和 `batch_extract`

**GUI 更新**

- 新增 YOLO 推理后端选择（auto/openvino/pytorch）
- 新增 运动检测开关和阈值配置
- 新增 最大采样帧数配置

### v7.6.0

**修复：Stage2重命名后文件查找**

- 新增 `resolve_media_path()` 三级回退查找，处理Stage2重命名后CSV中original_path指向旧文件名的问题
  - 第1级：original_path直接存在
  - 第2级：用final_name在同目录拼路径
  - 第3级：去掉`[关键词]_`前缀后按original_title stem搜索
- vision/audio/renamer/muxer 四个模块统一使用回退查找

**修复：Stage1b final_name格式统一**

- Stage1b确认写入格式从`[优化标题]`改为`[优化标题]_原始标题`，与Vision输出一致
- 原标题不变时不加格式（填入原标题操作保持原样）

**GUI Stage1b 批量操作**

- 新增批量按钮：选中行→需要视觉/不需要视觉/反选
- 新增全选/取消全选按钮
- AI优化进度条（实时显示百分比）

**Refiner 并发加速**

- Refiner改为并发批处理（ThreadPoolExecutor, 3线程并发）
- batch_size从5提升到10，速度约提升3倍

**数据修复脚本**

- 新增 `scripts/fix_bracket_only.py`：修CSV中纯中括号的final_name
- 新增 `scripts/fix_bracket_filenames.py`：修磁盘文件名+同步更新CSV
- 新增 `scripts/fix_csv_paths.py`：修CSV的original_path去掉`[关键词]_`前缀
- 新增 `scripts/fix_all_csvs.py`：批量对所有CSV执行修复

### v7.5.0

**新增：SQLite 数据库**

- 新增 `data/media.db` 数据库存储媒体元数据
- 表结构：`media_files`、`tags`、`media_tags`、`change_log`、`vlm_frames`
- 支持标签索引、搜索、改动历史记录
- VLM 帧保存到 `data/covers/<video_id>/`（仅首次）
- CLI 命令：`db init`、`db import`、`db list`、`db search`、`db show`、`db history`、`db stats`
- GUI 所有操作自动同步到数据库（扫描、音频识别、视觉识别、重命名）

**GUI 改进**

- 新增 CSV 状态栏 + 并发提示，实时显示当前 CSV 路径
- 日志区域改为 `ttk.PanedWindow`，可拖拽调整大小
- 修复 CSV 路径同步：使用 `trace_add` 实时同步各标签页 CSV 变量
- 修复 `_on_tab_changed`：切换标签页不再覆盖所有标签页的 CSV 路径

**字幕封装改进**

- 修复 `_get_output_path`：覆盖模式下使用不同的临时文件名，避免 ffmpeg "cannot edit in-place" 错误
- 安全覆盖策略：原始文件→备份→临时文件→原位置，三步安全替换
- 新增文件大小验证：防止覆盖时替换为损坏的小文件
- 改进错误日志：覆盖失败时记录完整错误信息和 ffmpeg 返回码

### v7.4.0

**稳定性：原子化 CSV 写入**

- 所有 CSV 写入操作改为原子化：写入临时文件 → `os.replace()` 原子替换
- 崩溃/断电不再导致 CSV 损坏或数据丢失
- 涉及 8 个写入点：scanner、cmd_audio、cmd_vision、gui/app.py (4处)、full_workflow.py

**VLM 空关键词重试**

- 视觉识别返回空关键词时，自动用强调格式的 prompt 重试一次
- 适用于视频全面分析模式、传统模式、图片模式
- 增大 `max_tokens` 1024→2048，防止响应截断导致关键词丢失

**VAD 参数调优**

- 更宽松的语音分段，短停顿不再切断语音：
  - `min_silence_ms`: 80→350ms（VAD 判定静音的最短时长）
  - `merge_gap`: 0.8→1.5s（微合并间隙阈值）
  - `long_gap`: 2.0→3.0s（语义打包长停顿阈值）
  - `min_speech_ms`: 150→250ms（过滤极短语音碎片）
  - `min_speech_ratio`: 0.4→0.3（最低语音占比）

**Scanner 括号前缀剥离**

- `--force` 扫描时自动去除文件名的 `[关键词]_` 前缀
- 以干净文件名重新判断 `needs_vision`，防止嵌套括号
- `generate_final_name()` 添加安全兜底，防止双重括号

**Per-directory 输出**

- 目录扫描自动输出到 `data/output/<目录名>/title_review.csv`
- 单文件扫描保持默认 `data/output/title_review.csv`
- 支持并行处理多个目录，互不干扰

**GUI Stage1b 改进**

- "确认写入CSV" 只写已修改行的 final_name，未修改行不受影响
- needs_vision/audio_recognized 切换即时写入 CSV，不需要点"确认写入"
- 右键菜单按功能分组，动态显示当前值和切换方向
- 修改行高亮（浅蓝色背景），一眼区分修改/未修改
- 新增"重置为原标题"菜单项，取消优化不标记为修改

**工作流脚本**

- 提取公共模块 `workflow_common.py`
- 新增专用脚本：
  - `workflow_scan_audio.py`：扫描 + 音频识别
  - `workflow_vision.py`：扫描 + 视觉识别
  - `workflow_reclassify.py`：强制重分类（剥离括号前缀）
  - `workflow_rename.py`：确认 + 重命名
  - `workflow_mux.py`：字幕封装
- `full_workflow.py` 重构为使用共享模块

**日志修正**

- 视觉识别日志准确区分 "YOLO基础模式" 和 "YOLO全面分析模式"
- 根据实际模型数量显示，不再误导用户

### v7.3.0

**优化：分区段帧选择策略**

- **均匀覆盖**：将采样帧等分为 vlm_frames 个区段，每段内独立选最优帧，保证全视频均匀覆盖
- 解决旧版全局 top-N 选帧导致后半段视频因置信度低被忽略的问题
- 区段内评分规则不变（置信度40% + 关键点30% + 姿态变化30%），无人体帧取段内中间帧
- 同时适用于基础模式和全面分析模式

**优化：字幕上下文去重**

- 多个帧落入同一字幕时间段时合并显示，避免重复发送相同字幕内容给 VLM
- 减少 VLM prompt 长度，降低 token 消耗

### v7.2.0

**新增功能：全面分析模式**

- **多模型支持**：全面分析模式现在使用三个YOLO模型（detect、pose、segment）
- **投票决策**：至少两个模型检测到人体才认为有人体，提高准确性
- **动态权重**：根据每个模型的置信度自动调整权重
- **详细上下文**：分别标注三个模型的结果来源传给VLM
- **穿着分析**：segment模型提供人体区域掩码和穿着色彩分析

**新增功能：字幕封装**

- **字幕封装模块**（新增 `muxer.py`）：
  - 将SRT字幕封装到视频容器中（MKV/MP4）
  - 自动检测字幕语言（中文/英文/日文/韩文）
  - 自动命名轨道并设置为默认轨道
  - 支持批量封装和重试失败操作
- **GUI集成**：
  - 在"Stage1c 视觉识别"标签页新增"字幕封装"配置区域
  - 提供封装开关、输出格式、文件处理方式等选项
  - 显示封装进度和状态
  - 支持重试失败的封装操作
- **配置选项**：
  - 新增 `[mux]` 配置节，支持自动封装、输出格式、文件处理等配置
  - 支持根据源视频格式自动选择容器
  - 支持创建新文件或覆盖原文件

**重大更新：音频处理系统重构**

- **VAD三层分段策略**：微合并→语义打包→静音过滤，替代旧的能量阈值分段
  - 第一层：间隙 < 0.8秒的相邻语音段合并，消除换气/停顿碎片
  - 第二层：按长停顿（>2秒）断开，打包成适合模型的块（最大25秒）
  - 第三层：过滤时长 < 1秒或语音占比 < 40%的低质量块
- **API拒绝自动重试**：被拒绝的长音频段按10秒切片自动重试
- **字幕后处理模块**（新增 `subtitle_postprocessor.py`）：
  - 拆分长字幕为多个短字幕
  - 过滤无效内容（时间戳列表、拒绝响应、分析报告等）
  - 格式化说话人标签
- **VLM字幕上下文**：视觉识别时自动匹配每帧对应的字幕时间段，增强VLM理解

**GUI改进**

- 新增 "Stage1c 音频识别" 标签页，独立配置VAD参数和字幕后处理
- 音频处理日志重定向到GUI运行日志框（与视觉识别一致）
- 移除视觉识别标签页的冗余音频配置区域
- 新增视觉识别调试窗口（`debug_window.py`）

**配置变更**

- 新增 `[audio.vad]` 配置节：`merge_gap`、`min_keep_duration`、`max_chunk`、`long_gap`、`min_duration`、`min_speech_ratio`
- 新增 `[audio.postprocess]` 配置节：`max_subtitle_duration`、`max_subtitle_chars`、`filter_invalid`、`format_text`
- 移除旧的 `[audio.adaptive]` 配置节（已被VAD策略替代）

**其他改进**

- scan 命令支持单个文件路径
- 修复配置文件路径计算错误
- 新增 `clean_transcription_text()` 清理API返回的非中文标注

### v7.1.0

**新增功能：字幕封装**

- **字幕封装模块**（新增 `muxer.py`）：
  - 将SRT字幕封装到视频容器中（MKV/MP4）
  - 自动检测字幕语言（中文/英文/日文/韩文）
  - 自动命名轨道并设置为默认轨道
  - 支持批量封装和重试失败操作
- **GUI集成**：
  - 在"Stage1c 视觉识别"标签页新增"字幕封装"配置区域
  - 提供封装开关、输出格式、文件处理方式等选项
  - 显示封装进度和状态
  - 支持重试失败的封装操作
- **配置选项**：
  - 新增 `[mux]` 配置节，支持自动封装、输出格式、文件处理等配置
  - 支持根据源视频格式自动选择容器
  - 支持创建新文件或覆盖原文件

**其他改进**

- 更新模型下载说明，新增一键下载脚本
- 修复 gitignore，保留 providers.json 配置

### v7.0.0

**重大更新：音频处理系统重构**

- **VAD三层分段策略**：微合并→语义打包→静音过滤，替代旧的能量阈值分段
  - 第一层：间隙 < 0.8秒的相邻语音段合并，消除换气/停顿碎片
  - 第二层：按长停顿（>2秒）断开，打包成适合模型的块（最大25秒）
  - 第三层：过滤时长 < 1秒或语音占比 < 40%的低质量块
- **API拒绝自动重试**：被拒绝的长音频段按10秒切片自动重试
- **字幕后处理模块**（新增 `subtitle_postprocessor.py`）：
  - 拆分长字幕为多个短字幕
  - 过滤无效内容（时间戳列表、拒绝响应、分析报告等）
  - 格式化说话人标签
- **VLM字幕上下文**：视觉识别时自动匹配每帧对应的字幕时间段，增强VLM理解

**GUI改进**

- 新增 "Stage1c 音频识别" 标签页，独立配置VAD参数和字幕后处理
- 音频处理日志重定向到GUI运行日志框（与视觉识别一致）
- 移除视觉识别标签页的冗余音频配置区域
- 新增视觉识别调试窗口（`debug_window.py`）

**配置变更**

- 新增 `[audio.vad]` 配置节
- 新增 `[audio.postprocess]` 配置节
- 移除旧的 `[audio.adaptive]` 配置节（已被VAD策略替代）

### v8.2.0 (最新)

**新增：日志系统和耗时追踪**
- 日志默认输出到 `logs/<日期>/` 目录，按天分目录，文件始终记录 DEBUG 级别
- 视觉处理分步耗时追踪：YOLO 推理、CLIP 差异度、帧选择、VLM API
- YOLO 详情日志：每帧 detect/pose/segment 推理耗时、置信度、关键点数
- CLIP 详情日志：分类耗时、穿着/动作/发型标签和置信度、差异度分数统计
- VLM API 日志：请求耗时、token 用量（prompt+completion+total）
- Debug 模式保存 `timing.json` 到调试目录

**新增：CLIP 帧差异度排序**
- CLIP 编码每帧计算语义差异度，差异最大的帧优先发给 VLM
- 帧选择权重调整：confidence 30% + keypoints 20% + pose 20% + 差异度 30%
- VLM Prompt 新增【帧差异度提示】，引导重点分析场景变化最大的帧
- 不使用 CLIP 时自动降级到原有逻辑，零影响

**新增：硅基流动 API 支持**
- 新增 siliconflow provider，使用 Qwen/Qwen3.6-35B-A3B 模型
- 使用 http.client 替代 urllib 解决 SSL 连接问题

**新增：扫描入库增强**
- 新增 `--sync-db` 参数：仅同步数据库，不生成 CSV
- `--sync-db` 同步完成后自动检查缺少视觉描述的记录，生成待处理 CSV
- `--force` 模式自动更新数据库
- GUI 扫描页面新增"同步数据库"选项

**新增：标题优化页面重构**
- 按钮分组（AI操作/编辑/过滤），布局更清晰
- 新增"最终文件名预览"列
- 多条件过滤下拉，撤销功能（10步）
- 修改行高亮加深

**新增：模型管理系统重构**
- 模型管理页面重写：卡片式布局，显示流水线角色和下载状态
- 模型配置统一到 default.toml + user.toml
- 新增 model_registry.py 和 model_downloader.py

**修复**
- 修复 stats_label AttributeError（模型管理页面初始化顺序）
- 修复主题切换不持久化（启动时从 config 读取主题）
- 修复视觉描述/关键词不写入数据库（CLI + GUI 两条路径）
- 修复 CLIP 模型状态检测（HF 目录名映射）
- 依赖包清理：移除 onnxruntime-gpu，新增 tomli 和 huggingface_hub

### v6.0.0

**重大更新：项目规范化重构**
- 项目结构重构为 Python 包（src/title_classifier）
- CLI 命令统一为 `title-classifier` 命令
- 配置文件分离到 config/ 目录
- 测试框架完善

**重大更新：YOLO视觉分析集成**
- 集成 YOLOv8，支持检测、姿态估计、实例分割
- 视频全面分析模式，每2秒采样一帧
- 智能帧选择，基于姿态变化、置信度、关键点可见性
- 视频摘要生成，包含姿态分布、变化时间线

**重大更新：关键词提取优化**
- 水印博主名字设为最优先级
- 聚焦穿着、姿势、行为三个维度
- 过滤诈骗网址，只保留博主昵称

**重大更新：final_name渐进式填充**
- 阶段1（scanner）：final_name = proposed_title
- 阶段1c（vision）：final_name = [关键词]_原文件名
- 每个阶段都有值，不会出现空值问题

**重大更新：SRT字幕生成功能**
- 视觉描述写入SRT开头
- SRT文件名使用final_name格式
- 支持音频字幕追加

**重大更新：GUI功能完善**
- Stage1b：AI优化结果预览表格，支持右键编辑
- Stage1c：YOLO检测器，YOLO模型多选
- Stage2：批量确认、批量清空功能

### v5.0.0
- CLIP 本地预分类，支持多标签输出
- 关键帧检测，基于帧差异自动检测视频关键帧
- 人体区域检测，专注人物穿着变化
- Embedding 变化检测，基于 CLIP embedding 相似度检测穿着变化
- 多帧 VLM，支持送入多帧给云端 VLM
- Stage1b AI优化支持预览编辑

### v4.0.0
- 支持图片文件（.jpg, .jpeg, .png, .bmp, .webp, .gif, .tiff）
- 人体检测预处理默认启用（YOLO 模型）
- 新增智能压缩、水印优先功能

**v8.2.0 更新：**

**模型管理系统重构：**
- 模型管理页面重写：卡片式布局，显示模型描述、流水线角色、下载状态
- 模型配置统一到 `config/default.toml` + `config/user.toml`，废弃 `config/models.json`
- 模型切换真正生效：detectors 从 config 读取用户选择的模型
- 新增 `core/model_registry.py`（声明式模型注册表）和 `core/model_downloader.py`（下载逻辑解耦）
- 依赖包清理：移除 `onnxruntime-gpu`，新增 `tomli`（Python 3.10 兼容）和 `huggingface_hub`（可选）

**扫描入库增强：**
- 新增 `--sync-db` 参数：仅同步数据库，不生成 CSV（适合全盘索引）
- `--sync-db` 同步完成后自动检查缺少视觉描述的记录，生成待处理 CSV
- `--force` 模式自动更新数据库
- GUI 扫描页面新增"同步数据库"选项（与"强制重新分类"互斥）

**标题优化页面重构：**
- 按钮分组：AI操作 / 编辑 / 过滤，布局更清晰
- 新增"最终文件名预览"列，实时显示 `[关键词]_原标题` 效果
- 多条件过滤下拉（needs_vision / audio_recognized / 已修改）
- 新增撤销功能（最近 10 步）
- 修改行高亮加深（`#c8e0ff`）

**其他修复：**
- 修复 `stats_label` AttributeError（模型管理页面初始化顺序）
- 修复主题切换不持久化（启动时从 config 读取主题）
- 修复视觉描述/关键词不写入数据库（CLI 和 GUI 两条路径）
- 修复 CLIP 模型状态检测（目录名映射修正）

### v3.0.0
- 新增阶段 1c：视觉理解提取关键词
- 新增 gcli API 支持

### v2.0.0
- 新增阶段 1b：AI 优化标题
- 新增自动跳过已分类文件

### v1.0.0
- 双阶段安全重命名
- Jieba 分词 + TF-IDF 关键词提取

---

## 许可证

MIT License

---

**如有建议或需求，欢迎反馈！**
