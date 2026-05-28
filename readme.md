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

工具好不好用，还是要看用工具的人，这个东西简洁明了，模型，提示词什么的你都可以自己换，符合你自己视频分类的需求。我的需求就是把自己的小学习资料加个名字列数据库好分类。这个东西就是一个处理框架，基本上就是最后的视觉大模型发挥主要的作用。中间的clip数据太少很难使用，人体检测模型可以替换。我在https://github.com/PINTO0309/PINTO_model_zoo 这个仓库扒拉的检测模型。可以自己查看一下。不过应该没人闲得无聊逛这些私人公共仓库吧（心虚）。最后就是这个东西是匹配个人需求的东西，你当然可以修改给自己使用。

---

## 核心功能

| 功能模块 | 特性描述 |
|---------|---------|
| **YOLO 视觉分析** | 集成 YOLOv8，支持检测、姿态估计、实例分割 |
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
| **无意义检测** | 程序自动识别 hash、Telegram 来源、IMG_xxx 等无意义标题 |
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

| 模型 | 大小 | 功能 |
|------|------|------|
| `yolov8n.pt` | 6MB | 人体检测 |
| `yolov8n-pose.pt` | 7MB | 姿态估计 |
| `yolov8n-seg.pt` | 7MB | 实例分割 |

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
| `--force` | 强制重新分类 |

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
| `--yolo-conf` | YOLO置信度阈值（默认0.5） |
| `--use-clip` | 使用CLIP预分类 |
| `--vlm-frames` | VLM帧数（默认10） |
| `--analysis-step` | 采样间隔秒数（默认2.0） |
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
| **Stage1c 视觉识别** | YOLO/UHD检测 + VLM识别 |
| **Stage2 重命名** | 执行重命名操作 |

### Stage1 扫描

支持选择**目录**或**单个媒体文件**：
- 选择目录：递归扫描所有子目录中的媒体文件
- 选择文件：只处理选中的单个媒体文件

### Stage1b 预览编辑功能

- **加载CSV**：加载CSV到预览表格
- **AI优化**：调用AI优化标题，自动加载结果
- **右键菜单**：
  - 编辑：双击或右键编辑优化结果
  - 采用原标题：使用原始文件名
  - 切换 needs_vision：手动标记是否需要视觉识别
  - 切换 audio_recognized：手动标记音频识别状态
  - 删除：从预览中移除
- **确认写入**：将优化结果写入CSV

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
merge_gap = 0.8          # 间隙小于此值的相邻语音段合并
min_keep_duration = 1.0  # 合并后仍不足此值的跳过

# 第二层：语义打包
max_chunk = 25.0         # 模型时长上限
long_gap = 2.0           # 间隙超过此值强制封口

# 第三层：静音过滤
min_duration = 1.0       # 最小块时长
min_speech_ratio = 0.4   # 最小语音占比
```

### Stage1c 视觉识别

- **检测器选择**：YOLO（默认）或 UHD
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

YOLOv8 是一个多功能视觉分析模型，支持：

1. **目标检测**（detect）：检测人体位置和边界框
2. **姿态估计**（pose）：17个关键点，识别动作姿态
3. **实例分割**（segment）：像素级人体区域分割

### 视频全面分析模式

当使用 `--use-yolo` 时，启用视频全面分析模式：

1. **高密度采样**：默认每2秒采样一帧
2. **姿态分析**：记录每帧的姿态、置信度、关键点
3. **智能帧选择**：选择10帧代表性帧
4. **视频摘要**：生成姿态分布、变化时间线
5. **VLM增强**：将YOLO上下文传给VLM

### 使用示例

```powershell
# 使用YOLO姿态估计
uv run title-classifier vision --use-yolo --yolo-model pose -p gcli

# 使用YOLO全面分析（检测+姿态+分割）
uv run title-classifier vision --use-yolo --yolo-model detect pose segment -p gcli

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
  ↓ 第一层：微合并（间隙 < 0.8秒合并）
  ↓ 第二层：语义打包（长停顿 > 2秒断开，最大块 25秒）
  ↓ 第三层：静音过滤（时长 < 1秒跳过，语音占比 < 40%跳过）
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
min_speech_ms = 150
min_silence_ms = 80
merge_gap = 0.8
min_keep_duration = 1.0
max_chunk = 25.0
long_gap = 2.0
min_duration = 1.0
min_speech_ratio = 0.4

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
│       │   └── renamer.py           # 重命名
│       │
│       ├── detectors/
│       │   ├── base.py              # 检测器基类
│       │   ├── uhd.py               # UHD人体检测
│       │   ├── yolo.py              # YOLO检测
│       │   └── clip.py              # CLIP分类
│       │
│       ├── providers/
│       │   ├── __init__.py          # Provider管理
│       │   └── base.py              # Provider基类
│       │
│       ├── utils/
│       │   ├── video.py             # 视频工具
│       │   ├── image.py             # 图片工具
│       │   ├── audio.py             # 音频处理（VAD分段 + API调用）
│       │   ├── subtitle_postprocessor.py  # 字幕后处理
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
│   ├── human_detection/             # UHD模型（已移除，不再使用）
│   └── yolo/                        # YOLO模型（自动下载）
│
├── scripts/
│   ├── download_models.py           # 一键下载所有模型
│   ├── download_clip.py
│   └── download_yolo_models.py
│
├── tests/                           # 测试文件
├── test/                            # 测试数据
├── data/
│   ├── output/                      # 输出目录
│   └── subtitles/                   # SRT字幕目录
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

### Q3：关键词不包含博主名

确保 VLM 提示词中水印博主名字为最高优先级。当前版本已默认设置。

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

---

## 更新日志

### v7.1.0（当前版本）

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
- Stage1c：YOLO/UHD检测器选择，YOLO模型多选
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
- 人体检测预处理默认启用（UHD 超轻量模型）
- 新增智能压缩、水印优先功能

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
