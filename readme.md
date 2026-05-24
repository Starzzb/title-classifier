# 视频/图片标题批量标准化重命名工具

## 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [快速开始](#快速开始)
- [GUI使用说明](#gui使用说明)
- [阶段一：扫描与生成](#阶段一扫描与生成)
- [阶段 1b：AI 优化标题](#阶段-1b-ai-优化标题)
- [阶段 1c：视觉理解提取关键词](#阶段-1c-视觉理解提取关键词)
- [阶段二：核对与重命名](#阶段二核对与重命名)
- [AI Provider 配置](#ai-provider-配置)
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
| 智能扫描 | 递归遍历指定目录，支持 10+ 种视频格式和 7 种图片格式 |
| 语义分析 | 基于 Jieba 分词 + TF-IDF 算法，自动提取标题核心关键词 |
| 无意义检测 | 程序自动识别 hash、Telegram 来源、IMG_xxx 等无意义标题 |
| 自动跳过 | 已分类的文件（以 `[` 开头的标题）自动跳过，避免重复处理 |
| AI 优化 | 支持 Ollama、智谱、gcli 三种 API，批量精简标题 |
| 视觉识别 | 支持 MiMo、gcli 视觉模型，从视频/图片提取关键词 |
| **CLIP 预分类** | 使用 OpenCLIP 进行本地预分类，支持多标签输出 |
| **关键帧检测** | 基于帧差异自动检测视频关键帧，捕捉场景变化 |
| **人体区域检测** | 使用 UHD 模型检测人体区域，专注人物穿着变化 |
| **Embedding 变化检测** | 基于 CLIP embedding 相似度检测穿着变化，更准确 |
| **多帧 VLM** | 支持送入多帧给云端 VLM，提高识别准确度 |
| 人体检测 | 使用 UHD 超轻量模型自动找到视频中包含人体的帧（默认启用） |
| 智能压缩 | 自动压缩图片/视频帧，避免 API 传输过大文件 |
| 水印优先 | 视觉识别优先提取水印、社交媒体名称、博主ID等关键信息 |
| 安全预览 | 三阶段设计，先生成待审表，人工确认后再执行，零风险操作 |
| 冲突避让 | 自动检测文件名冲突，智能追加序号（`_1`, `_2`），防止覆盖 |
| 增量模式 | 支持多次扫描追加记录，适合分批处理大型媒体库 |
| 目录排除 | 灵活排除特定目录（按名称/路径），避免扫描系统文件夹 |
| 详细日志 | 生成执行日志与错误报告，便于审计与问题排查 |

---

## 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.10+ | 推荐 3.12 |
| 操作系统 | Windows / macOS / Linux | 全平台兼容 |
| 磁盘空间 | >= 500MB | 包含模型文件和依赖 |
| 权限要求 | 目标目录读写权限 | 必需 |
| ffmpeg | 全局 PATH | 用于阶段 1c 视频帧提取 |

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
git clone https://github.com/Starzzb/tittle-classfication.git
cd tittle-classfication
```

### 3. 创建虚拟环境并安装依赖

```powershell
uv venv --python 3.12
uv sync
```

### 4. 下载模型文件

#### 4.1 人体检测模型（必须）

模型文件已包含在项目中：`models/human_detection/ultratinyod_res_anc8_w128_64x64_loese_distill.onnx`

#### 4.2 CLIP 模型（可选，用于本地预分类）

CLIP 模型会在首次使用时自动下载到 `models/clip/` 目录。

如需手动下载：
```powershell
uv run python download_clip.py
```

### 5. 配置 AI API（可选）

在项目根目录创建 `.env` 文件：

```env
# 智谱 API（用于文本优化）
ZHIPU_API_KEY=your_zhipu_key_here

# gcli API（用于视觉识别，推荐）
GCLI_API_KEY=your_gcli_key_here

# 小米 MiMo API（用于视觉识别）
MIMO_API_KEY=your_mimo_key_here
```

---

## 快速开始

### 命令行方式

```powershell
# 步骤1：扫描目录并生成待审表
uv run python stage1_extract_propose.py -d "F:\Videos"

# 步骤2：（可选）AI 优化标题
uv run python stage1b_ai_refine.py -p gcli

# 步骤3：视觉识别提取关键词（默认启用人体检测）
uv run python stage1c_vision_refine.py --use-clip -p gcli

# 步骤4：用编辑器打开 CSV，核对并修改
code title_review.csv

# 步骤5：模拟运行，预览结果
uv run python stage2_apply_rename.py --dry-run

# 步骤6：确认无误，执行重命名
uv run python stage2_apply_rename.py
```

### GUI 方式

```powershell
# 启动图形界面
uv run python gui.py
```

---

## GUI使用说明

### 启动 GUI

```powershell
uv run python gui.py
```

### 功能标签页

| 标签页 | 功能 |
|--------|------|
| **Stage1 扫描** | 扫描目录，生成待审表 |
| **Stage1b AI优化** | 使用AI优化标题（只处理needs_vision=false） |
| **Stage1c 视觉识别** | 使用CLIP+VLM提取关键词 |
| **Stage2 重命名** | 执行重命名操作 |

### 特色功能

- **鼠标悬停提示**：所有控件都有详细的功能说明
- **Provider自动检测**：下拉框自动显示可用的AI Provider
- **预览编辑**：Stage1b支持预览和编辑AI优化结果
- **右键菜单**：支持采用原标题、编辑、删除等操作

---

## 阶段一：扫描与生成

### 功能说明

递归扫描指定目录中的所有视频和图片文件，提取文件名，通过分词算法生成标准化建议名称，输出为 CSV 待审表。

### 基本用法

```powershell
# 扫描单个目录
uv run python stage1_extract_propose.py -d "F:\Download"

# 排除特定目录
uv run python stage1_extract_propose.py -d "F:\Download" --exclude-dir "temp"

# 强制重新分类（处理已有中括号的文件）
uv run python stage1_extract_propose.py -d "F:\Videos" --force-reclassify
```

### 参数说明

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--target-dir` | `-d` | **必需** | - | 要扫描的目录路径 |
| `--output` | `-o` | 可选 | `title_review.csv` | 输出待审表文件名 |
| `--append` | `-a` | 可选 | `False` | 启用追加模式 |
| `--exclude-dir` | - | 可选（可多次） | - | 排除的目录名或绝对路径 |
| `--force-reclassify` | - | 可选 | `False` | 强制重新分类 |

---

## 阶段 1b：AI 优化标题

### 功能说明

读取阶段一生成的 CSV，调用 AI 大语言模型对标题进行智能精简，去除冗余信息，只保留核心标题。

**处理规则**：
- `needs_vision=false` 的标题：AI 精简后写入 `final_name`
- `needs_vision=true` 的标题：**跳过**，留给阶段 1c 处理

### 支持的 Provider

| Provider | 默认模型 | 环境变量 | 特点 |
|----------|---------|---------|------|
| `ollama` | qwen2.5:7b-instruct-q4_K_M | - | 本地运行，免费 |
| `zhipu` | GLM-4.7-Flash | ZHIPU_API_KEY | 云端，中文优秀 |
| `gcli` | gemini-3-flash-preview | GCLI_API_KEY | 云端，推荐使用 |

### 基本用法

```powershell
# 使用 gcli API（推荐）
uv run python stage1b_ai_refine.py -p gcli

# 使用 Ollama（本地）
uv run python stage1b_ai_refine.py -p ollama

# 使用智谱 API
uv run python stage1b_ai_refine.py -p zhipu
```

---

## 阶段 1c：视觉理解提取关键词

### 功能说明

对于 `needs_vision=true` 的文件（hash、纯数字、IMG_xxx 等），使用视觉大模型从视频/图片中提取内容关键词。

**核心特性**：
- **CLIP 本地预分类**：使用 OpenCLIP 进行本地分类，支持多标签输出
- **关键帧检测**：基于帧差异自动检测视频关键帧，捕捉场景变化
- **人体区域检测**：使用 UHD 模型检测人体区域，专注人物穿着变化
- **Embedding 变化检测**：基于 CLIP embedding 相似度检测穿着变化
- **多帧 VLM**：支持送入多帧给云端 VLM，提高识别准确度

### 支持的 Provider

| Provider | 默认模型 | 环境变量 | 特点 |
|----------|---------|---------|------|
| `gcli`（推荐） | gemini-3-flash-preview | GCLI_API_KEY | 云端，免费 |
| `mimo` | mimo-v2-omni | MIMO_API_KEY | 小米自研视觉模型 |

### 基本用法

```powershell
# 默认使用 gcli，启用 CLIP
uv run python stage1c_vision_refine.py --use-clip -p gcli

# 使用 MiMo 模型
uv run python stage1c_vision_refine.py -p mimo

# 禁用人体检测
uv run python stage1c_vision_refine.py --no-frame-selector

# 模拟运行
uv run python stage1c_vision_refine.py --dry-run
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-clip` | 可选 | `False` | 启用 CLIP 预分类 |
| `--clip-threshold` | 可选 | `0.25` | CLIP 置信度阈值 |
| `--clip-frames` | 可选 | `5` | CLIP 分析的视频帧数 |
| `--vlm-frames` | 可选 | `3` | 送入 VLM 的帧数 |
| `--keyframe-threshold` | 可选 | `30.0` | 关键帧差异阈值 |
| `--max-keyframes` | 可选 | `8` | 最大关键帧数 |
| `--use-embedding-detection` | 可选 | `True` | 使用 Embedding 检测变化 |
| `--embedding-threshold` | 可选 | `0.75` | Embedding 相似度阈值 |
| `--no-frame-selector` | 可选 | `False` | 禁用人体检测 |
| `--max-image-size` | 可选 | `800` | 图片最大边长 |

---

## 阶段二：核对与重命名

### 功能说明

读取 CSV 待审表，仅对标记为"已确认"的记录执行安全重命名。

### 基本用法

```powershell
# 模拟运行（推荐首次执行）
uv run python stage2_apply_rename.py --dry-run

# 正式执行
uv run python stage2_apply_rename.py

# 批量确认所有记录
uv run python renamecsv.py
```

---

## AI Provider 配置

### 自定义 Provider

创建 `providers.json` 文件添加自定义 Provider：

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
    "description": "自定义Provider描述"
  }
}
```

### Provider 类型

| 类型 | 说明 | 支持阶段 |
|------|------|----------|
| `text` | 只支持文本 | 1b |
| `vision` | 只支持视觉 | 1c |
| `multi` | 都支持 | 1b + 1c |

### 可用性检测

系统会自动检测 Provider 可用性：
- 检查环境变量中的 API Key 是否存在
- 只显示可用的 Provider

---

## 常见问题

### Q1：提示"文件或目录损坏且无法读取"

```powershell
uv run python stage1_extract_propose.py -d "F:\" --exclude-dir "F:\Download\love"
```

### Q2：CSV 在 Excel 中打开乱码

1. 用 VS Code 打开 CSV
2. 右下角点击编码 -> "通过编码保存"
3. 选择 "UTF-8 with BOM"

### Q3：人体检测不准确

```powershell
# 降低置信度阈值
uv run python stage1c_vision_refine.py --conf-threshold 0.3

# 禁用人体检测
uv run python stage1c_vision_refine.py --no-frame-selector
```

### Q4：如何撤销重命名？

```powershell
# 执行前备份目录
robocopy "F:\Videos" "F:\Videos_Backup" /E /NFL /NDL /NJH /NJS
```

---

## 更新日志

### v5.0.0（当前版本）
- **重大更新**：CLIP 本地预分类，支持多标签输出
- **重大更新**：关键帧检测，基于帧差异自动检测视频关键帧
- **重大更新**：人体区域检测，专注人物穿着变化
- **重大更新**：Embedding 变化检测，基于 CLIP embedding 相似度检测穿着变化
- **重大更新**：多帧 VLM，支持送入多帧给云端 VLM
- **重大更新**：Stage1b AI优化支持预览编辑，只处理needs_vision=false的标题
- **重大更新**：统一 Provider 管理，支持自定义 Provider
- 新增纯色帧检测（黑屏/白屏），自动过滤无效帧
- 优化帧选择策略，跳过视频开头和结尾
- 新增 GUI 工具提示功能，鼠标悬停显示功能说明
- 新增 Stage1b 右键菜单：采用原标题、编辑、删除
- 新增 Provider 可用性自动检测
- GUI Provider 改为下拉框自动检测
- 修复 Windows 编码问题（特殊字符如❤等）
- 修复图片处理vlm_frames未初始化问题
- 修复`[未分类]`文件强制重分类问题

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
