# title-classifier CLI 参考手册

版本：8.1.0  
入口：`python -m title-classifier <command> [options]`  
打包后：`title-classifier <command>`（自动启动 GUI）或 `title-classifier-gui`

---

## 1. 安装与运行

```bash
# 开发模式（从源码根目录）
python -m title-classifier <command> [options]

# 已安装（pip install -e .）后
title-classifier <command> [options]
```

**运行环境：** 需在项目根目录（含 `config/`、`src/` 的子目录）下运行，否则相对路径（数据库、配置、输出）会错位。  
**数据库位置：** `<项目根>/data/media.db`（自动创建，WAL 模式）。  
**配置文件：** `config/default.toml`（内置） + `config/user.toml`（用户覆盖，可选）。

---

## 2. 全局参数

| 参数 | 说明 |
|------|------|
| `-v, --verbose` | 详细输出（DEBUG 级别日志） |
| `--log <path>` | 指定日志文件路径；不指定则默认写入 `logs/<日期>/<时分秒>.log` |

---

## 3. 子命令总览

| 命令 | 说明 |
|------|------|
| `scan` | 扫描目录/文件，生成待审 CSV 或同步数据库 |
| `vision` | 对 CSV 中的记录执行视觉识别（YOLO + CLIP + VLM） |
| `audio` | 音频识别，生成 SRT 字幕 |
| `rename` | 根据 CSV 执行文件重命名 |
| `refine` | AI 标题优化（待实现） |
| `gui` | 启动图形界面 |
| `db` | 数据库管理（init / import / list / search / show / history / stats） |

---

## 4. `scan` — 扫描

```
python -m title-classifier scan -d <dir> [options]
```

### 4.1 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-d, --dir` | str（必填） | — | 目标目录或单个媒体文件路径 |
| `-o, --output` | str | 自动 | 输出 CSV 路径。目录扫描时自动为 `<output-dir>/<dir名>/title_review.csv` |
| `--output-dir` | str | `data/output` | 输出目录根 |
| `-a, --append` | flag | false | 追加模式（在已有 CSV 末尾追加，而非覆盖） |
| `--exclude-dir` | str[] | `[]` | 排除的目录名（相对路径匹配，可多个） |
| `--force` | flag | false | 强制重新分类：剥离 `[xxx]_` 前缀后重新判断是否需要视觉识别 |
| `--sync-db` | flag | false | **数据库同步模式**：扫描全部文件同步到数据库，并生成待视觉识别 CSV |

### 4.2 两种模式

#### 模式 A：普通扫描（默认）

仅生成 CSV，不写数据库（除非配合 `--force`）。

```bash
# 扫描目录，输出到 data/output/<目录名>/title_review.csv
python -m title-classifier scan -d "D:\Telegram\shortvideo"

# 扫描单个文件
python -m title-classifier scan -d "D:\Telegram\shortvideo\xxx.mp4"

# 强制重分类（剥离前缀重新判断）
python -m title-classifier scan -d "D:\Telegram\shortvideo" --force

# 排除某些子目录
python -m title-classifier scan -d "D:\Telegram" --exclude-dir cached thumbs
```

**普通扫描的跳过逻辑：** 文件名以 `[` 开头且含 `]_` 前缀（即已被分类/规范化）的文件会被跳过（返回 `None`），不会出现在 CSV 中。只有"需要视觉识别"的文件才会输出。`--force` 会剥离前缀后重新判断。

#### 模式 B：数据库同步（`--sync-db`）

全量同步到数据库，不过滤已分类文件。每个文件都会 `find_match`（去重），已存在则更新，不存在则插入。

```bash
python -m title-classifier scan -d "D:\Telegram" --sync-db
python -m title-classifier scan -d "D:\Telegram" --sync-db --exclude-dir cached
```

**`sync_db` 内部流程（`scanner.py:sync_db`）：**

1. 递归扫描目录，收集所有媒体文件（跳过隐藏目录如 `.thumbs`、`@eaDir`）。
2. 对每个文件：
   - 计算 `clean_title`（剥离 `[xxx]_` 前缀）、`classified`（是否已分类）、`file_size`、`duration`、`resolution`。
   - 视频用 ffprobe（cv2 备用）取时长/分辨率；若两者都失败 → 判定为损毁，**删除文件**并插入 `文件损毁已删除` 记录。
   - 调用 `db.find_match()` 去重：
     - **路径精确匹配**：`current_path` 或 `original_path` 等于文件路径 → 更新元数据。
     - **内容匹配**：`original_title` + `file_size`(±0.1%) + `duration`(±0.5s) + `resolution` 全部一致，且旧路径已不存在 → 判定为文件移动，追踪到已有记录并更新路径。
   - 未匹配 → `insert_media` 插入新记录（`insert_media` 内部还会再 `find_match` 一次，按清理后的标题匹配）。
3. 标记该目录下已不存在于磁盘的记录为 `文件已移走`。
4. 查询该目录下缺少视觉描述的记录，生成待处理 CSV（`data/output/<目录名>/title_review.csv`），复用已有 CSV 中的视觉数据避免覆盖。

**返回状态：** `已规范化`（有 `[xxx]_` 前缀）、`待确认`（未分类）、`已完成`/`文件已移走`（恢复时）。

### 4.3 媒体文件扩展名

视频：`.mp4 .mkv .avi .mov .flv .wmv .webm .m4v .ts`  
图片：`.jpg .jpeg .png .bmp .webp .gif .tiff`

### 4.4 CSV 输出列

```
original_title, original_path, needs_vision, final_name, review_status,
audio_recognized, srt_path, vision_description, vision_keywords, vision_failed,
human_detected, detection_confidence, detection_timestamp, detection_method,
clip_clothing, clip_action, clip_hairstyle, clip_tags, clip_tags_json,
clip_confidence, clip_detail, vision_source, file_size, duration, resolution
```

---

## 5. `vision` — 视觉识别

```
python -m title-classifier vision -c <csv> [options]
```

读取 CSV，对需要视觉识别的视频逐条执行 YOLO 姿态检测 + CLIP 预分类 + VLM 大模型描述，结果写回 CSV 并可选自动导入数据库。

### 5.1 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-c, --csv` | str | `data/output/title_review.csv` | 输入 CSV 路径 |
| `-p, --provider` | str | `gcli` | AI Provider（覆盖配置 `providers.stage_providers.vision`） |
| `--use-yolo` | flag | false | 启用 YOLO 姿态检测（分析人体姿态，智能选帧） |
| `--comprehensive` | flag | false | 全面分析模式：使用 detect + pose + segment 三个模型投票 |
| `--yolo-conf` | float | 0.5 | YOLO 置信度阈值 |
| `--use-clip` | flag | false | 启用 CLIP 预分类 |
| `--clip-threshold` | float | 0.25 | CLIP 置信度阈值 |
| `--max-image-size` | int | 800 | 图片最大边长（像素） |
| `--vlm-frames` | int | 10 | VLM 帧数（采样间隔决定） |
| `--analysis-step` | float | 5.0 | YOLO 模式采样间隔（秒） |
| `--max-sample-frames` | int | 50 | 最大采样帧数上限 |
| `--device` | str | `auto` | 推理设备：`auto` / `cuda` / `cpu` |
| `--concurrent` | int | 4 | 并发处理视频数 |
| `--backend` | str | `openvino` | YOLO 后端：`auto` / `openvino` / `pytorch` |
| `--no-motion-detection` | flag | false | 禁用运动检测前置过滤 |
| `--motion-threshold` | float | 10.0 | 运动检测阈值（变化像素比例 %） |
| `--no-scene-detection` | flag | false | 禁用场景分段分析（仅 60s+ 视频生效） |
| `--scene-threshold` | float | 0.3 | 场景检测敏感度（0-1，越低切得越碎） |
| `--max-scenes` | int | 10 | 最大场景段数 |
| `--frames-per-scene` | int | 10 | 每场景取帧数 |
| `--all` | flag | false | 处理所有未识别文件（无视 needs_vision 标记） |
| `--debug` | flag | false | 调试模式，保存检测结果和 VLM 输入输出 |
| `--debug-dir` | str | `data/debug` | 调试数据输出目录 |
| `--retry-failed` | flag | false | 仅重试之前失败的行（`vision_failed=true`） |
| `--auto-import` | flag | false | 处理完成后自动将 CSV 导入数据库 |

### 5.2 筛选逻辑

默认处理满足以下条件的行：
- `needs_vision=true`
- `vision_keywords` 为空
- `vision_failed` 不为 `true`

`--all`：处理所有 `original_path` 非空、`vision_keywords` 为空、未失败的行。  
`--retry-failed`：仅处理 `vision_failed=true` 的行。

### 5.3 典型用法

```bash
# 基础视觉识别（单线程，默认 Provider）
python -m title-classifier vision -c data/output/shortvideo/title_review.csv

# 启用 YOLO + CLIP，GPU 加速，4 并发
python -m title-classifier vision -c data/output/shortvideo/title_review.csv \
  --use-yolo --use-clip --device cuda --concurrent 4

# 全面分析（三模型投票）
python -m title-classifier vision -c data/output/shortvideo/title_review.csv \
  --use-yolo --comprehensive

# 重试失败
python -m title-classifier vision -c data/output/shortvideo/title_review.csv --retry-failed

# 处理后自动导入数据库
python -m title-classifier vision -c data/output/shortvideo/title_review.csv --auto-import
```

---

## 6. `audio` — 音频识别

```
python -m title-classifier audio -c <csv> [options]
```

对视频执行语音识别，生成 SRT 字幕文件。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-c, --csv` | str | `data/output/title_review.csv` | 输入 CSV 路径 |
| `-p, --provider` | str | `mimo` | AI Provider |
| `--all` | flag | false | 处理所有未识别的视频 |

**筛选：** 默认处理 `needs_vision=true` 且 `audio_recognized!=true` 的视频行。SRT 输出到 CSV 所在目录的 `subtitles/` 子目录。

```bash
python -m title-classifier audio -c data/output/shortvideo/title_review.csv
```

---

## 7. `rename` — 重命名

```
python -m title-classifier rename -c <csv> [options]
```

根据 CSV 中的 `final_name` 和 `review_status` 执行文件重命名。仅处理 `review_status=已确认` 的记录。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-c, --csv` | str | `data/output/title_review.csv` | 输入 CSV 路径 |
| `--dry-run` | flag | false | 模拟运行（不实际重命名） |
| `--use-rclone` | flag | false | 使用 rclone 重命名（适合云盘） |
| `--rclone-path` | str | `rclone` | rclone 可执行文件路径 |
| `--max-workers` | int | 5 | 并行重命名线程数 |

```bash
# 预览
python -m title-classifier rename -c data/output/shortvideo/title_review.csv --dry-run

# 执行
python -m title-classifier rename -c data/output/shortvideo/title_review.csv
```

重命名后会同步更新数据库中的 `current_path` 和 `original_path`。

---

## 8. `db` — 数据库管理

```
python -m title-classifier db <action> [options]
```

### 8.1 子命令

| 动作 | 说明 |
|------|------|
| `init` | 初始化数据库表结构 |
| `import` | 从 CSV 导入数据 |
| `list` | 列出记录 |
| `search` | 搜索记录 |
| `show` | 查看单条记录详情 |
| `history` | 查看改动历史 |
| `stats` | 统计信息 |

### 8.2 `db import`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--csv` | str | 全部导入 | 指定单个 CSV 路径 |
| `--all` | flag | true | 导入 `data/output/` 下所有 CSV |

```bash
python -m title-classifier db init
python -m title-classifier db import --csv data/output/shortvideo/title_review.csv
python -m title-classifier db import   # 导入所有 CSV
```

`import_csv` 逻辑：逐行读取，按 `find_match` 去重；已存在则补全空字段，不存在则插入；同时导入标签（`vision_keywords` 按逗号分割）。

### 8.3 `db list`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--limit` | int | 20 | 显示数量 |
| `--offset` | int | 0 | 偏移量 |

```bash
python -m title-classifier db list --limit 50
```

### 8.4 `db search`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--query` | str | — | 搜索关键词（标题/名称模糊匹配） |
| `--tag` | str | — | 按标签搜索 |
| `--source` | str | — | 按来源筛选 |

```bash
python -m title-classifier db search --query "水手服"
python -m title-classifier db search --tag "JK"
```

### 8.5 `db show` / `db history`

```bash
python -m title-classifier db show 1234        # 查看 id=1234 的详情
python -m title-classifier db history 1234     # 查看 id=1234 的改动历史
```

### 8.6 `db stats`

```bash
python -m title-classifier db stats
# 输出：媒体总数、视频数、图片数、标签数、改动数、VLM 帧数、热门标签
```

---

## 9. `gui` — 图形界面

```bash
python -m title-classifier gui
```

启动 tkinter GUI。GUI 内部通过 `_run_command` 调用 CLI 命令，并通过 `_sync_csv_to_db` / `_gui_sync_to_db` 将结果同步回数据库。

---

## 10. 配置文件（config/default.toml）

配置通过 `config/default.toml`（内置）+ `config/user.toml`（用户覆盖）合并加载。`user.toml` 中只需写要覆盖的字段。

### 关键配置项

```toml
[general]
device = "auto"          # auto / cuda / cpu

[vision]
max_image_size = 800
vlm_frames = 10
analysis_step = 5.0      # YOLO 采样间隔（秒）
max_sample_frames = 50
motion_threshold = 10.0

[yolo]
model_type = "pose"
confidence = 0.5
backend = "openvino"      # auto / openvino / pytorch

[clip]
threshold = 0.25

[scene_detection]
enabled = true
threshold = 0.3
max_scenes = 10
frames_per_scene = 10

[providers]
default = "gcli"
[providers.stage_providers]
refine = "gcli"
vision = "gcli"
audio = "mimo"

[renamer]
use_rclone = false
max_workers = 5
```

---

## 11. 数据库 Schema

数据库：SQLite（WAL 模式），路径 `data/media.db`。

### 核心表

| 表 | 说明 |
|----|------|
| `media_files` | 媒体主表 |
| `video_fingerprints` | 视频指纹（file_size + duration 唯一） |
| `tags` | 标签字典 |
| `media_tags` | 媒体-标签关联（多对多） |
| `change_log` | 字段改动历史 |
| `vlm_frames` | VLM 抽帧记录 |
| `collections` | 收藏集（DBwatcher 扩展） |
| `bookmarks` | 书签（DBwatcher 扩展，FK CASCADE） |
| `play_history` | 播放进度（DBwatcher 扩展，FK CASCADE） |

### media_files 字段

| 字段 | 说明 |
|------|------|
| `id` | 自增主键 |
| `original_title` | 原始标题（清理后的文件名，剥离 `[xxx]_`） |
| `original_path` | 原始路径 |
| `current_path` | 当前路径（重命名/移动后更新） |
| `file_size` | 文件大小（字节） |
| `duration` | 时长（秒） |
| `resolution` | 分辨率（如 `1920x1080`） |
| `final_name` | 最终分类名称 |
| `vision_description` | VLM 视觉描述 |
| `vision_keywords` | 视觉关键词（逗号分隔） |
| `human_detected` | 是否检测到人体 |
| `detection_method` | 检测方法（如 `yolo`） |
| `needs_vision` | 是否需要视觉识别 |
| `audio_recognized` | 音频是否已识别 |
| `review_status` | 审核状态：`待确认` / `已确认` / `已规范化` / `已完成` / `文件已移走` / `文件损毁已删除` |
| `srt_path` | 字幕路径 |
| `fingerprint_id` | 关联指纹 |

### 去重匹配逻辑（`find_match`）

两级匹配，优先级从高到低：

1. **路径精确匹配**：`current_path` 或 `original_path` 等于查询路径 → 直接命中。
2. **内容匹配（追踪）**：`original_title` + `file_size`(±0.1%) + `duration`(±0.5s) + `resolution` 全部一致，且旧路径在磁盘上已不存在 → 判定为同一文件被移动/重命名，追踪到已有记录。

**注意：** `insert_media` 调用 `find_match` 时传入的 `original_title` 是清理后的（`strip_bracket_prefix`），而 `sync_db` 调用 `find_match` 时传入的是原始文件名（含 `[xxx]_` 前缀）。这可能导致同一文件在 `sync_db` 中先被当作新记录插入，后续 `insert_media` 内部再匹配到它。

---

## 12. 典型工作流

### 工作流 A：新入库（扫描 → 视觉 → 重命名）

```bash
# 1. 扫描目录，生成待审 CSV
python -m title-classifier scan -d "D:\Telegram\shortvideo"

# 2. 视觉识别（YOLO + CLIP + VLM）
python -m title-classifier vision -c data/output/shortvideo/title_review.csv \
  --use-yolo --use-clip --concurrent 4

# 3. 人工审核 CSV，将确认的行标记 review_status=已确认

# 4. 预览重命名
python -m title-classifier rename -c data/output/shortvideo/title_review.csv --dry-run

# 5. 执行重命名
python -m title-classifier rename -c data/output/shortvideo/title_review.csv
```

### 工作流 B：数据库同步（直接入库，不生成待审 CSV 主流程）

```bash
# 1. 全量同步到数据库
python -m title-classifier scan -d "D:\Telegram" --sync-db

# 2. 查看统计
python -m title-classifier db stats

# 3. 搜索
python -m title-classifier db search --query "关键词"
```

### 工作流 C：增量同步后视觉识别

```bash
# 1. 同步（新增文件入库，已存在的更新路径/元数据，消失的标记移走）
python -m title-classifier scan -d "D:\Telegram\shortvideo" --sync-db

# 2. 同步会自动生成 data/output/shortvideo/title_review.csv（含待视觉识别记录）

# 3. 对生成的 CSV 执行视觉识别
python -m title-classifier vision -c data/output/shortvideo/title_review.csv --use-yolo --auto-import
```

---

## 13. 审核状态说明

| 状态 | 含义 | 来源 |
|------|------|------|
| `待确认` | 未分类，需要处理 | 普通扫描插入 / sync_db 未分类 |
| `已规范化` | 已有 `[xxx]_` 前缀，跳过视觉 | sync_db 识别到前缀 |
| `已完成` | 视觉+音频处理完毕 | vision 处理完成 / 恢复时已有描述 |
| `已确认` | 人工审核通过，可重命名 | 人工在 CSV/GUI 中标记 |
| `文件已移走` | 磁盘上已不存在 | sync_db 检测到 / 手动标记 |
| `文件损毁已删除` | ffprobe+cv2 都读不到时长，文件已被删除 | sync_db 损毁检测 |

---

## 14. 常见问题

**Q: 同步后文件显示"文件已移走"但实际存在？**  
A: `sync_db` 按目录检测。若文件从 `D:\aria2\X` 移到 `D:\Telegram\Y`，旧记录仍指向 `D:\aria2\X`。需对新目录重新 `sync_db`，内容匹配会在旧路径消失后追踪到新路径。

**Q: 为什么同步后数据库没新增，CSV 却显示新增？**  
A: `sync_db` 日志的"新增"指走到 insert 分支的文件数，但 `insert_media` 内部会再次 `find_match`，若按清理标题匹配到已有记录则只更新路径不新增行。最终数据库行数可能不增加。

**Q: 重复记录怎么处理？**  
A: `find_match` 按 标题+大小+时长+分辨率 去重。大小/时长极接近但实际不同的文件可能被误判。建议人工核对后删除较早 id。

**Q: `sync_db` 会删除文件吗？**  
A: 仅当视频 ffprobe 和 cv2 都获取不到时长/分辨率时，判定为损毁并 `unlink` 文件。图片不会。

---

## 15. 环境变量

| 变量 | 说明 |
|------|------|
| `PYTHONIOENCODING` | 强制 UTF-8（Windows 控制台已自动设置） |

`.env` 文件会在 `vision`/`audio` 命令启动时自动加载（`os.environ.setdefault`）。
