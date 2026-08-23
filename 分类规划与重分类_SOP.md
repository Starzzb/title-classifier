# title-classifier 视频分类规划与重分类 操作手册（SOP）

**日期**: 2026-08-03
**环境**: Windows / PowerShell 5.1 / Python 3.10（含 ffprobe）
**状态**: ✅ 已验证可复现（2026-08-03 完整跑通 83 个视频）

---

## 1. 任务概述

对 `D:\Telegram\video` 目录下的视频，依据数据库 `media.db` 中保存的 VLM 描述文字（可参考标题），按 agent.md 决策树做**唯一分类**，并**按分类路径移动**到子目录。

整体流程分 4 步：
1. **读取数据库**：了解表结构（见 §3）
2. **精确匹配**：把每个视频文件与数据库记录对上（见 §5）
3. **分类判定**：按决策树给每条记录归入唯一路径（见 §6）
4. **执行移动**：按路径建目录并把文件移动进去（见 §7）

---

## 2. 关键路径

| 用途 | 路径 |
|---|---|
| 数据库 | `D:\python projects\myscripts\title-classifier\data\media.db` |
| 待分类视频目录 |如 `D:\Telegram\video\` 可以根据实际需要切换|
| 描述导出文件（数据库备份文本） | `D:\Telegram\video\all_descriptions.txt` |
| 分类规划输出 | `D:\Telegram\video\分类规划.md` |

> ⚠️ `all_descriptions.txt` 是数据库描述的文本导出（含 `=== ID=xxx ===` 段、文件名/标签/描述），但**以数据库为准**，该文件仅作参考。

---

## 3. 数据库情况（media.db，SQLite）

### 3.1 表清单（10 张）

| 表 | 行数(2026-08) | 用途 |
|---|---|---|
| `media_files` | 1978 | 主表：媒体文件与分类结果 |
| `tags` | 5719 | 标签字典（`id, name, category`） |
| `media_tags` | 15191 | 媒体↔标签关联（`confidence, source`） |
| `change_log` | 22829 | 变更审计（`field_name, old/new_value, change_source`） |
| `play_history` | 149 | 播放进度 |
| `bookmarks` | 30 | 收藏 |
| `collections` | 1 | 收藏夹 |
| `video_fingerprints` | 0 | 视频指纹（去重，空） |
| `vlm_frames` | 0 | VLM 抽帧记录（空） |
| `sqlite_sequence` | 4 | 自增计数 |

### 3.2 media_files 关键字段

`id, original_title, original_path, current_path, file_size, duration, resolution, file_hash, final_name, vision_description, vision_keywords, human_detected, detection_method, needs_vision, review_status, srt_path, fingerprint_id, created_at, updated_at, faststart, video_codec`

- `vision_description`：VLM 生成的**中文描述**（分类依据）
- `vision_keywords`：逗号/顿号分隔的标签词
- `final_name`：去扩展名的重命名后文件名（如 `[an80_水手服_...]_TG@COSSSDZH (4) (2)`）
- `original_title`：原始文件名（含扩展名）
- `review_status` 取值：`待确认 903 / 已规范化 525 / 已完成 448 / 已确认 30 / 文件已移走 72`
- `detection_method` 取值：`yolo 1212 / '' 665 / NULL 101`

### 3.3 重要坑

- ⚠️ **`file_hash` 全为 NULL**：无法做哈希级同一性校验，只能靠「文件大小 + 时长 + 分辨率」三者联合确认（见 §5.3）。
- 部分记录的 `vision_keywords` 是多段拼合（用 `，` 或 `；` 连接重复标签），读取时无需处理，仅展示用。
- `sqlite_sequence` 显示自增值约 4，但 `media_files` id 从 3361 到 6480，最大 id ≈ 视频总数 + 偏移，勿用 id 连续性做任何假设。

---

## 4. 文件命名结构

`D:\Telegram\video` 下视频文件名格式：
```
[标签1_标签2_...标签N]_原文件名.mp4
```
- 方括号内是下划线分隔的标签，括号外是原始标题（常含 `#话题`）。
- 数据库 `final_name` = 去掉扩展名的整个文件名（含 `[...]_` 前缀）。
- 数据库 `original_title` = 去掉 `[...]_` 前缀后的尾部（含扩展名）。
- 示例：
  - 文件名：`[an80_水手服_灰色百褶裙_...]_TG@COSSSDZH (4) (2).mp4`
  - `final_name`：`[an80_水手服_灰色百褶裙_...]_TG@COSSSDZH (4) (2)`
  - `original_title`：`TG@COSSSDZH (4) (2).mp4`

分类路径目录风格对齐 `G:\AdultResources\{01_Partnered\1.1_Hetero, 02_Solo\2.1_Cosplay, 03_Exhibition\3.2_IndoorPrivate, 04_Fetish\4.1_Foot, ...}`。

---

## 5. 精确匹配逻辑（★最重要）

### 5.1 匹配规则（按优先级）

对每个视频文件 `f`（先去掉扩展名得 `stem`）：
1. **`final_name == stem`**（精确匹配，无扩展名）
2. 若未命中：用正则 `^\[.*\]_(.+)$` 从文件名剥掉 `[...]_` 前缀得到 `tail`，**`original_title == tail`**
3. 仍未命中 → 视为未匹配，需人工检查（正常情况应全部命中）

> 曾实测：83 个视频中 53 个命中 `final_name`，30 个命中 `original_title`，0 个未匹配。

### 5.2 为什么不能只信文本匹配

只按文件名文本匹配可能把「同名但不同文件」错配，必须做同一性二次校验。

### 5.3 同一性校验（三重确认）

对每条文本匹配，用实际文件属性与数据库字段对照：
1. **文件大小**：`os.path.getsize(实际文件) == media_files.file_size`（字节级精确）
2. **时长**：`ffprobe` 探测 `format.duration`，与 `duration` 误差 `< 1 秒`
3. **分辨率**：`ffprobe` 探测视频流 `width x height`，与 `resolution` 一致

实测 30 个 original_title 匹配全部三重一致（含 608x1080、496x832 等竖屏分辨率）。

> 注意：有的 `resolution` 如 `1072x1904` 是旋转后的显示分辨率，`ffprobe` 读出的原始宽高可能带旋转角度，需以 `streams[0].width/height` 直接对比，差异大时检查 `rotation`。

---

## 6. 分类要求（agent.md 决策树摘要）

**只做分类规划与移动，不改数据库**（本次执行已移动文件；如后续需回写 DB 另议）。

### 6.1 一级主分类（优先级从高到低）

1. **伴侣互动性行为 Partnered**：明确≥两人之间的性行为过程
2. **单人自慰 Solo**：仅一人，核心为自我刺激
3. **露出/展示 Exhibition/Display**：核心是暴露身体/隐私部位，无性行为且非自慰为主
4. **恋物主导 Fetish-Focused**：核心是特定物品/身体部位（含恋足）
5. **ASMR/感官型**：以声音、触发音、低语为核心
6. **擦边/写真/软调 Teasing/Soft/Photo**：姿势、服装、氛围、轻微暗示，无明确性行为/自慰
7. **特殊形式 Special**：动画/CG、剧情向、指令引导等独立特征
8. **混合/无法判定 Mixed**：真兜底，仅描述完全模糊时使用

### 6.2 关键判定规则（实测重点）

- **写真/擦边 vs 真性爱/自慰必须严格区分**：
  - 只有「展示身材/摆姿势/脱衣/暴露/诱惑/写真」而无插入/抽插/自慰高潮等过程 → **第6类**
  - 出现两人及以上性交/抽插/插入/做爱/高潮 → **第1类**
  - 明确自我刺激过程（自慰/用手/玩具刺激私处至高潮）→ **第2类**
- **隐私部位暴露 ≠ 性行为/自慰**：
  - 单纯暴露/特写隐私部位 → 第3类或第6类
  - 伴随明确刺激过程才升级为自慰
- **实际踩线经验（本次 83 条）**：
  - 掀衣露乳、提裙露臀、张腿但**无露阴** → 第6类（6.2 动态擦边）
  - **拉开/拨开内裤、脱内裤、掰穴展示私处** → 第3类（3.2 室内/私密展示）
  - **手指拨弄私处/阴道插入/仿真阳具插入** → 第2类自慰（有 Cosplay 服装→2.1，无服装有道具→2.3，无服装徒手→2.2）
  - **足交（为男性服务至射精）** → 第1类伴侣性行为（优先于恋足）
  - **纯足部/腿部/丝袜展示，无伴侣互动** → 第4类恋物（4.1 恋足 / 4.3 服装恋物）
  - **3D 动画/CG 里无论多露骨性爱** → 第7类（7.3 动画/CG/非真人）
  - **群交/3P**：参与者为男+女且无异性互交 → 1.1；多人中**含同性接触**（如女女舔乳）→ 1.3 其他取向；纯两名女性 → 1.2.2 女同
  - **先自慰后与人性交** → 第1类（优先级更高）

### 6.3 二级/三级细分路径（本次实际用到的 12 个）

| 路径 | 中文标签 |
|---|---|
| `01_Partnered/1.1_Hetero` | 伴侣 · 异性 |
| `01_Partnered/1.2.2_Lesbian` | 伴侣 · 同性（女同） |
| `01_Partnered/1.3_OtherOrientation` | 伴侣 · 其他取向组合 |
| `02_Solo/2.1_Cosplay` | 自慰 · Cosplay/角色扮演 |
| `02_Solo/2.3_ToyAssisted` | 自慰 · 道具辅助 |
| `03_Exhibition/3.2_IndoorPrivate` | 露出 · 室内/私密 |
| `04_Fetish/4.1_Feet` | 恋物 · 恋足 |
| `04_Fetish/4.3_ItemClothing` | 恋物 · 物品/服装恋物 |
| `06_Teasing/6.1_PhotoStyle` | 擦边 · 写真风格 |
| `06_Teasing/6.2_DynamicTease` | 擦边 · 动态擦边/暗示 |
| `06_Teasing/6.3_AtmosphereRoleplay` | 擦边 · 氛围/角色软调 |
| `07_Special/7.3_AnimationCG` | 特殊 · 动画/CG/非真人 |

> 第5类 ASMR、2.2 普通自慰、7.1/7.2 等本次无样本，路径可按需添加。

### 6.4 唯一性与输出要求

- 每个视频**必须且只能**归入一个最细路径，绝不允许跨目录/多重归属。
- 分类理由须基于描述中的关键文字（一句话）。
- 分类规划 MD 每个视频附带：ID / 文件名 / 分类理由 / 标签 / 完整描述。

---

## 7. 执行步骤（一次性操作流程）

### 步骤 0：环境准备
- 确认 `python` 可用；确认 `ffprobe` 在 PATH（`Get-Command ffprobe`）。
- **PowerShell 管道给 python 传中文会乱码** → 一律把脚本写成 `.py` 文件再 `python xxx.py` 运行，不要用 `@'...'@ | python -` 传中文脚本。
- 输出中文时设置：`$env:PYTHONIOENCODING='utf-8'`；若在控制台打印中文再设 `[Console]::OutputEncoding=[System.Text.Encoding]::UTF8`。

### 步骤 1：读取数据库
```python
import sqlite3
con = sqlite3.connect(r"D:/python projects/myscripts/title-classifier/data/media.db")
cur = con.cursor()
for (name,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print(name, cur.execute("SELECT COUNT(*) FROM '%s'" % name).fetchone()[0])
# 读某条描述：
cur.execute("SELECT vision_description FROM media_files WHERE id=?", (id,))
```

### 步骤 2：生成待分类清单（匹配+提取）
匹配逻辑（核心代码）：
```python
import os, re, sqlite3
FOLDER = "D:/Telegram/video"
VID_EXT = {".mp4",".mov",".mkv",".avi",".flv",".m4v",".wmv",".webm"}
# ⚠️ 必须过滤扩展名，否则会把 all_descriptions.txt / 生成的 .md 当成视频！
vids = sorted(f for f in os.listdir(FOLDER)
              if os.path.isfile(os.path.join(FOLDER, f))
              and os.path.splitext(f)[1].lower() in VID_EXT
              and not f.startswith("all_descriptions"))
for f in vids:
    stem = os.path.splitext(f)[0]
    row = cur.execute("SELECT id, original_title, vision_description, vision_keywords "
                      "FROM media_files WHERE final_name=?", (stem,)).fetchone()
    if not row:
        tail = re.match(r"^\[.*\]_(.+)$", f).group(1)
        row = cur.execute("SELECT id, original_title, vision_description, vision_keywords "
                          "FROM media_files WHERE original_title=?", (tail,)).fetchone()
```

### 步骤 3：同一性校验（size + duration + resolution）
```python
import subprocess, json
def probe(path):
    r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=duration,width,height",
        "-show_entries","format=duration","-of","json",path],
        capture_output=True, text=True)
    return json.loads(r.stdout)
# 对照：os.path.getsize == file_size；|ff_duration - duration| < 1.0；ff_width x ff_height == resolution
```

### 步骤 4：分类判定
- 逐条读 `vision_description`，按 §6 决策树归入唯一路径。
- 维护一份 `id -> (路径, 理由)` 映射字典（脚本内置）。
- 校验：所有视频 id 都有映射、无重复、总数一致。

### 步骤 5：生成分类规划 MD
- 结构：`# 视频分类规划` → 统计概览表 → 按路径分组的详细结果。
- 每个视频：`**ID x / 文件名**：标题 → 分类理由` + 标签 + 描述（按 `；` 分段逐行）。

### 步骤 6：执行移动（本次已完成，下次对新区块重复）
```python
import shutil
for mid, path in id2path.items():
    f = id2file[mid]
    dst_dir = os.path.join(FOLDER, *path.split("/"))
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f)
    if os.path.exists(dst):
        print("DEST_EXISTS", f); continue
    shutil.move(os.path.join(FOLDER, f), dst)
```
- 移动后校验：递归统计每个子目录数量，与分类规划一致；根目录仅剩 `all_descriptions.txt` 与 `分类规划.md`。

---

## 8. 注意事项（踩坑清单）

1. **PowerShell 传中文**：`@'...'@ | python -` 会把中文按 ANSI 传导致乱码/正则报错（`re.error` 或 mojibake）。**必须用 .py 文件**。
2. **过滤非视频文件**：脚本要按扩展名过滤，否则会误把 `all_descriptions.txt`、`分类规划.md` 当视频，导致匹配失败。
3. **`file_hash` 为空**：不要尝试哈希比对；用 大小+时长+分辨率 三重校验。
4. **分辨率旋转**：`ffprobe` 原始宽高可能含 `rotation`，对比时如不一致检查该字段。
5. **original_title 匹配的尾串**：剥前缀用 `re.match(r"^\[.*\]_(.+)$")`，注意 `.*` 贪婪匹配，方括号内含中文逗号 `，` 也能正确处理。
6. **分类勿只看标签**：`vision_keywords` 常含多段拼合重复标签，判定一律以 `vision_description` 实际行为为准。
7. **决策树优先级不可跳**：自慰/性行为 > 恋物/露出；露阴≠自慰；足交为伴侣行为归 Partnered。
8. **每个视频唯一归位**：脚本要有 `id -> path` 完整性校验（总数=匹配数=已分类数）。
9. **移动用 `shutil.move`**（同盘等价 rename），目标已存在时先报错而非覆盖。

---

## 9. 复用脚本清单

本次产生的临时脚本（`C:\Users\28759\AppData\Local\Temp\opencode\`，可删除，逻辑已固化到本文档）：

| 脚本 | 用途 |
|---|---|
| `build_classification_list.py` | 生成待分类清单 MD（匹配+完整描述） |
| `build_classification.py` | 生成分类规划 MD（含 id→路径/理由 映射） |
| `verify_size.py` / `verify_size_orig.py` | 文件大小同一性校验 |
| `check_hash.py` | 检查 file_hash（确认全空） |
| `verify_probe.py` | ffprobe 时长+分辨率校验 |
| `move_classified.py` | 按路径移动文件 |

下次操作建议直接复用 `build_classification.py`（含完整 id→路径 映射字典）与 `move_classified.py`，只需更新文件夹与映射。
