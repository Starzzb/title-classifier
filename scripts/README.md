# Scripts

独立脚本，不随主程序安装。按用途分三类。

## 工作流脚本

一键执行完整流程（扫描→音频→视觉→封装→确认→重命名）。

| 脚本 | 用途 |
|------|------|
| `full_workflow.py` | 完整工作流入口，一个目录跑全流程 |
| `workflow_common.py` | 工作流公共工具（输出目录、CSV读写、断点续跑） |
| `workflow_vision.py` | 单独执行视觉识别 |
| `workflow_scan_audio.py` | 单独执行音频识别 |
| `workflow_mux.py` | 单独执行字幕封装 |
| `workflow_rename.py` | 单独执行重命名 |
| `workflow_reclassify.py` | 重新分类（force_reclassify） |

```bash
# 完整流程
python scripts/full_workflow.py "G:/Download"

# 断点续跑：中途中断后重新运行同一目录，自动跳过已完成步骤
python scripts/full_workflow.py "G:/Download"

# 多开并行：每个目录独立CSV和日志
python scripts/full_workflow.py "G:/love"
python scripts/full_workflow.py "G:/anime"   # 另一个窗口
```

## 数据修复脚本

一次性迁移工具，修复 CSV 和文件名中的中括号问题。

| 脚本 | 用途 | 写入位置 |
|------|------|----------|
| `fix_bracket_only.py` | CSV `final_name` 列：`[标题]` → `标题` | 直接修改原CSV |
| `fix_bracket_filenames.py` | 磁盘文件名：`[标题].mp4` → `标题.mp4`，同步更新CSV | 重命名文件 + 修改原CSV |
| `fix_csv_paths.py` | CSV `original_path` 列：去掉 `[关键词]_` 前缀 | 直接修改原CSV |
| `fix_all_csvs.py` | 批量对所有CSV执行 `fix_bracket_only` | 直接修改原CSV |
| `extract_original.py` | CSV `final_name` 列：`[关键词]_原标题` → `原标题` | 直接修改原CSV |

```bash
# 模拟运行（不修改文件）
python scripts/fix_bracket_only.py data/output/Download/title_review.csv --dry-run
python scripts/fix_bracket_filenames.py "G:/好的" --dry-run

# 实际执行
python scripts/fix_bracket_only.py data/output/Download/title_review.csv
python scripts/fix_bracket_filenames.py "G:/好的" "G:/Download"
python scripts/fix_csv_paths.py
python scripts/fix_all_csvs.py

# 提取原标题（[关键词]_原标题 → 原标题）
python scripts/extract_original.py data/output/Download/title_review.csv --dry-run
python scripts/extract_original.py --all
```

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `download_models.py` | 下载模型文件 |
| `download_yolo_models.py` | 下载 YOLO 模型 |
| `download_clip.py` | 下载 CLIP 模型 |
| `import_csv.py` | 从 CSV 导入数据到 SQLite 数据库 |
| `test_prompts.py` | 测试 AI prompt 效果 |

```bash
python scripts/import_csv.py                           # 导入所有CSV到数据库
python scripts/import_csv.py --csv "data/output/love/title_review.csv"  # 指定CSV
```

## 输出目录

脚本产生的日志、临时报告等写入 `scripts/output/`。

主数据输出（CSV、SRT、调试数据）仍写入 `data/output/<目录名>/`。
