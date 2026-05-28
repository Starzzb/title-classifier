"""
完整工作流脚本：扫描 → 音频识别 → 视觉识别 → 字幕封装 → 确认 → 重命名

用法:
    python scripts/full_workflow.py <目录路径> [--dry-run]

示例:
    python scripts/full_workflow.py "D:/aria2/love"
    python scripts/full_workflow.py "D:/aria2/love" --dry-run  # 仅模拟重命名

多开并行: 每个目录自动分配独立 CSV，可同时运行多个窗口
    python scripts/full_workflow.py "D:/aria2/love"
    python scripts/full_workflow.py "D:/aria2/anime"  # 另一个窗口
"""

import sys
import os
import csv
import time
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def get_csv_path(target_dir: str) -> str:
    """根据目标目录生成唯一的 CSV 路径，避免多实例冲突"""
    dir_name = Path(target_dir).resolve().name
    csv_dir = PROJECT_ROOT / "data" / "output" / dir_name
    csv_dir.mkdir(parents=True, exist_ok=True)
    return str(csv_dir / "title_review.csv")


def run_cmd(args: list, desc: str, timeout: int = 3600):
    """运行 CLI 命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"[步骤] {desc}")
    print(f"  命令: {' '.join(str(a) for a in args)}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [str(a) for a in args],
        cwd=str(PROJECT_ROOT),
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"[警告] 命令返回非零退出码: {result.returncode}")
    return result.returncode


def step_scan(target_dir: str, csv_path: str):
    """Step 1: 扫描目录"""
    cmd = [
        sys.executable, "-m", "title_classifier", "scan",
        "-d", target_dir,
        "-o", csv_path,
        "--force",
    ]
    return run_cmd(cmd, f"扫描目录: {target_dir}")


def step_audio(csv_path: str):
    """Step 2: 音频识别"""
    cmd = [
        sys.executable, "-m", "title_classifier", "audio",
        "--all", "-p", "mimo", "-c", csv_path,
    ]
    return run_cmd(cmd, "音频识别 (mimo)", timeout=1800)


def step_vision(csv_path: str):
    """Step 3: 视觉识别"""
    cmd = [
        sys.executable, "-m", "title_classifier", "vision",
        "--all", "-p", "gcli", "--use-yolo", "--comprehensive",
        "-c", csv_path,
    ]
    return run_cmd(cmd, "视觉识别 (gcli + YOLO全面分析)", timeout=3600)


def step_mux_subtitles(csv_path: str):
    """Step 4: 字幕封装到视频"""
    print(f"\n{'='*60}")
    print(f"[步骤] 字幕封装")
    print(f"{'='*60}\n")

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

    from title_classifier.utils.muxer import SubtitleMuxer

    muxer = SubtitleMuxer({
        "output_format": "auto",
        "file_handling": "new",
        "subtitle_processing": "direct",
    })

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[警告] CSV 为空，跳过字幕封装")
        return 0

    srt_dir = str(Path(csv_path).parent / "subtitles")
    success = 0
    skipped = 0

    for row in rows:
        original_path = row.get("original_path", "").strip()
        srt_path = row.get("srt_path", "").strip()

        if not original_path or not Path(original_path).exists():
            skipped += 1
            continue

        if not srt_path:
            skipped += 1
            continue

        VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts"}
        if Path(original_path).suffix.lower() not in VIDEO_EXT:
            skipped += 1
            continue

        if not Path(srt_path).is_absolute():
            srt_full = Path(srt_dir) / Path(srt_path).name
        else:
            srt_full = Path(srt_path)

        if not srt_full.exists():
            srt_same_dir = Path(original_path).with_suffix(".srt")
            if srt_same_dir.exists():
                srt_full = srt_same_dir
            else:
                print(f"  [跳过] SRT 不存在: {srt_full}")
                skipped += 1
                continue

        result = muxer.mux_subtitle(original_path, str(srt_full))
        if result.get("success"):
            output = result.get("output_path", "")
            print(f"  [完成] {Path(original_path).name} -> {Path(output).name if output else '?'}")
            success += 1
        else:
            print(f"  [失败] {Path(original_path).name}: {result.get('error', '未知错误')}")

    print(f"\n[字幕封装统计] 成功={success}, 跳过={skipped}")
    return 0


def step_confirm_all(csv_path: str):
    """Step 5: 将所有待确认记录标记为已确认"""
    print(f"\n{'='*60}")
    print(f"[步骤] 确认所有记录")
    print(f"{'='*60}\n")

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "review_status" not in fieldnames:
        fieldnames.append("review_status")

    confirmed = 0
    for row in rows:
        if row.get("review_status", "").strip() != "已确认":
            row["review_status"] = "已确认"
            confirmed += 1

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  已确认 {confirmed} 条记录 (共 {len(rows)} 条)")
    return 0


def step_rename(csv_path: str, dry_run: bool = False):
    """Step 6: 执行重命名"""
    mode = "模拟重命名" if dry_run else "执行重命名"
    cmd = [sys.executable, "-m", "title_classifier", "rename", "-c", csv_path]
    if dry_run:
        cmd.append("--dry-run")
    return run_cmd(cmd, mode)


def main():
    parser = argparse.ArgumentParser(description="完整工作流：扫描→音频→视觉→字幕封装→确认→重命名")
    parser.add_argument("target_dir", help="目标视频目录")
    parser.add_argument("--dry-run", action="store_true", help="仅模拟重命名，不实际执行")
    parser.add_argument("--skip-audio", action="store_true", help="跳过音频识别")
    parser.add_argument("--skip-vision", action="store_true", help="跳过视觉识别")
    parser.add_argument("--skip-mux", action="store_true", help="跳过字幕封装")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.exists():
        print(f"[错误] 目录不存在: {target_dir}")
        sys.exit(1)

    csv_path = get_csv_path(str(target_dir))
    print(f"[目录] {target_dir}")
    print(f"[CSV]  {csv_path}")

    start = time.time()

    # Step 1: 扫描
    step_scan(str(target_dir), csv_path)

    # Step 2: 音频识别
    if not args.skip_audio:
        step_audio(csv_path)

    # Step 3: 视觉识别
    if not args.skip_vision:
        step_vision(csv_path)

    # Step 4: 字幕封装
    if not args.skip_mux:
        step_mux_subtitles(csv_path)

    # Step 5: 确认所有记录
    step_confirm_all(csv_path)

    # Step 6: 重命名
    step_rename(csv_path, dry_run=args.dry_run)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"[完成] 总耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
