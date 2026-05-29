"""
完整工作流脚本：扫描 → 音频识别 → 视觉识别 → 字幕封装 → 确认 → 重命名

用法:
    python scripts/full_workflow.py <目录路径> [--dry-run]

示例:
    python scripts/full_workflow.py "D:/aria2/love"
    python scripts/full_workflow.py "D:/aria2/love" --dry-run  # 仅模拟重命名

多开并行: 每个目录自动分配独立 CSV 和日志，可同时运行多个窗口
    python scripts/full_workflow.py "D:/aria2/love"
    python scripts/full_workflow.py "D:/aria2/anime"  # 另一个窗口

断点续跑: 中途中断后重新运行同一目录，会自动跳过已完成的步骤

日志输出: 每个目录独立日志文件 data/output/<目录名>/workflow.log
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


def get_output_dir(target_dir: str) -> Path:
    """根据目标目录生成唯一的输出目录"""
    dir_name = Path(target_dir).resolve().name
    output_dir = PROJECT_ROOT / "data" / "output" / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_cmd(args: list, desc: str, log_path: Path):
    """运行 CLI 命令，实时输出到控制台和日志文件"""
    header = f"\n{'='*60}\n[步骤] {desc}\n  命令: {' '.join(str(a) for a in args)}\n{'='*60}\n"
    print(header)

    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(header)
        log_f.flush()

        proc = subprocess.Popen(
            [str(a) for a in args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()

        proc.wait()

        if proc.returncode != 0:
            warn = f"[警告] 命令返回非零退出码: {proc.returncode}\n"
            print(warn)
            log_f.write(warn)
            log_f.flush()

    return proc.returncode


def read_csv(csv_path: str) -> list:
    """读取 CSV 并返回行列表"""
    if not Path(csv_path).exists():
        return []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def check_audio_done(csv_path: str) -> bool:
    """检查音频识别是否全部完成"""
    rows = read_csv(csv_path)
    if not rows:
        return False
    video_ext = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts"}
    video_rows = [r for r in rows if Path(r.get("original_path", "")).suffix.lower() in video_ext]
    if not video_rows:
        return True
    return all(r.get("audio_recognized", "").strip().lower() == "true" for r in video_rows)


def check_vision_done(csv_path: str) -> bool:
    """检查视觉识别是否全部完成"""
    rows = read_csv(csv_path)
    if not rows:
        return False
    return all(r.get("vision_keywords", "").strip() for r in rows)


def step_scan(target_dir: str, csv_path: str, log_path: Path):
    """Step 1: 扫描目录"""
    cmd = [
        sys.executable, "-m", "title_classifier", "scan",
        "-d", target_dir,
        "-o", csv_path,
        "--force",
    ]
    return run_cmd(cmd, f"扫描目录: {target_dir}", log_path)


def step_audio(csv_path: str, log_path: Path):
    """Step 2: 音频识别（带断点续跑检测）"""
    if check_audio_done(csv_path):
        msg = "\n[跳过] 音频识别已完成，无需重复执行\n"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg)
        return 0

    cmd = [
        sys.executable, "-m", "title_classifier", "--log", str(log_path),
        "audio",
        "--all", "-p", "mimo", "-c", csv_path,
    ]
    return run_cmd(cmd, "音频识别 (mimo)", log_path)


def step_vision(csv_path: str, log_path: Path):
    """Step 3: 视觉识别（带断点续跑检测）"""
    if check_vision_done(csv_path):
        msg = "\n[跳过] 视觉识别已完成，无需重复执行\n"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg)
        return 0

    cmd = [
        sys.executable, "-m", "title_classifier", "--log", str(log_path),
        "vision",
        "--all", "-p", "gcli", "--use-yolo", "--comprehensive",
        "-c", csv_path,
    ]
    return run_cmd(cmd, "视觉识别 (gcli + YOLO全面分析)", log_path)


def step_mux_subtitles(csv_path: str, log_path: Path):
    """Step 6: 字幕封装到视频（覆写原视频）"""
    header = f"\n{'='*60}\n[步骤] 字幕封装\n{'='*60}\n"
    print(header)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(header)
        f.flush()

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
        "file_handling": "overwrite",
        "subtitle_processing": "direct",
    })

    rows = read_csv(csv_path)
    if not rows:
        msg = "[警告] CSV 为空，跳过字幕封装\n"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg)
        return 0

    srt_dir = str(Path(csv_path).parent / "subtitles")
    success = 0
    skipped = 0

    with open(log_path, "a", encoding="utf-8") as log_f:
        for row in rows:
            original_path = row.get("original_path", "").strip()
            final_name = row.get("final_name", "").strip()
            srt_path = row.get("srt_path", "").strip()

            if not original_path:
                skipped += 1
                continue

            # rename 后视频路径已变，用 final_name 构造
            if final_name:
                video_file = Path(original_path).parent / f"{final_name}{Path(original_path).suffix}"
            else:
                video_file = Path(original_path)

            if not video_file.exists():
                msg = f"  [跳过] 视频不存在: {video_file.name}\n"
                print(msg, end="")
                log_f.write(msg)
                skipped += 1
                continue

            if not srt_path:
                skipped += 1
                continue

            VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts"}
            if video_file.suffix.lower() not in VIDEO_EXT:
                skipped += 1
                continue

            if not Path(srt_path).is_absolute():
                srt_full = Path(srt_dir) / Path(srt_path).name
            else:
                srt_full = Path(srt_path)

            if not srt_full.exists():
                msg = f"  [跳过] SRT 不存在: {srt_full}\n"
                print(msg, end="")
                log_f.write(msg)
                skipped += 1
                continue

            result = muxer.mux_subtitle(str(video_file), str(srt_full))
            if result.get("success"):
                msg = f"  [完成] {video_file.name} (字幕已嵌入)\n"
                print(msg, end="")
                log_f.write(msg)
                success += 1
            else:
                msg = f"  [失败] {video_file.name}: {result.get('error', '未知错误')}\n"
                print(msg, end="")
                log_f.write(msg)

        summary = f"\n[字幕封装统计] 成功={success}, 跳过={skipped}\n"
        print(summary)
        log_f.write(summary)
        log_f.flush()

    return 0


def step_confirm_all(csv_path: str, log_path: Path):
    """Step 4: 将所有待确认记录标记为已确认"""
    header = f"\n{'='*60}\n[步骤] 确认所有记录\n{'='*60}\n"
    print(header)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(header)
        f.flush()

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

    msg = f"  已确认 {confirmed} 条记录 (共 {len(rows)} 条)\n"
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg)
    return 0


def step_rename(csv_path: str, log_path: Path, dry_run: bool = False):
    """Step 5: 执行重命名"""
    mode = "模拟重命名" if dry_run else "执行重命名"
    cmd = [sys.executable, "-m", "title_classifier", "rename", "-c", csv_path]
    if dry_run:
        cmd.append("--dry-run")
    return run_cmd(cmd, mode, log_path)


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

    # 输出目录：每个目标目录独立
    output_dir = get_output_dir(str(target_dir))
    csv_path = str(output_dir / "title_review.csv")
    log_path = output_dir / "workflow.log"

    # 写入日志头部
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[工作流启动] 目录: {target_dir}\n")
        f.write(f"[CSV] {csv_path}\n")
        f.write(f"[日志] {log_path}\n")
        f.write(f"[时间] {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    print(f"[目录] {target_dir}")
    print(f"[CSV]  {csv_path}")
    print(f"[日志] {log_path}")

    # Step 1: 扫描
    step_scan(str(target_dir), csv_path, log_path)

    # Step 2: 音频识别
    if not args.skip_audio:
        step_audio(csv_path, log_path)

    # Step 3: 视觉识别
    if not args.skip_vision:
        step_vision(csv_path, log_path)

    # Step 4: 确认所有记录
    step_confirm_all(csv_path, log_path)

    # Step 5: 重命名
    step_rename(csv_path, log_path, dry_run=args.dry_run)

    # Step 6: 字幕封装（覆写原视频）
    if not args.skip_mux:
        step_mux_subtitles(csv_path, log_path)

    # Step 6: 重命名
    step_rename(csv_path, log_path, dry_run=args.dry_run)

    elapsed = time.time() - start
    footer = f"\n{'='*60}\n[完成] 总耗时: {elapsed/60:.1f} 分钟\n{'='*60}\n"
    print(footer)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(footer)


if __name__ == "__main__":
    main()
