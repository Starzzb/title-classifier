"""
音频准备工作流：扫描 → 音频识别

用途：先完成音频转录，为后续视觉识别提供音频上下文
用法：python scripts/workflow_scan_audio.py <目录路径>

示例：
    python scripts/workflow_scan_audio.py "D:/aria2/love"
    python scripts/workflow_scan_audio.py "D:/aria2/love" --provider gcli

断点续跑：中途中断后重新运行，会自动跳过已完成的音频识别
"""

import sys
import time
import argparse
from pathlib import Path

from workflow_common import (
    get_output_dir, init_log, print_summary,
    step_scan, step_audio,
)


def main():
    parser = argparse.ArgumentParser(description="音频准备工作流：扫描 → 音频识别")
    parser.add_argument("target_dir", help="目标视频目录")
    parser.add_argument("--provider", default="mimo", help="音频识别 Provider (默认: mimo)")
    parser.add_argument("--force", action="store_true", help="强制重新扫描（剥离已有 [kw]_ 前缀）")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.exists():
        print(f"[错误] 目录不存在: {target_dir}")
        sys.exit(1)

    output_dir = get_output_dir(str(target_dir))
    csv_path = str(output_dir / "title_review.csv")
    log_path = output_dir / "workflow_audio.log"

    init_log(log_path, str(target_dir), csv_path)
    start = time.time()

    print(f"[目录] {target_dir}")
    print(f"[CSV]  {csv_path}")
    print(f"[日志] {log_path}")

    step_scan(str(target_dir), csv_path, log_path, force=args.force)
    step_audio(csv_path, log_path, provider=args.provider)

    print_summary(time.time() - start, log_path)


if __name__ == "__main__":
    main()
