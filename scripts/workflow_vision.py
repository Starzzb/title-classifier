"""
视觉识别工作流：扫描 → 视觉识别 → 确认 → 重命名

用途：跳过音频识别，直接进行视觉分析和重命名
用法：python scripts/workflow_vision.py <目录路径> [--dry-run]

示例：
    python scripts/workflow_vision.py "D:/aria2/love"
    python scripts/workflow_vision.py "D:/aria2/love" --dry-run  # 仅模拟重命名
    python scripts/workflow_vision.py "D:/aria2/love" --provider zhipu

适用场景：
    - 文件名有意义，不需要音频上下文
    - 只需要视觉识别关键词
    - 图片目录（无音频）
"""

import sys
import time
import argparse
from pathlib import Path

from workflow_common import (
    get_output_dir, init_log, print_summary,
    step_scan, step_vision, step_confirm_all, step_rename,
)


def main():
    parser = argparse.ArgumentParser(description="视觉识别工作流：扫描 → 视觉识别 → 确认 → 重命名")
    parser.add_argument("target_dir", help="目标视频目录")
    parser.add_argument("--provider", default="gcli", help="视觉识别 Provider (默认: gcli)")
    parser.add_argument("--force", action="store_true", help="强制重新扫描（剥离已有 [kw]_ 前缀）")
    parser.add_argument("--dry-run", action="store_true", help="仅模拟重命名，不实际执行")
    parser.add_argument("--skip-rename", action="store_true", help="跳过重命名（只做识别）")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.exists():
        print(f"[错误] 目录不存在: {target_dir}")
        sys.exit(1)

    output_dir = get_output_dir(str(target_dir))
    csv_path = str(output_dir / "title_review.csv")
    log_path = output_dir / "workflow_vision.log"

    init_log(log_path, str(target_dir), csv_path)
    start = time.time()

    print(f"[目录] {target_dir}")
    print(f"[CSV]  {csv_path}")
    print(f"[日志] {log_path}")

    step_scan(str(target_dir), csv_path, log_path, force=args.force)
    step_vision(csv_path, log_path, provider=args.provider)

    if not args.skip_rename:
        step_confirm_all(csv_path, log_path)
        step_rename(csv_path, log_path, dry_run=args.dry_run)

    print_summary(time.time() - start, log_path)


if __name__ == "__main__":
    main()
