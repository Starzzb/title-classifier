"""
重命名工作流：确认 → 重命名

用途：在 GUI 中手动编辑完成后，执行确认和重命名
用法：python scripts/workflow_rename.py <目录路径> [--dry-run]

示例：
    python scripts/workflow_rename.py "D:/aria2/love"
    python scripts/workflow_rename.py "D:/aria2/love" --dry-run  # 仅模拟

适用场景：
    - 在 Stage1b GUI 中手动编辑了标题
    - 在 Stage1b GUI 中切换了 needs_vision/audio_recognized
    - 想先模拟重命名看看效果，再实际执行
"""

import sys
import time
import argparse
from pathlib import Path

from workflow_common import (
    get_output_dir, init_log, print_summary,
    step_confirm_all, step_rename,
)


def main():
    parser = argparse.ArgumentParser(description="重命名工作流：确认 → 重命名")
    parser.add_argument("target_dir", help="目标视频目录")
    parser.add_argument("--dry-run", action="store_true", help="仅模拟重命名，不实际执行")
    parser.add_argument("--csv", help="指定 CSV 文件路径（默认自动查找）")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.exists():
        print(f"[错误] 目录不存在: {target_dir}")
        sys.exit(1)

    if args.csv:
        csv_path = args.csv
        log_path = Path(csv_path).parent / "workflow_rename.log"
    else:
        output_dir = get_output_dir(str(target_dir))
        csv_path = str(output_dir / "title_review.csv")
        log_path = output_dir / "workflow_rename.log"

    if not Path(csv_path).exists():
        print(f"[错误] CSV 文件不存在: {csv_path}")
        print("提示：请先运行 workflow_scan_audio.py 或 workflow_vision.py 生成 CSV")
        sys.exit(1)

    init_log(log_path, str(target_dir), csv_path)
    start = time.time()

    print(f"[目录] {target_dir}")
    print(f"[CSV]  {csv_path}")
    print(f"[日志] {log_path}")

    step_confirm_all(csv_path, log_path)
    step_rename(csv_path, log_path, dry_run=args.dry_run)

    print_summary(time.time() - start, log_path)


if __name__ == "__main__":
    main()
