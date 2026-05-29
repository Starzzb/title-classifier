"""
字幕封装工作流：将 SRT 字幕嵌入视频文件

用途：在重命名完成后，将生成的 SRT 字幕嵌入到视频文件中
用法：python scripts/workflow_mux.py <目录路径>

示例：
    python scripts/workflow_mux.py "D:/aria2/love"
    python scripts/workflow_mux.py "D:/aria2/love" --csv "data/output/love/title_review.csv"

适用场景：
    - 已完成视觉/音频识别和重命名
    - 想将 SRT 字幕嵌入到视频文件中（硬编码或软字幕）
    - 需要单独执行字幕封装步骤
"""

import sys
import time
import argparse
from pathlib import Path

from workflow_common import (
    get_output_dir, init_log, print_summary,
    step_mux_subtitles,
)


def main():
    parser = argparse.ArgumentParser(description="字幕封装工作流：SRT 嵌入视频")
    parser.add_argument("target_dir", help="目标视频目录")
    parser.add_argument("--csv", help="指定 CSV 文件路径（默认自动查找）")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.exists():
        print(f"[错误] 目录不存在: {target_dir}")
        sys.exit(1)

    if args.csv:
        csv_path = args.csv
        log_path = Path(csv_path).parent / "workflow_mux.log"
    else:
        output_dir = get_output_dir(str(target_dir))
        csv_path = str(output_dir / "title_review.csv")
        log_path = output_dir / "workflow_mux.log"

    if not Path(csv_path).exists():
        print(f"[错误] CSV 文件不存在: {csv_path}")
        print("提示：请先运行 workflow_scan_audio.py 或 workflow_vision.py 生成 CSV")
        sys.exit(1)

    init_log(log_path, str(target_dir), csv_path)
    start = time.time()

    print(f"[目录] {target_dir}")
    print(f"[CSV]  {csv_path}")
    print(f"[日志] {log_path}")

    step_mux_subtitles(csv_path, log_path)

    print_summary(time.time() - start, log_path)


if __name__ == "__main__":
    main()
