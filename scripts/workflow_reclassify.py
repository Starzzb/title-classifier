"""
重分类工作流：强制扫描（剥离括号前缀） → 音频 → 视觉 → 确认 → 重命名

用途：对已有 [关键词]_ 前缀的文件重新进行分类
用法：python scripts/workflow_reclassify.py <目录路径> [--dry-run]

示例：
    python scripts/workflow_reclassify.py "D:/aria2/love"
    python scripts/workflow_reclassify.py "D:/aria2/love" --skip-audio
    python scripts/workflow_reclassify.py "D:/aria2/love" --dry-run

与 full_workflow.py 的区别：
    - 使用 --force 扫描，自动剥离已有的 [关键词]_ 前缀
    - 以干净的文件名重新进行视觉/音频识别
    - 不会产生嵌套括号

适用场景：
    - 之前分类结果不理想，想重新识别
    - 从其他地方复制来的文件带有旧的关键词前缀
    - 想用新的 AI 模型重新分析
"""

import sys
import time
import argparse
from pathlib import Path

from workflow_common import (
    get_output_dir, init_log, print_summary,
    step_scan, step_audio, step_vision,
    step_confirm_all, step_rename, step_mux_subtitles,
)


def main():
    parser = argparse.ArgumentParser(description="重分类工作流：剥离括号前缀 → 重新识别 → 重命名")
    parser.add_argument("target_dir", help="目标视频目录")
    parser.add_argument("--dry-run", action="store_true", help="仅模拟重命名，不实际执行")
    parser.add_argument("--skip-audio", action="store_true", help="跳过音频识别")
    parser.add_argument("--skip-vision", action="store_true", help="跳过视觉识别")
    parser.add_argument("--skip-mux", action="store_true", help="跳过字幕封装")
    parser.add_argument("--audio-provider", default="mimo", help="音频识别 Provider (默认: mimo)")
    parser.add_argument("--vision-provider", default="gcli", help="视觉识别 Provider (默认: gcli)")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.exists():
        print(f"[错误] 目录不存在: {target_dir}")
        sys.exit(1)

    output_dir = get_output_dir(str(target_dir))
    csv_path = str(output_dir / "title_review.csv")
    log_path = output_dir / "workflow_reclassify.log"

    init_log(log_path, str(target_dir), csv_path)
    start = time.time()

    print(f"[目录] {target_dir}")
    print(f"[CSV]  {csv_path}")
    print(f"[日志] {log_path}")
    print(f"[模式] 重分类（强制扫描，剥离括号前缀）")

    # Step 1: 强制扫描（剥离括号前缀）
    step_scan(str(target_dir), csv_path, log_path, force=True)

    # Step 2: 音频识别
    if not args.skip_audio:
        step_audio(csv_path, log_path, provider=args.audio_provider)

    # Step 3: 视觉识别
    if not args.skip_vision:
        step_vision(csv_path, log_path, provider=args.vision_provider)

    # Step 4: 确认 + 重命名
    step_confirm_all(csv_path, log_path)
    step_rename(csv_path, log_path, dry_run=args.dry_run)

    # Step 5: 字幕封装
    if not args.skip_mux:
        step_mux_subtitles(csv_path, log_path)

    # Step 6: 最终重命名（处理字幕封装后的文件名变化）
    step_rename(csv_path, log_path, dry_run=args.dry_run)

    print_summary(time.time() - start, log_path)


if __name__ == "__main__":
    main()
