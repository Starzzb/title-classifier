"""CLI入口点"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

# Windows控制台UTF-8支持
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .core import Scanner, Refiner, VisionProcessor, Renamer


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_env(env_path: Path):
    """加载.env文件"""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def cmd_scan(args):
    """扫描命令"""
    scanner = Scanner(output_dir=args.output_dir)
    output = scanner.scan(
        target_dir=args.dir,
        output_file=args.output,
        append=args.append,
        exclude_dirs=args.exclude_dir,
        force_reclassify=args.force,
    )
    if output:
        print(f"[完成] 结果已保存至: {output}")


def cmd_refine(args):
    """优化命令"""
    refiner = Refiner(provider=args.provider)
    # TODO: 实现CSV读取和批量优化
    print("[待实现] AI标题优化")


def cmd_vision(args):
    """视觉识别命令"""
    import time

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[错误] CSV文件不存在: {csv_path}")
        return

    # 加载环境变量
    load_env(Path.cwd() / ".env")

    # 调试目录
    debug_dir = None
    if args.debug:
        debug_dir = args.debug_dir
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        print(f"[调试模式] 调试数据将保存到: {debug_dir}")

    # 初始化处理器（YOLO模式固定使用pose模型）
    processor = VisionProcessor(
        provider=args.provider,
        use_yolo=args.use_yolo,
        yolo_model="pose" if args.use_yolo else "detect",
        yolo_conf=args.yolo_conf,
        use_clip=args.use_clip,
        clip_threshold=args.clip_threshold,
        max_image_size=args.max_image_size,
        vlm_frames=args.vlm_frames,
        analysis_step=args.analysis_step,
        debug_dir=debug_dir,
    )

    if not processor.initialize():
        print("[错误] 初始化失败")
        return

    # 读取CSV
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if not rows:
        print("[警告] CSV为空")
        return

    # 确保字段存在
    for col in ["vision_description", "vision_keywords", "final_name", "srt_path"]:
        if col not in fieldnames:
            fieldnames.append(col)

    # 筛选需要视觉识别的记录
    pending = [
        (i, row)
        for i, row in enumerate(rows)
        if row.get("needs_vision", "").strip().lower() == "true"
        and not row.get("vision_keywords", "").strip()
    ]

    if args.all:
        pending = [
            (i, row)
            for i, row in enumerate(rows)
            if row.get("original_path", "").strip()
            and not row.get("vision_keywords", "").strip()
        ]

    print(f"共 {len(rows)} 条记录，待处理 {len(pending)} 条")

    if not pending:
        print("[完成] 无需处理")
        return

    # SRT输出目录
    srt_dir = str(csv_path.parent / "subtitles")

    # 批量处理
    total = len(pending)
    success = 0
    failed = 0

    for idx, (row_idx, row) in enumerate(pending):
        original_path = row.get("original_path", "").strip()
        original_title = row.get("original_title", "").strip()

        if not original_path or not Path(original_path).exists():
            print(f"[{idx+1}/{total}] 跳过（文件不存在）: {original_title[:40]}")
            failed += 1
            continue

        print(f"[{idx+1}/{total}] 处理: {original_title[:40]}")

        start_time = time.time()

        try:
            # 使用process_and_save一次性处理
            result = processor.process_and_save(
                video_path=original_path,
                title=original_title,
                original_title=original_title,
                srt_output_dir=srt_dir,
            )

            if "error" in result:
                print(f"  [错误] {result['error']}")
                failed += 1
                continue

            # 更新CSV行
            rows[row_idx]["vision_description"] = result.get("description", "")
            rows[row_idx]["vision_keywords"] = result.get("keywords", "")
            rows[row_idx]["final_name"] = result.get("final_name", original_title)
            rows[row_idx]["srt_path"] = result.get("srt_path", "")

            # 更新姿态信息
            video_summary = result.get("video_summary", {})
            if video_summary:
                rows[row_idx]["human_detected"] = "true" if video_summary.get("has_person") else "false"
                rows[row_idx]["detection_method"] = "yolo" if args.use_yolo else "uhd"

            elapsed = time.time() - start_time
            print(f"  [完成] {elapsed:.1f}秒")
            print(f"  关键词: {result.get('keywords', '')[:60]}")
            print(f"  final_name: {result.get('final_name', '')[:60]}")

            # 输出调试目录
            if args.debug and result.get("debug_dir"):
                print(f"  [调试] 数据已保存: {result['debug_dir']}")

            success += 1

        except Exception as e:
            print(f"  [错误] {e}")
            failed += 1

        # 每处理完一条立即保存CSV
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n[统计]")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  结果已保存至: {csv_path}")


def cmd_rename(args):
    """重命名命令"""
    renamer = Renamer(csv_path=args.csv)
    stats = renamer.rename(dry_run=args.dry_run)

    if "error" in stats:
        print(f"[错误] {stats['error']}")
        return

    print(f"\n[统计]")
    print(f"  已确认: {stats['confirmed']}")
    print(f"  成功: {stats['success']}")
    print(f"  跳过: {stats['skip']}")
    print(f"  冲突: {stats['conflict']}")
    print(f"  错误: {stats['error']}")


def cmd_audio(args):
    """音频识别命令"""
    import time
    from .utils.audio import AudioProcessor, load_audio_config

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[错误] CSV文件不存在: {csv_path}")
        return

    # 加载环境变量
    load_env(Path.cwd() / ".env")

    # 加载音频配置
    audio_config = load_audio_config()
    print(f"音频配置: 自适应分段={audio_config['adaptive_enabled']}, 静音跳过={audio_config['skip_silence']}, 阈值={audio_config['volume_threshold']}")

    # 初始化音频处理器
    audio_processor = AudioProcessor(provider=args.provider, config=audio_config)

    # 读取CSV
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if not rows:
        print("[警告] CSV为空")
        return

    # 确保字段存在
    for col in ["audio_recognized", "srt_path"]:
        if col not in fieldnames:
            fieldnames.append(col)

    # 筛选需要音频识别的记录
    pending = [
        (i, row)
        for i, row in enumerate(rows)
        if row.get("needs_vision", "").strip().lower() == "true"
        and row.get("audio_recognized", "").strip().lower() != "true"
    ]

    if args.all:
        pending = [
            (i, row)
            for i, row in enumerate(rows)
            if row.get("original_path", "").strip()
            and row.get("audio_recognized", "").strip().lower() != "true"
        ]

    print(f"共 {len(rows)} 条记录，待处理 {len(pending)} 条")

    if not pending:
        print("[完成] 无需处理")
        return

    # SRT输出目录
    srt_dir = str(csv_path.parent / "subtitles")
    Path(srt_dir).mkdir(parents=True, exist_ok=True)

    # 批量处理
    total = len(pending)
    success = 0
    failed = 0

    for idx, (row_idx, row) in enumerate(pending):
        original_path = row.get("original_path", "").strip()
        original_title = row.get("original_title", "").strip()

        if not original_path or not Path(original_path).exists():
            print(f"[{idx+1}/{total}] 跳过（文件不存在）: {original_title[:40]}")
            failed += 1
            continue

        print(f"[{idx+1}/{total}] 处理: {original_title[:40]}")

        start_time = time.time()

        try:
            # 生成SRT路径（使用原文件名）
            srt_name = Path(original_title).stem + ".srt"
            srt_path = str(Path(srt_dir) / srt_name)

            # 调用音频处理器
            result_path = audio_processor.process_video(
                video_path=original_path,
                output_srt=srt_path,
            )

            if result_path:
                # 更新CSV行
                rows[row_idx]["audio_recognized"] = "true"
                rows[row_idx]["srt_path"] = result_path

                elapsed = time.time() - start_time
                print(f"  [完成] {elapsed:.1f}秒")
                print(f"  SRT: {result_path}")
                success += 1
            else:
                print(f"  [警告] 音频识别无结果")
                failed += 1

        except Exception as e:
            print(f"  [错误] {e}")
            failed += 1

        # 每处理完一条立即保存CSV
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n[统计]")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  结果已保存至: {csv_path}")


def cmd_gui(args):
    """GUI命令"""
    try:
        from .gui.app import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"[错误] GUI模块加载失败: {e}")
        print("请确保已安装 tkinter")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        prog="title-classifier",
        description="视频标题分类和重命名工具",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # scan 命令
    scan_cmd = subparsers.add_parser("scan", help="扫描目录或单个文件")
    scan_cmd.add_argument("-d", "--dir", required=True, help="目标目录或单个媒体文件路径")
    scan_cmd.add_argument("-o", "--output", help="输出文件路径")
    scan_cmd.add_argument("--output-dir", default="data/output", help="输出目录")
    scan_cmd.add_argument("-a", "--append", action="store_true", help="追加模式")
    scan_cmd.add_argument("--exclude-dir", nargs="*", default=[], help="排除的目录")
    scan_cmd.add_argument("--force", action="store_true", help="强制重新分类")
    scan_cmd.set_defaults(func=cmd_scan)

    # refine 命令
    refine_cmd = subparsers.add_parser("refine", help="AI优化标题")
    refine_cmd.add_argument("-c", "--csv", default="data/output/title_review.csv", help="CSV文件路径")
    refine_cmd.add_argument("-p", "--provider", default="gcli", help="AI Provider")
    refine_cmd.set_defaults(func=cmd_refine)

    # vision 命令
    vision_cmd = subparsers.add_parser("vision", help="视觉识别")
    vision_cmd.add_argument("-c", "--csv", default="data/output/title_review.csv", help="CSV文件路径")
    vision_cmd.add_argument("-p", "--provider", default="gcli", help="AI Provider")
    vision_cmd.add_argument("--use-yolo", action="store_true", help="使用YOLO姿态检测（分析人体姿态，智能选择代表性帧）")
    vision_cmd.add_argument("--yolo-conf", type=float, default=0.5, help="YOLO置信度阈值")
    vision_cmd.add_argument("--use-clip", action="store_true", help="使用CLIP预分类")
    vision_cmd.add_argument("--clip-threshold", type=float, default=0.25, help="CLIP置信度阈值")
    vision_cmd.add_argument("--max-image-size", type=int, default=800, help="图片最大尺寸")
    vision_cmd.add_argument("--vlm-frames", type=int, default=10, help="VLM帧数（UHD模式使用，YOLO模式由采样间隔决定）")
    vision_cmd.add_argument("--analysis-step", type=float, default=2.0, help="YOLO模式采样间隔（秒，默认2秒）")

    vision_cmd.add_argument("--all", action="store_true", help="处理所有未识别的文件")
    vision_cmd.add_argument("--debug", action="store_true", help="启用调试模式，保存检测结果和VLM输入输出")
    vision_cmd.add_argument("--debug-dir", default="data/debug", help="调试数据输出目录")
    vision_cmd.set_defaults(func=cmd_vision)

    # audio 命令
    audio_cmd = subparsers.add_parser("audio", help="音频识别（为视觉识别做准备）")
    audio_cmd.add_argument("-c", "--csv", default="data/output/title_review.csv", help="CSV文件路径")
    audio_cmd.add_argument("-p", "--provider", default="mimo", help="AI Provider")
    audio_cmd.add_argument("--all", action="store_true", help="处理所有未识别的文件")
    audio_cmd.set_defaults(func=cmd_audio)

    # rename 命令
    rename_cmd = subparsers.add_parser("rename", help="执行重命名")
    rename_cmd.add_argument("-c", "--csv", default="data/output/title_review.csv", help="CSV文件路径")
    rename_cmd.add_argument("--dry-run", action="store_true", help="模拟运行")
    rename_cmd.set_defaults(func=cmd_rename)

    # gui 命令
    gui_cmd = subparsers.add_parser("gui", help="启动图形界面")
    gui_cmd.set_defaults(func=cmd_gui)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
