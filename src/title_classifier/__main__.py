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


def setup_logging(verbose: bool = False, log_file: str = None):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
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

    # 确定YOLO模型列表
    if args.use_yolo and args.comprehensive:
        # 全面分析模式：使用三个模型
        yolo_models = ["detect", "pose", "segment"]
        print("[全面分析模式] 使用三个YOLO模型: detect, pose, segment")
    elif args.use_yolo:
        # 基础YOLO模式：只使用pose模型
        yolo_models = ["pose"]
    else:
        yolo_models = ["pose"]

    # 初始化处理器
    processor = VisionProcessor(
        provider=args.provider,
        use_yolo=args.use_yolo,
        yolo_model="pose" if args.use_yolo else "detect",
        yolo_models=yolo_models,
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
                rows[row_idx]["detection_method"] = "yolo"

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

        # 每处理完一条立即保存CSV（原子化写入）
        from .utils.atomic_csv import atomic_write_csv
        atomic_write_csv(csv_path, rows, fieldnames)

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

    # 视频文件扩展名（音频识别只处理视频，跳过图片）
    VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts"}

    # 筛选需要音频识别的记录
    pending = [
        (i, row)
        for i, row in enumerate(rows)
        if row.get("needs_vision", "").strip().lower() == "true"
        and row.get("audio_recognized", "").strip().lower() != "true"
        and Path(row.get("original_path", "")).suffix.lower() in VIDEO_EXT
    ]

    if args.all:
        pending = [
            (i, row)
            for i, row in enumerate(rows)
            if row.get("original_path", "").strip()
            and row.get("audio_recognized", "").strip().lower() != "true"
            and Path(row.get("original_path", "")).suffix.lower() in VIDEO_EXT
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

        # 每处理完一条立即保存CSV（原子化写入）
        from .utils.atomic_csv import atomic_write_csv
        atomic_write_csv(csv_path, rows, fieldnames)

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


def cmd_db(args):
    """数据库管理命令"""
    from pathlib import Path

    db_path = str(Path("data/media.db"))
    from .core.db_store import MediaDB
    db = MediaDB(db_path)

    action = getattr(args, "db_action", None)

    if action == "init":
        db.init_schema()
        print(f"[完成] 数据库初始化: {db_path}")
        print(f"[目录] {Path('data/covers').mkdir(exist_ok=True) or 'data/covers'}")

    elif action == "import":
        db.init_schema()
        if args.csv:
            stats = db.import_csv(args.csv)
            print(f"[导入] {args.csv}")
        else:
            stats = db.import_all_csvs()
            print("[导入] 所有 CSV 文件")
        print(f"  新增: {stats['imported']}, 更新: {stats['updated']}, 标签: {stats['tags_added']}")

    elif action == "list":
        rows = db.list_all(limit=args.limit, offset=args.offset)
        total = db.count()
        print(f"[记录] 共 {total} 条，显示 {len(rows)} 条\n")
        for r in rows:
            tags = db.get_tags(r["id"])
            tags_str = ", ".join(tags[:3]) if tags else ""
            print(f"  [{r['id']}] {r['final_name'] or r['original_title'][:40]}")
            print(f"       路径: {r['original_path'][:60]}...")
            if tags_str:
                print(f"       标签: {tags_str}")
            print()

    elif action == "search":
        rows = db.search(query=args.query, tags=[args.tag] if args.tag else None, source=args.source)
        print(f"[搜索] 找到 {len(rows)} 条记录\n")
        for r in rows[:20]:
            print(f"  [{r['id']}] {r['final_name'] or r['original_title'][:40]}")
            print(f"       {r['original_path'][:70]}")
            print()

    elif action == "show":
        r = db.get_media(args.media_id)
        if not r:
            print(f"[错误] 记录不存在: {args.media_id}")
            return
        print(f"[记录 {r['id']}]")
        print(f"  原始标题: {r['original_title']}")
        print(f"  当前路径: {r['current_path']}")
        print(f"  最终名称: {r['final_name']}")
        print(f"  描述: {r['vision_description'] or '(无)'}")
        tags = db.get_tags(r['id'])
        print(f"  标签: {', '.join(tags) if tags else '(无)'}")
        print(f"  人体检测: {'是' if r['human_detected'] else '否'}")
        print(f"  音频识别: {'是' if r['audio_recognized'] else '否'}")
        print(f"  审核状态: {r['review_status']}")
        print(f"  创建时间: {r['created_at']}")
        print(f"  更新时间: {r['updated_at']}")
        frames = db.get_vlm_frames(r['id'])
        if frames:
            print(f"  VLM帧: {len(frames)} 张")

    elif action == "history":
        changes = db.get_changes(args.media_id)
        if not changes:
            print(f"[记录 {args.media_id}] 无改动历史")
            return
        print(f"[记录 {args.media_id}] 改动历史 ({len(changes)} 条)\n")
        for c in changes:
            print(f"  {c['changed_at']} [{c['change_source']}]")
            print(f"    {c['field_name']}: {c['old_value'][:30]} → {c['new_value'][:30]}")
            print()

    elif action == "stats":
        stats = db.get_stats()
        print(f"[统计]")
        print(f"  媒体文件: {stats['total_media']}")
        print(f"  视频: {stats['videos']}")
        print(f"  图片: {stats['images']}")
        print(f"  标签: {stats['total_tags']}")
        print(f"  改动记录: {stats['total_changes']}")
        print(f"  VLM帧: {stats['total_frames']}")
        if stats['top_tags']:
            print(f"\n  热门标签:")
            for t in stats['top_tags'][:5]:
                print(f"    {t['name']}: {t['count']} 次")

    db.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        prog="title-classifier",
        description="视频标题分类和重命名工具",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--log", help="日志文件路径（不指定则只输出到控制台）")

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
    vision_cmd.add_argument("--comprehensive", action="store_true", help="全面分析模式（使用detect/pose/segment三个模型，投票决策）")
    vision_cmd.add_argument("--yolo-conf", type=float, default=0.5, help="YOLO置信度阈值")
    vision_cmd.add_argument("--use-clip", action="store_true", help="使用CLIP预分类")
    vision_cmd.add_argument("--clip-threshold", type=float, default=0.25, help="CLIP置信度阈值")
    vision_cmd.add_argument("--max-image-size", type=int, default=640, help="图片最大尺寸")
    vision_cmd.add_argument("--vlm-frames", type=int, default=10, help="VLM帧数（由采样间隔决定）")
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

    # db 命令
    db_cmd = subparsers.add_parser("db", help="数据库管理")
    db_sub = db_cmd.add_subparsers(dest="db_action", help="数据库操作")
    db_init = db_sub.add_parser("init", help="初始化数据库")
    db_init.set_defaults(func=cmd_db)
    db_import = db_sub.add_parser("import", help="从CSV导入数据")
    db_import.add_argument("--csv", help="指定CSV文件路径")
    db_import.add_argument("--all", action="store_true", help="导入所有CSV（默认）")
    db_import.set_defaults(func=cmd_db)
    db_list = db_sub.add_parser("list", help="列出记录")
    db_list.add_argument("--limit", type=int, default=20, help="显示数量")
    db_list.add_argument("--offset", type=int, default=0, help="偏移量")
    db_list.set_defaults(func=cmd_db)
    db_search = db_sub.add_parser("search", help="搜索记录")
    db_search.add_argument("--query", help="搜索关键词")
    db_search.add_argument("--tag", help="按标签搜索")
    db_search.add_argument("--source", help="按来源筛选")
    db_search.set_defaults(func=cmd_db)
    db_show = db_sub.add_parser("show", help="查看单条记录")
    db_show.add_argument("media_id", type=int, help="记录ID")
    db_show.set_defaults(func=cmd_db)
    db_history = db_sub.add_parser("history", help="查看改动历史")
    db_history.add_argument("media_id", type=int, help="记录ID")
    db_history.set_defaults(func=cmd_db)
    db_stats = db_sub.add_parser("stats", help="统计信息")
    db_stats.set_defaults(func=cmd_db)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging(args.verbose, args.log)
    args.func(args)


if __name__ == "__main__":
    main()
