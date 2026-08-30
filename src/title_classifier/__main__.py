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


def _import_core():
    """延迟导入核心模块（支持打包模式）"""
    try:
        from .core import Scanner, Refiner, VisionProcessor, Renamer
        from .utils.file_resolve import resolve_media_path
    except ImportError:
        from title_classifier.core import Scanner, Refiner, VisionProcessor, Renamer
        from title_classifier.utils.file_resolve import resolve_media_path
    return Scanner, Refiner, VisionProcessor, Renamer, resolve_media_path


def _import_atomic_csv():
    """延迟导入 atomic_csv 模块（支持打包模式）"""
    try:
        from .utils.atomic_csv import atomic_write_csv, atomic_append_csv
    except ImportError:
        from title_classifier.utils.atomic_csv import atomic_write_csv, atomic_append_csv
    return atomic_write_csv, atomic_append_csv


def setup_logging(verbose: bool = False, log_file: str = None):
    """设置日志

    日志默认输出到 logs/<日期>/ 目录，按天分目录。
    控制台输出 INFO 级别，文件输出 DEBUG 级别。
    """
    from datetime import datetime as dt

    level = logging.DEBUG if verbose else logging.INFO

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 文件 handler（默认输出到 logs/<日期>/ 目录）
    handlers = [console_handler]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    else:
        # 默认按天分目录
        today = dt.now().strftime("%Y-%m-%d")
        log_dir = Path("logs") / today
        log_dir.mkdir(parents=True, exist_ok=True)
        log_filename = dt.now().strftime("%H%M%S") + ".log"
        log_path = log_dir / log_filename
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
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
    Scanner, _, _, _, _ = _import_core()

    # 需要写 DB 时创建连接
    db = None
    if args.sync_db or args.force:
        from .core.db_store import MediaDB
        db = MediaDB()
        db.init_schema()

    scanner = Scanner(output_dir=args.output_dir, db_store=db)

    if args.sync_db:
        # DB 同步模式：扫描全部文件，同步到数据库，并生成待处理 CSV
        scanner.sync_db(target_dir=args.dir, exclude_dirs=args.exclude_dir,
                        deep=args.deep, exclude_images=args.exclude_images)
    else:
        # 普通扫描 / 强制重分类：生成 CSV
        output = scanner.scan(
            target_dir=args.dir,
            output_file=args.output,
            append=args.append,
            exclude_dirs=args.exclude_dir,
            force_reclassify=args.force,
            exclude_images=args.exclude_images,
        )
        if output:
            print(f"[完成] 结果已保存至: {output}")


def cmd_refine(args):
    """优化命令"""
    _, Refiner, _, _, _ = _import_core()
    refiner = Refiner(provider=args.provider)
    # TODO: 实现CSV读取和批量优化
    print("[待实现] AI标题优化")


def cmd_vision(args):
    """视觉识别命令"""
    import time
    import traceback

    from .utils.crash_reporter import write_crash_dump

    try:
        _, _, VisionProcessor, _, resolve_media_path = _import_core()
    except Exception:
        crash_file = write_crash_dump(*sys.exc_info())
        print(f"[致命错误] 导入失败，崩溃报告: {crash_file}", file=sys.stderr)
        traceback.print_exc()
        return

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

    # 初始化数据库
    from .core.db_store import MediaDB
    db = MediaDB()
    db.init_schema()

    # 加载配置
    from .utils.config import load_merged_config
    cfg = load_merged_config()

    # 初始化处理器
    processor = VisionProcessor(
        config=cfg,
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
        max_sample_frames=args.max_sample_frames,
        debug_dir=debug_dir,
        device=args.device,
        motion_detection=not args.no_motion_detection,
        motion_threshold=args.motion_threshold,
        backend=args.backend,
        db_store=db,
        use_scene_detection=not args.no_scene_detection,
        scene_threshold=args.scene_threshold,
        max_scenes=args.max_scenes,
        frames_per_scene=args.frames_per_scene,
    )

    if not processor.initialize():
        print("[错误] 初始化失败")
        return

    # 确保YOLO模型已加载到GPU（防止子线程重复加载）
    if processor.yolo_detector and not processor.yolo_detector._loaded:
        processor.yolo_detector.load_model()
    print(f"[就绪] 模型加载完成，开始处理")

    # 读取CSV
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if not rows:
        print("[警告] CSV为空")
        return

    # 确保字段存在
    for col in ["vision_description", "vision_keywords", "final_name", "srt_path", "vision_failed",
                 "clip_clothing", "clip_action", "clip_hairstyle", "clip_tags", "clip_tags_json",
                 "clip_confidence", "clip_detail"]:
        if col not in fieldnames:
            fieldnames.append(col)

    # 筛选需要视觉识别的记录
    retry_failed = getattr(args, "retry_failed", False)
    if retry_failed:
        # 重试失败行：只处理 vision_failed=true 的行
        pending = [
            (i, row)
            for i, row in enumerate(rows)
            if row.get("vision_failed", "").strip().lower() == "true"
        ]
    else:
        pending = [
            (i, row)
            for i, row in enumerate(rows)
            if row.get("needs_vision", "").strip().lower() == "true"
            and not row.get("vision_keywords", "").strip()
            and row.get("vision_failed", "").strip().lower() != "true"
        ]

    if args.all:
        pending = [
            (i, row)
            for i, row in enumerate(rows)
            if row.get("original_path", "").strip()
            and not row.get("vision_keywords", "").strip()
            and row.get("vision_failed", "").strip().lower() != "true"
        ]

    print(f"共 {len(rows)} 条记录，待处理 {len(pending)} 条")

    if not pending:
        print("[完成] 无需处理")
        return

    # SRT输出目录
    srt_dir = str(csv_path.parent / "subtitles")

    # 并发数
    max_workers = min(getattr(args, "concurrent", 1), len(pending))
    if max_workers < 1:
        max_workers = 1

    # 线程安全锁
    import threading
    lock = threading.Lock()
    counter = {"success": 0, "failed": 0, "done": 0}
    total = len(pending)

    def process_one(idx, row_idx, row):
        """处理单个视频（线程安全）"""
        original_path = row.get("original_path", "").strip()
        original_title = row.get("original_title", "").strip()
        final_name = row.get("final_name", "").strip()

        resolved = resolve_media_path(original_path, final_name, original_title)
        if not resolved:
            with lock:
                counter["done"] += 1
                counter["failed"] += 1
                print(f"[{counter['done']}/{total}] 跳过（文件不存在）: {original_title[:40]}")
            return
        if resolved != original_path:
            with lock:
                print(f"  [路径回退] {Path(original_path).name} → {Path(resolved).name}")
        original_path = resolved

        with lock:
            counter["done"] += 1
            cur = counter["done"]
            print(f"[{cur}/{total}] 处理: {original_title[:40]}")

        start_time = time.time()

        try:
            result = processor.process_and_save(
                video_path=original_path,
                title=original_title,
                original_title=original_title,
                srt_output_dir=srt_dir,
            )

            if "error" in result:
                with lock:
                    counter["failed"] += 1
                    rows[row_idx]["vision_failed"] = "true"
                    print(f"  [错误] {result['error']}")
                    # 保存失败标记到CSV
                    atomic_write_csv, _ = _import_atomic_csv()
                    atomic_write_csv(csv_path, rows, fieldnames)
                return

            with lock:
                rows[row_idx]["vision_description"] = result.get("description", "")
                rows[row_idx]["vision_keywords"] = result.get("keywords", "")
                rows[row_idx]["final_name"] = result.get("final_name", original_title)
                rows[row_idx]["srt_path"] = result.get("srt_path", "")
                rows[row_idx]["vision_failed"] = "false"

                video_summary = result.get("video_summary", {})
                if video_summary:
                    rows[row_idx]["human_detected"] = "true" if video_summary.get("has_person") else "false"
                    rows[row_idx]["detection_method"] = "yolo"

                # 保存CLIP详细置信度
                if result.get("clip_detail"):
                    rows[row_idx]["clip_clothing"] = result.get("clip_clothing", "")
                    rows[row_idx]["clip_action"] = result.get("clip_action", "")
                    rows[row_idx]["clip_hairstyle"] = result.get("clip_hairstyle", "")
                    rows[row_idx]["clip_tags"] = result.get("keywords", "")
                    rows[row_idx]["clip_tags_json"] = result.get("clip_tags_json", "")
                    rows[row_idx]["clip_confidence"] = str(result.get("clip_confidence", ""))
                    rows[row_idx]["clip_detail"] = result.get("clip_detail", "")

                elapsed = time.time() - start_time
                counter["success"] += 1
                print(f"  [完成] {elapsed:.1f}秒")
                print(f"  关键词: {result.get('keywords', '')[:60]}")
                print(f"  final_name: {result.get('final_name', '')[:60]}")

                if args.debug and result.get("debug_dir"):
                    print(f"  [调试] 数据已保存: {result['debug_dir']}")

                # 每处理完一条立即保存CSV
                atomic_write_csv, _ = _import_atomic_csv()
                atomic_write_csv(csv_path, rows, fieldnames)

        except Exception as e:
            with lock:
                counter["failed"] += 1
                print(f"  [错误] {e}")

    # 执行
    if max_workers > 1:
        print(f"并发模式: {max_workers} 个视频同时处理")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(process_one, idx, row_idx, row): idx
                for idx, (row_idx, row) in enumerate(pending)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    with lock:
                        counter["failed"] += 1
                        print(f"  [线程错误] {e}")
    else:
        for idx, (row_idx, row) in enumerate(pending):
            process_one(idx, row_idx, row)

    print(f"\n[统计]")
    print(f"  成功: {counter['success']}")
    print(f"  失败: {counter['failed']}")
    print(f"  结果已保存至: {csv_path}")

    # 自动导入数据库
    if getattr(args, "auto_import", False):
        print(f"\n[自动导入] 正在将 CSV 数据导入数据库...")
        from .core.db_store import MediaDB
        _db = MediaDB()
        _db.init_schema()
        stats = _db.import_csv(csv_path)
        print(f"  导入: {stats['imported']}, 更新: {stats['updated']}, 标签: {stats['tags_added']}")
    else:
        print(f"\n[提示] 建议运行以下命令将 CSV 数据导入数据库:")
        print(f"  python -m title-classifier db import")


def cmd_rename(args):
    """重命名命令"""
    _, _, _, Renamer, _ = _import_core()

    # 读取配置
    try:
        from .utils.config import load_config
    except ImportError:
        from title_classifier.utils.config import load_config
    config = load_config()
    renamer_config = config.get("renamer", {})

    use_rclone = getattr(args, 'use_rclone', False) or renamer_config.get("use_rclone", False)
    rclone_path = getattr(args, 'rclone_path', None) or renamer_config.get("rclone_path", "rclone")
    max_workers = getattr(args, 'max_workers', None) or renamer_config.get("max_workers", 5)

    renamer = Renamer(
        csv_path=args.csv,
        use_rclone=use_rclone,
        rclone_path=rclone_path,
        max_workers=int(max_workers)
    )
    stats = renamer.rename(dry_run=args.dry_run)

    error_val = stats.get("error")
    if error_val and isinstance(error_val, str):
        print(f"[错误] {error_val}")
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

    _, _, _, _, resolve_media_path = _import_core()

    try:
        from .utils.audio import AudioProcessor, load_audio_config
    except ImportError:
        from title_classifier.utils.audio import AudioProcessor, load_audio_config

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
        final_name = row.get("final_name", "").strip()

        # 解析实际文件路径（Stage2重命名后用final_name回退查找）
        resolved = resolve_media_path(original_path, final_name, original_title)
        if not resolved:
            print(f"[{idx+1}/{total}] 跳过（文件不存在）: {original_title[:40]}")
            failed += 1
            continue
        if resolved != original_path:
            print(f"  [路径回退] {Path(original_path).name} → {Path(resolved).name}")
        original_path = resolved

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
        atomic_write_csv, _ = _import_atomic_csv()
        atomic_write_csv(csv_path, rows, fieldnames)

    print(f"\n[统计]")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  结果已保存至: {csv_path}")


def cmd_gui(args):
    """GUI命令"""
    try:
        try:
            from .gui.app import main as gui_main
        except ImportError:
            from title_classifier.gui.app import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"[错误] GUI模块加载失败: {e}")
        print("请确保已安装 tkinter")


def cmd_db(args):
    """数据库管理命令"""
    from pathlib import Path

    db_path = str(Path("data/media.db"))
    try:
        from .core.db_store import MediaDB
    except ImportError:
        from title_classifier.core.db_store import MediaDB
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


def is_packaged() -> bool:
    """检测是否为 PyInstaller 打包后的可执行文件"""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def main():
    """主函数"""
    # 如果是打包后的可执行文件，直接启动 GUI
    if is_packaged():
        try:
            # 打包模式使用绝对导入
            from title_classifier.gui.app import main as gui_main
            gui_main()
        except Exception as e:
            print(f"[错误] GUI 启动失败: {e}")
            import traceback
            traceback.print_exc()
            input("按 Enter 键退出...")
        return

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
    scan_cmd.add_argument("--sync-db", action="store_true", help="同步数据库并生成待视觉识别CSV")
    scan_cmd.add_argument(
        "--deep", action="store_true",
        help="sync-db 时跳过快速路径，强制对全部文件做 ffprobe/哈希全量探测",
    )
    scan_cmd.add_argument(
        "--exclude-images", action="store_true",
        help="排除图片文件（jpg/png/webp 等），仅扫描/同步视频",
    )
    scan_cmd.set_defaults(func=cmd_scan)

    # refine 命令
    refine_cmd = subparsers.add_parser("refine", help="AI优化标题")
    refine_cmd.add_argument("-c", "--csv", default="data/output/title_review.csv", help="CSV文件路径")
    refine_cmd.add_argument("-p", "--provider", default="gcli", help="AI Provider")
    refine_cmd.set_defaults(func=cmd_refine)

    # vision 命令
    vision_cmd = subparsers.add_parser("vision", help="视觉识别")
    from .utils.config import load_merged_config, get_config_value
    cfg = load_merged_config()
    gv = lambda key, fallback: get_config_value(cfg, key, fallback)

    vision_cmd.add_argument("-c", "--csv", default="data/output/title_review.csv", help="CSV文件路径")
    vision_cmd.add_argument("-p", "--provider", default=gv("providers.stage_providers.vision", "gcli"), help="AI Provider")
    vision_cmd.add_argument("--use-yolo", action="store_true", help="使用YOLO姿态检测（分析人体姿态，智能选择代表性帧）")
    vision_cmd.add_argument("--comprehensive", action="store_true", help="全面分析模式（使用detect/pose/segment三个模型，投票决策）")
    vision_cmd.add_argument("--yolo-conf", type=float, default=gv("yolo.confidence", 0.5), help="YOLO置信度阈值")
    vision_cmd.add_argument("--use-clip", action="store_true", help="使用CLIP预分类")
    vision_cmd.add_argument("--clip-threshold", type=float, default=gv("clip.threshold", 0.25), help="CLIP置信度阈值")
    vision_cmd.add_argument("--max-image-size", type=int, default=gv("vision.max_image_size", 640), help="图片最大尺寸")
    vision_cmd.add_argument("--vlm-frames", type=int, default=gv("vision.vlm_frames", 10), help="VLM帧数（由采样间隔决定）")
    vision_cmd.add_argument("--analysis-step", type=float, default=gv("vision.analysis_step", 5.0), help="YOLO模式采样间隔（秒，默认5秒）")
    vision_cmd.add_argument("--max-sample-frames", type=int, default=gv("vision.max_sample_frames", 50), help="最大采样帧数上限（默认50，超过此数会均匀分布到整个视频）")
    vision_cmd.add_argument("--device", default=gv("general.device", "cpu"), choices=["auto", "cuda", "cpu"], help="推理设备（cpu=默认, auto=自动检测, cuda=GPU需手动安装CUDA版PyTorch）")
    vision_cmd.add_argument("--concurrent", type=int, default=4, help="并发处理视频数（默认4，CPU多核并行）")
    vision_cmd.add_argument("--backend", default=gv("yolo.backend", "auto"), choices=["auto", "openvino", "pytorch"], help="YOLO推理后端（auto=自动检测, openvino=Intel/AMD CPU加速, pytorch=原始PyTorch）")
    vision_cmd.add_argument("--no-motion-detection", action="store_true", help="禁用运动检测前置过滤（默认启用）")
    vision_cmd.add_argument("--motion-threshold", type=float, default=gv("vision.motion_threshold", 8.0), help="运动检测阈值（变化像素比例%%，低于此值跳过YOLO推理，默认8.0）")
    vision_cmd.add_argument("--no-scene-detection", action="store_true", help="禁用场景分段分析（默认启用，仅60s以上视频生效）")
    vision_cmd.add_argument("--scene-threshold", type=float, default=gv("scene_detection.threshold", 0.3), help="场景检测敏感度（0-1，越低切得越碎，默认0.3）")
    vision_cmd.add_argument("--max-scenes", type=int, default=gv("scene_detection.max_scenes", 10), help="最大场景段数（默认10，超出则合并相邻小场景）")
    vision_cmd.add_argument("--frames-per-scene", type=int, default=gv("scene_detection.frames_per_scene", 10), help="每场景取帧数（默认10）")
    vision_cmd.add_argument("--all", action="store_true", help="处理所有未识别的文件")
    vision_cmd.add_argument("--debug", action="store_true", help="启用调试模式，保存检测结果和VLM输入输出")
    vision_cmd.add_argument("--debug-dir", default="data/debug", help="调试数据输出目录")
    vision_cmd.add_argument("--retry-failed", action="store_true", help="重试之前失败的行（vision_failed=true）")
    vision_cmd.add_argument("--auto-import", action="store_true", help="处理完成后自动将 CSV 数据导入数据库")
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
    rename_cmd.add_argument("--use-rclone", action="store_true", help="使用 rclone 重命名（适合云盘）")
    rename_cmd.add_argument("--rclone-path", default=None, help="rclone 可执行文件路径")
    rename_cmd.add_argument("--max-workers", type=int, default=None, help="并行重命名线程数")
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

    # 安装崩溃报告钩子（确保日志系统就绪后安装）
    from .utils.crash_reporter import install_excepthook as _install_crash_hook, write_crash_dump as _write_dump
    _install_crash_hook()

    try:
        args.func(args)
    except Exception:
        import traceback as _tb
        crash_file = _write_dump(*sys.exc_info())
        print(f"[致命错误] 命令执行失败，崩溃报告: {crash_file}", file=sys.stderr)
        _tb.print_exc()


if __name__ == "__main__":
    main()
