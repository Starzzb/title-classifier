"""文件扫描模块 - 简化版：只判断是否需要视觉识别"""

import re
import csv
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Set, Optional

logger = logging.getLogger(__name__)

# 过滤非关键依赖警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# 配置
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

def _is_thumbnail(file_path: Path) -> bool:
    """判断文件是否位于以 . 开头的隐藏目录中（如 .thumbs、.thumbnails 等）"""
    return any(part.startswith(".") for part in file_path.parent.parts if part not in ("/", "\\", ""))


def has_chinese(text: str) -> bool:
    """检查是否包含中文"""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def is_already_classified(name: str) -> bool:
    """检查是否已分类"""
    match = re.match(r"^\[([^\]]+)\]", name)
    if not match:
        return False
    tag_content = match.group(1)
    if tag_content == "未分类":
        return False
    return True


def strip_bracket_prefix(name: str) -> str:
    """去除文件名开头的 [关键词]_ 前缀，还原干净文件名"""
    return re.sub(r"^\[[^\]]*\]_?", "", name, count=1)


def is_needs_vision(original: str) -> bool:
    """判断是否需要视觉识别"""
    orig = original.strip()

    # 1. IMG/VID/VIDEO/DCIM 等设备前缀
    if re.match(r"^(IMG|VID|VIDEO|MOV|MP4|DCIM|P\d+)[_\-\s]?\d+", orig, re.IGNORECASE):
        return True

    # 2. Telegram 来源且无中文
    if re.search(r"(?i)telegram|(?<!\w)tg(?!\w)", orig) and not has_chinese(orig):
        cleaned = re.sub(r"(telegram|tg|@[\w]*|merged[-_]?\d*|[-_.\s\d\(\)\[\]【】]+)", "", orig, flags=re.IGNORECASE)
        cleaned = cleaned.strip("-_. ")
        if len(cleaned) < 3:
            return True

    # 3. 纯 hex/hash 或随机字母数字串
    if re.match(r"^[a-f0-9]{8,}$", orig, re.IGNORECASE):
        return True
    # 无分隔符的随机字母数字串（大小写混合，>=8位）
    name_no_ext = Path(orig).stem
    if re.match(r"^[a-zA-Z0-9]{8,}$", name_no_ext):
        has_upper = bool(re.search(r"[A-Z]", name_no_ext))
        has_lower = bool(re.search(r"[a-z]", name_no_ext))
        has_digit = bool(re.search(r"\d", name_no_ext))
        # 大小写混合 + 包含数字 = 大概率是随机串
        if has_upper and has_lower and has_digit:
            return True

    # 4. 去除括号后纯数字
    stripped = re.sub(r"[\(\)\[\]【】（）\s\-_.]+", "", orig)
    if stripped.isdigit():
        return True

    # 5. 纯数字/分隔符
    if re.match(r"^[\d\-_.\s\(\)\[\]【】（）]+$", orig):
        return True

    # 6. 中文日期格式
    if re.match(r"^\d{1,2}月\d{1,2}日(\s*[\(\（]\d+[\)\）])*\s*$", orig):
        return True

    # 7. 无中文 + 数字占比>70%
    if not has_chinese(orig):
        core = re.sub(r"[\(\)\[\]【】（）\-_.\s]+", "", orig)
        if core:
            digit_ratio = len(re.findall(r"\d", core)) / len(core)
            if digit_ratio > 0.7:
                return True

    # 8. 中文无意义标题
    if has_chinese(orig):
        cleaned = re.sub(r"[\(\)\[\]【】（）\s\-_.]+", "", orig)
        if cleaned == "视频":
            return True
        if re.match(r"^视\d+频$", cleaned):
            return True

    return False


class Scanner:
    """文件扫描器 - 简化版"""

    def __init__(self, output_dir: str = "data/output", db_store=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_store = db_store

    def scan(
        self,
        target_dir: str,
        output_file: str = None,
        append: bool = False,
        exclude_dirs: List[str] = None,
        force_reclassify: bool = False,
    ) -> str:
        """
        扫描目录或单个文件并生成待审表

        Args:
            target_dir: 目标目录或单个文件路径
            output_file: 输出文件路径
            append: 是否追加模式
            exclude_dirs: 排除的目录列表
            force_reclassify: 是否强制重新分类

        Returns:
            输出文件路径
        """
        target_path = Path(target_dir).resolve()
        if not target_path.exists():
            logger.error(f"路径不存在: {target_path}")
            return ""

        # 自动生成输出路径：目录扫描时创建子目录，单文件保持默认
        if output_file is None:
            if target_path.is_dir():
                dir_name = target_path.name
                output_dir = self.output_dir / dir_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = str(output_dir / "title_review.csv")
            else:
                output_file = str(self.output_dir / "title_review.csv")

        # 判断是单个文件还是目录
        if target_path.is_file():
            logger.info(f"处理单个文件: {target_path}")
            if _is_thumbnail(target_path):
                logger.info(f"跳过缩略图: {target_path}")
                return ""
            files = [target_path] if target_path.suffix.lower() in MEDIA_EXTENSIONS else []
            if not files:
                logger.warning(f"不支持的文件类型: {target_path}")
                return ""
        else:
            logger.info(f"开始递归扫描: {target_path}")
            files = self._scan_directory(target_path, exclude_dirs or [])
        
        logger.info(f"找到 {len(files)} 个媒体文件")

        # 处理文件
        rows = []
        for file_path in files:
            row = self._process_file(file_path, force_reclassify)
            if row:
                rows.append(row)

        # 保存结果
        if rows:
            self._save_csv(rows, output_file, append)
            logger.info(f"新增 {len(rows)} 条记录")

            # 同步到数据库
            if self.db_store:
                for row in rows:
                    self._sync_row_to_db(row)

        else:
            logger.info("没有新文件需要处理")

        return output_file

    def _sync_row_to_db(self, row: dict):
        """同步单行到数据库"""
        db = self.db_store
        original_path = row.get("original_path", "")
        if not original_path:
            return

        # 插入新记录（insert_media 内部会用 find_match 去重）
        data = {
            "original_title": row.get("original_title", ""),
            "original_path": original_path,
            "current_path": original_path,
            "needs_vision": row.get("needs_vision", "").lower() == "true",
            "final_name": row.get("final_name", ""),
            "review_status": row.get("review_status", "待确认"),
            "audio_recognized": row.get("audio_recognized", "").lower() == "true",
            "srt_path": row.get("srt_path", ""),
            "vision_description": row.get("vision_description", ""),
            "vision_keywords": row.get("vision_keywords", ""),
            "human_detected": row.get("human_detected", "").lower() == "true",
            "detection_method": row.get("detection_method", ""),
        }

        # 传递 file_size, duration, resolution（如果有的话）
        file_size = row.get("file_size")
        if file_size:
            try:
                data["file_size"] = int(file_size)
            except (ValueError, TypeError):
                pass

        duration = row.get("duration")
        if duration:
            try:
                data["duration"] = float(duration)
            except (ValueError, TypeError):
                pass

        resolution = row.get("resolution", "").strip()
        if resolution:
            data["resolution"] = resolution

        media_id = db.insert_media(data)

        # 导入标签
        keywords = row.get("vision_keywords", "")
        if keywords and media_id:
            db.add_tags_from_keywords(media_id, keywords, "scanner")

    def sync_db(self, target_dir: str, exclude_dirs: list = None):
        """扫描目录，将所有文件信息同步到数据库

        Args:
            target_dir: 目标目录
            exclude_dirs: 排除的目录列表
        """
        if not self.db_store:
            logger.error("未配置数据库连接，无法同步")
            return

        target_path = Path(target_dir).resolve()
        if not target_path.exists():
            logger.error(f"路径不存在: {target_path}")
            return

        if target_path.is_file():
            logger.error("sync_db 模式仅支持目录扫描")
            return

        logger.info(f"开始同步数据库: {target_path}")
        files = self._scan_directory(target_path, exclude_dirs or [])
        logger.info(f"找到 {len(files)} 个媒体文件，开始同步...")

        updated = 0
        inserted = 0
        skipped = 0
        deleted = 0

        for file_path in files:
            name = file_path.stem
            classified = is_already_classified(name)
            clean_title = strip_bracket_prefix(file_path.name)
            clean_name = strip_bracket_prefix(name)

            # 收集元数据
            file_size = None
            try:
                file_size = file_path.stat().st_size
            except OSError:
                pass

            duration = None
            resolution = ""
            if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                from ..utils.video import get_video_info
                vinfo = get_video_info(str(file_path))
                duration = vinfo.get("duration") or None
                resolution = vinfo.get("resolution", "")

                # ffprobe 和 cv2 都获取失败 → 损毁文件，删除
                if not duration and not resolution:
                    file_path.unlink(missing_ok=True)
                    logger.warning(f"[已删除] 损毁视频: {file_path.name}")
                    data = {
                        "original_title": clean_title,
                        "original_path": str(file_path),
                        "current_path": str(file_path),
                        "file_size": file_size,
                        "needs_vision": False,
                        "final_name": clean_name,
                        "review_status": "文件损毁已删除",
                    }
                    self.db_store.insert_media(data)
                    deleted += 1
                    continue

            elif file_path.suffix.lower() in IMAGE_EXTENSIONS:
                from ..utils.image import get_image_info
                iinfo = get_image_info(str(file_path))
                resolution = iinfo.get("resolution", "")

            data = {
                "original_title": clean_title,
                "original_path": str(file_path),
                "current_path": str(file_path),
                "file_size": file_size,
                "duration": duration,
                "resolution": resolution,
                "needs_vision": not classified,
                "final_name": clean_name,
                "review_status": "已规范化" if classified else "待确认",
            }

            existing = self.db_store.find_match(
                original_title=file_path.name,
                file_size=file_size,
                duration=duration,
                resolution=resolution,
                path=str(file_path),
            )

            # 未命中：计算内容指纹（L3 消歧），仍无 → 新建指纹+记录
            file_hash = None
            if not existing and file_size and duration:
                from ..utils.fingerprint import compute_partial_hash
                file_hash = compute_partial_hash(str(file_path))
                if file_hash:
                    existing = self.db_store.find_match(
                        original_title=file_path.name,
                        file_size=file_size,
                        duration=duration,
                        resolution=resolution,
                        file_hash=file_hash,
                    )

            if existing:
                changed = False

                # 如果之前标记为"文件已移走"，恢复状态
                if existing.get("review_status") == "文件已移走":
                    if existing.get("vision_description") or existing.get("vision_keywords"):
                        new_status = "已完成"
                    elif is_already_classified(existing.get("final_name", "")):
                        new_status = "已规范化"
                    else:
                        new_status = data["review_status"]
                    self.db_store.update_media(existing["id"], "review_status", new_status, "sync_db")
                    logger.info(f"[恢复] 文件已移走 → {new_status}: {clean_title}")
                    changed = True

                # 更新路径
                for field in ["original_path", "current_path"]:
                    new_val = data.get(field)
                    old_val = existing.get(field)
                    if new_val and new_val != old_val:
                        self.db_store.update_media(existing["id"], field, new_val, "sync_db")
                        changed = True

                # 补全缺失元数据
                for field in ["file_size", "duration", "resolution"]:
                    new_val = data.get(field)
                    old_val = existing.get(field)
                    if new_val and not old_val:
                        self.db_store.update_media(existing["id"], field, new_val, "sync_db")
                        changed = True

                # 关联指纹（L2/L3 命中时补上 fingerprint_id）
                if file_hash and not existing.get("fingerprint_id"):
                    fp_id = self.db_store.get_fingerprint_id(file_size, duration, file_hash)
                    self.db_store.link_fingerprint(existing["id"], fp_id)

                if changed:
                    updated += 1
                else:
                    skipped += 1
            else:
                # 新建记录并关联指纹
                fp_id = None
                if file_size and duration:
                    fp_id = self.db_store.get_fingerprint_id(file_size, duration, file_hash)
                if fp_id:
                    data["fingerprint_id"] = fp_id
                self.db_store.insert_media(data)
                inserted += 1

        logger.info(f"[完成] 数据库同步: 新增 {inserted}, 更新 {updated}, 无变化 {skipped}, 删除损毁 {deleted}")

        # 标记磁盘上已不存在的记录为"文件已移走"
        moved = 0
        target_str = str(target_path)
        db_rows = self.db_store.conn.execute(
            "SELECT id, current_path, original_path, review_status FROM media_files WHERE current_path LIKE ? OR original_path LIKE ?",
            (f"{target_str}%", f"{target_str}%")
        ).fetchall()
        for row in db_rows:
            if row["review_status"] == "文件已移走":
                continue
            cur = row["current_path"]
            orig = row["original_path"]
            cur_missing = cur and not Path(cur).exists()
            orig_missing = orig and not Path(orig).exists()
            if (cur_missing and orig_missing) or (cur_missing and not orig):
                self.db_store.update_media(row["id"], "review_status", "文件已移走", "sync_db")
                moved += 1
        if moved:
            logger.info(f"[同步] 标记 {moved} 个文件已移走")

        # 查询该目录下缺少视觉描述的记录，生成待处理 CSV
        self._generate_vision_csv(target_path)

    def _generate_vision_csv(self, target_path: Path):
        """查询 DB 中缺少视觉描述的记录，生成 CSV 供视觉识别使用

        只包含：
        1. 路径前缀匹配目标目录
        2. 缺少视觉描述
        3. review_status 不是 "文件已移走"
        4. current_path 实际存在

        生成时会合并已有 CSV 中的视觉数据，避免覆盖之前的处理结果。
        """
        output_dir = self.output_dir / target_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / "title_review.csv")

        # 加载已有 CSV 中的视觉数据（避免覆盖之前的处理结果）
        existing_vision = {}
        if Path(output_file).exists():
            try:
                with open(output_file, "r", encoding="utf-8-sig") as f:
                    for r in csv.DictReader(f):
                        path = r.get("original_path", "")
                        desc = r.get("vision_description", "").strip()
                        kw = r.get("vision_keywords", "").strip()
                        if path and (desc or kw):
                            existing_vision[path] = r
            except Exception:
                pass

        dir_prefix = str(target_path) + "%"
        rows = self.db_store.conn.execute(
            """SELECT * FROM media_files
               WHERE original_path LIKE ?
               AND (vision_description IS NULL OR vision_description = '')
               AND (vision_keywords IS NULL OR vision_keywords = '')
               AND review_status != '文件已移走'""",
            (dir_prefix,)
        ).fetchall()

        # 过滤掉实际不存在的文件
        valid_rows = []
        for row in rows:
            r = dict(row)
            current_path = r.get("current_path", "")
            if current_path and Path(current_path).exists():
                valid_rows.append(r)

        if not valid_rows:
            logger.info("[完成] 所有文件已完成视觉识别，无需生成 CSV")
            return

        csv_rows = []
        for r in valid_rows:
            path = r.get("original_path", "")
            # 如果已有 CSV 中包含该路径的视觉数据，直接复用
            if path in existing_vision:
                csv_rows.append(existing_vision[path])
                continue
            csv_rows.append({
                "original_title": r.get("original_title", ""),
                "original_path": path,
                "needs_vision": "true",
                "final_name": r.get("final_name", ""),
                "review_status": "待确认",
                "audio_recognized": "false",
                "srt_path": "",
                "vision_description": "",
                "vision_keywords": "",
                "vision_failed": "false",
                "human_detected": "",
                "detection_confidence": "",
                "detection_timestamp": "",
                "detection_method": "",
                "clip_clothing": "",
                "clip_action": "",
                "clip_hairstyle": "",
                "clip_tags": "",
                "clip_tags_json": "",
                "clip_confidence": "",
                "clip_detail": "",
                "vision_source": "",
                "file_size": r.get("file_size", ""),
                "duration": r.get("duration", ""),
                "resolution": r.get("resolution", ""),
            })

        self._save_csv(csv_rows, output_file, append=False)
        logger.info(f"[待视觉识别] 发现 {len(csv_rows)} 条记录缺少视觉描述，已生成 CSV: {output_file}")

    def _scan_directory(self, directory: Path, exclude_dirs: List[str]) -> List[Path]:
        """递归扫描目录"""
        files = []
        exclude_set = set(exclude_dirs)

        for item in directory.rglob("*"):
            # 检查是否在排除目录中
            rel_path = item.relative_to(directory)
            if any(excluded in str(rel_path) for excluded in exclude_set):
                continue

            if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
                if _is_thumbnail(item):
                    continue
                files.append(item)

        return sorted(files)

    def _process_file(self, file_path: Path, force_reclassify: bool = False) -> Optional[Dict]:
        """处理单个文件 - 判断是否需要视觉识别，并填入原标题作为final_name"""
        name = file_path.stem
        original_title = file_path.name

        # 检查是否已分类
        if is_already_classified(name) and not force_reclassify:
            return None

        # force_reclassify 时：剥离 [keywords]_ 前缀，还原干净文件名重新判断
        if force_reclassify:
            clean_name = strip_bracket_prefix(name)
            clean_title = strip_bracket_prefix(original_title)
        else:
            clean_name = name
            clean_title = original_title

        # 判断是否需要视觉识别（使用干净文件名）
        needs_vision = is_needs_vision(clean_title)

        # 默认填入干净原标题（去除扩展名）作为final_name
        final_name = clean_name

        # 获取文件大小（快速，毫秒级）
        try:
            file_size = file_path.stat().st_size
        except OSError:
            file_size = None

        # 视频文件：获取时长和分辨率
        duration = None
        resolution = ""
        if file_path.suffix.lower() in VIDEO_EXTENSIONS:
            from ..utils.video import get_video_info
            vinfo = get_video_info(str(file_path))
            duration = vinfo.get("duration") or None
            resolution = vinfo.get("resolution", "")

        return {
            "original_title": clean_title,
            "original_path": str(file_path),
            "needs_vision": str(needs_vision).lower(),
            "final_name": final_name,  # 默认填入原标题
            "review_status": "待确认",
            "audio_recognized": "false",
            "srt_path": "",
            "vision_description": "",
            "vision_keywords": "",
            "human_detected": "",
            "detection_confidence": "",
            "detection_timestamp": "",
            "detection_method": "",
            "clip_clothing": "",
            "clip_action": "",
            "clip_hairstyle": "",
            "clip_tags": "",
            "clip_tags_json": "",
            "clip_confidence": "",
            "clip_detail": "",
            "vision_source": "",
            "vision_failed": "false",
            "file_size": file_size,
            "duration": duration,
            "resolution": resolution,
        }

    def _save_csv(self, rows: List[Dict], output_file: str, append: bool = False) -> None:
        """保存CSV文件（原子化写入）"""
        from ..utils.atomic_csv import atomic_write_csv, atomic_append_csv

        fieldnames = [
            "original_title", "original_path",
            "needs_vision", "final_name", "review_status",
            "audio_recognized", "srt_path",
            "vision_description", "vision_keywords",
            "vision_failed",
            "human_detected", "detection_confidence", "detection_timestamp", "detection_method",
            "clip_clothing", "clip_action", "clip_hairstyle",
            "clip_tags", "clip_tags_json", "clip_confidence", "clip_detail", "vision_source",
            "file_size", "duration", "resolution",
        ]

        if append:
            atomic_append_csv(output_file, rows, fieldnames)
        else:
            atomic_write_csv(output_file, rows, fieldnames)

        logger.info(f"结果已保存至: {output_file}")
