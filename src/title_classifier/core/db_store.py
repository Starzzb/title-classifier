"""SQLite 数据库访问层"""

import sqlite3
import shutil
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


def _get_schema_path() -> Path:
    """获取 db_schema.sql 路径（支持打包模式）"""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller 打包模式
        return Path(sys._MEIPASS) / "title_classifier" / "core" / "db_schema.sql"
    else:
        # 开发模式
        return Path(__file__).parent / "db_schema.sql"


SCHEMA_PATH = _get_schema_path()


def _get_db_path():
    """获取默认数据库路径"""
    # 项目根目录/data/media.db
    root = Path(__file__).parent.parent.parent.parent
    return root / "data" / "media.db"


class MediaDB:
    """媒体文件数据库访问层"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(_get_db_path())
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def init_schema(self):
        """初始化表结构"""
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._migrate_fingerprints()
        self.conn.commit()
        logger.info(f"数据库初始化完成: {self.db_path}")

    def _migrate_fingerprints(self):
        """迁移旧版 video_fingerprints 表：去掉 UNIQUE(file_size, duration) 约束。

        注意：不能用 ALTER TABLE video_fingerprints RENAME TO xxx 方式，
        那会让 SQLite 自动改写 media_files.fingerprint_id 的外键引用指向旧表名。
        正确做法：新建表 → 复制数据 → 删旧表 → 新表改名回原名。
        """
        try:
            sql = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='video_fingerprints'"
            ).fetchone()
            if not sql:
                return
            if "UNIQUE(file_size, duration)" in sql[0]:
                logger.info("检测到旧版 video_fingerprints UNIQUE 约束，重建表...")
                self.conn.execute("PRAGMA foreign_keys=OFF")
                self.conn.executescript("""
                    CREATE TABLE video_fingerprints_new (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_size       INTEGER NOT NULL,
                        duration        REAL NOT NULL,
                        file_hash       TEXT,
                        first_seen      TEXT DEFAULT (datetime('now', 'localtime')),
                        last_seen       TEXT DEFAULT (datetime('now', 'localtime'))
                    );
                    INSERT INTO video_fingerprints_new (id, file_size, duration, file_hash, first_seen, last_seen)
                        SELECT id, file_size, duration, file_hash, first_seen, last_seen FROM video_fingerprints;
                    DROP TABLE video_fingerprints;
                    ALTER TABLE video_fingerprints_new RENAME TO video_fingerprints;
                    CREATE INDEX IF NOT EXISTS idx_fingerprint_size_dur ON video_fingerprints(file_size, duration);
                    CREATE INDEX IF NOT EXISTS idx_fingerprint_hash ON video_fingerprints(file_hash);
                """)
                self.conn.execute("PRAGMA foreign_keys=ON")
                logger.info("video_fingerprints 表已重建（去掉 UNIQUE 约束）")

            # 修复历史迁移遗留：media_files 外键误指向 video_fingerprints_old
            self._repair_fingerprint_fk()
        except Exception as e:
            logger.warning(f"video_fingerprints 迁移失败: {e}")

    def _repair_fingerprint_fk(self):
        """修复 media_files.fingerprint_id 外键误引用 video_fingerprints_old 的问题。

        旧版迁移用 ALTER TABLE RENAME 导致外键被改写为指向已删除的旧表名，
        这里检测并重建 media_files 表修正外键。
        """
        try:
            row = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='media_files'"
            ).fetchone()
            if not row:
                return
            if "video_fingerprints_old" not in row[0]:
                return
            logger.info("检测到 media_files 外键误引用 video_fingerprints_old，重建表...")
            self.conn.execute("PRAGMA foreign_keys=OFF")
            self.conn.executescript("""
                CREATE TABLE media_files_new (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_title  TEXT NOT NULL,
                    original_path   TEXT NOT NULL,
                    current_path    TEXT,
                    file_size       INTEGER,
                    duration        REAL,
                    resolution      TEXT,
                    file_hash       TEXT,
                    final_name      TEXT,
                    vision_description TEXT,
                    vision_keywords TEXT,
                    human_detected  INTEGER DEFAULT 0,
                    detection_method TEXT,
                    needs_vision    INTEGER DEFAULT 1,
                    audio_recognized INTEGER DEFAULT 0,
                    review_status   TEXT DEFAULT '待确认',
                    srt_path        TEXT,
                    fingerprint_id  INTEGER REFERENCES video_fingerprints(id),
                    created_at      TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at      TEXT DEFAULT (datetime('now', 'localtime')),
                    faststart INTEGER DEFAULT 0,
                    video_codec TEXT DEFAULT ''
                );
                INSERT INTO media_files_new (
                    id, original_title, original_path, current_path, file_size, duration, resolution,
                    file_hash, final_name, vision_description, vision_keywords, human_detected,
                    detection_method, needs_vision, audio_recognized, review_status, srt_path,
                    fingerprint_id, created_at, updated_at, faststart, video_codec
                )
                SELECT id, original_title, original_path, current_path, file_size, duration, resolution,
                    file_hash, final_name, vision_description, vision_keywords, human_detected,
                    detection_method, needs_vision, audio_recognized, review_status, srt_path,
                    fingerprint_id, created_at, updated_at, faststart, video_codec
                FROM media_files;
                DROP TABLE media_files;
                ALTER TABLE media_files_new RENAME TO media_files;
            """)
            self.conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_media_original_path ON media_files(original_path);
                CREATE INDEX IF NOT EXISTS idx_media_current_path ON media_files(current_path);
                CREATE INDEX IF NOT EXISTS idx_media_final_name ON media_files(final_name);
                CREATE INDEX IF NOT EXISTS idx_media_file_hash ON media_files(file_hash);
                CREATE INDEX IF NOT EXISTS idx_media_fingerprint ON media_files(fingerprint_id);
            """)
            self.conn.execute("PRAGMA foreign_keys=ON")
            logger.info("media_files 外键已修复")
        except Exception as e:
            logger.warning(f"media_files 外键修复失败: {e}")

    def close(self):
        """关闭连接"""
        self.conn.close()

    # ===== 基础 CRUD =====

    def insert_media(self, data: dict) -> int:
        """插入新记录，返回 media_id。如果匹配到已有记录则更新路径并返回旧 id"""
        existing = self.find_match(
            original_title=data.get("original_title", ""),
            file_size=data.get("file_size"),
            duration=data.get("duration"),
            resolution=data.get("resolution"),
            path=data.get("original_path") or data.get("current_path"),
        )

        # 未命中：尝试内容指纹消歧
        file_hash = data.get("file_hash")
        if not existing and not file_hash and data.get("file_size") and data.get("duration"):
            from ..utils.fingerprint import compute_partial_hash
            src = data.get("original_path") or data.get("current_path")
            if src:
                file_hash = compute_partial_hash(src)
                if file_hash:
                    existing = self.find_match(
                        original_title=data.get("original_title", ""),
                        file_size=data.get("file_size"),
                        duration=data.get("duration"),
                        resolution=data.get("resolution"),
                        file_hash=file_hash,
                    )

        if existing:
            # 匹配到已有记录，更新路径
            new_path = data.get("original_path", "")
            if new_path and new_path != existing.get("original_path"):
                self.update_media(existing["id"], "original_path", new_path, "match_update")
            current_path = data.get("current_path", new_path)
            if current_path and current_path != existing.get("current_path"):
                self.update_media(existing["id"], "current_path", current_path, "match_update")
            # 关联指纹
            if file_hash and not existing.get("fingerprint_id") and data.get("duration"):
                fp_id = self.get_fingerprint_id(data["file_size"], data["duration"], file_hash)
                self.link_fingerprint(existing["id"], fp_id)
            return existing["id"]

        # 新建记录：关联指纹
        fp_id = data.get("fingerprint_id")
        if not fp_id and file_hash and data.get("duration"):
            fp_id = self.get_fingerprint_id(data["file_size"], data["duration"], file_hash)

        cursor = self.conn.execute("""
            INSERT INTO media_files
                (original_title, original_path, current_path, file_size, duration,
                 resolution, final_name, vision_description, vision_keywords,
                 human_detected, detection_method, needs_vision, audio_recognized,
                 review_status, srt_path, fingerprint_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("original_title", ""),
            data.get("original_path", ""),
            data.get("current_path", data.get("original_path", "")),
            data.get("file_size"),
            data.get("duration"),
            data.get("resolution"),
            data.get("final_name"),
            data.get("vision_description"),
            data.get("vision_keywords"),
            int(data.get("human_detected", False)),
            data.get("detection_method"),
            int(data.get("needs_vision", True)),
            int(data.get("audio_recognized", False)),
            data.get("review_status", "待确认"),
            data.get("srt_path"),
            fp_id,
        ))
        self.conn.commit()
        media_id = cursor.lastrowid
        logger.debug(f"插入记录: id={media_id}, path={data.get('original_path')}")
        return media_id

    def update_media(self, media_id: int, field: str, value: any, source: str = ""):
        """更新字段并记录改动"""
        current = self.conn.execute(
            f"SELECT {field} FROM media_files WHERE id=?", (media_id,)
        ).fetchone()

        if current and str(current[0]) != str(value):
            old_value = str(current[0]) if current[0] is not None else ""
            self.log_change(media_id, field, old_value, str(value), source)

        self.conn.execute(
            f"UPDATE media_files SET {field}=?, updated_at=datetime('now','localtime') WHERE id=?",
            (value, media_id)
        )
        self.conn.commit()

    def get_media(self, media_id: int) -> Optional[dict]:
        """获取单条记录"""
        row = self.conn.execute("SELECT * FROM media_files WHERE id=?", (media_id,)).fetchone()
        return dict(row) if row else None

    def find_by_path(self, path: str) -> Optional[dict]:
        """按 original_path 查找"""
        row = self.conn.execute(
            "SELECT * FROM media_files WHERE original_path=?", (path,)
        ).fetchone()
        return dict(row) if row else None

    def find_by_current_path(self, path: str) -> Optional[dict]:
        """按 current_path 查找"""
        row = self.conn.execute(
            "SELECT * FROM media_files WHERE current_path=?", (path,)
        ).fetchone()
        return dict(row) if row else None

    def find_match(self, original_title: str, file_size: int = None, duration: float = None,
                   resolution: str = None, path: str = None, file_hash: str = None) -> Optional[dict]:
        """
        四级匹配逻辑：

        L1 路径精确匹配：path 精确匹配 current_path 或 original_path
           → 同一文件已入库（覆盖未变文件，零成本）。

        L2 粗指纹匹配：file_size + duration 唯一命中 video_fingerprints
           → 文件移动/重命名后，内容未变，靠粗指纹追踪（无需哈希）。

        L3 哈希精确匹配：file_hash 命中指纹 → 内容真实相同，可安全追踪
           （仅 L2 歧义或需要消歧时使用）。

        L4 标题兜底：剥离 [关键词]_ 前缀后的标题 + size + duration + resolution，
           且 DB 旧路径已不存在 → 图片 / 无指纹记录 / 哈希失败时兜底。
        """
        # L1 路径精确匹配（最高优先级）
        if path:
            row = self.conn.execute(
                "SELECT * FROM media_files WHERE current_path=? OR original_path=?",
                (path, path)
            ).fetchone()
            if row:
                return dict(row)

        # L2 粗指纹匹配：size + duration 唯一候选 → 内容未变即可追踪
        if file_size and duration:
            fps = self.conn.execute(
                """SELECT * FROM video_fingerprints
                   WHERE ABS(file_size - ?) / MAX(file_size, 1) <= 0.001
                     AND ABS(duration - ?) <= 0.5""",
                (file_size, duration)
            ).fetchall()
            if len(fps) == 1:
                media = self.conn.execute(
                    "SELECT * FROM media_files WHERE fingerprint_id=?",
                    (fps[0]["id"],)
                ).fetchone()
                if media:
                    self.update_fingerprint_last_seen(fps[0]["id"])
                    return dict(media)

        # L3 哈希精确匹配：内容真实相同，安全追踪（无需"旧路径不存在"）
        if file_hash:
            fp = self.find_fingerprint_by_hash(file_hash)
            if fp:
                media = self.conn.execute(
                    "SELECT * FROM media_files WHERE fingerprint_id=?",
                    (fp["id"],)
                ).fetchone()
                if media:
                    self.update_fingerprint_last_seen(fp["id"])
                    return dict(media)

        if not original_title:
            return None

        original_title = original_title.strip()
        if not original_title:
            return None

        # L4 标题兜底：剥离 [关键词]_ 前缀后的标题 + size + duration + resolution
        #    且旧路径已不存在 → 判定为文件移动/重命名，追踪到已有记录
        from .scanner import strip_bracket_prefix

        stripped = strip_bracket_prefix(original_title)
        if not stripped:
            return None

        # 候选：精确标题，或带 [xxx]_ 前缀的标题（在 Python 中剥离后比较）
        candidates = self.conn.execute(
            "SELECT * FROM media_files WHERE original_title = ? OR original_title LIKE ?",
            (stripped, "[%]%")
        ).fetchall()

        for c in candidates:
            if strip_bracket_prefix(c["original_title"]) != stripped:
                continue

            # 精确验证 size / duration / resolution（DB 为 NULL 时视为未知，不阻断）
            if file_size is not None and file_size > 0 and c["file_size"]:
                if abs(c["file_size"] - file_size) / max(c["file_size"], 1) > 0.001:
                    continue
            if duration is not None and duration > 0 and c["duration"]:
                if abs(c["duration"] - duration) > 0.5:
                    continue
            if resolution and c["resolution"] and c["resolution"] != resolution:
                continue

            old_cur = c["current_path"]
            old_orig = c["original_path"]
            old_cur_missing = old_cur and not Path(old_cur).exists()
            old_orig_missing = old_orig and not Path(old_orig).exists()
            if (old_cur_missing and old_orig_missing) or (old_cur_missing and not old_orig):
                return dict(c)

        return None

    def list_all(self, limit: int = 100, offset: int = 0) -> List[dict]:
        """列出所有记录"""
        rows = self.conn.execute(
            "SELECT * FROM media_files ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """统计记录数"""
        row = self.conn.execute("SELECT COUNT(*) FROM media_files").fetchone()
        return row[0]

    # ===== 去重 =====

    def find_fingerprint(self, file_size: int, duration: float) -> List[dict]:
        """按 file_size + duration 查找指纹（可能多个候选）"""
        rows = self.conn.execute(
            "SELECT * FROM video_fingerprints WHERE file_size=? AND duration=?",
            (file_size, duration)
        ).fetchall()
        return [dict(r) for r in rows]

    def find_fingerprint_by_hash(self, file_hash: str) -> Optional[dict]:
        """按 file_hash 精确查找指纹"""
        row = self.conn.execute(
            "SELECT * FROM video_fingerprints WHERE file_hash=?",
            (file_hash,)
        ).fetchone()
        return dict(row) if row else None

    def create_fingerprint(self, file_size: int, duration: float, file_hash: str = None) -> int:
        """创建指纹"""
        cursor = self.conn.execute(
            "INSERT INTO video_fingerprints (file_size, duration, file_hash) VALUES (?, ?, ?)",
            (file_size, duration, file_hash)
        )
        self.conn.commit()
        return cursor.lastrowid

    def link_fingerprint(self, media_id: int, fp_id: int):
        """将 media_files 记录关联到指纹"""
        if not fp_id:
            return
        self.conn.execute(
            "UPDATE media_files SET fingerprint_id=? WHERE id=?",
            (fp_id, media_id)
        )
        self.conn.commit()

    def update_fingerprint_last_seen(self, fp_id: int):
        """更新指纹的最后访问时间"""
        self.conn.execute(
            "UPDATE video_fingerprints SET last_seen=datetime('now','localtime') WHERE id=?",
            (fp_id,)
        )
        self.conn.commit()

    def get_fingerprint_id(self, file_size: int, duration: float, file_hash: str = None) -> int:
        """获取或创建指纹 ID。

        优先按 file_hash 精确匹配；无 file_hash 时按 size+duration 唯一候选匹配；
        否则创建新指纹。
        """
        if file_hash:
            fp = self.find_fingerprint_by_hash(file_hash)
            if fp:
                self.update_fingerprint_last_seen(fp["id"])
                return fp["id"]

        candidates = self.find_fingerprint(file_size, duration)
        if not file_hash and len(candidates) == 1:
            self.update_fingerprint_last_seen(candidates[0]["id"])
            return candidates[0]["id"]

        return self.create_fingerprint(file_size, duration, file_hash)

    # ===== 标签 =====

    def add_tag(self, media_id: int, tag_name: str, source: str = "vision", confidence: float = None):
        """添加标签"""
        tag = self.conn.execute("SELECT id FROM tags WHERE name=?", (tag_name,)).fetchone()
        if tag:
            tag_id = tag["id"]
        else:
            cursor = self.conn.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
            tag_id = cursor.lastrowid

        try:
            self.conn.execute(
                "INSERT INTO media_tags (media_id, tag_id, confidence, source) VALUES (?, ?, ?, ?)",
                (media_id, tag_id, confidence, source)
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass

    def add_tags_from_keywords(self, media_id: int, keywords: str, source: str = "vision"):
        """从逗号分隔的关键词批量添加标签"""
        if not keywords:
            return
        for kw in keywords.split(","):
            kw = kw.strip()
            if kw:
                self.add_tag(media_id, kw, source)

    def get_tags(self, media_id: int) -> List[str]:
        """获取媒体的所有标签"""
        rows = self.conn.execute("""
            SELECT t.name FROM tags t
            JOIN media_tags mt ON t.id = mt.tag_id
            WHERE mt.media_id = ?
            ORDER BY t.name
        """, (media_id,)).fetchall()
        return [r["name"] for r in rows]

    def search_by_tag(self, tag_name: str) -> List[dict]:
        """按标签搜索"""
        rows = self.conn.execute("""
            SELECT m.* FROM media_files m
            JOIN media_tags mt ON m.id = mt.media_id
            JOIN tags t ON mt.tag_id = t.id
            WHERE t.name LIKE ?
            ORDER BY m.updated_at DESC
        """, (f"%{tag_name}%",)).fetchall()
        return [dict(r) for r in rows]

    def get_all_tags(self) -> List[dict]:
        """获取所有标签"""
        rows = self.conn.execute("""
            SELECT t.name, t.category, COUNT(mt.media_id) as count
            FROM tags t
            LEFT JOIN media_tags mt ON t.id = mt.tag_id
            GROUP BY t.id
            ORDER BY count DESC
        """).fetchall()
        return [dict(r) for r in rows]

    # ===== 改动记录 =====

    def log_change(self, media_id: int, field_name: str, old_value: str, new_value: str, source: str = ""):
        """记录改动"""
        self.conn.execute("""
            INSERT INTO change_log (media_id, field_name, old_value, new_value, change_source)
            VALUES (?, ?, ?, ?, ?)
        """, (media_id, field_name, old_value, new_value, source))
        self.conn.commit()

    def get_changes(self, media_id: int) -> List[dict]:
        """获取媒体的所有改动历史"""
        rows = self.conn.execute("""
            SELECT * FROM change_log
            WHERE media_id = ?
            ORDER BY changed_at DESC
        """, (media_id,)).fetchall()
        return [dict(r) for r in rows]

    # ===== VLM 帧 =====

    def save_vlm_frame(self, media_id: int, frame_index: int, frame_path: str, timestamp: float = None):
        """保存 VLM 帧记录"""
        try:
            self.conn.execute("""
                INSERT INTO vlm_frames (media_id, frame_index, frame_path, timestamp)
                VALUES (?, ?, ?, ?)
            """, (media_id, frame_index, frame_path, timestamp))
            self.conn.commit()
        except sqlite3.IntegrityError:
            self.conn.execute("""
                UPDATE vlm_frames SET frame_path=?, timestamp=?
                WHERE media_id=? AND frame_index=?
            """, (frame_path, timestamp, media_id, frame_index))
            self.conn.commit()

    def get_vlm_frames(self, media_id: int) -> List[dict]:
        """获取媒体的所有 VLM 帧"""
        rows = self.conn.execute("""
            SELECT * FROM vlm_frames
            WHERE media_id = ?
            ORDER BY frame_index
        """, (media_id,)).fetchall()
        return [dict(r) for r in rows]

    # ===== 搜索 =====

    def search(self, query: str = None, tags: List[str] = None, source: str = None) -> List[dict]:
        """综合搜索"""
        sql = "SELECT DISTINCT m.* FROM media_files m"
        params = []
        joins = []
        wheres = []

        if tags:
            joins.append("JOIN media_tags mt ON m.id = mt.media_id")
            joins.append("JOIN tags t ON mt.tag_id = t.id")
            placeholders = ",".join(["?" for _ in tags])
            wheres.append(f"t.name IN ({placeholders})")
            params.extend(tags)

        if query:
            wheres.append("(m.original_title LIKE ? OR m.final_name LIKE ? OR m.vision_description LIKE ? OR m.vision_keywords LIKE ?)")
            params.extend([f"%{query}%"] * 4)

        if source:
            wheres.append("m.original_path LIKE ?")
            params.append(f"%{source}%")

        if joins:
            sql += " " + " ".join(joins)
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)

        sql += " ORDER BY m.updated_at DESC"

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ===== 统计 =====

    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = {}
        stats["total_media"] = self.count()
        stats["total_tags"] = self.conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        stats["total_changes"] = self.conn.execute("SELECT COUNT(*) FROM change_log").fetchone()[0]
        stats["total_frames"] = self.conn.execute("SELECT COUNT(*) FROM vlm_frames").fetchone()[0]

        # 按类型统计
        video_ext = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts")
        image_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff")

        stats["videos"] = self.conn.execute(
            "SELECT COUNT(*) FROM media_files WHERE original_path LIKE ?", ("%.mp4%",)
        ).fetchone()[0]
        stats["images"] = self.conn.execute(
            "SELECT COUNT(*) FROM media_files WHERE original_path LIKE ?", ("%.jpg%",)
        ).fetchone()[0]

        # 标签频率 Top 10
        stats["top_tags"] = self.get_all_tags()[:10]

        return stats

    # ===== CSV 导入 =====

    def import_csv(self, csv_path: str) -> dict:
        """从 CSV 导入数据"""
        import csv as csv_module

        csv_path = Path(csv_path)
        if not csv_path.exists():
            return {"error": f"CSV 不存在: {csv_path}"}

        stats = {"total": 0, "imported": 0, "updated": 0, "skipped": 0, "tags_added": 0}

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                stats["total"] += 1
                original_path = row.get("original_path", "")

                if not original_path:
                    stats["skipped"] += 1
                    continue

                # 尝试从CSV获取元数据（如果有列）
                file_size = None
                fs_str = row.get("file_size", "").strip()
                if fs_str:
                    try:
                        file_size = int(fs_str)
                    except ValueError:
                        pass

                duration = None
                dur_str = row.get("duration", "").strip()
                if dur_str:
                    try:
                        duration = float(dur_str)
                    except ValueError:
                        pass

                resolution = row.get("resolution", "").strip()

                existing = self.find_match(
                    original_title=row.get("original_title", ""),
                    file_size=file_size,
                    duration=duration,
                    path=original_path,
                )

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
                    "file_size": file_size,
                    "duration": duration,
                    "resolution": resolution,
                }

                if existing:
                    for field in data:
                        if data[field] and not existing.get(field):
                            self.update_media(existing["id"], field, data[field], "import_csv")
                    stats["updated"] += 1
                    media_id = existing["id"]
                else:
                    media_id = self.insert_media(data)
                    stats["imported"] += 1

                # 导入标签
                keywords = row.get("vision_keywords", "")
                if keywords:
                    self.add_tags_from_keywords(media_id, keywords, "import_csv")
                    stats["tags_added"] += len([k for k in keywords.split(",") if k.strip()])

        return stats

    def import_all_csvs(self, output_dir: str = "data/output") -> dict:
        """从所有 CSV 文件导入数据"""
        output_path = Path(output_dir)
        total_stats = {"total": 0, "imported": 0, "updated": 0, "skipped": 0, "tags_added": 0}

        for csv_file in output_path.rglob("title_review.csv"):
            stats = self.import_csv(str(csv_file))
            for k, v in stats.items():
                if k != "error" and k in total_stats:
                    total_stats[k] += v
            logger.info(f"导入 {csv_file}: {stats}")

        return total_stats
