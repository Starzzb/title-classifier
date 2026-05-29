"""SQLite 数据库访问层"""

import sqlite3
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "db_schema.sql"


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
        self.conn.commit()
        logger.info(f"数据库初始化完成: {self.db_path}")

    def close(self):
        """关闭连接"""
        self.conn.close()

    # ===== 基础 CRUD =====

    def insert_media(self, data: dict) -> int:
        """插入新记录，返回 media_id"""
        existing = self.find_by_path(data.get("original_path", ""))
        if existing:
            return existing["id"]

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
            data.get("fingerprint_id"),
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

    def find_fingerprint(self, file_size: int, duration: float) -> Optional[dict]:
        """查找指纹"""
        row = self.conn.execute(
            "SELECT * FROM video_fingerprints WHERE file_size=? AND duration=?",
            (file_size, duration)
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

    def update_fingerprint_last_seen(self, fp_id: int):
        """更新指纹的最后访问时间"""
        self.conn.execute(
            "UPDATE video_fingerprints SET last_seen=datetime('now','localtime') WHERE id=?",
            (fp_id,)
        )
        self.conn.commit()

    def check_duplicate(self, file_size: int, duration: float, file_hash: str = None) -> dict:
        """检查重复"""
        fp = self.find_fingerprint(file_size, duration)
        if not fp:
            fp_id = self.create_fingerprint(file_size, duration, file_hash)
            return {"is_duplicate": False, "fingerprint_id": fp_id}

        if file_hash and fp["file_hash"]:
            if fp["file_hash"] == file_hash:
                existing = self.conn.execute(
                    "SELECT id FROM media_files WHERE fingerprint_id=?", (fp["id"],)
                ).fetchone()
                self.update_fingerprint_last_seen(fp["id"])
                return {
                    "is_duplicate": True,
                    "fingerprint_id": fp["id"],
                    "existing_media_id": existing["id"] if existing else None
                }

        return {"is_duplicate": False, "fingerprint_id": fp["id"]}

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
            wheres.append("(m.original_title LIKE ? OR m.final_name LIKE ? OR m.vision_description LIKE ?)")
            params.extend([f"%{query}%"] * 3)

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

                existing = self.find_by_path(original_path)

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
