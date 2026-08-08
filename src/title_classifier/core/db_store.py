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
        这里检测并重建 media_files 表修正外键。列定义从 PRAGMA table_info
        动态读取，避免硬编码列清单导致后续新增列在重建时被丢弃。
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

            cols = self.conn.execute("PRAGMA table_info(media_files)").fetchall()
            col_defs = []
            for c in cols:
                cid, name, ctype, notnull, dflt, pk = c
                parts = [f'"{name}" {ctype}']
                if pk:
                    parts.append("PRIMARY KEY AUTOINCREMENT" if cid == 0 else "PRIMARY KEY")
                if notnull:
                    parts.append("NOT NULL")
                if dflt is not None:
                    # PRAGMA dflt_value 是原样文本，但表达式默认值已被去掉外层括号，
                    # SQLite 要求非字面量默认值必须加括号，统一包一层总是合法。
                    parts.append(f"DEFAULT ({dflt})")
                # 外键：把指向旧表名的引用重写为 video_fingerprints(id)
                if name == "fingerprint_id":
                    parts.append("REFERENCES video_fingerprints(id)")
                col_defs.append(" ".join(parts))

            self.conn.execute("PRAGMA foreign_keys=OFF")
            self.conn.execute(
                f'CREATE TABLE media_files_new ({", ".join(col_defs)})'
            )
            col_names = ", ".join(f'"{c[1]}"' for c in cols)
            self.conn.execute(
                f"INSERT INTO media_files_new ({col_names}) SELECT {col_names} FROM media_files"
            )
            self.conn.execute("DROP TABLE media_files")
            self.conn.execute("ALTER TABLE media_files_new RENAME TO media_files")
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
            file_hash=data.get("file_hash"),  # 调用方已算好则用于严格确认
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
        匹配确认流程：从宽松到严密，逐级确认两个视频确实是同一个。

        设计原则：
          - 收集候选要宽松（宁多勿漏）：路径 / 粗指纹 / 标题 / 哈希 都会成为候选。
          - 确认要严密（宁缺勿滥）：必须有足够强的证据才判定"是同一视频"，
            任何内容矛盾（哈希不同 / size 差异大）立即排除该候选。
          - 排除要宽松：一旦发现 hash 或 size 矛盾，立刻否定，不做多余推断。

        证据强度（由强到弱，取分最高者，0 分即排除）：
          E1 内容哈希一致：file_hash == 候选指纹 hash → 100% 同一（4 分）
          E2 路径命中 + size 一致（记录无 hash 可证伪）→ 高度同一（3 分）
          E3 size+duration 粗指纹命中，且候选指纹无 hash 可证伪 → 弱证据兜底（2 分）
          E4 剥离前缀标题 + size+duration+resolution + 旧路径不存在 → 最后兜底（1 分）
        """
        from .scanner import strip_bracket_prefix

        def _cand_fp_hash(cand: dict) -> Optional[str]:
            """候选记录的指纹 hash（无指纹则返回 None）"""
            if not cand.get("fingerprint_id"):
                return None
            row = self.conn.execute(
                "SELECT file_hash FROM video_fingerprints WHERE id=?",
                (cand["fingerprint_id"],),
            ).fetchone()
            return row["file_hash"] if row else None

        def _confirm(cand: dict) -> int:
            """严密确认候选 == 当前文件，返回证据分数（0 表示排除）。"""
            c_hash = _cand_fp_hash(cand)

            # 哈希矛盾 → 立即排除（宽松排除：宁可新建，不误判为同一）
            if file_hash and c_hash and file_hash != c_hash:
                return 0

            # E1 哈希一致 → 100% 同一
            if file_hash and c_hash and file_hash == c_hash:
                return 4

            # E2 路径命中 + size 无矛盾
            if path and (cand.get("current_path") == path or cand.get("original_path") == path):
                if file_size and cand.get("file_size"):
                    if abs(file_size - cand["file_size"]) / max(cand["file_size"], 1) > 0.001:
                        return 0  # size 矛盾 → 文件被替换
                # 路径命中且无 size 矛盾：
                #   候选无 hash（无法证伪）或调用方未提供 hash → 视为同一
                return 3

            # E3 粗指纹：size+duration 命中候选指纹，且候选指纹无 hash 可证伪
            if file_size and duration and cand.get("fingerprint_id"):
                fp = self.conn.execute(
                    "SELECT * FROM video_fingerprints WHERE id=?",
                    (cand["fingerprint_id"],),
                ).fetchone()
                if fp:
                    size_ok = abs(fp["file_size"] - file_size) / max(fp["file_size"], 1) <= 0.001
                    dur_ok = abs(fp["duration"] - duration) <= 0.5
                    if size_ok and dur_ok and not c_hash:
                        return 2  # 无 hash 可证伪，弱证据接受

            # E4 标题兜底：剥离前缀标题 + size/duration/resolution + 旧路径不存在
            stripped = strip_bracket_prefix(original_title or "")
            if stripped and strip_bracket_prefix(cand.get("original_title") or "") == stripped:
                if file_size and cand.get("file_size"):
                    if abs(file_size - cand["file_size"]) / max(cand["file_size"], 1) > 0.001:
                        return 0
                if duration and cand.get("duration"):
                    if abs(duration - cand["duration"]) > 0.5:
                        return 0
                if resolution and cand.get("resolution") and cand["resolution"] != resolution:
                    return 0
                old_cur = cand.get("current_path")
                old_orig = cand.get("original_path")
                old_cur_missing = old_cur and not Path(old_cur).exists()
                old_orig_missing = old_orig and not Path(old_orig).exists()
                if (old_cur_missing and old_orig_missing) or (old_cur_missing and not old_orig):
                    return 1
            return 0

        # ============ 宽松收集候选（宁多勿漏） ============
        candidates: dict[int, dict] = {}

        # C1 路径候选
        if path:
            row = self.conn.execute(
                "SELECT * FROM media_files WHERE current_path=? OR original_path=?",
                (path, path),
            ).fetchone()
            if row:
                candidates[row["id"]] = dict(row)

        # C2 粗指纹候选
        if file_size and duration:
            fps = self.conn.execute(
                """SELECT * FROM video_fingerprints
                   WHERE ABS(file_size - ?) / MAX(file_size, 1) <= 0.001
                     AND ABS(duration - ?) <= 0.5""",
                (file_size, duration),
            ).fetchall()
            for fp in fps:
                media = self.conn.execute(
                    "SELECT * FROM media_files WHERE fingerprint_id=?",
                    (fp["id"],),
                ).fetchone()
                if media:
                    candidates.setdefault(media["id"], dict(media))

        # C3 哈希候选（最强证据，单独优先）
        if file_hash:
            fp = self.find_fingerprint_by_hash(file_hash)
            if fp:
                media = self.conn.execute(
                    "SELECT * FROM media_files WHERE fingerprint_id=?",
                    (fp["id"],),
                ).fetchone()
                if media:
                    candidates.setdefault(media["id"], dict(media))

        # C4 标题候选
        stripped = strip_bracket_prefix(original_title or "")
        if stripped:
            for c in self.conn.execute(
                "SELECT * FROM media_files WHERE original_title = ? OR original_title LIKE ?",
                (stripped, "[%]%"),
            ).fetchall():
                if strip_bracket_prefix(c["original_title"]) == stripped:
                    candidates.setdefault(c["id"], dict(c))

        # ============ 严密确认：取证据分数最高的确认候选 ============
        best: Optional[dict] = None
        best_score = 0
        for cand in candidates.values():
            score = _confirm(cand)
            if score > best_score:
                best = cand
                best_score = score

        if best:
            fp_id = best.get("fingerprint_id")
            if fp_id:
                self.update_fingerprint_last_seen(fp_id)
            return best
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
