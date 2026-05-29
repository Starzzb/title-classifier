"""重命名模块"""

import csv
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class Renamer:
    """文件重命名器"""

    def __init__(self, csv_path: str = "data/output/title_review.csv", db_store=None):
        self.csv_path = Path(csv_path)
        self.db_store = db_store

    def rename(self, dry_run: bool = False) -> Dict:
        """
        执行重命名

        Args:
            dry_run: 是否模拟运行

        Returns:
            统计信息
        """
        if not self.csv_path.exists():
            logger.error(f"审核文件不存在: {self.csv_path}")
            return {"error": "文件不存在"}

        # 读取CSV
        tasks = self._read_csv()
        if not tasks:
            logger.warning("待审表为空")
            return {"total": 0}

        # 统计
        stats = {"confirmed": 0, "success": 0, "skip": 0, "conflict": 0, "error": 0}

        for task in tasks:
            status = task.get("review_status", "").strip()

            # 只处理已确认的记录
            if status != "已确认":
                stats["skip"] += 1
                continue

            stats["confirmed"] += 1

            # 获取路径
            original_path = Path(task["original_path"]).resolve()
            final_name = task.get("final_name", "").strip() or task["original_title"]

            if not original_path.exists():
                logger.warning(f"文件不存在: {original_path}")
                stats["error"] += 1
                continue

            # 构建新路径
            new_path = original_path.parent / f"{final_name}{original_path.suffix}"

            # 检查冲突
            if new_path.exists() and new_path != original_path:
                # 添加序号
                counter = 1
                while new_path.exists():
                    new_path = original_path.parent / f"{final_name}_{counter}{original_path.suffix}"
                    counter += 1
                stats["conflict"] += 1

            # 执行重命名
            if dry_run:
                logger.info(f"[模拟] {original_path.name} -> {new_path.name}")
                self._rename_srt(original_path, new_path, task.get("srt_path", ""), dry_run=True)
            else:
                try:
                    original_path.rename(new_path)
                    logger.info(f"[重命名] {original_path.name} -> {new_path.name}")
                    stats["success"] += 1
                    self._rename_srt(original_path, new_path, task.get("srt_path", ""), dry_run=False)
                    # 同步到数据库
                    self._sync_to_db(str(original_path), str(new_path))
                except Exception as e:
                    logger.error(f"重命名失败: {e}")
                    stats["error"] += 1

        return stats

    def _sync_to_db(self, old_path: str, new_path: str):
        """重命名后同步到数据库"""
        if not self.db_store:
            return
        try:
            media = self.db_store.find_by_path(old_path)
            if media:
                self.db_store.update_media(media["id"], "current_path", new_path, "rename")
                logger.debug(f"数据库路径更新: {old_path} -> {new_path}")
        except Exception as e:
            logger.warning(f"数据库同步失败: {e}")

    def _rename_srt(self, original_path: Path, new_path: Path, srt_path: str = "", dry_run: bool = False):
        """
        重命名字幕文件
        
        Args:
            original_path: 原视频路径
            new_path: 新视频路径
            srt_path: CSV中的srt_path字段（唯一来源）
            dry_run: 是否模拟运行
        """
        if not srt_path:
            return

        srt_file = Path(srt_path)
        if not srt_file.exists():
            return

        new_srt_name = new_path.stem + ".srt"
        new_srt_path = srt_file.parent / new_srt_name

        if srt_file == new_srt_path:
            return  # 名称相同，无需重命名

        if dry_run:
            logger.info(f"[模拟] SRT: {srt_file.name} -> {new_srt_name}")
        else:
            try:
                if new_srt_path.exists():
                    counter = 1
                    while new_srt_path.exists():
                        new_srt_name = f"{new_path.stem}_{counter}.srt"
                        new_srt_path = srt_file.parent / new_srt_name
                        counter += 1
                srt_file.rename(new_srt_path)
                logger.info(f"[SRT重命名] {srt_file.name} -> {new_srt_name}")
            except Exception as e:
                logger.warning(f"SRT重命名失败: {e}")

    def _read_csv(self) -> List[Dict]:
        """读取CSV文件"""
        try:
            with open(self.csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logger.error(f"读取CSV失败: {e}")
            return []
