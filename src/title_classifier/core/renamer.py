"""重命名模块"""

import csv
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

from ..utils.file_resolve import resolve_media_path

logger = logging.getLogger(__name__)


class Renamer:
    """文件重命名器"""

    def __init__(self, csv_path: str = "data/output/title_review.csv", db_store=None,
                 use_rclone: bool = False, rclone_path: str = "rclone", max_workers: int = 5):
        self.csv_path = Path(csv_path)
        self.db_store = db_store
        self.use_rclone = use_rclone
        self.rclone_path = rclone_path
        self.max_workers = max_workers

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

        # 收集需要重命名的任务
        rename_tasks = []

        for task in tasks:
            status = task.get("review_status", "").strip()

            # 只处理已确认的记录
            if status != "已确认":
                stats["skip"] += 1
                continue

            stats["confirmed"] += 1

            # 获取路径（Stage2重命名后用final_name回退查找）
            original_path_str = task.get("original_path", "").strip()
            original_title = task.get("original_title", "").strip()
            final_name = task.get("final_name", "").strip() or original_title

            resolved = resolve_media_path(original_path_str, final_name, original_title)
            if not resolved:
                logger.warning(f"文件不存在: {original_path_str}")
                stats["error"] += 1
                continue
            if resolved != original_path_str:
                logger.info(f"[路径回退] {Path(original_path_str).name} → {Path(resolved).name}")
            try:
                original_path = Path(resolved).resolve()
            except OSError:
                original_path = Path(resolved)

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

            rename_tasks.append((task, original_path, new_path))

        # 执行重命名
        if dry_run:
            for task, original_path, new_path in rename_tasks:
                logger.info(f"[模拟] {original_path.name} -> {new_path.name}")
                self._rename_srt(original_path, new_path, task.get("srt_path", ""), dry_run=True)
        elif self.use_rclone:
            # 使用 rclone 并行重命名
            stats = self._rename_parallel(rename_tasks, stats)
        else:
            # 串行重命名
            for task, original_path, new_path in rename_tasks:
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

    def _rename_parallel(self, rename_tasks: list, stats: Dict) -> Dict:
        """使用 rclone 并行重命名"""
        def rename_one(task_info):
            task, original_path, new_path = task_info
            try:
                success = self._rename_with_rclone(original_path, new_path)
                if success:
                    logger.info(f"[rclone重命名] {original_path.name} -> {new_path.name}")
                    self._rename_srt(original_path, new_path, task.get("srt_path", ""), dry_run=False)
                    self._sync_to_db(str(original_path), str(new_path))
                    return "success"
                else:
                    return "error"
            except Exception as e:
                logger.error(f"重命名异常: {e}")
                return "error"

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(rename_one, task_info): task_info 
                       for task_info in rename_tasks}
            for future in as_completed(futures):
                result = future.result()
                stats[result] += 1

        return stats

    def _rename_with_rclone(self, src: Path, dst: Path) -> bool:
        """使用 rclone moveto 重命名"""
        try:
            cmd = [self.rclone_path, "moveto", str(src), str(dst)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return True
            else:
                logger.error(f"rclone 失败: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"rclone 超时: {src.name}")
            return False
        except Exception as e:
            logger.error(f"rclone 异常: {e}")
            return False

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
