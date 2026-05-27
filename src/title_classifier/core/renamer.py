"""重命名模块"""

import csv
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class Renamer:
    """文件重命名器"""

    def __init__(self, csv_path: str = "data/output/title_review.csv"):
        self.csv_path = Path(csv_path)

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
            else:
                try:
                    original_path.rename(new_path)
                    logger.info(f"[重命名] {original_path.name} -> {new_path.name}")
                    stats["success"] += 1
                except Exception as e:
                    logger.error(f"重命名失败: {e}")
                    stats["error"] += 1

        return stats

    def _read_csv(self) -> List[Dict]:
        """读取CSV文件"""
        try:
            with open(self.csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logger.error(f"读取CSV失败: {e}")
            return []
