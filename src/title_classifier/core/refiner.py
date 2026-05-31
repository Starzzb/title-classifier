"""AI标题优化模块 - 并发批处理版"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..providers import call_text_api, get_provider_config, get_api_key
from ..utils.prompt_loader import get_prompt

logger = logging.getLogger(__name__)

# 每批处理的标题数量
BATCH_SIZE = 10
# 最大并发批次数
MAX_WORKERS = 3


class Refiner:
    """AI标题优化器 - 支持并发"""

    def __init__(self, provider: str = "gcli"):
        self.provider = provider
        self.config = get_provider_config(provider)

    def refine_batch(
        self,
        titles: List[str],
        progress_callback: Callable[[int, int, str], None] = None,
    ) -> List[str]:
        """
        并发分批优化标题

        Args:
            titles: 标题列表
            progress_callback: 进度回调函数 (当前索引, 总数, 当前标题) -> None

        Returns:
            优化后的标题列表（与输入等长、等序）
        """
        if not titles:
            return []

        total = len(titles)

        # 构建批次: [(batch_idx, batch_titles, start_index), ...]
        batches = []
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batches.append((batch_start, titles[batch_start:batch_end]))

        all_results: Dict[int, List[str]] = {}
        completed = 0

        def process_one(batch_start: int, batch_titles: List[str]) -> tuple:
            result = self._refine_single_batch(batch_titles)
            return batch_start, result

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_one, bs, bt): bs
                for bs, bt in batches
            }

            for future in as_completed(futures):
                bs = futures[future]
                try:
                    batch_start, batch_results = future.result()
                    all_results[batch_start] = batch_results
                    completed += len(batch_results)
                    if progress_callback:
                        # 报告整体进度
                        progress_callback(completed, total, f"批次 {batch_start // BATCH_SIZE + 1}")
                except Exception as e:
                    logger.error(f"批次 {bs} 失败: {e}")
                    batch_titles = [bt for b, bt in batches if b == bs][0]
                    all_results[bs] = list(batch_titles)  # 失败时保留原标题
                    completed += len(batch_titles)

        # 按原始顺序合并结果
        ordered = []
        for batch_start in sorted(all_results.keys()):
            ordered.extend(all_results[batch_start])

        return ordered[:total]

    def _refine_single_batch(self, titles: List[str]) -> List[str]:
        """优化单批标题"""
        prompt = self._build_batch_prompt(titles)
        result = call_text_api(self.provider, prompt)
        return self._parse_batch_response(result, len(titles), original_titles=titles)

    def _build_batch_prompt(self, titles: List[str]) -> str:
        """构建批量提示词"""
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))
        return (
            f"{get_prompt('refiner', 'system_header')}\n\n"
            f"{get_prompt('refiner', 'task_instruction')}\n\n"
            f"请精简以下文件名，每行一个，保持顺序，不要序号：\n{numbered}\n\n"
            f"{get_prompt('refiner', 'output_format')}"
        )

    def _parse_batch_response(self, response: str, count: int, original_titles: List[str] = None) -> List[str]:
        """解析批量响应

        Args:
            response: AI返回的响应文本
            count: 期望的结果数量
            original_titles: 原始标题列表（用于空行回退）

        Returns:
            解析后的结果列表
        """
        import re

        lines = response.strip().split("\n")
        results = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[.)\s、]+", "", line).strip()
            if cleaned:
                results.append(cleaned)

        while len(results) < count:
            results.append("")

        if original_titles:
            for i in range(min(len(results), len(original_titles))):
                if not results[i]:
                    results[i] = original_titles[i]

        return results[:count]
