"""全局 API 限流协调器

线程安全的跨线程退避：任一 worker 撞到 429，全体 worker 一起暂停（全局熔断窗口），
退避结束后按发车节流间隔依次放行，避免惊群再次撞墙。

典型用途：场景模式并发段分析 + 多视频 --concurrent 时共享同一 VLM API 配额。
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """429 指数退避 + 发车节流（线程安全，全局单例使用）"""

    def __init__(self, max_delay: float = 60.0, min_gap: float = 1.0):
        self._lock = threading.Lock()
        self._blocked_until = 0.0     # 全局熔断窗口结束时间 (monotonic)
        self._consecutive = 0         # 连续 429 次数
        self._next_slot = 0.0         # 下一个允许发车的时间 (monotonic)
        self._max_delay = max_delay
        self._min_gap = min_gap

    def wait_turn(self):
        """调用 API 前调用：若处于熔断窗口则等待；同时预约发车位（间隔 min_gap）"""
        with self._lock:
            now = time.monotonic()
            my_slot = max(self._blocked_until, self._next_slot, now)
            self._next_slot = my_slot + self._min_gap
            delay = my_slot - now
        if delay > 0:
            time.sleep(delay)

    def report_429(self) -> float:
        """报告一次 429：连续计数+1，全局熔断窗口按指数退避延长

        Returns:
            本次的退避秒数（4, 8, 16, 32, 60, 60... 封顶）
        """
        with self._lock:
            self._consecutive += 1
            delay = min(4.0 * (2 ** (self._consecutive - 1)), self._max_delay)
            self._blocked_until = time.monotonic() + delay
        logger.warning(f"API限流(429): 全局退避 {delay:.0f}s (连续第{self._consecutive}次)")
        return delay

    def report_success(self):
        """调用成功：清零连续 429 计数"""
        with self._lock:
            self._consecutive = 0

    @property
    def consecutive_429(self) -> int:
        with self._lock:
            return self._consecutive


# 全局单例：所有 VLM 调用（场景段并发、多视频并发）共享同一配额状态
vision_rate_limiter = RateLimiter()
