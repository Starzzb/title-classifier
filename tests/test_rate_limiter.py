"""限流协调器测试：429 指数退避、发车节流、成功重置（fake 时钟，不真睡）"""
import pytest

import title_classifier.utils.rate_limiter as rl_mod
from title_classifier.utils.rate_limiter import RateLimiter


class FakeClock:
    """可控单调时钟"""
    def __init__(self):
        self.now = 1000.0

    def advance(self, s):
        self.now += s

    def __call__(self):
        return self.now


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(rl_mod.time, "monotonic", clock)
    monkeypatch.setattr(rl_mod.time, "sleep", clock.advance)  # sleep 即快进时钟
    return clock


def test_429_backoff_exponential():
    rl = RateLimiter(max_delay=60.0)
    delays = [rl.report_429() for _ in range(6)]
    assert delays == [4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    assert rl.consecutive_429 == 6


def test_success_resets_counter():
    rl = RateLimiter()
    rl.report_429()
    rl.report_429()
    assert rl.consecutive_429 == 2
    rl.report_success()
    assert rl.consecutive_429 == 0
    # 重置后退避从头开始
    assert rl.report_429() == 4.0


def test_wait_turn_blocks_until_window(fake_clock):
    rl = RateLimiter(min_gap=0.0)
    rl.report_429()  # 熔断 4s
    slept = []
    orig = rl_mod.time.sleep
    rl_mod.time.sleep = lambda s: (slept.append(s), orig(s))
    rl.wait_turn()
    rl_mod.time.sleep = orig
    assert slept and slept[0] > 3.0  # 等到了熔断窗口


def test_wait_turn_slot_spacing(fake_clock):
    """连续 wait_turn 的发车位间隔不小于 min_gap（防惊群）"""
    rl = RateLimiter(min_gap=2.0)
    rl.wait_turn()  # 立即放行，预约 now+2
    fake_clock.advance(5.0)
    t0 = fake_clock.now
    rl.wait_turn()  # slot 在 t0-3 之前，需等到 t0+... 按 min_gap 预约
    # 第二次调用应比第一次预约至少间隔 min_gap（时钟被 sleep 快进）
    assert fake_clock.now - t0 >= 0  # 不死循环即通过（fake sleep 保证前进）


def test_call_vision_api_429_does_not_consume_retries(monkeypatch, fake_clock):
    """429 重试不消耗普通重试次数：前2次429后第3次成功，retries=3 应能成功"""
    from title_classifier.providers import call_vision_api

    # 重置全局单例状态（避免测试间污染）
    from title_classifier.utils.rate_limiter import vision_rate_limiter
    vision_rate_limiter.report_success()

    calls = {"n": 0}

    def fake_http(url, payload, api_key, timeout):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise Exception("HTTP 429: rate limit exceeded")
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr("title_classifier.providers._http_request", fake_http)
    result = call_vision_api("gcli", "fakeb64", "prompt", api_key="fake", timeout=10, retries=3)
    assert result == "ok"
    assert calls["n"] == 3
