"""崩溃报告工具 - 捕获未处理异常并写入日志文件"""

import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CRASH_DIR = Path("logs") / "crash"


def _ensure_crash_dir():
    CRASH_DIR.mkdir(parents=True, exist_ok=True)


def write_crash_dump(exc_type, exc_value, exc_tb):
    """将崩溃信息写入文件"""
    _ensure_crash_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crash_file = CRASH_DIR / f"crash_{timestamp}.log"

    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_text = "".join(tb_lines)

    with open(crash_file, "w", encoding="utf-8") as f:
        f.write(f"=== 崩溃报告 ===\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"类型: {exc_type.__name__}\n")
        f.write(f"值: {exc_value}\n")
        f.write(f"进程ID: {os.getpid()}\n")
        f.write(f"工作目录: {os.getcwd()}\n")
        f.write(f"\n=== 调用栈 ===\n")
        f.write(tb_text)
        f.write(f"\n=== 结束 ===\n")

    logger.critical(f"崩溃报告已保存: {crash_file}")
    return crash_file


def install_excepthook():
    """安装全局异常钩子，捕获所有未处理异常"""
    _ensure_crash_dir()

    def crash_hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        crash_file = write_crash_dump(exc_type, exc_value, exc_tb)
        try:
            print(f"\n[致命错误] 程序遇到未预期错误，崩溃报告已保存: {crash_file}", file=sys.stderr)
        except Exception:
            pass

    sys.excepthook = crash_hook
    return crash_hook
