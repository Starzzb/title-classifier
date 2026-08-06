"""文件内容指纹工具 - 使用 xxhash 对文件首尾采样生成指纹"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:
    import hashlib
    _HAS_XXHASH = False
    logger.warning("xxhash 未安装，回退使用 sha256")


def _hash_bytes(data: bytes) -> str:
    if _HAS_XXHASH:
        return xxhash.xxh64(data).hexdigest()
    return hashlib.sha256(data).hexdigest()[:16]


def compute_partial_hash(path, head_bytes: int = 8 * 1024 * 1024, tail_bytes: int = 8 * 1024 * 1024) -> str:
    """首尾采样 xxhash 指纹。

    文件大小 > head_bytes + tail_bytes 时，只读取头部和尾部；
    否则读取整个文件。

    Args:
        path: 文件路径
        head_bytes: 头部采样字节数（默认 8MB）
        tail_bytes: 尾部采样字节数（默认 8MB）

    Returns:
        xxh64 十六进制指纹字符串，读取失败返回 None
    """
    try:
        file_path = Path(path)
        total = file_path.stat().st_size

        if total <= head_bytes + tail_bytes:
            return _hash_bytes(file_path.read_bytes())

        if _HAS_XXHASH:
            h = xxhash.xxh64()
            with open(file_path, "rb") as f:
                h.update(f.read(head_bytes))
                f.seek(total - tail_bytes)
                h.update(f.read(tail_bytes))
            return h.hexdigest()

        # 无 xxhash 时的 sha256 回退
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            h.update(f.read(head_bytes))
            f.seek(total - tail_bytes)
            h.update(f.read(tail_bytes))
        return h.hexdigest()[:16]

    except Exception as e:
        logger.warning(f"计算文件指纹失败: {path}: {e}")
        return None
