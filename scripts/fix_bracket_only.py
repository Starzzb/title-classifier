"""修复脚本：去除仅被中括号包裹的 final_name 的括号

规则：
  [标题]        → 标题       （去掉括号）
  [关键词]_原标题 → 不动       （有后缀，保留）
  标题          → 不动       （无括号，跳过）

用法：
  python scripts/fix_bracket_only.py data/output/Download/title_review.csv
  python scripts/fix_bracket_only.py data/output/Download/title_review.csv --dry-run
"""

import csv
import re
import sys
import os
import tempfile
from pathlib import Path

# 处理控制台编码问题
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def atomic_write_csv(csv_path, rows, fieldnames, encoding="utf-8-sig"):
    """原子化写入CSV"""
    csv_path = os.path.abspath(csv_path)
    csv_dir = os.path.dirname(csv_path) or "."
    os.makedirs(csv_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=csv_dir, suffix=".tmp", prefix=".csv_")
    try:
        with os.fdopen(tmp_fd, "w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, csv_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# 匹配纯中括号包裹：整行就是 [xxx] 且后面没有其他内容
BRACKET_ONLY = re.compile(r"^\[([^\]]+)\]$")


def fix_bracket_only(csv_path: str, dry_run: bool = False) -> dict:
    """
    修复纯中括号包裹的 final_name。

    Args:
        csv_path: CSV 文件路径
        dry_run: 是否模拟运行

    Returns:
        统计信息 {"total": int, "fixed": int, "skipped": int}
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[错误] 文件不存在: {csv_path}")
        return {"error": "文件不存在"}

    # 读取
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    stats = {"total": len(rows), "fixed": 0, "skipped": 0}

    for i, row in enumerate(rows):
        final_name = row.get("final_name", "").strip()
        if not final_name:
            continue

        m = BRACKET_ONLY.match(final_name)
        if m:
            content = m.group(1)
            old = final_name
            row["final_name"] = content
            stats["fixed"] += 1
            if dry_run:
                print(f"  [模拟] {old} → {content}")
            else:
                print(f"  [修复] {old} → {content}")
        else:
            stats["skipped"] += 1

    # 写回
    if not dry_run and stats["fixed"] > 0:
        atomic_write_csv(str(csv_path), rows, fieldnames)

    print(f"\n[统计] 总计: {stats['total']}, 修复: {stats['fixed']}, 跳过: {stats['skipped']}")
    return stats


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/fix_bracket_only.py <csv_path> [--dry-run]")
        sys.exit(1)

    csv_path = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("[模拟运行] 不会修改文件\n")

    fix_bracket_only(csv_path, dry_run=dry_run)


if __name__ == "__main__":
    main()
