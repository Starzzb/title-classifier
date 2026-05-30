"""提取原标题脚本：[关键词]_原标题 → 原标题

从 final_name 中剥离 [关键词]_ 前缀，还原为原标题。

规则：
  [关键词]_原标题  → 原标题
  [关键词]         → 不动（无后缀，无法判断原标题）
  原标题           → 不动

用法：
  python scripts/extract_original.py data/output/Download/title_review.csv --dry-run
  python scripts/extract_original.py data/output/Download/title_review.csv
  python scripts/extract_original.py --all                 # 所有CSV
  python scripts/extract_original.py --all --dry-run       # 模拟
"""

import csv
import os
import re
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 匹配 [关键词]_原标题 格式
BRACKET_PREFIX = re.compile(r"^\[[^\]]*\]_?(.+)$")


def atomic_write_csv(csv_path, rows, fieldnames, encoding="utf-8-sig"):
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


def strip_bracket_prefix(name: str) -> str:
    """去掉 [关键词]_ 前缀，返回原标题。无前缀则返回原值。"""
    m = BRACKET_PREFIX.match(name)
    return m.group(1) if m else name


def extract_original(csv_path: str, dry_run: bool = False) -> dict:
    """
    从 final_name 提取原标题。

    Args:
        csv_path: CSV文件路径
        dry_run: 模拟运行

    Returns:
        {"total": int, "fixed": int, "skipped": int}
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[错误] 文件不存在: {csv_path}")
        return {"error": "文件不存在"}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    stats = {"total": len(rows), "fixed": 0, "skipped": 0}

    for row in rows:
        final_name = row.get("final_name", "").strip()
        if not final_name:
            stats["skipped"] += 1
            continue

        cleaned = strip_bracket_prefix(final_name)
        if cleaned != final_name:
            old = final_name
            row["final_name"] = cleaned
            stats["fixed"] += 1
            if dry_run:
                print(f"  [模拟] {old[:50]} → {cleaned[:50]}")
            else:
                print(f"  [修复] {old[:50]} → {cleaned[:50]}")
        else:
            stats["skipped"] += 1

    if not dry_run and stats["fixed"] > 0:
        atomic_write_csv(str(csv_path), rows, fieldnames)

    return stats


def find_all_csvs() -> list:
    project_dir = Path(__file__).parent.parent
    output_dir = project_dir / "data" / "output"
    if not output_dir.exists():
        return []
    return sorted(output_dir.rglob("title_review.csv"))


def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if dry_run:
        print("[模拟运行] 不会修改文件\n")

    if "--all" in sys.argv or not args:
        csvs = find_all_csvs()
        print(f"找到 {len(csvs)} 个CSV文件\n")
        total_fixed = 0
        for csv_path in csvs:
            print(f"{'='*50}")
            print(f"处理: {csv_path.parent.name}/{csv_path.name}")
            print(f"{'='*50}")
            stats = extract_original(str(csv_path), dry_run=dry_run)
            if "error" not in stats:
                total_fixed += stats["fixed"]
            print()
        print(f"\n全部完成！共修复 {total_fixed} 条记录")
    else:
        for csv_path in args:
            stats = extract_original(csv_path, dry_run=dry_run)
            if "error" not in stats:
                print(f"\n[统计] 总计: {stats['total']}, 修复: {stats['fixed']}, 跳过: {stats['skipped']}")


if __name__ == "__main__":
    main()
