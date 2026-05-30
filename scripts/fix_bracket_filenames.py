"""修复磁盘上纯中括号包裹的文件名：去除最外层[]

规则：
  [标题].mp4     → 标题.mp4       （去掉括号）
  [关键词]_标题.mp4 → 不动          （有后缀，保留）
  标题.mp4       → 不动           （无括号，跳过）

同时更新CSV中对应的 original_path。

用法：
  python scripts/fix_bracket_filenames.py G:\ --dry-run
  python scripts/fix_bracket_filenames.py G:\
  python scripts/fix_bracket_filenames.py G:\好的 G:\Download --dry-run
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

# 匹配纯中括号包裹的文件名：[任意内容].扩展名
BRACKET_FILE = re.compile(r"^\[([^\]]+)\](\.\w+)$")

# 媒体扩展名
MEDIA_EXT = {
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff",
    ".srt",
}


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


def scan_and_rename(target_dirs: list[str], dry_run: bool = False) -> dict:
    """
    扫描目录，重命名纯中括号包裹的文件，更新CSV。

    Returns:
        {"renamed": int, "csv_updated": int, "errors": int}
    """
    stats = {"renamed": 0, "csv_updated": 0, "errors": 0}

    # 收集所有需要重命名的文件: {old_path: new_path}
    renames = {}

    for target_dir in target_dirs:
        target = Path(target_dir)
        if not target.exists():
            print(f"[跳过] 目录不存在: {target}")
            continue

        if target.is_file():
            # 单文件处理
            files = [target]
        else:
            # 递归扫描目录
            files = list(target.rglob("*"))

        for f in files:
            if not f.is_file():
                continue
            if f.suffix.lower() not in MEDIA_EXT:
                continue

            m = BRACKET_FILE.match(f.name)
            if m:
                content = m.group(1)
                ext = m.group(2)
                new_name = f"{content}{ext}"
                new_path = f.parent / new_name
                renames[str(f)] = str(new_path)

    if not renames:
        print("[完成] 没有需要修复的文件")
        return stats

    print(f"找到 {len(renames)} 个需要去除中括号的文件\n")

    # 执行重命名
    for old_path, new_path in renames.items():
        old_name = os.path.basename(old_path)
        new_name = os.path.basename(new_path)

        if os.path.exists(new_path):
            print(f"  [冲突] 目标已存在: {new_name}")
            stats["errors"] += 1
            continue

        if dry_run:
            print(f"  [模拟] {old_name} → {new_name}")
        else:
            try:
                os.rename(old_path, new_path)
                print(f"  [重命名] {old_name} → {new_name}")
                stats["renamed"] += 1
            except Exception as e:
                print(f"  [错误] {old_name}: {e}")
                stats["errors"] += 1

    # 更新CSV中的 original_path
    if not dry_run and stats["renamed"] > 0:
        stats["csv_updated"] = _update_csvs(renames)
    elif dry_run:
        # 模拟运行时统计需要更新的CSV数
        stats["csv_updated"] = _count_csv_updates(renames)

    return stats


def _find_csv_files() -> list[Path]:
    """找到所有 title_review.csv"""
    project_dir = Path(__file__).parent.parent
    output_dir = project_dir / "data" / "output"
    if not output_dir.exists():
        return []
    return sorted(output_dir.rglob("title_review.csv"))


def _count_csv_updates(renames: dict) -> int:
    """统计需要更新多少个CSV文件"""
    count = 0
    for csv_path in _find_csv_files():
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig = row.get("original_path", "")
                    if orig in renames:
                        count += 1
                        break
        except Exception:
            pass
    return count


def _update_csvs(renames: dict) -> int:
    """更新所有CSV中的 original_path"""
    updated_files = 0

    for csv_path in _find_csv_files():
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames)
                rows = list(reader)
        except Exception:
            continue

        changed = False
        for row in rows:
            orig = row.get("original_path", "")
            if orig in renames:
                row["original_path"] = renames[orig]
                changed = True

        if changed:
            atomic_write_csv(str(csv_path), rows, fieldnames)
            updated_files += 1
            print(f"  [CSV更新] {csv_path}")

    return updated_files


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/fix_bracket_filenames.py <目录1> [目录2 ...] [--dry-run]")
        print("示例: python scripts/fix_bracket_filenames.py G:\\ --dry-run")
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]

    if dry_run:
        print("[模拟运行] 不会修改文件\n")

    stats = scan_and_rename(dirs, dry_run=dry_run)

    print(f"\n[统计] 重命名: {stats['renamed']}, CSV更新: {stats['csv_updated']}, 错误: {stats['errors']}")


if __name__ == "__main__":
    main()
