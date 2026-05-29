"""
CSV 导入工具：从现有 CSV 导入数据到 SQLite 数据库

用法:
    python scripts/import_csv.py                    # 导入所有 data/output/ 下的 CSV
    python scripts/import_csv.py --csv "path/to.csv"  # 导入指定 CSV

示例:
    python scripts/import_csv.py
    python scripts/import_csv.py --csv "data/output/love/title_review.csv"
    python scripts/import_csv.py --all
"""

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from title_classifier.core.db_store import MediaDB


def main():
    parser = argparse.ArgumentParser(description="从 CSV 导入数据到 SQLite 数据库")
    parser.add_argument("--csv", help="指定单个 CSV 文件路径")
    parser.add_argument("--all", action="store_true", help="导入所有 CSV 文件（默认行为）")
    parser.add_argument("--db", help="指定数据库路径（默认 data/media.db）")
    args = parser.parse_args()

    db_path = args.db or str(PROJECT_ROOT / "data" / "media.db")
    db = MediaDB(db_path)
    db.init_schema()

    print(f"[数据库] {db_path}")
    print(f"[当前记录数] {db.count()}")

    start = time.time()

    if args.csv:
        # 导入单个 CSV
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[错误] CSV 不存在: {csv_path}")
            sys.exit(1)

        print(f"\n[导入] {csv_path}")
        stats = db.import_csv(args.csv)
        elapsed = time.time() - start
        print(f"[完成] {elapsed:.1f}秒")
        print(f"  新增: {stats['imported']}")
        print(f"  更新: {stats['updated']}")
        print(f"  跳过: {stats['skipped']}")
        print(f"  标签: {stats['tags_added']}")
    else:
        # 导入所有 CSV
        print(f"\n[扫描] data/output/ 下的所有 CSV...")
        stats = db.import_all_csvs()
        elapsed = time.time() - start
        print(f"[完成] {elapsed:.1f}秒")
        print(f"  新增: {stats['imported']}")
        print(f"  更新: {stats['updated']}")
        print(f"  跳过: {stats['skipped']}")
        print(f"  标签: {stats['tags_added']}")

    print(f"\n[数据库统计]")
    db_stats = db.get_stats()
    print(f"  媒体文件: {db_stats['total_media']}")
    print(f"  视频: {db_stats['videos']}")
    print(f"  图片: {db_stats['images']}")
    print(f"  标签: {db_stats['total_tags']}")
    print(f"  改动记录: {db_stats['total_changes']}")
    print(f"  VLM帧: {db_stats['total_frames']}")

    db.close()


if __name__ == "__main__":
    main()
