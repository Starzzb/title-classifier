"""
回填 video_fingerprints：为数据库中在磁盘上存在的记录计算内容指纹并关联。

用法:
    python scripts/backfill_fingerprints.py [--target "D:\\Telegram"] [--db data/media.db] [--dry-run]

匹配策略:
    对每条磁盘上存在的 media_files 记录：
      1. 计算内容指纹（首尾采样 xxhash）
      2. 按指纹查找已有 video_fingerprints，存在则复用，否则新建
      3. 设置 media_files.fingerprint_id
    视觉描述 (vision_description) 和标签 (media_tags) 已存在数据库中，
    此脚本不触碰这些数据，只建立指纹关联。

    --target 可多次指定；不指定则处理所有在盘记录。
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from title_classifier.utils.fingerprint import compute_partial_hash


def main():
    parser = argparse.ArgumentParser(description="回填视频指纹")
    parser.add_argument("--db", default="data/media.db")
    parser.add_argument("--target", action="append", default=None, help="目标目录前缀，可多次指定")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
    parser.add_argument("--min-age-days", type=int, default=0,
                        help="只处理 created_at 距今超过 N 天的记录（0=全部）")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    query = """
        SELECT id, original_title, original_path, current_path,
               file_size, duration, fingerprint_id, vision_description
        FROM media_files
        WHERE (current_path IS NOT NULL AND current_path != '')
    """
    params = []
    if args.target:
        likes = []
        for t in args.target:
            likes.append("(current_path LIKE ? OR original_path LIKE ?)")
            params.extend([f"{t}%", f"{t}%"])
        query += " AND (" + " OR ".join(likes) + ")"
    if args.min_age_days:
        query += " AND created_at <= datetime('now','localtime', ?)"
        params.append(f"-{args.min_age_days} days")

    rows = c.execute(query, params).fetchall()
    print(f"待处理记录: {len(rows)}")

    t0 = time.time()
    done = 0
    skipped_no_file = 0
    skipped_has_fp = 0
    linked = 0
    created_fp = 0
    errors = 0

    for r in rows:
        if r["fingerprint_id"] is not None:
            skipped_has_fp += 1
            continue

        path = r["current_path"]
        if not Path(path).exists():
            skipped_no_file += 1
            continue

        if not r["file_size"] or not r["duration"]:
            skipped_no_file += 1
            continue

        fhash = compute_partial_hash(path)
        if not fhash:
            errors += 1
            continue

        fp = c.execute(
            "SELECT id FROM video_fingerprints WHERE file_hash=?",
            (fhash,)
        ).fetchone()

        if fp:
            fp_id = fp["id"]
        else:
            if args.dry_run:
                fp_id = -1
            else:
                cur = c.execute(
                    "INSERT INTO video_fingerprints (file_size, duration, file_hash) VALUES (?, ?, ?)",
                    (r["file_size"], r["duration"], fhash)
                )
                fp_id = cur.lastrowid
                created_fp += 1

        if args.dry_run:
            linked += 1
        else:
            c.execute(
                "UPDATE media_files SET fingerprint_id=? WHERE id=?",
                (fp_id, r["id"])
            )
            linked += 1

        done += 1
        if done % 50 == 0 or done == len(rows):
            print(f"  进度: {done}/{len(rows)}  ({time.time()-t0:.0f}s) 当前: {Path(path).name[:40]}")
            sys.stdout.flush()

    if not args.dry_run:
        conn.commit()

    print(f"\n[完成]")
    print(f"  处理: {done}, 关联指纹: {linked}, 新建指纹: {created_fp}")
    print(f"  跳过(已有指纹): {skipped_has_fp}")
    print(f"  跳过(文件不存在/无元数据): {skipped_no_file}")
    print(f"  失败: {errors}")
    print(f"  耗时: {time.time()-t0:.1f}s")
    conn.close()


if __name__ == "__main__":
    main()
