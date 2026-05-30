"""更新CSV中的original_path：去除文件名的中括号前缀"""

import csv
import os
import re
import tempfile
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def strip_bracket(name):
    return re.sub(r"^\[([^\]]+)\]_?", "", name, count=1)

def atomic_write_csv(csv_path, rows, fieldnames):
    csv_path = os.path.abspath(csv_path)
    csv_dir = os.path.dirname(csv_path) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=csv_dir, suffix=".tmp", prefix=".csv_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, csv_path)
    except Exception:
        try: os.unlink(tmp_path)
        except: pass
        raise

project_dir = Path(__file__).parent.parent
csv_files = sorted((project_dir / "data" / "output").rglob("title_review.csv"))

total_updated = 0
for csv_path in csv_files:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    changed = False
    updated = 0
    for row in rows:
        orig = row.get("original_path", "")
        if not orig:
            continue
        fname = os.path.basename(orig)
        clean = strip_bracket(fname)
        if clean != fname:
            # 文件名有 [关键词]_ 前缀，去掉
            new_path = str(Path(orig).parent / clean)
            row["original_path"] = new_path
            changed = True
            updated += 1

    if changed:
        atomic_write_csv(str(csv_path), rows, fieldnames)
        total_updated += updated
        print(f"[更新] {csv_path.parent.name}/{csv_path.name}: {updated} 条")
    else:
        print(f"[跳过] {csv_path.parent.name}/{csv_path.name}")

print(f"\n总计更新: {total_updated} 条")
