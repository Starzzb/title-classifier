"""批量修复所有CSV中的纯中括号final_name"""

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fix_bracket_only import fix_bracket_only

csvs = sorted(glob.glob("data/output/*/title_review.csv"))
print(f"找到 {len(csvs)} 个CSV文件\n")

total_fixed = 0
for csv_path in csvs:
    print(f"{'='*60}")
    print(f"处理: {csv_path}")
    print(f"{'='*60}")
    stats = fix_bracket_only(csv_path, dry_run=False)
    if "error" not in stats:
        total_fixed += stats["fixed"]
    print()

print(f"\n{'='*60}")
print(f"全部完成！共修复 {total_fixed} 条记录")
