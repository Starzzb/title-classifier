"""Stage2 重命名 Tab - 从主窗口提取的独立组件"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import ttkbootstrap as ttk

from .context import AppContext
from .app import ToolTip

PYTHON = sys.executable
PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()


class StageRenameTab(ttk.Frame):
    """Stage2 重命名标签页"""

    def __init__(self, master, ctx: AppContext, run_command=None, sync_csv_to_db=None, **kwargs):
        super().__init__(master, **kwargs)
        self.ctx = ctx
        self._run_command = run_command
        self._sync_csv_to_db = sync_csv_to_db
        self._build()

    def _build(self):
        # CSV文件
        csv_frame = ttk.LabelFrame(self, text="CSV文件")
        csv_frame.pack(fill=tk.X, padx=4, pady=4)

        csv_entry = ttk.Entry(csv_frame, textvariable=self.ctx.csv_var, width=60)
        csv_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(csv_frame, text="浏览...", command=self._browse_csv_s2).pack(side=tk.LEFT, padx=4)
        ToolTip(csv_entry, "包含final_name的CSV文件，用于批量重命名")

        # 批量操作
        batch_frame = ttk.LabelFrame(self, text="批量操作")
        batch_frame.pack(fill=tk.X, padx=4, pady=4)

        confirm_btn = ttk.Button(batch_frame, text="一键确认所有记录", command=self._batch_confirm)
        confirm_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(confirm_btn, "将CSV中所有记录的review_status设置为'已确认'\n\n"
                "只有review_status='已确认'的记录才会被重命名")

        clear_btn = ttk.Button(batch_frame, text="一键清空final_name", command=self._batch_clear_final_name)
        clear_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(clear_btn, "清空CSV中所有记录的final_name字段\n\n"
                "用于重置优化结果，重新开始处理")

        # 选项
        opt_frame = ttk.LabelFrame(self, text="选项")
        opt_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s2_dry_run_var = tk.BooleanVar(value=True)
        dry_run_cb = ttk.Checkbutton(opt_frame, text="模拟运行", variable=self.s2_dry_run_var)
        dry_run_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(dry_run_cb, "勾选后只显示重命名预览，不实际修改文件\n\n"
                "- 勾选：安全模式，只预览不执行\n"
                "- 取消勾选：实际执行重命名操作")

        # 执行按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        rename_btn = ttk.Button(btn_frame, text="执行重命名", command=self._run_rename)
        rename_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(rename_btn, "根据CSV中的final_name批量重命名文件\n\n"
                "重命名规则：\n"
                "- 只处理review_status='已确认'的记录\n"
                "- 新文件名 = final_name + 原扩展名\n"
                "- 如果目标文件已存在，自动添加序号")

    # ==================== 文件浏览 ====================

    def _browse_csv_s2(self):
        """浏览CSV文件"""
        initial = Path(PROJECT_DIR) / "data" / "output"
        file_path = filedialog.askopenfilename(title="选择CSV文件", initialdir=str(initial), filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.ctx.csv_var.set(file_path)

    # ==================== 批量操作 ====================

    def _batch_confirm(self):
        """批量确认所有记录"""
        csv_path = self.ctx.csv_var.get()
        if not Path(csv_path).exists():
            messagebox.showwarning("警告", "CSV文件不存在")
            return

        if not messagebox.askyesno("确认", "确定要将所有记录的review_status设置为'已确认'吗？"):
            return

        try:
            import csv
            # 读取CSV
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            # 确保字段存在
            if "review_status" not in fieldnames:
                fieldnames.append("review_status")

            # 批量设置
            count = 0
            for row in rows:
                if row.get("review_status") != "已确认":
                    row["review_status"] = "已确认"
                    count += 1

            # 保存CSV（原子化写入）
            from ..utils.atomic_csv import atomic_write_csv
            atomic_write_csv(csv_path, rows, fieldnames)

            print(f"[完成] 已确认 {count} 条记录")
            messagebox.showinfo("完成", f"已确认 {count} 条记录")

        except Exception as e:
            print(f"[错误] {e}")
            messagebox.showerror("错误", str(e))

    def _batch_clear_final_name(self):
        """批量清空final_name"""
        csv_path = self.ctx.csv_var.get()
        if not Path(csv_path).exists():
            messagebox.showwarning("警告", "CSV文件不存在")
            return

        if not messagebox.askyesno("确认", "确定要清空所有记录的final_name吗？"):
            return

        try:
            import csv
            # 读取CSV
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            # 批量清空
            count = 0
            for row in rows:
                if row.get("final_name"):
                    row["final_name"] = ""
                    count += 1

            # 保存CSV（原子化写入）
            from ..utils.atomic_csv import atomic_write_csv
            atomic_write_csv(csv_path, rows, fieldnames)

            print(f"[完成] 已清空 {count} 条记录的final_name")
            messagebox.showinfo("完成", f"已清空 {count} 条记录的final_name")

        except Exception as e:
            print(f"[错误] {e}")
            messagebox.showerror("错误", str(e))

    # ==================== 执行重命名 ====================

    def _run_rename(self):
        """运行重命名"""
        csv = self.ctx.csv_var.get()

        cmd = [PYTHON, "-m", "title_classifier", "rename", "-c", csv]

        if self.s2_dry_run_var.get():
            cmd.append("--dry-run")

        def on_rename_complete(returncode=None):
            # 同步重命名结果到数据库
            self._sync_csv_to_db(csv)

        self._run_command(cmd, callback=on_rename_complete)

    def _sync_csv_to_db(self, csv_path: str):
        """从 CSV 同步数据到数据库"""
        if not self.ctx.db:
            return
        try:
            import csv as csv_module
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    original_path = row.get("original_path", "")
                    if not original_path:
                        continue

                    # 尝试获取元数据
                    file_size = None
                    fs_str = row.get("file_size", "").strip()
                    if fs_str:
                        try:
                            file_size = int(fs_str)
                        except ValueError:
                            pass

                    duration = None
                    dur_str = row.get("duration", "").strip()
                    if dur_str:
                        try:
                            duration = float(dur_str)
                        except ValueError:
                            pass

                    resolution = row.get("resolution", "").strip()

                    existing = self.ctx.db.find_match(
                        original_title=row.get("original_title", ""),
                        file_size=file_size,
                        duration=duration,
                    )
                    if existing:
                        continue

                    data = {
                        "original_title": row.get("original_title", ""),
                        "original_path": original_path,
                        "current_path": original_path,
                        "needs_vision": row.get("needs_vision", "").lower() == "true",
                        "final_name": row.get("final_name", ""),
                        "review_status": row.get("review_status", "待确认"),
                        "audio_recognized": row.get("audio_recognized", "").lower() == "true",
                        "srt_path": row.get("srt_path", ""),
                        "vision_description": row.get("vision_description", ""),
                        "vision_keywords": row.get("vision_keywords", ""),
                        "human_detected": row.get("human_detected", "").lower() == "true",
                        "detection_method": row.get("detection_method", ""),
                        "file_size": file_size,
                        "duration": duration,
                        "resolution": resolution,
                    }
                    self.ctx.db.insert_media(data)
            print("[数据库] 扫描结果已同步")
        except Exception as e:
            print(f"[警告] 数据库同步失败: {e}")
