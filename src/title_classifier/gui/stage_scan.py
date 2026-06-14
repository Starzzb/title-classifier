"""扫描入库 Tab"""

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from pathlib import Path

from .context import AppContext
from .app import ToolTip

PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
PYTHON = __import__('sys').executable
DEFAULT_CSV = "data/output/title_review.csv"


class StageScanTab(ttk.Frame):
    """Stage1 扫描入库"""

    def __init__(self, parent, ctx: AppContext, run_command_callback=None):
        super().__init__(parent)
        self.ctx = ctx
        self._run_command = run_command_callback
        self._build_ui()

    def _build_ui(self):
        """构建Stage1扫描标签页"""
        tab = self

        # 目录/文件选择
        dir_frame = ttk.LabelFrame(tab, text="扫描目标")
        dir_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1_dir_var = tk.StringVar()
        dir_entry = ttk.Entry(dir_frame, textvariable=self.s1_dir_var, width=60)
        dir_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(dir_frame, text="浏览目录...", command=self._browse_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(dir_frame, text="选择文件...", command=self._browse_file).pack(side=tk.LEFT, padx=2)
        ToolTip(dir_entry, "选择要扫描的视频/图片目录或单个文件\n\n"
                "- 选择目录：递归扫描所有子目录\n"
                "- 选择文件：只处理选中的单个文件")

        # 输出文件
        out_frame = ttk.LabelFrame(tab, text="输出文件")
        out_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1_output_var = tk.StringVar(value=DEFAULT_CSV)
        out_entry = ttk.Entry(out_frame, textvariable=self.s1_output_var, width=60)
        out_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(out_entry, "扫描结果保存的CSV文件路径")

        # 选项
        opt_frame = ttk.LabelFrame(tab, text="选项")
        opt_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1_append_var = tk.BooleanVar()
        append_cb = ttk.Checkbutton(opt_frame, text="追加模式", variable=self.s1_append_var)
        append_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(append_cb, "勾选后新扫描结果追加到现有CSV文件，否则覆盖")

        self.s1_force_var = tk.BooleanVar()
        force_cb = ttk.Checkbutton(opt_frame, text="强制重新分类", variable=self.s1_force_var)
        force_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(force_cb, "勾选后即使文件已有分类标签也会重新处理")

        # 执行按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        scan_btn = ttk.Button(btn_frame, text="开始扫描", command=self._run_scan)
        scan_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(scan_btn, "扫描目录中的媒体文件，提取关键词，生成待审CSV表")

    def _browse_dir(self):
        """浏览目录"""
        dir_path = filedialog.askdirectory(title="选择扫描目录")
        if dir_path:
            self.s1_dir_var.set(dir_path)
            # 自动建议 per-directory 的 CSV 输出路径
            dir_name = Path(dir_path).resolve().name
            suggested = Path(PROJECT_DIR) / "data" / "output" / dir_name / "title_review.csv"
            self.s1_output_var.set(str(suggested))

    def _browse_file(self):
        """浏览文件"""
        filetypes = [
            ("媒体文件", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.webm *.m4v *.ts *.jpg *.jpeg *.png *.bmp *.webp"),
            ("视频文件", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.webm *.m4v *.ts"),
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("所有文件", "*.*"),
        ]
        file_path = filedialog.askopenfilename(title="选择媒体文件", filetypes=filetypes)
        if file_path:
            self.s1_dir_var.set(file_path)

    def _run_scan(self):
        """运行扫描"""
        dir_path = self.s1_dir_var.get()
        if not dir_path:
            messagebox.showwarning("警告", "请选择扫描目录")
            return

        # 自动计算 per-directory 的输出路径
        target = Path(dir_path).resolve()
        if target.is_dir():
            dir_name = target.name
            output = str(Path(PROJECT_DIR) / "data" / "output" / dir_name / "title_review.csv")
            self.s1_output_var.set(output)
        else:
            output = self.s1_output_var.get()

        cmd = [PYTHON, "-m", "title_classifier", "scan", "-d", dir_path, "-o", output]

        if self.s1_append_var.get():
            cmd.append("-a")
        if self.s1_force_var.get():
            cmd.append("--force")

        # 扫描完成后同步 CSV 路径到所有标签页 + 同步到数据库
        def on_scan_complete():
            self.ctx.csv_var.set(output)
            self._sync_csv_to_db(output)

        self._run_command(cmd, callback=on_scan_complete)

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
