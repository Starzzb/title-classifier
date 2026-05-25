"""视频标题分类工具 - 图形界面"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import sys
from pathlib import Path
from datetime import datetime

from ..providers import (
    get_available_providers, get_provider_config, get_api_key,
    check_provider_availability, get_provider_display_name,
    get_providers_for_gui, call_text_api, test_provider_connection,
)

PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
PYTHON = sys.executable
DEFAULT_CSV = "data/output/title_review.csv"


class ToolTip:
    """鼠标悬停提示工具类"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 25

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Microsoft YaHei", 9),
            wraplength=350,
        )
        label.pack(ipadx=4, ipady=2)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class LogRedirector:
    """将print输出重定向到GUI文本框"""

    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, msg):
        if msg.strip():
            self.text_widget.after(0, self._append, msg)

    def _append(self, msg):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, msg + "\n", self.tag)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass


class TitleClassifierApp(tk.Tk):
    """视频标题分类工具主窗口"""

    def __init__(self):
        super().__init__()
        self.title("视频标题分类工具 v6.0")
        self.geometry("900x850")
        self.minsize(800, 700)

        self.process = None
        self.running = False

        # 共享CSV路径变量
        self.csv_var = tk.StringVar(value=DEFAULT_CSV)

        self._build_ui()
        self._sync_csv()

    def _build_ui(self):
        """构建UI"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_stage1_tab()
        self._build_stage1b_tab()
        self._build_stage1c_tab()
        self._build_stage2_tab()

        # 切换标签页时同步CSV
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="运行日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(log_toolbar, text="清空日志", command=self._clear_log).pack(side=tk.RIGHT)
        self.stop_btn = ttk.Button(log_toolbar, text="停止", command=self._stop_process, state="disabled")
        self.stop_btn.pack(side=tk.RIGHT, padx=4)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=12, state="disabled", font=("Consolas", 9), wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.log_text.tag_configure("stdout", foreground="#cccccc")
        self.log_text.tag_configure("stderr", foreground="#ff6666")
        self.log_text.tag_configure("info", foreground="#66ccff")

        sys.stdout = LogRedirector(self.log_text, "stdout")
        sys.stderr = LogRedirector(self.log_text, "stderr")

    def _sync_csv(self):
        """同步所有标签页的CSV路径"""
        csv = self.csv_var.get()
        self.s1_output_var.set(csv)
        self.s1b_csv_var.set(csv)
        self.s1c_csv_var.set(csv)
        self.s2_csv_var.set(csv)

    def _on_tab_changed(self, event=None):
        """切换标签页时同步"""
        tab = self.notebook.select()
        tab_name = self.notebook.tab(tab, "text")
        if "Stage1 " in tab_name:
            current = self.s1_output_var.get()
        elif "Stage1b" in tab_name:
            current = self.s1b_csv_var.get()
        elif "Stage1c" in tab_name:
            current = self.s1c_csv_var.get()
        elif "Stage2" in tab_name:
            current = self.s2_csv_var.get()
        else:
            return
        self.csv_var.set(current)
        self._sync_csv()

    def _build_stage1_tab(self):
        """构建Stage1扫描标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Stage1 扫描")

        # 目录选择
        dir_frame = ttk.LabelFrame(tab, text="扫描目录")
        dir_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1_dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.s1_dir_var, width=60).pack(side=tk.LEFT, padx=4)
        ttk.Button(dir_frame, text="浏览...", command=self._browse_dir).pack(side=tk.LEFT, padx=4)

        # 输出文件
        out_frame = ttk.LabelFrame(tab, text="输出文件")
        out_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1_output_var = tk.StringVar(value=DEFAULT_CSV)
        ttk.Entry(out_frame, textvariable=self.s1_output_var, width=60).pack(side=tk.LEFT, padx=4)

        # 选项
        opt_frame = ttk.LabelFrame(tab, text="选项")
        opt_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1_append_var = tk.BooleanVar()
        ttk.Checkbutton(opt_frame, text="追加模式", variable=self.s1_append_var).pack(side=tk.LEFT, padx=4)

        self.s1_force_var = tk.BooleanVar()
        ttk.Checkbutton(opt_frame, text="强制重新分类", variable=self.s1_force_var).pack(side=tk.LEFT, padx=4)

        # 执行按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        ttk.Button(btn_frame, text="开始扫描", command=self._run_scan).pack(side=tk.LEFT, padx=4)

    def _build_stage1b_tab(self):
        """构建Stage1b AI优化标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Stage1b AI优化")

        # CSV文件
        csv_frame = ttk.LabelFrame(tab, text="CSV文件")
        csv_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1b_csv_var = tk.StringVar(value=DEFAULT_CSV)
        ttk.Entry(csv_frame, textvariable=self.s1b_csv_var, width=60).pack(side=tk.LEFT, padx=4)
        ttk.Button(csv_frame, text="浏览...", command=self._browse_csv_s1b).pack(side=tk.LEFT, padx=4)

        # Provider选择
        provider_frame = ttk.LabelFrame(tab, text="AI Provider")
        provider_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1b_provider_var = tk.StringVar(value="gcli")
        providers = get_providers_for_gui("1b")
        ttk.Combobox(provider_frame, textvariable=self.s1b_provider_var, values=providers, state="readonly").pack(
            side=tk.LEFT, padx=4
        )

        # 执行按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(btn_frame, text="AI优化", command=self._run_refine).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="加载CSV", command=self._load_s1b_preview).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="确认写入", command=self._confirm_s1b_results).pack(side=tk.LEFT, padx=4)

        # 预览表格
        preview_frame = ttk.LabelFrame(tab, text="优化结果预览")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 创建树形视图
        columns = ("original", "proposed", "refined")
        self.s1b_tree = ttk.Treeview(preview_frame, columns=columns, show="headings", selectmode="browse")
        self.s1b_tree.heading("original", text="原始标题")
        self.s1b_tree.heading("proposed", text="建议标题")
        self.s1b_tree.heading("refined", text="AI优化结果")
        self.s1b_tree.column("original", width=200)
        self.s1b_tree.column("proposed", width=150)
        self.s1b_tree.column("refined", width=250)

        # 滚动条
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.s1b_tree.yview)
        self.s1b_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.s1b_tree.pack(fill=tk.BOTH, expand=True)

        # 右键菜单
        self.s1b_context_menu = tk.Menu(self, tearoff=0)
        self.s1b_context_menu.add_command(label="编辑", command=self._s1b_edit)
        self.s1b_context_menu.add_command(label="采用原标题", command=self._s1b_use_original)
        self.s1b_context_menu.add_command(label="采用建议标题", command=self._s1b_use_proposed)
        self.s1b_context_menu.add_separator()
        self.s1b_context_menu.add_command(label="删除", command=self._s1b_delete)

        self.s1b_tree.bind("<Button-3>", self._s1b_show_context_menu)
        self.s1b_tree.bind("<Double-1>", self._s1b_edit)

        # 存储优化结果
        self.s1b_results = {}

    def _build_stage1c_tab(self):
        """构建Stage1c视觉识别标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Stage1c 视觉识别")

        # CSV文件
        csv_frame = ttk.LabelFrame(tab, text="CSV文件")
        csv_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1c_csv_var = tk.StringVar(value=DEFAULT_CSV)
        ttk.Entry(csv_frame, textvariable=self.s1c_csv_var, width=60).pack(side=tk.LEFT, padx=4)
        ttk.Button(csv_frame, text="浏览...", command=self._browse_csv_s1c).pack(side=tk.LEFT, padx=4)

        # Provider选择
        provider_frame = ttk.LabelFrame(tab, text="AI Provider")
        provider_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1c_provider_var = tk.StringVar(value="gcli")
        providers = get_providers_for_gui("1c")
        ttk.Combobox(provider_frame, textvariable=self.s1c_provider_var, values=providers, state="readonly").pack(
            side=tk.LEFT, padx=4
        )

        # 检测器选择
        det_frame = ttk.LabelFrame(tab, text="检测器")
        det_frame.pack(fill=tk.X, padx=4, pady=4)

        # 检测器类型选择（UHD或YOLO）
        self.s1c_detector_var = tk.StringVar(value="yolo")
        ttk.Radiobutton(det_frame, text="YOLO", variable=self.s1c_detector_var, value="yolo").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(det_frame, text="UHD", variable=self.s1c_detector_var, value="uhd").pack(side=tk.LEFT, padx=4)

        # YOLO模型选择（多选）
        ttk.Label(det_frame, text="YOLO模型:").pack(side=tk.LEFT, padx=4)
        self.s1c_yolo_detect_var = tk.BooleanVar(value=True)
        self.s1c_yolo_pose_var = tk.BooleanVar(value=True)
        self.s1c_yolo_segment_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(det_frame, text="detect", variable=self.s1c_yolo_detect_var).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(det_frame, text="pose", variable=self.s1c_yolo_pose_var).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(det_frame, text="segment", variable=self.s1c_yolo_segment_var).pack(side=tk.LEFT, padx=2)

        # CLIP选项
        self.s1c_use_clip_var = tk.BooleanVar()
        ttk.Checkbutton(det_frame, text="使用CLIP", variable=self.s1c_use_clip_var).pack(side=tk.LEFT, padx=4)

        # 分析参数
        param_frame = ttk.LabelFrame(tab, text="分析参数")
        param_frame.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(param_frame, text="采样间隔(秒):").pack(side=tk.LEFT, padx=4)
        self.s1c_analysis_step_var = tk.StringVar(value="2.0")
        ttk.Entry(param_frame, textvariable=self.s1c_analysis_step_var, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Label(param_frame, text="VLM帧数:").pack(side=tk.LEFT, padx=4)
        self.s1c_vlm_frames_var = tk.StringVar(value="10")
        ttk.Entry(param_frame, textvariable=self.s1c_vlm_frames_var, width=6).pack(side=tk.LEFT, padx=4)

        # 选项
        opt_frame = ttk.LabelFrame(tab, text="选项")
        opt_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1c_all_var = tk.BooleanVar()
        ttk.Checkbutton(opt_frame, text="处理所有未识别文件", variable=self.s1c_all_var).pack(side=tk.LEFT, padx=4)

        self.s1c_audio_var = tk.BooleanVar()
        ttk.Checkbutton(opt_frame, text="生成音频字幕", variable=self.s1c_audio_var).pack(side=tk.LEFT, padx=4)

        # 执行按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        ttk.Button(btn_frame, text="视觉识别", command=self._run_vision).pack(side=tk.LEFT, padx=4)

    def _build_stage2_tab(self):
        """构建Stage2重命名标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Stage2 重命名")

        # CSV文件
        csv_frame = ttk.LabelFrame(tab, text="CSV文件")
        csv_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s2_csv_var = tk.StringVar(value=DEFAULT_CSV)
        ttk.Entry(csv_frame, textvariable=self.s2_csv_var, width=60).pack(side=tk.LEFT, padx=4)
        ttk.Button(csv_frame, text="浏览...", command=self._browse_csv_s2).pack(side=tk.LEFT, padx=4)

        # 批量操作
        batch_frame = ttk.LabelFrame(tab, text="批量操作")
        batch_frame.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(batch_frame, text="一键确认所有记录", command=self._batch_confirm).pack(side=tk.LEFT, padx=4)
        ttk.Button(batch_frame, text="一键清空final_name", command=self._batch_clear_final_name).pack(side=tk.LEFT, padx=4)

        # 选项
        opt_frame = ttk.LabelFrame(tab, text="选项")
        opt_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s2_dry_run_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="模拟运行", variable=self.s2_dry_run_var).pack(side=tk.LEFT, padx=4)

        # 执行按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        ttk.Button(btn_frame, text="执行重命名", command=self._run_rename).pack(side=tk.LEFT, padx=4)

    # ==================== 文件浏览 ====================

    def _browse_dir(self):
        """浏览目录"""
        dir_path = filedialog.askdirectory(title="选择扫描目录")
        if dir_path:
            self.s1_dir_var.set(dir_path)

    def _browse_csv_s1b(self):
        """浏览CSV文件"""
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.s1b_csv_var.set(file_path)

    def _browse_csv_s1c(self):
        """浏览CSV文件"""
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.s1c_csv_var.set(file_path)

    def _browse_csv_s2(self):
        """浏览CSV文件"""
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.s2_csv_var.set(file_path)

    # ==================== 命令执行 ====================

    def _run_command(self, cmd, callback=None):
        """运行命令"""
        if self.running:
            messagebox.showwarning("警告", "有命令正在运行")
            return

        self.running = True
        self.stop_btn.configure(state="normal")

        def run():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                for line in self.process.stdout:
                    print(line.rstrip())
                self.process.wait()
                if self.process.returncode == 0:
                    print("\n[完成] 命令执行成功")
                else:
                    print(f"\n[错误] 命令执行失败，返回码: {self.process.returncode}")
            except Exception as e:
                print(f"\n[错误] {e}")
            finally:
                self.running = False
                self.stop_btn.configure(state="disabled")
                if callback:
                    callback()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _stop_process(self):
        """停止进程"""
        if self.process:
            self.process.terminate()
            self.process = None
            self.running = False
            self.stop_btn.configure(state="disabled")
            print("[停止] 命令已终止")

    def _clear_log(self):
        """清空日志"""
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")

    def _run_scan(self):
        """运行扫描"""
        dir_path = self.s1_dir_var.get()
        if not dir_path:
            messagebox.showwarning("警告", "请选择扫描目录")
            return

        output = self.s1_output_var.get()
        cmd = [PYTHON, "-m", "title_classifier", "scan", "-d", dir_path, "-o", output]

        if self.s1_append_var.get():
            cmd.append("-a")
        if self.s1_force_var.get():
            cmd.append("--force")

        self._run_command(cmd)

    def _run_refine(self):
        """运行AI优化"""
        csv = self.s1b_csv_var.get()
        provider = self.s1b_provider_var.get()

        cmd = [PYTHON, "-m", "title_classifier", "refine", "-c", csv, "-p", provider]
        self._run_command(cmd, callback=self._load_s1b_preview)

    def _load_s1b_preview(self):
        """加载CSV到预览表格"""
        csv_path = self.s1b_csv_var.get()
        if not Path(csv_path).exists():
            return

        try:
            import csv
            # 清空表格
            for item in self.s1b_tree.get_children():
                self.s1b_tree.delete(item)
            self.s1b_results.clear()

            # 读取CSV
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    original = row.get("original_title", "")
                    proposed = row.get("proposed_title", "")
                    final = row.get("final_name", "")
                    
                    # 插入到表格
                    item_id = self.s1b_tree.insert("", tk.END, values=(original, proposed, final))
                    self.s1b_results[item_id] = {
                        "row_index": i,
                        "original_title": original,
                        "proposed_title": proposed,
                        "final_name": final,
                    }

            print(f"[完成] 加载 {len(self.s1b_results)} 条记录")

        except Exception as e:
            print(f"[错误] 加载CSV失败: {e}")

    def _s1b_show_context_menu(self, event):
        """显示右键菜单"""
        item = self.s1b_tree.identify_row(event.y)
        if item:
            self.s1b_tree.selection_set(item)
            self.s1b_context_menu.post(event.x_root, event.y_root)

    def _s1b_edit(self, event=None):
        """编辑选中项"""
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        values = self.s1b_tree.item(item, "values")
        
        # 弹出编辑对话框
        dialog = tk.Toplevel(self)
        dialog.title("编辑标题")
        dialog.geometry("500x150")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="原始标题:").grid(row=0, column=0, padx=4, pady=4, sticky=tk.W)
        ttk.Label(dialog, text=values[0][:50]).grid(row=0, column=1, padx=4, pady=4, sticky=tk.W)

        ttk.Label(dialog, text="优化结果:").grid(row=1, column=0, padx=4, pady=4, sticky=tk.W)
        edit_var = tk.StringVar(value=values[2])
        ttk.Entry(dialog, textvariable=edit_var, width=50).grid(row=1, column=1, padx=4, pady=4)

        def confirm():
            new_value = edit_var.get()
            self.s1b_tree.item(item, values=(values[0], values[1], new_value))
            if item in self.s1b_results:
                self.s1b_results[item]["final_name"] = new_value
            dialog.destroy()

        ttk.Button(dialog, text="确认", command=confirm).grid(row=2, column=0, columnspan=2, pady=8)

    def _s1b_use_original(self):
        """采用原标题"""
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        values = self.s1b_tree.item(item, "values")
        original = values[0]
        
        self.s1b_tree.item(item, values=(original, values[1], original))
        if item in self.s1b_results:
            self.s1b_results[item]["final_name"] = original

    def _s1b_use_proposed(self):
        """采用建议标题"""
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        values = self.s1b_tree.item(item, "values")
        proposed = values[1]
        
        self.s1b_tree.item(item, values=(values[0], proposed, proposed))
        if item in self.s1b_results:
            self.s1b_results[item]["final_name"] = proposed

    def _s1b_delete(self):
        """删除选中项"""
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        self.s1b_tree.delete(item)
        if item in self.s1b_results:
            del self.s1b_results[item]

    def _confirm_s1b_results(self):
        """确认写入优化结果到CSV"""
        csv_path = self.s1b_csv_var.get()
        if not Path(csv_path).exists():
            messagebox.showwarning("警告", "CSV文件不存在")
            return

        if not self.s1b_results:
            messagebox.showwarning("警告", "没有优化结果可写入")
            return

        if not messagebox.askyesno("确认", "确定要将优化结果写入CSV吗？"):
            return

        try:
            import csv
            # 读取CSV
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            # 更新final_name
            updated = 0
            for item_id, result in self.s1b_results.items():
                row_index = result["row_index"]
                if row_index < len(rows):
                    rows[row_index]["final_name"] = result["final_name"]
                    updated += 1

            # 保存CSV
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"[完成] 已更新 {updated} 条记录")
            messagebox.showinfo("完成", f"已更新 {updated} 条记录")

        except Exception as e:
            print(f"[错误] 写入CSV失败: {e}")
            messagebox.showerror("错误", str(e))

    def _run_vision(self):
        """运行视觉识别"""
        csv = self.s1c_csv_var.get()
        provider = self.s1c_provider_var.get()

        cmd = [PYTHON, "-m", "title_classifier", "vision", "-c", csv, "-p", provider]

        # 检测器选择
        detector = self.s1c_detector_var.get()
        if detector == "yolo":
            cmd.append("--use-yolo")
            # 获取选中的YOLO模型
            models = []
            if self.s1c_yolo_detect_var.get():
                models.append("detect")
            if self.s1c_yolo_pose_var.get():
                models.append("pose")
            if self.s1c_yolo_segment_var.get():
                models.append("segment")
            if models:
                cmd.extend(["--yolo-model", ",".join(models)])
        # UHD不需要额外参数，默认使用

        if self.s1c_use_clip_var.get():
            cmd.append("--use-clip")

        # 添加分析参数
        analysis_step = self.s1c_analysis_step_var.get()
        if analysis_step:
            cmd.extend(["--analysis-step", analysis_step])

        vlm_frames = self.s1c_vlm_frames_var.get()
        if vlm_frames:
            cmd.extend(["--vlm-frames", vlm_frames])

        if self.s1c_all_var.get():
            cmd.append("--all")

        if self.s1c_audio_var.get():
            cmd.append("--audio")

        self._run_command(cmd)

    def _batch_confirm(self):
        """批量确认所有记录"""
        csv_path = self.s2_csv_var.get()
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

            # 保存CSV
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"[完成] 已确认 {count} 条记录")
            messagebox.showinfo("完成", f"已确认 {count} 条记录")

        except Exception as e:
            print(f"[错误] {e}")
            messagebox.showerror("错误", str(e))

    def _batch_clear_final_name(self):
        """批量清空final_name"""
        csv_path = self.s2_csv_var.get()
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

            # 保存CSV
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"[完成] 已清空 {count} 条记录的final_name")
            messagebox.showinfo("完成", f"已清空 {count} 条记录的final_name")

        except Exception as e:
            print(f"[错误] {e}")
            messagebox.showerror("错误", str(e))

    def _run_rename(self):
        """运行重命名"""
        csv = self.s2_csv_var.get()

        cmd = [PYTHON, "-m", "title_classifier", "rename", "-c", csv]

        if self.s2_dry_run_var.get():
            cmd.append("--dry-run")

        self._run_command(cmd)


def main():
    """启动GUI"""
    app = TitleClassifierApp()
    app.mainloop()


if __name__ == "__main__":
    main()
