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
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        ttk.Button(btn_frame, text="AI优化", command=self._run_refine).pack(side=tk.LEFT, padx=4)

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

        self.s1c_use_yolo_var = tk.BooleanVar()
        ttk.Checkbutton(det_frame, text="使用YOLO", variable=self.s1c_use_yolo_var).pack(side=tk.LEFT, padx=4)

        self.s1c_yolo_model_var = tk.StringVar(value="detect")
        ttk.Combobox(
            det_frame,
            textvariable=self.s1c_yolo_model_var,
            values=["detect", "pose", "segment"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=4)

        self.s1c_use_clip_var = tk.BooleanVar()
        ttk.Checkbutton(det_frame, text="使用CLIP", variable=self.s1c_use_clip_var).pack(side=tk.LEFT, padx=4)

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
        self._run_command(cmd)

    def _run_vision(self):
        """运行视觉识别"""
        csv = self.s1c_csv_var.get()
        provider = self.s1c_provider_var.get()

        cmd = [PYTHON, "-m", "title_classifier", "vision", "-c", csv, "-p", provider]

        if self.s1c_use_yolo_var.get():
            cmd.append("--use-yolo")
            cmd.extend(["--yolo-model", self.s1c_yolo_model_var.get()])

        if self.s1c_use_clip_var.get():
            cmd.append("--use-clip")

        self._run_command(cmd)

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
