"""视频标题分类工具 - 图形界面"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import subprocess
import threading
import sys
import logging
from pathlib import Path
from datetime import datetime

from ..providers import (
    get_available_providers, get_provider_config, get_api_key,
    check_provider_availability, get_provider_display_name,
    get_providers_for_gui, call_text_api, test_provider_connection,
)
from ..core.refiner import Refiner
from ..utils.muxer import SubtitleMuxer
from ..utils.file_resolve import resolve_media_path
from .context import AppContext

PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
PYTHON = sys.executable
DEFAULT_CSV = "data/output/title_review.csv"
THEME_NAME = "solar"


class ToolTip:
    """鼠标悬停提示"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
            font=("Microsoft YaHei", 9),
        )
        label.pack()

    def _hide(self, event):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class LogRedirector:
    """将stdout/stderr重定向到GUI日志框"""

    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, message):
        if message.strip():
            self.text_widget.after(0, self._append, message)

    def _append(self, message):
        self.text_widget.insert(tk.END, message + "\n", self.tag)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


class GUILogHandler(logging.Handler):
    """将Python logging重定向到GUI日志框"""

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.after(0, self._append, msg)

    def _append(self, message):
        self.text_widget.insert(tk.END, message + "\n", "info")
        self.text_widget.see(tk.END)


class TitleClassifierApp(ttk.Window):
    """视频标题分类工具主窗口"""

    def __init__(self):
        super().__init__(title="视频标题分类工具 v8.1", themename=THEME_NAME)
        self.geometry("900x850")
        self.minsize(800, 700)

        self.process = None
        self.running = False

        self._load_env()

        # 初始化数据库
        from ..core.db_store import MediaDB
        from ..utils.config import load_merged_config
        db = MediaDB()
        db.init_schema()

        # 共享上下文
        self.ctx = AppContext(
            db=db,
            csv_var=tk.StringVar(value=DEFAULT_CSV),
            config=load_merged_config(),
            user_config_path=PROJECT_DIR / "config" / "user.toml",
        )

        self._build_ui()

    def _load_env(self):
        """加载.env文件"""
        env_path = PROJECT_DIR / ".env"
        if not env_path.exists():
            return
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, val = line.partition("=")
                        os.environ.setdefault(key.strip(), val.strip())
        except Exception:
            pass

    def _build_ui(self):
        """构建UI"""
        # 菜单栏
        self._build_menu_bar()

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # CSV状态栏
        csv_bar = ttk.Frame(main_frame)
        csv_bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(csv_bar, text="当前CSV:").pack(side=tk.LEFT)
        csv_display = ttk.Label(csv_bar, textvariable=self.ctx.csv_var, foreground="#6688cc")
        csv_display.pack(side=tk.LEFT, padx=(2, 8))
        hint_label = ttk.Label(
            csv_bar,
            text="各阶段可同时运行，确保CSV文件路径一致即可",
            foreground="#888888",
            font=("Microsoft YaHei", 8),
        )
        hint_label.pack(side=tk.LEFT)
        ToolTip(hint_label, (
            "并发说明：\n"
            "- 每个阶段启动时锁定CSV路径，运行中切换不影响已启动的任务\n"
            "- 不同标签页可以同时运行，各自读写不同列\n"
            "- 注意：运行中请勿在Stage1b右键切换 needs_vision/audio_recognized"
        ))

        # 可调大小的上下分栏：上=标签页，下=日志
        self.pane = ttk.Panedwindow(main_frame, orient=tk.VERTICAL)
        self.pane.pack(fill=tk.BOTH, expand=True)

        # 上半部分：标签页
        notebook_frame = ttk.Frame(self.pane)
        self.pane.add(notebook_frame, weight=3)

        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 导入并创建各 Tab
        from .stage_scan import StageScanTab
        from .stage_refine import StageRefineTab
        from .stage_audio import StageAudioTab
        from .stage_vision import StageVisionTab
        from .stage_rename import StageRenameTab

        self.stage_scan = StageScanTab(self.notebook, self.ctx, self._run_command, self._sync_csv_to_db)
        self.notebook.add(self.stage_scan, text="扫描入库")

        self.stage_refine = StageRefineTab(self.notebook, self.ctx, self._run_command, self._sync_csv_to_db)
        self.notebook.add(self.stage_refine, text="标题优化")

        self.stage_audio = StageAudioTab(self.notebook, self.ctx, self._run_command, self._gui_sync_to_db)
        self.notebook.add(self.stage_audio, text="音频转录")

        self.stage_vision = StageVisionTab(self.notebook, self.ctx, self._run_command, self._gui_sync_to_db, self._sync_csv_to_db)
        self.notebook.add(self.stage_vision, text="视觉识别")

        self.stage_rename = StageRenameTab(self.notebook, self.ctx, self._run_command, self._sync_csv_to_db)
        self.notebook.add(self.stage_rename, text="批量重命名")

        # 切换标签页时更新状态栏
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # 下半部分：日志区域（可折叠）
        self.log_frame = ttk.LabelFrame(self.pane, text="运行日志")
        self.pane.add(self.log_frame, weight=1)
        self._log_expanded = True

        log_toolbar = ttk.Frame(self.log_frame)
        log_toolbar.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(log_toolbar, text="折叠", width=5, command=self._toggle_log).pack(side=tk.LEFT, padx=2)
        ttk.Button(log_toolbar, text="清空日志", command=self._clear_log).pack(side=tk.RIGHT)
        self.stop_btn = ttk.Button(log_toolbar, text="停止", command=self._stop_process, state="disabled")
        self.stop_btn.pack(side=tk.RIGHT, padx=4)
        self.ctx.stop_btn = self.stop_btn
        self.progress_label = ttk.Label(log_toolbar, text="", foreground="#888888")
        self.progress_label.pack(side=tk.LEFT, padx=4)
        self.ctx.progress_label = self.progress_label

        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=8, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.log_text.tag_configure("stdout", foreground="#cccccc")
        self.log_text.tag_configure("stderr", foreground="#ff6666")
        self.log_text.tag_configure("info", foreground="#66ccff")
        self.ctx.log_text = self.log_text

        # 重定向stdout/stderr
        sys.stdout = LogRedirector(self.log_text, "stdout")
        sys.stderr = LogRedirector(self.log_text, "stderr")

        # 重定向logging
        gui_handler = GUILogHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logging.getLogger().addHandler(gui_handler)

    def _toggle_log(self):
        """折叠/展开日志区"""
        if self._log_expanded:
            self.pane.forget(self.log_frame)
            self._log_expanded = False
        else:
            self.pane.add(self.log_frame, weight=1)
            self._log_expanded = True

    def _build_menu_bar(self):
        """构建菜单栏"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)

        # 主题子菜单
        theme_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="主题", menu=theme_menu)
        for theme_name in ["solar", "cosmo", "darkly", "flatly", "superhero", "cyborg"]:
            theme_menu.add_command(
                label=f"{'[当前] ' if theme_name == THEME_NAME else ''}{theme_name}",
                command=lambda t=theme_name: self._switch_theme(t),
            )

        view_menu.add_separator()
        view_menu.add_command(label="调试查看器...", command=self._open_debug_browser)

        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="设置...", command=self._open_settings)

    def _switch_theme(self, theme_name: str):
        """切换主题"""
        try:
            self.style.theme_use(theme_name)
            print(f"[信息] 主题已切换: {theme_name}")
        except Exception as e:
            print(f"[错误] 主题切换失败: {e}")

    def _open_debug_browser(self):
        """打开调试浏览器"""
        from tkinter import filedialog
        debug_dir = filedialog.askdirectory(
            title="选择调试目录",
            initialdir=str(PROJECT_DIR / "data" / "debug"),
        )
        if debug_dir:
            from .debug_window import open_debug_window
            open_debug_window(self, debug_dir)

    def _open_settings(self):
        """打开设置对话框"""
        try:
            from .settings import SettingsDialog
            SettingsDialog(self, self.ctx)
        except ImportError:
            print("[信息] 设置对话框尚未实现")

    def _on_tab_changed(self, event=None):
        """切换标签页时更新状态栏"""
        try:
            tab_text = self.notebook.tab(self.notebook.select(), "text")
            self.progress_label.configure(text=f"当前: {tab_text}")
        except Exception:
            pass

    def _run_command(self, cmd, callback=None):
        """运行子进程（线程安全）"""
        if self.running:
            print("[警告] 已有任务运行中")
            return

        self.running = True
        self.stop_btn.configure(state="normal")
        self.progress_label.configure(text="运行中...")

        def run_in_thread():
            try:
                self.process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    bufsize=1, encoding="utf-8", errors="replace",
                )
                for line in self.process.stdout:
                    print(line.rstrip())
                self.process.wait()
                returncode = self.process.returncode
            except Exception as e:
                print(f"[错误] {e}")
                returncode = -1
            finally:
                self.process = None
                self.running = False
                self.stop_btn.configure(state="disabled")
                self.progress_label.configure(text="完成" if returncode == 0 else "出错")
                if callback:
                    callback(returncode)

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

    def _stop_process(self):
        """停止当前运行的进程"""
        if self.process:
            try:
                self.process.terminate()
                print("[信息] 已发送停止信号")
            except Exception as e:
                print(f"[错误] 停止失败: {e}")

    def _clear_log(self):
        """清空日志"""
        self.log_text.delete("1.0", tk.END)

    def _sync_csv_to_db(self, csv_path: str):
        """同步CSV记录到数据库"""
        if not self.ctx.db:
            return
        try:
            from ..utils.atomic_csv import safe_read_csv
            rows, _ = safe_read_csv(csv_path)
            for row in rows:
                original_path = row.get("original_path", "").strip()
                if not original_path:
                    continue
                media = self.ctx.db.find_by_path(original_path)
                if not media:
                    data = {
                        "original_path": original_path,
                        "original_title": row.get("original_title", ""),
                        "final_name": row.get("final_name", ""),
                        "needs_vision": 1 if row.get("needs_vision", "").lower() == "true" else 0,
                    }
                    self.ctx.db.insert_media(data)
        except Exception as e:
            print(f"[警告] CSV同步到数据库失败: {e}")

    def _gui_sync_to_db(self, original_path: str, updates: dict, source: str = "gui"):
        """GUI 操作同步到数据库"""
        if not self.ctx.db:
            return
        try:
            media = self.ctx.db.find_by_path(original_path)
            if not media:
                data = {"original_path": original_path, "original_title": Path(original_path).name}
                data.update(updates)
                self.ctx.db.insert_media(data)
            else:
                for field, value in updates.items():
                    self.ctx.db.update_media(media["id"], field, value, source)
        except Exception as e:
            print(f"[警告] 数据库同步失败: {e}")


def main():
    """启动GUI"""
    app = TitleClassifierApp()
    app.mainloop()


if __name__ == "__main__":
    main()
