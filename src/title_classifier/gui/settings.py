"""设置对话框 - 外观 + API 配置 + 推理配置 + 全局参数"""

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from pathlib import Path

from .context import AppContext
from ..utils.config import load_merged_config, save_user_config, get_config_value


class SettingsDialog(ttk.Toplevel):
    """设置对话框"""

    def __init__(self, parent, ctx: AppContext):
        super().__init__(parent)
        self.title("设置")
        self.geometry("500x600")
        self.minsize(450, 500)
        self.transient(parent)
        self.grab_set()

        self.ctx = ctx
        self.config = load_merged_config()

        # 变量
        self._init_vars()
        self._build_ui()
        self._load_values()

    def _init_vars(self):
        """初始化变量"""
        # 外观
        self.theme_var = tk.StringVar()
        self.font_size_var = tk.StringVar()

        # API
        self.provider_var = tk.StringVar()
        self.api_key_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.timeout_var = tk.StringVar()

        # 推理
        self.device_var = tk.StringVar()
        self.backend_var = tk.StringVar()
        self.debug_dir_var = tk.StringVar()

        # 全局参数
        self.max_image_size_var = tk.StringVar()
        self.vlm_timeout_var = tk.StringVar()
        self.vlm_retry_var = tk.StringVar()

    def _build_ui(self):
        """构建UI"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 滚动区域
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 外观
        self._build_appearance_section(scroll_frame)

        # API 配置
        self._build_api_section(scroll_frame)

        # 推理配置
        self._build_inference_section(scroll_frame)

        # 全局参数
        self._build_global_section(scroll_frame)

        # 底部按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="恢复默认", command=self._reset_defaults).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="保存", command=self._save, bootstyle=PRIMARY).pack(side=tk.RIGHT, padx=4)

    def _build_appearance_section(self, parent):
        """外观区块"""
        frame = ttk.LabelFrame(parent, text="外观", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="主题:", width=12).pack(side=tk.LEFT)
        themes = ["solar", "cosmo", "darkly", "flatly", "superhero", "cyborg"]
        ttk.Combobox(row1, textvariable=self.theme_var, values=themes, state="readonly", width=20).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="字体大小:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.font_size_var, width=6).pack(side=tk.LEFT, padx=4)

    def _build_api_section(self, parent):
        """API 配置区块"""
        frame = ttk.LabelFrame(parent, text="API 配置", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Provider
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Provider:", width=12).pack(side=tk.LEFT)
        try:
            from ..providers import get_providers_for_gui
            providers = get_providers_for_gui()
        except Exception:
            providers = ["gcli", "mimo", "zhipu", "ollama"]
        ttk.Combobox(row1, textvariable=self.provider_var, values=providers, state="readonly", width=20).pack(side=tk.LEFT, padx=4)

        # API Key
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="API Key:", width=12).pack(side=tk.LEFT)
        self.api_key_entry = ttk.Entry(row2, textvariable=self.api_key_var, show="*", width=30)
        self.api_key_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="显示", width=5, command=self._toggle_api_key_visibility).pack(side=tk.LEFT)

        # 模型
        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="模型:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.model_var, width=30).pack(side=tk.LEFT, padx=4)

        # 超时
        row4 = ttk.Frame(frame)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text="超时(秒):", width=12).pack(side=tk.LEFT)
        ttk.Entry(row4, textvariable=self.timeout_var, width=6).pack(side=tk.LEFT, padx=4)

    def _build_inference_section(self, parent):
        """推理配置区块"""
        frame = ttk.LabelFrame(parent, text="推理配置", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="推理设备:", width=12).pack(side=tk.LEFT)
        ttk.Combobox(row1, textvariable=self.device_var, values=["auto", "cuda", "cpu"], state="readonly", width=10).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="YOLO后端:", width=12).pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.backend_var, values=["auto", "openvino", "pytorch"], state="readonly", width=10).pack(side=tk.LEFT, padx=4)

        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="调试目录:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.debug_dir_var, width=30).pack(side=tk.LEFT, padx=4)
        ttk.Button(row3, text="浏览...", command=self._browse_debug_dir).pack(side=tk.LEFT)

    def _build_global_section(self, parent):
        """全局参数区块"""
        frame = ttk.LabelFrame(parent, text="全局参数", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="最大图片尺寸:", width=14).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.max_image_size_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(row1, text="像素", foreground="#888888").pack(side=tk.LEFT)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="VLM超时(秒):", width=14).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.vlm_timeout_var, width=6).pack(side=tk.LEFT, padx=4)

        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="VLM重试次数:", width=14).pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.vlm_retry_var, width=6).pack(side=tk.LEFT, padx=4)

    def _load_values(self):
        """从配置加载当前值"""
        c = self.config

        # 外观
        self.theme_var.set(get_config_value(c, "theme.name", "solar"))
        self.font_size_var.set(str(get_config_value(c, "theme.font_size", "10")))

        # API
        self.provider_var.set(get_config_value(c, "providers.default", "gcli"))
        self.api_key_var.set(get_config_value(c, "api.key", ""))
        self.model_var.set(get_config_value(c, "providers.default_model", ""))
        self.timeout_var.set(str(get_config_value(c, "providers.timeout", 90)))

        # 推理
        self.device_var.set(get_config_value(c, "general.device", "auto"))
        self.backend_var.set(get_config_value(c, "yolo.backend", "auto"))
        self.debug_dir_var.set(get_config_value(c, "vision.debug_dir", "data/debug"))

        # 全局参数
        self.max_image_size_var.set(str(get_config_value(c, "vision.max_image_size", 1200)))
        self.vlm_timeout_var.set(str(get_config_value(c, "providers.timeout", 120)))
        self.vlm_retry_var.set(str(get_config_value(c, "vision.vlm_retry_count", 1)))

    def _save(self):
        """保存到 user.toml"""
        updates = {
            "theme": {
                "name": self.theme_var.get(),
                "font_size": int(self.font_size_var.get() or 10),
            },
            "providers": {
                "default": self.provider_var.get(),
                "timeout": int(self.timeout_var.get() or 90),
            },
            "api": {
                "key": self.api_key_var.get(),
            },
            "general": {
                "device": self.device_var.get(),
            },
            "yolo": {
                "backend": self.backend_var.get(),
            },
            "vision": {
                "debug_dir": self.debug_dir_var.get(),
                "max_image_size": int(self.max_image_size_var.get() or 1200),
                "vlm_retry_count": int(self.vlm_retry_var.get() or 1),
            },
        }

        if save_user_config(updates):
            # 更新上下文中的配置
            self.ctx.config = load_merged_config()

            # 尝试切换主题
            try:
                self.winfo_toplevel().style.theme_use(self.theme_var.get())
            except Exception:
                pass

            messagebox.showinfo("保存成功", "设置已保存到 config/user.toml\n部分设置需要重启才能生效", parent=self)
            self.destroy()
        else:
            messagebox.showerror("保存失败", "无法保存设置，请检查文件权限", parent=self)

    def _reset_defaults(self):
        """恢复默认值"""
        if messagebox.askyesno("确认", "确定要恢复默认设置吗？\n这将删除 user.toml 中的所有自定义配置", parent=self):
            try:
                user_path = Path(__file__).parent.parent.parent.parent / "config" / "user.toml"
                if user_path.exists():
                    user_path.unlink()
                self.config = load_merged_config()
                self._load_values()
                messagebox.showinfo("完成", "已恢复默认设置", parent=self)
            except Exception as e:
                messagebox.showerror("错误", f"恢复失败: {e}", parent=self)

    def _toggle_api_key_visibility(self):
        """切换 API Key 显示/隐藏"""
        if self.api_key_entry.cget("show") == "*":
            self.api_key_entry.configure(show="")
        else:
            self.api_key_entry.configure(show="*")

    def _browse_debug_dir(self):
        """浏览调试目录"""
        d = filedialog.askdirectory(initialdir=self.debug_dir_var.get(), parent=self)
        if d:
            self.debug_dir_var.set(d)
