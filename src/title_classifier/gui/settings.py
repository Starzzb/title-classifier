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
        self.geometry("520x680")
        self.minsize(450, 500)
        self.transient(parent)
        self.grab_set()

        # 设置窗口图标
        try:
            icon_path = Path(__file__).parent / "assets" / "icon.ico"
            if icon_path.exists():
                self.iconbitmap(str(icon_path))
        except Exception:
            pass

        self.ctx = ctx
        self.config = load_merged_config()

        # 变量
        self._init_vars()
        self._build_ui()
        self._load_values()

        # 窗口关闭时清理
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # 居中到父窗口
        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, px)}+{max(0, py)}")

    def _init_vars(self):
        """初始化变量"""
        # 外观
        self.theme_var = tk.StringVar()
        self.font_size_var = tk.StringVar()

        # 分阶段 Provider + Model 配置
        self.stage_refine_provider_var = tk.StringVar()
        self.stage_refine_model_var = tk.StringVar()
        self.stage_vision_provider_var = tk.StringVar()
        self.stage_vision_model_var = tk.StringVar()
        self.stage_audio_provider_var = tk.StringVar()
        self.stage_audio_model_var = tk.StringVar()
        self.timeout_var = tk.StringVar()

        # 推理
        self.device_var = tk.StringVar()
        self.backend_var = tk.StringVar()
        self.debug_dir_var = tk.StringVar()

        # 全局参数
        self.max_image_size_var = tk.StringVar()
        self.vlm_timeout_var = tk.StringVar()
        self.vlm_retry_var = tk.StringVar()

        # 标题优化
        self.refiner_batch_size_var = tk.StringVar()
        self.refiner_max_workers_var = tk.StringVar()

    def _build_ui(self):
        """构建UI"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 滚动区域
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 让 scroll_frame 宽度跟随 canvas
        def _on_canvas_resize(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮（只在鼠标悬停在canvas上时生效）
        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                pass
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scroll_frame.bind("<MouseWheel>", _on_mousewheel)

        # 外观
        self._build_appearance_section(scroll_frame)

        # API 配置
        self._build_api_section(scroll_frame)

        # 推理配置
        self._build_inference_section(scroll_frame)

        # 全局参数
        self._build_global_section(scroll_frame)

        # 标题优化配置
        self._build_refiner_section(scroll_frame)

        # 底部按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="恢复默认", command=self._reset_defaults).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="保存", command=self._save, bootstyle=PRIMARY).pack(side=tk.RIGHT, padx=4)

    def _build_appearance_section(self, parent):
        """外观区块"""
        frame = ttk.LabelFrame(parent, text="外观")
        frame.pack(fill=tk.X, pady=(0, 10), padx=10)

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
        """API 配置区块 — 每阶段独立 Provider + Model"""
        frame = ttk.LabelFrame(parent, text="模型配置")
        frame.pack(fill=tk.X, pady=(0, 10), padx=10)

        stages = [
            ("标题优化 (Refine)", "refine", self.stage_refine_provider_var, self.stage_refine_model_var),
            ("视觉识别 (Vision)", "vision", self.stage_vision_provider_var, self.stage_vision_model_var),
            ("音频识别 (Audio)", "audio", self.stage_audio_provider_var, self.stage_audio_model_var),
        ]

        self._stage_model_combos = {}

        for label, stage_key, provider_var, model_var in stages:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=4)

            ttk.Label(row, text=f"{label}:", width=16).pack(side=tk.LEFT)

            # Provider dropdown
            try:
                from ..providers import get_providers_for_gui
                providers = get_providers_for_gui(stage_key)
            except Exception:
                providers = ["gcli", "mimo"]
            provider_combo = ttk.Combobox(row, textvariable=provider_var, values=providers, state="readonly", width=12)
            provider_combo.pack(side=tk.LEFT, padx=4)

            # Model dropdown
            model_combo = ttk.Combobox(row, textvariable=model_var, width=28)
            model_combo.pack(side=tk.LEFT, padx=4)
            self._stage_model_combos[stage_key] = model_combo

            # Fetch button
            ttk.Button(row, text="🔄", width=3,
                       command=lambda s=stage_key, pv=provider_var: self._fetch_models_for_stage(s, pv)).pack(side=tk.LEFT, padx=2)

            # Update model list when provider changes
            provider_combo.bind("<<ComboboxSelected>>",
                                lambda e, s=stage_key, pv=provider_var: self._on_provider_changed(s, pv))

        # 超时
        row_timeout = ttk.Frame(frame)
        row_timeout.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(row_timeout, text="API 超时(秒):", width=16).pack(side=tk.LEFT)
        ttk.Entry(row_timeout, textvariable=self.timeout_var, width=6).pack(side=tk.LEFT, padx=4)

        # 高级配置按钮
        row_adv = ttk.Frame(frame)
        row_adv.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(row_adv, text="高级 API 配置...", command=self._open_api_config, bootstyle=INFO).pack(side=tk.LEFT)
        ttk.Label(row_adv, text="配置各 Provider 的 URL、API Key", foreground="#888888", font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=10)

    def _open_api_config(self):
        """打开 API 配置对话框"""
        from .api_config import APIConfigDialog
        APIConfigDialog(self)

    def _on_provider_changed(self, stage_key, provider_var):
        """切换 Provider 时清空 Model"""
        self._stage_model_combos[stage_key].set("")

    def _fetch_models_for_stage(self, stage_key, provider_var):
        """获取指定阶段 Provider 的模型列表"""
        import threading
        provider = provider_var.get()
        if not provider:
            messagebox.showwarning("提示", "请先选择 Provider", parent=self)
            return

        combo = self._stage_model_combos[stage_key]

        def do_fetch():
            try:
                from ..providers import get_provider_config, get_api_key
                import json, ssl, http.client
                from urllib.parse import urlparse

                config = get_provider_config(provider)
                api_key = get_api_key(provider)
                api_url = config.get("url", "")
                if not api_url:
                    raise Exception("URL 为空")

                parsed = urlparse(api_url)
                models_url = f"{parsed.scheme}://{parsed.hostname}/v1/models"

                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

                conn = http.client.HTTPSConnection(parsed.hostname, context=ctx, timeout=15)
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = "Bearer " + api_key

                conn.request("GET", "/v1/models", headers=headers)
                resp = conn.getresponse()
                data = resp.read().decode()
                conn.close()

                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")

                result = json.loads(data)
                model_list = [m.get("id", "") for m in result.get("data", [])]

                def update_ui():
                    if model_list:
                        combo["values"] = model_list
                    else:
                        messagebox.showinfo("提示", "无可用模型", parent=self)
                self.after(0, update_ui)

            except Exception as e:
                def show_error():
                    messagebox.showerror("错误", f"获取模型失败: {e}", parent=self)
                self.after(0, show_error)

        threading.Thread(target=do_fetch, daemon=True).start()

    def _build_inference_section(self, parent):
        """推理配置区块"""
        frame = ttk.LabelFrame(parent, text="推理配置")
        frame.pack(fill=tk.X, pady=(0, 10), padx=10)

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
        frame = ttk.LabelFrame(parent, text="全局参数")
        frame.pack(fill=tk.X, pady=(0, 10), padx=10)

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

    def _build_refiner_section(self, parent):
        """标题优化配置区块"""
        frame = ttk.LabelFrame(parent, text="标题优化")
        frame.pack(fill=tk.X, pady=(0, 10), padx=10)

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="批量大小:", width=14).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.refiner_batch_size_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(row1, text="条/批（建议5-15，越小越准确）", foreground="#888888").pack(side=tk.LEFT)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="并发批次数:", width=14).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.refiner_max_workers_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(row2, text="（建议1-5）", foreground="#888888").pack(side=tk.LEFT)

    def _load_values(self):
        """从配置加载当前值"""
        c = self.config

        # 外观
        self.theme_var.set(get_config_value(c, "theme.name", "solar"))
        self.font_size_var.set(str(get_config_value(c, "theme.font_size", "10")))

        # 分阶段 Provider + Model
        self.stage_refine_provider_var.set(get_config_value(c, "providers.stage_providers.refine", "gcli"))
        self.stage_refine_model_var.set(get_config_value(c, "providers.models.refine", ""))
        self.stage_vision_provider_var.set(get_config_value(c, "providers.stage_providers.vision", "gcli"))
        self.stage_vision_model_var.set(get_config_value(c, "providers.models.vision", ""))
        self.stage_audio_provider_var.set(get_config_value(c, "providers.stage_providers.audio", "mimo"))
        self.stage_audio_model_var.set(get_config_value(c, "providers.models.audio", ""))
        self.timeout_var.set(str(get_config_value(c, "providers.timeout", 90)))

        # 推理
        self.device_var.set(get_config_value(c, "general.device", "auto"))
        self.backend_var.set(get_config_value(c, "yolo.backend", "auto"))
        self.debug_dir_var.set(get_config_value(c, "vision.debug_dir", "data/debug"))

        # 全局参数
        self.max_image_size_var.set(str(get_config_value(c, "vision.max_image_size", 1200)))
        self.vlm_timeout_var.set(str(get_config_value(c, "providers.timeout", 120)))
        self.vlm_retry_var.set(str(get_config_value(c, "vision.vlm_retry_count", 1)))

        # 标题优化
        self.refiner_batch_size_var.set(str(get_config_value(c, "refiner.batch_size", 10)))
        self.refiner_max_workers_var.set(str(get_config_value(c, "refiner.max_workers", 3)))

    def _save(self):
        """保存到 user.toml"""
        updates = {
            "theme": {
                "name": self.theme_var.get(),
                "font_size": int(self.font_size_var.get() or 10),
            },
            "providers": {
                "timeout": int(self.timeout_var.get() or 90),
                "stage_providers": {
                    "refine": self.stage_refine_provider_var.get(),
                    "vision": self.stage_vision_provider_var.get(),
                    "audio": self.stage_audio_provider_var.get(),
                },
                "models": {
                    "refine": self.stage_refine_model_var.get(),
                    "vision": self.stage_vision_model_var.get(),
                    "audio": self.stage_audio_model_var.get(),
                },
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
            "refiner": {
                "batch_size": int(self.refiner_batch_size_var.get() or 10),
                "max_workers": int(self.refiner_max_workers_var.get() or 3),
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

    def _on_close(self):
        """窗口关闭时清理"""
        # 解绑全局事件
        try:
            self.unbind_all("<MouseWheel>")
        except Exception:
            pass
        self.destroy()

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

    def _browse_debug_dir(self):
        """浏览调试目录"""
        d = filedialog.askdirectory(initialdir=self.debug_dir_var.get(), parent=self)
        if d:
            self.debug_dir_var.set(d)
