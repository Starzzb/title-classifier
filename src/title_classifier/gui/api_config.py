"""API 配置对话框 - 管理所有 Provider 的 URL 和 API Key"""

import os
import json
import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from pathlib import Path
from typing import Dict, Any, Optional

from ..providers import get_all_providers, DEFAULT_PROVIDERS

# 项目根目录
PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
ENV_FILE = PROJECT_DIR / ".env"
PROVIDERS_JSON = PROJECT_DIR / "config" / "providers.json"


class APIConfigDialog(tk.Toplevel):
    """API 配置对话框"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("API 配置")
        self.geometry("600x500")
        self.resizable(False, False)

        # 设置窗口图标
        try:
            icon_path = Path(__file__).parent / "assets" / "icon.ico"
            if icon_path.exists():
                self.iconbitmap(str(icon_path))
        except Exception:
            pass

        # Make modal
        self.transient(parent)
        self.grab_set()

        # 加载当前配置
        self._load_current_config()

        # 构建 UI
        self._build_ui()

        # 居中到父窗口
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _load_current_config(self):
        """加载当前配置"""
        # 加载 .env 文件
        self.env_data = {}
        if ENV_FILE.exists():
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, val = line.partition("=")
                        self.env_data[key.strip()] = val.strip()

        # 加载 providers.json
        self.providers_data = {}
        if PROVIDERS_JSON.exists():
            with open(PROVIDERS_JSON, "r", encoding="utf-8") as f:
                self.providers_data = json.load(f)

        # 获取所有 provider 配置
        self.all_providers = get_all_providers()

    def _build_ui(self):
        """构建 UI"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        ttk.Label(
            main_frame,
            text="API 配置",
            font=("Microsoft YaHei", 14, "bold")
        ).pack(pady=(0, 5))

        ttk.Label(
            main_frame,
            text="配置各 AI 服务的 URL 和 API Key",
            foreground="#888888"
        ).pack(pady=(0, 10))

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

        # 鼠标滚轮
        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                pass
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scroll_frame.bind("<MouseWheel>", _on_mousewheel)

        # Provider 配置区域
        self.provider_widgets = {}
        for provider_id, provider_info in self.all_providers.items():
            self._build_provider_section(scroll_frame, provider_id, provider_info)

        # 底部按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="恢复默认",
            command=self._restore_defaults
        ).pack(side=tk.LEFT)

        ttk.Button(
            btn_frame,
            text="取消",
            command=self.destroy
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            btn_frame,
            text="保存",
            command=self._save,
            bootstyle=SUCCESS
        ).pack(side=tk.RIGHT)

    def _build_provider_section(self, parent, provider_id: str, provider_info: Dict[str, Any]):
        """构建单个 provider 的配置区域"""
        # 框架
        frame = ttk.LabelFrame(parent, text=provider_info.get("name", provider_id), padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # URL 输入
        url_frame = ttk.Frame(frame)
        url_frame.pack(fill=tk.X, pady=2)
        ttk.Label(url_frame, text="URL:", width=10).pack(side=tk.LEFT)

        # 获取当前 URL（优先使用 providers.json 中的自定义值，否则使用默认值）
        default_url = provider_info.get("url", "")
        custom_url = self.providers_data.get(provider_id, {}).get("url", default_url)

        url_entry = ttk.Entry(url_frame, width=50)
        url_entry.insert(0, custom_url)
        url_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # API Key 输入
        key_frame = ttk.Frame(frame)
        key_frame.pack(fill=tk.X, pady=2)
        ttk.Label(key_frame, text="API Key:", width=10).pack(side=tk.LEFT)

        # 获取当前 API Key
        env_key = provider_info.get("env_key", "")
        current_key = self.env_data.get(env_key, "") if env_key else ""

        key_entry = ttk.Entry(key_frame, show="*", width=40)
        key_entry.insert(0, current_key)
        key_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # 显示/隐藏按钮
        show_btn = ttk.Button(
            key_frame,
            text="显示",
            width=6,
            command=lambda e=key_entry: self._toggle_show(e)
        )
        show_btn.pack(side=tk.LEFT)

        # 模型显示（只读）
        model_frame = ttk.Frame(frame)
        model_frame.pack(fill=tk.X, pady=2)
        ttk.Label(model_frame, text="模型:", width=10).pack(side=tk.LEFT)
        ttk.Label(
            model_frame,
            text=provider_info.get("default_model", ""),
            foreground="#888888"
        ).pack(side=tk.LEFT, padx=5)

        # 描述
        if provider_info.get("description"):
            ttk.Label(
                frame,
                text=provider_info["description"],
                foreground="#888888",
                font=("Microsoft YaHei", 8)
            ).pack(anchor=tk.W, pady=(5, 0))

        # 保存引用
        self.provider_widgets[provider_id] = {
            "url_entry": url_entry,
            "key_entry": key_entry,
            "env_key": env_key,
            "default_url": provider_info.get("url", ""),
        }

    def _toggle_show(self, entry: ttk.Entry):
        """切换密码显示/隐藏"""
        if entry.cget("show") == "*":
            entry.configure(show="")
        else:
            entry.configure(show="*")

    def _save(self):
        """保存配置"""
        # 1. 保存 URL 到 providers.json
        providers_to_save = {}
        for provider_id, widgets in self.provider_widgets.items():
            url = widgets["url_entry"].get().strip()
            default_url = widgets["default_url"]

            # 只保存与默认值不同的 URL
            if url and url != default_url:
                if provider_id not in providers_to_save:
                    providers_to_save[provider_id] = {}
                providers_to_save[provider_id]["url"] = url

        # 保存 providers.json
        try:
            PROVIDERS_JSON.parent.mkdir(parents=True, exist_ok=True)
            with open(PROVIDERS_JSON, "w", encoding="utf-8") as f:
                json.dump(providers_to_save, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("错误", f"保存 providers.json 失败: {e}", parent=self)
            return

        # 2. 保存 API Key 到 .env
        env_lines = []
        if ENV_FILE.exists():
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                env_lines = f.readlines()

        # 更新或添加 API Key
        updated_keys = set()
        new_lines = []
        for line in env_lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                new_lines.append(line)
                continue

            if "=" in line_stripped:
                key, _, _ = line_stripped.partition("=")
                key = key.strip()

                # 查找对应的 provider
                found = False
                for provider_id, widgets in self.provider_widgets.items():
                    if widgets["env_key"] == key:
                        new_key = widgets["key_entry"].get().strip()
                        if new_key:
                            new_lines.append(f"{key}={new_key}\n")
                        else:
                            new_lines.append(f"{key}=\n")
                        updated_keys.add(key)
                        found = True
                        break

                if not found:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # 添加新的 API Key（如果 .env 中不存在）
        for provider_id, widgets in self.provider_widgets.items():
            env_key = widgets["env_key"]
            if env_key and env_key not in updated_keys:
                new_key = widgets["key_entry"].get().strip()
                if new_key:
                    new_lines.append(f"{env_key}={new_key}\n")

        # 写入 .env 文件
        try:
            with open(ENV_FILE, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except Exception as e:
            messagebox.showerror("错误", f"保存 .env 失败: {e}", parent=self)
            return

        # 3. 更新环境变量
        for provider_id, widgets in self.provider_widgets.items():
            env_key = widgets["env_key"]
            if env_key:
                new_key = widgets["key_entry"].get().strip()
                if new_key:
                    os.environ[env_key] = new_key
                elif env_key in os.environ:
                    del os.environ[env_key]

        messagebox.showinfo("成功", "API 配置已保存", parent=self)
        self.destroy()

    def _restore_defaults(self):
        """恢复默认配置"""
        if not messagebox.askyesno("确认", "确定要恢复默认配置吗？\n这将清除所有自定义 URL 和 API Key", parent=self):
            return

        # 清空所有输入框
        for provider_id, widgets in self.provider_widgets.items():
            widgets["url_entry"].delete(0, tk.END)
            widgets["url_entry"].insert(0, widgets["default_url"])
            widgets["key_entry"].delete(0, tk.END)

        # 删除 providers.json 中的自定义配置
        try:
            if PROVIDERS_JSON.exists():
                PROVIDERS_JSON.unlink()
        except Exception:
            pass

        # 清空 .env 文件中的 API Key
        try:
            if ENV_FILE.exists():
                ENV_FILE.unlink()
        except Exception:
            pass

        # 清空环境变量
        for provider_id, widgets in self.provider_widgets.items():
            env_key = widgets["env_key"]
            if env_key and env_key in os.environ:
                del os.environ[env_key]

        messagebox.showinfo("完成", "已恢复默认配置", parent=self)
