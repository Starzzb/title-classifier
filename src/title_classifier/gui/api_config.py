"""API 配置对话框 - 管理所有 Provider 的 URL、API Key 和模型"""

import os
import json
import threading
import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..providers import get_all_providers, DEFAULT_PROVIDERS

# 项目根目录
PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
ENV_FILE = PROJECT_DIR / ".env"
PROVIDERS_JSON = PROJECT_DIR / "config" / "providers.json"


def fetch_models_from_api(url: str, api_key: str = "", provider_type: str = "openai") -> List[str]:
    """从 API 获取可用模型列表"""
    import urllib.request

    try:
        # 构建模型列表 URL
        if provider_type == "ollama":
            # Ollama: /api/tags
            base_url = url.rstrip("/")
            if "/api/generate" in base_url:
                base_url = base_url.replace("/api/generate", "")
            models_url = f"{base_url}/api/tags"
        else:
            # OpenAI 兼容: /v1/models
            base_url = url.rstrip("/")
            if "/v1/chat/completions" in base_url:
                base_url = base_url.replace("/v1/chat/completions", "")
            elif "/chat/completions" in base_url:
                base_url = base_url.replace("/chat/completions", "")
            models_url = f"{base_url}/v1/models"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(models_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        # 解析模型列表
        if provider_type == "ollama":
            return [m.get("name", "") for m in data.get("models", [])]
        else:
            return [m.get("id", "") for m in data.get("data", [])]

    except Exception as e:
        raise Exception(f"获取模型失败: {e}")


class AddProviderDialog(tk.Toplevel):
    """添加自定义提供商对话框"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("添加自定义提供商")
        self.geometry("500x400")
        self.resizable(False, False)

        self.result = None

        # Make modal
        self.transient(parent)
        self.grab_set()

        self._build_ui()

        # 居中
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        """构建 UI"""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="添加自定义提供商",
            font=("Microsoft YaHei", 14, "bold")
        ).pack(pady=(0, 15))

        # 提供商 ID
        row1 = ttk.Frame(main_frame)
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="标识符:", width=12).pack(side=tk.LEFT)
        self.id_entry = ttk.Entry(row1, width=30)
        self.id_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Label(row1, text="英文，如 my-api", foreground="#888888").pack(side=tk.LEFT)

        # 名称
        row2 = ttk.Frame(main_frame)
        row2.pack(fill=tk.X, pady=5)
        ttk.Label(row2, text="显示名称:", width=12).pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(row2, width=30)
        self.name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # URL
        row3 = ttk.Frame(main_frame)
        row3.pack(fill=tk.X, pady=5)
        ttk.Label(row3, text="API URL:", width=12).pack(side=tk.LEFT)
        self.url_entry = ttk.Entry(row3, width=40)
        self.url_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Label(row3, text="OpenAI 兼容格式", foreground="#888888").pack(side=tk.LEFT)

        # Env Key
        row4 = ttk.Frame(main_frame)
        row4.pack(fill=tk.X, pady=5)
        ttk.Label(row4, text="环境变量名:", width=12).pack(side=tk.LEFT)
        self.env_key_entry = ttk.Entry(row4, width=30)
        self.env_key_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Label(row4, text="如 MY_API_KEY", foreground="#888888").pack(side=tk.LEFT)

        # 按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(20, 0))

        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="添加", command=self._add, bootstyle=SUCCESS).pack(side=tk.RIGHT)

    def _add(self):
        """添加提供商"""
        provider_id = self.id_entry.get().strip()
        name = self.name_entry.get().strip()
        url = self.url_entry.get().strip()
        env_key = self.env_key_entry.get().strip()

        if not provider_id or not name or not url:
            messagebox.showerror("错误", "标识符、名称和 URL 不能为空", parent=self)
            return

        # 验证 ID 格式
        if not provider_id.replace("-", "").replace("_", "").isalnum():
            messagebox.showerror("错误", "标识符只能包含英文、数字、- 和 _", parent=self)
            return

        self.result = {
            "id": provider_id,
            "name": name,
            "url": url,
            "env_key": env_key,
            "type": "multi",
            "requires_api_key": bool(env_key),
            "supports_1b": True,
            "supports_1c": True,
            "supports_audio": False,
            "description": "自定义提供商",
        }
        self.destroy()


class APIConfigDialog(tk.Toplevel):
    """API 配置对话框"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("API 配置")
        self.geometry("700x600")
        self.resizable(True, True)

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

        # 标题栏
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            header_frame,
            text="API 配置",
            font=("Microsoft YaHei", 14, "bold")
        ).pack(side=tk.LEFT)

        ttk.Button(
            header_frame,
            text="+ 添加自定义提供商",
            command=self._add_custom_provider,
            bootstyle=INFO
        ).pack(side=tk.RIGHT)

        # 滚动区域
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)

        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_resize(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                pass
        canvas.bind("<MouseWheel>", _on_mousewheel)
        self.scroll_frame.bind("<MouseWheel>", _on_mousewheel)

        # Provider 配置区域
        self.provider_widgets = {}
        for provider_id, provider_info in self.all_providers.items():
            self._build_provider_section(self.scroll_frame, provider_id, provider_info)

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
        # 判断是否为自定义提供商
        is_custom = provider_id not in DEFAULT_PROVIDERS

        # 框架
        label_text = provider_info.get("name", provider_id)
        if is_custom:
            label_text += " (自定义)"
        frame = ttk.LabelFrame(parent, text=label_text)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # 内部框架
        inner_frame = ttk.Frame(frame, padding=10)
        inner_frame.pack(fill=tk.X)

        # URL 输入
        url_frame = ttk.Frame(inner_frame)
        url_frame.pack(fill=tk.X, pady=2)
        ttk.Label(url_frame, text="URL:", width=10).pack(side=tk.LEFT)

        default_url = provider_info.get("url", "")
        custom_url = self.providers_data.get(provider_id, {}).get("url", default_url)

        url_entry = ttk.Entry(url_frame, width=50)
        url_entry.insert(0, custom_url)
        url_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # API Key 输入
        key_frame = ttk.Frame(inner_frame)
        key_frame.pack(fill=tk.X, pady=2)
        ttk.Label(key_frame, text="API Key:", width=10).pack(side=tk.LEFT)

        env_key = provider_info.get("env_key", "")
        current_key = self.env_data.get(env_key, "") if env_key else ""

        key_entry = ttk.Entry(key_frame, show="*", width=40)
        key_entry.insert(0, current_key)
        key_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        show_btn = ttk.Button(
            key_frame,
            text="显示",
            width=6,
            command=lambda e=key_entry: self._toggle_show(e)
        )
        show_btn.pack(side=tk.LEFT)

        # 模型选择
        model_frame = ttk.Frame(inner_frame)
        model_frame.pack(fill=tk.X, pady=2)
        ttk.Label(model_frame, text="模型:", width=10).pack(side=tk.LEFT)

        # 获取当前模型
        saved_model = self.providers_data.get(provider_id, {}).get("default_model", "")
        default_model = saved_model or provider_info.get("default_model", "")

        model_var = tk.StringVar(value=default_model)
        model_combo = ttk.Combobox(model_frame, textvariable=model_var, width=30)
        model_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # 如果有默认模型，加入下拉列表
        if default_model:
            model_combo["values"] = [default_model]

        # 获取模型按钮
        fetch_btn = ttk.Button(
            model_frame,
            text="获取模型",
            width=10,
            command=lambda pid=provider_id, pi=provider_info, mc=model_combo, ke=key_entry, ue=url_entry: 
                self._fetch_models(pid, pi, mc, ke, ue)
        )
        fetch_btn.pack(side=tk.LEFT)

        # 模型状态标签
        model_status = ttk.Label(model_frame, text="", foreground="#888888", width=20)
        model_status.pack(side=tk.LEFT, padx=5)

        # 删除按钮（仅自定义提供商）
        if is_custom:
            del_btn = ttk.Button(
                inner_frame,
                text="删除此提供商",
                bootstyle=DANGER,
                command=lambda pid=provider_id: self._delete_custom_provider(pid)
            )
            del_btn.pack(anchor=tk.E, pady=(5, 0))

        # 描述
        if provider_info.get("description"):
            ttk.Label(
                inner_frame,
                text=provider_info["description"],
                foreground="#888888",
                font=("Microsoft YaHei", 8)
            ).pack(anchor=tk.W, pady=(5, 0))

        # 保存引用
        self.provider_widgets[provider_id] = {
            "url_entry": url_entry,
            "key_entry": key_entry,
            "model_var": model_var,
            "model_combo": model_combo,
            "model_status": model_status,
            "env_key": env_key,
            "default_url": provider_info.get("url", ""),
            "is_custom": is_custom,
        }

    def _toggle_show(self, entry: ttk.Entry):
        """切换密码显示/隐藏"""
        if entry.cget("show") == "*":
            entry.configure(show="")
        else:
            entry.configure(show="*")

    def _fetch_models(self, provider_id: str, provider_info: Dict, model_combo: ttk.Combobox, 
                      key_entry: ttk.Entry, url_entry: ttk.Entry):
        """获取可用模型列表"""
        url = url_entry.get().strip()
        api_key = key_entry.get().strip()

        if not url:
            messagebox.showwarning("提示", "请先输入 URL", parent=self)
            return

        # 更新状态
        widgets = self.provider_widgets.get(provider_id, {})
        status_label = widgets.get("model_status")
        if status_label:
            status_label.configure(text="获取中...", foreground="#888888")

        # 在后台线程获取模型
        def fetch_thread():
            try:
                provider_type = "ollama" if provider_id == "ollama" else "openai"
                models = fetch_models_from_api(url, api_key, provider_type)
                
                # 在主线程更新 UI
                self.after(0, lambda: self._update_model_list(provider_id, model_combo, models, status_label))
            except Exception as e:
                self.after(0, lambda: self._fetch_error(provider_id, str(e), status_label))

        threading.Thread(target=fetch_thread, daemon=True).start()

    def _update_model_list(self, provider_id: str, model_combo: ttk.Combobox, 
                          models: List[str], status_label: ttk.Label):
        """更新模型列表"""
        if models:
            model_combo["values"] = models
            if status_label:
                status_label.configure(text=f"找到 {len(models)} 个模型", foreground="#28a745")
        else:
            if status_label:
                status_label.configure(text="未找到模型", foreground="#dc3545")

    def _fetch_error(self, provider_id: str, error: str, status_label: ttk.Label):
        """获取模型失败"""
        if status_label:
            status_label.configure(text="获取失败", foreground="#dc3545")
        messagebox.showerror("错误", f"获取模型失败:\n{error}", parent=self)

    def _add_custom_provider(self):
        """添加自定义提供商"""
        dialog = AddProviderDialog(self)
        self.wait_window(dialog)

        if dialog.result:
            provider_id = dialog.result["id"]

            # 检查是否已存在
            if provider_id in self.all_providers:
                messagebox.showerror("错误", f"提供商 '{provider_id}' 已存在", parent=self)
                return

            # 保存到 providers.json
            if provider_id not in self.providers_data:
                self.providers_data[provider_id] = {}

            self.providers_data[provider_id].update({
                "name": dialog.result["name"],
                "url": dialog.result["url"],
                "env_key": dialog.result["env_key"],
                "type": dialog.result["type"],
                "requires_api_key": dialog.result["requires_api_key"],
                "supports_1b": dialog.result["supports_1b"],
                "supports_1c": dialog.result["supports_1c"],
                "supports_audio": dialog.result["supports_audio"],
                "description": dialog.result["description"],
            })

            # 更新 all_providers
            self.all_providers[provider_id] = self.providers_data[provider_id]

            # 刷新 UI
            self._refresh_providers()

    def _delete_custom_provider(self, provider_id: str):
        """删除自定义提供商"""
        if not messagebox.askyesno("确认", f"确定要删除提供商 '{provider_id}' 吗？", parent=self):
            return

        # 从 providers_data 中删除
        if provider_id in self.providers_data:
            del self.providers_data[provider_id]

        # 从 all_providers 中删除
        if provider_id in self.all_providers:
            del self.all_providers[provider_id]

        # 刷新 UI
        self._refresh_providers()

    def _refresh_providers(self):
        """刷新提供商列表"""
        # 清空现有 widgets
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.provider_widgets.clear()

        # 重新构建
        for provider_id, provider_info in self.all_providers.items():
            self._build_provider_section(self.scroll_frame, provider_id, provider_info)

    def _save(self):
        """保存配置"""
        # 1. 保存所有配置到 providers.json
        providers_to_save = {}
        for provider_id, widgets in self.provider_widgets.items():
            provider_config = {}

            # URL
            url = widgets["url_entry"].get().strip()
            default_url = widgets["default_url"]
            if url and url != default_url:
                provider_config["url"] = url

            # 模型
            model = widgets["model_var"].get().strip()
            if model:
                provider_config["default_model"] = model

            # 如果是自定义提供商，保存完整配置
            if widgets["is_custom"] and provider_id in self.providers_data:
                provider_config.update(self.providers_data[provider_id])

            if provider_config:
                providers_to_save[provider_id] = provider_config

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

        for provider_id, widgets in self.provider_widgets.items():
            env_key = widgets["env_key"]
            if env_key and env_key not in updated_keys:
                new_key = widgets["key_entry"].get().strip()
                if new_key:
                    new_lines.append(f"{env_key}={new_key}\n")

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
        if not messagebox.askyesno("确认", "确定要恢复默认配置吗？\n这将清除所有自定义配置", parent=self):
            return

        # 删除 providers.json
        try:
            if PROVIDERS_JSON.exists():
                PROVIDERS_JSON.unlink()
        except Exception:
            pass

        # 清空 .env 文件
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

        # 重新加载配置
        self._load_current_config()
        self._refresh_providers()

        messagebox.showinfo("完成", "已恢复默认配置", parent=self)
