"""模型管理对话框 - 卡片式布局，显示、切换和下载 YOLO/CLIP 模型"""

import json
import threading
import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from pathlib import Path

from .context import AppContext
from ..core.model_registry import (
    MODEL_REGISTRY, MODELS_DIR, YOLO_DIR, CLIP_DIR,
    get_model_from_config, get_model_info,
)
from ..core.model_downloader import (
    get_file_size_mb, get_dir_size_mb, get_downloaded_size,
    get_model_status_yolo, get_model_status_clip,
    download_yolo_model, download_clip_model,
)
from ..utils.config import load_merged_config, save_user_config


class ModelManagerDialog(tk.Toplevel):
    """模型管理对话框"""

    def __init__(self, parent, ctx: AppContext):
        super().__init__(parent)
        self.title("模型管理")
        self.geometry("720x650")
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

        self.ctx = ctx

        # 三步走
        self._init_vars()
        self._build_ui()
        self._load_values()

        # 居中到父窗口
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    # ── 初始化 ──────────────────────────────────────────────────────────────

    def _init_vars(self):
        """从 config 加载当前模型选择"""
        self._migrate_models_json()
        config = load_merged_config()

        self.model_vars = {}
        for key in MODEL_REGISTRY:
            default = MODEL_REGISTRY[key]["default"]
            if key == "clip":
                val = config.get("models", {}).get("clip", default)
            else:
                val = config.get("models", {}).get(key, default)
            self.model_vars[key] = tk.StringVar(value=val)

    def _migrate_models_json(self):
        """一次性迁移：从 models.json 迁移到 user.toml"""
        models_json = Path(__file__).parent.parent.parent.parent / "config" / "models.json"
        if not models_json.exists():
            return

        config = load_merged_config()
        if "models" in config:
            return

        try:
            with open(models_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            mapping = {
                "detect": "yolo_detect",
                "pose": "yolo_pose",
                "segment": "yolo_segment",
                "clip": "clip",
            }
            # 旧 CLIP 名称映射到新的 HF 仓库全名
            clip_name_map = {
                "CLIP-ViT-B-16": "CLIP-ViT-B-16-laion2B-s34B-b88K",
                "CLIP-ViT-B-32": "CLIP-ViT-B-32-laion2B-s34B-b79K",
                "CLIP-ViT-L-14": "CLIP-ViT-L-14-laion2B-s32B-b82K",
            }
            models_cfg = {}
            for old_key, new_key in mapping.items():
                if old_key in data:
                    val = data[old_key]
                    if old_key == "clip" and val in clip_name_map:
                        val = clip_name_map[val]
                    models_cfg[new_key] = val

            if models_cfg:
                save_user_config({"models": models_cfg})
                models_json.unlink()
                print(f"[迁移] models.json 已迁移到 user.toml [models] 段")
        except Exception as e:
            print(f"[警告] models.json 迁移失败: {e}")

    # ── UI 构建 ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        """构建 UI"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        ttk.Label(
            main_frame,
            text="模型管理",
            font=("Microsoft YaHei", 14, "bold")
        ).pack(pady=(0, 10))

        # 滚动区域
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

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

        # 统计信息（先创建，卡片构建时会引用）
        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        self.stats_label = ttk.Label(stats_frame, text="", foreground="#888888")
        self.stats_label.pack(side=tk.LEFT)

        # 模型卡片
        self.card_widgets = {}
        for registry_key, info in MODEL_REGISTRY.items():
            self._build_card(scroll_frame, registry_key, info)

        # 底部按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame, text="打开模型目录",
            command=self._open_models_dir
        ).pack(side=tk.LEFT)

        ttk.Button(
            btn_frame, text="保存配置",
            command=self._save_config, bootstyle=SUCCESS
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            btn_frame, text="关闭",
            command=self.destroy
        ).pack(side=tk.RIGHT)

    def _build_card(self, parent, registry_key: str, info: dict):
        """构建单个模型卡片"""
        frame = ttk.LabelFrame(parent, text=info["name"])
        frame.pack(fill=tk.X, padx=5, pady=5)

        inner = ttk.Frame(frame, padding=10)
        inner.pack(fill=tk.X)

        # 描述 + 流水线角色
        ttk.Label(
            inner, text=info["description"],
            foreground="#666666", wraplength=600
        ).pack(anchor=tk.W, pady=(0, 2))

        if info.get("pipeline_role"):
            ttk.Label(
                inner, text=f"流水线作用: {info['pipeline_role']}",
                foreground="#999999", font=("Microsoft YaHei", 8),
                wraplength=600
            ).pack(anchor=tk.W, pady=(0, 8))

        # 当前状态
        status_frame = ttk.Frame(inner)
        status_frame.pack(fill=tk.X, pady=2)

        if info["category"] == "clip":
            status = get_model_status_clip(self.model_vars[registry_key].get())
        else:
            status = get_model_status_yolo(self.model_vars[registry_key].get())

        status_text = "已下载" if status["exists"] else "未下载"
        status_color = "#28a745" if status["exists"] else "#dc3545"
        size_text = f"{status['size_mb']:.1f} MB" if status["exists"] else ""

        ttk.Label(status_frame, text="当前:", width=10).pack(side=tk.LEFT)
        status_label = ttk.Label(
            status_frame,
            text=f"{self.model_vars[registry_key].get()}  {size_text}  {status_text}",
            foreground=status_color
        )
        status_label.pack(side=tk.LEFT)

        # 选择行
        select_frame = ttk.Frame(inner)
        select_frame.pack(fill=tk.X, pady=5)
        ttk.Label(select_frame, text="切换:", width=10).pack(side=tk.LEFT)

        options = list(info["options"].keys())
        model_var = self.model_vars[registry_key]
        combo = ttk.Combobox(
            select_frame, textvariable=model_var,
            values=options, state="readonly", width=20
        )
        combo.pack(side=tk.LEFT, padx=5)

        desc_label = ttk.Label(select_frame, text="", foreground="#888888")
        desc_label.pack(side=tk.LEFT, padx=5)

        # 下载行
        dl_frame = ttk.Frame(inner)
        dl_frame.pack(fill=tk.X, pady=5)

        progress_label = ttk.Label(dl_frame, text="", foreground="#888888")
        progress_label.pack(side=tk.LEFT, padx=5)

        download_btn = ttk.Button(
            dl_frame, text="下载选中模型",
            command=lambda rk=registry_key, mv=model_var, pl=progress_label:
                self._download_model(rk, mv.get(), pl)
        )
        download_btn.pack(side=tk.RIGHT)

        # CLIP 提示
        if info["category"] == "clip":
            ttk.Label(
                inner,
                text="注意: CLIP 模型较大，保存配置后需重启程序生效",
                foreground="#ffc107", font=("Microsoft YaHei", 8)
            ).pack(anchor=tk.W, pady=(5, 0))

        # 更新描述回调
        def on_change(event=None):
            selected = model_var.get()
            opt = info["options"].get(selected, {})
            desc = opt.get("desc", "")
            size = opt.get("size", 0)
            desc_label.configure(text=f"{desc}  ({size} MB)")

            # 更新状态
            if info["category"] == "clip":
                st = get_model_status_clip(selected)
            else:
                st = get_model_status_yolo(selected)

            st_text = "已下载" if st["exists"] else "未下载"
            st_color = "#28a745" if st["exists"] else "#dc3545"
            st_size = f"{st['size_mb']:.1f} MB" if st["exists"] else ""
            status_label.configure(
                text=f"{selected}  {st_size}  {st_text}",
                foreground=st_color
            )
            self._update_stats()

        combo.bind("<<ComboboxSelected>>", on_change)

        self.card_widgets[registry_key] = {
            "combo": combo,
            "desc_label": desc_label,
            "status_label": status_label,
            "progress_label": progress_label,
            "download_btn": download_btn,
            "on_change": on_change,
        }

    # ── 加载值 ──────────────────────────────────────────────────────────────

    def _load_values(self):
        """填充控件初始值"""
        for registry_key, widgets in self.card_widgets.items():
            widgets["on_change"]()
        self._update_stats()

    # ── 更新统计 ────────────────────────────────────────────────────────────

    def _update_stats(self):
        """更新模型统计信息"""
        total_size = get_downloaded_size()
        self.stats_label.configure(
            text=f"模型目录: {MODELS_DIR}  |  已下载模型总大小: {total_size:.1f} MB"
        )

    # ── 下载 ────────────────────────────────────────────────────────────────

    def _download_model(self, registry_key: str, model_name: str, progress_label: ttk.Label):
        """下载模型（通用）"""
        info = MODEL_REGISTRY[registry_key]
        opt = info["options"].get(model_name, {})
        size = opt.get("size", 0)

        # 检查已存在
        if info["category"] == "clip":
            status = get_model_status_clip(model_name)
        else:
            status = get_model_status_yolo(model_name)

        if status["exists"]:
            messagebox.showinfo("提示", f"模型 {model_name} 已存在", parent=self)
            return

        if not messagebox.askyesno(
            "确认下载",
            f"确定要下载 {model_name} ({size} MB) 吗？",
            parent=self
        ):
            return

        progress_label.configure(text="下载中...")
        widgets = self.card_widgets[registry_key]
        widgets["download_btn"].configure(state="disabled")

        def do_download():
            try:
                if info["category"] == "clip":
                    def clip_progress(status, cur, total):
                        self.after(0, lambda: progress_label.configure(
                            text=f"下载中: {status}"
                        ))
                    success = download_clip_model(model_name, clip_progress)
                else:
                    def yolo_progress(count, block_size, total_size):
                        if total_size > 0:
                            pct = min(100, count * block_size * 100 / total_size)
                            self.after(0, lambda: progress_label.configure(
                                text=f"下载中... {pct:.0f}%"
                            ))
                    success = download_yolo_model(model_name, yolo_progress)

                if success:
                    self.after(0, lambda: progress_label.configure(
                        text="下载完成", foreground="#28a745"
                    ))
                    self.after(0, self._update_stats)
                    # 刷新状态
                    self.after(0, widgets["on_change"])
                else:
                    self.after(0, lambda: progress_label.configure(
                        text="下载失败", foreground="#dc3545"
                    ))
            except Exception as e:
                self.after(0, lambda: progress_label.configure(
                    text=f"下载失败: {e}", foreground="#dc3545"
                ))
            finally:
                self.after(0, lambda: widgets["download_btn"].configure(state="normal"))

        threading.Thread(target=do_download, daemon=True).start()

    # ── 保存 ────────────────────────────────────────────────────────────────

    def _save_config(self):
        """保存模型配置到 user.toml"""
        models_cfg = {}
        for key, var in self.model_vars.items():
            models_cfg[key] = var.get()

        if save_user_config({"models": models_cfg}):
            # 更新 ctx.config
            self.ctx.config = load_merged_config()
            messagebox.showinfo(
                "保存成功",
                "模型配置已保存到 config/user.toml\n重启程序后生效",
                parent=self
            )
        else:
            messagebox.showerror("保存失败", "无法保存配置", parent=self)

    def _open_models_dir(self):
        """打开模型目录"""
        import subprocess
        subprocess.Popen(f'explorer "{MODELS_DIR}"')
