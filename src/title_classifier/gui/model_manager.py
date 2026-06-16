"""模型管理对话框 - 显示、切换和下载 YOLO/CLIP 模型"""

import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk as tkttk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request

# 项目目录
PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
MODELS_DIR = PROJECT_DIR / "models"
YOLO_DIR = MODELS_DIR / "yolo"
CLIP_DIR = MODELS_DIR / "clip"

# YOLO 模型配置
YOLO_MODELS = {
    "detect": {
        "name": "YOLO 检测模型",
        "description": "目标检测，识别物体位置",
        "current": "yolov8n.pt",
        "options": {
            "yolov8n.pt": {"size": 6.25, "desc": "Nano - 最快，精度最低"},
            "yolov8s.pt": {"size": 22.0, "desc": "Small - 平衡"},
            "yolov8m.pt": {"size": 52.0, "desc": "Medium - 较高精度"},
            "yolov8l.pt": {"size": 87.0, "desc": "Large - 高精度"},
            "yolov8x.pt": {"size": 131.0, "desc": "Extra - 最高精度"},
        },
        "download_base": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
    },
    "pose": {
        "name": "YOLO 姿态模型",
        "description": "人体姿态检测，分析人体关键点",
        "current": "yolo11m-pose.pt",
        "options": {
            "yolov8n-pose.pt": {"size": 6.52, "desc": "Nano - 最快"},
            "yolov8s-pose.pt": {"size": 22.42, "desc": "Small - 平衡"},
            "yolo11m-pose.pt": {"size": 40.49, "desc": "Medium - 推荐"},
        },
        "download_base": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
    },
    "segment": {
        "name": "YOLO 分割模型",
        "description": "实例分割，识别物体轮廓",
        "current": "yolov8n-seg.pt",
        "options": {
            "yolov8n-seg.pt": {"size": 6.74, "desc": "Nano - 最快"},
            "yolov8s-seg.pt": {"size": 23.0, "desc": "Small - 平衡"},
            "yolov8m-seg.pt": {"size": 54.0, "desc": "Medium - 较高精度"},
        },
        "download_base": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
    },
}

# CLIP 模型配置
CLIP_MODELS = {
    "name": "CLIP 视觉模型",
    "description": "视觉特征提取，用于图像理解和分类",
    "current": "CLIP-ViT-B-16",
    "options": {
        "CLIP-ViT-B-16": {"size": 650, "desc": "Base - 推荐，平衡性能和精度"},
        "CLIP-ViT-B-32": {"size": 350, "desc": "Base-32 - 更轻量"},
        "CLIP-ViT-L-14": {"size": 1700, "desc": "Large - 最高精度，需要更多内存"},
    },
}


def get_file_size_mb(path: Path) -> float:
    """获取文件大小（MB）"""
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def get_dir_size_mb(path: Path) -> float:
    """获取目录大小（MB）"""
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def download_file(url: str, dest: Path, progress_callback=None) -> bool:
    """下载文件"""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest), reporthook=progress_callback)
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


class ModelManagerDialog(tk.Toplevel):
    """模型管理对话框"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("模型管理")
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

        # 当前选择的模型
        self.selected_models = self._load_current_models()

        # 构建 UI
        self._build_ui()

        # 居中到父窗口
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _load_current_models(self) -> Dict[str, str]:
        """加载当前使用的模型配置"""
        config_file = PROJECT_DIR / "config" / "models.json"
        if config_file.exists():
            import json
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # 默认配置
        return {
            "detect": "yolov8n.pt",
            "pose": "yolo11m-pose.pt",
            "segment": "yolov8n-seg.pt",
            "clip": "CLIP-ViT-B-16",
        }

    def _save_current_models(self):
        """保存当前使用的模型配置"""
        config_file = PROJECT_DIR / "config" / "models.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.selected_models, f, indent=2)

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

        # YOLO 模型区域
        self.yolo_widgets = {}
        for model_type, model_info in YOLO_MODELS.items():
            self._build_yolo_section(scroll_frame, model_type, model_info)

        # CLIP 模型区域
        self._build_clip_section(scroll_frame)

        # 模型统计
        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill=tk.X, pady=(10, 0))

        self.stats_label = ttk.Label(stats_frame, text="", foreground="#888888")
        self.stats_label.pack(side=tk.LEFT)

        self._update_stats()

        # 底部按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="打开模型目录",
            command=self._open_models_dir
        ).pack(side=tk.LEFT)

        ttk.Button(
            btn_frame,
            text="保存配置",
            command=self._save_config,
            bootstyle=SUCCESS
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            btn_frame,
            text="关闭",
            command=self.destroy
        ).pack(side=tk.RIGHT)

    def _build_yolo_section(self, parent, model_type: str, model_info: Dict):
        """构建 YOLO 模型配置区域"""
        frame = ttk.LabelFrame(parent, text=model_info["name"])
        frame.pack(fill=tk.X, padx=5, pady=5)

        inner_frame = ttk.Frame(frame, padding=10)
        inner_frame.pack(fill=tk.X)

        # 描述
        ttk.Label(
            inner_frame,
            text=model_info["description"],
            foreground="#888888"
        ).pack(anchor=tk.W, pady=(0, 5))

        # 当前模型
        current_model = self.selected_models.get(model_type, model_info["current"])
        current_path = YOLO_DIR / current_model
        current_size = get_file_size_mb(current_path)
        current_status = "已下载" if current_path.exists() else "未下载"

        status_color = "#28a745" if current_path.exists() else "#dc3545"

        current_frame = ttk.Frame(inner_frame)
        current_frame.pack(fill=tk.X, pady=2)
        ttk.Label(current_frame, text="当前:", width=8).pack(side=tk.LEFT)
        ttk.Label(
            current_frame,
            text=f"{current_model} ({current_size:.1f} MB) - {current_status}",
            foreground=status_color
        ).pack(side=tk.LEFT)

        # 模型选择
        select_frame = ttk.Frame(inner_frame)
        select_frame.pack(fill=tk.X, pady=5)
        ttk.Label(select_frame, text="切换:", width=8).pack(side=tk.LEFT)

        # 模型下拉框
        model_var = tk.StringVar(value=current_model)
        model_combo = ttk.Combobox(
            select_frame,
            textvariable=model_var,
            values=list(model_info["options"].keys()),
            state="readonly",
            width=20
        )
        model_combo.pack(side=tk.LEFT, padx=5)

        # 模型描述
        model_desc_label = ttk.Label(select_frame, text="", foreground="#888888")
        model_desc_label.pack(side=tk.LEFT, padx=5)

        # 更新描述
        def on_model_change(event=None):
            selected = model_var.get()
            if selected in model_info["options"]:
                desc = model_info["options"][selected]["desc"]
                size = model_info["options"][selected]["size"]
                model_desc_label.configure(text=f"{desc} ({size:.1f} MB)")
                self.selected_models[model_type] = selected
                self._update_stats()

        model_combo.bind("<<ComboboxSelected>>", on_model_change)
        on_model_change()  # 初始化描述

        # 下载按钮
        btn_frame = ttk.Frame(inner_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        progress_label = ttk.Label(btn_frame, text="", foreground="#888888")
        progress_label.pack(side=tk.LEFT, padx=5)

        download_btn = ttk.Button(
            btn_frame,
            text="下载选中模型",
            command=lambda mt=model_type, mv=model_var, pl=progress_label: 
                self._download_yolo_model(mt, mv.get(), pl)
        )
        download_btn.pack(side=tk.RIGHT)

        # 保存引用
        self.yolo_widgets[model_type] = {
            "model_var": model_var,
            "model_combo": model_combo,
            "progress_label": progress_label,
        }

    def _build_clip_section(self, parent):
        """构建 CLIP 模型配置区域"""
        frame = ttk.LabelFrame(parent, text=CLIP_MODELS["name"])
        frame.pack(fill=tk.X, padx=5, pady=5)

        inner_frame = ttk.Frame(frame, padding=10)
        inner_frame.pack(fill=tk.X)

        # 描述
        ttk.Label(
            inner_frame,
            text=CLIP_MODELS["description"],
            foreground="#888888"
        ).pack(anchor=tk.W, pady=(0, 5))

        # 当前模型
        current_model = self.selected_models.get("clip", CLIP_MODELS["current"])
        clip_path = CLIP_DIR / f"models--laion--{current_model}"
        clip_size = get_dir_size_mb(clip_path)
        clip_exists = clip_path.exists() and clip_size > 100  # CLIP 模型应该大于 100MB

        status_color = "#28a745" if clip_exists else "#dc3545"
        current_status = "已下载" if clip_exists else "未下载"

        current_frame = ttk.Frame(inner_frame)
        current_frame.pack(fill=tk.X, pady=2)
        ttk.Label(current_frame, text="当前:", width=8).pack(side=tk.LEFT)
        ttk.Label(
            current_frame,
            text=f"{current_model} (~{clip_size:.0f} MB) - {current_status}",
            foreground=status_color
        ).pack(side=tk.LEFT)

        # 模型选择
        select_frame = ttk.Frame(inner_frame)
        select_frame.pack(fill=tk.X, pady=5)
        ttk.Label(select_frame, text="切换:", width=8).pack(side=tk.LEFT)

        self.clip_model_var = tk.StringVar(value=current_model)
        model_combo = ttk.Combobox(
            select_frame,
            textvariable=self.clip_model_var,
            values=list(CLIP_MODELS["options"].keys()),
            state="readonly",
            width=20
        )
        model_combo.pack(side=tk.LEFT, padx=5)

        # 模型描述
        self.clip_desc_label = ttk.Label(select_frame, text="", foreground="#888888")
        self.clip_desc_label.pack(side=tk.LEFT, padx=5)

        # 更新描述
        def on_clip_change(event=None):
            selected = self.clip_model_var.get()
            if selected in CLIP_MODELS["options"]:
                desc = CLIP_MODELS["options"][selected]["desc"]
                size = CLIP_MODELS["options"][selected]["size"]
                self.clip_desc_label.configure(text=f"{desc} (~{size} MB)")
                self.selected_models["clip"] = selected
                self._update_stats()

        model_combo.bind("<<ComboboxSelected>>", on_clip_change)
        on_clip_change()

        # 下载按钮
        btn_frame = ttk.Frame(inner_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.clip_progress_label = ttk.Label(btn_frame, text="", foreground="#888888")
        self.clip_progress_label.pack(side=tk.LEFT, padx=5)

        download_btn = ttk.Button(
            btn_frame,
            text="下载选中模型",
            command=self._download_clip_model
        )
        download_btn.pack(side=tk.RIGHT)

        # 提示
        ttk.Label(
            inner_frame,
            text="注意: CLIP 模型较大，下载后需重启程序生效",
            foreground="#ffc107",
            font=("Microsoft YaHei", 8)
        ).pack(anchor=tk.W, pady=(5, 0))

    def _update_stats(self):
        """更新模型统计信息"""
        total_size = 0.0

        # YOLO 模型
        for model_type in YOLO_MODELS:
            model_name = self.selected_models.get(model_type, "")
            if model_name:
                model_path = YOLO_DIR / model_name
                total_size += get_file_size_mb(model_path)

        # OpenVINO 模型
        openvino_dir = YOLO_DIR / "openvino"
        total_size += get_dir_size_mb(openvino_dir)

        # CLIP 模型
        clip_name = self.selected_models.get("clip", "")
        if clip_name:
            clip_path = CLIP_DIR / f"models--laion--{clip_name}"
            total_size += get_dir_size_mb(clip_path)

        self.stats_label.configure(text=f"模型目录: {MODELS_DIR}  |  已下载模型总大小: {total_size:.1f} MB")

    def _download_yolo_model(self, model_type: str, model_name: str, progress_label: ttk.Label):
        """下载 YOLO 模型"""
        model_path = YOLO_DIR / model_name

        if model_path.exists():
            messagebox.showinfo("提示", f"模型 {model_name} 已存在", parent=self)
            return

        # 确认下载
        model_info = YOLO_MODELS[model_type]["options"].get(model_name, {})
        size = model_info.get("size", 0)

        if not messagebox.askyesno("确认下载", f"确定要下载 {model_name} ({size:.1f} MB) 吗？", parent=self):
            return

        # 开始下载
        progress_label.configure(text="下载中...")
        download_btn = progress_label.master.winfo_children()[-1]  # 获取下载按钮
        download_btn.configure(state="disabled")

        def download_thread():
            url = YOLO_MODELS[model_type]["download_base"] + model_name

            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    percent = min(100, count * block_size * 100 / total_size)
                    self.after(0, lambda: progress_label.configure(text=f"下载中... {percent:.0f}%"))

            try:
                success = download_file(url, model_path, progress_hook)
                if success:
                    self.after(0, lambda: progress_label.configure(text="下载完成", foreground="#28a745"))
                    self.after(0, self._update_stats)
                else:
                    self.after(0, lambda: progress_label.configure(text="下载失败", foreground="#dc3545"))
            except Exception as e:
                self.after(0, lambda: progress_label.configure(text=f"下载失败: {e}", foreground="#dc3545"))
            finally:
                self.after(0, lambda: download_btn.configure(state="normal"))

        threading.Thread(target=download_thread, daemon=True).start()

    def _download_clip_model(self):
        """下载 CLIP 模型"""
        model_name = self.clip_model_var.get()
        model_path = CLIP_DIR / f"models--laion--{model_name}"

        if model_path.exists() and get_dir_size_mb(model_path) > 100:
            messagebox.showinfo("提示", f"模型 {model_name} 已存在", parent=self)
            return

        # 确认下载
        model_info = CLIP_MODELS["options"].get(model_name, {})
        size = model_info.get("size", 0)

        if not messagebox.askyesno("确认下载", f"确定要下载 {model_name} (~{size} MB) 吗？\n这可能需要几分钟时间。", parent=self):
            return

        # 开始下载
        self.clip_progress_label.configure(text="下载中...")

        def download_thread():
            try:
                # 使用 huggingface_hub 下载 CLIP 模型
                self.after(0, lambda: self.clip_progress_label.configure(text="正在初始化下载..."))

                # 检查是否安装了 huggingface_hub
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    self.after(0, lambda: self.clip_progress_label.configure(
                        text="需要安装 huggingface_hub: pip install huggingface_hub",
                        foreground="#dc3545"
                    ))
                    return

                # 下载模型
                repo_id = f"laion/{model_name}"
                self.after(0, lambda: self.clip_progress_label.configure(text=f"下载中: {repo_id}"))

                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,
                )

                self.after(0, lambda: self.clip_progress_label.configure(text="下载完成", foreground="#28a745"))
                self.after(0, self._update_stats)

            except Exception as e:
                self.after(0, lambda: self.clip_progress_label.configure(
                    text=f"下载失败: {str(e)[:50]}",
                    foreground="#dc3545"
                ))

        threading.Thread(target=download_thread, daemon=True).start()

    def _open_models_dir(self):
        """打开模型目录"""
        import subprocess
        subprocess.Popen(f'explorer "{MODELS_DIR}"')

    def _save_config(self):
        """保存模型配置"""
        self._save_current_models()
        messagebox.showinfo("成功", "模型配置已保存\n重启程序后生效", parent=self)
