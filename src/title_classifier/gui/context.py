"""共享上下文 - 各 Tab 之间共享的状态"""

import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AppContext:
    """各 Tab 共享的应用上下文"""

    # 数据库
    db: object = None  # MediaDB instance

    # 共享 CSV 路径（所有 Tab 共用同一个 StringVar）
    csv_var: tk.StringVar = field(default=None)

    # 子进程管理
    process: Optional[object] = None  # subprocess.Popen
    running: bool = False

    # 配置
    config: dict = field(default_factory=dict)  # 合并后的 default.toml + user.toml
    user_config_path: Optional[Path] = None  # config/user.toml

    # UI 引用（由 app.py 设置）
    stop_btn: object = None  # ttk.Button
    progress_label: object = None  # ttk.Label
    log_text: object = None  # scrolledtext.ScrolledText

    def __post_init__(self):
        if self.csv_var is None:
            self.csv_var = tk.StringVar(value="data/output/title_review.csv")
