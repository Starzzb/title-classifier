"""设置对话框 - 占位文件，完整实现在步骤4"""

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *


class SettingsDialog:
    """设置对话框（占位）"""

    def __init__(self, parent, ctx):
        from tkinter import messagebox
        messagebox.showinfo("设置", "设置对话框将在步骤4中实现", parent=parent)
