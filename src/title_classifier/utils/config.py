"""配置加载模块 - 支持 default.toml + user.toml 两层配置"""

from pathlib import Path
from typing import Any, Dict


PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "default.toml"
USER_CONFIG_PATH = PROJECT_DIR / "config" / "user.toml"


def _load_toml(path: Path) -> dict:
    """加载单个 TOML 文件"""
    if not path.exists():
        return {}
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import tomli
            with open(path, "rb") as f:
                return tomli.load(f)
        except ImportError:
            return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并字典，override 覆盖 base"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str = None) -> dict:
    """
    加载 TOML 配置文件（仅默认层）

    Args:
        config_path: 配置文件路径，默认为 config/default.toml

    Returns:
        配置字典
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)
    return _load_toml(config_path)


def load_merged_config() -> dict:
    """
    加载合并后的配置（default.toml + user.toml）

    user.toml 中的字段会覆盖 default.toml 中的同名字段

    Returns:
        合并后的配置字典
    """
    default = _load_toml(DEFAULT_CONFIG_PATH)
    user = _load_toml(USER_CONFIG_PATH)
    return _deep_merge(default, user)


def save_user_config(updates: dict) -> bool:
    """
    保存用户配置到 user.toml（只更新指定字段，不覆盖整个文件）

    Args:
        updates: 要更新的配置字典，支持嵌套

    Returns:
        是否保存成功
    """
    try:
        import tomli_w
    except ImportError:
        print("[错误] 需要安装 tomli-w: pip install tomli-w")
        return False

    # 加载现有 user.toml
    existing = _load_toml(USER_CONFIG_PATH)

    # 合并更新
    merged = _deep_merge(existing, updates)

    # 写入
    try:
        USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(USER_CONFIG_PATH, "wb") as f:
            tomli_w.dump(merged, f)
        return True
    except Exception as e:
        print(f"[错误] 保存用户配置失败: {e}")
        return False


def get_config_value(config: dict, dotted_key: str, default=None) -> Any:
    """
    用点号分隔的键从嵌套字典中获取值

    Args:
        config: 配置字典
        dotted_key: 点号分隔的键，如 "vision.max_image_size"
        default: 默认值

    Returns:
        配置值
    """
    keys = dotted_key.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
