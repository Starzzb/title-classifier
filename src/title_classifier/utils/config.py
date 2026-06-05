"""配置加载模块"""

from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    加载 TOML 配置文件

    Args:
        config_path: 配置文件路径，默认为 config/default.toml

    Returns:
        配置字典
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "default.toml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return {}

    try:
        import tomllib
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import tomli
            with open(config_path, "rb") as f:
                return tomli.load(f)
        except ImportError:
            return {}
