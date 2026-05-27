"""Provider基类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseProvider(ABC):
    """AI Provider 基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "")
        self.url = config.get("url", "")
        self.default_model = config.get("default_model", "")
        self.requires_api_key = config.get("requires_api_key", False)

    @abstractmethod
    def call_text(self, prompt: str, model: str = None, **kwargs) -> str:
        """调用文本API"""
        pass

    @abstractmethod
    def call_vision(self, image_b64: str, prompt: str, model: str = None, **kwargs) -> str:
        """调用视觉API"""
        pass

    def call_audio(self, audio_b64: str, prompt: str = None, model: str = None, **kwargs) -> str:
        """调用音频API（可选实现）"""
        raise NotImplementedError(f"{self.name} 不支持音频API")
