# src/aituber_llm_tts/factory.py

from typing import Any, Dict
from .tts_client_dummy import DummyTTSClient
from .tts_client_gemini import RealTTSClient, GeminiTTSConfig


def create_tts_client(config: Dict[str, Any]):
    """
    YAMLの config["tts"]  を読んで TTSClient を選択
    """
    tts_cfg = config.get("tts", {})
    engine = tts_cfg.get("engine", "dummy")

    if engine == "gemini":
        gem_cfg = GeminiTTSConfig(
            model=tts_cfg.get("model", "gemini-2.5-flash-preview-tts"),
            voice_name=tts_cfg.get("voice_name", "Kore"),
            sample_rate=tts_cfg.get("sample_rate", 24000),
        )
        return RealTTSClient(config=gem_cfg)

    return DummyTTSClient()
