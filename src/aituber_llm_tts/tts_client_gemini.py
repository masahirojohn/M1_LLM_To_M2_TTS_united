# src/aituber_llm_tts/tts_client_gemini.py

from __future__ import annotations
from dataclasses import dataclass
import os
import wave
import pathlib
from typing import Optional

from google import genai
from google.genai import types

from .contracts import TTSRequest, TTSResult
from .tts_client_base import ITTSClient


@dataclass
class GeminiTTSConfig:
    model: str = "gemini-2.5-flash-preview-tts"
    voice_name: str = "Kore"
    sample_rate: int = 24000
    num_channels: int = 1
    sample_width: int = 2
    tts_engine_name: str = "gemini-tts"


class RealTTSClient(ITTSClient):
    def __init__(self, config: Optional[GeminiTTSConfig] = None, api_key: Optional[str] = None):
        self.config = config or GeminiTTSConfig()
        api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self._client = genai.Client(api_key=api_key)

    def synthesize(self, req: TTSRequest) -> TTSResult:

        text = req.text
        style_prompt = req.tts_style_prompt or ""
        prompt = f"{style_prompt}. {text}" if style_prompt else text

        cfg = self.config

        response = self._client.models.generate_content(
            model=cfg.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=cfg.voice_name
                        )
                    )
                ),
            ),
        )

        pcm_bytes: bytes = response.candidates[0].content.parts[0].inline_data.data

        wav_path = pathlib.Path(req.wav_path)
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(cfg.num_channels)
            wf.setsampwidth(cfg.sample_width)
            wf.setframerate(cfg.sample_rate)
            wf.writeframes(pcm_bytes)

        bytes_per_frame = cfg.num_channels * cfg.sample_width
        num_frames = len(pcm_bytes) // bytes_per_frame
        audio_ms = int(round((num_frames / float(cfg.sample_rate)) * 1000.0))

        return TTSResult(
            schema_version="llm_tts_contracts_v0.2",
            session_id=req.session_id,
            utt_id=req.utt_id,
            wav_path=str(wav_path),
            audio_ms=audio_ms,
            sample_rate=cfg.sample_rate,
            num_channels=cfg.num_channels,
            fmt="wav",
            tts_engine=cfg.tts_engine_name,
            tts_preset=style_prompt,
            tts_kana=None,
        )
