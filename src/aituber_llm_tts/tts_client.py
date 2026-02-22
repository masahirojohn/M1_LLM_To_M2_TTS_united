from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import os
import wave
import json
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    from google import genai
    from google.genai import types as genai_types
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

from .config import AppConfig
from .llm_client import M1Output


@dataclass
class M2Output:
    schema_version: str
    session_id: str
    utt_id: str
    wav_path: str
    audio_ms: int
    sample_rate: int
    num_channels: int
    fmt: str
    tts_engine: str
    tts_preset: str
    tts_kana: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ========== Dummy TTS (従来どおりの機械音) ==========

class DummyTTSClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def synthesize(
        self,
        m1: M1Output,
        audio_path: Path,
        tts_style_prompt: Optional[str] = None,
    ) -> M2Output:
        """
        文字数ベースで適当な長さのサイン波ビープ音を生成。
        """
        pcm_bytes, sample_rate, audio_ms = self.synthesize_pcm_bytes(
            m1,
            tts_style_prompt=tts_style_prompt,
        )

        num_channels = 1
        sample_width = 2  # 16bit

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)

        return M2Output(
            schema_version=self._config.schema_version,
            session_id=m1.session_id,
            utt_id=m1.utt_id,
            wav_path=str(audio_path),
            audio_ms=audio_ms,
            sample_rate=sample_rate,
            num_channels=num_channels,
            fmt="wav",
            tts_engine="dummy-tts",
            tts_preset=tts_style_prompt or "",
            tts_kana=None,
        )

    def synthesize_pcm_bytes(
        self,
        m1: M1Output,
        tts_style_prompt: Optional[str] = None,
    ) -> Tuple[bytes, int, int]:
        """
        wavを書かずに、PCM16 mono の bytes を返す。
        Returns:
            pcm_bytes: bytes (PCM16 little-endian mono)
            sample_rate: int
            audio_ms: int
        """
        text = m1.text or ""
        cfg = self._config.tts

        # 長さはだいたい base_ms_per_char * 文字数（最低 base_ms_min）
        est_ms = max(cfg.base_ms_min, cfg.base_ms_per_char * max(1, len(text)))
        sample_rate = cfg.sample_rate
        num_channels = 1
        sample_width = 2  # 16bit
        bytes_per_frame = num_channels * sample_width

        n_frames = int(est_ms / 1000.0 * sample_rate)
        freq = 880.0  # A5 くらいのビープ

        buf = bytearray()
        for i in range(n_frames):
            t = i / sample_rate
            val = int(0.2 * 32767 * math.sin(2 * math.pi * freq * t))
            buf.extend(val.to_bytes(2, byteorder="little", signed=True))

        # 念のため bytes_per_frame で割り切れることを保証（Dummyは必ずOKだがガード）
        n_frames2 = len(buf) // bytes_per_frame
        audio_ms = int(round(n_frames2 / sample_rate * 1000.0))

        return (bytes(buf), int(sample_rate), int(audio_ms))


# ========== Gemini TTS ==========

class RealTTSClient:
    """
    Gemini TTS を使う実TTSクライアント。
    オーケストレータからは Dummy と同じインタフェースで呼び出せる。
    """

    def __init__(self, config: AppConfig, api_key: Optional[str] = None) -> None:
        if not _HAS_GEMINI:
            raise RuntimeError("google-genai がインストールされていません。`pip install google-genai` を実行してください。")

        self._config = config
        self._tts_cfg = config.tts

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY が設定されていません。環境変数に設定してください。")

        self._client = genai.Client(api_key=api_key)

    def synthesize(
        self,
        m1: M1Output,
        audio_path: Path,
        tts_style_prompt: Optional[str] = None,
    ) -> M2Output:
        pcm_bytes, sample_rate, audio_ms = self.synthesize_pcm_bytes(
            m1,
            tts_style_prompt=tts_style_prompt,
        )

        num_channels = 1
        sample_width = 2  # 16bit

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)

        return M2Output(
            schema_version=self._config.schema_version,
            session_id=m1.session_id,
            utt_id=m1.utt_id,
            wav_path=str(audio_path),
            audio_ms=audio_ms,
            sample_rate=sample_rate,
            num_channels=num_channels,
            fmt="wav",
            tts_engine="gemini-tts",
            tts_preset=tts_style_prompt or "",
            tts_kana=None,
        )

    def synthesize_pcm_bytes(
        self,
        m1: M1Output,
        tts_style_prompt: Optional[str] = None,
    ) -> Tuple[bytes, int, int]:
        """
        wavを書かずに、PCM16 mono の bytes を返す（方法Bの核）。
        Returns:
            pcm_bytes: bytes (PCM16 little-endian mono)
            sample_rate: int
            audio_ms: int
        """
        text = m1.text or ""
        style_prompt = tts_style_prompt or ""

        # シンプルに「スタイル指示 + セリフ」を1テキストにまとめる
        if style_prompt:
            prompt = f"{style_prompt}. {text}"
        else:
            prompt = text

        cfg = self._tts_cfg

        response = self._client.models.generate_content(
            model=cfg.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                            voice_name=cfg.voice_name
                        )
                    )
                ),
            ),
        )

        # ---- robust audio extraction (avoid NoneType crash) ----
        def _dump(obj) -> str:
            # google-genai objects often support model_dump()
            try:
                return json.dumps(obj.model_dump(), ensure_ascii=False)  # type: ignore
            except Exception:
                try:
                    return json.dumps(obj.__dict__, ensure_ascii=False)  # type: ignore
                except Exception:
                    return repr(obj)

        if not getattr(response, "candidates", None):
            raise RuntimeError(
                "TTS response has no candidates. "
                "Likely model/config/quota issue. response=" + _dump(response)
            )

        pcm_bytes_raw = None
        for cand in response.candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for p in parts:
                inline = getattr(p, "inline_data", None)
                if inline is None:
                    continue
                data = getattr(inline, "data", None)
                if data:
                    pcm_bytes_raw = data
                    break
            if pcm_bytes_raw:
                break

        if not pcm_bytes_raw:
            # no audio returned -> print the whole response to debug
            raise RuntimeError(
                "TTS response contains no inline audio data (content/parts missing). "
                "Check: (1) model is a TTS model, (2) response_modalities includes AUDIO, "
                "(3) quota/transient errors. response=" + _dump(response)
            )

        pcm_bytes = bytes(pcm_bytes_raw)

        sample_rate = int(cfg.sample_rate)
        num_channels = 1
        sample_width = 2  # 16bit
        bytes_per_frame = num_channels * sample_width
        n_frames = len(pcm_bytes) // bytes_per_frame
        audio_ms = int(round(n_frames / sample_rate * 1000.0))

        # ガード：端数（万一）があれば切り捨て（PCM16 mono の2byte境界を守る）
        trim_n = n_frames * bytes_per_frame
        if trim_n != len(pcm_bytes):
            pcm_bytes = pcm_bytes[:trim_n]

        return (pcm_bytes, sample_rate, int(audio_ms))


# ========== Factory ==========

def create_tts_client(config: AppConfig):
    """
    config.tts.engine に応じて Dummy / Gemini を切り替える。
    """
    engine = (config.tts.engine or "dummy").lower()
    if engine == "gemini":
        return RealTTSClient(config)
    return DummyTTSClient(config)
