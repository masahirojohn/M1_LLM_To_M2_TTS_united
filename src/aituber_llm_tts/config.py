from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class AppPaths:
    out_root: Path
    audio_subdir: str
    utt_log_filename: str

    @property
    def audio_dir(self) -> Path:
        return self.out_root / self.audio_subdir

    @property
    def utt_log_path(self) -> Path:
        return self.out_root / self.utt_log_filename


@dataclass
class LLMConfig:
    provider: str           # "dummy" or "openai"
    model: str
    temperature: float
    max_turns: int
    mode: str
    system_prompt_path: Path


@dataclass
class TTSConfig:
    engine: str            # "dummy" or "gemini"
    model: str             # "gemini-2.5-flash-preview-tts" など
    voice_name: str        # "Kore" など
    sample_rate: int       # 24000 など
    base_ms_per_char: int  # Dummy 用
    base_ms_min: int       # Dummy 用（最短長さ）


@dataclass
class SessionConfig:
    id_prefix: str
    seed: int

    # ★追加：配信テーマ（LLM へのヒント）
    episode_theme: str

    # ★追加：配信時間上限（ms）。指定無しなら None。
    max_total_ms: int | None = None


@dataclass
class AppConfig:
    schema_version: str
    llm: LLMConfig
    tts: TTSConfig
    paths: AppPaths
    session: SessionConfig
    emo_map_path: Path | None = None


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path)
    raw = load_yaml(config_path)

    schema_version = raw.get("schema_version", "llm_tts_contracts_v0.2")

    llm_raw = raw.get("llm", {})
    tts_raw = raw.get("tts", {})
    paths_raw = raw.get("paths", {})
    sess_raw = raw.get("session", {})

    # --- LLM ---
    llm = LLMConfig(
        provider=str(llm_raw.get("provider", "dummy")),
        model=str(llm_raw.get("model", "gpt-4.1-mini")),
        temperature=float(llm_raw.get("temperature", 0.7)),
        max_turns=int(llm_raw.get("max_turns", 5)),
        mode=str(llm_raw.get("mode", "monologue")),
        system_prompt_path=(
            config_path.parent
            / llm_raw.get("system_prompt_path", "system_prompt_m1.txt")
        ),
    )

    # --- TTS ---
    tts = TTSConfig(
        engine=str(tts_raw.get("engine", "dummy")),
        model=str(tts_raw.get("model", "gemini-2.5-flash-preview-tts")),
        voice_name=str(tts_raw.get("voice_name", "Kore")),
        sample_rate=int(tts_raw.get("sample_rate", 24000)),
        base_ms_per_char=int(tts_raw.get("base_ms_per_char", 90)),
        base_ms_min=int(tts_raw.get("base_ms_min", 600)),
    )

    # --- Paths ---
    out_root = Path(paths_raw.get("out_root", "out"))
    paths = AppPaths(
        out_root=out_root,
        audio_subdir=str(paths_raw.get("audio_subdir", "audio")),
        utt_log_filename=str(paths_raw.get("utt_log_filename", "utt_log.jsonl")),
    )

    # --- Session ---
    session = SessionConfig(
        id_prefix=str(sess_raw.get("id_prefix", "sess_")),
        seed=int(sess_raw.get("seed", 12345)),
        episode_theme=str(sess_raw.get("episode_theme", "雑談配信")),
        max_total_ms=(
            int(sess_raw["max_total_ms"]) if "max_total_ms" in sess_raw else None
        ),
    )

    # --- emo map ---
    emo_map_path = config_path.parent / "emo_tts_map.yaml"
    if not emo_map_path.exists():
        emo_map_path = None

    return AppConfig(
        schema_version=schema_version,
        llm=llm,
        tts=tts,
        paths=paths,
        session=session,
        emo_map_path=emo_map_path,
    )
