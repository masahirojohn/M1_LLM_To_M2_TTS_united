# src/aituber_llm_tts/orchestrator.py

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Dict, Any

from .config import AppConfig
from .emo_mapping import EmoTtsMapper
from .llm_client import DummyLLMClient, M1Output
from .tts_client import DummyTTSClient, M2Output


@dataclass
class SessionState:
    session_id: str
    global_t_ms: int = 0


class Orchestrator:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._emo_mapper = EmoTtsMapper.from_yaml(config.emo_map_path)
        self._llm = DummyLLMClient(config)
        self._tts = DummyTTSClient(config)

        # パス準備
        self._paths = config.paths
        self._paths.out_root.mkdir(parents=True, exist_ok=True)
        self._paths.audio_dir.mkdir(parents=True, exist_ok=True)

    def _append_utt_log(self, log_line: Dict[str, Any]) -> None:
        log_path = self._paths.utt_log_path
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

    def _make_session_id(self) -> str:
        # 実運用では UUID などを使う想定
        return f"{self._config.session.id_prefix}demo"

    def run_batch(self) -> SessionState:
        session_id = self._make_session_id()
        state = SessionState(session_id=session_id, global_t_ms=0)

        for turn_idx in range(self._config.llm.max_turns):
            # --- M1: LLM から次の発話を取得 ---
            m1: M1Output = self._llm.next_utterance(session_id=session_id)

            # emo_id から TTSスタイルプロンプトを補完
            style_prompt = self._emo_mapper.get_style_prompt(m1.emo_id)
            m1.tts_style_prompt = style_prompt

            # --- M2: TTS合成 ---
            audio_fname = f"{m1.utt_id}.wav"
            audio_path = self._paths.audio_dir / audio_fname

            m2: M2Output = self._tts.synthesize(
                m1=m1,
                audio_path=audio_path,
                tts_style_prompt=style_prompt,
            )

            utterance_offset_ms = state.global_t_ms

            # ログ 1 行分を組み立て
            log_line: Dict[str, Any] = {
                "schema_version": self._config.schema_version,
                "session_id": session_id,
                "utt_id": m1.utt_id,
                "utterance_offset_ms": utterance_offset_ms,
                "llm": m1.to_dict(),
                "tts": m2.to_dict(),
            }

            self._append_utt_log(log_line)

            # 次の発話のために global_t_ms を進める
            # TTS音声の長さ + LLMが提案した待ち時間
            state.global_t_ms += m2.audio_ms + m1.next_wait_ms

            print(
                f"[turn {turn_idx}] {m1.text} "
                f"(emo_id={m1.emo_id}, audio_ms={m2.audio_ms})"
            )

        print(f"Session {session_id} finished. total_t_ms={state.global_t_ms}")
        print(f"- Log : {self._paths.utt_log_path}")
        print(f"- Audio dir : {self._paths.audio_dir}")

        return state
