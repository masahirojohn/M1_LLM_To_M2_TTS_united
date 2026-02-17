# src/aituber_llm_tts/orchestrator.py

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Dict, Any

from .config import AppConfig
from .emo_mapping import EmoTtsMapper
from .llm_client import create_llm_client, M1Output
from .tts_client import create_tts_client, M2Output   # ★ここを修正


@dataclass
class SessionState:
    session_id: str
    global_t_ms: int = 0


class Orchestrator:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._emo_mapper = EmoTtsMapper.from_yaml(config.emo_map_path)
        self._llm = create_llm_client(config)
        self._tts = create_tts_client(config)   # ★Dummy から Factory に変更

        self._paths = config.paths
        self._paths.out_root.mkdir(parents=True, exist_ok=True)
        self._paths.audio_dir.mkdir(parents=True, exist_ok=True)

        self._history: List[Dict[str, Any]] = []
        self._utt_counter: int = 0

    def _append_utt_log(self, log_line: Dict[str, Any]) -> None:
        log_path = self._paths.utt_log_path
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

    def _make_session_id(self) -> str:
        return f"{self._config.session.id_prefix}demo"

    def _next_utt_id(self) -> str:
        self._utt_counter += 1
        return f"utt_{self._utt_counter:06d}"

    # ========== 従来のバッチモード ==========

    def run_batch(self) -> SessionState:
        session_id = self._make_session_id()
        state = SessionState(session_id=session_id, global_t_ms=0)

        for turn_idx in range(self._config.llm.max_turns):
            utt_id = self._next_utt_id()

            m1: M1Output = self._llm.next_utterance(
                session_id=session_id,
                utt_id=utt_id,
                mode=self._config.llm.mode,
                history=self._history,
                # viewer_comment は Phase1 互換のため None 固定
                viewer_comment=None,
            )

            style_prompt = self._emo_mapper.get_style_prompt(m1.emo_id)
            m1.tts_style_prompt = style_prompt

            audio_fname = f"{m1.utt_id}.wav"
            audio_path = self._paths.audio_dir / audio_fname

            m2: M2Output = self._tts.synthesize(
                m1=m1,
                audio_path=audio_path,
                tts_style_prompt=style_prompt,
            )

            utterance_offset_ms = state.global_t_ms

            log_line: Dict[str, Any] = {
                "schema_version": self._config.schema_version,
                "session_id": session_id,
                "utt_id": m1.utt_id,
                "utterance_offset_ms": utterance_offset_ms,
                "llm": m1.to_dict(),
                "tts": m2.to_dict(),
            }

            self._append_utt_log(log_line)

            self._history.append(
                {
                    "role": "assistant",
                    "text": m1.text,
                    "emo_id": m1.emo_id,
                }
            )

            state.global_t_ms += m2.audio_ms + m1.next_wait_ms

            print(
                f"[turn {turn_idx}] {m1.text} "
                f"(emo_id={m1.emo_id}, audio_ms={m2.audio_ms}, tts_engine={m2.tts_engine})"
            )

        print(f"Session {session_id} finished. total_t_ms={state.global_t_ms}")
        print(f"- Log : {self._paths.utt_log_path}")
        print(f"- Audio dir : {self._paths.audio_dir}")

        return state

    # ========== Phase2 用：インタラクティブ CLI モード ==========

    def run_interactive_cli(self) -> SessionState:
        """
        Phase2: 簡単インタラクション用 CLI ループ。
        - 1 行入力ごとに 1 発話を生成（コメントへのリアクションモード）
        - 空行 → viewer_comment=None で通常モノローグ
        - 'quit' / 'exit' で終了
        """
        session_id = self._make_session_id()
        state = SessionState(session_id=session_id, global_t_ms=0)

        print("=== Interactive CLI mode (Phase2) ===")
        print("入力して Enter で 1 発話リアクション。")
        print("空行 Enter → 通常モノローグ")
        print("'quit' または 'exit' → 終了\n")

        turn_idx = 0
        while True:
            try:
                user_raw = input("viewer_comment> ")
            except EOFError:
                # Ctrl+D などで終了
                print("\n[EOF] 終了します。")
                break

            if user_raw is None:
                user_raw = ""
            user_str = user_raw.strip()

            if user_str.lower() in ("quit", "exit"):
                print("[INFO] quit コマンドを受信したため終了します。")
                break

            viewer_comment: str | None = user_str if user_str != "" else None

            utt_id = self._next_utt_id()

            m1: M1Output = self._llm.next_utterance(
                session_id=session_id,
                utt_id=utt_id,
                mode=self._config.llm.mode,
                history=self._history,
                viewer_comment=viewer_comment,
            )

            style_prompt = self._emo_mapper.get_style_prompt(m1.emo_id)
            m1.tts_style_prompt = style_prompt

            audio_fname = f"{m1.utt_id}.wav"
            audio_path = self._paths.audio_dir / audio_fname

            m2: M2Output = self._tts.synthesize(
                m1=m1,
                audio_path=audio_path,
                tts_style_prompt=style_prompt,
            )

            utterance_offset_ms = state.global_t_ms

            log_line: Dict[str, Any] = {
                "schema_version": self._config.schema_version,
                "session_id": session_id,
                "utt_id": m1.utt_id,
                "utterance_offset_ms": utterance_offset_ms,
                "llm": m1.to_dict(),
                "tts": m2.to_dict(),
            }

            self._append_utt_log(log_line)

            self._history.append(
                {
                    "role": "assistant",
                    "text": m1.text,
                    "emo_id": m1.emo_id,
                }
            )

            state.global_t_ms += m2.audio_ms + m1.next_wait_ms

            print(
                f"[turn {turn_idx}] {m1.text} "
                f"(emo_id={m1.emo_id}, audio_ms={m2.audio_ms}, tts_engine={m2.tts_engine})"
            )
            print(f"  - viewer_comment_used: {m1.meta.get('viewer_comment_used')}")
            if "viewer_comment_text" in m1.meta:
                print(f"  - viewer_comment_text: {m1.meta.get('viewer_comment_text')}")

            turn_idx += 1

        print(f"\nInteractive session {session_id} finished. total_t_ms={state.global_t_ms}")
        print(f"- Log : {self._paths.utt_log_path}")
        print(f"- Audio dir : {self._paths.audio_dir}")

        return state
