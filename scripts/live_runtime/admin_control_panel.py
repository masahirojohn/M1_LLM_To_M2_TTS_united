from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from scripts.live_runtime.battle_runtime_admin_api import (
    DEFAULT_CONTROL_FILE,
    DEFAULT_INTERRUPT_FILE,
    DEFAULT_EVENT_FILE,
    DEFAULT_VAD_PROFILE_FILE,
    DEFAULT_LEADERSHIP_CONTROL_TEXT,
    DEFAULT_LEADERSHIP_INTERRUPT_TEXT,
    DEFAULT_LEADERSHIP_OPEN_TEXT,
    DEFAULT_NORMAL_CONVERSATION_TEXT,
    DEFAULT_ZOOM_LEADERSHIP_CONTROL_TEXT,
    clear_control,
    read_status,
    write_control,
    write_event,
    write_interrupt,
    write_leadership_open,
    write_leadership_start,
    write_zoom_leadership_start,
    write_mic_gate,
    write_normal_conversation,
    write_vad_profile,
)


def _append_ui_log(message: str) -> None:
    logs = st.session_state.setdefault("ui_logs", [])
    ts = time.strftime("%H:%M:%S")
    logs.append(f"[{ts}] {message}")

    # 増えすぎ防止
    st.session_state["ui_logs"] = logs[-50:]


def main() -> None:
    st.session_state.setdefault("ui_logs", [])

    st.set_page_config(
        page_title="Battle Runtime Admin",
        layout="wide",
    )

    st.title("Battle Runtime Admin Control Panel")

    st.caption(
        "battle_interrupt / battle_control / mic_gate / vad_profile を "
        "live txt file 経由で操作する管理者UI"
    )

    with st.sidebar:
        st.header("Files")

        control_file_str = st.text_input(
            "battle_control_live.txt",
            value=str(DEFAULT_CONTROL_FILE),
        )

        interrupt_file_str = st.text_input(
            "battle_interrupt_live.txt",
            value=str(DEFAULT_INTERRUPT_FILE),
        )

        event_file_str = st.text_input(
            "event_runtime_live.txt",
            value=str(DEFAULT_EVENT_FILE),
        )

        vad_profile_file_str = st.text_input(
            "vad_profile_live.txt",
            value=str(DEFAULT_VAD_PROFILE_FILE),
        )

        control_file = Path(control_file_str)
        interrupt_file = Path(interrupt_file_str)
        event_file = Path(event_file_str)
        vad_profile_file = Path(vad_profile_file_str)

        st.divider()

        st.write("Current paths")
        st.code(f"control:     {control_file}")
        st.code(f"interrupt:   {interrupt_file}")
        st.code(f"event:       {event_file}")
        st.code(f"vad_profile: {vad_profile_file}")
   
    st.header("基本操作")

    quick_interrupt_text = DEFAULT_LEADERSHIP_INTERRUPT_TEXT
    leadership_control_text = DEFAULT_LEADERSHIP_CONTROL_TEXT
    leadership_open_text = DEFAULT_LEADERSHIP_OPEN_TEXT
    zoom_leadership_control_text = DEFAULT_ZOOM_LEADERSHIP_CONTROL_TEXT

    b1, b2, b3, b4, b5 = st.columns(5)

    with b1:
        if st.button("今すぐ割り込め", type="primary", use_container_width=True):
            write_interrupt(
                interrupt_file=interrupt_file,
                text=quick_interrupt_text,
                priority="battle",
                expire_sec=60.0,
            )
            _append_ui_log("interrupt text=今すぐ割り込め")
            st.success("割り込みを投入しました。")

    with b2:
        if st.button("主導権奪取", type="primary", use_container_width=True):
            write_leadership_start(
                control_file=control_file,
                interrupt_file=interrupt_file,
                control_text=leadership_control_text,
                interrupt_text=quick_interrupt_text,
                priority="battle",
                expire_sec=60.0,
            )
            _append_ui_log("leadership start")
            st.success("主導権奪取を投入しました。")

    with b5:
        if st.button("Zoom主導権", type="primary", use_container_width=True):
            write_zoom_leadership_start(
                control_file=control_file,
                control_text=zoom_leadership_control_text,
            )
            _append_ui_log("zoom leadership control only")
            st.success("Zoom主導権 control を投入しました。")

    with b3:
        if st.button("主導権解除 OPEN", use_container_width=True):
            write_leadership_open(
                control_file=control_file,
                text=leadership_open_text,
            )
            _append_ui_log("leadership open")
            st.success("主導権解除 / OPEN を投入しました。")

    with b4:
        if st.button("通常会話へ戻す", use_container_width=True):
            write_normal_conversation(
                control_file=control_file,
                text=DEFAULT_NORMAL_CONVERSATION_TEXT,
            )
            _append_ui_log("normal conversation open")
            st.success("通常会話復帰を投入しました。")

    st.divider()

    st.header("マイクゲート")

    g1, g2 = st.columns(2)

    with g1:
        if st.button("MUTE 相手音声停止", type="primary", use_container_width=True):
            write_mic_gate(
                control_file=control_file,
                mic_gate="mute",
            )
            _append_ui_log("mic_gate=mute")
            st.success("mic_gate=mute を投入しました。")

    with g2:
        if st.button("OPEN 相手音声再開", use_container_width=True):
            write_mic_gate(
                control_file=control_file,
                mic_gate="open",
            )
            _append_ui_log("mic_gate=open")
            st.success("mic_gate=open を投入しました。")

    st.divider()

    st.header("イベント動画")

    event_id = st.text_input(
        "event_id",
        value="evt_001",
        help="event_catalog.json に登録済みの event_id を指定してください。",
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        if st.button("イベント投入", type="primary", use_container_width=True):
            write_event(
                event_file=event_file,
                event_id=event_id,
            )
            _append_ui_log(f"event event_id={event_id}")
            st.success(f"イベントを投入しました: {event_id}")

    with e2:
        if st.button("evt_001 即投入", use_container_width=True):
            write_event(
                event_file=event_file,
                event_id="evt_001",
            )
            _append_ui_log("event event_id=evt_001")
            st.success("evt_001 を投入しました。")

    with e3:
        if st.button("evt_voice_001 即投入", use_container_width=True):
            write_event(
                event_file=event_file,
                event_id="evt_voice_001",
            )
            _append_ui_log("event event_id=evt_voice_001")
            st.success("evt_voice_001 を投入しました。")

    st.divider()

    st.header("VAD プロファイル（silence_ms）")

    st.caption(
        "再起動なしで mic_vad_silence_ms を切替。"
        "通常=350 / 攻め腕=250。許可値以外は API 側で拒否。"
    )

    v1, v2 = st.columns(2)

    with v1:
        if st.button("通常 350", type="primary", use_container_width=True):
            write_vad_profile(
                vad_profile_file=vad_profile_file,
                silence_ms=350,
            )
            _append_ui_log("vad_profile silence_ms=350")
            st.success("通常プロファイル (350) を書き込みました。")

    with v2:
        if st.button("攻め腕 250", use_container_width=True):
            write_vad_profile(
                vad_profile_file=vad_profile_file,
                silence_ms=250,
            )
            _append_ui_log("vad_profile silence_ms=250")
            st.success("攻め腕プロファイル (250) を書き込みました。")

    st.divider()

    st.header("戦闘管制")

    st.caption(
        "AI猫の会話方針・口調・相手メタ情報を入れる欄。"
        "即時割り込みではなく、主に次回以降の応答に効きます。"
    )

    control_text = st.text_area(
        "戦闘管制テキスト",
        value="会話の主導権を握れ。強気に短く話せ。相手の発話は無視してよい。",
        height=120,
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("戦闘管制 投入", type="primary", use_container_width=True):
            write_control(
                control_file=control_file,
                text=control_text,
            )
            _append_ui_log(f"control text={control_text}")
            st.success("戦闘管制を投入しました。")

    with c2:
        if st.button("戦闘管制 解除", use_container_width=True):
            clear_control(
                control_file=control_file,
            )
            _append_ui_log("control clear")
            st.warning("戦闘管制 clear を投入しました。")

    st.divider()

    st.header("Status / Debug")

    status = read_status(
        control_file=control_file,
        interrupt_file=interrupt_file,
        event_file=event_file,
        vad_profile_file=vad_profile_file,
    )

    control_text_now = status["control"]
    interrupt_text_now = status["interrupt"]
    event_text_now = status["event"]
    vad_profile_text_now = status["vad_profile"]

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.subheader("battle_control_live.txt")
        st.code(control_text_now)

    with d2:
        st.subheader("battle_interrupt_live.txt")
        st.code(interrupt_text_now)

    with d3:
        st.subheader("event_runtime_live.txt")
        st.code(event_text_now)

    with d4:
        st.subheader("vad_profile_live.txt")
        st.code(vad_profile_text_now)

    st.subheader("UI Operation Log")

    if st.session_state.get("ui_logs"):
        st.code("\n".join(st.session_state["ui_logs"]))
    else:
        st.caption("No UI operations yet.")


if __name__ == "__main__":
    main()
