from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from scripts.live_runtime.battle_runtime_admin_api import (
    DEFAULT_CONTROL_FILE,
    DEFAULT_INTERRUPT_FILE,
    DEFAULT_EVENT_FILE,
    DEFAULT_VAD_PROFILE_FILE,
    DEFAULT_OBS_CONFIG_FILE,
    DEFAULT_OBS_LOCAL_CONFIG_FILE,
    DEFAULT_LEADERSHIP_CONTROL_TEXT,
    DEFAULT_LEADERSHIP_INTERRUPT_TEXT,
    DEFAULT_LEADERSHIP_OPEN_TEXT,
    DEFAULT_NORMAL_CONVERSATION_TEXT,
    DEFAULT_ZOOM_LEADERSHIP_CONTROL_TEXT,
    clear_control,
    read_obs_admin_state,
    read_status,
    write_control,
    write_event,
    write_interrupt,
    write_leadership_open,
    write_leadership_start,
    write_zoom_leadership_start,
    write_mic_gate,
    write_normal_conversation,
    write_obs_background,
    write_obs_bgm,
    write_atefuri_enabled,
    write_atefuri_zoom,
    write_smith_effect,
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
        "live txt file 経由で操作する管理者UI。"
        "OBS 背景静止画・BGM は WebSocket 直結。当てフリは発話開始で見せ消し"
        "（session_loop は sleep せず fire-and-forget。OFF で無効）。"
        "スミスは管理画面の明示ボタン（一斉表示＋音声フィルタ）。"
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

        obs_config_file_str = st.text_input(
            "obs_control_config.json",
            value=str(DEFAULT_OBS_CONFIG_FILE),
        )

        control_file = Path(control_file_str)
        interrupt_file = Path(interrupt_file_str)
        event_file = Path(event_file_str)
        vad_profile_file = Path(vad_profile_file_str)
        obs_config_file = Path(obs_config_file_str)
        obs_local_config_file = DEFAULT_OBS_LOCAL_CONFIG_FILE

        st.divider()

        st.write("Current paths")
        st.code(f"control:     {control_file}")
        st.code(f"interrupt:   {interrupt_file}")
        st.code(f"event:       {event_file}")
        st.code(f"vad_profile: {vad_profile_file}")
        st.code(f"obs_config:  {obs_config_file}")
        st.code(f"obs_local:   {obs_local_config_file}")

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

    st.header("OBS 背景静止画 / BGM")

    st.caption(
        "OBS WebSocket で背景静止画ソースと BGM メディアソースを切替。"
        "VirtualCam 内 BGV・AI PCM は対象外。"
        "接続失敗でも Live パイプラインは継続。"
        "当てフリは発話開始トリガ（この画面は ON/OFF と手動テスト）。"
    )

    obs_state = read_obs_admin_state(
        obs_config_file=obs_config_file,
        obs_local_config_file=obs_local_config_file,
    )
    probe = obs_state.get("probe") or {}
    ws_meta = obs_state.get("websocket") or {}
    sources = obs_state.get("sources") or {}

    if obs_state.get("config_error"):
        st.error(f"OBS config error: {obs_state['config_error']}")

    if probe.get("ok"):
        st.success(
            f"OBS connected: {probe.get('host')}:{probe.get('port')} "
            f"(obs {probe.get('obs_version')})"
        )
    else:
        st.warning(
            "OBS 未接続（Live は継続可）。"
            f" reason={probe.get('error') or 'unknown'}"
        )

    st.code(
        "host={host} port={port} password_set={pw}\n"
        "background_source={bg}\n"
        "bgm_source={bgm}\n"
        "atefuri_normal={an}\n"
        "atefuri_zoom={az}\n"
        "smith_clones={sc}\n"
        "smith_audio={sa} filter={sf}".format(
            host=ws_meta.get("host"),
            port=ws_meta.get("port"),
            pw=ws_meta.get("password_set"),
            bg=sources.get("background_image"),
            bgm=sources.get("bgm_media"),
            an=sources.get("atefuri_normal"),
            az=sources.get("atefuri_zoom"),
            sc=", ".join((obs_state.get("smith") or {}).get("clone_sources") or [])
            or "(none)",
            sa=(obs_state.get("smith") or {}).get("audio_source") or "",
            sf=(obs_state.get("smith") or {}).get("filter_name") or "",
        )
    )

    backgrounds = obs_state.get("backgrounds") or []
    bgm_items = obs_state.get("bgm") or []

    o1, o2 = st.columns(2)

    with o1:
        st.subheader("背景静止画")
        if not backgrounds:
            st.caption("catalog 空（in/obs_control_config.json を確認）")
        else:
            bg_labels = {
                f"{x['label']} ({x['id']})": x["id"] for x in backgrounds
            }
            bg_choice = st.selectbox(
                "背景を選択",
                options=list(bg_labels.keys()),
                key="obs_bg_select",
            )
            if st.button("背景を切替", type="primary", use_container_width=True):
                result = write_obs_background(
                    item_id=bg_labels[bg_choice],
                    obs_config_file=obs_config_file,
                    obs_local_config_file=obs_local_config_file,
                )
                if result.get("ok"):
                    _append_ui_log(
                        f"obs_background id={result.get('id')} "
                        f"source={result.get('source')}"
                    )
                    st.success(
                        f"背景切替 OK: {result.get('id')} → {result.get('path')}"
                    )
                else:
                    _append_ui_log(
                        f"obs_background FAIL id={result.get('id')} "
                        f"err={result.get('error')}"
                    )
                    st.error(f"背景切替失敗: {result.get('error')}")

    with o2:
        st.subheader("BGM")
        if not bgm_items:
            st.caption("catalog 空（in/obs_control_config.json を確認）")
        else:
            bgm_labels = {
                f"{x['label']} ({x['id']})": x["id"] for x in bgm_items
            }
            bgm_choice = st.selectbox(
                "BGM を選択",
                options=list(bgm_labels.keys()),
                key="obs_bgm_select",
            )
            if st.button("BGM を切替", type="primary", use_container_width=True):
                result = write_obs_bgm(
                    item_id=bgm_labels[bgm_choice],
                    obs_config_file=obs_config_file,
                    obs_local_config_file=obs_local_config_file,
                )
                if result.get("ok"):
                    _append_ui_log(
                        f"obs_bgm id={result.get('id')} "
                        f"source={result.get('source')}"
                    )
                    st.success(
                        f"BGM 切替 OK: {result.get('id')} → {result.get('path')}"
                    )
                else:
                    _append_ui_log(
                        f"obs_bgm FAIL id={result.get('id')} "
                        f"err={result.get('error')}"
                    )
                    st.error(f"BGM 切替失敗: {result.get('error')}")

    st.subheader("当てフリ（通常⇔ドアップ 見せ消し）")
    st.caption(
        "OBS に事前配置した通常ソースとドアップソースの可視性だけ切替。"
        "拡大は OBS 側の配置。トリガは AI 発話開始（first_audio）。"
        "default ON。OFF にすると発話開始でも切替しない。"
    )
    atefuri = obs_state.get("atefuri") or {}
    atefuri_on = bool(atefuri.get("enabled", True))
    st.write(
        f"now={'ON' if atefuri_on else 'OFF'} "
        f"(config default={'ON' if atefuri.get('enabled_default', True) else 'OFF'}) "
        f"scene={atefuri.get('scene') or '(current program)'}"
    )
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        if st.button("当てフリ ON", type="primary", use_container_width=True):
            result = write_atefuri_enabled(
                enabled=True,
                obs_config_file=obs_config_file,
                obs_local_config_file=obs_local_config_file,
            )
            _append_ui_log("atefuri enabled=ON")
            st.success("当てフリ ON（次の発話開始でズームソース）")
    with t2:
        if st.button("当てフリ OFF", use_container_width=True):
            result = write_atefuri_enabled(
                enabled=False,
                obs_config_file=obs_config_file,
                obs_local_config_file=obs_local_config_file,
            )
            restore = result.get("restore") or {}
            if restore.get("ok"):
                _append_ui_log("atefuri enabled=OFF restore=ok")
                st.warning("当てフリ OFF（通常ソースへ戻した）")
            else:
                _append_ui_log(
                    f"atefuri enabled=OFF restore_err={restore.get('error')}"
                )
                st.warning(
                    "当てフリ OFF を記録。"
                    f" 戻し: {restore.get('error') or 'skipped'}"
                )
    with t3:
        if st.button("テスト ズーム", use_container_width=True):
            result = write_atefuri_zoom(
                zoomed=True,
                obs_config_file=obs_config_file,
                obs_local_config_file=obs_local_config_file,
            )
            if result.get("ok"):
                _append_ui_log("atefuri test zoomed=1")
                st.success(
                    f"ズーム表示: {result.get('zoom_source')} "
                    f"(hide {result.get('normal_source')})"
                )
            else:
                _append_ui_log(f"atefuri test FAIL {result.get('error')}")
                st.error(f"ズーム失敗: {result.get('error')}")
    with t4:
        if st.button("テスト 通常", use_container_width=True):
            result = write_atefuri_zoom(
                zoomed=False,
                obs_config_file=obs_config_file,
                obs_local_config_file=obs_local_config_file,
            )
            if result.get("ok"):
                _append_ui_log("atefuri test zoomed=0")
                st.success(
                    f"通常表示: {result.get('normal_source')} "
                    f"(hide {result.get('zoom_source')})"
                )
            else:
                _append_ui_log(f"atefuri test FAIL {result.get('error')}")
                st.error(f"通常戻し失敗: {result.get('error')}")

    st.subheader("エージェントスミス（クローン＋音声フィルタ）")
    st.caption(
        "OBS 事前配置クローンを一斉表示し、AI拾い音声ソースのフィルタを ON。"
        "BGM には掛けない。transform / 増殖加速なし。session_loop 非触。"
        "開始＝表示＋フィルタON／リセット＝非表示＋フィルタOFF。"
    )
    smith = obs_state.get("smith") or {}
    st.write(
        f"clones={', '.join(smith.get('clone_sources') or []) or '(none)'} "
        f"audio={smith.get('audio_source') or '(unset)'} "
        f"filter={smith.get('filter_name') or '(unset)'} "
        f"scene={smith.get('scene') or '(current program)'}"
    )
    s1, s2 = st.columns(2)
    with s1:
        if st.button("スミス開始", type="primary", use_container_width=True):
            result = write_smith_effect(
                active=True,
                obs_config_file=obs_config_file,
                obs_local_config_file=obs_local_config_file,
            )
            if result.get("ok"):
                _append_ui_log(
                    "smith start clones="
                    + ",".join(result.get("applied") or [])
                    + f" filter={result.get('filter_name')}"
                )
                st.success(
                    f"スミス ON: clones={', '.join(result.get('applied') or [])} "
                    f"filter={result.get('audio_source')}/{result.get('filter_name')}"
                )
            else:
                _append_ui_log(f"smith start FAIL {result.get('error')}")
                st.error(f"スミス開始失敗: {result.get('error')}")
    with s2:
        if st.button("スミス リセット", use_container_width=True):
            result = write_smith_effect(
                active=False,
                obs_config_file=obs_config_file,
                obs_local_config_file=obs_local_config_file,
            )
            if result.get("ok"):
                _append_ui_log("smith reset ok")
                st.warning("スミス OFF（クローン非表示＋フィルタOFF）")
            else:
                _append_ui_log(f"smith reset FAIL {result.get('error')}")
                st.error(f"スミスリセット失敗: {result.get('error')}")

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
