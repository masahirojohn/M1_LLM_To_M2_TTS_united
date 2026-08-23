# Zoom 第三者運営手順（この機）

抜き出し元: `docs/PROGRESS.md` の「Zoom 運用」および Phase Z2b 配線。  
**矛盾したら PROGRESS を正とする。** このファイルは新規設計ではない。

対象: 第三者がこの AI/OBS 専用機で、遠隔 Zoom 参加者と会話を通すこと。  
コード変更・図A変更・B/O 再開はしない。

---

## 前提

- この機は **AI / OBS 専用**。人間は Zoom / SNS にこの機から参加しない。
- 本番の会話入力は **遠隔の相手声**。USB mic はローカル検証のみ（`mic=1` だけなら Banana 不要）。
- Voicemeeter は **Banana**（`voicemeeterpro.exe`）。**Standard は使わない。**
- 主導権の mute + interrupt は Zoom では使わない。

---

## 毎回（起動順）

1. **Voicemeeter Banana を起動して開いたまま**（`voicemeeterpro.exe`）。
2. Banana: 中央 VIRTUAL INPUTS の左（`Voicemeeter VAIO` / Voicemeeter Input）だけ **B1 点灯**。B2 / A2 / A3 はオフ。**CABLE を Hardware In に足さない。**
3. 必要なら A1 = ヘッドホン（遠隔モニター）。USB mic へ戻さない。
4. M1 セッションを起動する（下のコマンド）。ログに `[virtualcam_persistent][OK] device=Unity Video Capture` が出るまで待つ。
5. Zoom を開き、毎回 Mic / Speaker / カメラを確認する。先に Zoom を開いているならカメラをオフ→オン。

---

## Zoom 設定

| 項目 | 値 |
| --- | --- |
| Microphone | **CABLE Output** |
| Speaker | **Voicemeeter Input (VAIO)** |
| カメラ | **Unity Video Capture**（`virtualcam OK` のあと） |

使わない Speaker: CABLE Input / システムと同じ / VAIO3 / AUX / In 1–5。

---

## 配線（この機スナップショット 2026-08-22）

番号は **再クエリ必須**。下表は Banana 導入後の一例。旧 CABLE Input=6 は使わない（Banana 後に `Voicemeeter Out A4` へずれた）。

```text
[出]
  M1 --audio_device     → CABLE Input
  Zoom Microphone       → CABLE Output

[入]
  Zoom Speaker          → Voicemeeter Input (VAIO)     NEVER CABLE Input
  Banana: VAIO strip    → B1 のみ。CABLE を Hardware In に足さない
  M1 --mic_input_device → Voicemeeter Out B1           NEVER CABLE Output
```

| 役割 | スナップショット | 禁止 |
| --- | ---: | --- |
| CABLE Input（AI 出） | **23** | 旧 6 を決め打ちしない |
| Voicemeeter Out B1（M1 mic） | **9** | CABLE Output を mic にしない |
| CABLE Output（Zoom Mic） | **8** | `--mic_input_device` にしない |
| Voicemeeter Input VAIO（Zoom Speaker） | **19** | Speaker を 23 / VAIO3(14) にしない |
| USB mic | 1 | ローカル検証のみ |
| ヘッドホン | 15 | 任意モニター |

再クエリ:

```powershell
& C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"
```

MME 行で名前を照合し、`$cableIn`（CABLE Input）と `$micIn`（Voicemeeter Out B1）を置き換える。

---

## 禁止

- CABLE Output を `--mic_input_device` にする（AI 声が戻る）
- Zoom Speaker = CABLE Input（遠隔声が USB / B1 に乗らず idle になる）
- 主導権 mute + interrupt（Zoom では使わない）
- session_loop で PCM ミックスする
- 部屋スピーカー拾い
- Voicemeeter Standard

---

## 現行 JP Zoom コマンド（Z2b 同形）

番号は再クエリしてから埋める。以下はスナップショット 23 / 9。

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$session_id = "sess_z2b_jp_subj_$ts"
$log = "C:\dev\M1_LLM_To_M2_TTS_united\logs\$session_id.log"
$py = "C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe"
$cableIn = 23
$micIn = 9
Set-Location C:\dev\M1_LLM_To_M2_TTS_united
& $py `
  scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py `
  --session_id $session_id `
  --m1_repo_root C:\dev\M1_LLM_To_M2_TTS_united `
  --m3_repo_root C:\dev\M3_Live_API_1_united `
  --m0_repo_root C:\dev\M0_session_renderer_final_1 `
  --m35_repo_root C:\dev\M3.5_final `
  --prompt_dir C:\dev\M1_LLM_To_M2_TTS_united\configs\prompts `
  --pose_json C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json `
  --bg_video C:\dev\M3.5_final\in\with1.mp4 `
  --turns 4 `
  --gap_s 1.0 `
  --turn_idle_wait_s 2.5 `
  --turn_first_audio_timeout_s 60.0 `
  --mic_send_max_s 25.0 `
  --mic_input_device $micIn `
  --audio_device $cableIn `
  --inline_emo_tag_mode `
  --stream_mouth_m0_chunk_len_ms 120 `
  --audio_player_initial_buffer_ms 300 `
  --audio_player_rebuffer_target_ms 240 `
  --mic_vad_end_enabled `
  --mic_vad_rms_threshold 0.010 `
  --mic_vad_end_rms_threshold 0.020 `
  --mic_vad_min_voice_ms 240 `
  --mic_vad_silence_ms 350 `
  --mic_vad_min_listen_ms 800 `
  --battle_talkover_cut_in_on_interrupt `
  --battle_interrupt_file_immediate_send `
  --battle_interrupt_file C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_live.txt `
  --battle_interrupt_file_poll_s 0.05 `
  --battle_control_file C:\dev\M1_LLM_To_M2_TTS_united\in\battle_control_live.txt `
  --battle_control_file_poll_s 0.05 `
  --event_runtime_file C:\dev\M1_LLM_To_M2_TTS_united\in\event_runtime_live.txt `
  --event_runtime_file_poll_s 0.05 `
  --event_catalog_json C:\dev\M1_LLM_To_M2_TTS_united\in\event_catalog.json `
  --bg_override_file C:\dev\M1_LLM_To_M2_TTS_united\in\bg_override_live.txt `
  --no-fast_inmemory `
  --m0_worker_n 2 `
  --clean `
  --clean_fg `
  *>&1 | Tee-Object -FilePath $log
Write-Host "SESSION=$session_id LOG=$log CABLE_IN=$cableIn MIC_IN=$micIn"
```

起動後チェック: 遠隔発話で `has_spoken=True` と `input_audio_ms>0`。idle 連発かつ 0ms は Zoom Speaker が CABLE Input になっていないか確認する。

---

## EN（任意・同じ配線）

配線と Banana / Zoom 設定は JP と同じ。起動差分だけ:

- `--m3_repo_root C:\dev\M3_Live_API_1_english`
- `--m0_repo_root C:\dev\M0_session_renderer_final_1_english`
- `--prompt_dir C:\dev\M1_LLM_To_M2_TTS_united\configs\prompts_en`
- `--output_audio_transcription`
- `--stream_mouth_gt_glob data/knn_db/en_10files.phoneme_gt.f1f2.json`

pose / bg / m35 は JP 絶対パスのまま。
