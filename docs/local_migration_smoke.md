# Local Migration Smoke Memo

## 完了日
2026-05-01

## 到達点

Windowsローカル環境で以下を確認済み。

- WAV入力E2E OK
- mic bridge単体 OK
- realtime streaming OK
- realtime streaming pipe BG OK
- mic入力E2E 1コマンド化 OK

## 固定仕様

- audio_ms = SSOT
- step_ms = 40
- chunk_len_ms = 400
- fps = 25

## 成功ログ要点

### WAV E2E

session_id: sess_audio_input_e2e_local_002

- response_audio_bytes: 389764
- tool_calls: 1
- target_frames: 203
- global_png_n: 203
- final_mp4: OK

### mic bridge

session_id: mic_test_001

- input_audio_ms: 5000
- response_audio_bytes: 39842
- tool_calls: 1

### realtime streaming

session_id: sess_realtime_pipe_bg_local_001_debug

- audio_ms: 3000
- chunks: 8
- target_frames: 75
- final_mp4: OK

### realtime streaming pipe BG

session_id: sess_realtime_pipe_bg_local_002

- audio_ms: 3000
- chunks: 8
- target_frames: 75
- pipe_bg.mp4: OK

### mic入力E2E

session_id: sess_mic_input_e2e_local_001

- response_audio_bytes: 48962
- tool_calls: 1
- mouth_frames_n: 23
- expr_events_n: 2
- target_frames: 26
- global_png_n: 26
- final_mp4: OK

## ローカル実行時の注意

### APIキー

GEMINI_API_KEY を使用する。

PowerShell:

```powershell
setx GEMINI_API_KEY "YOUR_KEY"
setx GOOGLE_API_KEY ""

VSCode再起動後に反映。

venv追加依存
pip install simplejson sounddevice
ffmpeg

Windows本体にffmpegをインストールし、PATHを通す。

mic入力E2E固定コマンド
python scripts/live_runtime/run_mic_input_e2e_smoke.py `
  --session_id sess_mic_input_e2e_local_001 `
  --m1_repo_root C:\dev\M1_LLM_To_M2_TTS_united `
  --m3_repo_root C:\dev\M3_Live_API_1_united `
  --m0_repo_root C:\dev\M0_session_renderer_final_1 `
  --m35_repo_root C:\dev\M3.5_final `
  --pose_json C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json `
  --bg_video C:\dev\M3.5_final\in\with1.mp4 `
  --duration_s 5 `
  --clean