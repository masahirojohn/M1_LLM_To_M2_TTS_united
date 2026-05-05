# Local OBS Realtime Step1 Memo

## 到達点

Windowsローカル環境で、以下まで確認済み。

```text
mic input
→ Gemini Live API
→ audio_response.pcm / tool_call
→ mouth.json / expr.json
→ 400ms chunks
→ M0 chunkごとFG生成
→ persistent virtualcam
→ Unity Video Capture
→ OBS映像表示

audio_response.pcm
→ VB Cable
→ OBS音声ミキサー反応

成功確認済み
virtualcam外側常駐
複数turn連続実行
frame_offsetによるFG連番維持
OBS映像3回表示
音声3回再生

代表成功ログ：

[run_mic_input_obs_realtime_step1_loop][DONE]
  ok_turns: 3/3
  total_frames: 103
  watch_fg_dir: C:\dev\M1_LLM_To_M2_TTS_united\out\obs_stream_step1_loop\fg
実行コマンド
python scripts/live_runtime/run_mic_input_obs_realtime_step1_loop.py `
  --session_id sess_obs_realtime_step1_pcam_001 `
  --m1_repo_root C:\dev\M1_LLM_To_M2_TTS_united `
  --m3_repo_root C:\dev\M3_Live_API_1_united `
  --m0_repo_root C:\dev\M0_session_renderer_final_1 `
  --m35_repo_root C:\dev\M3.5_final `
  --pose_json C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json `
  --bg_video C:\dev\M3.5_final\in\with1.mp4 `
  --duration_s 5 `
  --audio_device 15 `
  --turns 3 `
  --gap_s 0.5 `
  --persistent_cam
成功条件
[realtime_step1_loop] start persistent virtualcam
[virtualcam_persistent][OK] device=Unity Video Capture

turn=1 → OBS映像表示
turn=2 → OBS映像表示
turn=3 → OBS映像表示

[run_mic_input_obs_realtime_step1_loop][DONE]
  ok_turns: 3/3
重要仕様
step_ms = 40
chunk_len_ms = 400
fps = 25
audio_ms がSSOT

追加・修正ファイル
scripts/live_runtime/run_virtualcam_persistent.py
scripts/live_runtime/run_mic_input_obs_persistent_smoke.py
scripts/live_runtime/run_mic_input_obs_realtime_step1.py
scripts/live_runtime/run_mic_input_obs_realtime_step1_loop.py
scripts/live_runtime/run_audio_input_m0_all_chunks_smoke.py
scripts/live_runtime/run_mic_input_e2e_smoke.py
現状の制約
Live API応答生成はまだturn単位
audio_response.pcmはファイル再生
M0はchunkごと subprocess
完全な常時マイク入力・常時Live API接続ではない