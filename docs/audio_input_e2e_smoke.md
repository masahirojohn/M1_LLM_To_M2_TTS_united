audio_input_e2e_smoke.md
概要

本ドキュメントは、Gemini Live API 音声入力 → 最終MP4生成までのE2Eスモーク手順を定義する。

対象：

・Codespacesで再現可能な音声入力パイプライン
・ローカル移行前の固定スモーク
パイプライン全体
WAV入力
↓
audio_stream_bridge
↓
audio_response.pcm + tool_calls
↓
M3 mouth解析
↓
mouth.json
↓
tool_call → expr.json
↓
chunk分割
↓
M0 全チャンクFG生成
↓
M3.5 BG合成
↓
MP4
実行コマンド（完全E2E）
cd /workspaces/M1_LLM_To_M2_TTS_united

PYTHONPATH=/workspaces/M1_LLM_To_M2_TTS_united/src:/workspaces/M3_Live_API_1_united/src:/workspaces/M3.5_final:$PYTHONPATH \
python scripts/live_runtime/run_audio_input_e2e_smoke.py \
  --session_id sess_audio_input_e2e_001 \
  --input_wav examples/test_input_16k_mono.fixed.wav \
  --pose_json /workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json \
  --bg_video /workspaces/M3.5_final/in/with1.mp4 \
  --duration_s 10 \
  --clean
成功条件
[run_audio_input_e2e_smoke][OK]

response_audio_bytes > 0
tool_calls >= 1
mouth_frames_n > 0
expr_events_n >= 2

global_png_n = target_frames

final_mp4 が生成される
出力構成
out/audio_input_e2e_smoke/<session_id>/

01_audio_input_smoke_pipeline/
02_audio_input_to_chunks/
03_m0_all_chunks/
04_m35_compose/
run_audio_input_e2e_smoke.summary.json

最終出力：

04_m35_compose/m3_5_composite.mp4
各ステージ詳細
1. Live API 音声入力
scripts/live_runtime/audio_stream_bridge.py

出力：

audio_response.pcm
live_tool_calls.json
2. Mouth解析
audio_response.pcm
→ MouthStreamerOC
→ mouth_timeline.formant.raw.json
→ kNN変換
→ mouth.json
3. Expression生成
live_tool_calls.json
→ expr.json

補正：

t_ms clamp:
min(t_ms, audio_ms - 40)
4. Chunk分割
400ms単位
step_ms = 40
fps = 25
5. M0
全チャンク描画
→ global FG PNG (連番)
6. M3.5
FG PNG + BG動画
→ 合成MP4
注意事項
・audio_ms が SSOT
・step_ms = 40 固定
・chunk_len_ms = 400 固定
・fps = 25 固定

・Live API:
  入力 → PCM16 16kHz
  出力 → PCM16 24kHz

・Codespacesではマイク不可
  → WAV/PCMでテスト
ローカル移行後のタスク
① マイク入力対応
--mode mic

使用：

sounddevice
② リアルタイム化
audio_stream_bridge
→ chunk_orchestrator
→ M0 streaming
→ M3.5 streaming
③ OBS接続
M3.5出力 → OBS
④ 最終構成
Microphone
→ Live API
→ mouth / expression
→ chunk streaming
→ M0
→ M3.5
→ OBS配信
ガード
・推測で回答しない
・実ログで確認する
・SSOT(audio_ms)を崩さない
・既存成功コードを壊さない
現在の到達点
Codespacesでの完全E2E成功
（音声入力 → MP4）