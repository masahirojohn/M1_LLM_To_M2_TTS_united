docs/live_runtime_smoke.md


# Live Runtime Smoke Tests（固定版）

本ドキュメントは、Gemini Live API 導入後の **動作確認コマンド（スモークテスト）固定版**です。  
今後の開発（streaming本番化・watcher統合）に入る前の **基準状態（SSOT）** として使用します。

---

# 0. 前提

```text
step_ms       = 40
chunk_len_ms  = 400
fps           = 25


1. 標準E2Eスモーク（合成あり）
コマンド
cd /workspaces/M1_LLM_To_M2_TTS_united

PYTHONPATH=/workspaces/M1_LLM_To_M2_TTS_united/src:/workspaces/M3_Live_API_1_united/src:$PYTHONPATH \
python scripts/run_runtime_session_live_e2e.py \
  --session_id sess_live_runtime_e2e_001 \
  --pose_json /workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json \
  --normal_bg_video /workspaces/M3.5_final/in/with1.mp4 \
  --mouth_model gemini-2.5-flash-native-audio-preview-12-2025 \
  --mouth_prompt "元気よく自己紹介して！短めで。" \
  --mouth_dev_stop_s 20 \
  --target_audio_ms 3000 \
  --expr_model gemini-3.1-flash-live-preview \
  --expr_prompt "短く自己紹介して。最初に必ず set_emotion を呼び、感情は smile にしてください。" \
  --expr_dev_stop_s 12 \
  --stop_after_first_tool_s 3

期待値
audio_ms     = 3000
plan_chunks  = 8
処理時間      = 1分以内


2. 高速検証（--skip_bridge）
コマンド
python scripts/run_runtime_session_live_e2e.py \
  --session_id sess_live_runtime_e2e_fast_001 \
  --pose_json /workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json \
  --normal_bg_video /workspaces/M3.5_final/in/with1.mp4 \
  --mouth_dev_stop_s 20 \
  --target_audio_ms 3000 \
  --expr_dev_stop_s 12 \
  --stop_after_first_tool_s 3 \
  --skip_bridge

期待値
audio_ms     = 3000
plan_chunks  = 8
処理時間      = 約20秒


3. Streaming v0（3秒）
コマンド
python scripts/run_live_runtime_streaming.py \
  --session_id sess_live_streaming_v0_001 \
  --pose_json /workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json \
  --normal_bg_video /workspaces/M3.5_final/in/with1.mp4 \
  --target_audio_ms 3000

期待値
audio_ms     = 3000
plan_chunks  = 8
ran_chunks   = 8


4. Streaming v0（5秒）
コマンド
python scripts/run_live_runtime_streaming.py \
  --session_id sess_live_streaming_v0_005s \
  --pose_json /workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json \
  --normal_bg_video /workspaces/M3.5_final/in/with1.mp4 \
  --target_audio_ms 5000

期待値
audio_ms     = 5000
plan_chunks  = 13
ran_chunks   = 13


5. 成功時の出力パス
例（3秒 streaming）：
out/live_runtime_streaming/sess_live_streaming_v0_001/
  ├─ mouth_live/
  ├─ expr_live/
  ├─ m0_in/
  │   ├─ mouth.json
  │   ├─ mouth.clamped.json
  │   └─ expr.json
  ├─ sessions/
  ├─ manifests/
  ├─ plans/
  └─ bridge_streaming_out/


6. 現在の固定ファイル（重要）
以下は修正済み前提で固定：
scripts/run_runtime_session_live_e2e.py
scripts/dev_live_to_m3_raw.py
scripts/dev_live_to_expression_json.py
scripts/run_live_runtime_streaming.py


7. 重要な仕様（SSOT）
audio_ms = SSOT
step_ms  = 40
chunk    = 400ms単位


8. ガード（必読）
推測でコード修正しない
SSOT（audio_ms）を崩さない
chunk境界（400ms）を変更しない
streaming v0 はあくまで疑似（本番は別）

以下を docs/live_runtime_realtime_smoke.md に追記してください。
追記箇所
「Live Runtime Realtime Smoke Test」の末尾
追記内容
# Streaming v0 Smoke（400ms逐次M0/M3.5）

## 目的

Live API 出力をもとに、400ms単位で M0 → M3.5 を逐次実行し、最後に確認用MP4を1本出力する。
---

## 実行コマンド

```bash
cd /workspaces/M1_LLM_To_M2_TTS_united

PYTHONPATH=/workspaces/M1_LLM_To_M2_TTS_united/src:/workspaces/M3_Live_API_1_united/src:$PYTHONPATH \
python scripts/live_runtime/run_live_runtime_realtime_streaming.py \
  --session_id sess_live_runtime_realtime_streaming_smoke_001 \
  --pose_json /workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json \
  --normal_bg_video /workspaces/M3.5_final/in/with1.mp4 \
  --target_audio_ms 3000 \
  --no_realtime_sleep


期待ログ
[run_live_runtime_realtime_streaming][OK]
  audio_ms     : 3000
  chunks       : 8
  target_frames: 75
  final_mp4    : /workspaces/M3.5_final/out/live_runtime_realtime_streaming/<session_id>/final/stage2/m3_5_composite.mp4


成功条件
M0:
frames 0..74 exist and are 4ch

M3.5:
chunks/000000〜000007/stage2/m3_5_composite.mp4 が出力される
final/stage2/m3_5_composite.mp4 が出力される
final MP4 が約3秒の正常な合成動画である


出力先
M1:
out/live_runtime_realtime_streaming/<session_id>/
  run_live_runtime_realtime_streaming.summary.json

M0:
out/live_runtime_realtime_streaming/<session_id>/
  fg_streaming/
  chunk_runs/fg_index.streaming.csv

M3.5:
out/live_runtime_realtime_streaming/<session_id>/
  chunks/000000/stage2/m3_5_composite.mp4
  ...
  chunks/000007/stage2/m3_5_composite.mp4
  final/stage2/m3_5_composite.mp4

注意
--no_realtime_sleep はスモーク高速化用。
実時間400ms間隔で流したい場合は外す。
---

# 5. Realtime Streaming 完全成功コマンド（SSOT）

## コマンド

cd /workspaces/M1_LLM_To_M2_TTS_united

PYTHONPATH=/workspaces/M1_LLM_To_M2_TTS_united/src:/workspaces/M3_Live_API_1_united/src:$PYTHONPATH \
python scripts/live_runtime/run_live_runtime_realtime_streaming.py \
  --session_id sess_final \
  --pose_json /workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json \
  --normal_bg_video /workspaces/M3.5_final/in/with1.mp4 \
  --target_audio_ms 3000

---

## 期待値（固定）

audio_ms       = 3000  
chunks         = 8  
target_frames  = 75  

FG:
  00000000.png ～ 00000074.png（連番欠損なし）

M0:
  各chunkごとに fg_local 出力 → global連結

M3.5:
  direct mode
  chunkごとMP4生成
  final MP4生成

---

## 成功ログ例（SSOT）

[run_live_runtime_realtime_streaming][OK]
  audio_ms     : 3000
  chunks       : 8
  target_frames: 75

---

## 注意（重要）

・このコマンドは「壊してはいけない基準」  
・以降の開発はこの結果を維持すること  
・常時worker化は別ファイルで実装すること（既存を改変しない）
---

# 6. Persistent Workers Safe Loop 成功確認

## コマンド概要

run_live_runtime_persistent_workers.py により、
既存成功版 run_live_runtime_realtime_streaming.py を session 単位で2回連続実行。

## 結果

sessions_n = 2

各session:

audio_ms       = 3000
step_ms        = 40
chunk_len_ms   = 400
fps            = 25
chunks         = 8
target_frames  = 75
returncode     = 0

final MP4 出力正常。
合成動画の目視確認も正常。

## 位置づけ

これは真の常時worker化ではなく、
既存成功版を壊さない安全な persistent loop 版。

短期方針:
M0 = subprocess のまま
M3.5 = chunk MP4 + final MP4確認

次フェーズ:
M0常駐化
OBS向けリアルタイム1本出力
