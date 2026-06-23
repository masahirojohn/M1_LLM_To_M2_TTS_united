# local_obs_realtime_low_latency.md

# 目的

OBSリアルタイム配信向けの低遅延化。

主目的：

batch pipeline
↓
streaming pipeline

への構造変換。

特に：

stream PCM
↓
stream mouth
↓
stream M0
↓
stream OBS

を実現すること。

---

# 現在の到達点

## 完了済

### 1. M0 persistent worker 化

subprocess/chunk 起動廃止。

TCP worker 常駐化済。

---

### 2. persistent virtualcam

UnityCapture 常駐。

OBSリアルタイム描画成功。

---

### 3. persistent audio player

audio subprocess/chunk 廃止。

---

### 4. stream PCM

audio_stream_bridge.py が：

pcm_stream_chunks/*.pcm

を逐次生成。

---

### 5. stream mouth

audio_stream_bridge.py 内で：

formant raw
↓
mouth.json

逐次更新。

KNN subprocess は廃止済。

in-process KNN 化済。

---

### 6. stream M0

watcher が：

mouth.json 更新
↓
120ms chunk
↓
M0 render

を逐次実行。

---

### 7. stream OBS

FG direct output
↓
persistent virtualcam

で OBS へ逐次送信。

---

# 現在の推奨設定

```text
model = gemini-3.1-flash-live-preview
api_version = v1alpha

duration_s = 1.5
mic_send_max_s = 0.6

early_response_trigger_s = 0.0

stream_mouth_m0_chunk_len_ms = 120
stream_mouth_knn_min_interval_s = 0.0



# OBS Realtime Low Latency Session Loop

## 目的

Live API session常駐 + stream mouth + stream M0 + OBS 出力により、初回FG表示遅延を削減する。

## 現在の到達点

1turn低遅延コアは成立。

最新ログ例：

```text
first audio        = 0.874 sec
mouth_json written = 0.973 sec
mouth_json seen    = 1.017 sec
frames ready       = 1.066 sec
first M0 done      = 1.271 sec

これにより、first FG は 1秒台前半に到達。

重要設定
model = gemini-3.1-flash-live-preview
api_version = v1alpha
mic_send_max_s = 0.6
stream_mouth_m0_chunk_len_ms = 120
M0 watcher start = turn_start_perf 設定後
1turn stable command
C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe `
  scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py `
  --session_id sess_live_session_loop_eventwake_001 `
  --m1_repo_root C:\dev\M1_LLM_To_M2_TTS_united `
  --m3_repo_root C:\dev\M3_Live_API_1_united `
  --m0_repo_root C:\dev\M0_session_renderer_final_1 `
  --m35_repo_root C:\dev\M3.5_final `
  --pose_json C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json `
  --bg_video C:\dev\M3.5_final\in\with1.mp4 `
  --turns 1 `
  --gap_s 0.8 `
  --turn_idle_wait_s 0.8 `
  --turn_first_audio_timeout_s 5.0 `
  --mic_send_max_s 0.6 `
  --audio_device 15 `
  --stream_mouth_m0_chunk_len_ms 120 `
  --clean `
  --clean_fg
現在の未完了

2turn以上の同一 Live API session 継続では、2回目が tool_call のみ返って audio が返らないケースがある。

そのため、2turn連続化は別途検証中。


## 次の具体案：2turn連続

次は **1turn成功構成を保ったまま、turnごとに mouth/M0 watcher を分離**します。

今の問題は、1turn目の mouth/M0/音声残りが2turn目に混ざることです。  
次の方針：

```text
turnごとに stream_mouth_dir / mouth.json / pcm_chunks_dir を分ける
turnごとに M0 watcher を起動
Live API session は常駐

その後の進展：
````md
# local_obs_realtime_low_latency.md

# Realtime OBS Low Latency Session Loop

## 目的

以下の realtime streaming pipeline の低遅延化。

```text
mic
↓
Gemini Live API
↓
stream mouth
↓
M0 persistent render
↓
virtualcam
↓
OBS
````

特に：

```text
2turn以降でも
1秒前後の応答を維持
```

を目標とする。

---

# 現在の到達点

達成済：

```text
tool_callなし
↓
inline emo_id
↓
stream mouth
↓
M0 persistent render
↓
OBS realtime
```

OBSで：

```text
3turn連続
音声＋映像
emo_id別表情
```

確認済。

---

# 重要な結論

## 主遅延要因

以前の構成：

```text
Live API
↓
tool_call(set_emotion)
↓
tool_response
↓
audio generation
```

では：

```text
2turn以降で
1.5秒前後
```

まで悪化。

現在：

```text
tool_call除去
↓
inline emo_id
```

へ変更。

結果：

```text
turn2/3 ≈ 0.6〜0.9 sec
```

まで改善。

---

# 現在の構成

```text
mic
↓
persistent Live API session
↓
_receive_loop
↓
stream mouth json
↓
M0 incremental render
↓
virtualcam persistent
↓
OBS
```

---

# 現在の重要ファイル

## session loop 本体

```text
scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py
```

---

## M0 watcher

```text
scripts/live_runtime/run_mic_input_obs_realtime_step1.py
```

重要関数：

```python
_watch_stream_mouth_and_render_m0()
```

---

## audio bridge

```text
scripts/live_runtime/audio_stream_bridge.py
```

---

## virtualcam

```text
scripts/live_runtime/virtualcam_persistent.py
```

---

# 現在の重要オプション

## inline emo mode

```text
--inline_emo_tag_mode
```

tool_callを使わず、
CLI側 emo_id を expr に直接反映。

---

## turn別 emo_id

```text
--inline_emo_ids 9_1,1_1,9_2
```

例：

```text
turn1 -> sad
turn2 -> happy/normal
turn3 -> sad
```

---

## audio冒頭drop

```text
--drop_initial_audio_ms 40
```

目的：

```text
[emo:...] 読み上げノイズ対策
```

現在は：

```text
40ms
```

で安定。

---

# 成功コマンド（現在の推奨）

```powershell
C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe `
  scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py `
  --session_id sess_live_session_loop_inline_emo_ids_001 `
  --m1_repo_root C:\dev\M1_LLM_To_M2_TTS_united `
  --m3_repo_root C:\dev\M3_Live_API_1_united `
  --m0_repo_root C:\dev\M0_session_renderer_final_1 `
  --m35_repo_root C:\dev\M3.5_final `
  --pose_json C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json `
  --bg_video C:\dev\M3.5_final\in\with1.mp4 `
  --turns 3 `
  --gap_s 1.5 `
  --turn_idle_wait_s 0.8 `
  --turn_first_audio_timeout_s 5.0 `
  --mic_send_max_s 0.6 `
  --audio_device 15 `
  --stream_mouth_m0_chunk_len_ms 120 `
  --inline_emo_tag_mode `
  --inline_emo_ids 9_1,1_1,9_2 `
  --drop_initial_audio_ms 40 `
  --response_trigger "短く返答してください。" `
  --clean `
  --clean_fg
```

---

# 期待ログ

```text
[session_loop][inline_emo] turn=1 emo_id=9_1
[session_loop][inline_emo] turn=2 emo_id=1_1
[session_loop][inline_emo] turn=3 emo_id=9_2
```

---

# 現在の性能

代表値：

```text
turn2_first_response_audio_chunk_sec ≈ 0.65〜0.9 sec
turn3_first_response_audio_chunk_sec ≈ 0.65〜0.9 sec
```

first FG：

```text
≈ 1.0〜1.2 sec
```

---

# 実装済み重要改善

## 1. M0 persistent worker

```text
m0_persistent_worker TCP server
```

---

## 2. virtualcam persistent

```text
Unity Video Capture
```

persistent送信。

---

## 3. turn別 audio watcher

global dir共有を廃止。

---

## 4. FG frame offset 連続化

以前：

```text
turn2で 00000000.png に戻る
```

現在：

```text
turn跨ぎで連番継続
```

これにより：

```text
OBSで後続turn映像が反映されない問題
```

解消。

---

# 現在の既知制約

## text chunk 未取得

現在：

```python
response_modalities=["AUDIO"]
```

のため、

```text
[debug][text_chunk]
```

は出ていない。

つまり：

```text
Live API text stream は未使用
```

状態。

現在の inline emo は：

```text
CLI inject
```

で動作。

---

# 次にやること

## 候補A

Live API text chunk 有効化。

```text
[emo:9_1]
```

などを実受信して parser 化。

---

## 候補B

expr streaming 化。

現在：

```text
expr_last_t_ms=0
```

固定。

将来的には：

```text
emo変化
blink
sleepy
```

などを realtime stream 化。

---

## 候補C

OBS + voice changer 構成。

将来的想定：

```text
Python
↓
virtual audio cable
↓
voice changer
↓
OBS
```

---

# 重要な現在の結論

現在のボトルネックは：

```text
M0ではなく
tool_call round-trip
```

だった可能性が高い。

inline emo方式で：

```text
2turn以降 ≈ 0.6〜0.9 sec
```

まで改善確認済。

```
```


#追加
# 2026-05 Realtime Low Latency Progress (Inline Emo Tag Mode)

## Goal

Replace delayed tool_call emotion flow with low-latency inline transcription emotion tags.

Old flow:

Live API
→ tool_call(set_emotion)
→ round-trip wait
→ emotion update
→ audio

New flow:

Live API audio
+ output_transcription
→ [emo:ID]
→ parser
→ expr.chunk.json update
→ M0
→ OBS

This removes tool_call round-trip latency.

---

# Current Stable Architecture

mic
↓
Gemini Live API (native audio)
↓
output_transcription.text
↓
[emo:ID] parser
↓
live_emo_override
↓
expr.chunk.json
↓
M0 persistent worker
↓
virtualcam_persistent
↓
OBS

---

# Important Discovery

Correct transcription field:

```python
server_content.output_transcription.text

NOT:

server_content.output_audio_transcription

This was the main reason transcription appeared missing.

Current Status

Stable:

persistent session loop
3-turn stable OBS rendering
inline emo parser
live emo override
M0 persistent
virtualcam persistent
race retry handling
first FG around ~1 sec

Confirmed logs:

[transcription][output] [emo:1_1]...
[transcription][emo_id] active_turn=1 emo_id=1_1
[stream_mouth_m0][live_emo_override] fallback=9_1 live=1_1
Important Runtime Behavior
output_transcription may arrive without audio

In some prompt patterns:

transcription arrives
audio does NOT arrive

Observed especially when prompts aggressively force emotion acting:

BAD example:

"とても眠そうに..."

This appears to bias the model toward text/transcription generation only.

Recommended production approach:

keep response_trigger short/simple
let system_instruction define emotion policy
let conversation context determine emo_id
Production Direction

Recommended production strategy:

system_instruction:
emo_id definitions
speaking tone definitions
inline tag rule
response_trigger:
minimal/simple
Live API:
decide emo_id dynamically from conversation context

Avoid over-controlling emo_id from response_trigger.

Current Recommended Runtime Flags
--inline_emo_tag_mode
--output_audio_transcription
--drop_initial_audio_ms 40
--stream_mouth_m0_chunk_len_ms 120

Current stable model:

gemini-3.1-flash-live-preview
api_version=v1alpha
Known Issues
1. output_transcription timing

Sometimes transcription arrives before audio chunks.

Current approach:

parser updates live_emo_id
next chunk uses override

This is acceptable for current latency targets.

2. Over-aggressive emotion prompts

Strong emotional forcing may suppress audio generation.

Production should avoid:

excessive emotional forcing
per-turn forced emo prompts
Important Conclusion

Main bottleneck was NOT M0.

Main bottleneck was:

batch emotion/control flow
↓
stream emotion/control flow

Migration to inline transcription solved the architectural issue.


---


#追記
## Realtime inline emo expression streaming

確認済み:

- output_transcription から inline [emo:ID] を取得
- live_emo_events として turn_state に蓄積
- expr.chunk.json を chunkごとに生成
- 1chunk内で expression timeline 複数eventを保持可能
- mouth終了後に live_emo_events が残っている場合、mouth最終frameをholdしてexpression描画を延長
- OBS上で happy -> surprised -> sad の切替を確認

成功ログ例:

```text
[perf][turn1_first_response_audio_chunk_sec] 0.647
[perf][stream_first_m0_done_from_turn_start_sec] 0.956
[stream_mouth_m0][live_emo_events] chunk=8 ... expr_timeline_n=2 ... 1_1 -> 2_0
[stream_mouth_m0][live_emo_events] chunk=11 ... expr_timeline_n=2 ... 2_0 -> 9_1
[stream_mouth_m0][live_emo_events] chunk=12 ... expr_timeline_n=1 ... 9_1

成功コマンド:

C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe `
  scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py `
  --session_id sess_live_session_loop_dev_emo_events_001 `
  --m1_repo_root C:\dev\M1_LLM_To_M2_TTS_united `
  --m3_repo_root C:\dev\M3_Live_API_1_united `
  --m0_repo_root C:\dev\M0_session_renderer_final_1 `
  --m35_repo_root C:\dev\M3.5_final `
  --pose_json C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json `
  --bg_video C:\dev\M3.5_final\in\with1.mp4 `
  --turns 1 `
  --gap_s 1.5 `
  --turn_idle_wait_s 1.2 `
  --turn_first_audio_timeout_s 6.0 `
  --mic_send_max_s 0.6 `
  --audio_device 15 `
  --stream_mouth_m0_chunk_len_ms 120 `
  --inline_emo_tag_mode `
  --inline_emo_queue_jsonl C:\dev\M1_LLM_To_M2_TTS_united\in\inline_emo_queue_probe.jsonl `
  --dev_live_emo_events_csv "1_1@600,2_0@1000,9_1@1400" `
  --drop_initial_audio_ms 40 `
  --output_audio_transcription `
  --response_trigger "短く返答してください。" `
  --clean `
  --clean_fg

## commit対象

```text
scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py
scripts/live_runtime/run_mic_input_obs_realtime_step1.py
docs/local_obs_realtime_low_latency.md


##追加
## Stable production baseline (2026-05)

Current recommended realtime settings:

- stream_mouth_m0_chunk_len_ms = 120
- Live API session reuse = ON
- reconnect_per_turn = OFF
- output_audio_transcription = ON

Reason:

80ms mode can achieve lower initial latency,
but Live API audio streaming becomes unstable across multi-turn sessions.
120ms currently provides stable realtime OBS output.



##追加
## Stable realtime baseline (2026-05)

Verified stable settings:

- stream_mouth_m0_chunk_len_ms = 120
- response_modalities = ["AUDIO"]
- output_audio_transcription = ON
- reconnect_per_turn = OFF
- turn_audio_retry_n = 0

Notes:

- 80ms mode can reduce initial latency,
  but multi-turn audio streaming becomes unstable.

- reconnect_per_turn and retry logic were tested,
  but current Gemini Live API behavior was less stable
  than a persistent single-session approach.

Current recommended production mode:
120ms + persistent session.




##追加最終20260520
## Stable realtime baseline

Recommended production settings:

- `stream_mouth_m0_chunk_len_ms = 120`
- persistent Live API session
- `response_modalities = ["AUDIO"]`
- `output_audio_transcription = ON`
- `reconnect_per_turn = OFF`
- `turn_audio_retry_n = 0`
- `skip_response_trigger = OFF`

Verified:

- 3 turns stable
- OBS video/audio output works
- `happy -> surprised -> sad` expression switching works
- `next_frame_offset` increments across turns
- first M0 completion is around 1 second

Stable command:

```powershell
C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe `
  scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py `
  --session_id sess_live_session_loop_120ms_3turn_baseline_001 `
  --m1_repo_root C:\dev\M1_LLM_To_M2_TTS_united `
  --m3_repo_root C:\dev\M3_Live_API_1_united `
  --m0_repo_root C:\dev\M0_session_renderer_final_1 `
  --m35_repo_root C:\dev\M3.5_final `
  --pose_json C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json `
  --bg_video C:\dev\M3.5_final\in\with1.mp4 `
  --turns 3 `
  --gap_s 2.0 `
  --turn_idle_wait_s 1.2 `
  --turn_first_audio_timeout_s 6.0 `
  --mic_send_max_s 0.6 `
  --audio_device 15 `
  --stream_mouth_m0_chunk_len_ms 120 `
  --inline_emo_tag_mode `
  --inline_emo_queue_jsonl C:\dev\M1_LLM_To_M2_TTS_united\in\inline_emo_queue_probe.jsonl `
  --dev_live_emo_events_csv "1_1@600,2_0@1000,9_1@1400" `
  --drop_initial_audio_ms 40 `
  --output_audio_transcription `
  --response_trigger "短く返答してください。" `
  --clean `
  --clean_fg

Dev-only / not recommended for production:

--stream_mouth_m0_chunk_len_ms 80
--reconnect_per_turn
--turn_audio_retry_n
--skip_response_trigger
response_modalities = ["TEXT", "AUDIO"]

Notes:

80ms can reduce first M0 completion slightly, but multi-turn audio is unstable.
reconnect_per_turn was tested as a recovery path, but current behavior is less stable than persistent session reuse.
skip_response_trigger did not reliably trigger audio response.
TEXT + AUDIO modality should not be used for the current Live API path.






##追記割り込み
Battle Interrupt Runtime Smoke
概要

Gemini Live API persistent session 上で、
Battle Runtime 用の「割り込み発話（interrupt speech）」を
リアルタイム挿入する最小MVP。

特徴：

persistent Live API session
OBS realtime output
M0 persistent worker
virtualcam persistent
queue-file based interrupt injection
turn_sent 後同期 interrupt send
実現できたこと

以下を実ログで確認済。

queue file から複数 interrupt line 読み込み
Battle interrupt send 成功
interrupt transcription 成功
audio chunk 継続
M0 chunk render 継続
OBS映像・音声反応あり
turn_sent 後同期方式で安定化

確認ログ例：

[battle_interrupt][after_turn_sent_queued]
[battle_interrupt][sent]
[perf][session_audio_chunk]
[realtime_step1][m0_chunk_done]
[virtualcam_persistent] sent=
__AUDIO_PLAYER_RESPONSE__
queue file 例
今すぐ割り込んで短くツッコんで
もう一回、さらに強く煽って
実行コマンド
C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe `
  scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py `
  --session_id sess_battle_interrupt_cli_120ms_3turn_001 `
  --m1_repo_root C:\dev\M1_LLM_To_M2_TTS_united `
  --m3_repo_root C:\dev\M3_Live_API_1_united `
  --m0_repo_root C:\dev\M0_session_renderer_final_1 `
  --m35_repo_root C:\dev\M3.5_final `
  --pose_json C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json `
  --bg_video C:\dev\M3.5_final\in\with1.mp4 `
  --turns 3 `
  --gap_s 2.0 `
  --turn_idle_wait_s 1.2 `
  --turn_first_audio_timeout_s 6.0 `
  --mic_send_max_s 0.6 `
  --audio_device 15 `
  --stream_mouth_m0_chunk_len_ms 120 `
  --inline_emo_tag_mode `
  --inline_emo_queue_jsonl C:\dev\M1_LLM_To_M2_TTS_united\in\inline_emo_queue_probe.jsonl `
  --dev_live_emo_events_csv "1_1@600,2_0@1000,9_1@1400" `
  --drop_initial_audio_ms 40 `
  --output_audio_transcription `
  --response_trigger "短く返答してください。" `
  --battle_interrupt_queue_file C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_queue.txt `
  --battle_interrupt_queue_interval_s 1.8 `
  --clean `
  --clean_fg
現状の制限

未実装：

audio routing 分離
self voice loopback防止
Zoom/TikTok相手音声 routing
mic gate
AI self-echo prevention

Battle Runtime本番化では、
audio routing layer の追加が必要。

注意:
Live API初回応答は不安定な場合がある。
成功判定は以下:
- after_turn_sent_queued
- sent
- transcription output
- session_audio_chunk
- m0_chunk_done
- AUDIO_PLAYER_RESPONSE
- OBS映像/音声反応

1回失敗した場合は同一コマンドを再実行する。




#追記=マイクスピーカー混線対策
Battle audio routing baseline:
AI audio output device = 15
OBS AI_CAT_AUDIO = CABLE Output
Desktop Audio = Disabled
Mic/Aux = Disabled
Headphone monitoring = OFF
1回目無反応の場合あり。2回目で成功確認。
成功条件:
- AI_CAT_AUDIO反応
- virtualcam_persistent sent
- AUDIO_PLAYER_RESPONSE device=15
- m0_chunk_done




追記セクション

# Battle Runtime Phase2
## File Polling Interrupt (2026-06)
保存版（Phase2 Battle Interrupt File Polling）
メイン実行コマンド
C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe `
  scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py `
  --session_id sess_battle_file_polling_prod_001 `
  --m1_repo_root C:\dev\M1_LLM_To_M2_TTS_united `
  --m3_repo_root C:\dev\M3_Live_API_1_united `
  --m0_repo_root C:\dev\M0_session_renderer_final_1 `
  --m35_repo_root C:\dev\M3.5_final `
  --pose_json C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json `
  --bg_video C:\dev\M3.5_final\in\with1.mp4 `
  --turns 1 `
  --gap_s 2.0 `
  --mic_send_max_s 0.6 `
  --ai_audio_output_device 15 `
  --stream_mouth_m0_chunk_len_ms 120 `
  --reconnect_per_turn `
  --bootstrap_audio_required `
  --bootstrap_retry_n 10 `
  --bootstrap_timeout_s 3.0 `
  --bootstrap_response_trigger "短く返答してください。" `
  --inline_emo_tag_mode `
  --inline_emo_queue_jsonl C:\dev\M1_LLM_To_M2_TTS_united\in\inline_emo_queue_probe.jsonl `
  --dev_live_emo_events_csv "1_1@600,2_0@1000,9_1@1400" `
  --drop_initial_audio_ms 40 `
  --output_audio_transcription `
  --response_trigger "短く返答してください。" `
  --battle_interrupt_file C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_live.txt `
  --battle_interrupt_file_poll_s 0.05 `
  --clean `
  --clean_fg
実運用 writer コマンド

実行前に空にする：

Set-Content `
  -Path C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_live.txt `
  -Value "" `
  -Encoding utf8

割り込み：

Set-Content `
  -Path C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_live.txt `
  -Value '{"type":"interrupt","text":"今すぐ短くツッコんで"}' `
  -Encoding utf8

煽り：

Set-Content `
  -Path C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_live.txt `
  -Value '{"type":"interrupt","text":"相手を軽く煽って"}' `
  -Encoding utf8

強め：

Set-Content `
  -Path C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_live.txt `
  -Value '{"type":"interrupt","text":"もっと強めに返して"}' `
  -Encoding utf8
Phase2 合格判定ログ

今回実際に通ったログです。

[battle_interrupt][file_pending]

[battle_interrupt][file_apply_as_control]

[transcription][output]
(内容が割り込み内容へ変化)

[perf][turn1_first_response_audio_chunk_sec]

[realtime_step1][m0_chunk_done]

__AUDIO_PLAYER_RESPONSE__

[battle_interrupt][file_consume_confirmed]


Phase3 Step1:
File Polling overwrite 実装済み。
未消費pendingがある状態で新指示が来た場合、最新指示のみ残す。
音声なしattemptではconsumeしない。
音声ありattempt成功後のみ file_consume_confirmed で消費。





docs/local_obs_realtime_low_latency.md 追記案
Battle Runtime Phase3 進捗
Phase3 Step1 : overwrite

実装済。

目的：

最新指示のみ採用

例：

煽れ
↓
優しく返せ

↓

優しく返せ

のみ保持。

Phase3 Step2 : priority metadata

実装済。

対応：

normal
battle
critical

未指定時：

normal

扱い。

例：

{
  "type":"interrupt",
  "priority":"battle",
  "text":"軽くツッコんで"
}
Phase3 Step3 : expire

実装済。

例：

{
  "type":"interrupt",
  "priority":"battle",
  "text":"軽くツッコんで",
  "expire_sec":30
}

期限切れ時：

[file_expired]

出力。

Phase3 Step4 : priority compare

実装済。

ルール：

critical > battle > normal

例：

battle pending
↓
normal投入

↓

battle維持

ログ：

[file_pending_keep_higher_priority]
現在のBattle Control JSON
{
  "type":"interrupt",
  "priority":"battle",
  "text":"軽くツッコんで",
  "expire_sec":30
}




Phase4 Step1 battle_control File Polling 化 完了

battle_control_live.txt を追加。
JSON:
{"type":"control","text":"軽くツッコむ口調で返して"}

仕様:
battle_control は apply 時点で consume する。
音声成功まで保持しない。
理由:
保持すると同じ control が retry attempt ごとに再注入され、audio_chunks=0 を固定化しやすい。

battle_interrupt との違い:
battle_interrupt = 音声成功まで保持
battle_control  = apply時consume




# Battle Runtime Phase4 進捗

## Phase4 Step1 : battle_control File Polling 化

実装済。

追加ファイル：

```text
in/battle_control_live.txt
```

形式：

```json
{
  "type":"control",
  "text":"軽くツッコむ口調で返して"
}
```

実装：

```text
battle_control_live.txt
↓
50ms polling
↓
battle_control_lines
↓
response_trigger 注入
```

---

## battle_control の仕様

battle_interrupt と異なり、

```text
apply時 consume
```

を採用。

理由：

```text
音声成功まで保持
```

にすると、

同じ control が retry attempt ごとに再注入され、

```text
audio_chunks=0
```

を固定化しやすいため。

---

## battle_interrupt の仕様

従来通り。

```text
音声成功時のみ consume
```

採用。

```text
file_consume_confirmed
```

まで保持。

---

## Phase4 Step2 : control + interrupt 同時投入

実装確認済。

同時投入：

```json
{"type":"control","text":"軽くツッコむ口調で返して"}
```

*

```json
{
  "type":"interrupt",
  "priority":"battle",
  "text":"今すぐ短くツッコんで",
  "expire_sec":60
}
```

結果：

```text
OBS映像・音声出力成功
```

---

## Phase4 Step3 : arbitration

実装済。

優先順位：

```text
interrupt > control > response_trigger
```

ログ：

```text
[battle_arbitration]
interrupt=True
control=True
winner=interrupt
```

---

## prompt順序

現在：

```text
【管理者制御】
...
【管理者割り込み予約】
...
```

固定。

control を土台、

interrupt を最終上書き指示として扱う。



Phase4 Step3-2 rollback確認 合格。
安定仕様:
battle_control = text only / apply時consume
battle_interrupt = priority + expire / 音声成功時consume
arbitration = interrupt > control
prompt順序 = control → interrupt
Step4 battle_control priority metadata は凍結。