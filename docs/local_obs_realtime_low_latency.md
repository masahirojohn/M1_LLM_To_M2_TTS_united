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
