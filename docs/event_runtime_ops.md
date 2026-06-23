# docs/event_runtime_ops.md

````md
# Event Runtime Operations Guide
Version: Phase9 Pattern A Complete
Status: Production Ready (Zoom Confirmed)

---

# 1. 概要

目的：

配信中・Zoom会話中に

管理者が任意タイミングで

イベント動画を挿入する。

---

実現済み構成

event_runtime_live.txt
↓
Battle Runtime
↓
bg_override_live.txt
↓
virtualcam
↓
OBS
↓
Zoom

確認済み：

- OBS表示
- Zoom表示
- Runtime継続
- イベント終了後自動復帰

---

# 2. 現在の対応範囲

## Pattern A

対応済み

内容：

イベント動画のみ

---

動作

通常配信
↓
イベント投入
↓
イベント動画表示
↓
duration経過
↓
通常配信へ復帰

---

例

- 驚き
- 照れ
- 土下座
- 激怒

---

# 3. イベント投入方法

## 管理UI

admin_control_panel.py

---

ボタン

evt_001 即投入

または

event_id入力
↓
イベント投入

---

## ファイル投入

event_runtime_live.txt

例

```json
{
  "type": "event",
  "event_id": "evt_001"
}
````

---

# 4. event_catalog.json

場所

```text
M1_LLM_To_M2_TTS_united/in/event_catalog.json
```

例

```json
{
  "evt_001": {
    "event_mode": "bg_only",
    "bg_video": "in/evt_001.mp4",
    "pose_json": "in/pose_timeline_evt_001.json",
    "duration_s": 3.0,
    "audio": false
  }
}
```

---

# 5. event_id命名

自由。

例

```text
evt_001
evt_002
rage_001
victory_001
embarrassed_001
```

制約なし。

---

# 6. 動画配置

場所

```text
C:\dev\M3.5_final\in\
```

例

```text
evt_001.mp4
rage_001.mp4
victory_001.mp4
```

---

event_catalog.json

では

```json
"bg_video":"in/evt_001.mp4"
```

と記述。

---

# 7. pose_json

場所

```text
C:\dev\M3.5_final\in\
```

例

```text
pose_timeline_evt_001.json
```

---

現状：

bg_only運用。

---

推奨

```json
disable_fg=true
```

---

# 8. Runtime構成

event_runtime_live.txt
↓
event watcher
↓
event_catalog resolve
↓
mic_gate=mute
↓
audio_player clear_queue
↓
bg_override write
↓
virtualcam switch
↓
duration待機
↓
mic_gate=open

---

採用理由

Live API停止不要。

安定。

---

# 9. bg_override_live.txt

場所

```text
C:\dev\M1_LLM_To_M2_TTS_united\in\
```

用途

virtualcam制御。

---

例

```json
{
  "type":"bg_override",
  "event_id":"evt_001",
  "bg_video":"C:\\dev\\M3.5_final\\in\\evt_001.mp4",
  "duration_s":3.0
}
```

Runtimeが自動生成。

手動編集不要。

---

# 10. Zoom運用

確認済み。

構成

OBS
↓
Unity Video Capture
↓
Zoom

---

Zoom設定

Camera

```text
Unity Video Capture
```

---

Microphone

```text
CABLE Output
```

---

Speaker

```text
CABLE Input
```

---

確認済み

* AI猫映像
* AI猫音声
* イベント動画
* イベント後自動復帰

---

# 11. 代表ログ

イベント開始

```text
[event_runtime][trigger]
[event_runtime][bg_override_written]
[virtualcam_persistent][bg_override]
```

---

イベント終了

```text
[event_runtime][mic_gate_open]
[virtualcam_persistent][bg_restore]
```

---

成功判定

```text
bg_override
↓
bg_restore
```

両方出力。

---

# 12. トラブルシュート

## ケース1

```text
bg_video_empty
```

原因

event_catalog未読込。

確認

```text
[event_runtime][catalog_loaded]
```

---

## ケース2

```text
missing bg_video
```

原因

mp4未配置。

確認

```text
C:\dev\M3.5_final\in\
```

---

## ケース3

イベント発火するが映像変わらない

確認

```text
[virtualcam_persistent][bg_override]
```

出ているか。

---

## ケース4

Zoomで見えない

確認

Camera

```text
Unity Video Capture
```

になっているか。

---

# 13. 現在の制限

未対応

Type B

---

内容

イベント動画＋イベント音声

例

* 必殺技
* 勝利演出
* チキンと言うな！

---

今後

audio_playerへ

```text
play_wav
play_mp4_audio
```

追加予定。

---

# 14. Phase9完了条件

達成済み

* Runtime稼働
* OBS表示
* Zoom表示
* 管理UI投入
* event_catalog連携
* bg_override切替
* 自動復帰

Status

PASS

```

このまま `docs/event_runtime_ops.md` として保存すれば、Phase9 Pattern A の運用手順・引継ぎ資料として使えます。
```




docs追記案です。そのまま `docs/event_runtime_ops.md` に追加可能です。

# Event Runtime Ops (Phase9)

## 概要

Phase9にて、Battle Runtime上からM3.5イベント動画をリアルタイム挿入する機能を実装。

対応：

```text
TypeA
動画のみイベント

TypeB
音声入りイベント
```

両方ともZoom実機確認済み。

---

# Runtime構成

通常：

```text
Gemini Live API
↓
Battle Runtime
↓
M0
↓
VirtualCam
↓
OBS
↓
Zoom
```

イベント時：

```text
event_runtime_live.txt
↓
Battle Runtime
↓
mic_gate=mute
↓
audio_player clear_queue
↓
BG Override
↓
イベント動画
↓
イベント終了
↓
mic_gate=open
↓
通常復帰
```

---

# Event Runtime File

ファイル：

```text
in/event_runtime_live.txt
```

形式：

```json
{
  "type":"event",
  "event_id":"evt_001"
}
```

overwrite方式。

queue方式は使用しない。

---

# Event Catalog

ファイル：

```text
in/event_catalog.json
```

---

## TypeA

動画のみ

```json
{
  "evt_001": {
    "event_mode": "bg_only",
    "bg_video": "in/evt_001.mp4",
    "pose_json": "in/pose_timeline_evt_001.json",
    "audio": false
  }
}
```

---

## TypeB

音声入りMP4

```json
{
  "evt_voice_001": {
    "event_mode": "bg_only",
    "bg_video": "in/evt_voice_001.mp4",
    "pose_json": "in/pose_timeline_evt_voice_001.json",
    "audio": true,
    "audio_source": "mp4",
    "post_control": "イベント後も激怒状態を維持して、強めに短く返して。"
  }
}
```

---

# MP4音声方式

採用。

event実行時：

```text
MP4
↓
ffmpeg
↓
wav抽出
↓
event_audio_cache
↓
play_wav
```

---

キャッシュ：

```text
out/event_audio_cache/
```

例：

```text
out/event_audio_cache/evt_voice_001.wav
```

---

# duration管理

duration_sは省略可能。

実行時：

```text
ffprobe
↓
duration自動取得
```

例：

```text
[event_runtime][duration_from_mp4]
duration_s=4.015
```

---

# BG Override

ファイル：

```text
in/bg_override_live.txt
```

イベント開始：

```text
bg_override
```

イベント終了：

```text
bg_restore
```

---

# post_control

目的：

```text
イベント後も感情維持
```

例：

```text
イベント動画
↓
激怒演出
↓
動画終了
↓
激怒状態で返答継続
```

実装：

```text
battle_control overwrite
```

利用。

新規システムなし。

---

# 管理UI

対象：

```text
scripts/live_runtime/admin_control_panel.py
```

実装済みボタン：

```text
evt_001 即投入

evt_voice_001 即投入
```

---

# Zoom確認済み

確認済み：

```text
OBS
↓
VirtualCam
↓
Zoom
```

確認内容：

```text
通常リップシンク
イベント動画
イベント音声
イベント終了後復帰
```

全てPASS。

---

# 運用ルール

event_idは任意。

推奨：

```text
evt_xxx
evt_voice_xxx
```

---

動画配置：

```text
M3.5_final/in/
```

例：

```text
evt_001.mp4
evt_voice_001.mp4
```

---

pose配置：

```text
M3.5_final/in/
```

例：

```text
pose_timeline_evt_001.json
pose_timeline_evt_voice_001.json
```

---

# 今後

未実装：

```text
複数event同時投入
event priority
event queue
```

現状は overwrite運用。

実戦運用上は十分。