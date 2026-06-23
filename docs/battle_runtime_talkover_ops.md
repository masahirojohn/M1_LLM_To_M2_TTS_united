# Battle Runtime Talk-Over 運用手順（Phase5 Step3-3）

## 目的

Battle Runtime において、

* 管理者指示
* battle_interrupt_live.txt
* immediate send
* Gemini Live API

を利用し、

相手発話中（mic送信中）に AI猫へ割り込み指示を送信する。

---

# 現在の到達点

Phase5 Step3-3 合格。

確認済み：

* battle_interrupt_live.txt 監視
* immediate send
* session.send_realtime_input()
* OBS映像正常
* OBS音声正常
* リップシンク正常
* 音声崩壊なし

確認ログ：

```text
[mic_send][BEGIN]

[battle_interrupt][file_pending_overwrite]

[battle_interrupt][send_begin]

[battle_interrupt][file_immediate_queued]

[battle_interrupt][sent]

[mic_send][DONE]
```

---

# 毎回の事前準備

battleファイルを空にする。

```powershell
Set-Content `
  -Path C:\dev\M1_LLM_To_M2_TTS_united\in\battle_control_live.txt `
  -Value "" `
  -Encoding utf8

Set-Content `
  -Path C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_live.txt `
  -Value "" `
  -Encoding utf8
```

---

# Battle Runtime 運用版起動コマンド

```powershell
C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe `
  scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py `
  --session_id sess_phase5_step3_3_battle_runtime_ops_001 `
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
  --mic_send_max_s 3.0 `
  --mic_input_device 1 `
  --audio_device 15 `
  --stream_mouth_m0_chunk_len_ms 120 `
  --drop_initial_audio_ms 40 `
  --battle_control_file C:\dev\M1_LLM_To_M2_TTS_united\in\battle_control_live.txt `
  --battle_control_file_poll_s 0.05 `
  --battle_interrupt_file C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_live.txt `
  --battle_interrupt_file_poll_s 0.05 `
  --battle_interrupt_file_immediate_send `
  --clean `
  --clean_fg
```

---

# 管理者割り込み投入

別ターミナル。

```powershell
Set-Content `
  -Path C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_live.txt `
  -Value '{"type":"interrupt","priority":"battle","text":"今すぐ一言だけ被せてツッコめ","expire_sec":60}' `
  -Encoding utf8
```

---

# 期待ログ

```text
[battle_interrupt][file_pending_overwrite]

[battle_interrupt][send_begin]

[battle_interrupt][file_immediate_queued]

[battle_interrupt][sent]
```

---

# 重要実装メモ

## 二重watcher問題

過去：

```text
session外 watcher
+
session内 watcher
```

の二重起動。

結果：

```text
file_pending_overwrite
のみ
```

で停止。

---

修正後：

```text
--battle_interrupt_file_immediate_send ON

↓

session外 watcher 起動しない
```

へ変更。

確認ログ：

```text
[battle_interrupt][session_file_watcher_start]
immediate_send=True
```

---

# PowerShell注意事項

バッククォート

```powershell
`
```

の後ろに空白を入れない。

NG例：

```powershell
--clean `
```

（末尾に空白あり）

これにより行継続が壊れる。

---

# 今後

Phase5 Step3-4

目的：

相手発話中の被せ会話（talk-over）の実運用確認。

対象：

* battle_control
* battle_interrupt
* VAD設定
* Live API挙動

確認事項：

* 被せ開始タイミング
* 誤発話有無
* 自己音声ループ有無

```
```



# Phase5 Step3-4 Soft Cut-In（合格）

## 目的

相手発話中に battle interrupt を発火し、
AI猫が相手の発話終了を待たずに応答開始できることを確認する。

---

## 実装方式

真abort方式は使用しない。

採用方式：

interrupt
↓
cut_in_event
↓
mic送信停止
↓
audio_stream_end
↓
response_trigger
↓
AI応答開始

（soft cut-in）

---

## 使用オプション

--battle_interrupt_file_immediate_send

--battle_talkover_receive_during_mic

--battle_talkover_cut_in_on_interrupt

---

## 外部投入スクリプト

scripts/live_runtime/dev_write_battle_interrupt_after_delay.py

例：

python scripts/live_runtime/dev_write_battle_interrupt_after_delay.py ^
  --delay_s 1.0 ^
  --interrupt_file in/battle_interrupt_live.txt ^
  --text "まだ話す気かよ！一回止まれ！"

---

## 合格ログ

[battle_talkover][cut_in_requested]

[mic_send][CUT_IN_STOP_BEFORE_SEND]

[battle_talkover][audio_stream_end_sent]

[perf][session_audio_chunk]

[realtime_step1][m0_chunk_done]

__AUDIO_PLAYER_RESPONSE__

---

## 結果

OBS映像あり

OBS音声あり

リップシンクあり

相手発話中に割り込み開始確認

---

## 注意

dev_battle_file_writer は検証専用。

本番では使用しない。

battle_interrupt_live.txt
または管理者UIから投入する。




# Phase5 Step4 Routing確認（合格）

## 目的

AI猫が自分自身の音声を聞いて再反応しないことを確認する。

---

## 確認結果

### Live API入力

```text
mic_input_device=1
```

実装：

```text
sounddevice.InputStream(device=args.mic_input_device)
```

Live APIへ送信される音声は mic_input_device のみ。

---

### AI音声出力

```text
audio_device=15
```

実装：

```text
_start_audio_player(... audio_device=args.ai_audio_output_device)
```

AI音声は device=15 へ出力される。

---

## Routing

### 入力系

```text
マイク
↓
mic_input_device=1
↓
Live API
```

### 出力系

```text
Live API
↓
AI音声
↓
audio_device=15
↓
VB-Cable
↓
OBS AI_CAT_AUDIO
```

---

## 自己音声ループ試験

テスト条件：

```text
3秒発話
↓
以後無言
↓
AI応答を観察
```

結果：

```text
AI応答は1回のみ
追加応答なし
再帰応答なし
```

自己音声ループは確認されなかった。

---

## 運用固定設定

### Runtime

```text
mic_input_device=1

audio_device=15
```

---

### OBS

使用：

```text
AI_CAT_AUDIO
```

無効：

```text
デスクトップ音声

マイク音声
```

---

### Windows

通常運用：

```text
CABLE Output
「このデバイスを聴く」
OFF
```

音声確認時のみ：

```text
CABLE Output
↓
このデバイスを聴く
ON

再生先
↓
ヘッドホン
```

---

## Phase5 Step4 判定

結果：

```text
PASS
```

AI音声とLive API入力は分離されている。

Battle Runtime運用構成として採用可能。





# Battle Runtime 運用メモ（2026-06 更新）

## 音声途切れ問題

### 症状

OBSでは映像は正常だが、

```text
音声がプツプツ途切れる
音声が意味不明になる
```

状態が発生。

---

### 原因

主因はネットワークではなかった。

確認結果：

```text
Wi-Fi
↓
スマホ有線テザリング
```

へ変更しても大幅改善なし。

---

### 実際に改善した対応

対象ファイル：

```text
scripts/live_runtime/dev_audio_chunk_player_persistent.py
```

音声再生キュー処理を改善。

結果：

```text
stream_mouth_m0_chunk_len_ms=200
```

で正常化。

さらに

```text
stream_mouth_m0_chunk_len_ms=120
```

へ戻しても正常。

---

### 現在の運用値

```text
--stream_mouth_m0_chunk_len_ms 120
```

固定。

---

## emo_id保護

### 問題

Live APIが定義外emo_idを返す。

実例：

```text
😺
angry
anger
```

---

### 現在の挙動

ログ：

```text
[emo_id][fallback] raw=😺 fallback=1_1
[emo_id][accepted] emo_id=1_1
```

---

### 現在の仕様

定義外emo_idは自動fallback。

例：

```text
😺
angry
anger
```

↓

```text
1_1
```

---

### 備考

M0/M3側の異常停止は防止済み。

ただし、

```text
Live APIが正しいemo_idを返しているか
```

は未解決。

将来改善候補。

---

## mic_gate

### 目的

相手音声をLive APIへ送るか停止するか制御。

---

### 状態

```text
open
mute
```

---

### 動作

open

```text
マイク送信あり
```

mute

```text
マイク送信なし
```

---

### 実証済ログ

mute

```text
[battle_mic_gate][file_set] state=mute
[battle_mic_gate][MUTED] i=0
[mic_send][DONE] sent_bytes=0
```

open

```text
[battle_mic_gate][file_set] state=open
[mic_send][DONE] sent_bytes=96000
```

---

## battle_control

### 目的

AI猫の会話方針を変更する。

例：

```text
強気に話せ
煽れ
短く返答しろ
主導権を握れ
```

---

### ファイル

```text
in/battle_control_live.txt
```

---

### JSON例

```json
{
  "type":"control",
  "text":"会話の主導権を握って強気に短く話せ"
}
```

---

## mic_gate + battle_control 同時投入

### JSON例

```json
{
  "type":"control",
  "mic_gate":"mute",
  "text":"会話の主導権を握って強気に短く話せ"
}
```

---

### 実証済

```text
mic_gate=mute
+
battle_control
```

同時反映成功。

---

## 自動投入ツール

対象：

```text
scripts/live_runtime/dev_write_battle_control_after_delay.py
```

---

### mute投入

```powershell
python scripts/live_runtime/dev_write_battle_control_after_delay.py `
  --delay_s 1 `
  --control_file in\battle_control_live.txt `
  --action set `
  --mic_gate mute `
  --text "会話の主導権を握って強気に短く話せ"
```

---

### open投入

```powershell
python scripts/live_runtime/dev_write_battle_control_after_delay.py `
  --delay_s 22 `
  --control_file in\battle_control_live.txt `
  --action set `
  --mic_gate open `
  --text "相手音声を再開する。ただし主導権は維持しろ"
```

---

## Battle Runtime 現在地

実証済：

```text
battle_interrupt
soft cut-in
talk-over
audio routing
emo fallback
mic_gate
battle_control
```

---

### 次候補

1.

```text
battle_interrupt
+
mic_gate
統合運用
```

2.

```text
Priority Arbitration
(critical/battle/normal)
```

3.

```text
管理者UI
(Streamlit)
```

---

## 本番運用固定値

```text
mic_input_device=1

audio_device=15

stream_mouth_m0_chunk_len_ms=120

battle_control_file_poll_s=0.2
```

---

## OBS

使用：

```text
AI_CAT_AUDIO
```

無効：

```text
デスクトップ音声
マイク音声
```

---

## 最終結論

Battle Runtime MVPは実運用レベル。

現在は

```text
battle_control
+
mic_gate
```

により

```text
相手音声停止
↓
主導権奪取
↓
相手音声再開
```

まで実証済み。




# 会話主導権モード（Conversation Leadership Mode）

## 目的

管理者が配信中に

```text
相手話中
↓
割り込み
↓
主導権奪取
↓
相手音声遮断
↓
AI猫が話し続ける
↓
相手音声再開
```

を実行できるようにする。

---

## 構成

会話主導権モードは以下の3機能の組み合わせ。

```text
battle_interrupt

battle_control

mic_gate
```

---

### battle_interrupt

役割：

```text
今すぐ割り込め
```

をLive APIへ送る。

---

### battle_control

役割：

```text
強気に話せ

煽れ

主導権を握れ

短く返せ
```

などの会話方針変更。

---

### mic_gate

役割：

```text
mute
open
```

による相手音声送信制御。

---

## 実装済ファイル

新規追加：

```text
scripts/live_runtime/dev_write_conversation_leadership_mode_after_delay.py
```

---

## 動作シーケンス

### Step1

control + mute

```json
{
  "type":"control",
  "mic_gate":"mute",
  "text":"会話の主導権を握れ。強気に短く話せ。相手の発話は無視してよい。"
}
```

---

### Step2

interrupt

```json
{
  "type":"interrupt",
  "priority":"battle",
  "text":"今すぐ割り込め",
  "expire_sec":60
}
```

---

### Step3

open

```json
{
  "type":"control",
  "mic_gate":"open",
  "text":"相手音声を再開する。ただし会話の主導権は維持し、短く強気に返答しろ。"
}
```

---

## 自動投入コマンド

```powershell
C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe `
  scripts/live_runtime/dev_write_conversation_leadership_mode_after_delay.py `
  --control_file C:\dev\M1_LLM_To_M2_TTS_united\in\battle_control_live.txt `
  --interrupt_file C:\dev\M1_LLM_To_M2_TTS_united\in\battle_interrupt_live.txt `
  --start_delay_s 3 `
  --interrupt_delay_s 5 `
  --open_delay_s 20
```

---

## 実証済ログ

### interrupt

```text
[battle_interrupt][file_immediate_queued]
[battle_interrupt][send_begin]
[battle_interrupt][sent]
```

---

### talk-over

```text
[battle_talkover][cut_in_requested]
[mic_send][CUT_IN_STOP_BEFORE_SEND]
[battle_talkover][audio_stream_end_sent]
```

---

### mute

```text
[battle_mic_gate][set] turn=1 state=mute
[mic_send][DONE] sent_bytes=0
```

---

### open

```text
[battle_mic_gate][file_set] state=open
[mic_send][CHUNK]
[mic_send][DONE] sent_bytes=8960
```

---

## テスト結果

結果：

```text
PASS
```

確認済：

```text
battle_interrupt
→ PASS

battle_control
→ PASS

mic_gate mute
→ PASS

mic_gate open
→ PASS

talk-over
→ PASS
```

---

## 運用上の意味

管理者は

```text
相手が話し続ける
↓
主導権奪取
```

を実行できる。

これにより、

```text
国際バトル

討論

煽り配信

リアクション配信
```

などで

AI猫が会話の主導権を握る運用が可能。

---

## 現在地

Battle Runtime MVP

完了。

完了済：

```text
talk-over

battle_interrupt

battle_control

audio routing

emo fallback

mic_gate

conversation leadership mode
```

---

## 次候補

### Option A

管理者UI

```text
Streamlit
```

目的：

```text
interrupt

control

mic_gate

主導権モード
```

をボタン化。

---

### Option B

Priority Arbitration

```text
critical

battle

normal
```

優先度制御。

---

## 推奨

現時点では

```text
管理者UI
```

を先に実装する。

理由：

既に運用機能は揃っているため、

次の価値は

```text
操作簡略化
```

である。






# Streamlit管理者UI

## 目的

配信中に管理者が

```text
battle_interrupt

battle_control

mic_gate

conversation_leadership_mode
```

をブラウザから操作できるようにする。

---

## 対象ファイル

```text
scripts/live_runtime/admin_control_panel.py
```

---

## 起動コマンド

```powershell
C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe `
  -m streamlit run `
  scripts/live_runtime/admin_control_panel.py
```

---

## 操作可能機能

### battle_interrupt

用途：

```text
今すぐ割り込め
```

を Runtime へ投入。

生成ファイル：

```text
in/battle_interrupt_live.txt
```

---

### battle_control

用途：

```text
強気に話せ

煽れ

主導権を握れ

短く返せ
```

などの会話方針変更。

生成ファイル：

```text
in/battle_control_live.txt
```

---

### mic_gate

用途：

```text
mute

open
```

制御。

ボタン：

```text
MUTE 相手音声停止

OPEN 相手音声再開
```

---

### conversation leadership mode

用途：

```text
主導権奪取
```

実行。

内部動作：

```text
battle_control
+
mic_gate=mute
+
battle_interrupt
```

を連続投入。

---

## UI Operation Log

追加済。

画面下部に

```text
interrupt

control

mute

open

leadership mode
```

投入履歴を表示。

---

## Status / Debug

追加済。

常時表示：

```text
battle_control_live.txt

battle_interrupt_live.txt
```

内容確認可能。

---

## 実証済テスト

### UI単体

確認済：

```text
Interruptボタン
→ PASS

MUTEボタン
→ PASS

OPENボタン
→ PASS

Control投入
→ PASS

主導権奪取
→ PASS
```

---

### Runtime統合

確認済：

```text
UI
↓
battle_interrupt
↓
cut-in
↓
talk-over
↓
mute
↓
open
↓
OBS映像
↓
OBS音声
```

PASS。

---

## 運用方法

推奨：

```text
Runtime起動
↓
Streamlit UI起動
↓
ブラウザから管理
```

---

## 現在地

Battle Runtime MVP

完了。

実装済：

```text
battle_interrupt

battle_control

talk-over

audio routing

emo fallback

mic_gate

conversation_leadership_mode

Streamlit管理者UI
```

---

## 将来計画

### Slack連携

目的：

```text
Slack
↓
管理者コマンド
↓
Runtime制御
```

---

### 想定構成

```text
Slack

Streamlit

↓

battle_runtime_api.py

↓

write_interrupt()

write_control()

write_leadership_mode()

↓

battle_interrupt_live.txt

battle_control_live.txt
```

---

## 次フェーズ候補

### Phase7 Step2

共通Writer API化

目的：

```text
Streamlit

Slack

将来のWebUI
```

から同じ処理を呼び出す。

---

### Phase8

Zoom/TikTok統合

目的：

```text
実戦運用対人バトル

会話主導権モード
```への接続。






Phase7 Step2 共通Writer API化
目的

管理者操作の実装を

Streamlit
Slack
将来のWebUI

で共通化する。

UI固有コードから

battle_control_live.txt

battle_interrupt_live.txt

への直接書き込みを排除し、

battle_runtime_admin_api.py

へ集約する。

追加ファイル
scripts/live_runtime/battle_runtime_admin_api.py
提供API
write_interrupt()

用途：

battle_interrupt投入
write_control()

用途：

battle_control投入
write_mic_gate()

用途：

mic_gate=open

mic_gate=mute

投入。

clear_control()

用途：

battle_control解除
write_leadership_start()

用途：

会話主導権モード開始

内部動作：

battle_control
+
mic_gate=mute
+
battle_interrupt

投入。

write_leadership_open()

用途：

主導権維持
+
mic_gate=open

へ移行。

write_normal_conversation()

用途：

通常会話へ復帰
read_status()

用途：

battle_control_live.txt

battle_interrupt_live.txt

状態取得。

Streamlit側変更

対象：

scripts/live_runtime/admin_control_panel.py

変更内容：

ローカルwriter関数削除

battle_runtime_admin_api.py
へ統一
実証済み
UI単体

確認済：

今すぐ割り込め
主導権奪取
主導権解除 OPEN
通常会話へ戻す
MUTE
OPEN
戦闘管制投入
戦闘管制解除

PASS。

Runtime統合

確認済：

Streamlit
↓
battle_runtime_admin_api
↓
battle_control_live.txt
battle_interrupt_live.txt
↓
Runtime
↓
OBS

PASS。

確認ログ：

[battle_interrupt][file_immediate_queued]

[battle_talkover][cut_in_requested]

[battle_control][file_pending_overwrite]

[battle_mic_gate][set] state=mute

[battle_mic_gate][file_set] state=open
現在の管理系アーキテクチャ
Streamlit UI
        │
        ▼
battle_runtime_admin_api.py
        │
        ├── write_interrupt()
        ├── write_control()
        ├── write_mic_gate()
        ├── write_leadership_start()
        └── write_leadership_open()
        │
        ▼
battle_interrupt_live.txt
battle_control_live.txt
        │
        ▼
Runtime Session Loop
運用方針

管理者は原則として

主導権奪取

主導権解除 OPEN

通常会話へ戻す

を中心に使用する。

以下は補助操作：

MUTE

OPEN

戦闘管制自由入力

今すぐ割り込め



---

# Phase8 Zoom実戦運用（確定版）

## 概要

Battle Runtime を Zoom へ接続する場合は、

```text
OBS
↓
Virtual Camera
↓
Zoom
```

構成を使用する。

Zoom専用実装は作らない。

YouTube / TikTok / Zoom を同一Runtimeで運用する。

---

# Zoom Audio Routing

## Zoom映像

Zoom設定

```text
Camera

Unity Video Capture
```

を選択。

---

## Zoom音声

Zoom設定

### Speaker

```text
CABLE Input (VB-Audio Virtual Cable)
```

### Microphone

```text
マイク (USB PnP Audio Device)
```

---

## Runtime

### mic_input_device

```text
3
```

使用。

実体：

```text
CABLE Output (VB-Audio Virtual Cable)
```

---

### audio_device

```text
14
```

使用。

実体：

```text
CABLE Input (VB-Audio Virtual Cable)
```

---

# Zoom主導権モード

## 使用するボタン

使用：

```text
Zoom主導権
```

---

使用しない：

```text
主導権奪取
```

理由：

```text
mic_gate=mute
battle_interrupt
```

が同時投入されるため。

Zoom実戦では不安定。

---

# Zoom主導権の動作

投入内容：

```text
会話の主導権を握れ。
強気に短く返せ。
ただし相手音声は止めるな。
```

実装：

```text
battle_control のみ
```

---

実行しない：

```text
battle_interrupt
mic_gate=mute
```

---

# Zoom実戦手順

## 起動

1

OBS起動

---

2

Zoom起動

---

3

スマホをZoomへ参加

---

4

管理画面起動

```powershell
C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe `
  -m streamlit run `
  scripts/live_runtime/admin_control_panel.py
```

---

5

Runtime起動

---

## 会話

スマホ側

```text
3～5秒話す
↓
黙る
```

---

AI猫

```text
返答
```

---

必要時

```text
Zoom主導権
```

クリック。

---

# 確認済み事項

確認済み：

```text
Zoom映像
```

OK

---

確認済み：

```text
Zoom音声
```

OK

---

確認済み：

```text
battle_control
```

OK

---

確認済み：

```text
Zoom主導権
```

OK

---

確認済み：

```text
AI猫 → Zoom音声返答
```

OK

---

確認済み：

```text
talk-over気味応答
```

発生確認。

---

# Phase8完了条件

以下成立。

```text
Zoom相手
↓
Runtime
↓
Live API
↓
M0
↓
OBS
↓
Zoom
```

映像・音声往復成功。

---

Phase8到達点：

```text
Battle Runtime基盤完成
↓
管理UI完成
↓
Zoom実戦接続完了
```
