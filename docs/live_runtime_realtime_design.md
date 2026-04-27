docs/live_runtime_realtime_design.md


コピペ完全版
# Live Runtime Realtime Design（v1）

本ドキュメントは、Gemini Live API ベースの
**完全リアルタイム配信構成（常時worker型）** の設計書である。

---

# 0. ゴール

```text
入力（Live API）
↓
40ms単位 mouth 更新
↓
400ms単位 chunk生成
↓
M0レンダリング
↓
M3.5合成
↓
リアルタイム出力（OBS等）

👉 「止めない」「待たない」構成

1. 全体構成
┌──────────────────────────┐
│   Live API (Audio)        │
└────────────┬─────────────┘
             ↓
     ┌───────────────┐
     │ mouth_worker   │
     └──────┬────────┘
            ↓ 40ms frames
     ring buffer (mouth)

┌──────────────────────────┐
│ Live API (Tool Call)     │
└────────────┬─────────────┘
             ↓
     ┌───────────────┐
     │ expression_worker │
     └──────┬────────┘
            ↓ latest state

     ┌──────────────────────┐
     │ chunk_orchestrator    │
     └─────────┬────────────┘
               ↓ 400ms
         M0 renderer
               ↓
         M3.5 compositor
               ↓
            Output


---

# **1.5 実装フェーズ区分（重要）**

本システムは以下の2段階で実装されている：

---

## ■ v0（完成済・現在のSSOT）

```text
run_live_runtime_realtime_streaming.py

特徴
・chunk単位逐次実行
 ・M0 → M3.5 を同期実行
 ・Live API と完全並列ではない
 ・安全実装（既存成功コード）
フロー
Live API
 → worker（単発実行）
 → chunk_orchestrator
 → M0（subprocess）
 → M3.5（subprocess）
状態
E2E OK
streaming OK
realtime OK
final MP4 OK
👉 現在の基準（絶対に壊さない）

■ v1（次フェーズ：常時worker版）
未実装（このドキュメントの設計対象）
目的
・完全リアルタイム化
 ・「止めない」構成
 ・低レイテンシ化

変更点
mouth_worker      常時起動
expression_worker 常時起動
chunk_orchestrator ループ化
M0               常駐化（subprocess削減）
M3.5             ストリーム合成

フロー
Live API（常時接続）
 ↓
 worker（非同期）
 ↓
 ring buffer
 ↓
 chunk_orchestrator（400ms tick）
 ↓
 M0（常駐）
 ↓
 M3.5（follow mode）
 ↓
 OBS / 出力

禁止事項
・v0コードを直接改変しない
 ・SSOT（40ms / 400ms）を変更しない
 ・既存E2E結果を壊さない

👉 実装方針
v0を残す
↓
新ファイルでv1を作る


2. 時間設計（SSOT）
step_ms       = 40ms
chunk_len_ms  = 400ms
fps           = 25


SSOT
リアルタイム中:
  → "時間経過" がSSOT

オフライン:
  → audio_ms がSSOT


3. mouth_worker
役割
Live API から PCM 受信
MouthStreamerOC に push
40ms単位でフレーム生成
出力
mouth_ring_buffer:
  [{t_ms, mouth_id, f1, f2, vad}, ...]


要件
・ブロックしない
・pushベース
・リングバッファ


4. expression_worker
役割
set_emotion tool_call 受信
最新状態を保持
出力
latest_expression:
  {
    "expression": "smile",
    "updated_at_ms": ...
  }


要件
・1件だけ保持（履歴不要）
・上書き方式


5. chunk_orchestrator
役割
400msごとに以下を生成：
mouth: 直近10フレーム
expression: latest
pose: slice


出力
chunk_input:
  pose.chunk.json
  mouth.chunk.json
  expr.chunk.json


フロー
loop (every 400ms):
  ↓
  mouth ring buffer から10フレーム取得
  ↓
  expression latest 取得
  ↓
  pose slice
  ↓
  M0
  ↓
  M3.5


6. M0 / M3.5
M0
入力:
  pose
  mouth
  expr

出力:
  FG PNG


M3.5
入力:
  FG
  BG

出力:
  合成フレーム / 動画


7. データフロー
Live API → mouth_worker → ring buffer
Live API → expression_worker → latest state

chunk_orchestrator:
  ↓
  M0
  ↓
  M3.5


8. 状態管理
mouth: ring buffer
expression: single state
pose: static or loop


9. 非同期設計
thread-1: mouth_worker
thread-2: expression_worker
thread-3: orchestrator


10. v0との差分
v0:
  まとめて生成 → chunk処理

realtime:
  常時更新 → chunk逐次処理


