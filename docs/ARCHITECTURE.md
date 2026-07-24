# Realtime Lip-Sync パイプライン — アーキテクチャ概要

> **親チャット（司令塔）向け SSOT。** ファイル役割・データフロー・目標設計を記述する。
> 絶対原則・禁止事項は `.cursorrules`、フェーズ進捗は `docs/PROGRESS.md` を参照。

## ベースラインと目標

| | 内容 |
| --- | --- |
| **現在のベースライン** | commit `e4c1204`（Phase10 STEP1 stable）— ローカル VAD 最終安定時点 |
| **作業ブランチ** | `feature/local-vad-restore` |
| **目標** | ローカル VAD（方式2）+ チャンク内同期一本道 + チャンク間並列 + データ量 jitter + 割り込み |

Phase10 ベースラインは **ファイル監視型**（receive 内 KNN + 別スレッドで player/M0）である。
Phase 2 以降、**図A（到着順並列パイプライン）** へ段階的に移行する。

---

## 主要ファイルと役割

### M1 リポジトリ（`scripts/live_runtime/`）

| ファイル | 役割 |
| --- | --- |
| **`run_mic_input_obs_realtime_session_loop.py`** | 全体オーケストレーター。Live API 接続、`_receive_loop()`、mic 送信、VAD、M0 watcher 起動、VirtualCam 連携を統括 |
| **`run_mic_input_obs_realtime_step1.py`** | 音声受信からリップシンク生成の架け橋。`_watch_stream_pcm_chunks()`（player へ PCM 送信）、`_watch_stream_mouth_and_render_m0()`（M0 描画監視）を提供 |
| **`dev_audio_chunk_player_persistent.py`** | AI 音声の常駐プレイヤー。PCM 受信 → jitter buffer → スピーカー出力。再生位置（`played_samples`）を管理 |
| **`run_virtualcam_persistent.py`** | M0 画像 + BGV を OBS VirtualCam へ出力 |

### M3 リポジトリ

| ファイル | 役割 |
| --- | --- |
| **`mouth_streamer_oc.py`** | Live API PCM を ~40ms 単位で受け取り、Formant 解析用 raw データを蓄積 |
| **`knn_from_formant_raw_to_mouth_timeline.py`** | Formant(raw) を KNN 検索し、口形タイムライン（`mouth.json`）を生成 |

### M0 リポジトリ

| ファイル | 役割 |
| --- | --- |
| **`m0_runner.py`** | M0 Worker（TCP サーバー）の起動・管理 |
| **`render_core.py`** | mouth / pose / expr を統合し PNG フレームを生成 |

### 中間データ

| データ | 役割 |
| --- | --- |
| **`mouth.json`** | KNN 出力。各 40ms 時点の口形 ID・時刻を保持するタイムライン |
| **M0 PNG フレーム** | リップシンク合成画像。`audio_ms` 範囲と紐付け |

---

## 3 層オーケストレーション（Phase10 ベースライン）

```text
run_mic_input_obs_realtime_session_loop.py
        │
        ├── run_mic_input_obs_realtime_step1.py
        │      ├─ _watch_stream_pcm_chunks()      → Audio Player へ PCM 送信
        │      └─ _watch_stream_mouth_and_render_m0() → M0 リップシンク生成
        │
        ├── dev_audio_chunk_player_persistent.py   → AI 音声を再生
        │
        └── run_virtualcam_persistent.py           → M0 画像 + BGV を OBS へ送信
```

一言で言うと:

- **session_loop** — 全体オーケストレーター
- **step1** — 音声 → リップシンク変換
- **audio_chunk_player** — 音声再生専用
- **virtualcam** — 映像出力専用

---

## Phase10 ベースラインのデータフロー

```text
Gemini Live API
      │
      ▼
_receive_loop()                         ← session_loop 内
      │
      ├──────────────────────→ pcm_stream_chunks/ → _watch_stream_pcm_chunks()
      │                                                    │
      │                                                    ▼
      │                                          dev_audio_chunk_player_persistent.py
      │                                                    │
      │                                                    ▼ スピーカー出力
      │
      └─→ mouth_streamer_oc.py
                │
                ▼
          knn_from_formant_raw_to_mouth_timeline.py
                │
                ▼
            mouth.json
                │
                ▼
      _watch_stream_mouth_and_render_m0()  ← session_loop → step1 経由
                │
                ▼
          m0_runner.py → render_core.py
                │
                ▼
          run_virtualcam_persistent.py → OBS
```

### Phase10 の入力制御（ローカル VAD）

- mic 送信: `_send_mic_once()` 内の RMS ベース VAD（`mic_vad_*` 引数）
- ターン終了: VAD 無音検知 → mic 停止 → `audio_stream_end=True`（または `--audio_stream_end_per_turn`）
- **目標（Phase 1）:** `activity_start/end` 方式2 へ置換。`drop_initial_audio_ms` / テキストレスポンストリガー撤去

---

## 目標アーキテクチャ（図A：到着順並列パイプライン）

```text
[Live API 受信ループ (_receive_loop)]
       │
       ├─► (音声 PCM チャンク到着)
       │         │
       │         ▼ 【非同期フォーク】 _process_pipeline_chunk_task（チャンクごと）
       │         │
       │         ▼ 【チャンク内は同期一本道】
       │         KNN（同期）→ M0 描画（同期・完了待ち）→ player enqueue
       │         │
       │         ▼
       │     [pipeline_enqueue_queue]
       │         │
       │         ▼ _pipeline_enqueue_dispatcher_loop
       │     [_enqueue_playback_audio_sync] → dev_audio_chunk_player_persistent.py
       │
       │     ※ 次チャンクは上記完了を待たず、別タスクとして並列開始
       │
       └─► [Virtual Cam / OBS 送信ループ]（再生 audio_ms に同期して描画）
```

### チャンク内（厳守）

`PCM 到着 → KNN → M0 描画（PNG 完了待ち）→ player enqueue`

- enqueue 時点で対応 chunk の mouth + M0 PNG が **必ず存在**
- M0 完了前の音声 enqueue は設計違反

### チャンク間（並列）

- チャンク N が M0 待ちでも、チャンク N+1 の処理は **ブロックしない**
- fast_worker / slow_worker 分離は **採用しない**

---

## 同期 ID（Sync SSOT）

音声の再生位置が唯一の時刻 ID。全レイヤーがこの ID で照合する。

```text
player.played_samples
  → player_local_ms / audio_ms
  → mouth.json frame（t_ms ≒ playback_origin_ms + audio_ms）
  → M0 PNG（同一 audio_ms 範囲）
  → virtualcam 表示 frame = 今鳴っている audio_ms
```

---

## バッファリング制御（目標）

| 概念 | 役割 | 初期値の目安 |
| --- | --- | --- |
| **初期ジッターバッファー** | 再生開始前に溜める M0 PNG 相当の音声データ量 | 300–500ms（CLI 可変・テストで最適化） |
| **再生開始フォールバック** | 蓄積不足時の強制再生開始 | ~1000ms |
| **M0 ハング救済** | 異常ハング時の `mouth_closed` 付与 → enqueue | ~500ms 超（CLI 可変） |

- 正常 M0 描画 ~250ms/chunk は enqueue 前コスト。初期ジッタが吸収する。
- 3 概念を混同しない（詳細は `.cursorrules` §1）。

### jitter / lip-sync delay の導入位置

- **PCM 側:** `dev_audio_chunk_player_persistent.py` の手前または内部
- **映像側:** M0 フレームを実際の音声再生時刻（`audio_ms`）に合わせて表示

M3 / M0 のコアロジック自体は大きく変えず、**player を基準に同期を取る**。

---

## 割り込み（Phase 4 目標）

| 状況 | 動作 |
| --- | --- |
| 通常再生中 | tail drain 完了まで `clear_queue` **禁止** |
| ユーザー割り込み発話 | **例外:** player + 描画キュー即時 clear、進行中 chunk タスク cancel |
| `generation_complete` | 入力終了マーカーのみ。clear_queue / ループ停止 **禁止** |

Phase10 → 新設計マッピングの詳細は `docs/PROGRESS.md` Phase 4 を参照。

---

## session_loop 内の主要関数（参照用）

| 関数 | 所在 | 役割 |
| --- | --- | --- |
| `_receive_loop()` | session_loop | Live API から PCM / transcription / tool call を受信 |
| `_send_mic_once()` | session_loop | mic PCM 送信 + ローカル VAD |
| `_watch_stream_pcm_chunks()` | step1 | PCM チャンクファイルを監視し player へ送信 |
| `_watch_stream_mouth_and_render_m0()` | step1 | mouth.json 等を監視し M0 Worker へ描画要求 |
| `_process_pipeline_chunk_task()` | session_loop（Phase 2 で実装） | チャンクごとの KNN→M0→enqueue 一本道 |

---

## 関連ドキュメント

| ファイル | 内容 |
| --- | --- |
| `.cursorrules` | 開発の絶対原則・禁止事項 |
| `docs/PROGRESS.md` | フェーズ進捗・Pass 基準 |
| `docs/streaming_design_notes.md` | 設計メモ（Sync SSOT 等） |

## 変更履歴

| 日付 | 内容 |
| --- | --- |
| 2026-07-24 | 初版（Phase10 ベースライン + 図A 目標設計を統合） |
