# Realtime Lip-Sync 再構築 — 進捗管理（SSOT）

> **親チャット（司令塔）がこのファイルを更新する。** 子チャットは直接編集しない。

## ベースライン

| 項目 | 値 |
| --- | --- |
| 起点 commit | `e4c1204`（Phase10 STEP1 stable restore before STEP2 retry） |
| 作業ブランチ | `feature/local-vad-restore` |
| 復元対象（session_loop） | `git checkout e4c1204 -- scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py` |
| 未 commit 3 ファイル | `run_mic_input_obs_realtime_step1.py`, `dev_audio_chunk_player_persistent.py`, `run_virtualcam_persistent.py` |
| 設計 SSOT | リポジトリ直下 `.cursorrules` |

## フェーズ一覧

| Phase | 名称 | 状態 | Pass 日 |
| --- | --- | --- | --- |
| 0 | ベースライン監査・計画 | `pending` | — |
| 1 | ローカル VAD 完全化 | `pending` | — |
| 2 | 一本道並列（session_loop + step1） | `pending` | — |
| 3 | データ量 jitter（audio_player） | `pending` | — |
| 4 | 割り込み・talkover 整合 | `pending` | — |
| 5 | VirtualCam SSOT | `pending` | — |
| 5b | fast_inmemory（任意） | `pending` | — |
| 6 | 総合 E2E | `pending` | — |

状態値: `pending` / `in_progress` / `pass` / `blocked`

---

## Phase 0: ベースライン監査・計画

**目的:** ロールバック後コードの現状把握。コード変更なし。

**Pass 基準:**
- [ ] 現行 vs `e4c1204` の差分表（4 ファイル）
- [ ] 旧 VAD 混在箇所リスト（`automatic_activity_detection`, `drop_initial_audio_ms`, テキストトリガー等）
- [ ] 3 ファイルのボトルネック候補リスト
- [ ] Phase 1 着手 Go/No-Go 判断

**子チャット報告:** （ここにサマリーを貼る）

---

## Phase 1: ローカル VAD 完全化

**目的:** server VAD 混在を解消し、方式2（クライアント VAD）に一本化。

**Pass 基準:**
- [ ] `automatic_activity_detection=False`
- [ ] `activity_start=True` / `activity_end=True` でターン完結
- [ ] `drop_initial_audio_ms` / テキストレスポンストリガー撤去または無効化
- [ ] `audio_stream_end_per_turn` を `activity_end` ベースに置換（通常ターン）
- [ ] 1 ターン E2E：ユーザー発話 → AI 音声返答

**主な対象:** `run_mic_input_obs_realtime_session_loop.py`

**子チャット報告:** （ここにサマリーを貼る）

---

## Phase 2: 一本道並列（session_loop + step1）

**目的:** チャンク内 KNN→M0→enqueue、チャンク間非ブロック並列。

**Pass 基準:**
- [ ] `_process_pipeline_chunk_task`（または同等）がチャンクごとに非同期起動
- [ ] チャンク内: KNN → M0 完了 → enqueue の順序ログ確認
- [ ] チャンク N の M0 待ち中もチャンク N+1 が処理開始されること
- [ ] `[sync][pipeline_chunk]` ログ出力（chunk_idx, knn_ms, m0_ms, enqueue_ms）
- [ ] supply_gap なし（enqueue 時 M0 PNG 存在）

**主な対象:** `run_mic_input_obs_realtime_session_loop.py`, `run_mic_input_obs_realtime_step1.py`

**子チャット報告:** （ここにサマリーを貼る）

---

## Phase 3: データ量 jitter（audio_player）

**目的:** M0 PNG 蓄積量トリガーで再生開始。固定 sleep 待機に依存しない。

**Pass 基準:**
- [ ] M0 PNG 相当 **300–500ms**（CLI 可変）蓄積で再生開始
- [ ] 蓄積不足時 **~1000ms** フォールバックで強制開始（無限ブロックなし）
- [ ] 連続再生でアンダーランなし
- [ ] ジッターバッファー値と M0 ハング救済タイムアウトが混同されていない

**主な対象:** `dev_audio_chunk_player_persistent.py`, session_loop の player 連携部分

**子チャット報告:** （ここにサマリーを貼る）

---

## Phase 4: 割り込み・talkover 整合

**目的:** ユーザー割り込み時のみ clear_queue 例外。Phase10 battle 機能を方式2 VAD に整合。

### Phase10 → 新設計マッピング（実装 SSOT）

| Phase10 | 新設計 | 備考 |
| --- | --- | --- |
| `mic_vad` 無音検知 → mic 送信停止 | 同左 + **`activity_end=True`** | 通常ターン終了 |
| `audio_stream_end_per_turn` | **`activity_end`** に置換 | テキストトリガーとセットで廃止 |
| `battle_talkover` cut-in | mic 停止 → **`activity_end`** → 割り込み例外 **`clear_queue`** → 進行中 chunk タスク cancel | tail 中は clear 禁止 |
| `battle_interrupt` テキスト送信 | Phase 1 方針に従い整理（emo 制御のみ残すか要確認） | Phase 0/1 報告を参照 |
| `drop_initial_audio_ms` | **撤去済みであること** | Phase 1 で確認 |

**Pass 基準:**
- [ ] 通常ターン: tail drain 前に clear_queue しない
- [ ] ユーザー割り込み時のみ clear_queue + 描画キュークリア
- [ ] talkover cut-in 後、新ターンが正常開始
- [ ] `generation_complete` 前後で clear_queue しない

**主な対象:** `run_mic_input_obs_realtime_session_loop.py`, `dev_audio_chunk_player_persistent.py`

**詳細実装仕様:** Phase 3 Pass 後、親チャットが本セクションを見て Phase 4 子プロンプトを生成する。Phase 0/1 の報告で emo/interrupt 方針が確定していれば、ここに追記する。

**子チャット報告:** （ここにサマリーを貼る）

---

## Phase 5: VirtualCam SSOT

**目的:** 表示 frame = 再生中 audio_ms。

**Pass 基準:**
- [ ] virtualcam が player の audio_ms と同期
- [ ] 口形ズレ・古い frame 残留なし
- [ ] `audio_ms` と無関係な sequential frame 消費なし

**主な対象:** `run_virtualcam_persistent.py`, session_loop M0/描画連携

**子チャット報告:** （ここにサマリーを貼る）

---

## Phase 5b: fast_inmemory（任意）

**前提:** Phase 5 Pass かつ `--no-fast_inmemory` で Phase 6 相当が Pass 済み。

**Pass 基準:**
- [ ] `--fast_inmemory` ON で Phase 6 相当テスト Pass
- [ ] 既存分岐構造を破壊していない

**子チャット報告:** （ここにサマリーを貼る）

---

## Phase 6: 総合 E2E

**Pass 基準:**
- [ ] 連続多ターン会話 Pass
- [ ] 割り込み + talkover Pass
- [ ] tail drain 完了ログ確認
- [ ] 主要メトリクス（queue_wait_ms, total_ms）が許容範囲

**子チャット報告:** （ここにサマリーを貼る）

---

## 変更履歴

| 日付 | 内容 |
| --- | --- |
| 2026-07-24 | 初版作成（Phase 0–6 定義、Phase 4 talkover マッピング含む） |
