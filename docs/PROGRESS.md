# Realtime Lip-Sync 再構築 — 進捗管理（SSOT）

> **親チャット（司令塔）がこのファイルを更新する。** 子チャットは直接編集しない。

## ベースライン

| 項目 | 値 |
| --- | --- |
| 起点 commit | `e4c1204`（Phase10 STEP1 stable restore before STEP2 retry） |
| 起点 tag | `phase10-local-vad-baseline` |
| マイルストーン tag | `phase2-pass` … `phase8-pass` / `phase9-pass`（任意） |
| 作業ブランチ | `feature/local-vad-restore` |
| 復元対象（session_loop） | `git checkout e4c1204 -- scripts/live_runtime/run_mic_input_obs_realtime_session_loop.py` |
| 未 commit 3 ファイル | 解消済み（Phase 0 監査時点で HEAD blob = e4c1204） |
| 設計 SSOT | `.cursorrules`（原則）+ `docs/ARCHITECTURE.md`（構造） |
| 子セルフ検証 | 各 Phase 完了前に子が検証コマンド実行。親はサマリーのみ確認（diff/ログ全文は読まない） |

## フェーズ一覧

| Phase | 名称 | 状態 | Pass 日 |
| --- | --- | --- | --- |
| 0 | ベースライン監査・計画 | `pass` | 2026-07-24 |
| 1 | ローカル VAD 完全化 | `pass` | 2026-07-24 |
| 2 | 一本道並列（session_loop + step1） | `pass` | 2026-07-24 |
| 3 | データ量 jitter（audio_player） | `pass` | 2026-07-25 |
| 4 | 割り込み・talkover 整合 | `pass` | 2026-07-25 |
| 5 | VirtualCam SSOT | `pass` | 2026-07-25 |
| 5b | fast_inmemory（任意） | `pass` | 2026-07-26 |
| 6 | 総合 E2E | `pass` | 2026-07-26（Pass-with-defer） |
| 7 | 性能（O(N²)/lock/供給）（任意） | `pass` | 2026-07-26（Pass-with-defer） |
| 8 | 品質・供給遅延（任意） | `pass` | 2026-07-26（Pass-with-defer） |
| 9 | M0 latency（任意） | `pass` | 2026-07-27 |
| 10 | 供給ストレス / inmemory ON ゲート | `pass` | 2026-07-27（Pass-with-defer） |
| 11 | battle / event 方式2 回帰（予約） | `pending` | —（今やる） |
| 12 | 待機モーション再配線（予約） | `pending` | — |
| 13 | マルチ M0（予約・将来 Go） | `pending` | —（11/12 後。今は未着手） |

状態値: `pending` / `in_progress` / `pass` / `blocked`

---

## Phase 0: ベースライン監査・計画

**目的:** ロールバック後コードの現状把握。コード変更なし。

**Pass 基準:**
- [x] 現行 vs `e4c1204` の差分表（4 ファイル）
- [x] 旧 VAD 混在箇所リスト（`automatic_activity_detection`, `drop_initial_audio_ms`, テキストトリガー等）
- [x] 3 ファイルのボトルネック候補リスト
- [x] Phase 1 着手 Go/No-Go 判断

**子チャット報告:**
- 対象 4 ファイルは `e4c1204` と blob 完全一致。working tree clean（「未 commit 3 ファイル」は解消済み）。
- 旧経路: `automatic_activity_detection` 未明示（SDK server VAD 残存しうる）、`activity_start/end` 未使用、`drop_initial_audio_ms`（default 120）、テキスト `response_trigger` 主経路、`audio_stream_end_per_turn`。
- 欠落: 本番にクライアント RMS `mic_vad` なし（固定 `mic_send_max_s`）。ARCHITECTURE の「mic_vad」記述と実装乖離 → Phase1 は残骸撤去＋方式2 新規配線。
- ボトルネックは Phase2–5 領域（ファイル監視 / 即再生 / sequential frame）。Phase1 阻害なし。
- **Go → Phase 1**

---

## Phase 1: ローカル VAD 完全化

**目的:** server VAD 混在を解消し、方式2（クライアント VAD）に一本化。

**設計決定:**
- `activity_start` / `activity_end` は **新規配線**（残すだけ禁止）。server VAD 旧分岐は無効化または削除。
- Pass は **1 ターン AI 音声返答 + activity ログ** 中心。リップシンク品質は要求しない。

**Pass 基準:**
- [x] `automatic_activity_detection=False`
- [x] `activity_start=True` / `activity_end=True` でターン完結（ログで確認）
- [x] `drop_initial_audio_ms` / テキストレスポンストリガー撤去または無効化
- [x] `audio_stream_end_per_turn` を `activity_end` ベースに置換（通常ターン）
- [x] 1 ターン E2E：ユーザー発話 → AI 音声返答（リップシンク品質は不問）

**主な対象:** `run_mic_input_obs_realtime_session_loop.py`

**子チャット報告:**
- AAD `disabled=True` + RMS `mic_vad_*` 新規配線 + `ActivityStart`/`ActivityEnd` でターン完結。
- 通常ターン: `response_trigger` default SKIP、`drop_initial_audio_ms` 強制 0、`audio_stream_end` 非送信。
- E2E 1ターン Pass（`logs/sess_phase1_client_vad_20260724_202421.log`）。セルフ検証すべてクリア。`clear_queue` 増加なし。
- 懸念（Pass 外）: 短無音で VAD 早め切れ／battle text interrupt は旧 trigger 依存 → Phase 4。
- 付随: `tools/phase1_play_synth_speech_to_cable.py`（任意ヘルパ）。
- **Pass → Phase 2**

---

## Phase 2: 一本道並列（session_loop + step1）

**目的:** チャンク内 KNN→M0→enqueue、チャンク間非ブロック並列。

**設計決定:**
- **supply_gap なし** = enqueue 時点で M0 PNG 存在（音声先行 enqueue 0 件）に限定。再生時 supply_gap / アンダーラン / 主観リップシンクは Phase 3 以降。
- チャンク間は並列開始可だが、**player への enqueue 順序は chunk_idx 到着順を厳守**（後続が先に M0 完了しても先行 chunk の enqueue 完了を待つ順序保証が必須）。
- テストは短い PCM ストリーム + ログベース Pass を主とする。`scripts/tools/verify_pipeline_logs.py` 最小版はこの Phase で作成後から義務化可。

**Pass 基準:**
- [x] チャンクごとに非同期起動されるパイプライン処理がある
- [x] チャンク内: KNN → M0 完了 → enqueue の順序ログ確認
- [x] チャンク N の M0 待ち中もチャンク N+1 が処理開始されること
- [x] `[sync][pipeline_chunk]` ログ出力（chunk_idx, knn_ms, m0_ms, enqueue_ms）
- [x] enqueue 時点 M0 PNG 存在（音声先行 enqueue 0 件）。再生時 gap は本 Phase 対象外
- [x] enqueue 順序 = chunk_idx 到着順

**主な対象:** `run_mic_input_obs_realtime_session_loop.py`, `run_mic_input_obs_realtime_step1.py`

**再始動メモ（2026-07-24）:**
- 子チャット Stop + Undo All。WIP 退避: commit `9e346eb` / tag `phase2-abort-backup` / `phase2-restart-base`（同一点）。
- **Step 0 Pass:** 不完全 Phase2 除去 → Clean Phase1 base → 本実装。
- **Phase 2 Pass:** 図A chunk task + 到着順 enqueue（pending 辞書・Condition）。オフライン順序 PASS + 実機 `sess_phase1_client_vad_20260724_223932` で verifier PASS（AUDIO_BEFORE_M0=0）。方式2 維持。
- tag: `phase2-pass` 付与済み（2026-07-24）。
- 申し送り Phase 3: player underrun 多数 → データ量 jitter。末尾 `ENQUEUE_BLOCKED` は音声先行せず（設計通り・tail 監視）。Tee-Object は UTF-16 → 検証は UTF-8 変換後。
- 成果物: `scripts/tools/verify_pipeline_logs.py`, `phase2_pipeline_order_offline.py`

**子チャット報告:** Phase 2 Pass（オフライン + 実機 verifier）。詳細は親チャット 2026-07-24。

---

## Phase 3: データ量 jitter（audio_player）

**目的:** M0 PNG 蓄積量トリガーで再生開始。固定 sleep 待機に依存しない。

**Pass 基準:**
- [x] M0 PNG 相当 **300–500ms**（CLI 可変）蓄積で再生開始
- [x] 蓄積不足時 **~1000ms** フォールバックで強制開始（無限ブロックなし）
- [x] 連続再生でアンダーランなし（Hotfix 後: 開始直後の fallback 誤爆連鎖は実機解消）
- [x] ジッターバッファー値と M0 ハング救済タイムアウトが混同されていない

**主な対象:** `dev_audio_chunk_player_persistent.py`, session_loop の player 連携部分

**状態メモ（2026-07-25）:**
- 初回: オフライン Pass も実機で微小片（samples=1）が `start_fallback` 誤爆 → Pass 保留 → Hotfix。
- Hotfix Pass: `TINY_PCM_DROP` で時計非搭載。実機 `PLAYOUT_START initial_buffer_ready pending_ms=370 target_ms=300`。`start_fallback` 誤爆 0。開始直後無音連鎖解消（`sess_phase1_client_vad_20260725_150506`）。
- tag: `phase3-pass` 付与済み（2026-07-25）。
- 申し送り（Pass 外）: ターン末途切れ／mid REBUFFERING（M0 700–1200ms vs jitter 300ms）／末尾 `ENQUEUE_BLOCKED`／tail の `AUDIO_BEFORE_M0` 疑い・task_cleanup。buffer 400–500ms 試験や tail 完了待ちは後続。**実 enqueue の音声先行は禁止のまま**（悪化させない）。
- ターン間 mic 待ちは運用/VAD メモ。

**子チャット報告:** Phase 3 Hotfix Pass。詳細は親チャット 2026-07-25。

---

## Phase 4: 割り込み・talkover 整合

**目的:** ユーザー割り込み時のみ clear_queue 例外。Phase10 battle 機能を方式2 VAD に整合。

### Phase10 → 新設計マッピング（実装 SSOT）

| Phase10 | 新設計 | 備考 |
| --- | --- | --- |
| `mic_vad` 無音検知 → mic 送信停止 | 同左 + **`activity_end=True`** | 通常ターン終了 |
| `audio_stream_end_per_turn` | **`activity_end`** に置換 | テキストトリガーとセットで廃止 |
| `battle_talkover` cut-in | mic 停止 → **`activity_end`** → 割り込み例外 **`clear_queue`** → 進行中 chunk タスク cancel | tail 中は clear 禁止 |
| `battle_interrupt` テキスト送信 | Phase 1: 通常ターンの text `response_trigger` は SKIP。battle の text interrupt/control は旧 trigger 合成依存が残存 → **本 Phase で方式2に整合**（emo 制御のみ残すか、activity ベースに置換。勝手に通常ターンへ trigger 復活させない） | Phase 1 懸念の清算 |
| `drop_initial_audio_ms` | **撤去済み**（強制 0） | Phase 1 確認済み |

**Pass 基準:**
- [x] 通常ターン: tail drain 前に clear_queue しない
- [x] ユーザー割り込み時のみ clear_queue + 描画キュークリア
- [x] talkover cut-in 後、新ターンが正常開始
- [x] `generation_complete` 前後で clear_queue しない

**主な対象:** `run_mic_input_obs_realtime_session_loop.py`, `dev_audio_chunk_player_persistent.py`

**詳細実装仕様:** Phase 3 Pass 済み。下記を遵守。
- 通常ターンは tail drain 完了まで `clear_queue` 禁止。
- 割り込み例外のみ clear（player + 描画キュー）+ 進行中 chunk タスク cancel。
- Phase 3 申し送りの mid REBUFFERING / 末尾途切れを「clear で隠す」のは禁止。
- tail の `AUDIO_BEFORE_M0` / `ENQUEUE_BLOCKED` を、音声先行 enqueue 許容で解消しない。

**子チャット報告:**
- 実機 `sess_phase4_talkover_20260725_155517`（talkover cut-in + interrupt file immediate）。
- 通常 clear 0／talkover のみ `clear_queue_sent` + draw_queue_cleared + chunk_tasks_cancelled=6。
- その後 turn=2..4 正常（ACTIVITY×4, SKIP×4）。`generation_complete` は marker_only no_clear_queue×4。
- 方式2 / 図A / Phase3 jitter 保護。clear で REBUFFERING を隠していない。
- 申し送り: cleared_samples=0（投入タイミング）／口先行→Phase5／mid REBUFFERING 継続。
- tag: `phase4-pass` 付与済み（2026-07-25）。
- **Pass → Phase 5**

---

## Phase 5: VirtualCam SSOT

**目的:** 表示 frame = 再生中 audio_ms。

**Pass 基準:**
- [x] virtualcam が player の audio_ms と同期
- [x] 口形ズレ・古い frame 残留なし
- [x] `audio_ms` と無関係な sequential frame 消費なし

**主な対象:** `run_virtualcam_persistent.py`, session_loop M0/描画連携

**子チャット報告:**
- `playback_state.json`（player_local_ms）公開 + virtualcam が audio_ms 照合。`sequential_idx=disabled`。連番/glob 消費廃止。
- 証拠: `sess_phase5_virtualcam_ssot_20260725_174249`（`[sync][virtualcam]`、audio_ms→frame）。主観追加: `sess_phase1_client_vad_20260725_220657`（4 turns、フリーズ/古い口形なし）。
- 方式2/図A/Phase3/Phase4 保護。Keep All 済み。tag: `phase5-pass` 付与済み（2026-07-25）。
- 申し送り: mid REBUFFERING/UNDERRUN 残（供給遅延・Phase3 申し送り）。clear で隠していない。step1 単体 virtualcam は未配線（session_loop 本線のみ）。
- **Pass → Phase 6**（5b は Phase 6 の `--no-fast_inmemory` Pass 後）

---

## Phase 5b: fast_inmemory（任意）

**前提:** Phase 5 Pass かつ Phase 6 が `--no-fast_inmemory` で Pass（本プロジェクトは Pass-with-defer 済み）。

**スコープ決定（2026-07-26）: 案 C 相当**
- **Phase 5b = `--fast_inmemory` ON Pass のみ**（`.cursorrules` §6 に一致）
- O(N²) / m0_lock / 供給遅延の Fix は **含めない** → **Phase 7**
- 理由: O(N²) は `--no-fast_inmemory` でも発生。inmemory と性能を同一 Phase にすると Pass/Fail が混線する

**Pass 基準:**
- [x] `--fast_inmemory` ON で Phase 6 相当テスト Pass（多ターン厳しい場合は原因記録＋親へ defer 可否）
- [x] 既存分岐構造を破壊していない（ON/OFF 両方報告）
- [x] m0_ms 分解・図A・VirtualCam audio_ms が ON でも維持

**子チャット報告:**
- ON: flush 省略＋streamer raw 投影、`knn_inmemory`（full rescan・O(N²) Fix なし）、`skip_archive_pcm`（write-on-demand）。証拠 `[fast_inmemory][ENABLED]`。
- ON ログ `sess_phase5b_inmemory_on_20260726_153931` turns=4 OK。OFF `..._154051` 非破壊。
- 図A 実 enqueue 先行 0 / VCam audio_ms / 分解ログ維持 / 通常 clear 0。性能 Fix なし。
- Phase 6 defer 残（REBUFFERING / ENQUEUE_BLOCKED / knn early→late / m0_lock）→ Phase 7。Keep All 済み。
- 主観 `sess_phase1_client_vad_20260726_154941`: ON で音声+口形。詰まりは defer 同系。
- tag: `phase5b-pass` 付与済み（2026-07-26）。
- **Pass → Phase 7 任意**

---

## Phase 6: 総合 E2E

**設計決定（2026-07-26 開始前申し送り）:**
- **m0_ms 分解ログ**を追加し、遅延の内訳を観測可能にする（Fix 目的ではない）。
- O(N²) 全件再スキャンは調査済み既知ボトルネック。Phase 6 では **観測のみ**。Fix は **Phase 7**（旧「5b+」表記を訂正）。
- 疑い（Phase5 ログ: chunk 増に伴い m0_ms/knn_ms 増大）: A) 毎チャンク全履歴 KNN 再実行、B) step1 の timeline 全件スキャン系。
- 多ターン Fail 時は原因記録＋ **defer 可**。無理な Hotfix で Pass しない。
- テストは `--no-fast_inmemory`。

**Pass 基準:**
- [x] 連続多ターン会話 Pass（Fail 時は原因記録＋defer 可。無理な Hotfix 禁止）→ **defer**（完走だが REBUFFERING/UNDERRUN/末尾ガード）
- [x] 割り込み + talkover Pass
- [x] tail drain 完了ログ確認（通常 clear=0 で確認。明示 tail_drain ログは未実装）
- [x] 主要メトリクス（queue_wait_ms, total_ms、および m0_ms 分解）を報告。許容外は隠さず記録

**子チャット報告:**
- m0_ms 分解キー追加（wait_mouth / lock / slice / disk / req_send / png_wait / verify / other）。helper: `phase6_metrics_summary.py`。
- 多ターン: 論理完走×4 だが Fail+defer（REBUFFERING/UNDERRUN、末尾 AUDIO_BEFORE_M0 ガード発火も enqueue 阻止）。Hotfix なし。
- talkover Pass。通常 clear=0。VirtualCam audio_ms。`--no-fast_inmemory`。
- 観測: knn_ms chunk 増で増大（疑い A）。m0_slice 軽微。m0_lock_ms + png_wait が支配的。
- 主観追加 `sess_phase1_client_vad_20260726_150119`: turn1 first_audio timeout、turn2–4 音声+口形。Keep All 済み。tag: `phase6-pass`。
- **Pass-with-defer → 任意で Phase 5b（inmemory）または Phase 7（性能）**

---

## Phase 7: 性能（O(N²) / lock / 供給）（任意）

**前提:** Phase 6 Pass（Pass-with-defer 可）。Phase 5b（inmemory）とは独立。順序はどちら先行でもよい。

**目的:** Phase 6 で defer した供給・計算量ボトルネックを、不変条件を壊さず改善する。

**対象テーマ（優先順は子が計測で決めてよい）:**
1. O(N²) 疑い A: knn_ms が chunk 増で増大（毎チャンク全履歴 KNN 再実行）→ incremental 等
2. m0_lock_ms: 並列 chunk 間の M0 lock 待ち
3. 供給遅延: REBUFFERING/UNDERRUN / 末尾 ENQUEUE_BLOCKED（音声先行 enqueue や clear 隠蔽で直さない）
4. turn1 first_audio timeout（起動遅延）監視・必要なら最小修正

**Pass 基準:**
- [x] knn_ms の chunk 増に対する単調悪化が改善（early vs late の比または傾きを報告）→ offline/live 改善確認
- [x] m0_ms 分解で lock/png_wait 等の改善または説明可能なトレードオフ → 構造は defer、トレードオフ説明済み
- [x] 多ターンで REBUFFERING/UNDERRUN が改善、または残存理由が計測で説明可能 → OFF REBUFFERING 10→6。残は defer
- [x] 図A / 方式2 / Sync SSOT / jitter / 割り込み例外 / fast_inmemory 分岐を壊していない
- [x] AUDIO_BEFORE_M0 実 enqueue 0（ガード発火は許容）
- [x] 改善が難しい項目は原因記録＋ defer 可

**子チャット報告:**
- knn: GTキャッシュ＋delta-only incremental（維持）。
- Hold 原因: exact PNG 欠落時の `idle_hold` 固着 → Hotfix `SSOT_CATCHUP`（audio_ms 以下の最新 PNG。未来 frame なし）。
- Hotfix After OFF `sess_phase1_client_vad_20260726_180003`: SSOT_WAIT 57→**0**、固着 35→**0**、CATCHUP=41。knn 23→40（1.8×、Phase6 2.9×より良）。主観大幅改善。
- defer: m0_lock/png_wait、残 REBUFFERING/軽い途切れ、ON 短確認未実施。
- tag: `phase7-pass` 付与済み（2026-07-26）。
- **Pass-with-defer → Phase 8（品質・供給遅延）任意**

---

## Phase 8: 品質・供給遅延（任意）

**前提:** Phase 0–7 完了（アーキテクチャ目的は達成）。本 Phase は **主観品質・供給遅延** の改善（M1 側）。

**スコープ決定（2026-07-26）:**
- 主対象: `run_mic_input_obs_realtime_step1.py`（疑い B: timeline 全件スキャン系）、**m0_lock 低減（M1 の lock 粒度・冗長呼び出しのみ）**、供給遅延（REBUFFERING/UNDERRUN）
- 主テスト: **`--no-fast_inmemory`**（本番優先）。fast_inmemory 分岐は維持・破壊禁止。ON は任意の短確認のみ
- 図A / Sync SSOT / 方式2 / jitter / 割り込み例外 / Phase7 knn incremental・SSOT_CATCHUP を壊さない
- 無理な Hotfix 禁止（音声先行 enqueue / clear 隠蔽 / 正常 M0 短タイムアウト打ち切り）
- **対象外:** M0 リポ修正・pose_transform 軽量化・FG PNG 往復削減・フル画像キャッシュ・Worker プール／マルチ M0 → **Phase 9（M0 latency）へ予約**
- **m0_lock:** 単一 Worker のままでは png_wait 起因の待ちは残りうる。計測で「lock ≈ png_wait」と判明したら Phase 9 へ defer 可（Pass-with-defer 可）

**Pass 基準:**
- [x] **主観改善** または REBUFFERING/UNDERRUN 明確改善 → 主観やや改善（`224337`）。REBUFFERING 6→14 は未達・Phase9
- [x] m0_lock_ms 改善または Phase 9 defer → M1 lock 粒度（queue_wait 153→0.7）+ png_wait 支配で Phase9 defer
- [x] 疑い B を閉じた → 非支配 + bisect
- [x] 図A / Sync SSOT / 方式2 / Phase7 成果維持（SSOT_WAIT=0、AUDIO_BEFORE 実 enqueue 0）
- [x] png_wait＝実レンダー支配 → Phase 9 へ defer（Pass-with-defer 可）

**子チャット報告:**
- step1: timeline slice bisect（疑い B 非支配で閉鎖）。`worker_owner` で slice/disk/png_wait 中に state lock 解放、covered peer 早期 return→enqueue。
- Cable `222230` / 実mic `224337`。queue_wait 大幅改善。REBUFFERING 未改善（png_wait≫jitter）。
- M0 リポ未編集。Worker プール・短タイムアウトなし。
- **Pass-with-defer（2026-07-26）。Keep All 可。Phase 9 = M0 latency（png_wait）。**
- tag: `phase8-pass` 付与済み（2026-07-26）。

---

## Phase 9: M0 latency（任意）

**前提:** Phase 8 Pass-with-defer。M1 側の疑い B / lock 粒度は完了。残る本丸は **単一 M0 Worker の実レンダー時間（png_wait）**。

**スコープ:**
- 主対象リポ: `C:\dev\M0_session_renderer_final_1`（render / runner / persistent worker 等。子が読んで特定）
- M1（`session_loop` / `step1` / player / virtualcam）は **計測・配線の最小変更のみ可**。図A / Sync SSOT / 方式2 / Phase7–8 成果を壊す再設計は禁止
- テーマ候補（計測で優先度決定）: 幾何変換コスト、FG PNG 往復、不要な再レンダー、キャッシュ（フル画像プリロードは費用対効果を計測してから）、Worker 内最適化。**プロセスプール／マルチ M0 は設計影響大のため、やるなら根拠＋親エスカレーション必須**
- 主テスト: M1 経由 E2E **`--no-fast_inmemory`**。Before 比較: Phase8 `sess_phase1_client_vad_20260726_224337`（または同等 OFF）
- **FG PNG:** ディスク経由をやめる変更は VirtualCam（または同等受け口）とセット。M0 単独で dump を切って Pass しない
- **非本命:** realtime は `write_mp4=False` のため ffmpeg mux 除去は主効果になりにくい
- **計測ファースト:** `m0_frame` を load/resize/transform/paste/imwrite 等に分解してから最大項を切る

**禁止:**
- 音声先行 enqueue / clear 隠蔽 / 正常 M0 短タイムアウト打ち切り
- Phase 8 と同一 Pass に混ぜない（本 Phase は M0 latency）
- M1 の knn incremental / SSOT_CATCHUP / worker_owner lock yield を不用意に巻き戻さない
- `affine_points` の無断有効化、根拠なしフルプリロード必須化

**Pass 基準:**
- [x] png_wait（または同等の M0 応答待ち）が Before 比で明確改善（p50/p90 等）
- [x] REBUFFERING/UNDERRUN **または** 主観（後半途切れ・口形詰まり）が Phase8 比で改善
- [x] M1 不変条件維持（図A、AUDIO_BEFORE_M0 実 enqueue 0、SSOT_WAIT 回帰なし、通常 clear 0、方式2）
- [x] lock≈png_wait 構造の説明が Before/After で更新されている（lock 影が png 短縮に追随するか）
- [x] 改善困難な項目は原因記録＋ defer 可（無理な Hotfix / 無断マルチ M0 で Pass しない）

**子チャット報告:**
- 計測分解: paste≈11 / hook≈9 / transform≈5 → blit・raw BGRA（VirtualCam セット）・近零 early-out。
- M0: `render_core` / `m0_runner`（`fg_format=bgra`）。M1 配線: virtualcam `.bgra`＋step1。図A/VAD/jitter 未変更。
- After: m0_frame p50≈9.5ms、png_wait p90 658→~380–392、REBUFFERING 14→3。Cable `135832`＋実mic `141053` 主観ほぼ正常。
- lock 影は残るが絶対値追随。マルチ M0 なし。残 REBUFFERING=3 / 末尾 ENQUEUE_BLOCKED は軽微。
- **Pass（2026-07-27）。Keep All 可。**
- 次ライン: Phase 10（供給ストレス / inmemory ON ゲート）→ 11/12 は予約

---

## Phase 10: 供給ストレス / inmemory ON ゲート

**前提:** Phase 0–9 完了。目的は **マルチプロセス実装ではなく要否判断のための定量データ**。

**注意:** 旧「Phase10 STEP1」ベースライン（tag `phase10-local-vad-baseline`）とは別物。本 Phase は新規の供給ストレステスト。

**スコープ決定（2026-07-27）:**
- 順序固定: **(1) `--fast_inmemory` ON 同ボリューム再ベース → (2) 長文（縦）→ (3) 多ターン耐久（横）**
- 比較軸: Phase9 OFF（例: `sess_phase1_client_vad_20260727_141053`）を残す。ON は M1 I/O 層（`png_wait` ゼロ前提にしない）
- 負荷再現: Live の「文数」は保証されない → `voice_s` / 固定シナリオ等で負荷を再現可能にする
- **禁止:** マルチ M0 実装、音声先行 enqueue、clear 隠蔽、図A/方式2/SSOT 破壊。無理な Hotfix で Pass しない

**Pass 基準:**
- [x] **Step1（ON 同ボリューム）:** Phase9 と同程度の負荷で `--fast_inmemory` ON。disk/queue_wait/後半 lock ドミノの改善または説明。不変条件維持
- [x] **Step2（長文・縦）:** REBUFFERING/UNDERRUN・png_wait/m0_ms の単調悪化を報告 → **マルチ M0 Go/No-Go を子が提案、親が承認**（実装は含めない）
- [x] **Step3（多ターン・横）:** 目安 10 ターンでターン間蓄積なし（または蓄積理由を記録）。マルチ判断とは分離可
- [x] 全 Step で不変条件維持（AUDIO_BEFORE_M0 実 enqueue 0、SSOT_WAIT 回帰なし、通常 clear 0、方式2、図A）

**子チャット報告:**
- 本番 `live_runtime/` コード変更なし。計測ヘルパ `tools/phase10_*` のみ。
- Step1 ON: knn/queue_wait/lock は P9 OFF 並み〜改善。png_wait は残（別レイヤ）。不変条件 OK。主観も概ね正常・末尾途切れ軽微。
- Step2: knn/m0/lock 単調悪化、後半描画停滞→AUDIO_BEFORE_M0 block（enqueue せず）。Live 長尺再現は弱いが固定台本で縦飽和は定量化。
- Step3: turns=10、ターン間 queue_wait 蓄積なし。
- 子マルチ提案: Conditional。**親確定（2026-07-27）: 将来 Go（運用で AI 2–3文以上想定）・今は未着手。実装順は Phase11→12→13。**
- **Pass-with-defer。Keep All 可（計測ヘルパ）。**

**運用判断（親）:**
- マルチ M0 実装は本 Phase に含めない → **Phase 13（予約）**
- 次: Phase 11（battle/event）→ Phase 12（待機モーション）→ Phase 13（マルチ M0）

---

## Phase 11: battle / event 方式2 回帰

**前提:** Phase 10 Pass。単一 Worker のまま機能・SSOT・割り込みを固める（マルチより先）。

**参照:** `docs/battle_runtime_talkover_ops.md` / `docs/event_runtime_ops.md`
**想定:** Phase4 talkover は一部済み。本番の主導権・event 動画・battle 経路を方式2（activity_start/end、通常 clear 禁止）で通し確認・必要最小修正。

**Pass 基準（骨子）:**
- [ ] battle / talkover / interrupt が方式2 で通し Pass（通常ターン clear 0、割り込み時のみ clear）
- [ ] event 動画経路が現行パイプラインで破綻しない（詳細は子プロンプト）
- [ ] 図A / Sync SSOT / jitter / Phase9–10 成果を壊さない
- [ ] マルチ M0 を導入しない

**子チャット報告:** （ここにサマリーを貼る）

---

## Phase 12: 待機モーション再配線（予約）

**状態:** 予約。Phase 11 後。
**想定:** 旧 Phase10 STEP2 idle silent PCM 相当を方式2 に再配線。相手発話中の BGV 停止を解消。Sync SSOT / jitter / VAD に触る。

---

## Phase 13: マルチ M0（予約・将来 Go）

**状態:** 予約。**要否は Go（将来必須）・今は未着手。** Phase 11/12 完了後に親が詳細プロンプトを定義。
**根拠（Phase10）:** 短〜中・多ターンは単一で実用可。連続長尺（単ターン実 M0 20+ chunk / 音声 8–10s 超、運用上は AI 2–3文以上）では単一 Worker が縦飽和。
**禁止（今）:** Phase 10–12 にマルチ実装を混ぜない。

---

## 変更履歴

| 日付 | 内容 |
| --- | --- |
| 2026-07-24 | 初版作成（Phase 0–6 定義、Phase 4 talkover マッピング含む） |
| 2026-07-24 | `docs/ARCHITECTURE.md` 追加に伴い SSOT 参照を更新 |
| 2026-07-24 | Phase 0 Pass。4 ファイル=e4c1204 一致。方式2 は新規配線必要。Phase 1 Go |
| 2026-07-24 | 子セルフ検証運用・tag `phase10-local-vad-baseline`・Phase1/2 設計決定を SSOT 反映 |
| 2026-07-24 | Phase 1 Pass。方式2 VAD + activity_start/end。Phase 2 Go |
| 2026-07-24 | Phase 2 abort。WIP=`9e346eb` / `phase2-restart-base`。Step0 付き再始動 |
| 2026-07-24 | Phase 2 Step 0 Pass。Clean Phase1 base。本実装 Go |
| 2026-07-24 | Phase 2 Pass。図A+到着順enqueue。underrun→Phase3。Phase 3 Go |
| 2026-07-25 | Phase 3 Pass 保留。実機微小chunkが start_fallback 誤爆。Hotfix 指示 |
| 2026-07-25 | Phase 3 Hotfix Pass（TINY_PCM_DROP + initial_buffer_ready）。Phase 4 Go |
| 2026-07-25 | Phase 4 Pass（talkover clear 例外 + generation_complete marker_only）。Phase 5 Go |
| 2026-07-25 | Phase 5 Pass（VirtualCam audio_ms SSOT）。REBUFFERING 残→Phase6監視。Phase 6 Go |
| 2026-07-26 | Phase 6 開始前: m0_ms 分解ログ必須。O(N²) は観測のみ（Fix=5b+）。多ターン Fail は defer 可 |
| 2026-07-26 | Phase 6 Pass-with-defer。分解ログ・talkover OK。多ターン供給遅延 defer。5b 任意 |
| 2026-07-26 | スコープ分割: Phase 5b=fast_inmemory のみ。O(N²)/lock/供給 Fix → Phase 7 |
| 2026-07-26 | Phase 5b Pass（inmemory ON/OFF）。性能 defer は Phase 7 へ |
| 2026-07-26 | `phase5b-pass` tag。Phase 7 子プロンプト投下 |
| 2026-07-26 | Phase 7 Hold。knn O(N²)改善は確認。SSOT_WAIT/主観劣化で Keep All 禁止 |
| 2026-07-26 | Phase 7 Pass-with-defer。SSOT_CATCHUP で固着回帰解消。Keep All 可 |
| 2026-07-26 | `phase7-pass` tag。Phase 8（品質・供給遅延）追加・子プロンプト投下 |
| 2026-07-26 | Phase 8 境界: M0 リポ/Worker プールは対象外→Phase 9 予約。m0_lock は M1 粒度のみ |
| 2026-07-26 | Phase 8 Pass-with-defer。疑いB閉鎖・M1 lock改善。REBUFFERING/png_wait→Phase9 |
| 2026-07-26 | `phase8-pass` tag。Phase 9（M0 latency）Pass 基準定義・子プロンプト投下 |
| 2026-07-26 | Phase 9 追記: FGは受け口セット、ffmpeg非本命、計測分解ファースト |
| 2026-07-27 | Phase 9 Pass。m0_frame/png_wait/REBUFFERING 改善。BGRA+VirtualCam セット。Keep All 可 |
| 2026-07-27 | Phase 10–12 追加。Phase 10=供給ストレス/inmemory ON ゲート（今やる）。11/12 予約 |
| 2026-07-27 | Phase 10 Pass-with-defer。マルチ=将来Go・未着手→Phase13。次=Phase11 |
