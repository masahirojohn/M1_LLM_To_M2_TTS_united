# Realtime Lip-Sync 再構築 — 進捗管理（SSOT）

> **親チャット（司令塔）がこのファイルを更新する。** 子チャットは直接編集しない。

## ベースライン

| 項目 | 値 |
| --- | --- |
| 起点 commit | `e4c1204`（Phase10 STEP1 stable restore before STEP2 retry） |
| 起点 tag | `phase10-local-vad-baseline` |
| マイルストーン tag | `phase2-pass` … `phase14-idle-bg`（`cf8ec05`） / `phase16-pass`（`6fc6507`） |
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
| 11 | battle / event 方式2 回帰 | `pass` | 2026-07-28 |
| 12 | 待機モーション再配線 | `pass` | 2026-07-28（Pass-with-defer） |
| 13 | マルチ M0 | `pass` | 2026-07-29（Pass-with-defer） |
| 13hf | claim/coverage Hotfix | `pass` | 2026-07-29（Pass-with-defer） |
| 14 | 中盤供給 / 末尾 IDLE_BG（任意） | `pass` | 2026-07-30（部分: IDLE_BGのみ。中盤 Fail→リバート） |
| 15 | wait_mouth（mouth×claim）（任意） | `pass` | 2026-07-30（Pass-with-defer。レース revert） |
| 16 | N スケール計測 A/B（任意） | `pass` | 2026-07-30（Pass-with-defer。N↑効果なし→default N=2） |
| 17 | mouth×claim 前線切り分け（任意） | `pass` | 2026-07-30（A=M3。本番 Fix なし） |
| 18 | M3 mouth 前線先行（任意） | `in_progress` | — |

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
- tag: `phase10-pass`（commit `7cd325e`）付与済み（2026-07-28）。

**運用判断（親）:**
- マルチ M0 実装は本 Phase に含めない → **Phase 13（予約）**
- 次: Phase 11（battle/event）→ Phase 12（待機モーション）→ Phase 13（マルチ M0）

---

## Phase 11: battle / event 方式2 回帰

**前提:** Phase 10 Pass。単一 Worker のまま機能・SSOT・割り込みを固める（マルチより先）。

**参照:** `docs/battle_runtime_talkover_ops.md` / `docs/event_runtime_ops.md`
**想定:** Phase4 talkover は一部済み。本番の主導権・event 動画・battle 経路を方式2（activity_start/end、通常 clear 禁止）で通し確認・必要最小修正。

**Pass 基準（骨子）:**
- [x] battle / talkover / interrupt が方式2 で通し Pass（通常ターン clear 0、割り込み時のみ clear）
- [x] event 動画経路が現行パイプラインで破綻しない
- [x] 図A / Sync SSOT / jitter / Phase9–10 成果を壊さない
- [x] マルチ M0 を導入しない

**子チャット報告:**
- 本番 `live_runtime/` 差分なし。回帰ヘルパ `tools/_phase11_feeder_*.py` のみ（任意）。
- talkover: `sess_phase11_talkover_20260728_134631` — clear=talkover のみ→turn2–4、gen_complete marker_only。
- event: `sess_phase11_event_20260728_134748` — override→restore、clear reason=event_runtime、SSOT OK。
- 主観 UI: `sess_phase11_subj_20260728_144019` — 割り込み×2＋evt_001、音声・口形破綻なし。主導権は未実施（API 1007 教訓でスキップ→後続可）。
- 申し送り: ops docs が旧 soft cut-in のまま（親が更新可）。event clear は player のみで chunk cancel なし（今回破綻なし）。
- **Pass（2026-07-28）。Keep All 可（任意ヘルパ）。次=Phase 12。**
- tag: `phase11-pass`（commit `0a140da`）付与済み（2026-07-28）。

---

## Phase 12: 待機モーション再配線

**前提:** Phase 11 Pass。単一 Worker のまま。

**目的:** 旧 Phase10 STEP2 idle silent PCM 相当を方式2 に再配線し、相手発話中／待機中に BGV が止まったままになる問題を解消する。

**スコープ:**
- 待機モーション（無音 PCM または同等）の再生・停止と Sync SSOT / jitter / VAD 境界の整合
- 主テスト: `--no-fast_inmemory` 優先。方式2維持
- **禁止:** マルチ M0、音声先行 enqueue、clear 隠蔽、図A 破壊

**Pass 基準（骨子）:**
- [x] 待機中に BGV／待機モーションが意図どおり動く（相手発話中の固着解消または仕様どおりの挙動をログ＋主観で説明）
- [x] ターン開始／割り込み／event と衝突しない（idle 解除が正しい）
- [x] Sync SSOT（audio_ms）を壊さない。通常 clear 0
- [x] 方式2 / 図A / Phase11 battle・event を回帰させない

**子チャット報告:**
- idle silent PCM を図A（KNN→M0→enqueue）＋ just-in-time。VirtualCam は非 PLAYING 時も BG＋last FG（`IDLE_BG_ADVANCE`）。
- ログ: idle2 / talkover。主観: `212320`（待機中 M0+BGV）、`213108`（talkover/event 整合）。通常 clear 0、AUDIO_BEFORE 0、SSOT_WAIT 0。
- defer: 待機中に稀に FG 抜け（単一 M0 vs idle 40ms）。致命ではない → 間隔緩和 or Phase13。
- **Pass-with-defer（2026-07-28）。Keep All 可。次=Phase 13。**
- tag: `phase12-pass`（commit `ce903b9`）付与済み（2026-07-28）。

---

## Phase 13: マルチ M0

**前提:** Phase 12 Pass-with-defer。要否は Phase10 で **将来 Go** 確定。単一 Worker の縦飽和・idle 供給を容量で緩和する。

**ハード目安（2026-07-28）:** 開発機 CPU **6コア**。default **N=2**（CLI 可変）。**6コア全振り禁止**。N 増加は計測根拠付き。初期上限目安 **3〜4**（例: cores-2）。メモリ（Worker×スプライトキャッシュ）を報告。

**最重要（Pass 上位制約）:**
1. マルチ化で単一 Worker より遅くなる経路を作らない（重いコピー／シリアライズ／ディスク往復の増加を避ける）
2. メモリ枯渇させない（N 増加時の RSS を Before/After で報告）

**必須原則:** 常駐プール（ターン毎 spawn 禁止）／VAD ターンに合わせ Worker flush・リセット／親死亡で子が残らない teardown（talkover cancel 整合）／巨大フレームコピー回避（共有メモリは任意）／図A＋enqueue 到着順／参考 RR→並べ替え構成は必須採用しない（chunk/job ディスパッチ可）

**目的:** 単一 M0 Worker 瓶颈を、不変条件を壊さず並列化（または同等の容量拡張）で緩和する。

**Pass 基準（骨子）:**
- [x] 長尺／高 chunk 負荷で png_wait・m0_lock・REBUFFERING（または idle FG 抜け）が単一比で改善 → lock max / late m0 / REB・多ターン CATCHUP は改善。**長尺主観リップは未達**
- [x] 単一比で悪化する経路がない（レイテンシ／I/O／RSS）→ メトリクス上は非回帰。主観口は N=2 長尺で悪化寄り → Hotfix
- [x] 図A / 方式2 / SSOT / idle・talkover 骨格維持
- [x] 常駐プール・ターンリセット・親死亡 teardown
- [x] 残件 defer 可

**子チャット報告:**
- 常駐プール N=2（clamp 1..4）、chunk claim、watermark、TCP reset、parent_pid teardown。N=1 フォールバック。
- A/B: lock/REB/CATCHUP 改善。長尺 N=2 主観は口フリーズ・IDLE_BG 増（N=1 より悪化寄り）。多ターンはメトリクス改善・主観「もう一息」。
- **Pass-with-defer（2026-07-29）。Keep All 可。N=3〜4 即スケール保留。**
- tag: `phase13-pass`（M1 commit `7d314af`）付与済み。M0 連携 commit `142aa82`（reset / parent_pid、M0 側 tag なし）。
- 残件: IDLE_BG／FG 供給／SSOT_CATCHUP。**実装 Hotfix の前に分析専用子で N1/N2 切り分け** → 親が Hotfix/Phase14 を発行。

---

## Phase 13 分析（実装なし・ゲート）

**目的:** 長尺 N=1 vs N=2 から FG 供給不足の主因レイヤを特定し、N増 Go/No-Go と Hotfix 対象を提案する。コード変更禁止。

**子チャット報告（受理 2026-07-29）:**
- N=2 は両 Worker 実効並列。lock / late m0 / CATCHUP は改善。
- 長尺主観悪化の主因: **末尾 claim/coverage 停止 → `rendered_end` 固定 → AUDIO_BEFORE 増 → REBUFFERING → IDLE_BG 連打**（N2 IDLE_BG 5→13、AUDIO_BEFORE 2→18）。
- **N増: 保留**。次手: Phase13 Hotfix（先に claim/coverage、その後必要なら IDLE_BG 表示）。表示だけで供給穴を隠さない。

---

## Phase 13 Hotfix: claim/coverage → IDLE_BG

**前提:** Phase13 Keep All / `phase13-pass` 済み。分析受理済み。
**順序:** (1) 末尾 claim/coverage 停止の修正 (2) 必要なら IDLE_BG 表示。供給穴を表示で隠さない。
**禁止:** 音声先行 enqueue、N=3〜4、clear／短タイムアウト隠蔽、6コア全振り。

**Pass 基準:**
- [x] 長尺 N2: AUDIO_BEFORE 大幅減、`rendered_end` 不当固定の解消（18→1、claims 114→149、rendered ~17880）
- [x] 修正順序遵守（claim 先、IDLE_BG 表示未実施）
- [x] 実 enqueue 先行 0 / 通常 clear 0 / SSOT_WAIT 0 / プール非破壊
- [ ] 長尺主観口フリーズ完全解消 → **未達・defer**（多ターン≒2文は実用レベル）

**子チャット報告:**
- mouth_ready を t_ms ベースに。coverage wait を stop_event 即打ち切り廃止＋ inflight 一時解放＋ dispatcher catch-up。
- After 長尺 `180454` / 多ターン `180926`。多ターン主観ほぼ正常・CATCHUP 119→1。
- defer: 長尺 2–3文目フリーズ感（中盤 REB）、末尾 m0_tail_uncovered + IDLE_BG ~180ms、表示順2、微 AUDIO_BEFORE。
- **Pass-with-defer（2026-07-29）。Keep All 可。N↑保留。残件→Phase 14（新子）。**
- tag: `phase13-hotfix`（commit `d1a881b`）付与済み（2026-07-29）。

---

## Phase 14: 中盤供給 / 末尾 IDLE_BG（任意）

**前提:** Phase13 Hotfix Pass-with-defer。N=2 維持。N=3〜4 は無断ではしない（計測オプション・親エスカレーション）。
**目的:** 長尺中盤 REBUFFERING／口フリーズ残差と、末尾短ギャップ IDLE_BG 連打を改善する。
**運用目安:** 〜2文/ターンは N=2 で実務可。長尺 4–5文は残差あり。

**任意分析（受理 2026-07-29）:** 長尺 `180454`、中盤 player≈2.5–7.6s。
- 全 REB で **player == rendered_end（gap=0）** → coverage 穴ではなく **pending 枯渇**
- inflight 6–7 常時 → N=2 稼働だが完了が再生に負け
- 中盤 m0: **lock≈48% / wait_mouth≈41% / png_wait≈9%**。claim 欠番なし
- **主因:** M0 前進スループット負け（残差本体は `m0_lock_ms`＝cond/pool 待ち、次いで wait_mouth）
- 供給リードしばしば 40–80ms（rebuffer_target=240 未満）→ underrun

**修正順（確定）:** (1) 中盤＝lock/cond・mouth 待ちの最小改善 (2) 末尾 IDLE_BG。表示で中盤を隠さない。

**Pass 基準（骨子）:**
- [ ] 長尺 N2 中盤の口フリーズ／REBUFFERING が `180454` 比で改善、または主因を計測で閉じる → **Fail**（主観未改善。REB 15→12 微減のみ。lock↓／wait_mouth↑＝付け替え）
- [x] 末尾 IDLE_BG 同一 audio_ms 連打が改善（11→0）— **部分達成**
- [x] 不変条件維持（AUDIO_BEFORE/SSOT/clear）。無断 N↑なし
- [x] 多ターン: Turn2以降非回帰寄り。Turn1 無音は API/session（Phase14 劣化にしない）

**子チャット報告:**
- 実装: cond 即 return＋常時 inflight 解放＋ catch-up sleep／virtualcam IDLE_BG。
- After 長尺 `223938` / 多ターン `224458`。
- **親判定 2026-07-29: 中盤 Fail。方針 (A) — 中盤差分をリバートし IDLE_BG のみ残す。Keep All はリバート後。N↑しない。**
- 次ライン（wait_mouth 本体 or N=3 計測）は **新親チャット**へ。

**子チャット報告（リバート後・受理）:**
- step1 / session_loop → `d1a881b`（phase13-hotfix）へ復帰。
- virtualcam のみ Phase14 IDLE_BG 残存。
- **Keep All 可（virtualcam のみ）。** 中盤は Fail のまま。Phase14 部分成果＝末尾 IDLE_BG。
- tag: `phase14-idle-bg`（commit `cf8ec05`）付与済み（2026-07-30）。

---

## Phase 15: wait_mouth（mouth 準備 × M0 claim）（任意）

**前提:** Phase14 (A) 完了。作業ベース = tag `phase14-idle-bg`（`cf8ec05`）。中盤リバート済み・IDLE_BG のみ残存。**Phase14 中盤再実装・リバートやり直しは禁止。**

**目的:** 長尺中盤の残差本体である **`wait_mouth`（mouth 準備と M0 claim の噛み合わせ）** を改善する。N↑より先。

**運用目安（継続）:** N=2 で 〜2文/ターンは実務可。長尺中盤 REBUFFERING／口フリーズが残課題。

**スコープ:**
- 主対象: mouth 準備タイミングと M0 claim／coverage 待ちの噛み合わせ（`wait_mouth` 支配の削減または説明可能な閉鎖）
- 主テスト: **`--no-fast_inmemory`・N=2**。多ターン非回帰は短確認
- Before 比較:
  - Hotfix 後長尺 N2: `logs/sess_phase10_step2_subj_20260729_180454`（中盤残差の基準）
  - Phase14 Fail（反面教師）: `logs/sess_phase10_step2_subj_20260729_223938`（lock↓／wait_mouth↑＝付け替え失敗。同型禁止）
- **任意・後段（親承認後のみ）:** N=3 計測 A/B。default 据え置き・6コア全振り禁止。無断 N↑禁止
- Phase14 IDLE_BG（virtualcam）は維持・破壊禁止

**禁止（継続）:**
- 音声先行 enqueue / clear 隠蔽 / 正常 M0 短タイムアウト打ち切り / 固定 sleep で隠す
- 旧 fast_worker/slow_worker 分離の復活
- 図A・方式2・Sync SSOT・jitter・idle silent・talkover/event の破壊
- lock↔wait_mouth の付け替えだけで「改善」と称すること（total／供給リードも見る）
- 6コア全振り／根拠なし N 拡大／default N 変更

**Pass 基準（骨子）:**
- [x] 長尺 N2 で中盤 REBUFFERING／口フリーズが Before（`180454`）比で改善、**または** wait_mouth を計測で閉じた上で次手段を提案 → **後者**（主観未達・計測閉鎖）
- [x] wait_mouth 改善が lock への付け替えだけで終わっていない → レース単体は付け替え型ではないが **効果なし**。coalesce は付け替え＋悪化で破棄
- [x] 不変条件概ね維持（SSOT_WAIT=0、通常 clear 0、到着順、方式2、N=2／N=1）。AUDIO_BEFORE 末尾微増は尾部残差
- [x] 無断 N↑なし。残件 defer

**子チャット報告（受理 2026-07-30）:**
- 仮説レース（mouth event clear→wait 空待ち）修正を試行。オフラインで空待ち減は確認。実機長尺は効き痕跡 0。
- coalesce 試行は lock↓／wait↑（Phase14 同型）→破棄。
- 主観 `sess_phase10_step2_subj_20260730_131822`: 2文目途中〜末尾口フリーズ。REB 15→15、wait/lock/m0 横ばい。**改善なし**。
- IDLE_BG=0 は Phase14 維持。

**親判定（2026-07-30）:**
- **Pass-with-defer**（主観未達で閉じる。wait_mouth 小修正では天井超え不可と計測閉鎖）。
- **Keep All: レース修正は revert**（効き無し。「正しさのみ」も残さない。ベースを `phase14-idle-bg` に戻す）。
- coalesce／cond 付け替え系は再禁止。
- 次: **Phase 16 = N スケール計測 A/B**（N↑本線解禁。default は計測完了まで N=2 据え置き）。

---

## Phase 16: N スケール計測 A/B（任意）

**前提:** Phase15 Pass-with-defer。作業ベース = tag `phase14-idle-bg`（`cf8ec05`）相当（Phase15 レース差分は revert 済み想定）。プールの N 可変は既存。**図A破壊・default N 無断変更禁止。**

**目的:** 長尺中盤の供給負けが **N 増加で緩和されるか** を同一台本で定量 A/B し、default 変更の Go/No-Go を親に提案する。**本 Phase は計測が主。default を N=3/4 に上げる実装判断は親が A/B 後に行う。**

**スコープ:**
- 同一長尺台本・`--no-fast_inmemory` で **N=2（基準）→ N=3 → N=4**
- 既存プール CLI の N 指定を使う。必要なら CLI 上限を **4 まで**許可（6コア全振り禁止は維持）
- 見る指標: 中盤 REBUFFERING／供給リード／wait_mouth・lock・m0 p50／主観口フリーズ／RSS（Worker×キャッシュ）
- Before 比較: Phase13hf 長尺 `logs/sess_phase10_step2_subj_20260729_180454` および本 Phase の N=2 再計測
- OBS GPU 割り当ては運用メモとして可（必須 Pass 条件にしない）

**禁止（継続）:**
- 音声先行 enqueue／clear 隠蔽／正常 M0 短タイムアウト／固定 sleep
- coalesce／cond 付け替え再燃、旧 fast/slow 分離
- 図A・方式2・SSOT・jitter・idle／talkover／event・Phase14 IDLE_BG 破壊
- **計測完了前に default N を変えない**
- 6コア全振り（N>4 や全コア占有）禁止

**Pass 基準（骨子）:**
- [x] N=2/3/4 の同一台本表（REB・供給リード・wait/lock/m0・主観・RSS）を報告
- [x] default 据え置きのまま。N↑の Go/No-Go を親へ提案 → **No-Go（default=N=2 維持）**
- [x] 不変条件維持
- [x] 無理な Hotfix なし（計測＋主観閉鎖）

**子チャット報告（受理 2026-07-30）:**
- Phase15 レース revert 済。本番コード差分なし。計測ヘルパ `phase16_n_scale_probe.py` のみ。
- 長尺ストレス N=2/3/4: REB・wait/lock/m0・供給リードは横ばい。Worker 均等稼働（遊休ではない）。RSS は N にほぼ線形。
- 主観長尺 N=4 `sess_phase10_step2_subj_20260730_140150`（Before 同等尺）: 2文目途中〜末尾口フリーズ・**改善なし**（REB 15→15）。
- CLI 上限 1..4 既存のまま。default 未変更。

**親判定（2026-07-30）:**
- **Pass-with-defer**。N↑では中盤供給負けは閉じない（計測＋主観で閉鎖）。
- **default = N=2 維持**（N=3 推奨せず。N=4 は CLI 上限のみ）。N スケールは本線から外す。
- Keep All 可（計測ヘルパのみ）。
- tag: `phase16-pass`（commit `6fc6507`、ヘルパ `phase16_n_scale_probe.py`）付与済み（2026-07-30）。
- 次: **Phase 17 = mouth×claim 前線切り分け**（先に計測。実装は親承認後）。

---

## Phase 17: mouth×claim 前線切り分け（任意・分析優先）

**前提:** Phase16 Pass-with-defer。作業ベース = `phase14-idle-bg`（`cf8ec05`）相当。default **N=2**。N↑は本線外。

**目的:** 長尺中盤の供給負けが **「口（mouth/KNN）が本当に遅い」** のか **「口はあるのに claim 側で待たされている」** のかを計測で切り分け、次の実装対象（M3 or M1）を親に提案する。

**スコープ（本 Phase = 切り分けが主）:**
- 長尺中盤で時系列比較: mouth 最新 `t_ms` vs claim が欲しい `until_ms` vs `knn_ms` / `wait_mouth`（必要なら供給リード・player audio_ms）
- 判定ルール（親確定）:
  - **口が本当に遅い**（mouth 前線が claim/再生に追いつかない）→ M3（mouth_streamer / KNN）分析・改善を次候補
  - **口はあるのに待たされている**（mouth は先行／十分だが wait_mouth や claim が進まない）→ M1 の mouth↔claim 前進を次候補
- 主テスト: `--no-fast_inmemory`・N=2。Before 尺または実務近い長尺でよい
- 観測用の最小ログ／計測ヘルパは可。**本番の挙動変更・「改善」実装は親が切り分け受理後に別指示**（本 Phase で勝手に Fix しない）
- OBS→GPU は任意運用メモ（N↑の代替にしない・Pass 必須条件にしない）

**禁止（継続）:**
- 音声先行 enqueue／clear 隠蔽／短タイムアウト／固定 sleep／後続遅れをジッタで隠す
- coalesce／cond 付け替え、旧 fast/slow、無断 N↑、default N 変更、6コア全振り
- 図A・到着順 enqueue・方式2・SSOT・Phase14 IDLE_BG 破壊

**Pass 基準（骨子）:**
- [x] 中盤帯の mouth vs claim vs knn/wait の表または時系列要約
- [x] **A（M3）** を根拠付きで提案（B 未検出）
- [x] 不変条件維持（本番 Fix なし・計測ヘルパ＋観測ログのみ）
- [x] 次 Phase 実装スコープ案を親承認待ちと明記

**子チャット報告（受理 2026-07-30）:**
- ヘルパ `phase17_mouth_claim_frontier_probe.py`＋観測 `mouth_claim_frontier`（挙動非変更）。
- 中盤: knn 完了時 mouth は until に対し常時 ~80ms、claim_t1 に対し 80–160ms 遅れ。claim_ahead（B）=0。
- knn_ms は軽い → ボトルネックは streamer／窓／emit の口タイムライン供給。
- vs player は薄い先行＋時々負。high_wait はすべて claim_behind。

**親判定（2026-07-30）:**
- **Pass**。切り分け **A（M3）** 確定。本線は M3 mouth 前線先行。
- 「M1 緩める」はゲート定義の設計論のみ可。口形捨て／mouth_closed 埋めで wait を消すのは禁止。
- Keep All 可（ヘルパ＋観測ログ）。任意長尺で frontier 再確認は Phase18 で可。
- 次: **Phase 18 = M3 mouth 前線先行**。

---

## Phase 18: M3 mouth 前線先行（任意）

**前提:** Phase17 Pass。作業ベース = `phase16-pass`（`6fc6507`）＋ Phase17 Keep（ヘルパ／観測）。default **N=2**。N↑本線外。

**目的:** mouth_streamer／formant 窓／KNN emit が PCM／claim より **2–4 frame（約 80–160ms）先行**できるよう、計測→最小改善する。目標: knn 時点で `claim_gap≤0` 比率↑、中盤 `wait_mouth`↓、長尺中盤口フリーズ改善。

**スコープ:**
- 主対象: M3（mouth_streamer / formant 窓 / KNN emit）。M1 は観測・配線の最小のみ（図A・ゲートの口形要件を捨てない）
- 主テスト: `--no-fast_inmemory`・N=2。Before: `180454`／Phase16 N2。frontier ログで gap を再確認可
- 順序: **計測で遅れ箇所を特定 → 最小改善 → Before 比**

**禁止（継続＋本 Phase 強調）:**
- 初期ジッター延長で中盤 REB を「直した」ことにする（初期ジッタは開始前のみ）
- 直列補給負けモデルで設計する／口形一致を捨てる／mouth_closed 埋めで wait_mouth を消す
- 根拠なし N↑、coalesce／付け替え、音声先行 enqueue、clear 隠蔽、短タイムアウト
- 図A・到着順 enqueue・方式2・SSOT・Phase14 IDLE_BG 破壊
- `.cursorrules` §2 時間軸図・誤解禁止に反する説明で Pass しない

**Pass 基準（骨子）:**
- [ ] knn 時点の mouth vs until/claim_gap が Before 比で改善（claim_gap≤0 比率↑ または 構造的 +80ms の縮小）
- [ ] 中盤 wait_mouth↓、または主観（中盤口フリーズ）改善。未達なら計測で次手段を提案（Pass-with-defer 可）
- [ ] 不変条件維持（AUDIO_BEFORE 実 enqueue 0 目安、SSOT_WAIT、通常 clear 0、到着順、方式2、N=2）
- [ ] 付け替え・ジッタ隠蔽・口形捨てでないこと

**親メモ:** 子は `docs/PROGRESS.md` を編集しない。

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
| 2026-07-28 | `phase10-pass` tag（`7cd325e`）付与確認 |
| 2026-07-28 | Phase 11 Pass。talkover/event 方式2 通し。次=Phase12 待機モーション |
| 2026-07-28 | `phase11-pass` tag（`0a140da`）付与確認 |
| 2026-07-28 | Phase 12 Pass-with-defer。idle silent 図A再配線。次=Phase13 マルチ M0 |
| 2026-07-28 | `phase12-pass` tag（`ce903b9`）付与確認 |
| 2026-07-28 | Phase 13 追記: N=2 default・6コア全振り禁止・常駐プール・非回帰上位制約 |
| 2026-07-29 | Phase 13 Pass-with-defer。プール達成・長尺主観未達→Hotfix(IDLE_BG)。N↑保留 |
| 2026-07-29 | `phase13-pass` tag（M1 `7d314af`）。M0 `142aa82` 連携。Hotfix 新子へ |
| 2026-07-29 | Phase13 分析専用子を先に投下（Hotfix 実装は分析受理後） |
| 2026-07-29 | 分析受理: 主因=末尾 claim/coverage。Hotfix 順序=claim先→IDLE_BG |
| 2026-07-29 | Phase13 Hotfix Pass-with-defer。AUDIO_BEFORE大穴解消。残=中盤REB/末尾IDLE→Phase14 |
| 2026-07-29 | `phase13-hotfix` tag（`d1a881b`）付与確認 |
| 2026-07-29 | Phase14 分析受理: 中盤REB主因=lock/cond+wait_mouth（gap=0）。実装 Go |
| 2026-07-29 | Phase14 中盤 Fail。方針(A) 中盤リバート＋IDLE_BGのみ。次=新親 |
| 2026-07-30 | Phase14 (A) 完了。virtualcam IDLE_BG のみ Keep。中盤は hotfix 復帰 |
| 2026-07-30 | `phase14-idle-bg` tag（`cf8ec05`）。本親クローズ。次=新親（wait_mouth） |
| 2026-07-30 | 新親着手。Phase15=`wait_mouth`（mouth×claim）定義・子プロンプト発行。N↑は後段・親承認後 |
| 2026-07-30 | Phase15 Pass-with-defer。主観未達・wait_mouth 計測閉鎖。レース revert。N↑本線→Phase16 |
| 2026-07-30 | Phase16=N=2/3/4 計測 A/B 定義・子プロンプト発行。default 据え置き・上限4・6全振り禁止 |
| 2026-07-30 | Phase16 Pass-with-defer。N↑効果なし（計測＋主観）。default=N=2 確定。N本線外 |
| 2026-07-30 | `phase16-pass` tag（`6fc6507`、計測ヘルパのみ）付与確認 |
| 2026-07-30 | Phase17=mouth×claim 前線切り分け定義・子プロンプト発行（分析優先・Fixは親承認後） |
| 2026-07-30 | `.cursorrules` §2: 時間軸図・直列補給負け誤解禁止・中盤第一疑いを復元追記 |
| 2026-07-30 | Phase17 Pass。切り分け A=M3。Keep=ヘルパ＋観測。本線=M3 前線先行 |
| 2026-07-30 | Phase18=M3 mouth 前線先行定義・子プロンプト発行 |
