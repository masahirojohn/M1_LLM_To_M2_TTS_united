# Realtime Lip-Sync 再構築 — 進捗管理（SSOT）

> **親チャット（司令塔）がこのファイルを更新する。** 子チャットは直接編集しない。

## ベースライン

| 項目 | 値 |
| --- | --- |
| 起点 commit | `e4c1204`（Phase10 STEP1 stable restore before STEP2 retry） |
| 起点 tag | `phase10-local-vad-baseline` |
| マイルストーン tag | M1: … `phase29hf-pass`（`a656270`） / `phase30-pass`（`4d19fb9`）／M3: … `phase24-pass`（`abf8226`）／M0: `phase24-pass`（`af3dc72`） |
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
| 18 | M3 mouth 前線先行（任意） | `pass` | 2026-07-30（Pass-with-defer。主観未達→頭止めへ） |
| 19 | 到着順 enqueue 頭止め（任意） | `pass` | 2026-07-30（Pass-with-defer。HOL否定・観測のみ） |
| 20 | Phase18 回帰切り分け／ベース復帰 | `pass` | 2026-07-31（183212型復帰。因果emit Keep） |
| 21 | M3 VAD buffer/index（任意） | `pass` | 2026-07-31（長尺≥10s 口回復） |
| 22 | 長尺散発欠落／等速残差（任意） | `pass` | 2026-07-31（Pass-with-defer。hold-extend） |
| 23 | 中盤REB1／終端停止残差（任意） | `pass` | 2026-08-01（Hold。差分なし・縫い目 defer） |
| 24 | メモリ／キャッシュ削減（任意） | `pass` | 2026-08-01（Pass-with-defer。RSS頭打ち） |
| 25 | N 再計測 A/B（削減後）（任意） | `pass` | 2026-08-02（Pass-with-defer。default=N=2 再確認） |
| 26 | 残差指紋の横断切り分け（任意） | `pass` | 2026-08-02（実装なし。本命=終端＋CATCHUP） |
| 27 | 終端／ENQUEUE_BLOCKED（任意） | `pass` | 2026-08-03（Pass-with-defer。AB/BLOCK 0） |
| 28 | 多ターン後半 CATCHUP／VCam（任意） | `pass` | 2026-08-03（Pass-with-defer。主観横ばい） |
| 29 | sync_meta 更新タイミング（任意） | `pass` | 2026-08-03（Pass-with-defer。T1–7 OK・T8偽ゼロ残） |
| 29hf | sync_meta commit 偽ゼロ Hotfix | `pass` | 2026-08-03（Pass-with-defer。T8偽ゼロ閉鎖） |
| 30 | 中盤供給／REB／画像欠落（候補B）（任意） | `pass` | 2026-08-04（Pass-with-defer。post_gen閉鎖） |
| 31 | 発話中 mid-real 供給／残欠落（任意） | `pass` | 2026-08-04（Hold。差分なし・品質打ち切り） |
| 32 | battle/talkover/event 短回帰（任意） | `pass` | 2026-08-04（自動＋主観OK。差分なし） |
| 33 | 運用耐久（任意） | `pass` | 2026-08-05（Pass-with-defer。24t完走・凍結可） |
| freeze | 品質凍結マイルストーン | `pass` | 2026-08-05（現行 Keep 一式＝運用ベース） |

> **品質ラインはここまで。** 以降の本線は下節「[新ライン: レスポンス／VAD プロファイル](#新ライン-レスポンスvad-プロファイル2026-08-05)」（Phase `R1`, `R2`, …）。

状態値: `pending` / `in_progress` / `pass` / `blocked`

### 長尺品質ライン（2026-07-30〜・以降の品質 Phase に適用）

設計切り替え期はログ重視でよかったが、**現状の最大課題は長尺の2文目以降のリップ詰まり**のため、以降は次を採用する。

1. **長尺主観が主 Pass 条件**（Before 比で口の詰まり改善）。改善なし／悪化は **Fail または revert**（ログだけ良いは Pass にしない）
2. ログは改悪検知・不変条件・切り分け用（AUDIO_BEFORE 実 enqueue 0、enqueue 到着順、SSOT_WAIT、通常 clear 0、方式2、N=2 default 等は必須）
3. メトリクス改善のみ・主観横ばい／悪化 → Fail／revert。Pass-with-defer は「次手が明確なとき」のみ
4. 多ターン非回帰は維持条件（長尺のために多ターンを壊さない）

**子セルフゲート（親主観依頼の前に必須）:** 長尺ログ／mouth を時間帯分割（例 0–2s / 2s–end）。各帯の mouth_id nonzero 比率・ユニーク数・変化回数を表で報告。2文目以降帯で口形が実質変化なし／ほぼ全程 0 → **自動 Fail・Hold**（改善報告禁止）。PCM RMS が高いのに vad_active 後半全程 0 → Fail・Hold（改悪候補）。ゲート未達で親に主観依頼しない。

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
- tag: `phase17-pass`（commit `93f0820`）付与済み（2026-07-30）。
- 次: **Phase 18 = M3 mouth 前線先行**。

---

## Phase 18: M3 mouth 前線先行（任意）

**前提:** Phase17 Pass。作業ベース = tag `phase17-pass`（`93f0820`）。default **N=2**。N↑本線外。

**目的:** mouth_streamer／formant 窓／KNN emit が PCM／claim より **2–4 frame（約 80–160ms）先行**できるよう、計測→最小改善する。**本物 KNN 口形のまま**前倒し（口形品質を捨てない）。目標: knn 時点で `claim_gap≤0` 比率↑、中盤 `wait_mouth`↓、長尺中盤口フリーズ改善。

**スコープ:**
- 主対象: M3（mouth_streamer / formant 窓 / KNN emit）。M1 は観測・配線の最小のみ（図A・ゲートの口形要件を捨てない）
- 主テスト: `--no-fast_inmemory`・N=2。Before: `180454`／Phase16 N2。frontier ログで gap を再確認可
- 順序: **計測で遅れ箇所を特定 → 最小改善 → Before 比**
- **本 Phase では頭止め修正・M0 使い方見直し・N 再計測に手を出さない**（結果報告で分岐提案のみ）

**Phase18 の結果で分岐（次 Phase・親が発行。子は実装しない）:**
1. mouth 前線は改善したが **なお等速 enqueue が崩れる** → 到着順 enqueue 頭止め（先行 chunk 停滞）を計測→最小修正（図Aの順序は維持）
2. mouth は揃うのに **M0／png_wait 支配** → M0 側（既存 N=2 プールの使い方含む）
3. それでも長尺だけ不足 → N は **再計測オプション**（default は安易に上げない。Phase16 結論を尊重）

**当面やらない（全 Phase 共通の継続禁止）:**
- ジッタ延長で中盤隠し、付け替え系、口形品質捨て、6コア全振り

**禁止（継続＋本 Phase 強調）:**
- 初期ジッター延長で中盤 REB を「直した」ことにする（初期ジッタは開始前のみ）
- 直列補給負けモデルで設計する／口形一致を捨てる／mouth_closed 埋めで wait_mouth を消す
- 根拠なし N↑、coalesce／付け替え、音声先行 enqueue、clear 隠蔽、短タイムアウト
- 図A・到着順 enqueue・方式2・SSOT・Phase14 IDLE_BG 破壊
- `.cursorrules` §2 時間軸図・誤解禁止に反する説明で Pass しない

**Pass 基準（骨子）:**
- [x] knn 時点の mouth vs until/claim_gap が Before 比で改善 → **until−mouth_cov 常時0、claim_gap≤0 ~33%、wait_mouth↓**
- [x] 中盤 wait_mouth↓、または主観改善 → wait↓は達成。**主観中盤口フリーズは未達** → Pass-with-defer
- [x] 不変条件維持
- [x] 付け替え・ジッタ隠蔽・口形捨てでないこと
- [x] 分岐提案 → **1 頭止め**

**子チャット報告（受理 2026-07-30）:**
- M3: formant 未来半窓待ち（+120ms）が構造遅れの主因。**step 完了で因果 emit**＋窓をバッファへクランプ（本物 KNN。口形捨てなし）。
- Keep: `mouth_streamer_oc.py` のみ。M3 無関係 dirty（knn timeline）は除外。M1 は selfcheck ヘルパのみ。
- 主観 `sess_phase10_step2_subj_20260730_183212`: 音声正常・**2–3文目〜末尾口フリーズ**。M3 効果は維持（cov gap=0、wait≈21）。REB=3 がフリーズ開始と一致。until≥12s で **queue_wait p50≈509**、enqueue_order pending 多発 → 口は揃ったのに等速 enqueue 崩れ。

**親判定（2026-07-30）:**
- **Pass-with-defer**。M3 前線先行は計測 Pass。主観 Fail を正式受理（M3 Keep は維持・主因ではない）。
- Keep: M3 `mouth_streamer_oc.py` 因果 emit。tag: `phase18-pass`（M3 commit `0bfd8bb`、repo `M3_Live_API_1_united`）付与済み（2026-07-30）。
- 次本線: **Phase 19 = 到着順 enqueue 頭止め**（計測→最小修正。図Aの順序は維持）。
- 分岐 2（M0）／3（N再計測）は頭止め否定後まで後回し。
- ジッタ延長・付け替え・口形捨て・N↑で隠さない。

---

## Phase 19: 到着順 enqueue 頭止め（任意）

**前提:** Phase18 Pass-with-defer。作業ベース = M1 `phase17-pass`（`93f0820`）＋ M3 `phase18-pass`（`0bfd8bb`、因果 emit）。default **N=2**。mouth 前線は揃っている前提。

**目的:** 口が揃った後も中盤で等速 enqueue が崩れる主因として、**到着順 enqueue の頭止め（先行 chunk 停滞で後続完了分が player に届かない）** を計測し、図Aの順序を維持したまま最小修正する。

**スコープ:**
- 主対象: M1 の到着順 enqueue／dispatcher／先行 chunk 待ち（頭止めの観測と最小緩和）。順序保証（chunk_idx 到着順）は壊さない
- 主テスト: `--no-fast_inmemory`・N=2。Before: Phase18 主観 `183212`／メトリクス `180805`。M3 Keep は維持・巻き戻さない
- 順序: **計測で頭止めを確認 → 最小修正 → Before 比（queue_wait／pending／REB／主観）**
- M0／N↑は本 Phase の主対象外（頭止め否定後に親が分岐）

**禁止（継続）:**
- ジッタ延長で中盤隠し、付け替え、口形捨て、音声先行 enqueue、clear 隠蔽、短タイムアウト
- 到着順の破棄（後続を先行 chunk より先に player へ出すこと）
- 根拠なし N↑、6コア全振り、図A・方式2・SSOT・Phase14 IDLE_BG・Phase18 M3 成果の破壊
- `PROGRESS.md` 編集（親のみ）

**Pass 基準（骨子）:**
- [x] 頭止めが Before 比で計測改善、または主因を計測で閉じる → **閉じた（主因ではない）**
- [ ] 中盤 REB／主観口フリーズ改善 → **未達**（挙動修正は悪化で破棄）
- [x] enqueue 到着順維持・不変条件 OK
- [x] ジッタ／口形捨て／付け替えで隠していない

**子チャット報告（受理 2026-07-30）:**
- `order_wait≈0`／`push_wait≈queue_wait` → dispatcher HOL ではない。late qw の本体は直列 KNN（push_turn）待ち。
- 挙動修正2案（push-before-acquire／release-around-push）は悪化 → 破棄。Keep = 観測フィールドのみ。
- その後の長尺主観は改悪方向（1文目からフリーズ／全程閉じ口の例あり）。多ターン≈2文は完走 → VAD常時死にではない。

**親判定（2026-07-30）:**
- **Pass-with-defer**（分岐1＝HOL 仮説の閉鎖）。Keep = 観測のみ。挙動差分は残さない。
- **次は分岐2（M0）に進まない。** 先に長尺品質のベース復帰（Phase18 回帰切り分け）。
- VAD 新 Phase を本線にしない（改悪のマッチポンプ禁止）。VAD 全程0はベース復帰後に再発するときだけ独立対応。
- 以降の品質 Phase は「長尺品質ライン」＋子セルフゲートを適用。

---

## Phase 20: Phase18 回帰切り分け／ベース復帰

**前提:** Phase19 Pass-with-defer。フルロールバック（Phase0／マルチ前）はしない。作業は **ピンポイント**: Phase18 M3 因果 emit の bisect／revert／副作用最小修正。

**目的:** 長尺主観を **「1文目リップOK・2文目途中フリーズ」**（Phase18 Keep 直後〜`183212` 型）まで復帰させる。1文目からフリーズ／全程閉じ口の改悪を止める。

**スコープ:**
- 第一候補: M3 `phase18-pass`（`0bfd8bb`）因果 emit の bisect／revert／副作用最小修正
- M1 Phase19 観測は可。Phase19 挙動修正は再投入禁止
- 復帰確認: 長尺主観＋子セルフゲート。多ターン非回帰の短確認
- 復帰後に残る中盤フリーズは次本線（M0／等速）。N↑後回し
- VAD 全程0は本線化しない（復帰後再発時のみ）

**禁止:**
- ジッタ延長・付け替え・口形捨て・音声先行・clear 隠蔽・短タイムアウト・N↑・6全振り
- フルロールバック、VAD 新 Phase としての前進扱い
- ゲート未達での「改善報告」／親への主観依頼
- `PROGRESS.md` 編集

**Pass 基準（骨子）— 長尺品質ライン適用:**
- [x] 長尺主観が「1文目OK・2文目途中フリーズ」まで復帰 → **`134138` で到達**（全程閉じ口改悪ではない）
- [x] 子セルフゲート通過（6–10s 変化あり。10s 以降の死は残差として次へ）
- [x] 不変条件維持。多ターン非回帰 OK（`133246`）
- [x] 復帰手段明記 → **コード変更なし（因果 emit Keep）。Phase18 は改悪主因否定**

**子チャット報告（受理 2026-07-31）:**
- bisect: 因果 emit ON≈OFF（同一 PCM）→ Phase18 は全程閉じ口改悪の主因ではない。revert 益なし → Keep。
- 改悪 `223210`/`190747`: 10s まで無音→以降に音声集中＋高 RMS・vad≈0。
- 主観追記: 多ターン `133246` OK。長尺 `134138` = 183212 型復帰（1文目終わり頃から口フリーズ。last nonzero≈10.2s）。
- 残差仮説: **`max_buffer_s=10` トリム後も EnergyVAD が絶対 step でスライス → 以降 vad_active=0 固定**（多ターンは短いので出ない）。

**親判定（2026-07-31）:**
- **Pass**（復帰目標達成）。Phase18 因果 emit **Keep 維持**（revert しない）。
- 次本線: **Phase 21 = M3 VAD buffer/index 最小修正**（マッチポンプの「VAD新Phase化」ではない。復帰確認後の残差本丸）。
- M0／N↑は後回し。ジッタ延長で隠さない。

---

## Phase 21: M3 VAD buffer/index（任意）

**前提:** Phase20 Pass。作業ベース = M1 現行＋ M3 `phase18-pass`（因果 emit Keep）。default **N=2**。長尺品質ライン＋子セルフゲート適用。

**目的:** `max_buffer_s=10` トリムと EnergyVAD の絶対 step スライスが噛み合い、**音声があるのに vad_active=0 → 閉じ口**になる問題を最小修正する。長尺で 10s 以降／2文目以降の口フリーズを Before（`134138`／`183212`）比で改善する。

**スコープ:**
- 主対象: M3（VAD バッファ／index／トリム後の step 整合）。M1 は観測・配線の最小のみ
- 主テスト: 長尺 N=2・`--no-fast_inmemory`。多ターン短確認（非回帰）
- Before: `logs/sess_phase10_step2_subj_20260731_134138`（復帰後残差）／`183212`
- 順序: 仮説確認（トリム後 index）→ 最小修正 → セルフゲート → 長尺主観

**禁止:**
- ジッタ延長・付け替え・口形捨て・音声先行・clear 隠蔽・短タイムアウト・N↑・6全振り
- 図A・方式2・SSOT・Phase18 因果 emit の無断破壊
- ゲート未達での改善報告／親主観依頼
- `PROGRESS.md` 編集
- 「VAD新Phase」として Phase20 改悪をすり替える説明（本 Phase は復帰後残差の本丸）

**Pass 基準（骨子）— 長尺品質ライン:**
- [x] 長尺主観で 2文目以降／10s 以降の口フリーズが Before 比で改善 → **大幅改善**（`145030`）
- [x] セルフゲート: 高 RMS 帯で vad/mouth 回復（10–20s nz 2.4%→98%）
- [x] 多ターン非回帰（`145500`）。不変条件維持
- [x] トリム後も相対 index で VAD 追従する説明と一致

**子チャット報告（受理 2026-07-31）:**
- 仮説再現→ VAD スライスをトリム後バッファ相対 index に修正（emit 側と同型）。
- After `144410`／主観 `145030`: ≥10s 口死は閉鎖。5–6文目まで基本維持。散発欠落は残差。
- Phase18 因果 emit Keep。M1 本番差分なし。knn dirty 除外。

**親判定（2026-07-31）:**
- **Pass**（長尺品質ライン主条件達成）。Keep = M3 `mouth_streamer_oc.py` の VAD 相対 index のみ。
- tag: `phase21-pass`（M3 commit `02dd1aa`）付与済み（2026-07-31）。
- 残件 defer: 長尺散発 PNG 欠落／多ターン末の軽い切れ → **Phase22（任意）**。M0／N↑はなお後回し可。
- メモリは未評価のまま並行観測可（本線差し替えなし）。

---

## Phase 22: 長尺散発欠落／等速残差（任意）

**前提:** Phase21 Pass。M3 VAD 崖は閉鎖。作業ベース = M3 `phase21-pass`（`02dd1aa`）＋因果 emit Keep。M1 は現行（Phase19 観測のみ可）。default **N=2**。長尺品質ライン＋子セルフゲート適用。

**目的:** VAD 口死とは別の残差—長尺の**散発欠落**（ところどころ口／PNG 抜け）、等速 enqueue／REB 残、多ターン末の軽い切れ—を計測で切り分け、最小改善する。主観悪化は Fail・revert。

**スコープ:**
- Before: Phase21 主観 `logs/sess_phase10_step2_subj_20260731_145030`（大幅改善済み・散発欠落残）／多ターン `145500`
- 主テスト: 長尺 N=2・`--no-fast_inmemory`。多ターン短確認（非回帰）
- 切り分け優先: 散発欠落が mouth／M0 png／enqueue／VirtualCam のどこか → 最小修正は該当層のみ
- 並行観測（本線化しない）: RSS 時系列・可能なら qsize。単調悪化が主因指紋なら親へメモリ Phase 提案のみ
- M3 VAD／因果 emit は巻き戻さない。N↑は安易に本線化しない（Phase16 結論）

**禁止:**
- ジッタ延長・付け替え・口形捨て・音声先行 enqueue・clear 隠蔽・短タイムアウト・frame drop 間引き・先制 gc.collect
- 図A・到着順・方式2・SSOT・Phase14 IDLE_BG・Phase18/21 M3 成果の破壊
- ゲート未達での改善報告／親主観依頼
- `PROGRESS.md` 編集

**Pass 基準（骨子）— 長尺品質ライン:**
- [x] 長尺主観で散発欠落／途切れが Before（`145030`）比で改善 → **大幅改善**（局在化した残差のみ）
- [x] セルフゲート通過（VAD 崖再発なし）
- [x] 多ターン非回帰。不変条件維持
- [x] 切り分けと修正層一致 → 等速崩れ（claim 120ms vs until ~40ms → wait_mouth）に step1 最小修正

**子チャット報告（受理 2026-07-31）:**
- mouth/VAD 穴・中盤 png_verified 多発は否定。主因＝ claim 窓が playback until を跨ぐ wait_mouth スパイク→pending 枯渇 REB。
- step1: until クランプ mouth ゲート＋render 専用 hold-extend（last mouth_id、`mouth_closed` ではない）。
- 主観 `230158`: 最後まで音声＋リップ基本維持。2文目途中一瞬フリーズ＋終端間際停止。REB 5→2、wait_mouth≈0。
- 多ターン `230719` 非回帰。RSS 単調悪化の主因指紋なし。M3 未変更。session_loop 観測 dirty は Keep 外。

**親判定（2026-07-31）:**
- **Pass-with-defer**。散発「ところどころ」型から局在残差へ。Keep = M1 step1 のみ（hold-extend 承認）。
- tag: `phase22-pass`（M1 commit `a26efcf`、step1 のみ）付与済み（2026-07-31）。session_loop・M3 未含。
- M3 Phase18/21 Keep。Phase19 session_loop 観測は Keep に含めない。
- 残件 → **Phase23（任意）**: 中盤 REB1回＋終端停止。ジッタ延長で隠さない。

---

## Phase 23: 中盤 REB1／終端停止残差（任意）

**前提:** Phase22 Pass-with-defer。作業ベース = M1 `phase22-pass`（`a26efcf`、step1 hold-extend）＋ M3 `phase21-pass`（`02dd1aa`）。default **N=2**。長尺品質ライン＋子セルフゲート適用。

**目的:** Phase22 後に局在した残差—(1) 長尺中盤の一瞬フリーズ（REB 1回級）(2) 終端間際の音声・リップ停止—を計測で切り分け、最小改善する。主観悪化は Fail・revert。

**スコープ:**
- Before: Phase22 主観 `logs/sess_phase10_step2_subj_20260731_230158`（REB@~2.5s/~9.9s＋終端≈18.9s）／多ターン `230719`
- 主テスト: 長尺 N=2・`--no-fast_inmemory`。多ターン短確認（非回帰）
- 切り分け: 中盤 REB と終端停止は同一原因か別か。層は mouth／M0／enqueue／player／tail のいずれか1つに絞る
- Phase22 hold-extend は維持が原則。副作用（sticky）が主因なら縮小は可（口形捨て・closed 埋めは禁止）
- M3 VAD／因果 emit・到着順・N↑安易本線化は禁止。RSS 並行観測可（本線差し替えなし）

**禁止:**
- ジッタ延長・付け替え・口形捨て・音声先行 enqueue・clear 隠蔽・短タイムアウト・frame drop・先制 gc.collect
- 図A・方式2・SSOT・Phase18/21/22 成果の無断破壊
- ゲート未達での改善報告／親主観依頼
- `PROGRESS.md` 編集

**Pass 基準（骨子）— 長尺品質ライン:**
- [ ] 長尺主観で中盤一瞬フリーズおよび／または終端停止が Before（`230158`）比で改善 → **未達（改善候補なし）**
- [x] 切り分け実施 → 中盤と終端は別因。縫い目試行はトレードオフで不採用
- [x] 本番差分なし（`phase22-pass` 復帰）。悪化を残していない

**子チャット報告（受理 2026-08-01）:**
- 中盤 REB@~2.5s＝idle 浅バッファ、@~9.9s＝idle→real 縫い目（STOP_ON_REAL_AUDIO）。終端＝push_wait／tail（別件）。
- 縫い目修正2案は改悪または REB残＋AUDIO_BEFORE → 破棄。hold-extend 未縮小。M3 未変更。
- 主観依頼なし。多ターン未実施。

**親判定（2026-08-01）:**
- **Hold**（Keep All 不可・本番差分なし）。縫い目の「許容定義決め」は今はやらない（仕上げ論点・低優先）。
- 終端／縫い目／idle 浅バッファは既知メモとして defer。
- Phase21/22 Keep（M3 VAD・step1 hold-extend）維持。
- 次本線: **Phase 24 = メモリ／キャッシュ削減**（運用耐久・長尺化備え。残 REB1 の特効薬扱いしない）。その後 N=2/3/4 再計測（default 据え置き）。

---

## Phase 24: メモリ／キャッシュ削減（任意）

**前提:** Phase23 Hold。作業ベース = M1 `phase22-pass`（`a26efcf`）＋ M3 `phase21-pass`（`02dd1aa`）。default **N=2**。長尺品質ライン＋子セルフゲート適用。

**目的:** 数十分配信／長尺化に備えた **運用耐久**のため、メモリ／キャッシュを計測→削減する。Phase22 時点で RSS 単調悪化が主因指紋ではなかったため、**中盤 REB1 の特効薬扱いをしない**。品質非回帰が必須。

**スコープ:**
- 計測ファースト: 長尺（および可能なら耐久）で親／M0 Worker の RSS 時系列、可能なら主要 Queue qsize、既存 wait/lock/png/queue/push/order_wait
- 削減候補（計測で優先度決定）: M0 スプライト／フレームキャッシュ、ターン境界 flush、不要な保持・コピー、履歴肥大など
- 主テスト: 長尺 N=2・`--no-fast_inmemory`＋品質セルフゲート。可能なら多ターン連続または長時間耐久
- Before 品質: `230158`／Phase22 水準を非回帰基準とする
- その後（別 Phase）: N=2/3/4 再計測（default 据え置き）。本 Phase では N↑しない

**禁止:**
- frame drop／deque 間引き、先制 gc.collect を改善手段にする
- ジッタ延長・付け替え・口形捨て・音声先行 enqueue・clear 隠蔽・短タイムアウト
- 図A・方式2・SSOT・Phase18/21/22 Keep の破壊
- 根拠なし N↑、6コア全振り
- ゲート未達での「品質改善」主張、`PROGRESS.md` 編集

**Pass 基準（骨子）:**
- [x] 長尺 RSS 時系列（削減前後）→ 親 **頭打ち**（+8–9MB 級）。M0 は有界プラトー
- [x] 長尺品質非回帰（`165233` 正常。ゲート OK）
- [x] 耐久: 12t A/B で親頭打ち・品質ほぼ同帯（明確な Phase24 回帰なし）。28t 悪化は今回再現せず→ばらつき／尺要因メモ
- [x] 不変条件維持。残 REB／縫い目／終端は主 Pass にしていない

**子チャット報告（受理 2026-08-01）:**
- 最大項: MouthStreamer 全履歴 flush→disk 再読込。削減: M3 compact flush／M1 in-mem KNN＋flush 間引き＋finalize／M0 turn reset 実 flush。
- 主観最終: 長尺 `165233` OK。12t A vs B2 ほぼ同帯（回帰なし）。残るターン末フリーズ／CATCHUP は baseline にもあり次線。
- 旧B `214408` は stash 未復帰走行→归因無効・破棄。

**親判定（2026-08-01）:**
- **Pass-with-defer**。メモリ目的達成寄り。Keep All 可（**M1 + M3 + M0** の Phase24 削減のみ）。
- tag `phase24-pass` 付与済み: M1 `2510f1c` / M3 `abf8226` / M0 `af3dc72`（2026-08-01）。push 未実施。
- 孤児 `m0_persistent_worker` のホスト掃除は **運用として可**（品質 Fix 手段にしない。先制 gc.collect 禁止は維持）。
- 次: **Phase25 = 削減後 N=2/3/4 再計測**（default 据え置き・親承認後発行）。縫い目／終端はなお defer。

---

## Phase 25: N 再計測 A/B（削減後）（任意）

**前提:** Phase24 Pass-with-defer。作業ベース = M1/M3/M0 各 `phase24-pass`（`2510f1c` / `abf8226` / `af3dc72`）＋品質 Keep（hold-extend／VAD／因果 emit）。**default N=2 据え置き**（計測完了まで変更禁止）。長尺品質ライン＋子セルフゲート適用。

**目的:** メモリ削減後ベースで **N=2（基準）→ N=3 → N=4** を同一条件再計測し、RSS／品質／供給指標を更新したうえで、default 変更の Go/No-Go を親に提案する。Phase16（N↑だけでは中盤を解かない）を尊重しつつ、削減後の前提を更新する。**本 Phase は計測が主。無断で default を変えない。**

**スコープ:**
- 同一長尺台本・`--no-fast_inmemory` で N=2/3/4（各1本以上。可能なら主観も）
- 見る指標: 中盤〜後半の主観口、REB、供給リード、wait_mouth/lock/png/m0、queue/push/order_wait、**RSS（親＋M0 Worker、子プロセス込み）**
- Before 参照: Phase16（N↑効果なし）／Phase24 品質（`165233`・12t B2）
- CLI 上限 1..4 既存のまま。6コア全振り禁止
- 縫い目／終端／残 REB1 は観測メモ可。本 Phase の「N で直す」対象にしない
- 本番の挙動「改善」実装は原則不要（計測ヘルパのみ可）

**禁止:**
- **計測完了前に default N を変えること**
- ジッタ延長・付け替え・口形捨て・音声先行・clear 隠蔽・短タイムアウト・frame drop・先制 gc
- 図A・方式2・SSOT・Phase21/22/24 Keep の破壊
- N>4／6コア全振り、`PROGRESS.md` 編集
- ゲート未達での「N↑で改善」主張

**Pass 基準（骨子）:**
- [x] N=2/3/4 の同一条件比較表（長尺＋多ターン主観）
- [x] default 据え置き。親へ Go/No-Go → **No-Go（default=N=2 維持）**
- [x] 不変条件維持。Phase24 RSS 頭打ちは N↑で破綻しないが線形増
- [x] 無理な Hotfix なし（計測ヘルパのみ）

**子チャット報告（受理 2026-08-02）:**
- 長尺≈20s: REB・mid wait/lock/m0 は N 横ばい。RSS 線形（302→355→417）。ゲートは全通過だが N↑優位なし。
- 多ターン8t主観: N=2 やや悪化感（1回）、N=3 主観微差「少し良い」だが REB は非改善、N=4 は REB 明確悪化。
- Phase16 結論を削減後でも再確認: N↑で中盤/後半口問題は閉じない。品質改善は Phase18/21/22 Keep 側。

**親判定（2026-08-02）:**
- **Pass-with-defer**。計測＋主観で閉鎖。
- **default = N=2 維持（No-Go）**。N=3 非推奨。N=4 は CLI 上限のみ。
- **N=2 vs N=3 微差の再テストは不要**（親希望なし）。
- Keep All 可（計測ヘルパのみ。本番／default 未変更）。tag `phase25-pass` は任意・必須ではない。
- 基本線は一通り閉鎖（VAD／口前線／hold-extend／メモリ／N=2 再確認）。
- 次: **Phase26 = 残差指紋の横断切り分けのみ（実装なし）** → 最多指紋へ最小修正（見込み順: 終端 → 多ターン後半 ※同型なら合流 → 縫い目は最後）。縫い目許容定義 Phase は作らない。N↑・ジッタ延長・frame drop は再開しない。

---

## Phase 26: 残差指紋の横断切り分け（任意・分析のみ）

**前提:** Phase25 Pass-with-defer。作業ベース = 各 `phase24-pass`（品質 Keep: VAD／因果 emit／hold-extend）。default **N=2**。長尺品質ライン＋子セルフゲート適用。

**目的:** ターン後半の散発フリーズ／画像欠落（ばらつきあり）について、既知残差ラベルを横断して **指紋を切り分けるだけ**。実装・「改善」Fix はしない。

**切り分け対象（例）:**
- idle→real 縫い目（STOP_ON_REAL_AUDIO 等）
- 中盤 REB（数件級）
- 終端停止／末尾 ENQUEUE_BLOCKED・push_wait
- 多ターン後半フリーズ／SSOT_CATCHUP スパイク
- その他（口 nz 落ち・png 欠・queue/push 等）

**スコープ:**
- 既存ログ＋必要なら N=2・`--no-fast_inmemory` の長尺／多ターン再計測（観測のみ）
- 出力: 症状×指紋×層の表、最多指紋、次最小修正の優先順提案（親承認待ち）
- **実装なし。** 本番差分を残さない

**修正見込み順（親確定・実装は次 Phase）:**
1. 終端
2. 多ターン後半（終端と同型なら合流）
3. 縫い目は最後（許容定義専用 Phase は作らない）

**禁止:**
- 実装 Fix、N↑、ジッタ延長、frame drop、付け替え、口形捨て、音声先行 enqueue
- 縫い目「許容定義」Phase の新設・本線化
- `PROGRESS.md` 編集、ゲート未達での改善主張

**Pass 基準（骨子）:**
- [x] 残差ラベル横断の指紋表（長尺10＋多ターン10）
- [x] 最多指紋と層を根拠付きで特定
- [x] 次最小修正の優先順を親承認待ちで提案（実装しない）
- [x] 本番差分なし（観測ヘルパのみ）

**子チャット報告（受理 2026-08-02）:**
- セッションヒット最多は縫い目だが、現残差の重症度本命は **終端 BLOCKED/AUDIO_BEFORE** ＋ **多ターン後半 CATCHUP（virtualcam）**。
- BLOCKED ≡ AUDIO_BEFORE（同数）。hang_used=0。長尺 keep は終端 push スパイク減衰・CATCHUP=0。
- 多ターン後半: BLOCKED は終端と同型で合流可。CATCHUP は別層。

**親判定（2026-08-02）:**
- **Pass**。Keep＝観測ヘルパのみ可。
- 多ターン後半は **2チケット分離**: **Phase27=終端/BLOCKED → Phase28=CATCHUP/virtualcam**（1 Phase 混在禁止）。
- `push_wait`／`order_wait` 分解ログは **Phase27 の観測として最小復元可**（別 Phase 不要）。
- 縫い目許容定義 Phase は作らない。縫い目 Fix は最後。N↑・ジッタ・frame drop 再開しない。

---

## Phase 27: 終端／ENQUEUE_BLOCKED（任意）

**前提:** Phase26 Pass。作業ベース = 各 `phase24-pass`＋品質 Keep。default **N=2**。長尺品質ライン＋子セルフゲート適用。

**目的:** `FP_TAIL_PUSH_BLOCK`（ターン末 `ENQUEUE_BLOCKED` / `AUDIO_BEFORE_M0` ガード経路）を計測→最小修正し、長尺間欠および多ターン末の途切れ／ガード発火を Before 比で改善する。**CATCHUP／virtualcam は触らない**（Phase28）。

**スコープ:**
- 主対象: enqueue／tail 層。必要なら `push_wait`／`order_wait` 観測の最小復元
- Before: 多ターン `140224`（AB/BLOCK=5）／長尺 keep `165233`（BLOCK=0 のこともある＝間欠）
- 主テスト: N=2・`--no-fast_inmemory`。長尺＋多ターン短確認
- Phase28（CATCHUP）と混在させない

**禁止:**
- virtualcam CATCHUP「直し」を本 Phase に混ぜる
- ジッタ延長・付け替え・口形捨て・音声先行 enqueue・clear 隠蔽・短タイムアウト・frame drop・N↑
- 縫い目本線化、`PROGRESS.md` 編集、ゲート未達での改善主張
- Phase18/21/22/24 Keep の破壊

**Pass 基準（骨子）— 長尺品質ライン:**
- [x] 終端 BLOCKED・AUDIO_BEFORE 発火減 → **multi/long とも 0**（hold-extend 後 enqueue）
- [x] セルフゲート通過。CATCHUP を本 Phase 成果にしない
- [x] 多ターン非回帰・不変条件維持
- [x] 主観: 終端途切れ型は改善。前半ターン末の口詰まり残 → Phase28

**子チャット報告（受理 2026-08-03）:**
- push/order 観測復元＋ターン末 hold-extend（音声先行なし）。AB/BLOCK 5→0。
- 親主観 `135918`（N=2）: T1–4 末口詰まり残、T5 以降概ね正常。CATCHUP 1028→659（観測のみ）。
- 終端追加 Hotfix なし。N=4 `134914` は参考のみ。

**親判定（2026-08-03）:**
- **Pass-with-defer**。Keep All 可（観測復元＋終端 hold-extend）。
- tag: `phase27-pass`（M1 commit `dff3e62`）付与済み（2026-08-03）。
- 次: **Phase28 = CATCHUP／VirtualCam**（`FP_MULTI_LATE_CATCHUP`）。終端と混在させない。
- 縫い目本線化・N↑・ジッタ・frame drop 再開なし。

---

## Phase 28: 多ターン後半 CATCHUP／VirtualCam（任意）

**前提:** Phase27 Pass-with-defer。作業ベース = M1 `phase27-pass`（`dff3e62`）＋各 `phase24-pass` 品質 Keep。default **N=2**。長尺品質ライン＋子セルフゲート適用。

**目的:** `FP_MULTI_LATE_CATCHUP`（virtualcam SSOT_CATCHUP 後半スパイク）を計測→最小修正し、多ターン前半〜後半のターン末口詰まり／画像欠落を Before 比で改善する。**終端 BLOCKED 経路・enqueue 追加直しはしない**（Phase27 済み）。

**スコープ:**
- 主対象: virtualcam／Sync SSOT 表示側（CATCHUP の原因層）
- Before: 多ターン `logs/sess_phase11_subj_20260803_135918`（Phase27 After・CATCHUP≈659、T1–4 詰まり）／更に旧 `140224`（CATCHUP≈1028）も参照可
- 主テスト: 多ターン N=2・`--no-fast_inmemory`（本丸）。長尺は非回帰短確認
- audio_ms 照合を壊さない。sequential frame 消費を復活させない

**禁止:**
- 終端 hold-extend の巻き戻し／enqueue 音声先行で CATCHUP を隠す
- ジッタ延長・付け替え・口形捨て・clear 隠蔽・短タイムアウト・frame drop・N↑
- 縫い目本線化、`PROGRESS.md` 編集、ゲート未達での改善主張
- Phase18/21/22/24/27 Keep の破壊

**Pass 基準（骨子）— 長尺品質ライン:**
- [ ] 多ターン主観でターン末口詰まり／画像欠が Before（`135918`）比で改善 → **横ばい（悪化なし・明確改善なし）**
- [x] CATCHUP 主因（ターン境界 sticky）を計測で特定し VCam defer を入れた。件数改善は親主観ランでは未再現 → Pass-with-defer
- [x] AB/BLOCK=0 維持。長尺非回帰
- [x] Phase27 非破壊

**子チャット報告（受理 2026-08-03）:**
- 主因: 新 `frame_offset` 適用中に旧ターン PCM が残る → 誤照合で旧 FG sticky。
- VCam: sync_meta 採用 defer（`SSOT_META_DEFER`）＋欠番スキップ。enqueue／終端未変更。
- 子 After `142439`: CATCHUP 659→508・gap 縮小。親主観 `144546`: CATCHUP=986・体感横ばい（計測≠主観）。長尺 `145147` 非回帰 OK。

**親判定（2026-08-03）:**
- **Pass-with-defer**。Keep All 可（VCam META_DEFER）。主観クリアは未達。
- tag: `phase28-pass`（M1 commit `d71afcf`、virtualcam のみ）付与済み（2026-08-03）。
- 次本線: **Phase29 = 候補 A**（session_loop で sync_meta の frame_offset／base 更新を旧キュー排水後 or 新ターン初回 enqueue 時へ）。表示 SSOT の根。
- 候補 B（ターン末供給遅れ）は A の後、または A 否定後。1 Phase 混在禁止。
- 縫い目・N↑・ジッタ・frame drop 再開なし。

---

## Phase 29: sync_meta 更新タイミング（任意）

**前提:** Phase28 Pass-with-defer。作業ベース = M1 `phase28-pass`（`d71afcf`、VCam META_DEFER）＋ `phase27-pass`（終端 hold-extend）。default **N=2**。長尺品質ライン＋子セルフゲート適用。

**目的:** sync_meta の `frame_offset`／base 更新を **旧キュー排水後、または新ターン初回 enqueue 時** にずらし、旧 PCM 再生中の新 offset 誤適用を根から防ぐ。多ターン T 末口詰まりを Before（`135918`／`144546`）比で改善する。

**スコープ:**
- 主対象: session_loop の sync_meta 書き込みタイミング（表示 SSOT の根）
- Phase28 VCam defer は維持（両立可）。enqueue 音声先行・終端 hold-extend 巻き戻し禁止
- Before: `144546`（P28 主観横ばい）／`135918`（P27 After）
- 主テスト: 多ターン N=2・`--no-fast_inmemory`。長尺非回帰短確認
- 候補 B（供給遅れ）は実装しない（観測メモ可）

**禁止:**
- VCam／enqueue／終端の広い同時改修（本 Phase は sync_meta タイミング）
- ジッタ延長・付け替え・口形捨て・音声先行・clear 隠蔽・短タイムアウト・frame drop・N↑
- sequential frame 復活、`PROGRESS.md` 編集、ゲート未達での改善主張

**Pass 基準（骨子）— 長尺品質ライン:**
- [x] 多ターン主観で T 末詰まりが Before 比で改善 → **全体改善**（T4/T5一瞬・T8のみ残）
- [x] ターン境界 sticky／誤 META が計測で改善 → **T1–T7 CATCHUP=0**。T8 は偽ゼロ commit 残
- [x] AB/BLOCK=0・Phase27/28 Keep 維持・長尺非回帰（早期終了は API 側・改悪ではない）
- [x] フル Pass 保留 → Pass-with-defer＋狭い Hotfix

**子チャット報告（受理 2026-08-03）:**
- sync_meta を初回 enqueue 直前 commit（`played+pending`）。子 After CATCHUP 986→2。
- 親主観 `175711`: 全体改善。T1–7 境界 CATCHUP=0。T8 のみ `base_played_samples=0` 偽ゼロ commit → 巨大 audio_ms CATCHUP。
- 長尺 `180308`: 品質非回帰。90s 未達は generation_complete 早期（Before 同型）。

**親判定（2026-08-03）:**
- **Pass-with-defer**。Keep All 可（候補 A＝enqueue 時 commit）。
- tag: `phase29-pass`（M1 commit `6a35cd7`、session_loop のみ）付与済み（2026-08-03）。T8 偽ゼロ Hotfix は次。
- 次: **Phase29 Hotfix**（狭い）— commit 時 `played==0` かつ直前ターンで再生が進んでいたら再読込／playback_ref 優先／commit 拒否して再試行。
- 候補 B・ジッタ・N↑は Hotfix 後。縫い目本線化なし。

---

## Phase 29 Hotfix: sync_meta commit 偽ゼロ

**前提:** Phase29 Pass-with-defer。作業ベース = M1 `phase29-pass`（`6a35cd7`）＋ `phase28-pass`／`phase27-pass`。default **N=2**。

**目的:** ターン跨ぎ commit で `playback_state` が一時的に `played_samples=0`（空ファイル／レース）を読んで `base=0` のまま新 `frame_offset` を載せる不具合を潰す。T8 型の境界 CATCHUP／sticky を防ぐ。

**スコープ:**
- session_loop の sync_meta commit 読み取りのみ（最小）
- 候補 A の意図（排水後／初回 enqueue 時）は維持
- Before: `logs/sess_phase11_subj_20260803_175711`（T8 偽ゼロ）
- 主テスト: 多ターン N=2（8ターン以上推奨）。長尺非回帰短確認

**禁止:**
- 候補 B 本線化、ジッタ延長、N↑、frame drop、音声先行、Phase27/28 巻き戻し
- `PROGRESS.md` 編集、ゲート未達でのフル Pass 主張

**Pass 基準:**
- [x] 多ターンで `base_played_samples=0` の誤 commit が再発しない → T2–T8 すべて非ゼロ。T8 型消滅
- [x] 最終ターン型 CATCHUP／sticky が Before（`175711` T8）比で改善 → CATCHUP 全ターン 0
- [x] T1–7 境界 CATCHUP=0 を回帰させない
- [x] AB/BLOCK=0・Phase27–29 Keep 維持。主観で偽ゼロ型フリーズは解消（残は欠落寄り＝候補 B）

**子チャット報告（受理 2026-08-03）:**
- commit 読み取り堅牢化＋ last-good seed＋偽ゼロ時 COMMIT_DEFER。候補 A 維持。
- 親主観 `213604`: 全体まずまず。軽い詰まり＋画像欠落残（劇的改善ではない）。偽ゼロ型は閉鎖。
- 長尺 `214228`: 正常・一瞬欠落×2。リップフリーズなし。

**親判定（2026-08-03）:**
- **Pass-with-defer**。Keep All 可。sync_meta 偽ゼロ系は再オープンしない。
- tag: `phase29hf-pass`（M1 commit `a656270`、session_loop のみ）付与済み（2026-08-03）。
- 次本線: **Phase30 = 候補 B**（中盤 REB／供給／画像欠落）。Phase27–29hf Keep 維持。

---

## Phase 30: 中盤供給／REB／画像欠落（候補 B）（任意）

**前提:** Phase29hf Pass-with-defer。作業ベース = M1 `phase29hf-pass`（`a656270`）＋ `phase28-pass`／`phase27-pass`。default **N=2**。長尺品質ライン＋子セルフゲート適用。

**目的:** ターン中盤〜ターン末の **供給遅れ／REB／画像欠落**（候補 B）を計測で切り分け、最小改善する。主観の軽いリップ詰まり・欠落を Before（`213604`／`214228`）比で改善する。

**スコープ:**
- 主対象: M0／hold-extend 帯／pending・queue／供給リード等（表示 META 偽ゼロではない）
- Before: 多ターン `logs/sess_phase11_subj_20260803_213604`／長尺 `214228`
- 主テスト: 多ターン＋長尺 N=2・`--no-fast_inmemory`
- sync_meta 偽ゼロ／VCam META_DEFER／終端 BLOCKED は再設計しない（回帰させない）
- 縫い目は本線にしない（最後）。N↑・ジッタ延長禁止

**禁止:**
- sync_meta 偽ゼロ系の再オープン、ジッタ延長、N↑、frame drop、音声先行、口形捨て
- Phase27–29hf Keep の破壊、`PROGRESS.md` 編集
- ゲート未達での改善報告

**Pass 基準（骨子）— 長尺品質ライン:**
- [x] 多ターン／長尺主観で中盤詰まり・画像欠落が Before 比で改善 → **全体改善**（T2/T6 末の軽残差のみ）
- [x] REB／供給: post_gen REB 3→0、wait_mouth max 大幅減。idle 浅い REB は未着手（件数増あり・本線外）
- [x] CATCHUP 境界 sticky・AB/BLOCK=0 非回帰
- [x] 不変条件・セルフゲート維持

**子チャット報告（受理 2026-08-04）:**
- `generation_complete` で短短落なら即 live hold-extend（`clear_queue` 非触）。META／偽ゼロ未着手。
- 親主観 multi `140212`: 全体改善。T2/T6 末のみ軽詰まり。長尺有効 `142929`: 悪くない・欠落×2・口フリーズなし。無効ラン `140913` 除外。

**親判定（2026-08-04）:**
- **Pass-with-defer**。Keep All 可（gen_complete 早期 mouth hold-extend）。
- tag: `phase30-pass`（M1 commit `4d19fb9`、session_loop のみ）付与済み（2026-08-04）。
- Phase27–29hf 非破壊。偽ゼロ系再オープンなし。
- 残件 defer: idle 浅い REB／mid-real 供給ギャップ／散発画像欠落。縫い目本線化・N↑・ジッタ延長はしない。
- 次: **Phase31 = 発話中 mid-real 供給／残欠落のみ**（idle／縫い目は本線外）。効き薄なら品質実装打ち切り→運用耐久へ。

---

## Phase 31: 発話中 mid-real 供給／残欠落（任意）

**前提:** Phase30 Pass-with-defer。作業ベース = M1 `phase30-pass`（`4d19fb9`）＋ Phase27–29hf Keep。default **N=2**。長尺品質ライン＋子セルフゲート適用。

**目的:** 発話中（real PCM）の **mid-real 供給ギャップ／散発画像欠落／軽いターン末詰まり**（例: T2/T6）を計測→最小改善する。idle 浅いバッファ REB・縫い目は本線外。

**スコープ:**
- Before: 多ターン `logs/sess_phase11_subj_20260804_140212`／長尺 `142929`
- 主テスト: 多ターン＋長尺 N=2・`--no-fast_inmemory`
- 層候補: mid-real pending／M0 供給ペース／hold 帯（post_gen は Phase30 済み・再設計しない）
- **効き薄判定:** 主観・REB が Before 比で明確改善しない → Pass-with-defer または Hold で品質実装を打ち切り、次は運用耐久（親発行）。無理に縫い目／N↑へ逃げない

**禁止:**
- idle／縫い目本線化、N↑、ジッタ延長、frame drop、音声先行、偽ゼロ系再オープン
- Phase27–30 Keep 破壊、`PROGRESS.md` 編集
- ゲート未達での改善報告

**Pass 基準（骨子）— 長尺品質ライン:**
- [x] 効き薄を計測で示して打ち切り提案 → **真 mid-real REB=0・供給崩壊なし。修正なしで Hold**
- [x] mid-real 指標の説明（既閉鎖）
- [x] AB/BLOCK=0・CATCHUP 境界・post_gen 維持（未改変）
- [x] idle REB を「直した」ことにしていない

**子チャット報告（受理 2026-08-04）:**
- mid-real REB（PLAYING・pl≥200）=0。pending 浅底も実発話中 0。一層の最小修正なし。
- 残は主観の軽い末詰まり／散発欠落のみ（単一レバーなし）。N↑禁止済み。

**親判定（2026-08-04）:**
- **Hold**（コード差分なし・`phase30-pass` 据え置き）。品質実装打ち切り。Keep 新規なし。
- Phase27–30 Keep 維持。親主観再依頼不要。
- 次: **Phase32 = battle/talkover/interrupt/event 短回帰**（品質新規 Fix なし）。その後 **運用耐久**。品質凍結マイルストーンは耐久後。
- やらない: mid-real 追加 Phase、idle／縫い目本線化、N↑、ジッタ、音声先行、口形捨て。

---

## Phase 32: battle / talkover / event 短回帰（任意）

**前提:** Phase31 Hold。作業ベース = M1 `phase30-pass`（`4d19fb9`）＋ Phase27–29hf Keep（同一ツリー）。default **N=2**。方式2維持。

**目的:** Phase27–30 積み上げ後に、battle／talkover／interrupt／event 経路が方式2・clear 例外・割り込み整合で非回帰であることを短確認する。**品質の新規 Fix はしない。**

**スコープ:**
- 参照: `docs/battle_runtime_talkover_ops.md` / `docs/event_runtime_ops.md`（ops が古い場合は実装・ログ優先）
- 主テスト: talkover cut-in、必要なら interrupt file、event 動画経路。`--no-fast_inmemory`・N=2
- 壊れていれば **最小修正のみ**（回帰修復）。長尺品質チューニング・縫い目／idle REB 本線化はしない
- Before 参照: Phase11 talkover/event 実績（方式2通し）

**禁止:**
- 品質新規改善、N↑、ジッタ延長、frame drop、音声先行、偽ゼロ再オープン
- idle／縫い目本線化、`PROGRESS.md` 編集
- 通常ターンでの clear 乱用（割り込み例外以外）

**Pass 基準（骨子）:**
- [x] talkover: 割り込み時のみ clear＋新ターン正常。通常 clear 0
- [x] event: 経路破綻なし。図A／SSOT 維持
- [x] 方式2維持。Phase27–30 Keep 非破壊
- [x] 回帰修復以外の品質 Fix なし（差分なし）

**子チャット報告（受理 2026-08-04）:**
- 自動短回帰 talkover／event OK。主観 `224606`（admin 割込×2＋evt_001）: 音声・リップ基本正常、割込・event 正常。
- clear=talkover×2＋event×1 のみ。AB/BLOCK/SSOT_WAIT/CATCHUP=0。本番差分なし。

**親判定（2026-08-04）:**
- **Pass**。Keep All 可（差分なし）。tag 任意・必須ではない。
- 次: **Phase33 = 運用耐久**（数十分／多ターン連続。RSS・孤児 Worker・品質非回帰監視）。
- 耐久後に問題なければ品質凍結マイルストーン。N↑・ジッタ・frame drop・偽ゼロ再オープン禁止。

---

## Phase 33: 運用耐久（任意）

**前提:** Phase32 Pass。作業ベース = M1 `phase30-pass`（`4d19fb9`）＋ Phase27–29hf Keep。default **N=2**。品質新規 Fix は原則しない。

**目的:** 数十分または多ターン連続で運用耐久を確認する。RSS 頭打ち（Phase24）、孤児 M0 Worker なし、品質が長尺品質ラインから大きく崩れないことを示す。

**スコープ:**
- 主テスト: 多ターン連続（目安 20–40t 以上）および／または長時間セッション。`--no-fast_inmemory`・N=2
- 観測: 親＋M0 RSS 時系列、孤児 Worker、AB/BLOCK、通常 clear、方式2、粗い主観（致命フリーズの有無）
- 壊れていれば最小修復のみ。品質チューニング・縫い目／idle／N↑本線化はしない
- Phase24 RSS サンプラ等の再利用可

**禁止:**
- N↑、ジッタ延長、frame drop、音声先行、口形捨て、偽ゼロ再オープン
- 品質新規チューニング、`PROGRESS.md` 編集
- 通常 clear 乱用

**Pass 基準（骨子）:**
- [x] 耐久ラン完走 → 親主観 24t 完走（`150656`）。子先行の API 1008 は環境依存
- [x] RSS 頭打ち／プラトー（親・M0）
- [x] 孤児 M0 なし（親終了連動で十分）
- [x] 致命的全体破綻なし。AB/BLOCK=0・通常 clear 0・方式2。局所フリーズは凍結後監視
- [x] 品質新規 Fix なし

**子チャット報告（受理 2026-08-05）:**
- 単一接続で API 1008 落ちあり → reconnect_per_turn で 24t 成功も記録。
- 親主観 `150656`: 24t 正常完走・reconnect なしでも可。T3–4／T10–12 に局所フリーズ（長尺後半残差＝Phase31 打ち切りと同型）。
- RSS 頭打ち・孤児なし・CATCHUP=0・偽ゼロ非再発。

**親判定（2026-08-05）:**
- **Pass-with-defer**。Keep All 可（コード差分なし）。
- **品質凍結マイルストーン**を宣言（下記）。局所フリーズは凍結後の監視／将来品質 Phase。縫い目・N↑・ジッタに戻さない。
- 運用: 多ターンで reconnect_per_turn は **必須ではない**（本ラン実証）。API 1008 時の退避として可。コード default 変更はしない。
- RSS サンプラ残骸のホスト掃除は運用として可。

---

## 品質凍結マイルストーン（2026-08-05）

**運用ベース（現行 Keep 一式）:**
| 層 | tag / commit | 内容 |
| --- | --- | --- |
| M1 | `phase30-pass`（`4d19fb9`）＋ `phase29hf-pass`（`a656270`）＋ `phase29-pass`／`phase28-pass`／`phase27-pass` | hold-extend／sync_meta／VCam META_DEFER／偽ゼロ対策 等 |
| M3 | `phase24-pass`（`abf8226`）＋ `phase21-pass`／`phase18-pass` | compact flush／VAD 相対 index／因果 emit |
| M0 | `phase24-pass`（`af3dc72`） | turn reset 実 flush 等 |
| 運用 | default **N=2**・`--no-fast_inmemory` 優先・方式2 | reconnect_per_turn は API 1008 退避（必須ではない） |

**凍結後もやらない（再開禁止）:** N↑本線化、ジッタ延長で中盤隠し、frame drop、音声先行 enqueue、口形捨て、偽ゼロ系の不用意な再設計、idle／縫い目の品質本線化。

**凍結後の監視（将来 Phase 候補・品質本線には戻さない）:** 長尺後半の局所口フリーズ（m0_ms 上昇相関）、境界 CATCHUP、累積 rebuffer、API 1008。

**役割転換（2026-08-05〜）:** リップ品質の追加チューニングは原則しない。以降の本線は **プロダクト／運用機能**（下節「新ライン」）。品質 Phase 番号（0–33 / freeze）はクローズ済み履歴として維持する。

---

## 新ライン: レスポンス／VAD プロファイル（2026-08-05〜）

> **進捗ファイル方針:** 別ファイルは作らない。本節＋下表で R ラインを管理する（親のみ編集。子は直接編集しない）。
> リップ再構築（Phase0–33／freeze）とは **番号空間を分離**（`R1`, `R2`, …）。品質チューニング・縫い目・N↑・ジッタ隠しには戻らない。

### 前提（実装 SSOT）

| 項目 | 内容 |
| --- | --- |
| 方式 | 方式2（`automatic_activity_detection=False`＋ローカル RMS VAD＋`activity_start`/`activity_end`） |
| VAD 実装 | **RMS `mic_vad_*` CLI**（Silero/WebRTC 一般論に引きずられない） |
| 体感「話し終わり〜反応」 | クライアント silence 待ち＋API `end→first_pcm`。合計の過半は API（R1 計測） |
| 関連 CLI（例） | `--mic_vad_end_enabled` / `--mic_vad_silence_ms` / `--mic_vad_min_voice_ms`（240）/ `--mic_vad_min_listen_ms`（800）／RMS 閾値類 |
| 運用ベース継承 | default **N=2**・`--no-fast_inmemory` 優先・図A・品質凍結 Keep 一式 |
| レスポンス運用ベース | 通常 **350**／攻め腕 **250**（管理画面 `vad_profile_live.txt`）。commit `eb26a9c` / tag `phase-r2-pass` |
| レスポンス本線 | **クローズ**（2026-08-06）。API `end→first` 短縮はバックログ（当面不要） |

### R フェーズ一覧（レスポンス／VAD・クローズ）

| Phase | 名称 | 状態 | Pass 日 |
| --- | --- | --- | --- |
| R1 | レスポンス遅延計測＋silence A/B＋短回帰 | `pass` | 2026-08-06 |
| R1b | silence 攻め腕短検証（250 vs 350） | `pass` | 2026-08-06（分岐 A） |
| R2 | runtime ファイル切替（350／250） | `pass` | 2026-08-06（Pass-with-defer: UI→R2b） |
| R2b | 管理画面→`vad_profile_live.txt` 書込 | `pass` | 2026-08-06 |

### プロファイル（確定・運用中）

| プロファイル | `mic_vad_silence_ms` | 用途 | 状態 |
| --- | ---: | --- | --- |
| 通常 | **350** | 日常運用 | 確定 |
| 攻め腕（バトル） | **250** | 反応優先 | 確定 |

### プロダクト本線（続き）

| Phase | 名称 | 状態 | Pass 日 |
| --- | --- | --- | --- |
| I1 | 無言→AI アイドル発話（会話後） | `pass` | 2026-08-09（Pass-with-defer: 初手→I1b） |
| I1b | 初手無言（セッション開始時の口火） | `pass` | 2026-08-09（候補A） |
| B1 | BGV 時計特定（調査のみ） | `pass` | 2026-08-09 |
| B2 | BGV↔audio_ms ズレ定量＋方式選択材料 | `pass` | 2026-08-09 |
| B3 | PLAYING 中 `played_audio_ms→BG frame` 実装 | `pass` | 2026-08-09（Pass-with-followup→B3hf） |
| B3hf | BG 固着／playback_state 読取競合 Hotfix | `pass` | 2026-08-10（Pass-with-followup→B3hf2） |
| B3hf2 | Turn 先頭 bg_pos 固着／境界スナップショット | `pass` | 2026-08-10 |
| B4 | T2 残ズレ：pose／幾何切り分け（分析のみ） | `pass` | 2026-08-10（主因 **B**） |
| B5 | pose スライス時計を BGV 継続 index へ結ぶ | `pass` | 2026-08-11（主観 Fail・方式 Keep→B5hf） |
| B5hf | pose_base snapshot／missing→0 Hotfix | `pass` | 2026-08-11（Pass-with-defer→B5hf2） |
| B5hf2 | ターン境界の誤 pose_base freeze | `pass` | 2026-08-11（Bライン **Pass-with-defer**・運用 Keep） |
| B6 | 合成瞬間 display_bg↔pose Δ（調査のみ） | `pass` | 2026-08-23（仮説2厚い・方式C採用） |
| B7 | 境で進んだ BG を pose/M0 と同じ枚へ（方式C） | `pass` | 2026-08-24（Pass-with-defer。② クローズ） |
| （defer） | 起動 BUFFERING idle Δ／EN turn_local 窓／貼り位置／口−音 | — | 第二手法なし。Colab・貼り・口−音は開かない |
| O1 | OBS WebSocket＋管理画面＋背景静止画/BGM切替 | `pass` | 2026-08-12（flat local Hotfix Keep） |
| O2 | 当てフリ（OBS事前配置ソースの見せ消し） | `pass` | 2026-08-13 |
| O3 | エージェントスミス＋音声フィルタ | `pass` | 2026-08-14 |
| （後続可） | スミス増殖加速（順次表示・delay短縮） | — | 管理画面側のみ。session_loop sleep 禁止 |
| X1 | blink 頻度（emo_id 分岐・M3生成） | `pass` | 2026-08-15（Pass-with-defer→X1b Live） |
| X1b | Live expression に blink 挿入配線 | `pass` | 2026-08-15 |
| F1 | M0 FG 黒縁除去（unpremultiply 試行） | `pass` | 2026-08-15（Fail→Revert。blit Keep） |
| F1s | 現行スプライトで合成主観（黒縁まだ問題か） | `pass` | 2026-08-16（**A**＝目立たない→F1クローズ） |
| （任意後日） | きれいスプライト資産更新 | — | 必須ではない。差替時は位置・25fps・4ch 短確認。**JP 本線に混ぜない** |
| （本線移管） | 英語版 Realtime | → EN-RT | **RT+LIVE1 クローズ済（2026-08-19）**。番号は `EN-RT0/1/2`＋`EN-LIVE1`。JP リポ現状維持 |
| （本線） | 英語検証／配信 | → EN-DUR / Z | **Z2b Pass（2026-08-22）**。遠隔受信=Banana。EN 短確認は任意。**main マージ済（2026-08-25）** |
| V1 | Live 声 prebuilt 固定（Aoede） | `pass` | 2026-08-26（Pass-with-note。Kore 名は Keep 不可） |
| P1 | EN 本番システムプロンプト差し替え＋テスト | `pass` | 2026-08-27（Pass-with-note。口 barge-in＝既存 talkover） |
| E1 | イベント動画 catalog 最大10＋管理画面プルダウン | `pass` | 2026-08-27（Pass-with-defer。完了ゲート→E1b） |

### Bライン申し送り（2026-08-25・main マージ済）

- 運用 Keep: 方式 A／pose=BGV 絶対 index／B3hf2 sync／方式C（境スナップ）／IDLE_BG_ADVANCE／`[B6_DELTA]`（tag `phase-b7-pass` = `813c641`。main FF 済）
- **今の本線 = なし。** E1 Pass-with-defer。希望順④は指名待ち（子は出さない）
- E1 Keep: catalog プルダウン（最大10・既存2件・決め打ちボタンなし）。イベント中 sequential_from_0（B3/B7 不使用）。open 直後 frame0 seek。復帰は古い pose lock を捨てる。SSOT=M1 `in/event_catalog.json`（M3.5 `in/` スキャンしない）
- P1 Keep: `--prompt_dir` 2系統（`prompts_en`＝20あり30空 / `prompts_en_battle`＝20空30=Studio）。interrupt/leadership は今の prompt_dir 横。JP fallback。口 barge-in＝既存 talkover `clear_queue`（mute は切らない）。二重 InputStream は監視。`main` FF 済（`f117d51` / `phase-p1-pass`）
- V1 Keep: 本番声 = `speech_config` prebuilt **Aoede**（両分岐）。camelCase wire（`t_live_speech_config`）。`--voice_name` CLI なし。JP/EN 同一接続。Kore 名は Keep しない。`main` FF 済（`b4f7d5c` / `phase-v1-pass`）
- ②（BGV顔Y vs M0顔Y）は B7 Pass で閉じた。B8／第二手法は出さない
- 定常 PLAYING・高速上下: Δ 中央0 最大1。EN PLAYING 最大275は消えた
- defer: 起動 BUFFERING の idle境最大（115/116）／EN ターン境最大131（turn_local→absolute 窓）。主観の顔Yは JP/EN とも解消
- 開かない: 貼り位置／口−音／pose 先送り／IDLE 廃止／`audio_ms` オフセット／pose.json／Colab／多BGV
- JP+EN。代表 BGV 1本。常時「表示枚目=pose枚目」
- **VirtualCam 内 BGV（猫の体・合成用動画）≠ OBS「背景」**。OBS 制御は VirtualCam BGV を切替対象にしない

### 次本線希望順（②以降は出さない）

| # | 内容 | 状態 |
| --- | --- | --- |
| 0 | 本マージ（B7 ベース） | **済**（2026-08-25。`phase-b7-pass` / `813c641` を `main` FF。本 PROGRESS 追記も FF） |
| 1 | Live 声の女性一本化（LiveConnectConfig の speech_config／女性 prebuilt。prompt だけではない） | **V1 Pass-with-note**（2026-08-26）。Keep=**Aoede**＋camelCase wire。`main` FF 済（`b4f7d5c` / `phase-v1-pass`） |
| 2 | EN 本番システムプロンプト差し替え＋テスト（prompt_dir。20/30 役割維持。割り込み／主導権定型の英語化） | **P1 Pass-with-note**（2026-08-27）。`main` FF 済（`f117d51` / `phase-p1-pass`） |
| 3 | イベント動画 catalog 最大10＋管理画面プルダウン | **E1 Pass-with-defer**（2026-08-27）。完了ゲートは E1b 予約。ブランチ `feature/event-catalog-admin` |
| 4 | 第三者向け「主要コマンド＋事前準備」docs（PROGRESS・ops_zoom・合格コマンドから抜く。チャット全文の要約にしない。ops_zoom は再発行しない） | 指名待ち |

出さない: 二重 Live 自己対戦。B8／貼り／口−音／Colab／N↑／ジッタ延長／session_loop ミックス。

### OBS 音声ルーティング（親定義・子へ固定）

**AI 音声**は現行 player→PC 再生デバイスのまま OBS が音声ソースとして拾う。**BGM**は OBS メディアソースのみを WebSocket で切替し、AI PCM／VirtualCam／M0 に混ぜない。

### Zoom 運用（Z1 出＋Z2b 入・2026-08-22）

この機は **AI/OBS 専用**。人間は Zoom/SNS にこの機から参加しない。USB mic はローカル検証のみ（`mic=1` だけなら Banana 不要）。本番会話入力は遠隔の相手声。当面 Zoom。本線は YouTube / TikTok（音声ゲストなら同じ「受信再生 → 仮想 mic → `--mic_input_device`」）。コメント／text interrupt のみなら音声受信バスは本番必須ではない。

**毎回:** 先に **Voicemeeter Banana** を起動して開いたまま（`voicemeeterpro.exe`。Standard は使わない）。Zoom の Mic/Speaker を確認。主導権 mute+interrupt は Zoom では使わない。

**Banana:** 中央 VIRTUAL INPUTS 左（`Voicemeeter VAIO` / Voicemeeter Input）だけ **B1 点灯**。B2/A2/A3 オフ。CABLE を Hardware In に足さない。A1=ヘッドホンは遠隔モニター任意。USB mic を B1 に足してよいが Gate には使わない。

**Zoom:** Microphone = **CABLE Output**。Speaker = **Voicemeeter Input (VAIO)**。禁止: CABLE Input / システムと同じ / VAIO3 / AUX / In 1–5。カメラ = **Unity Video Capture**（`[virtualcam_persistent][OK]` のあと）。

**M1:** `--audio_device` / `--mic_input_device` は **再クエリ**（決め打ち禁止）。この機の導入後スナップショット: CABLE Input **23** / Voicemeeter Out B1 **9** / CABLE Output **8**（mic 禁止）/ VAIO **19**。旧 CABLE Input=6 は Banana 後に `Voicemeeter Out A4` へずれた。CABLE Output を mic にしない。

**やらない:** session_loop ミックス、ジッタ延長、B/O 再開、`prompts_en_dur`、部屋スピーカー拾い。

遅延は継続観察（Gate にしない）。

第三者向け抜き出し（SSOT は本節）: `docs/ops_zoom_third_party.md`

### 当面の非本線（今は実装させない）

- API `end→first` 短縮
- pose 先送り／IDLE_BG_ADVANCE 廃止／`audio_ms` オフセット／pose.json 先修正／多BGV／Colab pose
- 貼り位置／口−音／起動 BUFFERING idle Δ／EN turn_local 窓（B7 defer。第二手法禁止）
- Phase12 系の idle silent PCM／縫い目の品質本線化
- 英語版 CTC（EN-RT3。当面やらない）
- EN-DB1 / 欠 view
- Zoom 運営マニュアル再発行（`docs/ops_zoom_third_party.md` 済み）
- スミス増殖加速（任意）
- 当てフリの Python scale／座標拡大
- Slack トリガ本番化
- B全面再オープン／リップ品質本線化
- 再生中 mic の新 barge-in 経路／二重 InputStream 本線化／主導権を Zoom で使う
- イベント完了まで AI を待つゲート（E1b。親設計ロック前は出さない。session_loop sleep／ジッタ延長／M0停止ゲート禁止）

### 全プロダクト Phase 共通禁止（子）

- `docs/PROGRESS.md` 無断編集、凍結 Keep 破壊、N↑、ジッタ延長で症状隠し、frame drop、音声先行 enqueue、口形捨て
- リップ品質・縫い目・Silero 再設計・API end→first 本線化
- VAD 350/250 プロファイルと talkover／割り込みの破壊

---

## Phase R1: レスポンス遅延計測＋silence A/B＋短回帰

**目的:** 「話し終わり〜サーバ反応」の間の内訳をログで確定し、`mic_vad_silence_ms` 短縮が支配的かを Before/After で示す。低遅延／通常プロファイルの候補値を短回帰付きで提案する。**管理画面切替は R2（本 Phase では実装しない）。**

**スコープ:**
1. **計測（推測禁止）:** 話し終わり → `activity_end` 送信 → 先着 AI PCM／`first_audio`（既存ログ名に合わせる）までの時間を分解。既存ログがあれば活用、不足なら最小の計測ログ追加のみ
2. **A/B:** 通常（例 silence 600）vs 低遅延（例 300–400）。`silence` が支配的かを数値で示す
3. **短回帰:** talkover／短発話で早期切断・言いよどみ誤終了が悪化しないこと。方式2・図A・凍結 Keep 非破壊

**スコープ外（R1 禁止）:**
- 管理画面／HTTP／control ファイルによるランタイム切替（→ R2）
- アイドル発話、BGV ズレ、OBS 制御
- リップ品質チューニング、N↑、ジッタ延長、frame drop、音声先行 enqueue

**Pass 基準:**
- [x] 間の内訳表（区間定義＋ms）。どの区間が支配的か一文で結論
- [x] Before（silence≈600）/ After（低遅延候補）の同一条件比較。`mic_vad_silence_ms` 支配の可否を数値で示す
- [x] 低遅延・通常プロファイル案（silence と必要なら関連 `mic_vad_*` の推奨値）→ **親判定で 350 固定に更新（下記）**
- [x] talkover／短発話の短回帰 OK（早期切断・誤終了の悪化なし、または許容条件を親承認用に明記）
- [x] 方式2・図A・凍結 Keep・N=2・`--no-fast_inmemory` 非破壊（破壊差分なし／親承認可能な最小差分のみ）
- [x] 親向けサマリーのみ（diff/ログ全文貼付禁止）。次 Phase（R2）への提案 3 行以内

**主な対象（子が特定）:** `run_mic_input_obs_realtime_session_loop.py` 周辺の `mic_vad_*` / `activity_end` / first_audio 計測。計測ヘルパは `tools/` 可。

**設計メモ:**
- 実装 SSOT は RMS `mic_vad_*`。外部 VAD ライブラリ前提の再設計はしない
- Keep: `[resp_timing]` 計測ログ＋ `tools/_phase_r1_parse_resp_timing.py`（挙動・default CLI 不変）

**子報告要約（2026-08-06）:**
- 自動 A/B（n=3）: silence待ち 617→374（Δ−243）。合計過半は API `end→first_pcm`。短発話・talkover sil=350 悪化なし。
- 主観 A/B（n=6）: silence待ち 616→377（Δ−239）。合計 1675→1546（Δ−129、API 揺らぎで埋もれ）。体感は「少し早いかも」程度。品質非破壊。
- Keep: `session_loop` の `[resp_timing]`＋ parse／ハーネス任意。

**親判定（2026-08-06）:**
- **Pass。** 計測目的達成。プロファイル案を更新:
  - **不採用:** 通常=600／低遅延=350 の二値切替（体感差小・無駄）
  - **運用ベース候補:** **silence=350 固定**（min_voice=240, min_listen=800, rms 現行）
  - **R2（600/350 切替）保留**
  - **次=R1b:** 攻め腕 silence 200–250 vs 350（短回帰＋主観）。OKかつ体感差明確→切替を 350/200–250 に再定義して R2。NG／体感差なし→切替なし・350 固定でレスポンス本線クローズ
  - API `end→first` 短縮は今はやらない（別トラック）

---

## Phase R1b: silence 攻め腕短検証（200–250 vs 350）

**目的:** バトル用の攻め腕として silence **200–250** が使えるかだけを短く検証する。運用ベースは **350**。管理画面実装はしない。

**スコープ:**
1. A/B: silence **350（基準）** vs **200 または 250（攻め腕・どちらか一方を先に）**。他 `mic_vad_*` 固定
2. 計測: 既存 `[resp_timing]` で silence待ち／end→first／合計（新規ログ原則不要）
3. 短回帰: 短発話＋talkover（言いよどみ誤終了・早期切断）
4. 主観: ユーザー実機 1 本（体感差が明確か／悪化なしか）

**スコープ外:**
- R2 管理画面実装、API `end→first` 短縮、Silero 等の VAD 再設計
- リップ品質・N↑・ジッタ隠し・600 への復帰提案

**Pass 分岐（親が判定）:**
- **A:** 短回帰 OK かつ主観で体感差が明確 → プロファイルを「通常=350／攻め腕=採用値」に再定義 → R2 Go
- **B:** 早期切断・誤終了 NG、または体感差なし → **350 固定のみ**。R2 なしでレスポンス本線クローズ可
- default CLI 変更は親承認後のみ（R1b では起動フラグ比較に留める）

**Pass 基準（報告）:**
- [x] 内訳表（350 vs 攻め腕）
- [x] 短発話・talkover 短回帰
- [x] 主観 1 本の結論（明確差／差なし／悪化）
- [x] 推奨: 採用値 or 捨てて 350 のみ
- [x] 非破壊確認

**子報告要約（2026-08-06）:**
- 自動: silence待ち 376→254（Δ−122）。短発話・talkover@250 OK。コード差分なし。200 未実施。
- 主観@250: 「全体的にかなり速くなった。満足レベル」。早期切断訴えなし。silence待ち avg 258ms。
- 系列: silence待ち 600=616 / 350=377 / 250=258。合計は API 揺らぎありつつ 250 で主観満足。

**親判定（2026-08-06）:**
- **Pass・分岐 A。** 通常=350／攻め腕=250。200 不要。次=**R2**（管理画面切替）。default CLI 変更は R2 設計時に親承認。

---

## Phase R2: 管理画面から VAD プロファイル切替（350／250）

**目的:** コマンド再起動なしで `mic_vad_silence_ms` を **通常=350／攻め腕=250** に切替できること。永続化・デフォルト・安全弁を含む。

**スコープ:**
1. 切替手段を提案→親承認後に実装（ファイル／HTTP／既存 control 等。別プロセス管理画面想定）
2. 出せる値は当面 **350 と 250 のみ**（極端値禁止の安全弁）
3. 永続化・起動時デフォルト（推奨 default=350。CLI 上書きとの優先順位を明記）
4. 切替確認: `[resp_timing]` の silence待ち≈設定値
5. 方式2・図A・凍結 Keep・N=2・`--no-fast_inmemory` 非破壊。本線ロジック変更は最小（フラグ差替え相当）

**スコープ外:**
- API `end→first` 短縮、Silero 再設計、リップ品質、200 追加、600 復活
- アイドル発話、BGV ズレ、OBS 制御

**Pass 基準:**
- [x] 手段の設計メモ（親承認済み）＋実装
- [x] 350↔250 を再起動なしで切替できる
- [x] 永続化・default・安全弁（許可値以外拒否）
- [x] 切替後 `[resp_timing]` で silence待ち≈設定値を確認
- [x] 短回帰（短発話 or talkover どちらか最小）非破壊
- [x] 親向けサマリーのみ
- [ ] 管理画面 UI から書込 → **defer R2b**

**設計決定（2026-08-06 親 Go）:**
- 手段: 既存 battle/event と同型の **ファイル watch**（例 `in/vad_profile_live.txt`）。HTTP 新設なし（管理画面は当該ファイルを書くだけ）
- 許可値: **350 / 250 のみ**。他は拒否＋現状維持
- 永続: 同ファイルで可（`out/runtime/vad_profile.json` は任意・二重化しないなら不要）
- 起動初期値: **CLI 明示 > 永続ファイル > default 350**
- runtime: 起動後は **control ファイルが常に上書き可**（CLI は初期値のみ。管理画面切替が本旨のためロックしない）
- 切替確認: `[resp_timing]` silence待ち≈設定値

**子報告要約（2026-08-06）:**
- `in/vad_profile_live.txt` watch。default 350。reject＋現状維持。CLI 初期のみ／runtime file 上書き。
- Live: set 350→250、silence待ち 372→256。短回帰 OK。selfcheck OK。
- Keep: `session_loop`＋`in/vad_profile_live.txt`＋R2 tools／ハーネス。

**親判定（2026-08-06）:**
- **Pass-with-defer。** runtime 切替は完了。管理画面 UI 書込は **R2b**。
- Keep All 可（R2 runtime）。commit/tag は R2b 後でも可（今すぐなら R2 分のみでも可）。

---

## Phase R2b: 管理画面→`vad_profile_live.txt` 書込

**目的:** 既存 Streamlit 管理画面から 通常=350／攻め腕=250 を押し、`in/vad_profile_live.txt` へ書くだけ。HTTP 新設なし。session_loop 本線は触らない（R2 済み）。

**主な対象:**
- `scripts/live_runtime/admin_control_panel.py`
- `scripts/live_runtime/battle_runtime_admin_api.py`（`write_*` と同型の `write_vad_profile`）
- 既定パス: `in/vad_profile_live.txt`（R2 と一致）

**スコープ:**
1. UI: 「通常 350」「攻め腕 250」ボタン（または同等）。許可値以外は出さない
2. API: ファイル書込ヘルパ（battle/event 同型）
3. 確認: 画面操作→ファイル内容→（可能なら）既存 Live で `[vad_profile][set]`／`[resp_timing]`。Live 再計測は最小で可
4. 不安定なら攻め腕 UI を捨て 350 固定に戻してよい（親へ報告）

**スコープ外:** session_loop 再設計、API end→first、リップ品質、HTTP サーバ新設、200/600 追加

**Pass 基準:**
- [x] 管理画面から 350/250 をファイルに書ける
- [x] 書込形式が R2 watch と互換（session_loop が受理）
- [x] 最小動作確認（ファイル or Live 1 切替）
- [x] 親向けサマリーのみ

**子報告要約（2026-08-06）:**
- `write_vad_profile`＋ Streamlit ボタン（通常350／攻め腕250）。形式 `250\n`/`350\n`。HTTP・session_loop 非変更。
- Live 主観: UI→file→`[vad_profile][set] 350→250` 確認。silence待ち turn1≈379 → turn3/4≈254–258。主観「turn3以降早い」。攻め腕 UI 維持。

**親判定（2026-08-06）:**
- **Pass。** レスポンス／VAD プロファイル本線（R1→R1b→R2→R2b）クローズ。
- 運用: 通常=350／攻め腕=250（管理画面）。起動 CLI を 350 固定にする必要なし。
- 次本線: **I1 アイドル発話**（2026-08-09）。API `end→first`／BGV／OBS は後続。

---

## Phase I1: 無言→AI アイドル発話

**目的:** ユーザー無言が数秒続いたら AI が自ら発話を開始する。  
**Phase12 idle silent PCM（待機モーション／無音 PCM）とは別機能。** 混同・改修しない。

**スコープ:**
1. **調査（先）:** 既存のターン境界・`activity_start/end`・client content 送信・`turn_idle_wait_s`・battle/event との関係を洗い、トリガー手段を **5 行以内で親に提案**（承認前に本実装しない）
2. **実装（親 Go 後）:** 無言タイマー（CLI 可変・初期案 5–8s）満了 → AI 発話開始。発話中／ユーザー発話中／talkover 中は発火しない
3. **衝突回避:** talkover／割り込み／VAD 350・250 プロファイルを壊さない。ユーザー発話でタイマーリセット／キャンセル
4. **観測:** 発火ログ（例 `[idle_utterance][fire]`）＋ first_audio まで。短回帰（通常ターン＋talkover 1 本）

**スコープ外:**
- Phase12 `idle_silent_pcm` の品質・縫い目チューニング
- API `end→first` 短縮、Silero 再設計、リップ品質、BGV ズレ、OBS 制御
- 管理画面への idle タイマー UI（必要なら後続。I1 は CLI＋動作で可）

**Pass 基準:**
- [x] 設計提案が親承認済み
- [x] 無言 N 秒後に AI 発話が始まる（ログ＋主観または自動）— **会話成立後**
- [x] ユーザー発話／talkover で誤発火しない・割り込み可能
- [x] VAD 350/250・方式2・図A・N=2・`--no-fast_inmemory`・品質凍結 Keep 非破壊
- [x] Phase12 idle silent と混同した改修がない
- [x] 親向けサマリーのみ
- [ ] セッション初手無言 → **defer I1b**（実運用で必要）

**設計決定（2026-08-09 親 Go）:**
- トリガー: **client text**（会話後パス）。server VAD なし。Phase12 `idle_silent_pcm` 非改修
- 計時: `--idle_utterance_s`。cooldown 1s。`turn_idle_wait_s` とは別

**子報告要約（2026-08-09）:**
- 会話後無言: fire→first_audio OK。talkover 短回帰 OK。初手 text のみは `audio_chunks=0`（activity 未追加・報告どおり）
- 主観（idle=3s）: 発話↔AI↔無言↔idle AI を複数回確認。誤発火なし。運用は **3s が妥当**（6s は長い）

**親判定（2026-08-09）:**
- **Pass-with-defer。** Keep: I1 実装。**`--idle_utterance_s` default を 3.0 に変更**（I1 Keep に含む／I1b 着手時に同時で可）
- 初手無言（相手が最初に話さない→AI が口火）は **I1b**（SNS バトル実運用のため必須）
- commit/tag は I1b 後まとめで可（今すぐ I1 のみでも可）

---

## Phase I1b: 初手無言（セッション開始時の口火）

**目的:** セッション開始後、ユーザーが一度も話さないまま無言タイマー満了したときも AI が音声で口火を切る。会話後パス（I1）は維持。

**背景:** text のみでは Live が初手で黙る場合あり（実測）。activity 全面偽装は禁止だが、**初手専用の最小手段**は親承認のうえ検討可（server VAD 復活は禁止）。

**スコープ:**
1. 先に手段を 5 行で親提案（承認前に本実装しない）
   - 既知: 会話後は text のみで OK。初手は text のみ NG
   - 候補例: 初手のみ `activity_start`→text→`activity_end`（ユーザー音声なし）。他手段があれば優先検討
2. 親 Go 後実装。発火条件: セッション内にユーザー発話（または prior turn）がまだ無い＋無言タイマー満了
3. 会話後パスは I1 のまま（余計な activity を付けない）
4. `--idle_utterance_s` default **3.0**（未反映なら同時変更）
5. 短回帰: 初手 idle→first_audio／通常ユーザー発話ターン／talkover。VAD 350/250・図A・方式2 非破壊。Phase12 非改修

**スコープ外:** BGV、OBS、API end→first、リップ品質、server VAD、idle silent 品質

**Pass 基準:**
- [x] 設計が親承認済み
- [x] 初手無言で fire→first_audio
- [x] 会話後パス非回帰
- [x] talkover／通常ターン非破壊
- [x] default idle=3.0
- [x] 親向けサマリーのみ

**設計決定（2026-08-09 親 Go）:**
- 会話後: text のみ（I1）。activity を付けない
- 初手 v1（Fail・Revert 済）: `activity_start`→text→`activity_end`（PCM なし）→ Live 沈黙
- 初手 v2 **採用（候補A）:** 初手のみ `activity_start` → silent PCM ×`min_voice_blocks` → `activity_end` → text
- 会話後パスへ silent PCM／activity を漏らさない。server VAD／response_trigger／Phase12 改修禁止
- `--idle_utterance_s` default=3.0

**子報告要約（2026-08-09）:**
- 初手: `first_turn_prime=1` → silent×6 → end → text → first_audio / chunks=13
- 会話後: `first_turn_prime=0` / text のみ → first_audio 非回帰。talkover OK
- 主観 4t: T1/T2 無言→AI OK、T3–T4 発話後→AI OK

**親判定（2026-08-09）:**
- **Pass。** Keep = I1 + I1b 候補A。commit `bb0ef93` / tag `phase-i1-pass`。アイドル発話本線クローズ。
- 次本線: **B1 BGV 時計特定（調査のみ）**。OBS／API end→first は後続。

---

## Phase B1: BGV 時計特定（調査のみ・実装禁止）

**目的:** BGV↔M0 FG の顔位置・サイズ・向きが徐々にズレる問題について、**まず BGV が今どの時計で進んでいるかを実コード／実ログで一つに確定する。** 修正実装はしない。

**前提:**
- 音声↔M0 は `played_audio_ms` SSOT で同期済
- ズレは BGV↔M0 FG。ターン開始でリセット想定
- 資料: ユーザー手元「新チャット引継ぎ資料（ローカル版 BGV・M0同期ズレ解消）」＋本節

**スコープ（調査のみ）:**
1. 実コードで BGV 進行時計を **必ず一つに確定**（推測禁止）  
   候補: `wall` / `chunk local` / `playlist absolute` / `played_audio_ms`／他なら明記
2. 根拠: ファイル・関数名・式・（可能なら）ログ引用パス。`bg_frame_idx` / `bg_start_ms` / playlist の関係を表にする
3. M0（`played_audio_ms` 系）との差・ズレが蓄積しうる理由
4. 修正方針案のみ（実装・diff 禁止）。資料の同期候補は未決のまま列挙可  
   - played_audio_ms→bg_frame_idx 直接  
   - BGV 側 hold/drop（**口／FG frame drop ではない**。BGV 側の可否は時計確定後に親判断）  
   - 微小速度補正

**スコープ外（B1 禁止）:**
- いかなる同期修正の実装・「300ms ずらせば直る」決め打ち
- 口形 frame drop、音声先行 enqueue、ジッタ延長で隠し、リップ品質本線化、N↑、Silero
- OBS 制御、API end→first、VAD／アイドル発話の再設計
- Phase12 idle silent 品質チューニング

**参照起点（子が実在パスを特定）:**
- M1: `run_mic_input_obs_realtime_session_loop.py` / audio_chunk_player（`played_audio_ms`・ジッタ）
- M0: `tools/run_chunk.py` / `render_core.py`（pose/mouth/expr `t_ms`）
- M3.5: `m3_5/bg_scheduler.py` / `compositor_direct.py` / `run_virtualcam_persistent.py` 等（`bg_frame_idx`・`bg_start_ms`・playlist）

**Pass 基準:**
- [x] BGV 時計を一つに確定（根拠付き）
- [x] M0 時計との差分説明（なぜ徐々ズレうるか）
- [x] 修正方針案（未決列挙可・採用は親）。実装なし
- [x] 親向けサマリーのみ（diff/ログ全文禁止）

**子報告要約（2026-08-09）:**
- Live BGV 時計 = **virtualcam FPS 順次デコード**（毎ループ `cap.read()` → `sleep_until_next_frame`）。`played_*` 非参照。wall ペース。
- FG = `played_audio_ms` → `target_frame`。`bg_frame_idx`/`bg_start_ms`/playlist は Live 未配線（M3.5 offline のみ）。
- 蓄積点: `IDLE_BG_ADVANCE` で非 PLAYING 中も BG 進行・FG 停止。ターン開始は FG sync_meta 更新のみで BG `VideoCapture` はリセットしない。
- コード変更なし。

**親判定（2026-08-09）:**
- **Pass。** 時計確定を受理。
- **IDLE_BG_ADVANCE は単純廃止しない**（Phase12/14 意図的 Keep。待機 BGV 固着の再発リスク）。
- 次=**B2**: ズレ定量→方式選択材料。PLAYING 中は `played_audio_ms→BG frame` 寄せを第一候補として検討。M3.5 playlist 丸ごと移植禁止。実装修正は B3（親承認後）。

---

## Phase B2: BGV↔audio_ms ズレ定量＋方式選択材料

**目的:** Live 本線で「BG 暗黙 frame」と `audio_ms` の同時系列を最小観測し、REBUFFERING／ターン境界での Δ 蓄積を定量する。方式は親が決める。**本格同期修正の実装はしない**（観測ログのみ可）。

**前提（B1 確定）:**
- BG = virtualcam FPS 順次（wall ペース）。FG = `played_audio_ms`
- `IDLE_BG_ADVANCE` 単純廃止禁止（待機中 BGV 維持が Keep）
- M3.5 offline `BgScheduler`/playlist の Live 丸ごと移植禁止

**スコープ:**
1. 挙動不変の最小観測ログ（例: 同時刻の `audio_ms`／推定 BG frame or read 回数／player state）
2. REBUFFERING・ターン境界・定常 PLAYING での Δ(bg, audio_ms) 定量表
3. 方式比較材料（実装せず提案のみ）:
   - **第一候補方向:** PLAYING 中 `played_audio_ms → BG frame` 寄せ
   - BGV 側 hold/drop（口／FG drop ではない。IDLE_BG_ADVANCE との共存設計を含める）
   - 微小速度補正
4. 親が B3 方式を選べる結論（推奨1つ＋棄却理由）

**スコープ外:**
- 同期本実装、300ms 決め打ち、口形捨て、ジッタ延長隠し、N↑、リップ品質、OBS、API end→first
- IDLE_BG_ADVANCE の無条件削除、playlist 移植

**Pass 基準:**
- [x] Δ 定量表（区間・条件付き）
- [x] 方式比較と推奨（親決定用）。本実装なし
- [x] IDLE_BG_ADVANCE／待機 BGV への影響を明記
- [x] 親向けサマリーのみ

**子報告要約（2026-08-09）:**
- 主因: 非 PLAYING 中 IDLE_BG_ADVANCE で BG 進行＋ターンで BG 非リセット。定常 PLAYING は共進で Δ≈0。
- 推奨 A: PLAYING 中のみ `played_audio_ms→BG frame`。IDLE_BG_ADVANCE 維持。速度補正は主手段棄却。
- Keep: 観測 `[B2_OBS]`＋`phase_b2_bg_delta_quant.py`（挙動変更なし）。

**親判定（2026-08-09）:**
- **Pass。** 方式 **A を採用** → B3 実装 Go。
- IDLE_BG_ADVANCE 維持。playlist／微小速度補正は入れない。hold/drop は A の実装手段として可。

---

## Phase B3: PLAYING 中 `played_audio_ms→BG frame` 実装

**目的:** PLAYING 中だけ BG を `played_audio_ms`（＋必要なら loop）で決め、REB／ターン復帰で audio に再ロックする。非 PLAYING は現行 `IDLE_BG_ADVANCE` 維持。

**方式（親確定）:**
- PLAYING: `played_audio_ms → BG frame`（loop 可）
- 非 PLAYING: 順次 `cap.read()` 継続（IDLE_BG_ADVANCE Keep）
- 禁止: playlist 移植、速度補正を主手段、口／FG drop、ジッタ延長隠し、300ms 決め打ち、IDLE_BG_ADVANCE 廃止

**スコープ:**
1. `run_virtualcam_persistent.py` に最小実装
2. REB／ターン復帰で audio に再ロック（持ち越し Δ を切る）
3. 既存 `[B2_OBS]` または同等で Before/After Δ 改善を示す
4. 短回帰: 待機 BGV が固着しない／方式2・図A・N=2・`--no-fast_inmemory`・品質凍結 Keep 非破壊
5. 主観またはログで顔位置ズレ改善（可能な範囲）

**Pass 基準:**
- [x] PLAYING 中 BG が audio_ms に追従（ログ根拠）
- [x] 非 PLAYING で IDLE_BG_ADVANCE 維持（待機固着なし）
- [x] REB／ターン後の持ち越し Δ が改善
- [x] 短回帰・Keep 非破壊
- [x] 親向けサマリーのみ

**子報告要約（2026-08-09）:**
- 方式 A: PLAYING=`audio→BG`（相対ロック＋seek）、非 PLAYING=順次／IDLE_BG_ADVANCE。`[B3_BG_RELOCK]`。
- selfcheck PASS。持ち越し Δ 切断を確認。talkover API 本線は未／割り込み主観で非破壊。
- 主観: 全体は顔位置適合改善。Turn2 で上下ズレ時間帯あり → `bg_mode=audio` なのに `bg_pos` 固定（FG/audio 進行）をログで説明可能。

**親判定（2026-08-09）:**
- **Pass-with-followup。** Keep = B3 本体。
- 次=**B3hf**: playback_state 単一読取で BG/FG 共通化、UNKNOWN 時 a_ms 固着防止、必要なら seek 実フレーム検証。pose.json 切り分けは Hotfix 後の再主観。
- UTF-16 ログは定量ツール側の追従を任意（本線外でも可）。

---

## Phase B3hf: BG 固着／playback_state 読取競合 Hotfix

**目的:** PLAYING 中に `bg_mode=audio` なのに `bg_pos` が固まり FG だけ進む区間を潰す。方式 A・IDLE_BG_ADVANCE は維持。

**疑い（主観ログ根拠）:**
- Turn2: RELOCK 後〜REB 前まで `bg_pos` 固定・`audio_ms` 進行 → 読取競合／UNKNOWN 固着／seek 失敗が第一候補
- pose.json 単独より B3 実装側を先に直す

**スコープ:**
1. playback_state（または同等）の **単一読取**で BG/FG が同じ `audio_ms` を使う
2. UNKNOWN 瞬断時の `a_ms` 固着防止（誤って BG を止め続けない）
3. 必要なら seek 後の実フレーム検証（失敗時のフォールバックをログ付きで）
4. Before/After: 同型主観または `[B2_OBS]` で `bg_mode=audio` 中の `bg_pos` 停滞区間が消える／短縮
5. IDLE_BG_ADVANCE・方式2・図A・品質凍結 Keep 非破壊

**スコープ外:** playlist、速度補正主手段、口／FG drop、ジッタ延長、pose 本線化、OBS

**Pass 基準:**
- [x] 固着メカニズムを特定または有力仮説を閉じる（ログ根拠）
- [x] Hotfix 実装＋`bg_pos` 停滞の改善根拠 — **T1 audio 中停滞は解消。T2 先頭停滞は残**
- [x] 非 PLAYING／IDLE_BG_ADVANCE 非破壊
- [x] 親向けサマリー（再主観依頼の要否を明記）

**子報告要約（2026-08-10）:**
- 原因: playback_state 二重読取。BG が None/UNKNOWN で a_ms を RELOCK 値に固着、FG だけ再読取で進行。
- Fix: 1 tick 1 スナップショット＋last-good PLAYING／seek 検証。selfcheck stall 解消。
- 主観: T1 停滞消えた。T2 先頭は `bg_mode=audio` で bg_pos 固定が残（≈2.4–3.4s）。SEEK_FAIL=0。待ち時間ズレは境界別件の示唆。

**親判定（2026-08-10）:**
- **Pass-with-followup。** Keep = B3hf（二重読取閉鎖は有効）。Fail にはしない。
- 次=**B3hf2**: sync_meta も含め BG/FG resolve 1回／診断ログ（a_ms_bg/desired/bg_pos/a_ms_fg）／T2 先頭固着閉鎖。pose 切り分けはその後。待ち時間（T1→T2 seq なし PLAYING のまま）は同 Phase で切り分けメモ可・本線は T2 先頭固着。

---

## Phase B3hf2: Turn 先頭 bg_pos 固着／境界スナップショット

**目的:** Turn 開始直後に `bg_mode=audio` なのに `bg_pos` が数秒固定する残件を閉じる。B3hf の 1-tick snapshot は維持。

**スコープ:**
1. `playback_state` だけでなく **sync_meta も BG/FG 単一スナップショット**（resolve 1回）
2. 診断ログ: `a_ms_bg` / `desired` / `bg_pos` / `a_ms_fg`（または同等）を併記し分岐を閉じる
3. T2 先頭固着の Before/After（実ログまたは再現ハーネス）
4. T1→T2 待ちで PLAYING のまま `reason=turn` RELOCK する場合の影響を切り分けメモ（本線は先頭固着。別チケット化してよい）
5. IDLE_BG_ADVANCE・方式 A・品質凍結 Keep 非破壊

**スコープ外:** playlist、速度補正主手段、口／FG drop、ジッタ延長、IDLE 廃止、OBS、pose 本線化（切り分けは Hotfix 後）

**Pass 基準:**
- [x] T2（または同型）先頭の bg_pos 停滞が解消／大幅短縮（ログ根拠）
- [x] 診断ログで原因分岐が閉じている
- [x] 短回帰・IDLE_BG_ADVANCE 非破壊
- [x] 再主観要否を明記

**子報告要約（2026-08-10）:**
- 主因: `lock_audio_ms=0` falsy（`or a_ms` で毎 tick 潰れ → desired=lock_bg 固定）。`is not None` で修正。
- sync_meta adopt→resolve 1回共用。`[B3hf2_SNAP]`。selfcheck stall 解消。B3/B3hf selfcheck PASS。
- 主観: T2 先頭固着クローズ（SNAP一致・追従）。**T2 上下ズレは残**（同期分岐型ではない）。待ち時間ズレは本測未再現→バックログ。

**親判定（2026-08-10）:**
- **Pass。** Keep = B3hf2。BGV runtime 同期本線（B1–B3hf2）クローズ可。
- 次=**B4** pose／幾何切り分け（分析のみ）。OBS は B4 で A（または runtime でない）確認後に着手可（Colab 完了待ち不要）。

---

## Phase B4: T2 残ズレ — pose／幾何切り分け（分析のみ）

**目的:** 残る T2 上下ズレについて、主因を **A/B/C の一つに確定**する。実装修正・Colab 改修はしない。

**前提:**
- BGV runtime 同期（方式 A＋二重読取＋falsy-0）は閉じた前提
- 対象主観ログ例: `sess_phase11_subj_20260810_215344`（残ズレ帯）
- 待ち時間 reason=turn RELOCK ズレはバックログ（本線外）

**スコープ（分析のみ）:**
1. 残ズレ帯の `audio_ms`／`bg_pos` と、当該 BGV フレーム顔位置 vs M0／`pose.json` を照合
2. 主因を次の **一つ**に確定:
   - **A)** `pose.json`（ETL 資産）ずれ
   - **B)** M0 適用／幾何変換側
   - **C)** なお runtime 残
3. 根拠表（ファイル・フレーム・数値）。推測で実装提案しない
4. A 確定時: 後続は資産再生成→差替検証（本 Phase ではやらない）。B/C なら親が次手再定義

**スコープ外:**
- 実装修正、Colab スクリプト改善（別途進行・Cursor 本線外）
- playlist／速度補正／口 FG drop／ジッタ／IDLE 廃止、OBS 実装（B4 判定後）

**Pass 基準:**
- [x] A/B/C を一つに確定（根拠付き）
- [x] 残ズレ帯の照合表
- [x] 次手提案 3 行（実装しない）
- [x] コード変更なしが原則

**子報告要約（2026-08-10）:**
- **B 確定**（M0 適用／pose インデックス）。A 否定（pose[i]↔BGV i 整合）。C 否定（B3hf2 sync 維持）。
- T2: BGV=`bg_pos=274+audio/40` なのに M0 は turn ローカル `pose[audio/40]`。`p_use.ty−p_bg.ty` absmean≈41.5。

**親判定（2026-08-10）:**
- **Pass（分析）。** 主因 B。
- 次=**B5** pose 適用修正。資産再生成は本因対応にしない。OBS は **B5 後**。待ち時間 RELOCK ズレはバックログ。

---

## Phase B5: pose スライス時計を BGV 継続 index へ結ぶ

**目的:** turn ローカル `t_ms=0` 起点の pose 適用をやめ、表示中 BGV（`bg_pos` / `frame_offset` 相当の絶対 index／絶対 t_ms）と pose を一致させ、T2 上下ズレを解消する。

**前提（B4）:**
- ETL `pose.json` 資産そのものは主因ではない（A 否定）
- BGV runtime sync は閉じた前提（C 否定・方式 A Keep）
- Colab／pose 再生成は本因対応にしない

**スコープ:**
1. 実コードで「誰が turn ローカル pose スライスを切っているか」を特定（M0 / session_loop / virtualcam の適用点）
2. **最小修正:** pose 参照を BGV 継続 index（または等価な絶対 t_ms）へ結ぶ。等価なら親承認可能な別手段可
3. Before/After: T2 帯で `pose_i ≈ bg_pos`（または ty 差が主観帯で縮小）をログ／数値で示す
4. 短回帰: 方式2／図A／N=2／`--no-fast_inmemory`／IDLE_BG_ADVANCE／B3 sync Keep 非破壊
5. 主観: T2 上下ズレ改善（親または子）

**スコープ外:**
- pose 資産のフル再生成を本因対応にする、Colab 改修、OBS 実装
- playlist／速度補正／口 FG drop／ジッタ延長／IDLE 廃止
- 待ち時間 RELOCK 別件の本線化

**Pass 基準:**
- [x] 適用点特定＋修正（最小）
- [x] T2 定量改善（pose↔bg 一致または ty 差縮小）— ideal lock で ty absmean 0
- [x] 短回帰・Keep 非破壊
- [ ] 主観で T2 残ズレ改善 → **Fail**（悪化。方式は正しい）
- [x] 親向けサマリーのみ

**子報告要約（2026-08-11）:**
- 方式: `bg_cursor.json` → turn 初回 `pose_base_frame` → pose のみ絶対 t_ms。mouth は turn 局所のまま。
- 定量: T2 ideal lock で ty 差 0。selfcheck 群 PASS。
- 主観 Fail（232331／232453）: T1 で idle 初回 freeze が RELOCK と恒常 Δ≈−27；T4 で `bg_cursor=missing`→`pose_base=0`（gap 数百）。T2/T3 の turn RELOCK 時は gap≈−2 で方向正しい。

**親判定（2026-08-11）:**
- **主観 Fail／方式 Keep。** 正しい設計＝pose も BGV 継続絶対 index。悪化は snapshot／missing→0 の運用バグ。
- 次=**B5hf**（下記）。方式破棄しない。検証不能時のみ B5 一時 Revert 可（親指示時）。
- OBS・Colab・資産再生成は今やらない。

---

## Phase B5hf: pose_base snapshot／missing→0 Hotfix

**目的:** B5 方式（pose＝BGV 継続絶対 index）を維持したまま、誤 `pose_base` 固定を潰し主観の上下ズレ悪化を戻す／改善する。

**スコープ:**
1. **cursor 欠落時 `pose_base=0` 禁止** — retry／last-good／失敗時は turn ローカルへ**明示**フォールバック（黙って 0 にしない）
2. **T1:** idle 初回即 freeze やめ、PLAYING／RELOCK 系の `ideal_base`（≈`bg − audio/step`）へ寄せる（または等価）
3. 回帰: 主観再測＋ターンごと `|B5_POSE_BASE − 直後 RELOCK bg|` を定量（数フレーム以内を目標）
4. Keep: 方式 A／IDLE_BG_ADVANCE／B3hf2／図A／N=2／`--no-fast_inmemory`／品質凍結。mouth は turn 局所のまま

**スコープ外:** 方式破棄、OBS、Colab／資産再生成、playlist／速度補正／口 FG drop／ジッタ延長／IDLE 廃止

**Pass 基準:**
- [x] missing→0 経路が閉じている（ログ根拠）
- [x] T1 の恒常 Δ（idle freeze vs RELOCK）が解消／大幅縮小
- [x] ターンごと |POSE_BASE−RELOCK| 定量表
- [x] 主観: 前 B5 比で悪化なし／T2 残ズレ改善 — T1/T2/T4 OK、**T3 残**
- [x] 短回帰・Keep 非破壊

**子報告要約（2026-08-11）:**
- missing→0 禁止（retry／last-good／`turn_local`）。`ok_audio` で ideal_base freeze。cursor に audio_ms／ideal_base 追加。
- 主観 162517: T1/T2/T4 顔位置ほぼ一致。T3 のみ大ズレ。|gap| T1/T2/T4≈0–4、**T3≈156**（pose_base=301 vs RELOCK ideal=457）。
- T3 根因: 前ターン PLAYING のまま新ターンが旧 cursor を `ok_audio` freeze → 後の enter_playing RELOCK と恒常Δ。

**親判定（2026-08-11）:**
- **Pass-with-defer。** Keep = B5hf。次=**B5hf2**（ターン境界の誤 freeze）。方式破棄しない。OBS・資産再生成はまだ不要。

---

## Phase B5hf2: ターン境界の誤 pose_base freeze

**目的:** 前ターン PLAYING 残留 cursor を新ターンで即 `ok_audio` freeze しない。新 fo の RELOCK／整合後にだけ絶対 `pose_base` を固定する。

**スコープ:**
1. `ok_audio` freeze は **当ターン fo と cursor／RELOCK が整合した後のみ**（cursor に fo 付与、または fo↑後は当面 seq／provisional `turn_local`）
2. ターン開始後、新 fo の `enter_playing`／`turn` RELOCK 前は freeze 禁止（provisional turn_local 維持）
3. 短回帰＋主観: T3 相当（待ち短く前ターン PLAYING 残留）＋ BGV 上下大振幅帯
4. Keep: 方式 A／IDLE_BG_ADVANCE／B3hf2／図A／N=2／`--no-fast_inmemory`／mouth turn 局所／B5 絶対 index 方式／B5hf missing→0 禁止

**スコープ外:** 方式破棄、OBS、Colab／資産再生成、playlist／速度補正／口 FG drop／ジッタ／IDLE 廃止

**Pass 基準:**
- [x] T3 型 |POSE_BASE−RELOCK| が数フレーム級（ログ）
- [x] T1/T2/T4 非回帰
- [x] 主観で T3 大ズレ改善（親再測可）
- [x] Keep 非破壊

**子報告要約（2026-08-11）:**
- cursor に `frame_offset`。`ok_audio` freeze は fo 一致時のみ。不一致は `fo_wait` provisional。
- selfcheck: T3 指紋 |gap| 156→0。主観 164529: T1–T4 OK。T3 大ズレ閉鎖。T2 高速上下で数フレーム遅れ気味（|gap|≈4、必須 Hotfix なし）。

**親判定（2026-08-11）:**
- **Pass-with-defer（Bライン運用 Keep）。** B5hf2 まで合格。仕上げは Colab/新資産後に再開可。
- 次本線=**OBS 制御（O1→O2→O3）**。

---

## Phase B6: 合成瞬間 display_bg↔pose Δ（調査のみ・実装禁止）

**目的:** 再開条件「ズレ再発」で B 仕上げに入る。B 全面再オープンではない。高速上下で、**合成瞬間**の表示 BGV 枚（`display_bg_idx`）と pose/M0 枚（`pose_idx`）の Δ を測る。オフセット実装は親が Δ を見てから。

**再開前提（ユーザー確定 2026-08-23）:**
1. 最優先 = BGV顔位置 vs M0顔位置（Y）。口−音は別チケット
2. JP と EN の両方
3. 今は代表 BGV 1本。後で複数ランダムループ。1本専用オフセット禁止。常時「表示枚目=pose枚目」
4. Colab しない。Colab pose は横の微細用。資産の上下が大きく違う問題ではない
5. pose だけ先送り禁止（体と頭が離れる）。オフセットするなら表示 BGV と pose を同じ index のまま

**設計 SSOT（ユーザー）:**
- BGV N枚目 → pose.json[N]（40ms）→ M0 も N。`audio_ms` が選ぶ枚は同じ N であるべき
- 同じ N 同士で目視ズレ → pose/貼り位置（今は第一仮説にしない）
- Live の上下追従遅れ → 合成瞬間に BGV枚 ≠ pose/M0枚 が本命候補。`pose.json` を先に直さない
- ユーザーは **仮説2（idle/ターン境で BGV だけ進行）** を厚く見る。計測項目は変えない
- Δ>0 → 枚を合わせる（方式C）。Δ=0 → 貼り位置

**相談確定（初手・守る）:**
- 「`audio_ms` がカウンター／OS遅延」は相対 Y ズレの第一因にしない（同じ時計なら相対は保つ）
- 優先度2の +40〜80ms を初手実装にしない（B1 が禁止した決め打ち）
- B3 Keep が Live で生きているかを先に見る（PLAYING 中 `played_audio_ms→BG frame`）

**スコープ（調査のみ）:**
1. B3 Keep が Live で生きているか（PLAYING 中 BG が `played_audio_ms` で選ばれているか。死んでいればその根拠）
2. 高速上下帯の合成瞬間で `display_bg_idx` と `pose_idx` の Δ 表（JP + EN、代表 BGV 1本）
3. 仮説2を厚く見るため、idle/ターン境の行を表に分ける（計測式は同じ）
4. 判定材料のみ: Δ が 1–3 なら遅れ側を同じ枚に合わせる候補／Δ=0 なら貼り位置候補。実装しない

**スコープ外（B6 禁止）:**
- いかなる同期／オフセット／pose 先送りの実装
- `audio_ms` 全体 +40〜80、ジッタ延長、口 frame drop、音声先行 enqueue
- B 全面再設計、方式 A／pose=絶対 index／IDLE_BG_ADVANCE 破棄
- Colab、新 pose 再生成、多 BGV、1本専用オフセット
- N↑、図A破壊、本番 prompts 編集、session_loop で Zoom 受信ミックス
- `docs/PROGRESS.md` 編集

**Pass 基準:**
- [x] B3 Keep 生死を根拠付きで一つに
- [x] JP + EN、高速上下、合成瞬間 Δ 表（idle/ターン境を分けてよい）
- [x] Δ の代表値（中央／最大／境での符号）と「枚合わせ候補 vs 貼り候補」1行
- [x] コード変更なし（観測ログ追加のみ可。挙動不変）
- [x] 親向けサマリーのみ（diff／ログ全文禁止）

**子報告要約（2026-08-23）:**
- B3 Keep **生き**。PLAYING 中 `_b3_desired_bg_frame` → `_read_bg_at_frame`（`bg_mode=audio`）。JP/EN とも `[B3_BG_RELOCK]` と `[B3hf2_SNAP] a_ms_bg=a_ms_fg / desired=bg_pos`
- Δ=display_bg_idx−pose_idx（+ = BG進み）。定常 PLAYING: JP 中央0 最大11 / EN 中央1 最大275。高速上下: JP 0/11 / EN 2/275
- idle境: JP 中央4 最大456 / EN 50/696。ターン境: JP 263/626 / EN 18/883。符号はすべて BG進み
- 結論: **枚合わせ候補**。仮説2が厚い。観測 `[B6_DELTA]` は挙動不変（virtualcam / step1 + 集計 `phase_b6_delta_quant.py`）

**親判定（2026-08-23）:**
- **Pass。** 表採用。方式 **C** 採用（境で進んだ BG を遅れ側 pose/M0 と同じ枚へ）。
- 出さない: pose 先送り／IDLE_BG_ADVANCE 廃止／`audio_ms` オフセット／pose.json 先修正。B3 Keep。`[B6_DELTA]` Keep 可
- EN 定常 PLAYING 最大275は境漏れ候補。次=**B7** で境揃え後に残るか見る。

---

## Phase B7: 境で進んだ BG を pose/M0 と同じ枚へ（方式C）

**目的:** idle/ターン境で BGV だけ進んだ Δ を、**表示 BG を遅れ側の pose/M0 枚に戻す**ことで閉じる。pose は先送りしない。同じ N のまま揃える。

**方式（親確定）:**
- 方式 C: 境イベント（idle入り／idle明け／ターン開始／enter_playing・RELOCK）で `display_bg_idx = pose_idx`（進んだ BG を戻す）
- PLAYING 中は B3 Keep（`played_audio_ms→BG frame`）
- 非 PLAYING の IDLE_BG_ADVANCE（待機モーション）は廃止しない。毎 idle tick で BG を pose に吸い付けて待機 BGV を固着させない
- pose＝BGV 絶対 index Keep。1本専用オフセット禁止

**スコープ:**
1. 境での最小実装（表示 BG を pose/M0 と同じ枚へ）。適用点は子が特定（合成 tick または BG 選択）
2. After を B6 と同じ表で出す（JP+EN、同じ Δ 式、同じ区間分け）。`[B6_DELTA]` 再利用
3. EN 定常 PLAYING 最大 Δ が境揃え後に残るか（漏れなら境分類の再掲のみ。勝手に第二手法を足さない）
4. 短回帰: 待機 BGV 固着なし／B3 生き／方式2・図A・N=2・`--no-fast_inmemory`・品質凍結 Keep 非破壊
5. 主観: 境と高速上下で BGV顔 vs M0顔（Y）。口−音は見ない

**スコープ外（B7 禁止）:**
- pose 先送り、IDLE_BG_ADVANCE 廃止、`audio_ms` +40〜80／全体オフセット、pose.json 先修正
- 口 frame drop、音声先行 enqueue、ジッタ延長、N↑、図A破壊
- Colab、多 BGV、1本専用オフセット、B 全面再設計
- 本番 prompts、session_loop Zoom ミックス、`docs/PROGRESS.md` 編集

**Pass 基準:**
- [x] 境（idle / ターン）の Δ 中央が数フレーム級へ縮小（Before は B6 表）。最大は defer
- [x] 定常 PLAYING が 0〜1 近傍を維持。EN PLAYING 最大の残否を明記 — **消えた**（275→1）
- [x] IDLE_BG_ADVANCE 維持（待機 BGV が止まらない）
- [x] B3 Keep 維持（PLAYING 中 `bg_mode=audio`）
- [x] 短回帰・Keep 非破壊。親向けサマリーのみ
- [x] 主観: JP 顔Y解消＋ EN `sess_en_live1_subj_20260824_140931` 音声後4t 一致

**子報告要約（2026-08-24）:**
- 適用: `run_virtualcam_persistent.py` の BG 選択＋合成 tick。境は idle入り / fo↑ターン開始 / enter_playing・turn RELOCK のみ。`display_bg_idx=pose_idx`。pose 非移動
- After: 定常 PLAYING JP/EN 中央0 最大1。高速上下 中央0 最大1。JP ターン境 中央0 最大7。EN ターン境 中央0 最大131。idle境 JP 中央0 最大115 / EN 中央5 最大116
- IDLE 固着なし（境後 seq 進行。毎 tick 吸い付けなし）。B3 生き。Revert 悪化なし
- 残: idle最大=起動 BUFFERING（初回 enter_playing 前）。EN ターン最大=turn_local→absolute 窓。第二手法なし
- 変更: `run_virtualcam_persistent.py` / `phase_b7_boundary_snap_selfcheck.py`
- JP 主観ログ Tee 不発（`134238` は rss のみ）。判定は手元主観＋無人定量＋ EN 主観ログ

**親判定（2026-08-24）:**
- **Pass-with-defer。** 方式C Keep。②（BGV顔Y vs M0顔Y）クローズ。B8／第二手法なし
- Fail/Revert にしない（冒頭無反応＋ズレは T1 実PCMまで idle_silent＋IDLE_BG_ADVANCE。enter_playing で BG を pose へ戻す）
- 開かない: 貼り位置／口−音／Colab／pose.json／`audio_ms` オフセット
- 次本線=**V1**（2026-08-25 着手）。commit/tag `phase-b7-pass`（`813c641`）＋ `main` FF は **完了**（2026-08-25）

---

## Phase V1: Live 声 Kore 固定

**目的:** Live 本番の声を女性 prebuilt **Kore** に一本化する。prompt に「女性」と書くだけでは足りない。`LiveConnectConfig.speech_config` で固定する。

**確定事実:**
- 本番 = `run_mic_input_obs_realtime_session_loop.py` の `_build_live_config`
- 現状 `speech_config` 未設定 → 声が混在（Live 未指定の既定は Puck＝男性）
- JP/EN は同じ Live 接続。`if language` 禁止。`language_code` の JP/EN 分岐も出さない（声だけ固定）
- 非 Live TTS 既定は Kore。公式 Live 例も Kore。**声名は Kore 固定**（Aoede 等の試聴・A/B・CLI 切替は本 Phase 外）
- Kore がこの model（既定 `gemini-3.1-flash-live-preview`）で拒否されたときだけ 1 行で止めて親へ。30 声を漁らない

**スコープ:**
1. `_build_live_config` の両分岐（tools あり/なし）に  
   `speech_config=SpeechConfig(voice_config=VoiceConfig(prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Kore")))`
2. 既存 `[session_loop][live_config]` に `voice_name=Kore` を出す
3. 本番経路だけ。死んでいる probe / worker の LiveConnectConfig は触らない（まだ Live 音声を出しているなら同じ Kore。推測で広げない）
4. `--voice_name` CLI は作らない（一本化＝固定）

**スコープ外（V1 禁止）:**
- prompt / prompts_en / JP prompts 編集（希望順②）
- B 再開、貼り、口−音、N↑、ジッタ延長、音声先行 enqueue、session_loop Zoom ミックス、二重 Live
- 言語 if、Puck 残置、声のランダム、Aoede A/B
- `docs/PROGRESS.md` 編集

**Pass 基準:**
- [x] ログに `voice_name`（当初 Kore。Keep は **Aoede**）＋ `voice_wire` / `speech_setup`
- [x] 主観: EN 女性。JP は中性として同一 Aoede の言語差を受容（これ以上の声漁り・prompt 女性化は②）
- [x] Keep 非破壊: 方式2・図A・N=2・`--no-fast_inmemory`・jitter 300/240・silence 350・B7 方式C・B3
- [x] 親向けサマリーのみ。追加 A/B・CLI・言語 if なし

**子報告要約（2026-08-25〜26）:**
- `_build_live_config` 両分岐に `speech_config`。本番は `--inline_emo_tag_mode` のため tools なし。probe / worker / `--voice_name` CLI 未追加
- 初手 Kore は connect OK だが、google-genai 2.10.0 mldev が snake_case で送り Live が無視 → Puck 既定のまま。`t_live_speech_config` を camelCase alias dump
- Kore は wire 到達後も JP 主観が男性 → **Aoede 1本**（A/B・30声漁りではない）
- オペレーター主観: EN `134218` 女性。JP `134051` / `134121` 中性。同一 Aoede の言語差として受容
- 変更: `run_mic_input_obs_realtime_session_loop.py` のみ。Keep All は親（commit / tag）

**親判定（2026-08-26）:**
- **Pass-with-note。** 声一本化は成立。Kore 名は Keep できない
- Keep = prebuilt **Aoede**（両分岐）＋ camelCase wire。Puck 既定に戻さない。`--voice_name` CLI なし。JP/EN 同じ接続。`language_code` / `if language` なし
- 出さない: 追加の声 A/B、CLI、JP/EN 言語 if、prompt 編集、B/O 再開
- 次本線=**P1**（2026-08-26 着手）。Keep All（commit / tag `phase-v1-pass`）は親実施済。`main` FF 済
- Keep All（commit / tag `phase-v1-pass`）は **親が実施**。子は commit / tag / Keep しない

---

## Phase P1: EN 本番システムプロンプト差し替え＋テスト

**目的:** 本番 EN の system prompt を通常／バトル 2 系統の `--prompt_dir` に分け、割り込み／主導権定型を EN dir 横の英語ファイルから読む。Studio 検証済文面はバトル正。要約・改作しない。

**確定事実（再調査で覆すな）:**
- 本番 EN = `--prompt_dir configs/prompts_en`。長さ SSOT は今 `00_base` の "Keep replies short, about one sentence."
- loader は dir 内を全部連結する → **同じ dir に 20 と 30 の長さ方針を両方書くな**。切替は `--prompt_dir` 2系統
- `prompts_en_dur` は DUR 検証用。本番例にしない。通常パスに残すな
- JP `configs/prompts` は無断上書き禁止
- `_build_battle_interrupt_prompt` と admin 定型は日本語の「短く／1文」。control は system 一文に負ける。主導権＝床取り（mute）。「短く話せ」は用途と逆
- `if language` 禁止。英語定型は **prompt_dir 横の同名ファイル**（JP dir は日本語のまま）
- Studio 文面 SSOT = `US_system_prompt_1.md`（Downloads 同名可）。**prompts_en_battle の 10/30**。通常 `00/20` に丸ごと載せるな
- Studio 自己対戦は文面の事前検証。M1 で二重 Live は出さない。Pass は本番 EN 1本＋人間/interrupt
- 「short, snappy」はテンポ。00_base の1文硬拘束ではない。バトル＝さえぎられるまで。few-shot は 2–4 文でよい
- V1 Keep（Aoede＋camelCase）は触るな。声を prompt で女性化しない
- 現行 `prompts_en/00_base` の emo_id 規則と「[emo:] を読み上げるな」「Never reply in Japanese」は残す。M1 は inline_emo 必須
- `[System: Phase/Action]` は既存 interrupt/control ファイルへ載せるだけ。session_loop に新プロトコルを足すな

**スコープ:**
1. `configs/prompts_en`: `00_base` から1文硬拘束を外す。`20_normal`＝1〜2文。`30_battle` は空のまま
2. 新 `configs/prompts_en_battle`: 同じ 00（1文なし）＋ Studio 原文を `_studio_source.md` にコピーしてから 10/30 へ分割。`20_normal` は空。`30`＝さえぎられるまで話してよい
3. 割り込み／主導権定型の英語ファイルを EN prompt_dir 横に置き、admin と session_loop ラッパは **今の prompt_dir から読む**。英語文面は床取り（短く話せ／1文で返せ、を書くな）
4. テスト（ローカル mic。Banana/Zoom 不要。Zoom で主導権 mute+interrupt は使うな）:
   - EN 通常 dir・必須3フラグ・Aoede Keep・2–3t: 英語で 1〜2文（DUR の 60s 化は Fail）
   - EN battle dir 短確認: 一文で終わらず、割り込みで切れる
   - 英語 interrupt が実際に飛ぶこと
5. V1 Keep・方式2・図A・N=2・`--no-fast_inmemory`・jitter 300/240・silence 350・B7 方式C 非破壊

**スコープ外（P1 禁止）:**
- JP prompts 編集、`prompts_en_dur` を本番化、`--no-skip` 本線化、I1 を長文装置化
- session_loop に language if、Aoede/camelCase 改変、B 再開、N↑、ジッタ延長、音声先行 enqueue、二重 Live、③ catalog
- Studio 文面の要約・改作・「もっと短く」、声の prompt 女性化
- `docs/PROGRESS.md` 編集。commit / tag / Keep All（親がやる）

**Pass 基準:**
- [x] 上のテスト＋ JP prompts 未変更＋ 通常/バトルが prompt_dir で切替
- [x] Studio 文面が battle 10/30 に載り、通常 00/20 に丸ごと入っていない
- [x] 親向けサマリーのみ。口 barge-in は既存 talkover 例外に接続（140322 Fail → 175716 Pass）

**子報告要約（2026-08-26〜27）:**
- 通常 dir: 20あり・30空。battle dir: 20空・30=Studio。同居なし。`_studio_source.md` は Downloads と SHA256 一致。emo は両 00 に残置
- interrupt/leadership は今の `--prompt_dir` から読む。JP dir に無いときは日本語 fallback。英語 wrap は飛ぶ（sidebar raw が JP でも Live へは EN）
- 無人: 通常 `212912` 1〜2文。battle `213855` 一文で終わらず＋管理 interrupt cut_in
- オペレータ通常: `133310` / `133646`（1〜2文。60s 化なし。かぶりは EN-LIVE1 観察・Gate にしない）
- battle `140322` は口で止まらない＝実装漏れ Fail。口 barge-in を既存 talkover `clear_queue` に接続（mute は切らない）
- 再主観 `175716`: 口で停止。`barge_in` playback_watch + mic_voice → `clear_queue:talkover_cut_in`。PLAYING 中 Δ 0±1。cut 後 BUFFERING の Δ は B 再開しない
- JP prompts / `prompts_en_dur` / Aoede wire 未変更。`if language` なし

**親判定（2026-08-27）:**
- **Pass-with-note。** 口 barge-in を既存 talkover 例外に接続。二重 InputStream は監視（本線化しない）
- Fail にしない: admin sidebar 既定 JP（Live wrap は EN）。先頭〜10s 無音＝接続待ち。かぶり気味。cut 後 BGV Δ
- 出さない: この子への追作業、再生中 mic の新経路、B 再開、ジッタ延長、主導権 Zoom、二重 Live、③④
- Keep All（commit / tag `phase-p1-pass`）は **親が実施**。子は commit / tag / Keep しない。`in/*.txt` は入れない。`main` FF 済。次=**E1**

---

## Phase E1: イベント動画 catalog 最大10＋管理画面プルダウン

**目的:** `in/event_catalog.json` を最大10件の SSOT にし、管理画面は catalog プルダウンから `event_runtime_live.txt` へ投入する。既存 event_runtime（VirtualCam 挿入）は壊さない。

**確定事実（覆すな）:**
- 現行 catalog は 2件（`evt_001` / `evt_voice_001`）。投入は `event_runtime_live.txt`
- 管理画面は event_id 手入力＋決め打ち2ボタン。catalog を読んでいない
- イベント動画＝既存 event_runtime（VirtualCam 側の挿入）。OBS「背景」静止画/BGM（O1）とは別。VirtualCam 内 BGV（猫の体）は切替対象外
- 画面コンパクト化は同時で可（制御を消すな。expander 等で短くするだけ）

**スコープ:**
1. catalog は **最大10**。今の2件は残す。実ファイルの無い id を捏造するな（空枠8個を作るな）
2. 管理画面は catalog のプルダウンから選んで投入。決め打ち2ボタンは不要（dropdown が正）
3. 既存 write_event / session_loop の event 経路は壊すな。新プロトコル・session_loop sleep・OBS でイベント動画再生は出さない
4. コンパクトは同じ PR で可。主導権／VAD／OBS／当てフリ／スミスは残す
5. 短確認: 画面に catalog（≤10）が出る。1件選ぶと `event_runtime_live.txt` にその event_id。Live が走っていれば既存どおり挿入（Zoom 不要）

**スコープ外（E1 禁止）:**
- B 再開、ジッタ延長、N↑、音声先行 enqueue、主導権 Zoom、二重 Live、④ docs
- JP prompts / `prompts_en_dur` / P1 文面の改作
- event を OBS WS 再生に付け替える、catalog を 10 超、ダミー mp4 量産
- P1 子の再利用。`docs/PROGRESS.md` 編集。commit / tag / Keep All（親がやる）

**Pass 基準:**
- [x] プルダウン＝catalog、最大10、既存2件生存、1件投入がファイルに出る
- [x] `evt_001` が頭から順再生（末尾飛びクローズ）— `232324` first_frame idx=0 total=73
- [x] Keep 非破壊（P1 / V1 / 方式2 / 図A / N=2 / jitter / B7 PLAYING / O1 音声ルーティング）
- [x] 親向けサマリーのみ。完了ゲートは defer（E1b）

**子報告要約（2026-08-27）:**
- catalog 2件（捏造なし）。決め打ち2ボタン廃止。投入は既存 `write_event`
- 管理画面は M1 `in/event_catalog.json` のみ読む。M3.5 `in/` スキャンしない
- compact: expander。主導権／VAD／OBS／当てフリ／スミスは残置
- イベント中 sequential_from_0。復帰時は古い pose lock を捨てる（`220810` 顔ずれ → `222518`/`222628` 主観OK）
- E1hf: `232324` `evt_001` 発話中投入で頭から最後まで。first_frame idx=0 total=73。override 中 B7 なし。復帰 0→102（古い pose 63 ではない）。`pos_after_count=0` で FRAME_COUNT→末尾仮説は否定
- ②未実施: turn2 first_audio は restore 前。完了ゲートなし

**親判定（2026-08-27）:**
- **Pass-with-defer。** 完了ゲートは E1b（親設計ロック後の新子）。session_loop sleep／ジッタ延長／M0停止ゲートは出さない
- Keep All（commit / tag `phase-e1-pass`）は **親が実施**。子はしない。`in/*.txt` は入れない
- 出さない: ④、E1b 今すぐ、B 再開、ジッタ延長。次本線=なし（指名待ち）

---

## Phase O1: OBS WebSocket＋管理画面＋背景静止画/BGM切替

**目的:** 既存管理画面（Streamlit／`battle_runtime_admin_api`）から OBS WebSocket で、**OBS側背景静止画**と **BGM** を配信中に指定・切替できる。

**確定前提:**
- OBS「背景」＝OBS 側静止画ソース（将来 OBS 側背景動画もありうる）。**VirtualCam 内 BGV（M0 合成用・透過）は切替対象外・触らない**
- 音声: **AI＝player→PC デバイスを OBS が拾う／BGM＝OBS メディアソースのみ**（AI PCM に混ぜない）
- 本番トリガ＝管理画面（Slack 例は参考のみ）。パスワード等は設定化（ハードコード禁止）
- session_loop を `sleep` でブロックしない（OBS I/O は別タスク／プロセス／非ブロッキング）

**スコープ:**
1. `obsws-python`（または同等）で OBS WebSocket 接続（host/port/password 設定化）
2. 管理画面に背景静止画・BGM の一覧指定／切替 UI（既存 panel 共通化）
3. メディア／画像ファイル配置方針（リポ内 or 設定パス）と OBS ソース名マッピング
4. 接続失敗時の安全なログ／UI 表示（Live パイプラインは継続）
5. 短確認: 切替が OBS に反映。図A／方式2／Bライン／VirtualCam・pose・口形 **非触**

**スコープ外（O2/O3）:** 当てフリ、スミス、音声フィルタ、Python scale、VirtualCam BGV 切替

**Pass 基準:**
- [x] WebSocket 接続（設定化）＋管理画面から背景静止画切替
- [x] 管理画面から BGM 切替（OBS メディアソース）
- [x] AI/BGM ルーティング遵守・session_loop 非ブロック・B/VCam 非破壊
- [x] 親向けサマリーのみ。O2 提案 3 行以内

**子報告要約（2026-08-12〜13）:**
- Admin Streamlit→OBS WS 直結。`obs_control_config.json`＋local／env。既定ソース `OBS_BG_Still`／`OBS_BGM`。
- 実機: 背景静止画／BGM 切替成功。flat local の password 非マージ → Hotfix（websocket.* 正規化）。example から実パスワード除去。
- session_loop／VirtualCam／B 非触。

**親判定（2026-08-13）:**
- **Pass。** Keep = O1＋local schema Hotfix。パスワードは gitignore local か env（コミット禁止）。
- 次=**O2** 当てフリ（事前配置見せ消しのみ）。

---

## Phase O2: 当てフリ（OBS 事前配置ソースの見せ消し）

**目的:** AI 発話開始に合わせ、OBS 上の通常ソース⇔事前ドアップソースを **見せ消しのみ**で切替する。default ON・管理画面で OFF。

**確定前提:**
- Python scale／座標拡大 **禁止**。M0／VirtualCam で拡大しない
- OBS に通常用・ズーム用を事前配置。制御は可視性切替のみ
- トリガ: AI 発話開始（既存 first_audio／PLAYING 相当の観測点を子が特定→最小配線）。本番 UI は管理画面（ON/OFF）
- Slack 本番トリガ禁止。session_loop を sleep でブロックしない
- VirtualCam 内 BGV・Bライン・口形パイプライン非触
- 音声ルーティングは O1 どおり（AI＝player→OBS拾い／BGM＝OBS メディア）

**スコープ:**
1. 設定に通常／ズームソース名＋当てフリ default ON
2. 管理画面トグル OFF で無効化
3. AI 発話開始→ズームソース表示／通常非表示（終了または次境界での戻し方針を短く決めて実装）
4. OBS 未接続時は失敗しても Live 継続
5. 短確認: ON で発話開始時に切替／OFF で切替なし。O1 背景/BGM 非破壊

**スコープ外:** スミス（O3）、増殖加速、Python transform アニメ、B 仕上げ

**Pass 基準:**
- [x] default ON で発話開始時に見せ消し切替（実機または同等確認）
- [x] 管理画面 OFF で無効
- [x] scale/座標拡大なし・session_loop 非ブロック・B/VCam 非破壊
- [x] 親向けサマリー＋O3 申し送り 3 行

**子報告要約（2026-08-13）:**
- IN=`first_audio`（bootstrap/warmup 除外）→ `to_thread` 見せ消し。OUT=ターン終了。live overlay OFF で無効＋即通常戻し。
- 実機 `sess_phase11_subj_20260813_152850`: turn1–4 すべて zoomed=1→0。OFF テストも正常。
- Keep: 見せ消しのみ／O1 非触／B・VCam・図A 非破壊。

**親判定（2026-08-13）:**
- **Pass。** Keep = O2。次=**O3** スミス＋音声フィルタ（増殖加速は後続可）。

---

## Phase O3: エージェントスミス＋音声フィルタ

**目的:** 管理画面 **1 ボタン**で、OBS 事前配置クローンソースを表示し、OBS 音声フィルタを ON にして「群唱感」を出す。

**確定前提:**
- クローンは OBS 事前配置（同一 VirtualCam／映像ソースの参照コピー想定）。Python で動的ソース生成は必須にしない
- 制御は可視性＋フィルタ enable（必要なら最小の transform は親承認後のみ。連続 scale アニメ禁止）
- 当てフリ `first_audio` トリガは流用しない（明示ボタン／コマンド）
- AI PCM と BGM メディアは混ぜない（O1 ルーティング維持）
- session_loop を sleep でブロックしない（増殖間隔は管理画面プロセス／別タスク側）
- VirtualCam 内 BGV・Bライン・口形・図A・方式2 非触
- 増殖加速（delay 短縮）は **後続可**（O3 では固定間隔 or 全表示の最小版で可）

**スコープ:**
1. 設定: クローンソース名リスト＋音声ソース名＋フィルタ名（例「スミス効果」）
2. 管理画面 1 ボタン: フィルタ ON＋クローンを順に（または一斉に）表示
3. OFF／リセットボタン（表示解除＋フィルタ OFF）推奨
4. OBS 未接続時は失敗しても Live 継続
5. O1/O2 非破壊の確認

**スコープ外:** 増殖加速の本格演出、Slack 本番トリガ、B 仕上げ、当てフリ改修

**Pass 基準:**
- [x] 1 ボタンでクローン表示＋音声フィルタ ON（実機確認）
- [x] リセット／OFF で元に戻せる
- [x] session_loop 非ブロック・AI/BGM 非混線・B/VCam/O1/O2 非破壊
- [x] 親向けサマリーのみ

**子報告要約（2026-08-14）:**
- 最小版: 一斉表示＋`SetSourceFilterEnabled`。`audio_source` は AI 拾いのみ（BGM 拒否）。session_loop／first_audio 非流用。
- 実機 `sess_phase11_subj_20260814_143226`: 音声・リップ OK。O2 当てフリ維持。turn4 でスミス完了（admin WS）。
- 既定フィルタ名 **`Smith_Effect`**（ASCII）。OBS 側も同名に揃える。

**親判定（2026-08-14）:**
- **Pass。** Keep = O3。OBS 本線（O1–O3）クローズ可。commit `62798bf` / tag `phase-o3-pass`。
- 増殖加速は後続可。次本線=**X1 blink → F1 黒縁**（英語版は X1+F1 後）。

---

## Phase X1: blink 頻度（emo_id 分岐）

**目的:** expression 生成の blink 間隔を emo_id で分岐する。新ファイル不要・最小分岐のみ。

**仕様（正式）:**
| emo_id | `blink_interval_ms` |
| --- | ---: |
| `9_1` / `9_2` | **3000** |
| それ以外 | **5000**（現行 10000 から変更） |

**スコープ:**
1. Live／生成経路で `emo_id` が blink 挿入に届くか **1行確認**（届かない場合は到達点を特定してから分岐）
2. M3: `build_session_expression_timeline_from_chunks.py` または `_insert_blinks` 系に最小分岐
3. 生成 expression で間隔が仕様どおり（数値）
4. 短回帰: 異常な増減なし。主観: 明らかに速すぎ／遅すぎない
5. 図A／口形／OBS／Bライン非破壊

**スコープ外:** F1 黒縁、英語版、B 仕上げ、リップ品質本線化

**Pass 基準:**
- [x] emo_id 到達確認（1行）
- [x] 9_1/9_2→3s、他→5s の生成根拠
- [x] 短回帰＋主観 OK — **生成パス**。Live 主観は未到達
- [x] 親向けサマリーのみ

**子報告要約（2026-08-15）:**
- M3 `build_session_expression_timeline_from_chunks.py`: `--auto_blink` 時 `_insert_blinks`＋emo 分岐。数値 PASS。
- Live: `auto_blink=False`、expr ほぼ t=0 n=1。主観で定期 blink なし（未配線）。idle_silent も同じビルダ経由で未挿入。

**親判定（2026-08-15）:**
- **Pass-with-defer。** Keep = X1 M3 生成分岐。次=**X1b** Live 配線。F1 は X1b 後。

---

## Phase X1b: Live expression に blink 挿入配線

**目的:** Live 経路でも X1 と同じ間隔（9_1/9_2=3000、他=5000）で blink を入れる。待機中（idle_silent）も対象。

**スコープ:**
1. step1 の `_default_expr_chunk`／`_expr_chunk_from_live_emo_events`（または同等到達点）に最小挿入
2. 間隔は X1 と同一。120ms チャンク内の単純 +interval は不可 → **絶対時刻（chunk_start_ms 等）で間引き**
3. idle_silent も同じビルダ経由（待機 6s+ で blink≥1 が主観条件）
4. ログに `source=auto_blink`（または同等）で識別可能に
5. 図A／口形／OBS／B／F1 非破壊。新ファイル不要が原則

**スコープ外:** M0 黒縁、英語版、B 仕上げ、リップ本線化、OBS 改修、M3 生成パスの再設計

**Pass 基準:**
- [x] 待機 6s+ で blink ≥1（ログ根拠）
- [x] 発話中 3s/5s で異常増減なし
- [x] X1 仕様どおりの emo 分岐
- [x] 短回帰・Keep 非破壊・主観 OK

**子報告要約（2026-08-15）:**
- step1: `_insert_live_auto_blinks`＋絶対グリッド。idle_silent も対象。`source=auto_blink`。
- 主観 165005: 待機 blink あり（T1–3）。発話 3s/5s 正常。図A／OBS／B 非破壊。

**親判定（2026-08-15）:**
- **Pass。** Keep = X1b Live 配線。次=**F1** 黒縁（比較レビュー→親 Go→実装）。

---

## Phase F1: M0 FG 黒縁除去（render_core 最小移植）

**目的:** ローカル現行 `render_core.py` を SSOT とし、合格版から **straight-alpha 復元（黒縁対策）差分のみ**を最小移植する。合格版フル置換禁止。

**推奨フロー:**
1. **現行スプライト**で unpremultiply 移植 → edge RGB が黒潰れでない（数値 Pass）
2. **最新スプライト差替**（検証前までに実施可）→ M3.5 合成主観（黒縁・緑白フリンジ・位置）

**手順:** まず現行 vs 合格版の比較レビューのみ → 親 Go 後に実装。

**スコープ:**
- 黒縁差分のみ（warp は現行経路で使う場合のみ）
- edge RGB 数値＋M3.5 合成主観
- 25fps・4ch・位置・view 維持

**禁止:** 合格版フル置換、left7/atlas/disable_fg 等の別機能混入、M0常駐・cache・render loop 破壊、図A／OBS／B 破壊

**Pass 基準:**
- [x] 比較レビューが親承認済み
- [x] 現行スプライトで edge RGB 数値 — **過補正で Revert**（黒潰れは現行で非再現）
- [ ] 最新スプライト差替後の合成主観 — **未着手（不要になった可能性）**
- [x] 位置・view・25fps・4ch 維持（Revert 後）
- [x] 親向けサマリーのみ

**子報告要約（2026-08-15）:**
- unpremultiply → edge 白飽和（sat 全画素）。即 Revert。
- 実装前 Live FG edge は既に黒でない（≈[54–63,130–173,66–75]）。資料の premul≈[7,17,8] は現行 `_blit_bgra` では再現せず。

**親判定（2026-08-15）:**
- **Pass-with-defer（手法 Fail・経路 Keep）。** unpremultiply は現行経路では不適。Keep = 実装前 `render_core`（`_blit_bgra`）。
- **unpremultiply を差替スプライトで再試行しない。**
- 次=**F1s**: 現行スプライトのまま M3.5／VirtualCam 合成主観 1 本。「黒縁がまだ問題か」だけ判定。
  - 目立たない → F1 クローズ。きれいスプライトは後日資産更新。
  - まだ目立つ → きれいスプライト差替 A/B（資産仮説）。render_core 再改修は原因再定義後。

---

## Phase F1s: 現行スプライトで合成主観（実装なし）

**目的:** 現行 `_blit_bgra`＋現行スプライトで、合成時の黒縁が運用上まだ問題かを主観 1 本で決める。コード変更なし。

**Pass 分岐:**
- [x] **A:** 黒縁目立たない → F1 クローズ
- [ ] B: まだ目立つ → きれいスプライト差替（今回は未達）

**子／親主観要約（2026-08-16）:**
- 親 SSOT: `sess_phase11_subj_20260816_141008`（緑バック）。黒縁／白／緑とも運用上目立たず。
- 子オフライン拡大の緑フリンジは運用距離では問題なし。口後半軽止まりは F1s 対象外。

**親判定（2026-08-16）:**
- **F1s Pass（A）。** F1 ラインクローズ。Keep = `_blit_bgra`＋現行スプライト。
- 出さない: render_core 再改修、unpremultiply 再試行、必須差替 A/B。
- きれいスプライトは後日任意。英語版本線は下節 **EN-RT**（2026-08-17 着手）。

---

## 新ライン: 英語版 Realtime（EN-RT・2026-08-17〜）

> **進捗ファイル方針:** 別ファイルは作らない。本節＋下表で EN ラインを管理する（親のみ編集。子は直接編集しない）。
> JP 品質凍結（Phase0–33／freeze）および R/I/B/O/X/F とは **番号空間を分離**（`EN-RT0`, `EN-RT1`, `EN-RT2`, `EN-LIVE1`, `EN-DUR1`, `EN-DUR2`, `Z1`）。
> **計画 SSOT:** `C:\Users\john\Desktop\EN_m3\english_m3_local_integration_precheck.md`（親は計画を作り直さない）。Phase2 オフライン口形パイプライン詳細は `english_mouth_pipeline_phase2.md`。
> JP リポは **現状維持（サブ運用）**。英語は **別リポをメイン運用**。同一リポ `if language==ja/en` は採用しない。

### 前提（実装 SSOT・Keep）

| 項目 | 内容 |
| --- | --- |
| M1 | 同一リポ `C:\dev\M1_LLM_To_M2_TTS_united`（`feature/local-vad-restore`）。JP コマンド同形＋ `--m3_repo_root` / `--m0_repo_root` 切替のみ |
| M3 JP（触らない） | `C:\dev\M3_Live_API_1_united` |
| M3 EN（本線） | `C:\dev\M3_Live_API_1_english`（EN-RT0 で作成） |
| M0 JP（触らない） | `C:\dev\M0_session_renderer_final_1` |
| M0 EN（本線） | `C:\dev\M0_session_renderer_final_1_english`（EN-RT0 で骨格。9 mouth スプライトは EN tree 作成後・EN-RT1 前） |
| Sync | `audio_ms` / `played_audio_ms` が唯一の時刻 ID。変更禁止 |
| 図A / 方式2 | Keep。音声先行 enqueue・口形捨て・ジッタ延長隠し禁止 |
| 運用ベース | N=2・`--no-fast_inmemory`・B/O/X1 Keep。B/O 本線の不用意再開禁止 |
| コード内パス | 新規ハードコード絶対パスを増やさない。EN リポ内は JP と同様の相対構造 |
| CTC | **当面やらない**（資料 EN-RT3。必要時のみ） |

### EN フェーズ一覧

| Phase | 名称 | 状態 | Pass 日 |
| --- | --- | --- | --- |
| EN-RT0 | リポ骨格＋成果物配置＋パス確認 | `pass` | 2026-08-18（Pass-with-defer → EN-RT0b） |
| EN-RT0b | atlas.en.json 配置＋ mouth_ch.png ファイル名 SSOT | `pass` | 2026-08-18 |
| EN-RT1 | English kNN → 9 mouth → M0（Realtime） | `pass` | 2026-08-18（Pass-with-defer: Live主観→EN-LIVE1、id6 GT→EN-DB1） |
| EN-RT2 | Primary Stress Event → kNN 未送信 frame → 160ms Hold | `pass` | 2026-08-18 |
| EN-LIVE1 | 英語 Live 主観 | `pass` | 2026-08-19（Fail→hf 再走 Pass） |
| EN-LIVE1hf | EN kNN ファイル経路 load | `pass` | 2026-08-19 |
| EN-DUR1 | 英語 1ターン長文ストレス（口頭） | `hold` | 2026-08-20（図A短ターンOK・長文未達） |
| EN-DUR1b | 英語 1ターン無人長尺（trigger 例外） | `hold` | 2026-08-20（trigger到達・90s未達） |
| EN-DUR1c | DUR専用 prompt_dir＋idle 0 無人長尺 | `pass` | 2026-08-20（Pass-with-defer。手元長文・無人PCM未着） |
| EN-DUR2 | 英語 多ターン（12t。4t 延長） | `pass` | 2026-08-20 |
| Z1 | Zoom 配信検証（JP + EN） | `pass` | 2026-08-21（片方向。再開しない） |
| Z2 | Zoom 受信→M1 mic（OS 分離ミックス） | `hold` | 2026-08-22（受信 virtual mic 無し。Fail ではない） |
| Z2b | Voicemeeter Banana 導入＋JP スマホ確認 | `pass` | 2026-08-22 |
| （backlog） | EN-DB1 knn GT 穴（id6=0 / id4=1） | — | Hold の前提ではない。DB 再作成時 |
| （backlog） | angry `leftdown` / `rightup` 欠 | — | 資産追加時は `_` なし。今は捏造しない |
| （backlog） | EN 顔上下動時 BGV↔M0 ズレ | — | B 残差の EN 再発。EN-DUR2 T6 / Z1 でも観察。今は再開しない |
| （backlog） | EN 本番システムプロンプト運用 | — | 方針のみ確定（下節）。**今は prompts_en / JP prompts 非編集** |
| EN-RT3 | CTC Alignment（Wav2Vec2 等） | `deferred` | 当面やらない |

### 日本語併用（切替のみ）

```text
JP: --m3_repo_root C:\dev\M3_Live_API_1_united  --m0_repo_root C:\dev\M0_session_renderer_final_1
EN: --m3_repo_root C:\dev\M3_Live_API_1_english --m0_repo_root C:\dev\M0_session_renderer_final_1_english
    --stream_mouth_gt_glob data/knn_db/en_10files.phoneme_gt.f1f2.json
EN Live:
  --prompt_dir C:\dev\M1_LLM_To_M2_TTS_united\configs\prompts_en
  --output_audio_transcription
pose/bg は当面 JP/M3.5 絶対パスのまま（口スプライトだけ M0 EN）。
```

Live API 自体は共通。切替は repo root・gt_glob・（Live時）prompt_dir。JP 既定 `configs/prompts` は上書きしない。

### Desktop 成果物ギャップ（親確認・2026-08-17）

ルート: `C:\Users\john\Desktop\EN_m3`

| 項目 | 判定 |
| --- | --- |
| `mouth_schema.py` / `knn_predictor.py` | **Desktop 直下に存在**（当初「未同梱」懸念は解消。正規先は `src/m3p/live/`） |
| `english-m3-phase2-snapshot.bundle` | **検出済**（子・`C:\Users\john\Desktop\EN_m3\english-m3-phase2-snapshot.bundle`、約20.5MB）。EN-RT0 は未展開で正（配置リスト外） |
| 9 mouth PNG | `EN_m3` 未検出は正しい。一次ソース `C:\Users\john\Desktop\EN_img9` → M0 EN `assets/` 構造維持コピー済（+576、合計774） |
| スプライトファイル名 | **PNG SSOT = `mouth_ch.png`**（親追記 2026-08-18）。`mouth_sh.png` は今後使わない。論理 key / mouth_id 6 は `mouth_sh` 維持（knn_db 互換）。atlas は `mouth_sh`→`mouth_ch.png` |
| スプライト view 名 | **`_` なしに統一**（angry も leftdown 等。ユーザーが EN_img9 と M0 EN を手動正規化済）。欠: angry `leftdown`（用意漏れ）と `rightup`。追加時も `_` なし |
| `atlas.en.json` | front 最小＋欠 view は同 mouth の front fallback（EN-RT1）。他 view フォルダは実在。全 view 展開は任意 |

正規相対配置（資料準拠・EN リポ内）:

```text
src/m3p/live/mouth_schema.py
src/m3p/live/knn_predictor.py
src/m3p/live/primary_stress_text_events.py
tools/run_english_mouth_pipeline.py
tools/knn_from_formant_raw_to_mouth_timeline.py
tools/apply_g2p_override_to_mouth_id_timeline_stress_hold.py
tools/apply_silence_gate_to_m0_mouth_timeline.py
tools/convert_mouth_id_to_m0_mouth.py
tools/offset_mouth_timeline.py
tools/smooth_mouth_timeline.py
data/knn_db/en_10files.phoneme_gt.f1f2.json
data/knn_db/en_1.phoneme_gt.f1f2.json … en_10.…
data/phonemes/1.phonemes.json … 10.…
data/formant_raw/1.formant.raw.json … 10.…
```

M0 EN（資料準拠）:

```text
configs/smoke_pose_english_front.yaml
timelines/mouth/en/mouth_current_en.json
```

---

## Phase EN-RT0: リポ骨格＋成果物配置＋パス確認

**目的:** 英語メイン運用の箱を作る。Realtime kNN 配線・Hold・M1 ロジック変更はしない。

**設計決定:**
- JP `M3_Live_API_1_united` / `M0_session_renderer_final_1` / M1 作業ツリーへ EN 成果物を混入しない
- M3 EN は JP M3 をベースに **`.git` 除外コピー** → 新規 `git init`（JP origin を共有しない）。bundle があれば補助、無くても Desktop ファイルで可
- EN リポ内パスは JP と同じ相対構造。新規絶対パス・`if language` 切替は禁止
- 9 mouth スプライト一次ソースは `C:\Users\john\Desktop\EN_img9`。M0 EN `assets/` へ **構造維持コピー**（子実施・手作業不要）。口形ファイル名は `mouth_schema` と突合し、不一致は捏造せず欠落表へ。JP 6 mouth 流用禁止

**Pass 基準:**
- [x] `C:\dev\M3_Live_API_1_english` が存在し、上表の正規相対パスに成果物が載っている
- [x] `C:\dev\M0_session_renderer_final_1_english` 骨格あり（yaml / mouth_current_en を正規名で配置）
- [x] JP M3/M0/M1 に EN ファイル追加なし（確認方法をサマリーに明記）
- [x] `mouth_schema.py` が `src/m3p/live/` から import できる（9 id = 0–8）
- [x] 教師 DB `data/knn_db/en_10files.phoneme_gt.f1f2.json` が存在する
- [x] `(8)` `(1)` `(3)` が正規名に整理済み
- [x] `EN_img9` → M0 EN `assets/` 構造維持コピー済み（件数・代表パス・mouth_schema 突合をサマリー）
- [x] M1 / session_loop / 図A / 方式2 を変更していない
- [x] 親向けサマリーのみ（diff/ログ全文なし）

**子報告要約（2026-08-18）:**
- M3 EN initial commit `2f961bc`（孤立 git）。M0 EN は `.git` なし。
- PNG: コピー前198（JP sprites 残）→ 後774（+576）。代表 `assets/normal/front/mouth_o.png`。M3 EN PNG=0。
- 突合時: `mouth_sh.png` は `normal/front` のみ1、他63は `mouth_ch.png`。親追記で **ファイル名は mouth_ch に統一済**（`mouth_sh.png` 不使用）。
- bundle 検出・未展開は正。`atlas.en.json` 当時未配置。`<expr>/<view>/` 維持（JP の `<expr>_<view>` 未平坦化は正）。

**親判定（2026-08-18）:**
- **Pass-with-defer。** 箱・配置・JP非混入は達成。Keep = 別リポ＋相対構造＋PNG構造維持。
- defer → **EN-RT0b:** Desktop `atlas.en.json` を M0 EN へ配置。PNG SSOT=`mouth_ch.png`。論理 key `mouth_sh`（id 6）は維持。
- 出さない: 平坦化、bundle 展開、Realtime kNN、knn_db の mouth_key 一括リネーム、JP 混入。
- M0 EN に git なしは残メモ（EN-RT0 Fail にしない）。

---

## Phase EN-RT0b: atlas.en.json 配置＋ mouth_ch.png ファイル名 SSOT

**目的:** EN-RT0 の defer を閉じ、EN-RT1 の M0 参照がファイルとして解決できるようにする。Realtime 配線はしない。

**設計決定:**
- `C:\Users\john\Desktop\atlas.en.json` → `C:\dev\M0_session_renderer_final_1_english\assets\atlas.en.json`
- **PNG ファイル名 SSOT = `mouth_ch.png`。`mouth_sh.png` は使わない。**
- **論理 key / mouth_id 6 は `mouth_sh` のまま**（`convert_mouth_id_to_m0_mouth` と knn_db の `mouth_key` を一括変更しない）
- `mouth_schema.sprite_name` と atlas のパス末尾だけ `mouth_ch.png` にする。atlas キー `mouth_sh` は残し、同じ PNG を指す。`mouth_ch` キーを足してよい
- Desktop atlas は `front/mouth_*.png`。実体は `assets/<expr>/<view>/`。**平坦化禁止。** atlas パスが 0 件ヒットなら `normal/front/` 接頭辞の最小修正のみ（EN コピー。Desktop 原本は触らなくてよい）
- JP M0/M3/M1 書き込み禁止。M3 EN に PNG を置かない

**Pass 基準:**
- [x] M0 EN `assets/atlas.en.json` が存在し、yaml の `atlas.atlas_json` と一致
- [x] M0 EN に `mouth_sh.png` が 0。`mouth_ch.png` が view 横断で存在する
- [x] `mouth_schema` の id6 `sprite_name` が `mouth_ch.png`。import で id 0–8 維持
- [x] atlas 上の mouth パスが、少なくとも `normal/front` の 9 枚に Test-Path で当たる（sh ではなく ch）
- [x] knn_db の `mouth_key` は未一括リネーム
- [x] 構造維持（`<expr>/<view>/`）。Realtime / 図A / 方式2 / M1 未変更
- [x] JP 非混入。親向けサマリーのみ

**子報告要約（2026-08-18）:**
- atlas コピー後 `front/` は 0 件 → EN コピーのみ `normal/front/` 接頭辞。Desktop 原本未編集。
- `mouth_sh.png`=0 / `mouth_ch.png`=64（`normal/front` の1枚をリネーム、捏造なし）。
- schema id6 `(mouth_sh, mouth_ch.png)`。atlas 9/9 HIT。`mouth_ch` キーも HIT。
- 残差: `{expr}_{view}` vs `<expr>/<view>/`、angry の `left_down` vs 他 `leftdown`、`rightup` 欠、atlas は front のみ。

**親判定（2026-08-18）:**
- **Pass。** ファイル名 SSOT と atlas front 接続は閉じた。Keep = `mouth_ch.png`＋論理 key `mouth_sh`＋`<expr>/<view>/`。
- 次=**EN-RT1**。平坦化しない。renderer/atlas 側でパス解決。Stress Hold / CTC は対象外。

---

## Phase EN-RT1: English kNN → 9 mouth → M0（Realtime）

**目的:** 資料どおり Realtime kNN を主経路にする。English 9 mouth で chunk 生成し M0 EN が描けること。Primary Stress Hold は EN-RT2。CTC はやらない。

**設計決定:**
- 経路: Formant → Realtime kNN（`k=9`、DB=`data/knn_db/en_10files.phoneme_gt.f1f2.json` のみ）→ mouth_id 0–8 → 120ms chunk → M0 EN
- M1 起動は JP と同形＋ `--m3_repo_root` / `--m0_repo_root` を EN へ。`if language` 禁止
- session_loop の `data/knn_db/*.f1f2.json` は EN では JP 残骸＋ `en_1..10` とマージ DB が混ざる。**EN 起動は `en_10files` 単独**（CLI 追加は可。JP default は変えない）
- M1 は EN knn を `mod._load_knn_db` で読む。EN ツール側にエイリアスを足してよい（M1 の図Aは触らない）
- スプライトは **平坦化しない**。M0 EN のみ `<expr>/<view>/` を解決。欠 view は atlas fallback（front）。`left_down` vs `leftdown`、`rightup` 欠は捏造せず欠落表
- atlas は front 最小で Pass 可。全 view 展開は必須にしない
- PNG SSOT=`mouth_ch.png`。論理 key は `mouth_sh`
- 運用 Keep: 方式2、図A、N=2、`--no-fast_inmemory`、audio_ms、音声先行 enqueue 禁止
- JP M3/M0 と M1 の JP 経路は壊さない。EN 専用変更に閉じる

**Pass 基準:**
- [ ] EN kNN が mouth_id 0–8 を出す（`en_10files`、k=9）。JP 6 mouth DB を混在させていない
- [ ] M0 EN が 9 mouth PNG を `<expr>/<view>/` から読める（少なくとも normal/front）
- [ ] 図A維持: KNN→M0完了→enqueue。AUDIO_BEFORE_M0=0
- [ ] Live 1 ターン、または同等の realtime chunk 経路（親は Live 1 ターン推奨）
- [ ] Stress Hold / CTC 未配線
- [ ] JP 非混入。親向けサマリーのみ

**Pass 基準:**
- [x] EN kNN が mouth_id 0–8 を出す（`en_10files`、k=9）。JP 6 mouth DB を混在させていない — **接続済。分布は DB 偏りで 4/6 未出**
- [x] M0 EN が 9 mouth PNG を `<expr>/<view>/` から読ける（少なくとも normal/front）
- [x] 図A維持: KNN→M0完了→enqueue。AUDIO_BEFORE_M0=0 — **enqueue 順未変更として確認**
- [x] Live 1 ターン、または同等の realtime chunk 経路 — **同等経路で確認。Live 主観は defer**
- [x] Stress Hold / CTC 未配線
- [x] JP 非混入。親向けサマリーのみ

**子報告要約（2026-08-18）:**
- DB 219 points / k=9 / EN は `--stream_mouth_gt_glob` で `en_10files` のみ。JP glob default 維持。
- M0 EN は `<expr>/<view>/`。欠 view は同 mouth の front。id6 key=`mouth_sh` → PNG=`mouth_ch.png`。
- 代替検証 mouth_id: `{0:833,1:9,2:61,3:8,5:14,7:94,8:33}` n=1052。4/6 未出。en_10files は id6=0 / id4=1。
- Live 1ターン未実施（この子セッションに mic/VCam/OBS なし）。
- 親追記: angry view 名はユーザーが `_` なしへ正規化済。欠は angry `leftdown` と `rightup`。

**親判定（2026-08-18）:**
- **Pass-with-defer。** Realtime kNN→9 mouth→M0 EN は接続。Keep = en_10files / k=9 / `<expr>/<view>/` / JP default glob 維持 / `--stream_mouth_gt_glob`。
- defer の整理（下表）。**次本線=EN-RT2**（Hold）。Live 主観は EN-LIVE1。id6 GT は EN-RT2 の Gate にしない（Hold は primary stress 母音）。

| 残件 | 扱い |
| --- | --- |
| angry view `_` なし正規化 | **済（ユーザー手動）。Keep。** 子作業なし |
| angry `leftdown` / `rightup` 欠 | backlog。資産追加まで捏造しない |
| atlas 非 front 展開 | 任意。fallback Keep |
| en_10files id6=0 / id4=1 | **EN-DB1 backlog。** RT2 非Gate |
| Live 主観 | **EN-LIVE1。** 先に `--prompt_dir configs/prompts_en`（同名 txt）。JP `configs/prompts` は触らない。親承認後 |

**EN プロンプト切替（親確定・Live 前必須）:**
- 既存 CLI `--prompt_dir` を使う。`if language` も JP ファイル上書きも禁止
- JP 既定: `<m1>/configs/prompts`
- EN: `<m1>/configs/prompts_en` に **同名ファイル一式** を新規作成し、起動時だけ `--prompt_dir` を向ける
- EN-RT2 では prompt 作業をしない

---

## Phase EN-RT2: Primary Stress Event → 未送信 kNN frame → 160ms Hold

**目的:** 資料 EN-RT2。G2P は主経路にしない。Realtime kNN の結果に対し、**未送信 frame だけ** Primary Stress 母音を 160ms Hold する。

**設計決定:**
- 経路: Transcript → g2p_en → Primary Stress Event → 未送信 kNN frame → 160ms Hold → chunk / M0
- 既存 `src/m3p/live/primary_stress_text_events.py` を使う。word onset は OFF。`min_hold_ms=160`
- revision は検知のみ。既送信 chunk の取消しはしない
- Hold は **enqueue 前の未送信 mouth frame のみ**。図A（KNN→M0→enqueue）と音声先行 enqueue 禁止は維持
- id6 GT 空は本 Phase の Fail 理由にしない（Hold 対象は primary stress 母音）
- JP プロンプト / `--prompt_dir` / Live 主観は対象外。CTC 禁止
- `if language` 禁止。JP M3/M0 書き込み禁止

**Pass 基準:**
- [x] Primary Stress Event が incremental transcript から出る
- [x] 未送信 kNN frame に 160ms Hold が乗る（既送信は不変）
- [x] 図A / AUDIO_BEFORE_M0=0 / 方式2 維持
- [x] 親向けサマリーのみ

**子報告要約（2026-08-18）:**
- Hold は `_process_audio_chunk_knn_sync` 末尾（KNN 後、M0/enqueue 前）。JP は hold モジュール無し → no-op。
- sent = 当該 chunk push 直前の `pipeline_audio_end_ms`。未送信 = `t_ms >= sent_ms`。VAD 非活性 / mouth_id=0 は Hold しない。revision は検知のみ。
- オフライン: 母音 run 40ms→160ms。`t_ms < sent_ms` 不変。windows=6 / applied=6 / lengthened=4。
- Live 未実施。`--output_audio_transcription` 既定 OFF。runtime に g2p-en + nltk。

**親判定（2026-08-18）:**
- **Pass。** Keep = 未送信のみ 160ms Hold、図A順（KNN→Hold→M0→enqueue）、JP no-op。
- **Live 主観はまだ不要。** EN-LIVE1 は prompt 切替と親承認のあと。CTC / EN-DB1 / 欠 view はやらない。

**EN-LIVE1 前提（親承認済み・2026-08-18 発行）:**
- `configs/prompts_en/` 同名 txt ＋ `--prompt_dir`（JP `configs/prompts` 非上書き）
- EN 起動に `--output_audio_transcription`
- g2p-en + nltk（M1 `.venv` は子が投入済）
- transcript 遅れは仕様（既送信は戻せない）
- pose/bg は JP/M3.5 資産のまま可

---

## Phase EN-LIVE1: 英語 Live 主観

**目的:** JP 実働主観コマンドを差分だけで英語 Live に通し、口形・Hold が破綻しないことを見る。実装本線（RT）の新規設計はしない。

**設計決定:**
- 起動は JP 確認済みコマンドがベース。作り直さない
- 必須差分: m3/m0 EN、`--prompt_dir` EN、`--output_audio_transcription`、`--stream_mouth_gt_glob` = `en_10files` のみ
- `configs/prompts_en` は同名 txt の新規。JP `configs/prompts` は無断上書き禁止。`if language` 禁止
- pose/bg/m35 は **JP 絶対パスのまま**（口スプライトは `--m0_repo_root` EN）
- Keep: 方式2 VAD、N=2、`--no-fast_inmemory`、jitter 300/240、talkover/event/OBS 系ファイル、mic/audio device
- スコープ外: CTC、EN-DB1、欠 view、B/O 再開、ジッタ延長

**Pass 基準:**
- [x] `configs/prompts_en` が同名で存在し、JP `configs/prompts` が未変更
- [x] 英語で音声返答する（4 ターン想定、主観） — hf 再走で Pass
- [x] 口形が変化する（固着しない） — 短いフリーズは観察（Gate にしない）
- [x] Hold でクラッシュ／図A破壊なし。AUDIO_BEFORE_M0 悪化なし
- [x] JP 経路を壊す変更なし
- [x] 親向けサマリーのみ

**子報告要約（2026-08-18）:**
- SESSION `sess_en_live1_subj_20260818_222706`。transcription は英語 4 ターン。player/OBS は無音。mouth/M0 未生成。
- AUDIO_BEFORE_M0=0 だが emitted=0 / PLAYING=0（enqueue なし）。
- Blocker: Live KNN が `ModuleNotFoundError: m3p.live.knn_predictor` を 475 回。session_loop が先に JP `m3p` を bind し、EN knn スクリプトの import が JP パッケージを見る。

**親判定（2026-08-18）:**
- **Fail。** prompts_en と起動差分は Keep。KNN import を **EN-LIVE1hf** で直す。
- 採用: Hold と同様の **ファイル経路 load**（`knn_predictor` / 必要なら `mouth_schema`）。`sys.path` 追加だけでは既 import の JP `m3p` を解けない。
- 出さない: JP M3 へ EN ファイル混入、`if language`、PYTHONPATH で `m3p` 全体差替（MouthStreamerOC が壊れる）、ジッタ延長、CTC。

**親再判定（2026-08-19）:**
- **Pass。** Blocker は EN-LIVE1hf で解消。SESSION `sess_en_live1_subj_20260819_140651`。
- Keep: prompts_en / en_10files / EN m3・m0 切替 / `_preload_knn_script_live_deps`
- 観察（非Gate）: ターン後半の短い口フリーズ（turn2 tail の M0 lock/png_wait 候補）。トーク被りは応答が速いため。ジッタ延長しない。

---

---

## Phase EN-LIVE1hf: EN kNN をファイル経路 load

**目的:** EN knn スクリプトが JP `m3p` に吸い込まれないようにする。主観は hotfix 後に同じ EN コマンドで再走。

**設計決定:**
- 挿入点: `_load_knn_runtime_module` の `exec_module` 前
- `knn_script` の M3 root に `src/m3p/live/knn_predictor.py` があれば、Hold と同じ `_load_module_from_path` で `mouth_schema` → `knn_predictor` を `sys.modules` に載せてから knn スクリプトを exec
- ファイルが無い JP M3 は no-op（現行どおり）
- 図A・gt_glob・prompts・jitter は触らない

**Pass 基準:**
- [x] EN Live で `knn_predictor` ModuleNotFoundError が 0
- [x] chunks_n / mouth が生成される
- [x] JP knn 経路を壊す変更なし（ファイル無しなら preload しない）
- [x] 同じ EN-LIVE1 コマンドで再走。親向けサマリー＋主観

**子報告要約（2026-08-19）:**
- `_preload_knn_script_live_deps` を `_load_knn_runtime_module` の exec 直前。Hold と同じ `_load_module_from_path`。JP は no-op。
- ModuleNotFoundError=0。`[knn][preload]` 1 回。chunks_n=262。knn_done=enqueue_done=m0_done=483。AUDIO_BEFORE_M0=0。
- Hold applied>0 が 21。MouthStreamerOC は JP bind のまま。

**親判定（2026-08-19）:**
- **Pass。** Keep = `_preload_knn_script_live_deps`。EN-LIVE1 を Pass に戻す。
- 英語実装本線（RT+LIVE1）はクローズ。**次本線=検証ライン EN-DUR1 → EN-DUR2 → Z1**（2026-08-20）。CTC / EN-DB1 / 欠 view は出さない。push/merge は本線外（ユーザー判断）。

---

## 新ライン: 英語検証／配信（EN-DUR / Z・2026-08-20〜）

> 実装本線（EN-RT / EN-LIVE1）はクローズ。以降は **運用確認**（長尺→多ターン→Zoom）。図A非破壊。品質 Phase の再開ではない。
> コマンドは **一から作らない**。EN-DUR は EN-LIVE1 4t をベース。JP 長尺の synth / `response_trigger` / `silence 600` は盲目コピー禁止。
> Zoom は別 Phase。着手時にユーザーが旧コマンドを添付する。

### コマンド差分（親固定・EN-DUR1）

ベース = EN-LIVE1 4t（Pass）。JP 長尺は「1ターン長文」の意図だけ借りる。

| 項目 | EN-LIVE1 4t | JP 長尺 | EN-DUR1 |
| --- | --- | --- | --- |
| `--m3_repo_root` / `--m0_repo_root` | EN | JP | **EN Keep** |
| `--prompt_dir` | `configs/prompts_en` | `configs/prompts` | **prompts_en Keep**（JP prompts 非上書き） |
| `--output_audio_transcription` | ON | ON | Keep |
| `--stream_mouth_gt_glob` | `en_10files` | （なし） | **en_10files Keep** |
| `--turns` | 4 | 1 | **1** |
| `--gap_s` | 1.0 | 0.5 | 1.0 Keep（1t では実質無影響） |
| `--turn_idle_wait_s` | 2.5 | 3.0 | 2.5 Keep |
| `--turn_first_audio_timeout_s` | 60 | 90 | **90**（長文生成待ち。ジッタではない） |
| `--mic_send_max_s` | 25 | 18 | **25 Keep**（EN 運用。JP 18 に下げない） |
| `--mic_vad_silence_ms` | 350 | 600 | **350 Keep**（R クローズ後の通常。600 に戻さない） |
| synth cable | なし | あり | **なし**（実 mic。EN-LIVE1 と同じ） |
| `--response_trigger` | なし | JP 長文 txt | **なし**（盲目コピー禁止。オペレーターが英語で長返答を依頼） |
| `--audio_priority_mode` / `--no-knn_inmemory` / `--skip_archive_pcm` / `--knn_incremental` / `--m0_worker_port` / `--mic_vad_debug` | なし | あり | **なし** |
| 出力デバイス CLI | `--audio_device 19` | `--ai_audio_output_device 19` | **`--audio_device 19` Keep** |
| N=2 / `--no-fast_inmemory` / jitter 300/240 | あり | あり | Keep |
| battle / event / bg_override ファイル | あり | なし | **EN-LIVE1 Keep** |
| pose / bg / m35 | JP 絶対パス | 同 | Keep（口スプライトだけ M0 EN） |

EN-DUR2（予約）: 同じ EN-LIVE1 4t を `--turns 12` へ。必要なら `gap_s` / `turn_idle_wait_s` のみ。通常パスへ trigger を残さない。
Z1（予約）: JP と EN の両方。コマンドはユーザー添付後に親が固定。今は着手しない。

---

## Phase EN-DUR1: 英語 1ターン長文ストレス

**目的:** EN-LIVE1（短 4t）の延長として、**1 ターン長文**で口・供給・図Aが破綻しないことを主観＋ログ要約で確認する。新規設計・品質チューニングはしない。

**設計決定:**
- 起動は EN-LIVE1 4t コマンドがベース。上表の差分以外は変えない
- JP 長尺の synth / trigger / silence 600 はコピーしない
- オペレーターが英語で「長く話して」と依頼する（テキストトリガ新設禁止）
- Keep: 方式2、図A、N=2、`--no-fast_inmemory`、audio_ms SSOT、jitter 300/240、silence 350
- 短い口フリーズ（EN-LIVE1 観察）は Gate にしない。ジッタ延長で隠さない
- スコープ外: EN-DUR2、Z1、CTC、EN-DB1、欠 view、B/O 再開、本番システムプロンプト、push/merge
- 実装は **Blocker があるときだけ**親へ報告。勝手に直さない（LIVE1hf 型は親承認後）

**Pass 基準:**
- [ ] 1 ターンで英語の長め返答が最後まで鳴る（主観）
- [ ] 口形が変化する（全程固着しない）。短い局所フリーズは観察メモ可
- [ ] 図A非破壊: AUDIO_BEFORE_M0=0、enqueue 到着順、通常 `clear_queue` 0（割り込み例外以外）
- [ ] 方式2 / N=2 / `--no-fast_inmemory` / jitter 300/240 / silence 350 維持
- [ ] JP M3/M0 / `configs/prompts` 非破壊。親向けサマリーのみ（diff/ログ全文なし）

**Fail:** クラッシュ、無音、口が全程固着、AUDIO_BEFORE_M0>0、JP 経路破壊、ジッタ延長や口形捨てで症状隠し。

**子報告（2026-08-20）:**
- 142252 / 143157。AI 音声・口変化あり。AUDIO_BEFORE_M0=0、clear 実体 0、ModuleNotFound 0。Keep 維持。コード差分なし。
- 長文未達。transcription は短文（世間話／tuna 2文途中切れ）。
- 主因候補: (1) `prompts_en/00_base_system.txt` が "Keep replies short, about one sentence."（JP prompts も1文。未編集） (2) `turn_idle_wait_s=2.5` 対 player pending（tail 未 drain でプロセス終了しうる）。図A破壊ではない。
- 運用メモ: EN 主観はオペレーター手元起動を推奨。

**親判定（2026-08-20）:**
- **Hold。** 短ターンとしての口・供給・図Aは動く。1ターン長文ストレスは未達。EN-DUR2 に進まない。
- 口頭「長く」だけでは `prompts_en` の1文制約を越えられない。`prompts_en` 本番は触らない（B 不採用。本番プロンプト運用は backlog）。
- ユーザー希望 = **無人長尺のあと手元主観**。次 = **EN-DUR1b**（C+D の DUR 専用例外）。通常ターンの trigger 本線復活はしない。

---

## Phase EN-DUR1b: 英語 1ターン無人長尺（DUR 専用 trigger 例外）

**目的:** EN-DUR1 の長文未達を、**無人 1 ターン長尺**で再現し、その後オペレーター手元主観も取る。図A非破壊。品質チューニング・prompts_en 本番変更はしない。

**設計決定（親 Go・C+D）:**
- 通常パスは `--skip_response_trigger` default True のまま。argparse 既定を変えない
- DUR 起動だけ `--no-skip_response_trigger` + **EN 専用** 英語長文 txt（新ファイル。JP `phase10_stress_long_trigger.txt` を使わない／日本語で話せと書かない）
- 無人: 既存 `tools/phase1_play_synth_speech_to_cable.py` で VAD 窓に入れる。**方式2は維持**（synth は activity 用。server VAD 復活禁止）
- `turn_idle_wait_s` を **20**（再生 tail drain 待ち。ジッタ 300/240 は変えない）
- 入れない: silence 600、`audio_priority_mode`、`--no-knn_inmemory`、`--skip_archive_pcm`、`--knn_incremental`、`--m0_worker_port`、`--mic_vad_debug`
- `prompts_en` / JP `configs/prompts` / JP M3/M0 は触らない
- session_loop 本線ロジック変更なし（既存 CLI のみ）

### コマンド差分（親固定・EN-DUR1b）

ベース = EN-DUR1 コマンド。追加例外だけ。

| 項目 | EN-DUR1 | EN-DUR1b |
| --- | --- | --- |
| `--turns` | 1 | 1 Keep |
| `--mic_vad_silence_ms` | 350 | **350 Keep** |
| jitter | 300/240 | Keep |
| `--turn_idle_wait_s` | 2.5 | **20**（drain。ジッタではない） |
| `--response_trigger` | なし | **DUR 起動のみ** `tools/en_dur1_stress_long_trigger.txt` |
| `--skip_response_trigger` | default True | この起動だけ `--no-skip_response_trigger` |
| synth | なし | **無人ランのみ**。delay_s=4 / voice_s=2.5 / tail_silence_s=1.0 |
| 無人 mic / synth device | mic=1（実マイク） | **ケーブル組**。案: synth `--device 6` → `--mic_input_device 4`（JP 長尺と同じ組）。EN-LIVE1 の mic 1 は手元主観用。違うならユーザーが番号置換 |
| 手元主観 | — | synth なし、`--mic_input_device 1`、trigger は残す（prompts_en 1文のまま口頭だけでは長文にならない） |

**手順:** (1) 無人ラン → ログ要約 (2) 同じ trigger で手元主観（synth なし）。EN-DUR2 にはまだ進まない。

**Pass 基準:**
- [ ] 無人 1 ターンで英語が **短文で終わらず** 長く続く（目安: 数十秒以上。90s 未達は観察可、2〜3言で終わりは Fail）
- [ ] 口形が変化する。短い局所フリーズは観察メモ（Gate にしない）
- [ ] 最後まで鳴る／プロセスが tail 前に殺して切らない（`pending_ms` 対 idle 20 を要約）
- [ ] AUDIO_BEFORE_M0=0、通常 clear 0、方式2 / N=2 / no-fast_inmemory / jitter 300/240 / silence 350
- [ ] default skip_trigger 未変更。JP prompts / prompts_en 未変更
- [ ] 手元主観メモ（ユーザー実施。子はコマンドを渡す）
- [ ] 親向けサマリーのみ

**Fail:** 短文のまま、無音、図A破壊、既定 skip を False に恒久化、silence 600 混入、prompts_en 上書き。

**子報告（2026-08-20）:**
- 無人 `sess_en_dur1b_unatt_20260820_145056`: Fail。synth delay 4 より先に I1 `idle_utterance_s=3` が発火。`first_turn_prime` が JP idle 短文を text 送信。`--no-skip` でも `already_fired_in_mic` で EN 長尺 trigger 未送。mic `has_spoken=False`。transcription 2文数秒。AUDIO_BEFORE_M0=0。
- 手元 `sess_en_dur1b_subj_20260820_145614`: trigger 送信。体感 4–5文。モデルが「short のみ許可」と明言。90s なし。queued≈25s / [OK] played≈15s pending≈10s。AUDIO_BEFORE_M0=0。図A/口/音声は短〜中ターンとして動く。
- コード: `tools/en_dur1_stress_long_trigger.txt` 新規のみ。session_loop / argparse skip 既定 / prompts_en 未変更。
- 観察（本線外）: EN 化後の BGV↔M0 上下ズレが JP より大きい気がする → 既存 backlog。DUR では触らない。

**親判定（2026-08-20）:**
- **Hold。** trigger 例外パスは手元で到達。無人は I1 競合で未到達。90s Gate は `prompts_en` 一文制約が勝つ。EN-DUR2 に進まない。
- 本番 `prompts_en` は触らない（backlog「EN 本番システムプロンプト」）。I1 default 3.0 も変えない。
- 次=**EN-DUR1c**: DUR 専用 `configs/prompts_en_dur`（コピー＋長さ行のみ）＋無人は `--idle_utterance_s 0`（既存 CLI。I1 Keep 破壊ではない）。

---

## Phase EN-DUR1c: DUR専用 prompt_dir＋idle 0 無人長尺

**目的:** 長文ストレスを、本番 `prompts_en` を変えずに通す。I1 競合を無人ランだけ既存 CLI で避ける。図A非破壊。session_loop 本線は触らない。

**設計決定:**
- `configs/prompts_en` は **コピーして** `configs/prompts_en_dur` を新設。同名ファイル一式。**長さ行だけ**緩める（"Keep replies short, about one sentence." を長尺許可に置換）。emo 規則はそのまま
- JP `configs/prompts` と本番 `prompts_en` は非上書き
- `--prompt_dir` を DUR 起動だけ `prompts_en_dur` へ。`if language` 禁止
- 無人: `--idle_utterance_s 0`（既存。`<=0` で I1 無効）。default 3.0 は変えない。synth delay 4 はそのまま可
- trigger 例外は 1b Keep（`--no-skip` + `tools/en_dur1_stress_long_trigger.txt`）
- `--turn_idle_wait_s 45`（90s 相当の drain。ジッタ 300/240 は据え置き）
- 入れない: silence 600、session_loop 改修、I1 既定変更、B/O 再開
- EN BGV↔M0 上下ズレは backlog。本 Phase で触らない

### コマンド差分（親固定・EN-DUR1c）

ベース = EN-DUR1b。追加例外だけ。

| 項目 | EN-DUR1b | EN-DUR1c |
| --- | --- | --- |
| `--prompt_dir` | `prompts_en` | **`prompts_en_dur`（DUR 専用）** |
| `--idle_utterance_s` | default 3 | **無人・手元とも 0**（I1 既定は維持） |
| `--turn_idle_wait_s` | 20 | **45**（drain） |
| trigger / --no-skip | DUR のみ | Keep |
| synth 無人 | delay 4 / device 6 / mic 4 | Keep |
| 手元 mic | 1 | Keep（synth なし） |
| jitter / silence 350 / N=2 | Keep | Keep |

**手順:** (1) `prompts_en_dur` 作成 (2) 無人ラン (3) 手元主観コマンドをユーザーへ渡す。EN-DUR2 にはまだ進まない。

**Pass 基準:**
- [ ] 無人で EN 長尺 trigger が送られる（I1 `already_fired_in_mic` でスキップされない）
- [ ] 英語が短文拒否で終わらず、**数十秒以上**続く（90s は目標。未達でも 20s+ 連続なら親が Pass-with-defer 可）
- [ ] 口形変化。短いフリーズは観察。AUDIO_BEFORE_M0=0、通常 clear 0
- [ ] `prompts_en` / JP prompts / I1 default 3.0 / skip default True 未変更
- [ ] 手元主観メモ（ユーザー実施）
- [ ] 親向けサマリーのみ

**Fail:** また 2〜3言、I1 が先に発火、prompts_en 上書き、図A破壊、ジッタ延長。

**子報告（2026-08-20）:**
- 無人 `sess_en_dur1c_unatt_20260820_151530`: trigger 到達・I1 非発火。実 PCM 0（mic4 `no_speech`）。`first_audio timeout`。再生は idle_silent のみ。AUDIO_BEFORE_M0=0。
- 手元 `sess_en_dur1c_subj_20260820_222002`: 主観 OK（かなり長文・口・音量）。`audio_ms≈31.6s`。AUDIO_BEFORE_M0=0、clear 実体 0。90s トピック完走ではない。`generation_complete` なし（drain 観察。本線非接触）。
- 差分: `configs/prompts_en_dur/` 新規（長さ行のみ）。`prompts_en` / session_loop / I1 default / skip default 未変更。

**親判定（2026-08-20）:**
- **Pass-with-defer。** 長文能力は手元で証明（支配層= system 長さ行）。無人 Fail はケーブル/mic PCM 未着であり、長文欠如ではない。90s 完走は非Gate。
- Keep: `prompts_en_dur` は **DUR 検証用**。通常パスへ残さない。`prompts_en` 一文は DUR 検証 Keep であり、プロダクト最終形ではない。
- defer: 無人ケーブル、90s 完走、`generation_complete` なし、BGV delta。
- 次本線=**EN-DUR2**（12t・本番 `prompts_en`）。prompt 本番化は backlog。

### 本番長さ方針（backlog・今は実装しない）

相談方針を親が採用。session_loop 新スイッチは作らない。今は `prompts_en` / JP prompts を触らない。

| 層 | 内容 |
| --- | --- |
| 長さ SSOT | prompt。将来 `00_base` から1文硬拘束を外す。通常=`20_normal`（1〜2文）。バトル=さえぎられるまで話してよい |
| 起動切替 | 長尺配信だけ `--prompt_dir`（DUR の 60–90s テスト文面は本番に使わない） |
| 主導権 | 床取り（mic mute＋割り込み）。定型「短く話せ」は用途と逆 → 後で直す。単独では一文のまま終わる |
| やらない | `prompts_en` 恒久を 60–90s にする、`--no-skip` 本線化、I1 を長文装置化、案2（00_base 都度編集） |

---

## Phase EN-DUR2: 英語 多ターン（12t）

**目的:** EN-LIVE1 の 4t 運用確認を **12 turns** へ延長する。長尺ストレスではない。図A非破壊。

**設計決定:**
- ベース = EN-LIVE1 4t。差分は `--turns 12` のみ
- `--prompt_dir` = 本番 `configs/prompts_en`（一文のまま。短返信は Fail にしない）
- DUR 例外を残さない: `prompts_en_dur` / `--idle_utterance_s 0` / `--no-skip_response_trigger` / `--response_trigger`
- I1 default 3.0・skip default True・jitter 300/240・silence 350・N=2・`--no-fast_inmemory`
- `gap_s=1.0` / `turn_idle_wait_s=2.5` / `turn_first_audio_timeout_s=60` は LIVE1 Keep
- session_loop 非変更。prompts_en / JP prompts 非編集
- 手元 mic=1。synth なし。子は mic 非占有
- BGV↔M0 / 本番プロンプト運用は backlog

**Pass 基準:**
- [x] 12 ターン完走（またはオペレーターが実施したターン数を明記。途中落ちは Fail）
- [x] 各ターンで英語音声と口変化（短文でよい）
- [x] AUDIO_BEFORE_M0=0、通常 clear 0、方式2 / N=2 / no-fast_inmemory 維持
- [x] DUR 例外フラグなし（prompt_dir=prompts_en、idle 既定、skip True）
- [x] 親向けサマリーのみ。コード差分なしが正

**Fail:** クラッシュ、無音連続、口全程固着、AUDIO_BEFORE_M0>0、DUR 例外の持ち込み、prompts_en 編集。

**子報告（2026-08-20）:**
- SESSION `sess_en_dur2_subj_20260820_231430`。12/12。AUDIO_BEFORE_M0=0、clear 実体 0、timeout 0、pipeline 1479 揃い。hang/mouth_closed/supply_gap=0。
- 主観: 音声・口は大きな問題なし。各ターンは自然な数文。T12 は I1 自発（本番相当）。後半固着・無音なし。
- 観察（非Gate）: T6 BGV↔M0 上下ずれ。返答のかぶり／~2s ばらつき。idle fire 2（T7, T12）。REBUFFERING 実体 35（後半偏りなし）。
- コード差分なし。DUR 例外なし。

**親判定（2026-08-20）:**
- **Pass。** Keep = EN-LIVE1 4t と同形の EN 12t 運用（`prompts_en` / skip True / I1=3 / N=2 / no-fast_inmemory / jitter 300/240 / silence 350）。
- T6 上下ずれ → 既存 backlog。かぶり・遅延ばらつきは Gate にしない。
- 次本線=**Z1**（ユーザーが Zoom 旧コマンドを添付するまで着手しない）。prompt 本番化は backlog。

### DUR 検証成果物（Keep All / commit / tag）

運用 Keep All（session_loop を DUR 例外付きにする）は **不要**。EN-DUR2 はコード差分なし。本番起動は `prompts_en`。

検証ファイル（`configs/prompts_en_dur/`、`tools/en_dur1_stress_long_trigger.txt`）は **任意のローカル commit 可**（再現用。本番 default にはしない）。tag / push は必須ではない。ユーザー判断。

---

## Phase Z1: Zoom 配信検証（JP + EN）

**目的:** JP と EN の両方で、Zoom 相手に AI 音声＋VirtualCam 映像／口が届くこと。コード改造しない。図A非破壊。品質 Phase ではない。

**設計決定:**
- 資料 `Zoomテスト用フルコマンド.md` は **ルーティングのみ採用**。旧 client-VAD フルコマンドは丸ごと復元禁止
- 中心 = VB-CABLE ＋ `--audio_device`。USB mic → session_loop → Live API → AI 音声 → CABLE Input → Zoom
- Zoom: Microphone = CABLE Output / Speaker = CABLE Input。カメラ = **Unity Video Capture**（OBS VirtualCam 決め打ちは誤り。下節 Keep）
- ベース = 現行合格（JP 凍結 Live ／ EN = EN-LIVE1 4t 同形＝EN-DUR2 から `--turns 4`）。核の差分は `--audio_device` のみ
- CABLE Input 番号は `sounddevice.query_devices()` で今取る。旧 14 も今の 19 も決め打ちしない
- `--mic_input_device` は現行どおり実マイク（通常 1）。CABLE にしない
- JP 4t → EN 4t。12t／DUR 長尺は出さない
- 主導権 mute+interrupt は Zoom では使わない（資料どおり。Gate にしない）
- ハウリングは観察メモ（Gate にしない。設定見直しは親）

### コマンド差分（親固定・Z1）

| 項目 | 現行合格 | 旧 Zoom md | Z1 |
| --- | --- | --- | --- |
| `--audio_device` | 19（ローカル再生） | 14（当時 CABLE Input） | **クエリした CABLE Input** |
| `--mic_input_device` | 1（実マイク） | 1 | **1 Keep** |
| `--turns` | EN-DUR2 は 12 | 4 | **4** |
| `--mic_vad_silence_ms` | 350 | 900 | **350 Keep** |
| `--mic_send_max_s` | 25 | 8 | **25 Keep** |
| `--turn_first_audio_timeout_s` | 60 | 12 | **60 Keep** |
| `--gap_s` | 1.0 | 2.0 | **1.0 Keep** |
| skip / trigger | default True | `--response_trigger` | **skip True。trigger 禁止** |
| `drop_initial_audio_ms` | 強制 0 | 40 | **触らない（強制0 Keep）** |
| `prompts_en_dur` / idle 0 | 使わない | — | **使わない** |
| JP m3/m0 | united / JP M0 | united | **JP Keep** |
| EN m3/m0 | english + prompts_en + en_10files | — | **EN Keep** |
| jitter / N=2 / no-fast_inmemory | 300/240 / 2 / OFF | なし | Keep |

**コピー禁止（旧 md）:** `drop_initial_audio_ms`、`response_trigger` / `--no-skip`、silence 900、`mic_send_max_s 8`、timeout 12、`gap_s 2.0`、`mic_vad_debug`、旧 `--audio_device 14`、JP/EN リポ取り違え。

**Pass 基準:**
- [x] device 番号をクエリで確定し、旧 14 を使っていない
- [x] JP Zoom: 相手に日本語音声。VirtualCam 映像・口が変化
- [x] EN Zoom: 同上（英語）。`prompts_en` Keep
- [x] AUDIO_BEFORE_M0=0、通常 clear 0、方式2 / N=2 / `--no-fast_inmemory` 維持
- [x] コード差分なし。短文・短い口フリーズは Fail にしない

**Fail:** 旧フルコマンド復元、図A破壊、session_loop 改修、CABLE を mic にする、prompts_en_dur、B/O 再開。

**子報告（2026-08-21）:**
- CABLE Input = **6**（MME Output）。旧 14 未使用。コード差分なし。
- JP `sess_z1_jp_subj_20260821_140722`: 初回 Zoom=OBS VirtualCam は映像なし。カメラを **Unity Video Capture** にして再走。4t とも相手側に JP 音声＋リップ。AUDIO_BEFORE_M0=0、clear 0、`real_audio_started=4`。
- EN 実体 `sess_z1_jp_subj_20260821_141501`（SESSION 名は JP のまま。ログは EN Keep）。先頭十数秒無反応のあと 4t EN 音声＋リップ。AUDIO_BEFORE_M0=0、clear 0、`real_audio_started=4`。
- 観察（非Gate）: 顔上下動時の M0↔BGV 位置ずれ（既存 backlog）。EN 再走の session_id 未置換。

**親判定（2026-08-21）:**
- **Pass。** Keep = クエリした CABLE Input への `--audio_device` ＋ Zoom カメラ **Unity Video Capture** ＋ 起動順（`[virtualcam_persistent][OK]` 後に Zoom が掴む）。O1–O3 は OBS 制御 Keep のまま。B/O 再開しない。
- 運用メモ: EN 再走は `$session_id = "sess_z1_en_subj_$ts"` を置換する（今回未置換は Gate にしない）。
- 検証ライン（EN-DUR + Z1 片方向）クローズ。**次本線=Z2**（2026-08-22。受信→M1。Z1 再開しない）。prompt 本番化・BGV ずれは backlog。push/merge 本線外。

---

## Phase Z2: Zoom 受信→M1 mic（OS 分離ミックス）

**目的:** スマホ遠隔参加者が AI と会話できること。Zoom 受信音声を M1 `--mic_input_device` に載せる。AI 出力（CABLE Input）と混ぜない。session_loop で PCM ミックスしない。図A非破壊。

**事実（Z1 再走・採用）:**
- `sess_z1_jp_subj_20260822_164959` / `131201`。AI は USB `--mic_input_device 1` しか聞いていない
- 164959: T1 のみ USB 720ms でかみ合い。T2–T3 は ACTIVITY なし・`input_audio_ms=0`・idle
- 131201: 4t とも idle（USB に有声音なし）
- Zoom Speaker=CABLE Input のため相手声は仮想ケーブル側に落ち、USB に乗らない。ヘッドホン可聴 ≠ Live API

**設計決定:**
- 第一手段 = **OS 音声グラフ**（Voicemeeter 等）。M1 第二入力実装ではない
- Z1 Keep（出）：`--audio_device` = クエリした CABLE Input（この機は 6）。Zoom Microphone = CABLE Output。カメラ = Unity Video Capture
- Z2 新設（入）：Zoom Speaker を CABLE Input にしない。Zoom 受信だけを virtual mic へ（USB 手元を同じ bus に足してよい）。`--mic_input_device` = その virtual mic（クエリ。決め打ち禁止）
- **CABLE Output を `--mic_input_device` にしない**（AI 声が戻る）。旧 docs のその配線は使わない
- EN は JP Pass 後の短確認（親が書く）。今は JP 4t＋スマホ参加
- 主導権 mute+interrupt は Zoom では使わない

### 配線（親固定・番号はクエリで埋める）

```text
[出・Z1]  M1 --audio_device --> CABLE Input
          Zoom Microphone   --> CABLE Output

[入・Z2]  Zoom Speaker      --> 受信専用デバイス（Voicemeeter VAIO 等）
                              NEVER CABLE Input
          M1 --mic_input_device --> 受信(+任意 USB) の virtual mic
                              NEVER CABLE Output
```

**コピー禁止:** `drop_initial_audio` / `response_trigger` / silence 900 / 旧フルコマンド / CABLE Output を mic / ジッタ延長 / B/O / `prompts_en_dur` / `if language` / session_loop PCM ミックス

**Pass 基準:**
- [ ] device クエリ表（CABLE / Voicemeeter / USB）と配線レシピ 1 枚
- [ ] 遠隔スマホ発話で `has_spoken`・`input_audio_ms>0`・transcription が相手内容に一致。idle 連発は Fail
- [ ] AI 返答がかみ合う（主観）。自己エコーなし（AI 音声で VAD 誤発火しない）
- [ ] Z1 片方向回帰なし（相手に AI 音声＋口）
- [ ] AUDIO_BEFORE_M0=0、通常 clear 0。session_loop 非変更
- [x] Voicemeeter 等が無い／配線不能なら **Hold＋親へ**（コードミックス提案は勝手にしない）

**子報告（2026-08-22）:**
- コマンド未走行。コード差分なし。Voicemeeter プロセス／VAIO 無し。仮想ケーブルは 1 対（Input 6 / Output 3）のみ。
- ステレオミキサー(2) は Realtek ループバック。USB ヘッドホン再生は乗らない。CABLE Output=3 を mic にはしていない。
- `$cableIn=6` は埋まる。`$micIn` は埋められない。

**親判定（2026-08-22）:**
- **Hold（配線不能）。Fail ではない。** EN に進まない。手段欠落で止めない。
- 次=**Z2b**: 同じ AI 機に **Voicemeeter Banana** 導入 → 再クエリ → 配線 → JP 4t＋スマホ。ケーブル2本目でも可だが推奨は Banana。
- 出さない: session_loop ミックス、CABLE Output を mic、部屋スピーカー拾い、Z1 再開、別 PC。

---

## Phase Z2b: Voicemeeter Banana 導入＋JP スマホ確認

**目的:** 受信専用 virtual mic をこの AI 機に作り、Z2 Gate を実施する。session_loop 非変更。図A非破壊。

**設計決定:**
- 導入対象 = **Voicemeeter Banana**（公式のみ）。同じ AI/OBS 専用機。別 PC は買わない
- 導入後に再クエリ。番号決め打ち禁止
- 配線は Z2 親固定のまま。CABLE 1 対は Z1 Keep（**番号は Banana 後に変わる。再クエリ**）
- USB mic=1 はローカル検証のみ。Gate は **遠隔スマホ発話**
- EN 短確認は任意（配線は同じ。親がコマンドだけ出す）

### Banana 配線（親固定・この機スナップショット 2026-08-22）

```text
[出・Z1]  M1 --audio_device --> CABLE Input (23)     ※旧6は使わない
          Zoom Microphone   --> CABLE Output (8)

[入・Z2b] Zoom Speaker      --> Voicemeeter Input VAIO (19)   NEVER 23 / VAIO3(14)
          Banana: VAIO strip → B1 のみ。B2/A2/A3 オフ。CABLE を Hardware In に足さない
          M1 --mic_input_device --> Voicemeeter Out B1 (9)    NEVER CABLE Output (8)
          A1 = ヘッドホン (15) 任意。USB mic へ戻さない
```

**Pass 基準:**
- [x] Banana 導入後のクエリ表に VAIO / Voicemeeter Output がある
- [x] 配線レシピ 1 枚（番号埋め）
- [x] JP 4t＋スマホ: 遠隔で `has_spoken`・`input_audio_ms>0`・transcription 一致。idle 連発は Fail
- [x] かみ合い（主観）。自己エコーなし。Z1 片方向回帰なし
- [x] AUDIO_BEFORE_M0=0、通常 clear 0。session_loop 非変更

**子報告（2026-08-22）:**
- Banana 導入後に番号ずれ。CABLE Input **23**（旧6は Voicemeeter Out A4）。`--mic_input_device` **9**（Out B1）。CABLE Output **8** は mic にしていない。
- `sess_z2b_jp_subj_20260822_223714`: スマホ遠隔 4t。音声＋リップ、会話成立。4t とも `has_spoken=True` / idle 0 / `input_audio_ms` 2400–1600。
- `220748` は Speaker=CABLE Input で未到達（混線再現。Keep の反証）。
- 遅延は継続観察。コード差分なし。図A・方式2・N=2・no-fast_inmemory・jitter 300/240・silence 350。

**親判定（2026-08-22）:**
- **Pass。** Keep = Banana 常駐＋VAIO→B1 のみ＋再クエリ番号。Z1 片方向は維持。
- 運用マニュアルは上節「Zoom 運用」に統合。
- EN 短確認は任意（同じ配線。ユーザー希望時）。本線の次はユーザー判断（YouTube/TikTok 受信は同じレシピ）。push/merge 本線外。

---

## 変更履歴

X1+F1 commit 後。別ブランチ。スプライト 6→9＋M3英語 knn。JP 本線 Keep 巻き込み禁止。オリエン／差分は改めて添付。

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
| 2026-07-30 | `phase17-pass` tag（`93f0820`）付与確認 |
| 2026-07-30 | Phase18=M3 mouth 前線先行定義・子プロンプト発行 |
| 2026-07-30 | Phase18以降の分岐順を SSOT 反映（頭止め→M0→N再計測オプション。当面ジッタ/付け替え/口形捨て/6全振り禁止） |
| 2026-07-30 | Phase18 Pass-with-defer。M3因果emit計測Pass・主観未達。Keep=M3。次=頭止め（Phase19） |
| 2026-07-30 | M3 `phase18-pass` tag（`0bfd8bb`）付与確認 |
| 2026-07-30 | Phase19=到着順 enqueue 頭止め定義・子プロンプト発行 |
| 2026-07-30 | Phase19 Pass-with-defer。HOL否定（order_wait≈0）。Keep=観測のみ。挙動修正破棄 |
| 2026-07-30 | 長尺品質ライン導入（主観主Pass・子セルフゲート）。次=Phase20 Phase18回帰／ベース復帰（VAD本線化禁止） |
| 2026-07-30 | Phase20定義・子プロンプト発行 |
| 2026-07-31 | Phase20 Pass。183212型復帰。因果emit Keep（改悪主因否定）。次=M3 VAD buffer/index |
| 2026-07-31 | Phase21定義・子プロンプト発行（長尺品質ライン＋セルフゲート適用） |
| 2026-07-31 | Phase21 Pass。VAD相対indexで≥10s口回復。Keep=M3 mouth_streamer。残=散発欠落→Phase22任意 |
| 2026-07-31 | M3 `phase21-pass` tag（`02dd1aa`）付与確認 |
| 2026-07-31 | メモリは未評価・Phase21と並行観測可（本線差し替えなし）とメモ |
| 2026-07-31 | Phase22=長尺散発欠落／等速残差 本定義・子プロンプト発行 |
| 2026-07-31 | Phase22 Pass-with-defer。step1 hold-extend Keep。残=中盤REB1＋終端停止→Phase23任意 |
| 2026-07-31 | M1 `phase22-pass` tag（`a26efcf`、step1のみ）付与確認 |
| 2026-08-01 | Phase23=中盤REB1／終端停止 本定義・子プロンプト発行 |
| 2026-08-01 | Phase23 Hold。縫い目／終端 defer。次=Phase24 メモリ／キャッシュ削減→その後N再計測 |
| 2026-08-01 | Phase24定義・子プロンプト発行 |
| 2026-08-01 | Phase24 Pass-with-defer。親RSS頭打ち・品質非回帰。Keep=M1+M3+M0。次=N再計測（Phase25） |
| 2026-08-01 | `phase24-pass` tag 付与確認: M1 `2510f1c` / M3 `abf8226` / M0 `af3dc72`（push未） |
| 2026-08-01 | Phase25=削減後 N=2/3/4 再計測 本定義・子プロンプト発行（default据え置き） |
| 2026-08-02 | Phase25 Pass-with-defer。削減後も N↑効果なし。default=N=2 再確認。N2/N3微差再テスト不要 |
| 2026-08-02 | 基本線閉鎖メモ。Phase26=残差指紋横断切り分け（実装なし）定義・子プロンプト発行 |
| 2026-08-02 | Phase26 Pass。本命=終端BLOCKED＋多ターンCATCHUP。2チケット分離→P27/P28 |
| 2026-08-02 | Phase27=終端/ENQUEUE_BLOCKED 定義・子プロンプト発行（push観測復元可・CATCHUP混ぜない） |
| 2026-08-03 | Phase27 Pass-with-defer。AB/BLOCK 0・終端途切れ改善。残=T1–4口詰まり→Phase28 CATCHUP |
| 2026-08-03 | M1 `phase27-pass` tag（`dff3e62`）付与確認 |
| 2026-08-03 | Phase28=CATCHUP/VirtualCam 定義・子プロンプト発行 |
| 2026-08-03 | Phase28 Pass-with-defer。VCam META_DEFER Keep。主観横ばい。次=P29 sync_meta更新タイミング |
| 2026-08-03 | M1 `phase28-pass` tag（`d71afcf`、virtualcamのみ）付与確認 |
| 2026-08-03 | Phase29定義・子プロンプト発行（候補A。候補Bは後） |
| 2026-08-03 | Phase29 Pass-with-defer。候補A効果（T1–7）。T8偽ゼロ残→Hotfix |
| 2026-08-03 | M1 `phase29-pass` tag（`6a35cd7`、session_loopのみ）付与確認 |
| 2026-08-03 | Phase29 Hotfix定義・子プロンプト発行 |
| 2026-08-03 | Phase29hf Pass-with-defer。T8偽ゼロ閉鎖。残=候補B（中盤供給／欠落）→Phase30 |
| 2026-08-03 | M1 `phase29hf-pass` tag（`a656270`）付与確認 |
| 2026-08-03 | Phase30=候補B 定義・子プロンプト発行 |
| 2026-08-04 | Phase30 Pass-with-defer。post_gen REB閉鎖・主観全体改善。残=idle浅い/mid-real/散発欠落 |
| 2026-08-04 | M1 `phase30-pass` tag（`4d19fb9`）付与確認 |
| 2026-08-04 | Phase31=発話中 mid-real 供給／残欠落 定義・子プロンプト発行（idle/縫い目本線外・効き薄なら耐久へ） |
| 2026-08-04 | Phase31 Hold。mid-real既閉鎖・品質打ち切り。次=短回帰(P32)→運用耐久→品質凍結 |
| 2026-08-04 | Phase32=battle/talkover/event 短回帰 定義・子プロンプト発行 |
| 2026-08-04 | Phase32 Pass。自動＋主観OK・差分なし。次=運用耐久（Phase33）→品質凍結 |
| 2026-08-04 | Phase33=運用耐久 定義・子プロンプト発行 |
| 2026-08-05 | Phase33 Pass-with-defer。24t完走・RSS頭打ち。品質凍結マイルストーン宣言 |
| 2026-08-05 | 運用メモ: reconnect_per_turn は API1008退避（必須ではない・default変更なし） |
| 2026-08-05 | 役割転換: リップ品質凍結維持。新ライン＝レスポンス／VAD プロファイル（R1/R2）。別進捗ファイルは作らず本 PROGRESS に追記。Phase R1 定義・子プロンプト発行 |
| 2026-08-06 | Phase R1 計測計画 Go。同一 perf_ms で speech_end/activity_end/first_audio。A/B silence 600 vs 350。実装→計測へ |
| 2026-08-06 | Phase R1 Pass。silence待ちは設定どおり短縮可・合計過半は API。主観で 600→350 は体感差小。運用ベース=350 固定。600/350 切替不採用・R2 hold。次=R1b（200–250 vs 350） |
| 2026-08-06 | Phase R1b Pass・分岐A。通常=350／攻め腕=250（主観満足・短回帰OK）。200不要。R2=管理画面切替へ |
| 2026-08-06 | Phase R2 手段 Go: ファイル watch（350/250）。起動=CLI>file>350、runtime は file 上書き可。実装へ |
| 2026-08-06 | Phase R2 Pass-with-defer。runtime 切替 OK。管理画面 UI 書込→R2b |
| 2026-08-06 | Phase R2b Pass。管理画面→vad_profile 書込＋Live set 確認。レスポンス本線クローズ可 |
| 2026-08-06 | R2+R2b commit `eb26a9c` / tag `phase-r2-pass` |
| 2026-08-09 | レスポンス本線クローズ確定。次本線=I1 アイドル発話（Phase12 idle silent とは別）。BGV/OBS は予約のみ |
| 2026-08-09 | Phase I1 設計 Go: client text トリガー＋`--idle_utterance_s` default6。実装→検証へ |
| 2026-08-09 | Phase I1 Pass-with-defer。会話後 idle OK・主観3s妥当。default→3。初手無言→I1b |
| 2026-08-09 | Phase I1b 設計 Go: 初手のみ activity_start→text→activity_end。実装→検証へ |
| 2026-08-09 | I1b v1 Fail（空 activity+text・audio0）Revert。再 Go=候補A（silent PCM→end→text） |
| 2026-08-09 | Phase I1b Pass（候補A）。初手／会話後／talkover／主観OK。アイドル本線クローズ可 |
| 2026-08-09 | I1/I1b commit `bb0ef93` / tag `phase-i1-pass` |
| 2026-08-09 | 次本線=B1 BGV 時計特定（調査のみ）。実装・300ms決め打ち禁止。子プロンプト発行 |
| 2026-08-09 | Phase B1 Pass。Live BGV=virtualcam FPS 順次。IDLE_BG_ADVANCE 単純廃止禁止。次=B2 定量 |
| 2026-08-09 | Phase B2 Pass。方式A採用（PLAYING中 audio→BG、IDLE維持）。次=B3 実装 |
| 2026-08-09 | Phase B3 Pass-with-followup。方式A Keep。Turn2 bg_pos固着→B3hf |
| 2026-08-10 | Phase B3hf Pass-with-followup。二重読取閉鎖・T1停滞解消。T2先頭残→B3hf2 |
| 2026-08-10 | Phase B3hf2 Pass。falsy-0 閉鎖。BGV runtime 同期クローズ可。次=B4 pose/幾何分析。OBSはB4後可 |
| 2026-08-10 | B2–B3hf2 commit `87ecb2d` / tag `phase-b3-pass` |
| 2026-08-10 | Phase B4 Pass。主因B（turnローカル pose）。次=B5 適用修正。OBSはB5後 |
| 2026-08-11 | Phase B5 主観 Fail・方式 Keep。pose絶対indexは正しい。snapshot/missing0 →B5hf |
| 2026-08-11 | Phase B5hf Pass-with-defer。T1/missing閉じ。T3境界誤freeze→B5hf2 |
| 2026-08-11 | Phase B5hf2 Pass。fo整合でT3閉鎖。B5系クローズ可。OBS発行可。T2数フレームは任意BL |
| 2026-08-11 | B5〜B5hf2 commit `9b1466c` / tag `phase-b5-pass`。Bライン Pass-with-defer |
| 2026-08-11 | 次本線=OBS。O1=WS+管理画面+背景静止画/BGM。O2=当てフリ見せ消し。O3=スミス。子プロンプト発行 |
| 2026-08-13 | Phase O1 Pass。実切替OK＋flat local Hotfix Keep。次=O2 当てフリ |
| 2026-08-13 | Phase O2 Pass。実機4ターン見せ消しOK＋OFF確認。Keep=first_audio IN／turn-end OUT。次=O3 スミス |
| 2026-08-14 | Phase O3 Pass。スミス一斉表示＋Smith_Effect。OBS本線(O1–O3)クローズ可。増殖加速は後続 |
| 2026-08-14 | O1–O3 commit `62798bf` / tag `phase-o3-pass` |
| 2026-08-15 | 次本線=X1 blink（9_1/9_2=3s・他=5s）→F1 黒縁。英語版はX1+F1後。子プロンプト発行 |
| 2026-08-15 | Phase X1 Pass-with-defer。M3生成分岐 Keep。Live未配線→X1b。F1はX1b後 |
| 2026-08-15 | Phase X1b Pass。Live auto_blink。次=F1 黒縁（比較レビュー→Go→実装） |
| 2026-08-15 | Phase F1 比較レビュー Go。unpremultiply 1箇所のみ。過補正なら即Revert |
| 2026-08-15 | F1 unpremultiply Fail→Revert。現行blitは既にstraight。次=F1s合成主観1本 |
| 2026-08-16 | F1s Pass（A）。黒縁運用上問題なし。F1クローズ。英語版はX1+X1b+F1s後可 |
| 2026-08-17 | 英語版本線着手。計画SSOT=`english_m3_local_integration_precheck.md`。EN-RT0/1/2 を PROGRESS 対応づけ（CTC=EN-RT3 defer）。M3 EN=`C:\dev\M3_Live_API_1_english` / M0 EN=`C:\dev\M0_session_renderer_final_1_english`。JP 現状維持。EN-RT0 子プロンプト発行 |
| 2026-08-17 | 9 mouth PNG 一次ソース確定 `C:\Users\john\Desktop\EN_img9`（576枚・emotion/view）。M0 EN `assets/` へ構造維持コピーを EN-RT0 に含める。EN-RT1 の未着 Blocker は解消見込み。子プロンプト手順5差し替え |
| 2026-08-18 | EN-RT0 Pass-with-defer。M3 EN `2f961bc` / M0 EN 骨格 / PNG +576。JP 非混入。次=EN-RT0b（`C:\Users\john\Desktop\atlas.en.json` 配置＋ PNG SSOT=`mouth_ch.png`、論理 key `mouth_sh` 維持）。子プロンプト発行 |
| 2026-08-18 | EN-RT0b Pass。atlas `normal/front` 9/9、`mouth_ch.png`=64 / `mouth_sh.png`=0。次=EN-RT1（Realtime kNN→9 mouth→M0。平坦化しない） |
| 2026-08-18 | EN-RT0/0b を EN リポに commit+tag。M3 `2f961bc`/`en-rt0-pass`、`3817a79`/`en-rt0b-pass`。M0 EN `baf0052`/`en-rt0b-pass`（PNG は JP 同様 gitignore）。M1 未 commit。push なし |
| 2026-08-18 | EN-RT1 Pass-with-defer。kNN en_10files/k=9 と M0 `<expr>/<view>/` 接続。Live主観は EN-LIVE1（`--prompt_dir configs/prompts_en`、JP prompts 非上書き）。id6 GT は EN-DB1。次本線=EN-RT2 |
| 2026-08-18 | EN-RT2 Pass。未送信 frame に 160ms Hold。図A=KNN→Hold→M0→enqueue。Live主観はまだ不要。次予約=EN-LIVE1 |
| 2026-08-18 | EN-LIVE1 発行。JP 主観コマンド差分。prompts_en 新規、pose/bg は JP のまま。CTC/DB1/欠view 対象外 |
| 2026-08-18 | EN-LIVE1 Fail。transcription 英語4tだが KNN が JP m3p を見て knn_predictor 欠。prompts_en Keep。hotfix=ファイル経路 load |
| 2026-08-19 | EN-LIVE1hf Pass。Hold と同じファイル経路で knn_predictor を EN 先載せ。再走 `sess_en_live1_subj_20260819_140651`。ModuleNotFoundError=0、chunks_n=262、英語4t 主観 Pass。EN-LIVE1 を Pass に戻す。短い口フリーズとトーク被りは観察。英語本線クローズ可 |
| 2026-08-20 | 検証ライン着手。EN-DUR1（1t 長文）→ EN-DUR2（12t）→ Z1（Zoom JP+EN）。コマンドは EN-LIVE1 4t ベース。JP 長尺の synth/trigger/silence600 はコピーしない。push/merge 本線外。EN-DUR1 子プロンプト発行 |
| 2026-08-20 | EN-DUR1 Hold。142252/143157 は口・図A OK だが短文のみ。主因=prompts_en 1文＋ idle 2.5 の tail。次=EN-DUR1b（C+D: EN 専用 trigger＋synth 無人＋idle 20）。通常 trigger 本線化・prompts_en 本番変更・silence 600 は禁止。子プロンプト発行 |
| 2026-08-20 | EN-DUR1b Hold。無人は I1 3s が synth 4s より先→trigger 未送。手元は trigger 到達だが prompts_en 一文が勝ち 90s Fail。図A Keep。次=EN-DUR1c（prompts_en_dur コピー＋idle 0＋drain 45）。BGV↔M0 ズレは backlog |
| 2026-08-20 | EN-DUR1c Pass-with-defer。手元 222002 長文・口OK（audio_ms≈31.6s）。無人は PCM 未着。支配層=system 長さ行。本番 prompt 方針は backlog（案1主＋案3補助・今は非編集）。次=EN-DUR2（12t・prompts_en）。子プロンプト発行 |
| 2026-08-20 | EN-DUR2 Pass。`sess_en_dur2_subj_20260820_231430` 12/12。AUDIO_BEFORE_M0=0。DUR 例外なし・差分なし。T6 BGV ずれは backlog。次=Z1（Zoom 旧コマンド添付後）。DUR1c の Keep All/tag は必須ではない |
| 2026-08-21 | Z1 Go。資料=ルーティングのみ。現行合格 4t の `--audio_device` をクエリした CABLE Input へ。旧 client-VAD フルコマンド復元禁止。子プロンプト発行 |
| 2026-08-21 | Z1 Pass。CABLE Input=6。JP 140722 / EN 実体 141501。相手側に音声＋リップ。カメラ SSOT=Unity Video Capture（OBS VirtualCam 決め打ちは誤り）。コード差分なし。検証ラインクローズ。次本線なし |
| 2026-08-22 | Z2 Go。片方向 Z1 は維持。受信は OS グラフで AI 出力と分離。CABLE Output を mic にしない。session_loop ミックス禁止。JP 4t＋スマホ優先。子プロンプト発行 |
| 2026-08-22 | Z2 Hold（Fail ではない）。VAIO 無し・ケーブル1対のみ。次=Z2b Banana 導入＋再クエリ＋JP スマホ。子プロンプト発行 |
| 2026-08-22 | Z2b Pass。`sess_z2b_jp_subj_20260822_223714` 遠隔4t成立。CABLE In=23 / mic=B1=9。Banana 常駐＋VAIO→B1 を運用 SSOT。220748 は Speaker=CABLE In で未到達。EN 短確認は任意 |
| 2026-08-23 | Zoom 第三者運営手順を `docs/ops_zoom_third_party.md` に抜き出し（PROGRESS「Zoom 運用」が SSOT）。コード非変更 |
| 2026-08-23 | 新親着任。本線=② B仕上げ再開（ズレ再発）。B6=調査のみ（合成瞬間 display_bg↔pose Δ、JP+EN）。実装・+40〜80・pose先送り・Colab禁止。① Zoomマニュアルは触らない |
| 2026-08-23 | Phase B6 Pass。B3生き。定常PLAYING Δ≈0〜1。idle/ターン境でBG進み（仮説2）。方式C採用。次=B7（境でBGをpose/M0枚へ戻す）。pose先送り/IDLE廃止/audio_msオフセット/pose.json先修正は出さない |
| 2026-08-24 | Phase B7 Pass-with-defer。方式C Keep。定常PLAYING/高速上下 Δ最大1。EN PLAYING最大275消滅。JP+EN主観で顔Y一致。②クローズ。次本線なし。貼り/口−音/第二手法は開かない |
| 2026-08-25 | `main` へ `feature/local-vad-restore` を FF-only マージ（`813c641` / `phase-b7-pass`）。希望順のみ記録（1=女性声 2=EN本番prompt 3=event catalog 4=第三者コマンドdocs）。①以降はまだ出さない |
| 2026-08-25 | 希望順①着手。Phase V1=Live 声 Kore 固定。ブランチ `feature/live-voice-kore`（from `main` `354d1e4`）。②以降は出さない |
| 2026-08-26 | Phase V1 Pass-with-note。Keep=Aoede＋camelCase wire（snake_case は Live 無視→Puck）。Kore 名は Keep 不可。JP 中性は同一 Aoede の言語差として受容。②は出さない |
| 2026-08-26 | `main` へ `feature/live-voice-kore` を FF-only（`b4f7d5c` / `phase-v1-pass`）。希望順②着手。Phase P1。ブランチ `feature/en-prod-prompts`。③④は出さない |
| 2026-08-27 | Phase P1 Pass-with-note。prompts_en / prompts_en_battle 切替。口 barge-in＝既存 talkover。140322 Fail→175716 Pass。Keep All は親。③④は出さない |
| 2026-08-27 | `main` へ `feature/en-prod-prompts` を FF-only（`f117d51` / `phase-p1-pass`）。希望順③着手。Phase E1。ブランチ `feature/event-catalog-admin`。④は出さない |
| 2026-08-27 | E1 途中。dropdown Keep 候補。まだ Pass/Keep All しない。次=E1hf（evt_001 末尾飛び・open直後 frame0 seek）。完了ゲートは E1b 予約 |
| 2026-08-27 | Phase E1 Pass-with-defer。evt_001 頭から順再生（232324）。完了ゲートは E1b。Keep All は親。④は出さない |
