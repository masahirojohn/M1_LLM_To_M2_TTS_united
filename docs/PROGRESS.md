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
| レスポンス運用ベース（R1 後） | **`mic_vad_silence_ms=350` 固定候補**（他 `mic_vad_*` 据え置き）。旧 600 は残す理由なし |

### R フェーズ一覧

| Phase | 名称 | 状態 | Pass 日 |
| --- | --- | --- | --- |
| R1 | レスポンス遅延計測＋silence A/B＋短回帰 | `pass` | 2026-08-06 |
| R1b | silence 攻め腕短検証（250 vs 350） | `pass` | 2026-08-06（分岐 A） |
| R2 | runtime ファイル切替（350／250） | `pass` | 2026-08-06（Pass-with-defer: UI→R2b） |
| R2b | 管理画面→`vad_profile_live.txt` 書込 | `pass` | 2026-08-06 |

### プロファイル（R1b 確定）

| プロファイル | `mic_vad_silence_ms` | 用途 | 状態 |
| --- | ---: | --- | --- |
| 通常 | **350** | 日常運用 | 確定 |
| 攻め腕（バトル） | **250** | 反応優先。短回帰 OK・主観で満足到達 | 確定 |
| （不採用）旧通常 | 600 | 体感差小 | 廃止 |
| （見送り）200 | 200 | 250 で満足のため未実施 | 不要 |

**方針:** 切替は **通常=350／攻め腕=250**。R2 で管理画面（別プロセス）から再起動なし切替。default CLI 据え置き（親承認後に default=350 化を検討可）。API `end→first` 短縮は別トラック（今はやらない）。200 は不要。

### 当面の非本線（後続・今は実装させない）

- 無言数秒後の AI アイドル発話（レスポンス系の次候補メモ）
- **BGV↔M0 位置／向きの徐々ズレ**（音声↔M0 は `played_audio_ms` SSOT 同期済。BGV 時計の特定が先。着手時は wall / playlist / played_audio_ms のどれかを実コード特定が第一タスク）
- OBS BGM／背景切替などの OBS 制御

### 全 R Phase 共通禁止（子）

- `docs/PROGRESS.md` 無断編集、凍結 Keep 破壊、N↑、ジッタ延長で症状隠し、frame drop、音声先行 enqueue、口形捨て
- 「ローカル VAD だから遅れは不可避」で打ち切らない（パラメータで縮む前提を **計測で** 検証する）
- リップ品質・縫い目・idle 品質本線化への回帰

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
- **Pass。** レスポンス／VAD プロファイル本線（R1→R1b→R2→R2b）クローズ可。
- 運用: 通常=350／攻め腕=250（管理画面）。起動 CLI を 350 固定にする必要なし（default 解決＋ファイルで足りる）。
- 次本線候補（未発行）: API `end→first` は別トラック／アイドル発話／BGV↔M0／OBS。リップ品質には戻らない。

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
