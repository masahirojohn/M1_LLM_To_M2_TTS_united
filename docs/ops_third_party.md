# 第三者向け 主要コマンド＋事前準備

抜き出し元: `docs/PROGRESS.md`（申し送り・Zoom 運用・EN-LIVE1 必須差分・P1・E1/E1b Keep）と `docs/ops_zoom_third_party.md`（Banana / Zoom 配線 / JP Zoom コマンド / EN 起動差分）。  
**矛盾したら PROGRESS を正とする。** このファイルは新規設計ではない。フラグ・手順の発明はしない。

対象: 第三者がこの機で **JP ローカル / JP Zoom / EN 通常 / EN battle / イベント投入** を取り違えないこと。  
Zoom 配線と JP Zoom コマンドの全文は `docs/ops_zoom_third_party.md` を見る（ここへコピーしない）。

コード変更・図A変更・B/O 再開はしない。

---

## 1. 事前準備

- この機は **AI / OBS 専用**。人間は Zoom / SNS にこの機から参加しない。
- Voicemeeter は **Banana**（`voicemeeterpro.exe`）。**Standard は使わない。**
- USB mic はローカル検証のみ。`mic=1` だけなら Banana 不要。本番の会話入力は遠隔の相手声。
- 毎回 Banana を使うなら、先に **Banana を起動して開いたまま**。
- `--audio_device` / `--mic_input_device` は **再クエリ**（番号の永久決め打ち禁止）。

再クエリ（`docs/ops_zoom_third_party.md` のコピー）:

```powershell
& C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"
```

MME 行で名前を照合する。ops_zoom の 23 / 9 / 8 / 19 は Banana 導入後スナップショットの一例。旧 CABLE Input=6 は使わない。

| 用途 | `--m3_repo_root` | `--m0_repo_root` |
| --- | --- | --- |
| JP | `C:\dev\M3_Live_API_1_united` | `C:\dev\M0_session_renderer_final_1` |
| EN | `C:\dev\M3_Live_API_1_english` | `C:\dev\M0_session_renderer_final_1_english` |

M1 は同一リポ `C:\dev\M1_LLM_To_M2_TTS_united`。切替は `--m3_repo_root` / `--m0_repo_root`。`if language` は使わない。

起動後、ログに `[virtualcam_persistent][OK] device=Unity Video Capture` が出るまで待つ。Zoom カメラはそれ以降。

---

## 2. 共通 Keep フラグ

運用ベース（壊したら Fail）:

| 項目 | 値 |
| --- | --- |
| 方式2 | クライアント VAD（`automatic_activity_detection=False`＋ローカル RMS＋`activity_start` / `activity_end`） |
| N | `--m0_worker_n 2` |
| inmemory | `--no-fast_inmemory` |
| jitter | `--audio_player_initial_buffer_ms 300` / `--audio_player_rebuffer_target_ms 240` |
| silence | `--mic_vad_silence_ms 350` |

JP Zoom コマンド本体のその他フラグは `docs/ops_zoom_third_party.md` の合格コマンドをそのまま使う（旧 VAD フルコマンドは復元しない）。

---

## 3. JP ローカルと JP Zoom

### JP ローカル（USB mic）

- `--mic_input_device` = USB mic（再クエリ。ローカル検証）。Banana 任意。
- `--audio_device` = ローカル再生デバイス（再クエリ。CABLE Input ではない）。
- `--m3_repo_root` / `--m0_repo_root` は上表の JP。
- `--prompt_dir C:\dev\M1_LLM_To_M2_TTS_united\configs\prompts`
- 共通 Keep は §2。

コマンド本体は `docs/ops_zoom_third_party.md` の「現行 JP Zoom コマンド」と同形。差し替えるのは device 番号（USB mic / ローカル再生）だけ。ops_zoom の `$cableIn=23` / `$micIn=9` をローカルに転用しない。

### JP Zoom

配線（Banana / Zoom Mic=CABLE Output / Speaker=VAIO / Cam=Unity Video Capture）と JP Zoom コマンド全文は **`docs/ops_zoom_third_party.md`**。ここへコピーしない。

主導権 mute+interrupt は Zoom では使わない。

---

## 4. EN Live

配線と Banana / Zoom 設定は JP Zoom と同じ（`docs/ops_zoom_third_party.md`）。起動差分だけ（同ファイルおよび PROGRESS EN-LIVE1 / P1 のコピー）:

必須3フラグ:

- `--prompt_dir C:\dev\M1_LLM_To_M2_TTS_united\configs\prompts_en`  
  または `--prompt_dir C:\dev\M1_LLM_To_M2_TTS_united\configs\prompts_en_battle`
- `--output_audio_transcription`
- `--stream_mouth_gt_glob data/knn_db/en_10files.phoneme_gt.f1f2.json`

リポ:

- `--m3_repo_root C:\dev\M3_Live_API_1_english`
- `--m0_repo_root C:\dev\M0_session_renderer_final_1_english`

pose / bg / m35 は **JP 絶対パスのまま**（ops_zoom JP コマンドの `--pose_json` / `--bg_video` / `--m35_repo_root` を変えない）。口スプライトだけ M0 EN。

| `--prompt_dir` | 用途 |
| --- | --- |
| `configs/prompts_en` | 通常（20あり・30空。1〜2文） |
| `configs/prompts_en_battle` | バトル（20空・30=Studio。さえぎられるまで） |

`prompts_en_dur` は本番にしない。JP `configs/prompts` は上書きしない。同じ dir に 20 と 30 の長さ方針を両方書かない。

ローカル EN は JP ローカルと同じく USB mic + ローカル再生（再クエリ）。Zoom EN は ops_zoom の配線のまま、上の起動差分だけ足す。

---

## 5. イベント

- 管理画面は catalog **プルダウン**から投入。決め打ちボタンはない。
- catalog SSOT = M1 `in/event_catalog.json`。**M3.5 `in/` はスキャンしない。**
- 現行 2 件: `evt_001` / `evt_voice_001`。最大 10。実ファイルの無い id を足さない。
- 投入先は既存 `event_runtime_live.txt`（VirtualCam 挿入）。OBS「背景」静止画/BGM とは別。VirtualCam 内 BGV（猫の体）は切替対象外。
- 追加イベントは **25fps**。尺を duration / 音に合わせる。60fps 長尺は別 Phase。
- イベント中は頭から順再生（sequential_from_0）。open 直後は frame0。B3/B7 はイベント中使わない。
- イベント中の Live 声は出さない（override 中の Live PCM は drop。Hold しない）。イベント wav（`evt_voice_001`）は潰さない。

---

## 6. 禁止

- 主導権 mute+interrupt を Zoom で使う
- CABLE を Banana Hardware In に足す
- CABLE Output を `--mic_input_device` にする（AI 声が戻る）
- `prompts_en_dur` を本番起動に使う
- device 番号の永久決め打ち（旧 CABLE Input=6 を含む）
- session_loop で PCM ミックスする
- 二重 Live、旧 VAD フルコマンド復元、Voicemeeter Standard
- Zoom Speaker = CABLE Input
