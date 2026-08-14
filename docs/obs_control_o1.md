# OBS 制御（Phase O1）— 配置とソース名マッピング

管理画面（Streamlit）プロセスが `obsws-python` で OBS WebSocket に直結する。  
`session_loop`／VirtualCam 内 BGV／AI PCM には接続しない。

## 用語

| 用語 | 意味 | O1 |
| --- | --- | --- |
| OBS 背景 | OBS シーン上の**静止画ソース** | 切替対象 |
| BGM | OBS の**メディアソース** | 切替対象 |
| VirtualCam 内 BGV | M0 合成用の透過動画 | **対象外・非触** |

## 音声ルーティング

- AI 音声: player → PC 再生デバイスを OBS が拾う（本モジュールは触らない）
- BGM: `sources.bgm_media` のファイルだけ切替（AI PCM に混ぜない）

## 設定

| ファイル | 役割 |
| --- | --- |
| `in/obs_control_config.json` | host/port/timeout・OBSソース名・カタログ（コミット可） |
| `in/obs_control_config.local.json` | パスワード等の上書き（gitignore・任意） |
| 環境変数 | `OBS_WS_HOST` / `OBS_WS_PORT` / `OBS_WS_PASSWORD` が最優先 |

パスワードをリポジトリに書かないこと。

**local.json の形（どちらも可）:**

```json
{ "websocket": { "password": "..." } }
```

```json
{ "host": "localhost", "port": 4455, "password": "..." }
```

（後者は flat。実装側で `websocket.*` に正規化してから merge する。）

## OBS 側の事前準備（手動・1回）

1. obs-websocket（OBS 28+ 内蔵）を有効化し、port / password を設定と一致させる
2. シーンに次を作成し、**ソース名を config と一致**させる  
   - 画像ソース `OBS_BG_Still`（背景静止画）  
   - メディアソース `OBS_BGM`（BGM・ループ推奨）  
   - （O2）`OBS_AI_Normal` / `OBS_AI_Zoom`（同一 VirtualCam の通常配置とドアップ配置）
   - （O3）`OBS_Smith_1` 等のクローン（初期非表示）＋ AI拾い音声ソース `OBS_AI_Audio` にフィルタ `Smith_Effect`
3. AI 映像（VirtualCam）と AI 音声デバイスは従来どおり。O1 は背景静止画と BGM、O2 は上記2ソースの見せ消し、O3 はクローン見せ＋音声フィルタ。

## ファイル配置

```text
media/obs_bg/*.png|jpg   … 背景静止画（config.backgrounds[].path）
media/obs_bgm/*.wav|mp3  … BGM（config.bgm[].path）
```

相対パスはリポジトリルート基準。差し替えは同パス上書き、または config にエントリ追加。

## 切替 API（実装）

- 背景: `SetInputSettings` → image `file`
- BGM: `SetInputSettings` → media `local_file` + 可能なら Restart

接続失敗時は UI／戻り値で通知し、Live パイプラインは継続。

## Phase O2: 当てフリ（見せ消し）

OBS に通常用・ドアップ用ソースを**事前配置**し、可視性だけ切替する。Python / M0 / VirtualCam で拡大しない。

| 設定 | 既定 |
| --- | --- |
| `sources.atefuri_normal` | `OBS_AI_Normal` |
| `sources.atefuri_zoom` | `OBS_AI_Zoom` |
| `atefuri.enabled` | `true`（管理画面で OFF 可。runtime は `in/obs_atefuri_live.json`） |

- トリガ: session_loop `_receive_loop` の `first_audio_sec` 初回セット（bootstrap/warmup 除外）。`asyncio.create_task` + `to_thread`（sleep でループを止めない）
- 戻し: 当該ターン終了時に通常ソースへ。OFF 時は管理画面側でも通常へ戻す
- 背景/BGM API は非変更

OBS 準備（O1 に追加）: 同一シーンに `OBS_AI_Normal` と `OBS_AI_Zoom`（VirtualCam 等のクローンをドアップ配置）を作り、名前を config と一致させる。

## Phase O3: エージェントスミス（クローン見せ＋音声フィルタ）

管理画面の**明示ボタン**で、事前配置クローンを一斉表示し、AI拾い音声ソースのフィルタを ON にする。  
当てフリ `first_audio` は使わない。`session_loop` 非触。Python transform / 連続 scale なし。

| 設定 | 既定 |
| --- | --- |
| `smith.clone_sources` | `OBS_Smith_1` / `OBS_Smith_2` / `OBS_Smith_3` |
| `smith.audio_source` | `OBS_AI_Audio`（player を拾う OBS 音声ソース。**BGM ではない**） |
| `smith.filter_name` | `Smith_Effect`（OBS 側で事前作成したフィルタ） |
| `smith.scene` | 空＝現在のプログラムシーン |

- 開始: クローン可視 ON ＋ フィルタ enable
- リセット: クローン可視 OFF ＋ フィルタ disable
- 増殖加速（順次表示・delay 短縮）は後続。O3 は一斉表示の最小版
- `smith.audio_source == sources.bgm_media` は拒否（BGM にフィルタを掛けない）

OBS 準備（O2 に追加）:

1. 同一シーンに VirtualCam（または本体映像）のクローンを `OBS_Smith_1` 等の名前で事前配置し、初期は非表示
2. AI 音声を拾っているソース（例 `OBS_AI_Audio`）にフィルタ `Smith_Effect` を作成（コーラス／ディレイ等。BGM ソースには付けない）
3. ソース名・フィルタ名を config と一致させる
