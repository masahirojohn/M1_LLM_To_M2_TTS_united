# M1_LLM_To_M2_TTS
# aituber_llm_tts

猫キャラクター AItuber システムの **M1 (LLM) + M2 (TTS) + オーケストレータ** の
バッチ版デモリポジトリです。

- 契約仕様は `contracts_llm_tts_v0.2` を前提とします。:contentReference[oaicite:13]{index=13}
- `emo_id` をソース・オブ・トゥルースとし、  
  `emo_id -> Gemini TTS 用スタイルプロンプト` マッピングで TTS スタイルを決定します。
- TTS の `audio_ms` が時間軸の真実となり、将来 M3’ / M0 / M3.5 に連携します。

## セットアップ

```bash
pip install -r requirements.txt

## Local Windows Smoke

Local Windows validation completed.

Confirmed:

- WAV input E2E
- microphone input bridge
- realtime streaming
- realtime streaming + pipe BG
- microphone input E2E one-command smoke

See:

```text
docs/local_migration_smoke.md


## 2. README追記

```md
## Local OBS Smoke

Windowsローカル環境で、mic入力からOBSへの映像・音声出力smokeを確認済み。

確認済み：

- mic input → Gemini Live API
- mouth / expression 生成
- M0 FG PNG生成
- Unity Video Capture 経由でOBS映像表示
- VB-Audio Virtual Cable 経由でOBS音声入力

詳細：

```text
docs/local_obs_smoke.md