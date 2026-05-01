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