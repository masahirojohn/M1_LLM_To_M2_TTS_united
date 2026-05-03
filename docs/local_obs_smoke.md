# Local OBS Smoke Memo

## 到達点

Windowsローカル環境で、mic入力からOBS出力までの1ターンsmokeを確認済み。

```text
mic input
→ Gemini Live API
→ audio_response.pcm / tool_call
→ mouth.json / expr.json
→ chunk
→ M0 FG PNG
→ FG+BG composite
→ Unity Video Capture
→ OBS映像表示

audio_response.pcm
→ VB Cable
→ OBS音声ミキサー反応

成功ログ

session_id: sess_obs_live_001

[obs_av][OK] virtualcam=Unity Video Capture
[obs_av] audio_device=15
[obs_av] frames=43
[obs_av][DONE]
[run_mic_input_obs_smoke][DONE]

OBS側：

映像：Unity Video Capture 表示OK
音声：CABLE Output (VB-Audio Virtual Cable) メーター反応OK
重要な注意

現在の run_mic_input_obs_smoke.py は、完全常時リアルタイムではなく、1ターンOBS出力smoke。

mic録音
→ Gemini応答生成
→ M0/M3.5生成
→ OBSへ再生

そのため、マイク発話に対して即時にOBSが反応するのではなく、生成完了後に映像・音声がOBSへ出力される。

OBS設定
映像
ソース追加
→ 映像キャプチャデバイス
→ Unity Video Capture
音声
ソース追加
→ 音声入力キャプチャ
→ CABLE Output (VB-Audio Virtual Cable)
必要ツール
OBS Studio
UnityCapture
VB-Audio Virtual Cable
ffmpeg
Python venv
Python追加依存
pip install pyvirtualcam sounddevice simplejson
固定コマンド
python scripts/live_runtime/run_mic_input_obs_smoke.py `
  --session_id sess_obs_live_001 `
  --m1_repo_root C:\dev\M1_LLM_To_M2_TTS_united `
  --m3_repo_root C:\dev\M3_Live_API_1_united `
  --m0_repo_root C:\dev\M0_session_renderer_final_1 `
  --m35_repo_root C:\dev\M3.5_final `
  --pose_json C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json `
  --bg_video C:\dev\M3.5_final\in\with1.mp4 `
  --duration_s 5 `
  --audio_device 15
APIキー注意

PowerShellで一時設定：

Remove-Item Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

永続化する場合：

setx GEMINI_API_KEY "YOUR_GEMINI_API_KEY"
setx GOOGLE_API_KEY ""

VSCode再起動後に反映。