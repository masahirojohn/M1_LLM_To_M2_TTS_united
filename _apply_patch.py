import sys
p = r'c:\dev\M1_LLM_To_M2_TTS_united\scripts\live_runtime\run_mic_input_obs_realtime_session_loop.py'
with open(p, 'rb') as f:
    raw = f.read()
# Detect CRLF
crlf = b'\r\n' in raw
text = raw.decode('utf-8')
# Normalize for matching using \n
norm = text.replace('\r\n', '\n')

old1 = '    ap.add_argument("--inline_emo_tag_mode", action="store_true")\n    ap.add_argument("--drop_initial_audio_ms", type=int, default=120)'
new1 = '    ap.add_argument("--inline_emo_tag_mode", action="store_true")\n    ap.add_argument("--inline_emo_id", default="1_1")\n    ap.add_argument("--drop_initial_audio_ms", type=int, default=120)'
print('found1:', norm.count(old1))
norm = norm.replace(old1, new1)

old2 = '                    mouth_updated_event=mouth_updated_event,\n                    m0_worker_proc=None,'
new2 = (
    '                    mouth_updated_event=mouth_updated_event,\n'
    '                    inline_emo_id=(\n'
    '                        str(args.inline_emo_id)\n'
    '                        if bool(args.inline_emo_tag_mode)\n'
    '                        else None\n'
    '                    ),\n'
    '                    m0_worker_proc=None,'
)
print('found2:', norm.count(old2))
norm = norm.replace(old2, new2)

# Write back in original line ending
out = norm.replace('\n', '\r\n') if crlf else norm
with open(p, 'wb') as f:
    f.write(out.encode('utf-8'))
print('done; crlf=', crlf)
