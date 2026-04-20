#!/usr/bin/env python3
from pathlib import Path
import json
import shutil

src = Path("/workspaces/M1_LLM_To_M2_TTS_united/out/dev_m1_stub/m1_unified_output_9_2_long_end.json")
inbox = Path("/workspaces/M1_LLM_To_M2_TTS_united/out/runtime_watch_inbox")

tmp_path = inbox / "test_010.part.json"
final_path = inbox / "test_010.json"

obj = json.loads(src.read_text(encoding="utf-8"))

tmp_path.write_text(
    json.dumps(obj, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# 完成後に rename
shutil.move(str(tmp_path), str(final_path))

print("[OK]", final_path)