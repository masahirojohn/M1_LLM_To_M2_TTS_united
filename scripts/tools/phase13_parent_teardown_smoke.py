#!/usr/bin/env python3
"""Phase 13 short: parent death must tear down M0 worker children."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path


def main() -> int:
    m0_repo = Path(r"C:\dev\M0_session_renderer_final_1")
    worker = m0_repo / "src" / "m0_persistent_worker.py"
    py = Path(r"C:\dev\M1_LLM_To_M2_TTS_united\.venv\Scripts\python.exe")
    port = 39451

    parent = subprocess.Popen(
        [str(py), "-c", "import time; time.sleep(60)"],
    )
    child = subprocess.Popen(
        [
            str(py),
            str(worker),
            "--tcp",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--parent_pid",
            str(parent.pid),
        ],
        cwd=str(m0_repo),
    )
    time.sleep(1.2)
    if child.poll() is not None:
        print("PARENT_TEARDOWN: FAIL worker_exited_early", child.returncode)
        parent.kill()
        return 1

    parent.kill()
    try:
        parent.wait(timeout=3)
    except Exception:
        pass

    deadline = time.time() + 6.0
    while time.time() < deadline:
        if child.poll() is not None:
            print(
                "PARENT_TEARDOWN: PASS",
                f"worker_pid={child.pid}",
                f"rc={child.returncode}",
            )
            return 0
        time.sleep(0.2)

    child.kill()
    print("PARENT_TEARDOWN: FAIL worker_still_alive", child.pid)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
