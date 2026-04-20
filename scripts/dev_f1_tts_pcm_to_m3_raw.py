#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

from aituber_llm_tts.config import load_config
from aituber_llm_tts.tts_client import RealTTSClient
from aituber_llm_tts.m1_unified_adapter import unified_to_legacy_m1output

# M3側（PYTHONPATHでM3/srcを通すこと）
from m3p.live.mouth_streamer_oc import MouthStreamerOC, MouthOCConfig


def project_streamer_to_raw(streamer_json_path: Path) -> dict:
    data = json.loads(streamer_json_path.read_text(encoding="utf-8"))
    frames_in = data.get("frames", [])
    frames_out = []
    for fr in frames_in:
        frames_out.append(
            {
                "t_ms": fr.get("t_ms"),
                "vad_active": int(fr.get("vad_active", 0) or 0),
                "f1_hz": fr.get("f1_hz", None),
                "f2_hz": fr.get("f2_hz", None),
                "src": "mouth_streamer_oc",
            }
        )
    return {
        "version": "m3p.mouth.timeline.v1",
        "step_ms": int(data.get("step_ms", 40)),
        "frames": frames_out,
        "meta": data.get("meta", {}),
    }


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _ceil_div(a: int, b: int) -> int:
    return int((a + b - 1) // b)


def _align_end_stream_events_to_audio_ms(
    events: list[dict],
    *,
    audio_ms: int,
    step_ms: int,
) -> list[dict]:
    """
    end_stream event の t_ms を audio_ms の40msグリッド切り上げ位置へ補正する。
    他の event はそのまま保持。
    """
    aligned_t_ms = _ceil_div(int(audio_ms), int(step_ms)) * int(step_ms)

    out = []
    for ev in events:
        if not isinstance(ev, dict):
            continue

        ev2 = dict(ev)
        if str(ev2.get("event_mode", "")) == "end_stream":
            ev2["t_ms"] = int(aligned_t_ms)
            ev2["duration_ms"] = 0
        out.append(ev2)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    # config / input
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--unified_json", required=True)
    ap.add_argument("--out_dir", required=True)

    # identity
    ap.add_argument("--session_id", default="")
    ap.add_argument("--utt_id", default="")

    # SSOT / M3
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--analysis_sr", type=int, default=16000)
    ap.add_argument("--vad_energy_thr", type=float, default=0.0004)

    # Phase5 / backpressure
    ap.add_argument("--max_buffer_ms", type=int, default=3000)
    ap.add_argument("--sleep_consumer_ms", type=int, default=0)

    # outputs
    ap.add_argument("--out_legacy_m1_json", default="m1_legacy.json")
    ap.add_argument("--out_events_json", default="events.json")
    ap.add_argument("--out_obs_actions_json", default="obs_actions.json")
    ap.add_argument("--out_streamer_json", default="streamer.json")
    ap.add_argument("--out_raw_json", default="mouth_timeline.formant.raw.json")
    ap.add_argument("--out_audio_meta_json", default="audio_meta.json")
    ap.add_argument("--out_profile_json", default="profile.json")
    args = ap.parse_args()

    prof = {
        "format": "phase5_profile.v1",
        "step_ms": int(args.step_ms),
        "max_buffer_ms": int(args.max_buffer_ms),
        "sleep_consumer_ms": int(args.sleep_consumer_ms),
        "t_start_ms": _now_ms(),
        "input_mode": "unified_json",
    }

    cfg = load_config(args.config)
    tts = RealTTSClient(cfg)

    unified_path = Path(args.unified_json)
    unified = json.loads(unified_path.read_text(encoding="utf-8"))

    session_id = args.session_id or str(unified.get("session_id", "sess_real_01"))
    utt_id = args.utt_id or str(unified.get("utt_id", "utt_0001"))

    # unified -> legacy M1Output
    m1 = unified_to_legacy_m1output(
        unified,
        schema_version=cfg.schema_version,
        mode=cfg.llm.mode,
    )

    # 念のためCLI引数優先で上書き
    m1.session_id = session_id
    m1.utt_id = utt_id

    pcm_bytes, sample_rate, audio_ms = tts.synthesize_pcm_bytes(m1)
    prof["t_tts_done_ms"] = _now_ms()

    # eventのt_ms補正
    unified["events"] = _align_end_stream_events_to_audio_ms(
        unified.get("events", []),
        audio_ms=int(audio_ms),
        step_ms=int(args.step_ms),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy_m1_json = out_dir / args.out_legacy_m1_json
    events_json = out_dir / args.out_events_json
    obs_actions_json = out_dir / args.out_obs_actions_json
    streamer_json = out_dir / args.out_streamer_json
    raw_json = out_dir / args.out_raw_json
    audio_meta_json = out_dir / args.out_audio_meta_json
    profile_json = out_dir / args.out_profile_json

    legacy_m1_json.write_text(
        json.dumps(m1.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    events_json.write_text(
        json.dumps(unified.get("events", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    obs_actions_json.write_text(
        json.dumps(unified.get("obs_actions", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # M3 streamer init
    m3_cfg = MouthOCConfig(
        step_ms=int(args.step_ms),
        analysis_sr=int(args.analysis_sr),
        input_sr_default=int(sample_rate),
        vad_energy_thr=float(args.vad_energy_thr),
    )
    ms = MouthStreamerOC(
        out_json=str(streamer_json),
        session_id=str(session_id),
        cfg=m3_cfg,
    )

    # PCM16 mono -> 40ms push
    bytes_per_sample = 2
    chunk_samples = int(sample_rate * int(args.step_ms) / 1000)
    chunk_bytes = chunk_samples * bytes_per_sample

    pcm_bytes = pcm_bytes[: (len(pcm_bytes) // 2) * 2]

    max_chunks = max(1, _ceil_div(int(args.max_buffer_ms), int(args.step_ms)))
    q: "queue.Queue[bytes | None]" = queue.Queue(maxsize=max_chunks)
    max_put_wait_s = max(0.001, float(int(args.max_buffer_ms)) / 1000.0)

    stats = {
        "max_queue_size": int(max_chunks),
        "max_queue_seen": 0,
        "chunks_enqueued": 0,
        "chunks_pushed": 0,
    }

    def consumer() -> None:
        first_push = True
        sleep_s = max(0.0, float(int(args.sleep_consumer_ms)) / 1000.0)
        while True:
            b = q.get()
            try:
                if b is None:
                    return
                ms.push_pcm16_mono(b, input_sr=int(sample_rate))
                stats["chunks_pushed"] += 1
                if first_push:
                    prof["t_first_push_ms"] = _now_ms()
                    first_push = False
                if sleep_s > 0:
                    time.sleep(sleep_s)
            finally:
                q.task_done()

    th = threading.Thread(target=consumer, daemon=True)
    th.start()

    first_enqueue = True
    for off in range(0, len(pcm_bytes), chunk_bytes):
        b = pcm_bytes[off : off + chunk_bytes]
        if not b:
            break
        try:
            q.put(b, timeout=max_put_wait_s)
        except queue.Full:
            raise RuntimeError(
                f"Backpressure: queue stayed full > {int(args.max_buffer_ms)}ms "
                f"(max_chunks={max_chunks}, step_ms={int(args.step_ms)}). "
                "Consumer is too slow or stalled."
            )
        stats["chunks_enqueued"] += 1
        if first_enqueue:
            prof["t_first_enqueue_ms"] = _now_ms()
            first_enqueue = False
        qs = q.qsize()
        if qs > stats["max_queue_seen"]:
            stats["max_queue_seen"] = qs

    q.put(None, timeout=max_put_wait_s)
    q.join()
    th.join(timeout=10.0)

    ms.finalize()
    prof["t_finalize_done_ms"] = _now_ms()

    audio_meta = {
        "format": "audio_meta.v1",
        "session_id": str(session_id),
        "utt_id": str(utt_id),
        "step_ms": int(args.step_ms),
        "sample_rate": int(sample_rate),
        "audio_ms": int(audio_ms),
    }
    audio_meta_json.write_text(
        json.dumps(audio_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # M2 SSOT: audio_ms を別JSONに書き出す
    audio_meta = {
        "format": "audio_meta.v1",
        "session_id": str(args.session_id),
        "utt_id": str(args.utt_id),
        "step_ms": int(args.step_ms),
        "sample_rate": int(sample_rate),
        "audio_ms": int(audio_ms),
    }
    audio_meta_json.write_text(json.dumps(audio_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_obj = project_streamer_to_raw(streamer_json)
    raw_json.write_text(
        json.dumps(raw_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prof["session_id"] = str(session_id)
    prof["utt_id"] = str(utt_id)
    prof["sample_rate"] = int(sample_rate)
    prof["audio_ms"] = int(audio_ms)
    prof["speech_text"] = str(unified.get("speech", {}).get("text", ""))
    prof["event_count"] = len(unified.get("events", []))
    prof["obs_action_count"] = len(unified.get("obs_actions", []))
    prof["stats"] = stats
    profile_json.write_text(
        json.dumps(prof, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[OK] session_id =", session_id)
    print("[OK] utt_id =", utt_id)
    print("[OK] speech =", unified.get("speech", {}).get("text", ""))
    print("[OK] sample_rate =", sample_rate)
    print("[OK] audio_ms (M2 SSOT) =", audio_ms)
    print("[OK] wrote:", legacy_m1_json.as_posix())
    print("[OK] wrote:", events_json.as_posix())
    print("[OK] wrote:", obs_actions_json.as_posix())
    print("[OK] wrote:", audio_meta_json.as_posix())
    print("[OK] wrote:", raw_json.as_posix())
    print("[OK] wrote:", profile_json.as_posix())


if __name__ == "__main__":
    main()