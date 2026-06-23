#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


RESPONSE_PREFIX = "__AUDIO_PLAYER_RESPONSE__ "


def _send(obj: dict) -> None:
    print(RESPONSE_PREFIX + json.dumps(obj, ensure_ascii=False), flush=True)


def _device_arg(device):
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        return device


class PersistentPcmPlayer:
    def __init__(self, *, sr: int, device, channels: int = 1) -> None:
        self.sr = int(sr)
        self.device = _device_arg(device)
        self.channels = int(channels)

        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.lock = threading.Lock()

        self.current: np.ndarray | None = None
        self.current_pos = 0

        self.played_samples = 0
        self.queued_samples = 0
        self.underrun_count = 0
        self.play_requests = 0
        self.last_underrun_log_t = 0.0

        self.stream = sd.OutputStream(
            samplerate=self.sr,
            device=self.device,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
            blocksize=0,
        )
        self.stream.start()

        print(
            "[audio_chunk_player_persistent][STREAM_START]",
            f"sr={self.sr}",
            f"device={self.device}",
            f"channels={self.channels}",
            flush=True,
        )

    def close(self) -> None:
        try:
            self.stream.stop()
        finally:
            self.stream.close()

    def enqueue(self, audio: np.ndarray) -> dict:
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        if audio.ndim != 1:
            audio = audio.reshape(-1)

        self.q.put(audio)

        with self.lock:
            self.play_requests += 1
            self.queued_samples += int(len(audio))
            qsize = self.q.qsize()

        return {
            "queued_samples": int(len(audio)),
            "queue_size": int(qsize),
            "played_samples": int(self.played_samples),
            "underrun_count": int(self.underrun_count),
        }

    def clear_queue(self) -> dict:
        cleared_chunks = 0
        cleared_samples = 0

        with self.lock:
            if self.current is not None:
                remaining = max(0, len(self.current) - int(self.current_pos))
                cleared_samples += int(remaining)

            self.current = None
            self.current_pos = 0

            while True:
                try:
                    chunk = self.q.get_nowait()
                except queue.Empty:
                    break

                cleared_chunks += 1
                cleared_samples += int(len(chunk))

            qsize = self.q.qsize()

        return {
            "ok": True,
            "cmd": "clear_queue",
            "cleared_chunks": int(cleared_chunks),
            "cleared_samples": int(cleared_samples),
            "queue_size": int(qsize),
            "played_samples": int(self.played_samples),
            "underrun_count": int(self.underrun_count),
        }

    def _next_chunk(self) -> np.ndarray | None:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def _callback(self, outdata, frames: int, time_info, status) -> None:
        if status:
            print(
                "[audio_chunk_player_persistent][STATUS]",
                str(status),
                flush=True,
            )

        out = np.zeros(int(frames), dtype=np.int16)
        filled = 0

        while filled < frames:
            if self.current is None or self.current_pos >= len(self.current):
                self.current = self._next_chunk()
                self.current_pos = 0

                if self.current is None:
                    with self.lock:
                        self.underrun_count += 1
                        underrun_count = self.underrun_count

                    now = time.monotonic()
                    if now - self.last_underrun_log_t >= 1.0:
                        self.last_underrun_log_t = now
                        print(
                            "[audio_chunk_player_persistent][UNDERRUN]",
                            f"count={underrun_count}",
                            f"filled={filled}",
                            f"frames={frames}",
                            flush=True,
                        )
                    break

            remain_out = frames - filled
            remain_cur = len(self.current) - self.current_pos
            n = min(remain_out, remain_cur)

            out[filled : filled + n] = self.current[
                self.current_pos : self.current_pos + n
            ]

            self.current_pos += n
            filled += n

        outdata[:, 0] = out

        with self.lock:
            self.played_samples += int(filled)


def _load_wav_as_int16_mono(path: Path, *, target_sr: int) -> np.ndarray:
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"missing wav: {path}")

    with wave.open(str(path), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_width = int(wf.getsampwidth())
        sr = int(wf.getframerate())
        frames = int(wf.getnframes())
        raw = wf.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"only 16bit PCM wav supported: {path} sampwidth={sample_width}")

    if sr != int(target_sr):
        raise ValueError(f"wav sr mismatch: path={path} sr={sr} target_sr={target_sr}")

    audio = np.frombuffer(raw, dtype=np.int16)

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

    return audio.reshape(-1)


def _play_pcm_chunk(
    *,
    player: PersistentPcmPlayer,
    pcm: str,
    sr: int,
    device,
    chunk_ms: int,
    start_chunk: int,
    req_is_single_chunk: bool = False,
) -> dict:
    pcm_path = Path(pcm).resolve()
    if not pcm_path.exists():
        raise FileNotFoundError(pcm_path)

    audio = np.frombuffer(pcm_path.read_bytes(), dtype=np.int16)

    samples_per_chunk = int(sr * chunk_ms / 1000)

    if bool(req_is_single_chunk):
        chunk_audio = audio
    else:
        start = int(start_chunk) * samples_per_chunk
        end = min(len(audio), start + samples_per_chunk)
        chunk_audio = audio[start:end]

    if len(chunk_audio) == 0:
        return {
            "ok": True,
            "cmd": "play",
            "chunk": int(start_chunk),
            "samples": 0,
            "skipped": True,
        }

    stats = player.enqueue(chunk_audio)

    return {
        "ok": True,
        "cmd": "play",
        "chunk": int(start_chunk),
        "samples": int(len(chunk_audio)),
        "device": device,
        **stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--chunk_ms", type=int, default=400)
    args = ap.parse_args()

    player = PersistentPcmPlayer(
        sr=int(args.sr),
        device=args.device,
        channels=1,
    )

    print("[audio_chunk_player_persistent][READY]", flush=True)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                req = json.loads(line)
                cmd = req.get("cmd")

                if cmd == "quit":
                    _send({"ok": True, "cmd": "quit"})
                    return 0

                if cmd == "clear_queue":
                    _send(player.clear_queue())
                    continue

                if cmd == "play_wav":
                    wav_path = Path(str(req.get("path", ""))).resolve()
                    audio = _load_wav_as_int16_mono(
                        wav_path,
                        target_sr=int(req.get("sr", args.sr)),
                    )

                    stats = player.enqueue(audio)

                    _send({
                        "ok": True,
                        "cmd": "play_wav",
                        "path": str(wav_path),
                        "chunk": int(req.get("chunk_id", -1)),
                        "samples": int(len(audio)),
                        "device": req.get("device", args.device),
                        **stats,
                    })
                    continue

                if cmd != "play":
                    raise ValueError(f"unknown cmd: {cmd}")

                res = _play_pcm_chunk(
                    player=player,
                    pcm=req["pcm"],
                    sr=int(req.get("sr", args.sr)),
                    device=req.get("device", args.device),
                    chunk_ms=int(req.get("chunk_ms", args.chunk_ms)),
                    start_chunk=int(req.get("start_chunk", 0)),
                    req_is_single_chunk=bool(req.get("single_chunk", False)),
                )
                _send(res)

            except Exception as e:
                _send({
                    "ok": False,
                    "error": str(e),
                })

    finally:
        player.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
