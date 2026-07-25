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
    """PCM player with data-amount jitter (not fixed-sleep wait).

    Distinct knobs (do not conflate with M0 hang timeout in session_loop/step1):
      - initial_buffer_ms: queued audio before first playout (target ~300-500ms)
      - start_fallback_ms: wall-clock force-start if target not reached (~1000ms)
      - rebuffer_target_ms: refill target after active-playback underrun
      - min_start_pcm_ms: ignore sub-threshold PCM for queue/clock/fallback (Hotfix)
    """

    STATE_BUFFERING = "BUFFERING"
    STATE_PLAYING = "PLAYING"
    STATE_REBUFFERING = "REBUFFERING"

    def __init__(
        self,
        *,
        sr: int,
        device,
        channels: int = 1,
        initial_buffer_ms: int = 300,
        start_fallback_ms: int = 1000,
        rebuffer_target_ms: int = 240,
        min_start_pcm_ms: int = 20,
    ) -> None:
        self.sr = int(sr)
        self.device = _device_arg(device)
        self.channels = int(channels)

        self.initial_buffer_ms = max(0, int(initial_buffer_ms))
        self.start_fallback_ms = max(0, int(start_fallback_ms))
        self.rebuffer_target_ms = max(0, int(rebuffer_target_ms))
        self.min_start_pcm_ms = max(0, int(min_start_pcm_ms))

        self.initial_buffer_samples = int(
            self.sr * self.initial_buffer_ms / 1000
        )
        self.rebuffer_target_samples = int(
            self.sr * self.rebuffer_target_ms / 1000
        )
        # Sub-threshold PCM must not arm BUFFERING clock / start_fallback.
        self.min_start_pcm_samples = int(
            self.sr * self.min_start_pcm_ms / 1000
        )
        if self.min_start_pcm_ms > 0 and self.min_start_pcm_samples < 1:
            self.min_start_pcm_samples = 1

        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.lock = threading.Lock()

        self.current: np.ndarray | None = None
        self.current_pos = 0

        self.played_samples = 0
        self.queued_samples = 0
        self.play_requests = 0

        # Unplayed samples: remaining in current + all queued chunks.
        self.pending_samples = 0

        self.state = self.STATE_BUFFERING
        self.playout_started_at: float | None = None
        # Wall clock starts on first enqueue while BUFFERING (not stream open).
        self.buffering_started_at: float | None = None

        self.active_playback_underrun_count = 0
        self.rebuffer_count = 0
        # Alias for older response consumers / log greps.
        self.underrun_count = 0

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
            f"initial_buffer_ms={self.initial_buffer_ms}",
            f"initial_buffer_samples={self.initial_buffer_samples}",
            f"start_fallback_ms={self.start_fallback_ms}",
            f"rebuffer_target_ms={self.rebuffer_target_ms}",
            f"rebuffer_target_samples={self.rebuffer_target_samples}",
            f"min_start_pcm_ms={self.min_start_pcm_ms}",
            f"min_start_pcm_samples={self.min_start_pcm_samples}",
            flush=True,
        )

        print(
            "[audio_chunk_player_persistent][BUFFERING]",
            "reason=stream_start",
            "pending_samples=0",
            "pending_ms=0.000",
            f"target_ms={self.initial_buffer_ms}",
            f"start_fallback_ms={self.start_fallback_ms}",
            f"min_start_pcm_ms={self.min_start_pcm_ms}",
            flush=True,
        )

    def _samples_to_ms(self, samples: int) -> float:
        return float(samples) * 1000.0 / float(self.sr)

    def _stats_locked(self) -> dict:
        pending_samples = max(0, int(self.pending_samples))
        return {
            "state": str(self.state),
            "pending_samples": pending_samples,
            "pending_ms": round(self._samples_to_ms(pending_samples), 3),
            "played_samples": int(self.played_samples),
            "queued_samples_total": int(self.queued_samples),
            "play_requests": int(self.play_requests),
            "active_playback_underrun_count": int(
                self.active_playback_underrun_count
            ),
            "underrun_count": int(self.underrun_count),
            "rebuffer_count": int(self.rebuffer_count),
            "playout_started_at": self.playout_started_at,
            "initial_buffer_ms": int(self.initial_buffer_ms),
            "start_fallback_ms": int(self.start_fallback_ms),
            "rebuffer_target_ms": int(self.rebuffer_target_ms),
            "min_start_pcm_ms": int(self.min_start_pcm_ms),
        }

    def _emit_playout_transition(self, transition: dict, stats: dict) -> None:
        print(
            f"[audio_chunk_player_persistent][{transition['log']}]",
            f"reason={transition['reason']}",
            f"pending_samples={stats['pending_samples']}",
            f"pending_ms={stats['pending_ms']:.3f}",
            f"target_ms={transition['target_ms']}",
            f"start_fallback_ms={self.start_fallback_ms}",
            f"rebuffer_count={stats['rebuffer_count']}",
            flush=True,
        )

    def _enter_playing_locked(self, *, reason: str, target_ms: int) -> dict:
        self.state = self.STATE_PLAYING
        if self.playout_started_at is None:
            self.playout_started_at = time.monotonic()
            log_name = "PLAYOUT_START"
        else:
            log_name = "PLAYING"
        return {
            "log": log_name,
            "reason": reason,
            "target_ms": int(target_ms),
        }

    def _maybe_start_from_fallback_locked(self) -> dict | None:
        """Force PLAYING if partial *effective* data waited too long.

        Guard: never start with near-zero pending (tiny-PCM false arm).
        """
        if self.state != self.STATE_BUFFERING:
            return None
        if self.start_fallback_ms <= 0:
            return None
        if self.buffering_started_at is None:
            return None
        # C: reject fallback when pending is below meaningful floor.
        if self.pending_samples < self.min_start_pcm_samples:
            return None

        elapsed_ms = (time.monotonic() - self.buffering_started_at) * 1000.0
        if elapsed_ms < float(self.start_fallback_ms):
            return None

        return self._enter_playing_locked(
            reason="start_fallback",
            target_ms=self.start_fallback_ms,
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

        audio = np.ascontiguousarray(audio)
        added_samples = int(len(audio))

        # A: drop sub-threshold PCM — do not queue, do not arm clock.
        if (
            self.min_start_pcm_samples > 0
            and added_samples > 0
            and added_samples < self.min_start_pcm_samples
        ):
            with self.lock:
                stats = self._stats_locked()
                qsize = self.q.qsize()
            print(
                "[audio_chunk_player_persistent][TINY_PCM_DROP]",
                f"samples={added_samples}",
                f"pending_ms={self._samples_to_ms(added_samples):.3f}",
                f"min_start_pcm_ms={self.min_start_pcm_ms}",
                f"min_start_pcm_samples={self.min_start_pcm_samples}",
                f"state={stats['state']}",
                flush=True,
            )
            return {
                "queued_samples": 0,
                "queue_size": int(qsize),
                "dropped_tiny_pcm": True,
                "dropped_samples": int(added_samples),
                **stats,
            }

        transition: dict | None = None

        with self.lock:
            self.q.put(audio)

            self.play_requests += 1
            self.queued_samples += added_samples
            self.pending_samples += added_samples

            if self.state == self.STATE_BUFFERING:
                # B: clock starts only when effective PCM is first queued.
                if (
                    self.buffering_started_at is None
                    and added_samples >= self.min_start_pcm_samples
                ):
                    self.buffering_started_at = time.monotonic()

                if self.pending_samples >= self.initial_buffer_samples:
                    transition = self._enter_playing_locked(
                        reason="initial_buffer_ready",
                        target_ms=self.initial_buffer_ms,
                    )
                else:
                    transition = self._maybe_start_from_fallback_locked()

            elif (
                self.state == self.STATE_REBUFFERING
                and self.pending_samples >= self.rebuffer_target_samples
            ):
                transition = self._enter_playing_locked(
                    reason="rebuffer_target_ready",
                    target_ms=self.rebuffer_target_ms,
                )

            qsize = self.q.qsize()
            stats = self._stats_locked()

        if transition is not None:
            self._emit_playout_transition(transition, stats)

        return {
            "queued_samples": int(added_samples),
            "queue_size": int(qsize),
            **stats,
        }

    def clear_queue(self, *, reason: str = "clear_queue") -> dict:
        """Clear pending PCM. Intended for user-interrupt exception only.

        After clear, state returns to BUFFERING so the next enqueue path can
        rebuild the data-amount jitter (initial_buffer / TINY_PCM_DROP intact).
        """
        cleared_chunks = 0
        cleared_samples = 0
        reason_s = str(reason or "clear_queue")

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

            self.pending_samples = 0
            self.state = self.STATE_BUFFERING
            self.playout_started_at = None
            self.buffering_started_at = None

            qsize = self.q.qsize()
            stats = self._stats_locked()

        print(
            "[audio_chunk_player_persistent][BUFFERING]",
            f"reason=clear_queue:{reason_s}",
            "pending_samples=0",
            "pending_ms=0.000",
            f"target_ms={self.initial_buffer_ms}",
            f"start_fallback_ms={self.start_fallback_ms}",
            f"min_start_pcm_ms={self.min_start_pcm_ms}",
            flush=True,
        )

        return {
            "ok": True,
            "cmd": "clear_queue",
            "reason": reason_s,
            "cleared_chunks": int(cleared_chunks),
            "cleared_samples": int(cleared_samples),
            "queue_size": int(qsize),
            **stats,
        }

    def get_status(self) -> dict:
        with self.lock:
            qsize = self.q.qsize()
            current_remaining_samples = 0
            if self.current is not None:
                current_remaining_samples = max(
                    0,
                    int(len(self.current)) - int(self.current_pos),
                )
            stats = self._stats_locked()

        return {
            "ok": True,
            "cmd": "get_status",
            "queue_size": int(qsize),
            "current_remaining_samples": int(current_remaining_samples),
            "drained": bool(
                int(stats["pending_samples"]) == 0
                and int(qsize) == 0
                and int(current_remaining_samples) == 0
            ),
            **stats,
        }

    def _next_chunk_locked(self) -> np.ndarray | None:
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

        frame_count = int(frames)
        out = np.zeros(frame_count, dtype=np.int16)
        filled = 0
        rebuffer_log: dict | None = None
        fallback_transition: dict | None = None
        fallback_stats: dict | None = None

        with self.lock:
            if self.state == self.STATE_BUFFERING:
                # Data-amount wait: output silence; do NOT count as underrun.
                # Fallback may fire here if enqueues stopped after partial fill.
                fallback_transition = self._maybe_start_from_fallback_locked()
                if fallback_transition is not None:
                    fallback_stats = self._stats_locked()

            if self.state == self.STATE_PLAYING:
                while filled < frame_count:
                    if self.current is None or self.current_pos >= len(
                        self.current
                    ):
                        self.current = self._next_chunk_locked()
                        self.current_pos = 0

                        if self.current is None:
                            self.state = self.STATE_REBUFFERING
                            self.active_playback_underrun_count += 1
                            self.underrun_count = (
                                self.active_playback_underrun_count
                            )
                            self.rebuffer_count += 1
                            rebuffer_log = {
                                "pending_samples": max(
                                    0, int(self.pending_samples)
                                ),
                                "filled": int(filled),
                                "frames": int(frame_count),
                                "active_playback_underrun_count": int(
                                    self.active_playback_underrun_count
                                ),
                                "rebuffer_count": int(self.rebuffer_count),
                            }
                            break

                    remain_out = frame_count - filled
                    remain_cur = len(self.current) - self.current_pos
                    n = min(remain_out, remain_cur)

                    out[filled : filled + n] = self.current[
                        self.current_pos : self.current_pos + n
                    ]
                    self.current_pos += n
                    filled += n
                    self.pending_samples = max(
                        0, int(self.pending_samples) - int(n)
                    )
                    self.played_samples += int(n)

            # BUFFERING / REBUFFERING: leave zeros (silence). Not idle underrun spam.

        outdata[:, 0] = out

        if fallback_transition is not None and fallback_stats is not None:
            self._emit_playout_transition(fallback_transition, fallback_stats)

        if rebuffer_log is not None:
            pending_ms = self._samples_to_ms(rebuffer_log["pending_samples"])
            print(
                "[audio_chunk_player_persistent][REBUFFERING]",
                "reason=active_playback_depleted",
                f"pending_samples={rebuffer_log['pending_samples']}",
                f"pending_ms={pending_ms:.3f}",
                f"filled={rebuffer_log['filled']}",
                f"frames={rebuffer_log['frames']}",
                f"target_ms={self.rebuffer_target_ms}",
                f"active_playback_underrun_count="
                f"{rebuffer_log['active_playback_underrun_count']}",
                f"rebuffer_count={rebuffer_log['rebuffer_count']}",
                flush=True,
            )
            # Keep grep-compatible underrun signal for continuous-play checks.
            print(
                "[audio_chunk_player_persistent][UNDERRUN]",
                f"count={rebuffer_log['active_playback_underrun_count']}",
                f"filled={rebuffer_log['filled']}",
                f"frames={rebuffer_log['frames']}",
                "phase=active_playback",
                flush=True,
            )


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
        raise ValueError(
            f"only 16bit PCM wav supported: {path} sampwidth={sample_width}"
        )

    if sr != int(target_sr):
        raise ValueError(
            f"wav sr mismatch: path={path} sr={sr} target_sr={target_sr}"
        )

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
    ap.add_argument(
        "--initial_buffer_ms",
        type=int,
        default=300,
        help="Data-amount jitter before first playout (M0-PNG-equivalent audio ms).",
    )
    ap.add_argument(
        "--start_fallback_ms",
        type=int,
        default=1000,
        help="Force playout if initial buffer not reached (avoid infinite wait).",
    )
    ap.add_argument(
        "--rebuffer_target_ms",
        type=int,
        default=240,
        help="Data-amount target to resume after active-playback underrun.",
    )
    ap.add_argument(
        "--min_start_pcm_ms",
        type=int,
        default=20,
        help=(
            "Ignore PCM shorter than this for queue/clock/fallback "
            "(drops tiny leading slices like samples=1)."
        ),
    )
    args = ap.parse_args()

    if int(args.initial_buffer_ms) < 0:
        raise SystemExit("--initial_buffer_ms must be >= 0")
    if int(args.start_fallback_ms) < 0:
        raise SystemExit("--start_fallback_ms must be >= 0")
    if int(args.rebuffer_target_ms) < 0:
        raise SystemExit("--rebuffer_target_ms must be >= 0")
    if int(args.min_start_pcm_ms) < 0:
        raise SystemExit("--min_start_pcm_ms must be >= 0")

    player = PersistentPcmPlayer(
        sr=int(args.sr),
        device=args.device,
        channels=1,
        initial_buffer_ms=int(args.initial_buffer_ms),
        start_fallback_ms=int(args.start_fallback_ms),
        rebuffer_target_ms=int(args.rebuffer_target_ms),
        min_start_pcm_ms=int(args.min_start_pcm_ms),
    )

    print(
        "[audio_chunk_player_persistent][READY]",
        f"state={player.state}",
        f"initial_buffer_ms={player.initial_buffer_ms}",
        f"start_fallback_ms={player.start_fallback_ms}",
        f"rebuffer_target_ms={player.rebuffer_target_ms}",
        f"min_start_pcm_ms={player.min_start_pcm_ms}",
        flush=True,
    )

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
                    _send(
                        player.clear_queue(
                            reason=str(req.get("reason", "clear_queue"))
                        )
                    )
                    continue

                if cmd == "get_status":
                    _send(player.get_status())
                    continue

                if cmd == "play_wav":
                    wav_path = Path(str(req.get("path", ""))).resolve()
                    audio = _load_wav_as_int16_mono(
                        wav_path,
                        target_sr=int(req.get("sr", args.sr)),
                    )
                    stats = player.enqueue(audio)
                    _send(
                        {
                            "ok": True,
                            "cmd": "play_wav",
                            "path": str(wav_path),
                            "chunk": int(req.get("chunk_id", -1)),
                            "samples": int(len(audio)),
                            "device": req.get("device", args.device),
                            **stats,
                        }
                    )
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
                _send({"ok": False, "error": str(e)})

    finally:
        player.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
