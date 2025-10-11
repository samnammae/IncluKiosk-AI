""" 오디오 입출력 유틸리티 """
import os
import sys
import shutil
import tempfile
import time
from typing import Optional, List

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import queue

from .config import (
    STT_SAMPLE_RATE,
    STT_SILENCE_SEC,
    STT_MAX_DURATION,
    STT_CALIB_SEC,
    STT_SENSITIVITY,
    STT_MIN_SPEECH_SEC,
    STT_INITIAL_SILENCE_TIMEOUT,
    PREFERRED_DEVICE_KEYWORDS
)


def list_input_devices() -> List[dict]:
    """녹음 가능한 입력 장치 리스트"""
    devices = sd.query_devices()
    input_devs = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            input_devs.append({"index": idx, "name": d["name"], **d})
    return input_devs


def find_input_device_index(
    prefer_keywords: tuple = PREFERRED_DEVICE_KEYWORDS
) -> Optional[int]:
    """
    선호 키워드 기반으로 입력 장치를 자동 선택.
    없으면 기본(None)을 반환하여 sounddevice 기본장치 사용.
    """
    try:
        devices = list_input_devices()
        if not devices:
            return None
        lowered = [
            {
                **d,
                "lname": (d["name"] or "").lower(),
                "lhost": str(d.get("hostapi", "")).lower()
            }
            for d in devices
        ]
        for kw in prefer_keywords:
            for d in lowered:
                if kw.lower() in d["lname"] or kw.lower() in d["lhost"]:
                    return d["index"]
        best = max(lowered, key=lambda x: x.get("max_input_channels", 0))
        return best["index"]
    except Exception:
        return None


def record_audio(
    duration: int = 5,
    sample_rate: int = 16000,
    channels: int = 1,
    device: Optional[int] = None
) -> str:
    """duration초 동안 마이크 녹음하여 임시 WAV 파일 경로 반환"""
    if duration <= 0:
        raise ValueError("duration은 1 이상이어야 합니다.")
    if channels not in (1, 2):
        raise ValueError("channels는 1 또는 2만 지원합니다.")

    sd.default.samplerate = sample_rate
    if device is not None:
        sd.default.device = (device, None)

    frames = int(duration * sample_rate)
    print(f"[STT] (fixed) 녹음 시작: {duration}s @ {sample_rate}Hz (device={device})")
    audio = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="int16")
    sd.wait()
    print("[STT] 녹음 종료")

    if channels == 2:
        audio = audio.mean(axis=1, dtype=audio.dtype)

    tmpdir = tempfile.mkdtemp(prefix="stt_rec_")
    wav_path = os.path.join(tmpdir, "input.wav")
    wav_write(wav_path, sample_rate, audio)
    print(f"[STT] (fixed) WAV saved: {wav_path}  bytes={os.path.getsize(wav_path)}")
    return wav_path


def record_until_silence(
    sample_rate: int = STT_SAMPLE_RATE,
    device: Optional[int] = None,
    frame_ms: int = 30,
    silence_sec: float = STT_SILENCE_SEC,
    max_total_sec: float = STT_MAX_DURATION,
    calib_sec: float = STT_CALIB_SEC,
    sensitivity: float = STT_SENSITIVITY,
    min_speech_sec: float = STT_MIN_SPEECH_SEC,
    initial_silence_timeout: float = STT_INITIAL_SILENCE_TIMEOUT
) -> Optional[str]:
    """침묵 감지까지 자동 녹음, WAV 파일 경로 반환 (없으면 None)"""
    
    frames_per_block = max(1, int(sample_rate * frame_ms / 1000))
    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print(f"[STT] stream status: {status}", file=sys.stderr)
        q.put(indata.copy())

    print(f"[STT] (auto) 스트림 시작 @ {sample_rate}Hz, frame={frame_ms}ms, device={device}")
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=frames_per_block,
        device=device,
        callback=cb
    ):
        # 1) 주변 소음 기준치 측정
        calib_blocks = max(1, int(calib_sec * 1000 / frame_ms))
        baseline_vals = []
        start_t = time.time()
        while len(baseline_vals) < calib_blocks and (time.time() - start_t) < (calib_sec + 1.0):
            block = q.get()
            baseline_vals.append(float(np.mean(np.abs(block))))

        baseline = float(np.median(baseline_vals)) if baseline_vals else 50.0
        threshold = baseline * sensitivity + 100.0
        print(f"[STT] 기준치={baseline:.1f}, 임계값={threshold:.1f} (sens={sensitivity})")

        # 2) 발화 구간 녹음
        speech_started = False
        frames = []
        silence_frames_needed = max(1, int(silence_sec * 1000 / frame_ms))
        min_speech_frames = max(1, int(min_speech_sec * 1000 / frame_ms))
        consecutive_silence = 0
        initial_silence_frames = 0
        initial_silence_limit = max(1, int(initial_silence_timeout * 1000 / frame_ms))
        t0 = time.time()

        while True:
            elapsed = time.time() - t0
            if elapsed > max_total_sec:
                print("[STT] max_total_sec 도달로 종료")
                break

            block = q.get()
            amp = float(np.mean(np.abs(block)))
            is_voice = amp > threshold

            if not speech_started:
                if is_voice:
                    speech_started = True
                    frames.append(block)
                    print(f"[STT] ▶ 발화 시작 (t={elapsed:.2f}s, amp={amp:.1f})")
                else:
                    initial_silence_frames += 1
                    if initial_silence_frames >= initial_silence_limit:
                        print(f"[STT] ✖ 초기 침묵 {initial_silence_timeout:.1f}초 도달 → 음성 없음으로 종료")
                        return None
                    continue
            else:
                frames.append(block)
                if is_voice:
                    if consecutive_silence > 0:
                        print(f"[STT] 발성 재개 (consecutive_silence reset, {consecutive_silence} frames)")
                    consecutive_silence = 0
                else:
                    consecutive_silence += 1
                    if consecutive_silence % 10 == 0:
                        sec = consecutive_silence * frame_ms / 1000.0
                        print(f"[STT] ...침묵 누적 {sec:.2f}s / 필요 {silence_sec:.2f}s")

                    if consecutive_silence >= silence_frames_needed and len(frames) >= min_speech_frames:
                        print("[STT] ◀ 침묵 지속으로 종료")
                        break

        if not frames:
            print("[STT] 음성 구간이 탐지되지 않음")
            return None

        audio = np.concatenate(frames, axis=0)
        if audio.ndim > 1:
            audio = audio.reshape(-1,)

        tmpdir = tempfile.mkdtemp(prefix="stt_rec_")
        wav_path = os.path.join(tmpdir, "input.wav")
        wav_write(wav_path, sample_rate, audio.astype(np.int16))

        dur = len(audio) / float(sample_rate)
        print(f"[STT] (auto) WAV saved: {wav_path}  dur={dur:.2f}s  bytes={os.path.getsize(wav_path)}")
        return wav_path


def play_audio_file(path: str) -> None:
    """
    파일 확장자에 맞춰 재생.
    - .mp3: mpg123
    - .wav: aplay
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        player = shutil.which("mpg123")
        if player:
            os.system(f"{player} -q '{path}'")
        else:
            print("[TTS] 경고: mpg123가 없어 재생을 생략합니다.")
    elif ext == ".wav":
        player = shutil.which("aplay")
        if player:
            os.system(f"{player} -q '{path}'")
        else:
            print("[TTS] 경고: aplay가 없어 재생을 생략합니다.")
    else:
        print("[TTS] 알 수 없는 포맷, 재생 생략:", path)


def play_if_exists(path: Optional[str], pause_after: float = 0.25) -> None:
    """파일이 존재하면 재생하고 약간의 지연"""
    if not path:
        return
    if os.path.isfile(path):
        print(f"[SND] 프리사운드 재생: {path}")
        play_audio_file(path)
        if pause_after > 0:
            time.sleep(pause_after)
    else:
        print(f"[SND] 프리사운드 없음(건너뜀): {path}")