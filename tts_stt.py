# tts_stt.py
# --------------------------------------------
# Raspberry Pi용 TTS/STT 유틸리티 (모듈 + CLI 겸용)
# - tts_play(text, lang="ko-KR", voice=..., speaking_rate=1.0, pitch=0.0)
# - stt_once(mode="auto", duration=5, sample_rate=16000, language_code="ko-KR", device=None, ...)
# --------------------------------------------
import os
import sys
import json
import time
import shutil
import tempfile
from typing import Optional, List

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import queue

# 기본 안내 멘트(프로젝트 명세 반영)
DEFAULT_CHAT_GUIDE = "안녕하세요. 음성으로 주문을 도와드릴게요."

# Google Cloud SDKs
try:
    from google.cloud import texttospeech
    from google.cloud import speech
except Exception as e:
    raise RuntimeError(
        "google-cloud-texttospeech, google-cloud-speech 패키지를 설치하세요.\n"
        "pip install google-cloud-texttospeech google-cloud-speech"
    ) from e


# =========================
# 오디오 입출력 유틸
# =========================
def list_input_devices() -> List[dict]:
    """녹음 가능한 입력 장치 리스트"""
    devices = sd.query_devices()
    input_devs = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            input_devs.append({"index": idx, "name": d["name"], **d})
    return input_devs


def find_input_device_index(prefer_keywords: tuple = ("usb", "mic", "seeed", "respeaker")) -> Optional[int]:
    """
    선호 키워드 기반으로 입력 장치를 자동 선택.
    없으면 기본(None)을 반환하여 sounddevice 기본장치 사용.
    """
    try:
        devices = list_input_devices()
        if not devices:
            return None
        lowered = [({**d, "lname": (d["name"] or "").lower(), "lhost": str(d.get("hostapi", "")).lower()})
                   for d in devices]
        for kw in prefer_keywords:
            for d in lowered:
                if kw.lower() in d["lname"] or kw.lower() in d["lhost"]:
                    return d["index"]
        best = max(lowered, key=lambda x: x.get("max_input_channels", 0))
        return best["index"]
    except Exception:
        return None


def record_audio(duration: int = 5, sample_rate: int = 16000, channels: int = 1, device: Optional[int] = None) -> str:
    """
    duration초 동안 마이크 녹음하여 임시 WAV 파일 경로 반환
    """
    if duration <= 0:
        raise ValueError("duration은 1 이상이어야 합니다.")
    if channels not in (1, 2):
        raise ValueError("channels는 1 또는 2만 지원합니다.")

    sd.default.samplerate = sample_rate
    if device is not None:
        sd.default.device = (device, None)  # (input, output)

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
    return wav_path


def record_until_silence(
    sample_rate: int = 16000,
    device: Optional[int] = None,
    frame_ms: int = 30,
    silence_sec: float = 0.8,
    max_total_sec: float = 15.0,
    calib_sec: float = 0.4,
    sensitivity: float = 2.0,
    min_speech_sec: float = 0.3,
) -> Optional[str]:
    """
    사용자가 말하는 동안 녹음하고, 침묵(silence_sec)이 지속되면 자동 종료.
    - frame_ms: 프레임 길이(ms) (30~40ms 권장)
    - silence_sec: 이 시간 동안 연속 침묵이면 종료
    - max_total_sec: 총 녹음 상한
    - calib_sec: 시작 전 주변소음 기준치 측정 시간
    - sensitivity: 기준치 * sensitivity 보다 크면 발성으로 판단
    - min_speech_sec: 최소 발화 시간(너무 짧은 호흡/노이즈 방지)
    반환: 임시 WAV 파일 경로 (없으면 None)
    """
    frames_per_block = max(1, int(sample_rate * frame_ms / 1000))
    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print(f"[STT] stream status: {status}", file=sys.stderr)
        q.put(indata.copy())

    # 입력 스트림 열기
    print(f"[STT] (auto) 스트림 시작 @ {sample_rate}Hz, frame={frame_ms}ms, device={device}")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                        blocksize=frames_per_block, device=device, callback=cb):
        # 1) 주변 소음 기준치 측정
        calib_blocks = max(1, int(calib_sec * 1000 / frame_ms))
        baseline_vals = []
        start_t = time.time()
        while len(baseline_vals) < calib_blocks and (time.time() - start_t) < (calib_sec + 1.0):
            block = q.get()
            baseline_vals.append(float(np.mean(np.abs(block))))

        baseline = float(np.median(baseline_vals)) if baseline_vals else 50.0
        threshold = baseline * sensitivity + 100.0  # 바닥잡기용 offset
        print(f"[STT] 기준치={baseline:.1f}, 임계값={threshold:.1f} (sens={sensitivity})")

        # 2) 발화 구간 녹음
        speech_started = False
        frames = []
        silence_frames_needed = max(1, int(silence_sec * 1000 / frame_ms))
        min_speech_frames = max(1, int(min_speech_sec * 1000 / frame_ms))
        consecutive_silence = 0
        t0 = time.time()

        while True:
            # 총 길이 상한
            if (time.time() - t0) > max_total_sec:
                print("[STT] max_total_sec 도달로 종료")
                break

            block = q.get()
            amp = float(np.mean(np.abs(block)))
            is_voice = amp > threshold

            if not speech_started:
                if is_voice:
                    speech_started = True
                    frames.append(block)
                else:
                    # 발화 전에는 버퍼를 모으지 않음 (원하면 pre-roll 구현 가능)
                    continue
            else:
                frames.append(block)
                if is_voice:
                    consecutive_silence = 0
                else:
                    consecutive_silence += 1
                    # 최소 발화 시간 보장 + 침묵 종료
                    if consecutive_silence >= silence_frames_needed and len(frames) >= min_speech_frames:
                        print("[STT] 침묵 지속으로 종료")
                        break

        if not frames:
            print("[STT] 음성 구간이 탐지되지 않음")
            return None

        audio = np.concatenate(frames, axis=0)
        # 1D int16로 보장
        if audio.ndim > 1:
            audio = audio.reshape(-1,)

        tmpdir = tempfile.mkdtemp(prefix="stt_rec_")
        wav_path = os.path.join(tmpdir, "input.wav")
        wav_write(wav_path, sample_rate, audio.astype(np.int16))
        return wav_path


def _play_audio_file(path: str) -> None:
    """
    파일 확장자에 맞춰 재생.
    - .mp3: mpg123
    - .wav: aplay
    둘 다 없으면 재생 생략(파일만 생성)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        player = shutil.which("mpg123")
        if player:
            os.system(f"{player} -q '{path}'")
        else:
            print("[TTS] 경고: mpg123가 없어 재생을 생략합니다. (파일만 생성됨)")
    elif ext == ".wav":
        player = shutil.which("aplay")
        if player:
            os.system(f"{player} -q '{path}'")
        else:
            print("[TTS] 경고: aplay가 없어 재생을 생략합니다. (파일만 생성됨)")
    else:
        print("[TTS] 알 수 없는 포맷, 재생 생략:", path)


# =========================
# TTS
# =========================
def tts_play(
    text: str,
    lang: str = "ko-KR",
    voice: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    out_path: Optional[str] = None,
    audio_encoding: str = "MP3",
) -> str:
    """
    텍스트를 음성으로 합성하고 재생.
    반환값: 생성된 오디오 파일 경로
    - voice 미지정 시 언어에 맞는 기본 음색 선택 시도
    - audio_encoding: "MP3" 또는 "LINEAR16/WAV" 지원
    """
    if not text or not text.strip():
        raise ValueError("TTS 텍스트가 비었습니다.")

    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("[TTS] 경고: GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않았습니다.", file=sys.stderr)

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    if voice is None:
        voice_params = texttospeech.VoiceSelectionParams(language_code=lang)
    else:
        voice_params = texttospeech.VoiceSelectionParams(language_code=lang, name=voice)

    if audio_encoding.upper() == "MP3":
        audio_cfg = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch,
        )
        ext = ".mp3"
    elif audio_encoding.upper() in ("LINEAR16", "WAV"):
        audio_cfg = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate,
            pitch=pitch,
        )
        ext = ".wav"
    else:
        raise ValueError("audio_encoding은 'MP3' 또는 'LINEAR16/WAV'만 지원합니다.")

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_cfg,
    )

    if out_path is None:
        out_dir = tempfile.mkdtemp(prefix="tts_out_")
        out_path = os.path.join(out_dir, f"tts{ext}")

    with open(out_path, "wb") as f:
        f.write(response.audio_content)

    _play_audio_file(out_path)
    return out_path


def tts_guide_default(text: str = DEFAULT_CHAT_GUIDE, lang: str = "ko-KR", voice: Optional[str] = None) -> str:
    """
    기본 안내 멘트를 재생하는 헬퍼.
    websocket.py의 CHAT_ORDER_ON 처리 흐름에서 바로 호출하기 좋음.
    """
    return tts_play(text=text, lang=lang, voice=voice)


# =========================
# STT
# =========================
def stt_once(
    mode: str = "auto",           # "auto": 침묵기반 자동종료, "fixed": 고정길이
    duration: int = 5,            # mode="auto"일 때는 '최대' 총 녹음시간으로 사용
    sample_rate: int = 16000,
    language_code: str = "ko-KR",
    device: Optional[int] = None,
    silence_sec: float = 0.8,
    max_duration: Optional[float] = None,  # None이면 duration 사용
    calib_sec: float = 0.4,
    sensitivity: float = 2.0,
    min_speech_sec: float = 0.3,
) -> str:
    """
    STT 단발 인식:
    - mode="auto": 사용자가 말하는 동안만 녹음하고 침묵이 지속되면 자동 종료
      * 최대 길이: max_duration(미지정 시 duration 값)
    - mode="fixed": duration초 만큼 고정 녹음
    """
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("[STT] 경고: GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않았습니다.", file=sys.stderr)

    wav_path = None
    try:
        if mode == "fixed":
            wav_path = record_audio(duration=duration, sample_rate=sample_rate, channels=1, device=device)
        else:
            total_cap = float(max_duration if max_duration is not None else duration)
            wav_path = record_until_silence(
                sample_rate=sample_rate,
                device=device,
                frame_ms=30,
                silence_sec=silence_sec,
                max_total_sec=total_cap,
                calib_sec=calib_sec,
                sensitivity=sensitivity,
                min_speech_sec=min_speech_sec,
            )
            if wav_path is None:
                # 음성 감지 실패: 빈 문자열 반환
                return ""

        client = speech.SpeechClient()
        with open(wav_path, "rb") as f:
            content = f.read()
        audio_msg = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            audio_channel_count=1,
            enable_automatic_punctuation=True,
        )
        print("[STT] 인식 요청 전송...")
        resp = client.recognize(config=config, audio=audio_msg)
        print("[STT] 인식 응답 수신")
        if not resp.results:
            return ""
        return resp.results[0].alternatives[0].transcript
    except Exception as e:
        print(f"[STT] 오류: {e}", file=sys.stderr)
        return ""
    finally:
        # 임시 파일 정리
        try:
            if wav_path:
                base = os.path.dirname(wav_path)
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                if os.path.isdir(base):
                    os.rmdir(base)
        except Exception:
            pass


# =========================
# 간단 CLI (테스트용)
# =========================
def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="TTS/STT Helper (Raspberry Pi)")
    sub = parser.add_subparsers(dest="cmd")

    p_tts = sub.add_parser("tts", help="텍스트를 합성해 재생")
    p_tts.add_argument("--text", type=str, required=True, help="합성할 텍스트")
    p_tts.add_argument("--lang", type=str, default="ko-KR")
    p_tts.add_argument("--voice", type=str, default=None, help="보이스 이름 (예: ko-KR-Wavenet-B)")
    p_tts.add_argument("--rate", type=float, default=1.0, help="말하기 속도(0.25~4.0)")
    p_tts.add_argument("--pitch", type=float, default=0.0, help="피치(-20.0~20.0)")
    p_tts.add_argument("--enc", type=str, default="MP3", choices=["MP3", "LINEAR16", "WAV"])
    p_tts.add_argument("--out", type=str, default=None, help="저장 경로(미지정 시 임시 폴더)")

    p_stt = sub.add_parser("stt", help="마이크로 녹음 후 인식")
    p_stt.add_argument("--mode", type=str, default="auto", choices=["auto", "fixed"], help="녹음 모드")
    p_stt.add_argument("--duration", type=int, default=5, help="fixed: 길이(초) / auto: 최대 길이(초)")
    p_stt.add_argument("--sr", type=int, default=16000, help="샘플레이트")
    p_stt.add_argument("--lang", type=str, default="ko-KR", help="인식 언어 코드")
    p_stt.add_argument("--device", type=int, default=None, help="입력 장치 인덱스 (미지정 시 자동)")
    p_stt.add_argument("--silence", type=float, default=0.8, help="auto: 침묵 종료 임계(초)")
    p_stt.add_argument("--max", type=float, default=None, help="auto: 최대 녹음 시간(초)")
    p_stt.add_argument("--calib", type=float, default=0.4, help="auto: 주변소음 기준 측정(초)")
    p_stt.add_argument("--sens", type=float, default=2.0, help="auto: 민감도(높을수록 더 큰 소리에만 반응)")

    p_list = sub.add_parser("list", help="입력 장치 나열")

    args = parser.parse_args()

    if args.cmd == "tts":
        path = tts_play(
            text=args.text,
            lang=args.lang,
            voice=args.voice,
            speaking_rate=args.rate,
            pitch=args.pitch,
            out_path=args.out,
            audio_encoding=args.enc,
        )
        print(json.dumps({"ok": True, "path": path}, ensure_ascii=False))
    elif args.cmd == "stt":
        device = args.device
        if device is None:
            device = find_input_device_index()
        text = stt_once(
            mode=args.mode,
            duration=args.duration,
            sample_rate=args.sr,
            language_code=args.lang,
            device=device,
            silence_sec=args.silence,
            max_duration=args.max,
            calib_sec=args.calib,
            sensitivity=args.sens,
        )
        print(json.dumps({"ok": True, "text": text}, ensure_ascii=False))
    elif args.cmd == "list":
        devs = list_input_devices()
        print(json.dumps(devs, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
