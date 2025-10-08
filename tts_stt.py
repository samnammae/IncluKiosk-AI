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

import requests


DEFAULT_CHAT_GUIDE = "안녕하세요. 음성으로 주문을 도와드릴게요." # 기본 안내 멘트
DEFAULT_ERROR_GUIDE = "죄송합니다, 말씀을 정확히 인식하지 못했습니다. 다시 한번 말씀해 주시겠어요?" # 오류 안내 멘트
DEFAULT_CANCEL_GUIDE = "인식되는 음성이 없어 주문이 취소되었습니다." # 무응답 2회 시 주문 취소 안내 멘트
DEFAULT_PRE_SOUND = "start_recording.mp3" # STT 시작 안내 사운드 기본 경로

STT_SILENCE_SEC = 1.2        # 침묵 종료 임계(초)
STT_MAX_DURATION = 60.0      # 최대 녹음 시간(초)
STT_CALIB_SEC = 0.4          # 주변소음 측정 시간(초)
STT_SENSITIVITY = 2.0        # 음성 감지 민감도
STT_MIN_SPEECH_SEC = 0.2     # 최소 발화 시간(초)
STT_ENGINE = "naver"         # 기본 STT 엔진
STT_SAMPLE_RATE = 16000      # 1초에 음성 몇번 녹음? (네이버 추천 값)
STT_PRE_SOUND_PAUSE = 0.25   # 프리 사운드 후 대기 시간
STT_INITIAL_SILENCE_TIMEOUT = 5.0  # 처음부터 말이 없으면 5초 후 종료

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
# 공용 유틸
# =========================
def _mask_secret(s: Optional[str]) -> str:
    if not s:
        return ""
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}{'*'*(len(s)-8)}{s[-4:]}"

# =========================
# Naver Clova Speech Recognition (CSR) helper
# =========================
def _lang_google_to_naver(language_code: str) -> str:
    """
    'ko-KR' -> 'Kor', 'en-US' -> 'Eng', 'ja-JP' -> 'Jpn', 'zh' -> 'Chn'
    NAVER_CSR_LANG 환경변수가 있으면 그것을 우선 사용.
    """
    env_lang = os.environ.get("NAVER_CSR_LANG")
    if env_lang:
        return env_lang

    lc = (language_code or "").lower()
    if lc.startswith("ko"):
        return "Kor"
    if lc.startswith("en"):
        return "Eng"
    if lc.startswith("ja") or lc.startswith("jp"):
        return "Jpn"
    if lc.startswith("zh") or "cn" in lc:
        return "Chn"
    # 기본값
    return "Kor"

def _stt_naver_csr_from_wav(
    wav_path: str,
    language_code: str = "ko-KR",
    timeout: float = 30.0,
) -> str:
    """
    네이버 CSR REST API로 WAV(16kHz, mono, 16-bit) 파일을 전송해 텍스트를 얻는다.
    응답은 JSON({"text": "..."})) 또는 평문 텍스트일 수 있으므로 모두 대응.
    """
    key_id = os.environ.get("NAVER_CSR_KEY_ID")
    key = os.environ.get("NAVER_CSR_KEY")
    if not key_id or not key:
        raise RuntimeError("NAVER_CSR_KEY_ID / NAVER_CSR_KEY 환경변수가 필요합니다.")

    base_url = os.environ.get("NAVER_CSR_URL", "https://naveropenapi.apigw.ntruss.com/recog/v1/stt")
    lang = _lang_google_to_naver(language_code)

    url = f"{base_url}?lang={lang}"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": key_id,
        "X-NCP-APIGW-API-KEY": key,
        "Content-Type": "application/octet-stream",
    }

    with open(wav_path, "rb") as f:
        data = f.read()

    t0 = time.time()
    resp = requests.post(url, headers=headers, data=data, timeout=timeout)
    elapsed = (time.time() - t0) * 1000.0
    print(f"[STT][NAVER] HTTP {resp.status_code} {resp.reason} in {elapsed:.1f} ms  content-type={resp.headers.get('Content-Type')}")
    resp.raise_for_status()

    # JSON 우선 시도
    text = ""
    try:
        j = resp.json()
        # 일반적으로 {"text": "..."} 형태
        text = j.get("text", "")
        print(f"[STT][NAVER] JSON parsed: {text[:120]!r}{'...' if len(text)>120 else ''}")
    except Exception:
        # JSON이 아니면 평문으로 처리
        text = resp.text.strip()
        print(f"[STT][NAVER] Plain parsed: {text[:120]!r}{'...' if len(text)>120 else ''}")

    return text or ""


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
    print(f"[STT] (fixed) WAV saved: {wav_path}  bytes={os.path.getsize(wav_path)}")
    return wav_path


def record_until_silence(
    sample_rate: int = STT_SAMPLE_RATE,
    device: Optional[int] = None,   # 반환: 임시 WAV 파일 경로 (없으면 None)
    frame_ms: int = 30,             # 프레임 길이(ms) (30~40ms 권장)
    silence_sec: float = STT_SILENCE_SEC,       # 침묵 시간
    max_total_sec: float = STT_MAX_DURATION,    # 최대 녹음 시간
    calib_sec: float = STT_CALIB_SEC,         # 시작 전 주변소음 기준치 측정 시간
    sensitivity: float = STT_SENSITIVITY,       # sensitivity 보다 크면 발성으로 판단
    min_speech_sec: float = STT_MIN_SPEECH_SEC,    # 최소 발화 시간
    initial_silence_timeout: float = STT_INITIAL_SILENCE_TIMEOUT 
) -> Optional[str]:
    
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
        initial_silence_frames = 0
        initial_silence_limit = max(1, int(initial_silence_timeout * 1000 / frame_ms))
        t0 = time.time()

        while True:
            # 총 길이 상한
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

                    # 최소 발화 시간 보장 + 침묵 종료
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


def _play_if_exists(path: Optional[str], pause_after: float = 0.25) -> None:
    """
    파일이 존재하면 재생하고, 약간의 지연 후 리턴.
    - pause_after: 재생 직후 마이크 입력이 섞이지 않도록 대기 (초)
    """
    if not path:
        return
    if os.path.isfile(path):
        print(f"[SND] 프리사운드 재생: {path}")
        _play_audio_file(path)
        if pause_after > 0:
            time.sleep(pause_after)
    else:
        # 조용히 건너뜀
        print(f"[SND] 프리사운드 없음(건너뜀): {path}")


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

def _stt_google_from_wav(wav_path: str, sample_rate: int, language_code: str) -> str:
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
    print("[STT][GOOGLE] recognize() 호출")
    resp = client.recognize(config=config, audio=audio_msg)
    print("[STT][GOOGLE] 응답 수신")
    if not resp.results:
        return ""
    text = resp.results[0].alternatives[0].transcript
    print(f"[STT][GOOGLE] Text: {text[:120]!r}{'...' if len(text)>120 else ''}")
    return text


# =========================
# STT
# =========================
def stt_once(
    mode: str = "auto",
    duration: int = 60,
    sample_rate: int = STT_SAMPLE_RATE,  
    language_code: str = "ko-KR",
    device: Optional[int] = None,
    silence_sec: float = STT_SILENCE_SEC,         
    max_duration: Optional[float] = STT_MAX_DURATION, 
    calib_sec: float = STT_CALIB_SEC,           
    sensitivity: float = STT_SENSITIVITY,       
    min_speech_sec: float = STT_MIN_SPEECH_SEC,  
    pre_sound: Optional[str] = DEFAULT_PRE_SOUND,
    pre_sound_pause: float = STT_PRE_SOUND_PAUSE,
    engine: str = None,
    initial_silence_timeout: float = STT_INITIAL_SILENCE_TIMEOUT, 
) -> str:
    """
    STT 단발 인식:
    - mode="auto": 사용자가 말하는 동안만 녹음하고 침묵이 지속되면 자동 종료
      * 최대 길이: max_duration(미지정 시 duration 값) -> 기본 60초
    - mode="fixed": duration초 만큼 고정 녹음
    """
    # ★ 엔진 결정 (우선순위: 인자 > 환경변수 > 기본 'google')
    eng = (engine or os.environ.get("STT_ENGINE") or "google").lower()
    if eng not in ("google", "naver"):
        eng = "google"
    print(f"[STT] Engine selected = {eng}  (mode={mode}, sr={sample_rate}, lang={language_code})")

    if eng == "google" and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("[STT] 경고: GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않았습니다.", file=sys.stderr)

    _play_if_exists(pre_sound, pause_after=pre_sound_pause)

    wav_path = None
    try:
        if mode == "fixed":
            print(f"[STT] mode=fixed  duration={duration}s  device={device}")
            wav_path = record_audio(duration=duration, sample_rate=sample_rate, channels=1, device=device)
        else:
            total_cap = float(max_duration if max_duration is not None else duration)
            print(f"[STT] mode=auto  silence_sec={silence_sec}s  min_speech_sec={min_speech_sec}s  max_total={total_cap}s  device={device}")
            wav_path = record_until_silence(
                sample_rate=sample_rate,
                device=device,
                frame_ms=30,
                silence_sec=silence_sec,
                max_total_sec=total_cap,
                calib_sec=calib_sec,
                sensitivity=sensitivity,
                min_speech_sec=min_speech_sec,
                initial_silence_timeout=initial_silence_timeout
            )
            if wav_path is None:
                print("[STT] 음성 감지 실패 → 빈 문자열 반환")
                return ""

        # ===== 엔진 분기 =====
        if eng == "naver":
            print("[STT] Naver CSR로 인식 요청 전송...")
            text = _stt_naver_csr_from_wav(wav_path, language_code=language_code)
            print("[STT] 인식 응답 수신(Naver CSR)")
        else:
            text = _stt_google_from_wav(wav_path, sample_rate, language_code)

        print(f"[STT] 최종 텍스트 길이: {len(text)} chars")
        return text

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
                print(f"[STT] 임시 파일/폴더 정리 완료: {wav_path}")
        except Exception as ce:
            print(f"[STT] 임시 파일 정리 중 예외: {ce}", file=sys.stderr)

def stt_once_with_error_handling(
    on_error_callback=None,      # STT_ERR 전송 콜백
    on_error_end_callback=None,  # ERR_END 전송 콜백
    error_guide_text: str = DEFAULT_ERROR_GUIDE,
    error_guide_lang: str = "ko-KR",
    error_guide_voice: Optional[str] = None,
    **stt_kwargs  # stt_once에 전달할 모든 인자
) -> tuple[str, bool]:
    """
    STT 실행 + 오류 시 자동 안내
    
    Returns:
        (text, success): 
            - text: 인식된 텍스트 (오류 시 빈 문자열)
            - success: 성공 여부 (True/False)
    """
    try:
        text = stt_once(**stt_kwargs)
        
        # 빈 문자열 = 인식 실패로 간주
        if not text or not text.strip():
            print("[STT] 음성 인식 실패 → 오류 처리 시작")
            
            # 1. 오류 발생 알림 (STT_ERR)
            if on_error_callback:
                try:
                    on_error_callback()
                    print("[STT] STT_ERR 콜백 실행 완료")
                except Exception as e:
                    print(f"[STT] STT_ERR 콜백 실패: {e}", file=sys.stderr)
            
            # 2. 오류 안내 음성 재생
            try:
                print(f"[STT] 오류 안내 TTS 재생: \"{error_guide_text}\"")
                tts_play(
                    text=error_guide_text,
                    lang=error_guide_lang,
                    voice=error_guide_voice,
                    audio_encoding="LINEAR16"  # WAV로 재생
                )
                print("[STT] 오류 안내 TTS 재생 완료")
            except Exception as e:
                print(f"[STT] 오류 안내 TTS 실패: {e}", file=sys.stderr)
            
            # 3. 오류 안내 종료 알림 (ERR_END)
            if on_error_end_callback:
                try:
                    on_error_end_callback()
                    print("[STT] ERR_END 콜백 실행 완료")
                except Exception as e:
                    print(f"[STT] ERR_END 콜백 실패: {e}", file=sys.stderr)
            
            return "", False  # 빈 문자열, 실패
        
        # 성공
        return text, True
        
    except Exception as e:
        print(f"[STT] 예외 발생: {e}", file=sys.stderr)
        
        # 예외 발생 시에도 동일한 오류 처리
        if on_error_callback:
            try:
                on_error_callback()
            except Exception:
                pass
        
        try:
            tts_play(
                text=error_guide_text,
                lang=error_guide_lang,
                voice=error_guide_voice,
                audio_encoding="LINEAR16"
            )
        except Exception:
            pass
        
        if on_error_end_callback:
            try:
                on_error_end_callback()
            except Exception:
                pass
        
        return "", False