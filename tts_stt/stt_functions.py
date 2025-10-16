""" STT (Speech-to-Text) 함수들 """
import os
import sys
import time
from typing import Optional

import requests

try:
    from google.cloud import speech
except Exception as e:
    raise RuntimeError(
        "google-cloud-speech 패키지를 설치하세요.\n"
        "pip install google-cloud-speech"
    ) from e

from .config import (
    STT_SAMPLE_RATE,
    STT_SILENCE_SEC,
    STT_MAX_DURATION,
    STT_CALIB_SEC,
    STT_SENSITIVITY,
    STT_MIN_SPEECH_SEC,
    STT_PRE_SOUND_PAUSE,
    STT_INITIAL_SILENCE_TIMEOUT,
    DEFAULT_PRE_SOUND
)
from .audio_utils import record_audio, record_until_silence, play_if_exists


def lang_google_to_naver(language_code: str) -> str:
    """Google 언어 코드를 Naver 언어 코드로 변환"""
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
    return "Kor"


def stt_naver_from_wav(
    wav_path: str,
    language_code: str = "ko-KR",
    timeout: float = 30.0,
) -> str:
    """네이버 CSR REST API로 WAV 파일 전송해 텍스트 얻기"""
    key_id = os.environ.get("NAVER_CSR_KEY_ID")
    key = os.environ.get("NAVER_CSR_KEY")
    if not key_id or not key:
        raise RuntimeError("NAVER_CSR_KEY_ID / NAVER_CSR_KEY 환경변수가 필요합니다.")

    base_url = os.environ.get(
        "NAVER_CSR_URL",
        "https://naveropenapi.apigw.ntruss.com/recog/v1/stt"
    )
    lang = lang_google_to_naver(language_code)

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
    print(f"[STT][NAVER] HTTP {resp.status_code} {resp.reason} in {elapsed:.1f} ms")
    resp.raise_for_status()

    text = ""
    try:
        j = resp.json()
        text = j.get("text", "")
        print(f"[STT][NAVER] JSON parsed: {text[:120]!r}")
    except Exception:
        text = resp.text.strip()
        print(f"[STT][NAVER] Plain parsed: {text[:120]!r}")

    return text or ""


def stt_google_from_wav(wav_path: str, sample_rate: int, language_code: str) -> str:
    """Google STT API로 WAV 파일 인식"""
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
    print(f"[STT][GOOGLE] Text: {text[:120]!r}")
    return text


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
    STT 단발 인식
    - mode="auto": 침묵 감지 자동 종료
    - mode="fixed": duration초 고정 녹음
    """
    eng = (engine or os.environ.get("STT_ENGINE") or "google").lower()
    
    print(f"[DEBUG] engine 파라미터: {engine}")
    print(f"[DEBUG] STT_ENGINE 환경변수: {os.environ.get('STT_ENGINE')}")
    print(f"[DEBUG] 최종 선택된 엔진: {eng}")
    
    if eng not in ("google", "naver"):
        eng = "google"
    print(f"[STT] Engine selected = {eng}  (mode={mode}, sr={sample_rate}, lang={language_code})")

    if eng == "google" and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("[STT] 경고: GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않았습니다.", file=sys.stderr)

    play_if_exists(pre_sound, pause_after=pre_sound_pause)

    wav_path = None
    try:
        if mode == "fixed":
            print(f"[STT] mode=fixed  duration={duration}s  device={device}")
            wav_path = record_audio(duration=duration, sample_rate=sample_rate, channels=1, device=device)
        else:
            total_cap = float(max_duration if max_duration is not None else duration)
            print(f"[STT] mode=auto  silence_sec={silence_sec}s  max_total={total_cap}s")
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

        if eng == "naver":
            print("[STT] Naver CSR로 인식 요청 전송...")
            text = stt_naver_from_wav(wav_path, language_code=language_code)
        else:
            text = stt_google_from_wav(wav_path, sample_rate, language_code)

        print(f"[STT] 최종 텍스트 길이: {len(text)} chars")
        return text

    except Exception as e:
        print(f"[STT] 오류: {e}", file=sys.stderr)
        return ""
    finally:
        if wav_path:
            try:
                base = os.path.dirname(wav_path)
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                if os.path.isdir(base):
                    os.rmdir(base)
                print(f"[STT] 임시 파일/폴더 정리 완료: {wav_path}")
            except Exception as ce:
                print(f"[STT] 임시 파일 정리 중 예외: {ce}", file=sys.stderr)