""" TTS (Text-to-Speech) 함수들 """
import os
import sys
import tempfile
from typing import Optional

try:
    from google.cloud import texttospeech
except Exception as e:
    raise RuntimeError(
        "google-cloud-texttospeech 패키지를 설치하세요.\n"
        "pip install google-cloud-texttospeech"
    ) from e

from .audio_utils import play_audio_file


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

    play_audio_file(out_path)
    return out_path