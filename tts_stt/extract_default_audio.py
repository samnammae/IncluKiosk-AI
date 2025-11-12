#!/usr/bin/env python3
"""
기본 안내 메시지를 오디오 파일로 미리 추출하는 스크립트
실행: python extract_default_audio.py
"""
import os
import sys
from pathlib import Path

try:
    from google.cloud import texttospeech
except ImportError:
    print("Error: google-cloud-texttospeech 패키지가 필요합니다.")
    print("설치: pip install google-cloud-texttospeech")
    sys.exit(1)


def extract_audio(
    text: str,
    output_path: str,
    lang: str = "ko-KR",
    voice: str = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_encoding: str = "LINEAR16"
):
    """텍스트를 오디오 파일로 변환하여 저장"""
    if not text or not text.strip():
        raise ValueError("텍스트가 비어있습니다.")
    
    print(f"[TTS] 추출 중: '{text[:50]}...' -> {output_path}")
    
    # Google TTS 클라이언트 생성
    client = texttospeech.TextToSpeechClient()
    
    # 음성 합성 입력
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # 음성 파라미터 설정
    if voice is None:
        voice_params = texttospeech.VoiceSelectionParams(language_code=lang)
    else:
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang,
            name=voice
        )
    
    # 오디오 설정
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
    
    # 음성 합성 요청
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_cfg,
    )
    
    # 파일 저장
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(response.audio_content)
    
    file_size = os.path.getsize(output_path)
    print(f"[TTS] 저장 완료: {output_path} ({file_size:,} bytes)")


def main():
    """기본 안내 메시지 오디오 파일 추출"""
    
    # 환경 변수 확인
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않았습니다.")
        print("Google Cloud 서비스 계정 JSON 키 파일 경로를 설정하세요.")
        sys.exit(1)
    
    # 출력 디렉토리 설정 (현재 스크립트와 같은 위치 또는 지정된 경로)
    # 패키지 디렉토리 내 audio 폴더에 저장
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "audio"
    output_dir.mkdir(exist_ok=True)
    
    print(f"출력 디렉토리: {output_dir}")
    print("=" * 60)
    
    # 추출할 기본 메시지 정의
    messages = {
        # 파일명: (텍스트, 설명)
        "chat_guide.wav": (
            "안녕하세요. 음성으로 주문을 도와드릴게요.",
            "주문 시작 안내"
        ),
        "error_guide.wav": (
            "죄송합니다, 말씀을 정확히 인식하지 못했습니다. 다시 한번 말씀해 주시겠어요?",
            "음성 인식 오류 안내"
        ),
        "cancel_guide.wav": (
            "인식되는 음성이 없어 주문이 취소되었습니다.",
            "주문 취소 안내"
        ),
        "height_guide.wav": (
            "높이 조절 중입니다.",
            "높이 조절 안내"
        ),
        "calib_guide.wav": (
            "눈 보정을 실행 중입니다. 화면 중앙을 바라봐주세요.",
            "눈 보정 안내"
        ),
        "mode_guide.wav": (
            "안녕하세요. 음성 주문을 원하시면 주먹을 쥐어주세요.",
            "모드 선택 안내"
        ),
    }
    
    # TTS 설정
    lang = "ko-KR"
    voice = None  # None이면 기본 음성 사용
    speaking_rate = 1.0
    pitch = 0.0
    audio_encoding = "LINEAR16"  # WAV 형식
    
    # 각 메시지를 오디오 파일로 추출
    success_count = 0
    fail_count = 0
    
    for filename, (text, description) in messages.items():
        output_path = output_dir / filename
        print(f"\n[{description}]")
        try:
            extract_audio(
                text=text,
                output_path=str(output_path),
                lang=lang,
                voice=voice,
                speaking_rate=speaking_rate,
                pitch=pitch,
                audio_encoding=audio_encoding
            )
            success_count += 1
        except Exception as e:
            print(f"[ERROR] 실패: {e}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"추출 완료: 성공 {success_count}개, 실패 {fail_count}개")
    print(f"저장 위치: {output_dir}")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()