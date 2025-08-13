import os
import traceback
from google.cloud import texttospeech

# 서비스 계정 키 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/pi/IncluKiosk/gcp-tts-key.json"

try:
    # 사용자 입력 받기
    text = input("읽어줄 문장을 입력하세요: ").strip()
    if not text:
        raise ValueError("입력된 문장이 없습니다.")

    # TTS 클라이언트 초기화
    client = texttospeech.TextToSpeechClient()

    # 요청 구성
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR", name="ko-KR-Wavenet-B"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # TTS 요청
    print("TTS 요청 중...")
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    print("TTS 응답 완료.")

    # 음성 저장
    output_path = "output.mp3"
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    print(f"음성 파일 저장 완료: {output_path}")

    # 음성 재생
    os.system(f"mpg123 {output_path}")

except Exception as e:
    print("\n❗에러 발생:")
    print(e)
    traceback.print_exc()
