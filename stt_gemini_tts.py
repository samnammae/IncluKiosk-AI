import os
import traceback
import pyaudio
import wave
from google.cloud import texttospeech
from google.cloud import speech
from google import genai
from dotenv import load_dotenv

# ── 환경 변수(필수) ───────────────────────────────────────────────
load_dotenv("/home/pi/IncluKiosk/config.env")
# GCP STT/TTS 인증키
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/home/pi/IncluKiosk/gcp-tts-key.json")
# Gemini API Key (export GEMINI_API_KEY="...") 로 설정해두세요.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── 장치 유틸 ────────────────────────────────────────────────────
def list_input_devices():
    p = pyaudio.PyAudio()
    print("\n[Input-capable devices]")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            print(f"  {i}: {info.get('name')} (in={info.get('maxInputChannels')}, rate={info.get('defaultSampleRate')})")
    p.terminate()

def find_input_device_index(prefer_keywords=("respeaker", "usb", "mic", "seeed")):
    p = pyaudio.PyAudio()
    target = None
    prefer = [k.lower() for k in prefer_keywords]
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name = str(info.get("name", "")).lower()
            max_in = int(info.get("maxInputChannels", 0))
            if max_in > 0:
                if any(k in name for k in prefer):
                    target = i
                    break
                if target is None:
                    target = i
    finally:
        p.terminate()
    return target

# ── 녹음 ─────────────────────────────────────────────────────────
def record_audio(filename="recorded_audio.wav", duration=5, sample_rate=16000):
    chunk = 1024
    fmt = pyaudio.paInt16
    channels = 1

    audio = pyaudio.PyAudio()
    device_index = find_input_device_index()

    try:
        if device_index is None:
            print("❗ 입력 가능한 장치를 찾지 못했습니다.")
            list_input_devices()
            raise RuntimeError("No input-capable device found")

        dev_info = audio.get_device_info_by_index(device_index)
        print(f"🎛️  Using input device index={device_index}, name='{dev_info.get('name')}'")

        stream = audio.open(format=fmt,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=chunk)

        print(f"🎤 {duration}초 동안 말씀해 주세요...")
        frames = []
        for _ in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

        print("🔴 녹음 완료!")

        stream.stop_stream()
        stream.close()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(fmt))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

    except Exception:
        list_input_devices()
        raise
    finally:
        audio.terminate()

    return filename

# ── STT(Google Speech-to-Text) ───────────────────────────────────
def speech_to_text_from_file(wav_path: str, sample_rate=16000, lang="ko-KR"):
    client = speech.SpeechClient()
    with open(wav_path, "rb") as f:
        content = f.read()

    audio_msg = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code=lang,
    )

    print("🔄 STT 요청 중...")
    resp = client.recognize(config=config, audio=audio_msg)

    text = ""
    conf = None
    if resp.results:
        best = resp.results[0].alternatives[0]
        text = best.transcript
        conf = best.confidence
        print(f"✅ STT 텍스트: '{text}' (신뢰도 {conf:.2%})")
    else:
        print("❌ 음성을 인식하지 못했습니다.")

    return text, conf

# ── Gemini 호출 ──────────────────────────────────────────────────
def ask_gemini(prompt: str, model="gemini-2.5-flash"):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("🤖 Gemini 응답 생성 중...")
    resp = client.models.generate_content(model=model, contents=prompt)
    text = getattr(resp, "text", None) or ""
    print(f"✅ Gemini 응답: {text}")
    return text

# ── TTS(Google Text-to-Speech) ───────────────────────────────────
def speak_text(text: str, voice_name="ko-KR-Wavenet-B", out="output.mp3"):
    if not text:
        print("ℹ️ 읽을 텍스트가 비어 있어 TTS를 건너뜁니다.")
        return
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", name=voice_name)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    print("🔄 TTS 합성 중...")
    resp = client.synthesize_speech(input=synthesis_input, voice=voice, audio=audio_config)
    with open(out, "wb") as f:
        f.write(resp.audio_content)
    print(f"🔊 재생: {out}")
    os.system(f"mpg123 {out}")

# ── 파이프라인 실행 ──────────────────────────────────────────────
def stt_gemini_tts(duration_sec=5):
    try:
        wav = record_audio(duration=duration_sec, sample_rate=16000)
        text, conf = speech_to_text_from_file(wav, sample_rate=16000, lang="ko-KR")

        if not text:
            print("🙅 STT 결과가 비었습니다. 다시 시도해 주세요.")
            return

        reply = ask_gemini(text)  # STT 텍스트를 Gemini에 그대로 전달
        speak_text(reply)

        # 임시 파일 정리
        if os.path.exists(wav):
            os.remove(wav)
    except Exception as e:
        print("\n❗오류 발생:")
        print(e)
        traceback.print_exc()

def main():
    print("=" * 50)
    print("🎙️  STT → Gemini → TTS (One-shot)")
    print("=" * 50)
    while True:
        sel = input("\n⏺ 녹음 길이를 초 단위로 입력하세요 (기본 5, 종료:q): ").strip()
        if sel.lower() == "q":
            break
        try:
            dur = int(sel) if sel.isdigit() else 5
        except:
            dur = 5
        stt_gemini_tts(duration_sec=dur)

if __name__ == "__main__":
    main()
