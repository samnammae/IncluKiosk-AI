import asyncio
import websockets
import subprocess
import json
import sys
from pathlib import Path
import os
from functools import partial
import tts_stt
from asyncio import Event

from set_height.linear_actuator.linear_actuator_controller import on_shutdown
import atexit

import time
import psutil

# 미리 추출된 오디오 사용 여부
USE_PREGENERATED_GUIDE = os.environ.get("USE_PREGENERATED_GUIDE", "true").lower() == "true"

# 전역 변수에 추가
height_set_processing = False
height_last_request_time = 0
HEIGHT_DEBOUNCE_SEC = 2.0     # ⬅️ 최소 2초 간격

stt_fail_count = 0  # TTS 무응답(실패) 횟수 카운터

# ============ 가상환경 Python 경로 명시 ============
VENV_PYTHON = "/home/pi/IncluKiosk/IncluKiosk_venv/bin/python"
PYTHON = VENV_PYTHON  # ← 수정 (기존: sys.executable)
BASE_DIR = Path(__file__).resolve().parent

PIR_WORKER = str(BASE_DIR / "pir_sensor" / "pir_worker.py")
EYE_SCRIPT = str(BASE_DIR / "eye_tracking" / "worker.py")
HEIGHT_WORKER = str(BASE_DIR / "set_height" / "worker.py")

workers = {"PIR": None}
clients = set()  # 프론트엔드 클라이언트들

# 런처 핸들
eye_proc = None
height_proc = None
height_set_processing = False  # 처리 중 플래그
eye_calib_processing = False  # 캘리브레이션 진행 중 플래그
eye_calib_completed = False   # 캘리브레이션 완료 플래그 추가
mode_select_processing = False  # 모드 선택 진행 중 플래그
chat_order_processing = False  # 대화 주문 진행 중 플래그
normal_order_processing = False # 일반 주문 진행 중 플래그
eye_order_processing = False # 아이트래킹 주문 진행 중 플래그
eye_ready_flag = False

eye_ready_event = None    # eye 워커 준비 신호 대기
touch_active = False        # 터치 중 여부(브로드캐스트용)

# === 내부 통신용 === #
internal_workers = set()
frontend_ws = None    # 프론트엔드의 WebSocket 연결

atexit.register(on_shutdown)    # 최종 프로그램 종료시에 리니어액추에이터 높이 낮추기

async def send_json(ws, payload: dict):
    try:
        await ws.send(json.dumps(payload, ensure_ascii=False))
    except Exception:
        clients.discard(ws)

async def broadcast_json(payload: dict):
    dead = []
    raw = json.dumps(payload, ensure_ascii=False)
    for ws in list(clients):
        try:
            await ws.send(raw)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)
        
import psutil  # 파일 상단에 추가

def count_eye_worker_processes():
    """실행중인 eye_tracking.worker 프로세스 개수 확인"""
    count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'eye_tracking.worker' in ' '.join(cmdline):
                count += 1
                print(f"🔵[Hub] [DEBUG] 발견된 프로세스: PID={proc.info['pid']}, CMD={' '.join(cmdline)}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return count

# === 라즈베리파이 내부 워커에게 메시지 전송 === #
async def send_to_internal_worker(payload: dict):
    if not internal_workers:
        return
    raw = json.dumps(payload, ensure_ascii=False)
    dead = []
    for ws in list(internal_workers):
        try:
            await ws.send(raw)
        except Exception as e:
            print(f"🔵[Hub] [Hub→Worker] 전송 실패: {e}")
            dead.append(ws)
    for ws in dead:
        internal_workers.discard(ws)

# === front로 메시지 전송 === #
async def send_to_front(payload: dict):
    if frontend_ws:
        try:
            await frontend_ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            print(f"🔵[Hub] [Hub→Front] 전송 실패: {e}")
            clients.discard(frontend_ws)

# === pir 센서 관련 ===
async def start_pir(websocket=None):
    if workers["PIR"] and (workers["PIR"].poll() is None):
        return
    proc = subprocess.Popen([PYTHON, PIR_WORKER])
    workers["PIR"] = proc

async def stop_pir(websocket=None):
    proc = workers.get("PIR")
    if not proc or (proc.poll() is not None):
        workers["PIR"] = None
        return
    proc.terminate()
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, proc.wait, 3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        workers["PIR"] = None

async def clear_pir_state_and_notify_frontend():
    workers["PIR"] = None
    await broadcast_json({"type": "PIR_OFF"})

# ====== 프로세스 런처 유틸 ======
def is_running(p):
    return (p is not None) and (p.poll() is None)

def pir_running():
    p = workers.get("PIR")
    return (p is not None) and (p.poll() is None)

def eye_running():
    global eye_proc
    return is_running(eye_proc)

def height_running():
    global height_proc
    return is_running(height_proc)

def stop_proc(p):
    if not is_running(p):
        return None
    try:
        p.terminate()
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()
    except Exception:
        pass
    return None

def start_eye():
    global eye_proc
    # 시스템 전체 프로세스 개수 확인
    total_count = count_eye_worker_processes()
    print(f"🔵[Hub] [DEBUG] 🔍 시스템에서 실행중인 eye_tracking.worker 프로세스: {total_count}개")
    
    print(f"🔵[Hub] [DEBUG] start_eye() 호출됨")
    print(f"🔵[Hub] [DEBUG] 현재 eye_proc 상태: {eye_proc}")
    print(f"🔵[Hub] [DEBUG] eye_proc 실행중? {is_running(eye_proc)}")
    if is_running(eye_proc):
        return
    env = os.environ.copy()
    # X 세션 접근 보장 (pi 사용자 기준)
    env.setdefault("DISPLAY", ":0")
    env.setdefault("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
    eye_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "eye_tracking.worker"],  # ← VENV_PYTHON 사용
        stdout=None,
        stderr=None,
        env=env,
        cwd=str(BASE_DIR)
    )
    print(f"🔵[Hub] [DEBUG] ✅ 새로운 eye_proc 생성됨 (PID: {eye_proc.pid})")

# === 높이 조절 관련 ===
async def start_height_worker():
    """높이 조절 워커 시작"""
    global height_proc, height_set_processing
    
    # 이미 처리 중이면 무시
    if height_set_processing:
        print("🔵[Hub] [Height] 이미 처리 중 (디바운싱)")
        return
    
    if is_running(height_proc):
        print("🔵[Hub] [Height] 이미 실행 중")
        return
    
    # 처리 시작 플래그 설정
    height_set_processing = True
    await asyncio.sleep(0.1)
    
    print("🔵[Hub] [Height] 워커 시작")
    
    loop = asyncio.get_running_loop()
    try:
        # await loop.run_in_executor(None, tts_stt.play_height_guide_message)
        await loop.run_in_executor(
            None, 
            partial(tts_stt.play_height_guide_message, use_pregenerated=USE_PREGENERATED_GUIDE)
        )
    except Exception as e:
        print(f"🔵[Hub] [TTS] 취소 안내 실패: {e}", file=sys.stderr)
    
    # 🆕 에러 로그 파일에 기록
    log_file = open("/tmp/height_worker.log", "w")
    
    height_proc = subprocess.Popen(
        ["sudo", "-E", PYTHON, "-m", "set_height.worker"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR)
    )
    print(f"🔵[Hub] [Height] 프로세스 PID: {height_proc.pid}")
    print(f"🔵[Hub] [Height] 로그 파일: /tmp/height_worker.log")

    # 프로세스 종료를 감시해서 플래그를 적절히 되돌리기
    async def _watch():
        global height_set_processing, height_proc
        
        # ⬇️ 로컬 변수에 저장 (경합 상태 방지!)
        proc_to_watch = height_proc
        
        if proc_to_watch:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, proc_to_watch.wait)
            print("🔵[Hub] [Height] 워커 종료 감지")
            
            try:
                with open("/tmp/height_worker.log", "r") as f:
                    log_content = f.read()
                    if log_content.strip():
                        print("=== Height Worker 로그 ===")
                        print(log_content)
                        print("=== 로그 끝 ===")
            except Exception as e:
                print(f"🔵[Hub] [Height] 로그 읽기 실패: {e}")
            
            # ⬇️ 전역 변수가 아직 이 프로세스를 가리킬 때만 초기화
            if height_proc == proc_to_watch:
                height_set_processing = False
                height_proc = None
    
    asyncio.create_task(_watch())

async def stop_height_worker():
    """높이 조절 중단 (graceful shutdown)"""
    global height_proc, height_set_processing
    
    # ⬇️ 로컬 변수에 저장 (경합 상태 방지!)
    proc = height_proc
    
    if not is_running(proc):
        print("🔵[Hub] [Height] 이미 중단됨")
        height_set_processing = False
        height_proc = None
        return
    
    print("🔵[Hub] [Height] 중단 명령 전송")
    await send_to_internal_worker({"type": "HEIGHT_SET_OFF"})
    
    # 프로세스 종료 대기 (최대 5초)
    try:
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, proc.wait),  # ⬅️ proc 사용
            timeout=5.0
        )
        print("🔵[Hub] [Height] 정상 종료됨")
    except asyncio.TimeoutError:
        print("🔵[Hub] [Height] 5초 대기 후 강제 종료")
        # ⬇️ None 체크 추가
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except:
                try:
                    proc.kill()
                except:
                    pass
    finally:
        height_proc = None
        height_set_processing = False

async def stop_all_workers_safely():
    """카메라/TPU를 사용하는 모든 워커를 안전하게 정지"""
    print("🔵[Hub] [Safety] 모든 워커 정지 시작...")
    
    # 1. 내부 워커들에게 정지 신호 전송
    await send_to_internal_worker({"type": "STOP_ALL"})
    await asyncio.sleep(0.5)
    
    # 2. 프로세스 종료
    global eye_proc
    eye_proc = stop_proc(eye_proc)
    await stop_height_worker()
    await stop_pir()
    
    # 3. eye_proc가 완전히 종료될 때까지 대기 (추가!)
    if eye_proc is not None:
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, eye_proc.wait),
                timeout=3.0
            )
            print("🔵[Hub] [Safety] eye_proc 종료 확인")
        except:
            pass
    
    # 4. 카메라 해제 대기
    await asyncio.sleep(1.5)
    print("🔵[Hub] [Safety] 모든 워커 정지 완료")
    
async def stop_workers(eye=False, height=False, pir=False):
    """요청한 워커만 안전하게 정지"""
    print(f"🔵[Hub] [Safety] 선택적 정지 요청: eye={eye}, height={height}, pir={pir}")

    # 1) 내부 워커에게 STOP_ALL 브로드캐스트는 하지 않음
    #    (특정 워커만 끄는 동작이므로)
    #    필요하면 타입별 메시지 보내기

    # 2) 눈 워커
    global eye_proc
    if eye and eye_running():
        eye_proc = stop_proc(eye_proc)
        await asyncio.sleep(0.5)

    # 3) 높이 워커
    if height and height_running():
        await stop_height_worker()

    # 4) PIR 워커
    if pir and pir_running():
        await stop_pir()

    print("🔵[Hub] [Safety] 선택적 정지 완료")

# 대화 주문 핸들러
async def handle_chat_order_on(websocket=None):    
    # # 1. 모든 워커 정지
    # await stop_all_workers_safely()
    
    # 2. 안내 TTS 재생
    loop = asyncio.get_running_loop()
    try:
        # await loop.run_in_executor(None, tts_stt.play_chat_guide_message)
        await loop.run_in_executor(
            None,
            partial(tts_stt.play_chat_guide_message, use_pregenerated=USE_PREGENERATED_GUIDE)
        )
        print("🔵[Hub] [TTS] 안내 종료 → TTS_OFF 전송")
        await send_to_front({"type": "TTS_OFF"})
    except Exception as e:
        print(f"🔵[Hub] [TTS] 안내 실패: {e}", file=sys.stderr)
        await send_to_front({"type": "TTS_ERROR", "message": f"Guide TTS failed: {e}"})

# ====== TTS/STT 핸들러 ======
async def handle_tts_on(websocket, data):
    """TTS_ON 메시지 처리"""
    text = data.get("message") or data.get("text") or ""
    if not str(text).strip():
        await send_json(websocket, {"type": "TTS_ERROR", "message": "Missing TTS text"})
        return
    
    lang = data.get("lang", "ko-KR")
    voice = data.get("voice", None)
    rate = float(data.get("speakingRate", 1.0))
    pitch = float(data.get("pitch", 0.0))
    enc = data.get("audioEncoding", "LINEAR16")
    
    loop = asyncio.get_running_loop()
    try:
        print(f"🔵[Hub] [TTS] 응답 시작: \"{text[:40]}...\" ({enc})")
        await loop.run_in_executor(
            None,
            partial(tts_stt.tts_play, text, lang, voice, rate, pitch, None, enc)
        )
        print("🔵[Hub] [TTS] 응답 종료 → TTS_OFF 전송")
        await send_json(websocket, {"type": "TTS_OFF"})
    except Exception as e:
        print(f"🔵[Hub] [TTS] 오류: {e}", file=sys.stderr)
        await send_json(websocket, {"type": "TTS_ERROR", "message": str(e)})


async def handle_stt_on(websocket, data, loop):
    """STT_ON 메시지 처리"""
    global stt_fail_count, chat_order_processing

    # 1. 파라미터 추출
    duration = int(data.get("duration", tts_stt.STT_MAX_DURATION))
    sample_rate = int(data.get("sampleRate", tts_stt.STT_SAMPLE_RATE))
    language_code = data.get("languageCode", "ko-KR")
    device_idx = data.get("deviceIndex")
    
    # 2. 장치 자동 선택
    if device_idx is None:
        device_idx = tts_stt.find_input_device_index()
        print(f"🔵[Hub] [STT] deviceIndex 자동 선택: {device_idx}")
    
    if device_idx is None:
        await send_json(websocket, {"type": "STT_ERROR", "message": "No input-capable device found"})
        return
    
    # 3. STT 실행
    def run_stt():
        return tts_stt.stt_once(
            mode="auto",
            duration=duration,
            sample_rate=sample_rate,
            language_code=language_code,
            device=device_idx,
            silence_sec=float(data.get("silence", tts_stt.STT_SILENCE_SEC)),
            max_duration=float(data.get("maxDuration", tts_stt.STT_MAX_DURATION)),
            calib_sec=float(data.get("calib", tts_stt.STT_CALIB_SEC)),
            sensitivity=float(data.get("sensitivity", tts_stt.STT_SENSITIVITY)),
            min_speech_sec=float(data.get("minSpeech", tts_stt.STT_MIN_SPEECH_SEC)),
            engine=data.get("engine", tts_stt.STT_ENGINE)
        )
    
    try:
        print(f"🔵[Hub] [STT] (auto) 녹음 시작: max {duration}s @ {sample_rate}Hz (device={device_idx})")
        transcript = await loop.run_in_executor(None, run_stt)
        
        # 🔹 STT가 끝나는 동안 화면이 나가서 대화주문이 종료된 경우 → 결과 무시
        if not chat_order_processing:
            print("🔵[Hub] [STT] chat_order_processing=False → STT 결과 무시 (화면 이탈 중)")
            # 실패 카운트도 초기화해 두는 것이 깔끔
            stt_fail_count = 0
            return
        
        if transcript and transcript.strip():
            # 성공!
            stt_fail_count = 0
            print(f"🔵[Hub] [STT] 성공 결과: \"{transcript}\"")
            await send_json(websocket, {"type": "STT_OFF", "message": transcript})
        else:
            # 실패 - 모든 처리는 handle_stt_failure에서
            await handle_stt_failure(websocket, loop, language_code)
    
    except Exception as e:
        print(f"🔵[Hub] [STT] 예외 발생: {e}", file=sys.stderr)
        await send_json(websocket, {"type": "STT_ERROR", "message": str(e)})


async def handle_stt_failure(websocket, loop, language_code="ko-KR"):
    """STT 실패 처리 (통합)"""
    global stt_fail_count
    stt_fail_count += 1
    print(f"🔵[Hub] [STT] 실패 처리 ({stt_fail_count}/2)")
    
    # 2회 실패 시 주문 취소
    if stt_fail_count >= 2:
        print("🔵[Hub] [ORDER] 무응답 2회 도달 → 주문 취소 플로우 실행")
        
        await send_to_front({"type": "ORDER_CANCEL"})
        
        loop = asyncio.get_running_loop()
        try:
            # await loop.run_in_executor(None, tts_stt.play_cancel_guide_message)
            await loop.run_in_executor(
                None,
                partial(tts_stt.play_cancel_guide_message, use_pregenerated=USE_PREGENERATED_GUIDE)
            )
        except Exception as e:
            print(f"🔵[Hub] [TTS] 취소 안내 실패: {e}", file=sys.stderr)
        
        await send_to_front({"type": "CANCEL_END"})
        
        stt_fail_count = 0
    
    # 1회 실패 시 오류 안내
    else:
        # 1. STT_ERR 전송
        await send_to_front({"type": "STT_ERR"})
        
        # 2. 오류 안내 TTS 재생
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, 
                partial(
                    tts_stt.play_error_guide_message, 
                    lang=language_code,
                    use_pregenerated=USE_PREGENERATED_GUIDE
                )
            )
        except Exception as e:
            print(f"🔵[Hub] [TTS] 오류 안내 실패: {e}", file=sys.stderr)
        
        # 3. ERR_END 전송
        await send_json(websocket, {"type": "ERR_END"})

async def handle_frontend(websocket):
    global eye_proc, frontend_ws, stt_fail_count, eye_calib_processing, eye_calib_completed, mode_select_processing, chat_order_processing, normal_order_processing, eye_order_processing
    print("🔵[Hub] 클라이언트 연결됨")
    
    # 프론트 연결 저장
    frontend_ws = websocket
    clients.add(websocket)
    stt_fail_count = 0

    try:
        async for raw in websocket:
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("JSON 객체가 아님")
            except Exception as e:
                await send_json(websocket, {"type": "ERROR", "message": f"Invalid JSON: {e}"})
                continue

            msg_type = data.get("type")
            msg_text = data.get("message")
            if not msg_type:
                await send_json(websocket, {"type": "ERROR", "message": "Missing 'type' field"})
                continue

            print(f"🔵[Hub] [Front→Hub] 수신: {msg_type} - {msg_text}")

            # === 잠금화면 / PIR ===
            if msg_type == "PIR_ON":
                if mode_select_processing:
                    mode_select_processing = False
                    print(f"🔵[Hub] [PIR_ON] mode_select_processing is True → {mode_select_processing}")
                await start_pir(websocket)

            elif msg_type == "PIR_OFF":
                await send_to_internal_worker({"type": "PIR_OFF"})
                await asyncio.sleep(1.0)
                await stop_pir(websocket)

            # === 높이조절 ===
            elif msg_type == "HEIGHT_SET_ON":                
                # ⬇️ 디바운싱 체크 추가
                global height_last_request_time
                now = time.time()
                
                if now - height_last_request_time < HEIGHT_DEBOUNCE_SEC:
                    print(f"🔵[Hub] [Height] ⚠️ 디바운스 무시 ({now - height_last_request_time:.1f}초 < {HEIGHT_DEBOUNCE_SEC}초)")
                    continue
                
                height_last_request_time = now
                await start_height_worker()

            elif msg_type == "HEIGHT_SET_CANCEL":
                await stop_height_worker()
                await send_to_front({"type": "HEIGHT_SET_CANCEL"})

            # === 조정/보정 ===
            elif msg_type == "EYE_CALIB_ON":
                # 이미 캘리브레이션 완료되었다면 무시
                if eye_calib_completed:
                    print("🔵[Hub] [EYE_CALIB] ✅ 이미 캘리브레이션 완료됨 - 무시")
                    continue

                # 중복 방지: 완료 신호를 받기 전에는 재요청 무시
                if eye_calib_processing:
                    print("🔵[Hub] [EYE_CALIB] ⚠ 이미 진행 중 - 무시")
                    continue

                eye_calib_processing = True  # ✅ 완료 전까지 유지

                # 높이/PIR만 안전 정지
                await stop_workers(eye=False, height=True, pir=True)

                global eye_ready_event, eye_ready_flag

                # 워커 실행 상태 확인
                need_wait_ready = False
                if not eye_running():
                    # 워커 새로 실행 → READY를 새로 기다려야 함
                    print("🔵[Hub] [DEBUG] 🔄 eye_running() == False → start_eye() 호출 예정")
                    loop = asyncio.get_running_loop()
                    try:
                        # await loop.run_in_executor(None, tts_stt.play_calib_guide_message)
                        await loop.run_in_executor(
                            None,
                            partial(tts_stt.play_calib_guide_message, use_pregenerated=USE_PREGENERATED_GUIDE)
                        )
                    except Exception as e:
                        print(f"🔵[Hub] [TTS] 취소 안내 실패: {e}", file=sys.stderr)
                    start_eye()
                    eye_ready_flag = False              
                    need_wait_ready = True
                else:
                    # 이미 실행 중인데 READY를 보낸 적이 없다면 대기
                    print(f"🔵[Hub] [DEBUG] ✓ 이미 실행중 (PID: {eye_proc.pid if eye_proc else 'None'})")
                    need_wait_ready = (not eye_ready_flag)

                if need_wait_ready:
                    eye_ready_event = asyncio.Event()
                    try:
                        await asyncio.wait_for(eye_ready_event.wait(), timeout=30.0)
                        print("🔵[Hub] [EYE_CALIB] worker READY 확인")
                    except asyncio.TimeoutError:
                        print("🔵[Hub] [EYE_CALIB] ❌ READY 타임아웃 발생 → 프론트로 오류 전송")
                        eye_calib_processing = False    # ❗ 실패 시 진행중 플래그 해제
                        await send_to_front({"type": "EYE_CALIB_ERR", "message": "카메라 초기화 타임아웃"})
                        continue
                else:
                    print("🔵[Hub] [EYE_CALIB] 이미 READY 상태 - 대기 생략")

                # 캘리브레이션 명령 전송 (진행중 플래그는 COMPLETE에서만 내림)
                await send_to_internal_worker({"type": "EYE_CALIB_ON"})
                print("🔵[Hub] [EYE_CALIB] 명령 전송 완료")
                

            elif msg_type == "MODE_SELECT_ON":
                # 뒤로가기 대비 모든 주문 상태를 확실히 초기화
                chat_order_processing = False
                normal_order_processing = False
                eye_order_processing = False
                stt_fail_count=0
                
                print(f"🔵[Hub] in MODE_SELECT_ON, mode_select_processing is {mode_select_processing}")
                
                # 대화 주문 등에서 넘어올 때, 재생 중이던 TTS가 있으면 끊기
                try:
                    tts_stt.stop_audio_playback()
                    print("🔵[Hub] [MODE_SELECT] 기존 TTS 재생 중단 요청")
                except Exception as e:
                    print(f"🔵[Hub] [MODE_SELECT] TTS 중단 중 예외: {e}", file=sys.stderr)
                
                if mode_select_processing:
                    print("🔵[Hub] [MODE_SELECT] ⚠ 이미 진행 중 - 무시")
                    continue
                
                mode_select_processing = True
                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(
                        None,
                        partial(tts_stt.play_mode_guide_message, use_pregenerated=USE_PREGENERATED_GUIDE)
                    )
                except Exception as e:
                    print(f"🔵[Hub] [TTS] 취소 안내 실패: {e}", file=sys.stderr)
                
                if eye_calib_completed:
                    if not is_running(eye_proc):
                        print("🔵[Hub] [MODE_SELECT] ⚠ eye_tracking_worker가 실행중이지 않음")
                        await send_to_front({"type": "EYE_ORDER_UNABLE"}) # 추후에 고려해볼 것
                    else:
                        await send_to_internal_worker({"type": "MOUSE_ON"})
                        print("🔵[Hub] [MODE_SELECT] 마우스 제어 ON 명령 전송 완료")
                else:
                    if is_running(eye_proc):
                        await send_to_internal_worker({"type": "MOUSE_OFF"})
                    print("🔵[Hub] [MODE_SELECT] 아이트래킹 마우스 제어를 사용할 수 없음")

            # === 모드 선택 → 대화/일반/눈 ===
            elif msg_type == "CHAT_ORDER_ON":
                if chat_order_processing == True:
                    print("🔵[Hub] [CHAT_ORDER_ON] 이미 대화 주문 진행 중 - 무시")
                    continue
                
                mode_select_processing = False
                chat_order_processing = True
                # 워커는 유지하되 마우스 제어만 OFF
                if is_running(eye_proc):
                    await send_to_internal_worker({"type": "MOUSE_OFF"})
                    print("🔵[Hub] [CHAT_ORDER] 마우스 제어 OFF (워커 유지)")
                    
                await handle_chat_order_on(websocket)

            elif msg_type == "NORMAL_ORDER_ON":
                if normal_order_processing == True:
                    print("🔵[Hub] [NORMAL_ORDER_ON] 이미 일반 주문 진행 중 - 무시")
                    continue
                
                mode_select_processing = False                
                normal_order_processing = True
                # 워커는 유지하되 마우스 제어만 OFF
                if is_running(eye_proc):
                    await send_to_internal_worker({"type": "MOUSE_OFF"})
                    print("🔵[Hub] [NORMAL_ORDER] 마우스 제어 OFF (워커 유지)")

            elif msg_type == "EYE_ORDER_ON":
                if eye_order_processing == True:
                    print("🔵[Hub] [CHAT_ORDER_ON] 이미 대화 주문 진행 중 - 무시")
                    continue
                
                mode_select_processing = False
                eye_order_processing = True
                # eye_tracking_worker가 이미 실행중이면 그대로 유지 (캘리브레이션 보존!)
                if not is_running(eye_proc):
                    print("🔵[Hub] [EYE_ORDER] ⚠️ eye_tracking_worker가 실행중이지 않음. EYE_CALIB_ON을 먼저 실행하세요.")
                    await send_to_front({"type": "ERROR", "message": "Please calibrate first (EYE_CALIB_ON)"})
                else:
                    # 주먹 인식 OFF + 마우스 제어 ON (캘리브레이션 계속 유지)
                    await send_to_internal_worker({"type": "EYE_ORDER_ON"})
                    print("🔵[Hub] [EYE_ORDER] 명령 전송 완료 (캘리브레이션 유지)")

            # === 대화주문 중 TTS/STT ===
            elif msg_type == "TTS_ON":
                if not chat_order_processing:
                    continue
                await handle_tts_on(websocket, data)

            elif msg_type == "STT_ON":
                if not chat_order_processing:
                    continue
                loop = asyncio.get_running_loop()
                await handle_stt_on(websocket, data, loop)

            # === ALL_RESET: 모든 기능 완전 정지 ===
            elif msg_type == "ALL_RESET":
                # 어떤 화면이든 간에, 재생 중인 TTS가 있으면 먼저 중단
                try:
                    tts_stt.stop_audio_playback()
                    print("🔵[Hub] [ALL_RESET] 기존 TTS 재생 중단 요청")
                except Exception as e:
                    print(f"🔵[Hub] [ALL_RESET] TTS 중단 중 예외: {e}", file=sys.stderr)
                
                # 모든 상태 플래그 리셋
                eye_calib_completed = False
                eye_calib_processing = False
                mode_select_processing = False
                chat_order_processing = False
                normal_order_processing = False
                eye_order_processing = False
                
                # 1. 내부 워커들에게 정지 신호 먼저 전송
                await send_to_internal_worker({"type": "STOP_ALL"})
                await asyncio.sleep(0.5)
                
                # 2. 모든 프로세스 종료
                eye_proc = stop_proc(eye_proc)
                await stop_height_worker()
                await stop_pir()
                await asyncio.sleep(0.5)
                
                # 4. 프론트에 완료 알림
                await send_to_front({"type": "ALL_RESET_COMPLETE"})
                print("🔵[Hub] [ALL_RESET] 완료")

            else:
                await send_json(websocket, {"type": "ERROR", "message": f"Unknown type: {msg_type}"})

    except websockets.exceptions.ConnectionClosed:
        print("🔵[Hub] 클라이언트 연결 끊김")
    finally:
        clients.discard(websocket)
        stt_fail_count = 0
        frontend_ws = None

# === 라즈베리파이 내부 통신 핸들러 ===
async def handle_internal_worker(websocket):
    """eye_tracking_worker 및 height_worker의 WebSocket 연결 처리 (내부 통신용)"""
    global internal_workers, height_proc, height_set_processing, eye_ready_event, eye_calib_completed, eye_calib_processing, eye_ready_flag
    internal_workers.add(websocket)
    print("🔵[Hub] [Internal Worker] 연결됨 (포트 8766)")
    
    try:
        async for raw in websocket:
            data = json.loads(raw)
            msg_type = data.get("type")
            print(f"🔵[Hub] [Worker→Hub] 수신: {msg_type}")
            
            if msg_type == "EYE_READY":
                print("🔵[Hub] eye worker READY")
                eye_ready_flag = True         
                if eye_ready_event is not None:
                    eye_ready_event.set()

            elif msg_type == "EYE_CALIB_COMPLETE":
                print("🔵[Hub] ✅ 캘리브레이션 완료 확인")
                eye_calib_completed = True
                eye_calib_processing = False         
                await send_to_front({"type": "EYE_CALIB_END"})

            elif msg_type == "EYE_CALIB_ERR":
                print(f"🔵[Hub] ❌ 캘리브레이션 실패: {data.get('message', '알 수 없는 오류')}")
                eye_calib_completed = False
                eye_calib_processing = False          
                await send_to_front({"type": "EYE_CALIB_ERR"})

            # PIR 감지
            elif msg_type == "PIR_DETECTED":
                await send_to_front({"type": "PIR_DETECTED"})
                
            # PIR 종료 감지
            elif msg_type == "PIR_END":
                await send_to_front({"type": "PIR_END"})
                
            # PIR 워커 정상 종료 (WebSocket 연결 끊김 감지)
            elif msg_type == "PIR_WORKER_EXIT":
                await send_to_front({"type": "PIR_END"})
                workers["PIR"] = None
            
            # 주먹 감지
            elif msg_type == "FIST_DETECTED":
                await send_to_front({"type": "FIST_DETECTED"})
            
            # 높이 조절 완료
            elif msg_type == "HEIGHT_SET_END":
                await send_to_front({"type": "HEIGHT_SET_END"})
                height_proc = None
                height_set_processing = False
            
            # 높이 조절 타임아웃
            elif msg_type == "HEIGHT_SET_TIMEOUT":
                print("🔵[Hub] ⚠️ 높이 조절 타임아웃 (30초간 미감지)")
                await send_to_front({"type": "HEIGHT_SET_CANCEL"})
                height_proc = None
                height_set_processing = False
            
            # 높이 조절 취소됨
            elif msg_type == "HEIGHT_SET_CANCEL":
                print("🔵[Hub] 🚫 높이 조절 취소됨")
                await send_to_front({"type": "HEIGHT_SET_CANCEL"})
                height_proc = None
                height_set_processing = False
                
            
            # 높이 조절에 에러가 발생
            elif msg_type == "HEIGHT_SET_ERR":
                print("🔵[Hub] ⚠️ 높이 조절 에러 발생")
                await send_to_front({"type": "HEIGHT_SET_ERR"})
                height_proc = None
                height_set_processing = False
    
    except websockets.exceptions.ConnectionClosed:
        print("🔵[Hub] [Internal Worker] 연결 끊김")
        # PIR 워커가 비정상 종료된 경우도 처리
        if workers.get("PIR") and workers["PIR"].poll() is None:
            print("🔵[Hub] PIR 워커 비정상 종료 감지")
            await send_to_front({"type": "PIR_END"})
            workers["PIR"] = None
    finally:
        try:
            internal_workers.discard(websocket)
        except Exception:
            pass

async def main():
    print("=" * 60)
    print("서버 시작됨")
    print("  - 프론트엔드: ws://0.0.0.0:8765")
    print("  - Internal Workers: ws://localhost:8766")
    print("=" * 60)
    
    # 프론트엔드용 서버 (포트 8765)
    frontend_server = websockets.serve(handle_frontend, "0.0.0.0", 8765)
    
    # eye_tracking_worker + height_worker용 내부 서버 (포트 8766)
    internal_server = websockets.serve(handle_internal_worker, "localhost", 8766)
    
    async with frontend_server, internal_server:
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())