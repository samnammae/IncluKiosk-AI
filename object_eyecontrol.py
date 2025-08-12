import argparse
import asyncio
import signal
import sys
import time
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import websockets

# ===== 로깅 설정 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [EYE] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== 인자 파싱 =====
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["mode_select", "eye_only"], default="mode_select")
parser.add_argument("--server", default="ws://localhost:8765")
args = parser.parse_args()

MODE = args.mode            # mode_select: 눈+주먹, eye_only: 눈만
SERVER_URI = args.server

# ===== 종료 관리 =====
_running = True
_loop = None

def _handle_stop(signum, frame):
    global _running
    _running = False
    logger.info(f"Signal {signum} received, stopping...")

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)

# ===== pyautogui 설정 =====
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

try:
    screen_w, screen_h = pyautogui.size()
    logger.info(f"화면 크기: {screen_w}x{screen_h}")
except Exception as e:
    logger.error(f"화면 크기 획득 실패: {e}")
    screen_w, screen_h = 1920, 1080  # 기본값

# ===== MediaPipe 초기화 =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = None

if MODE == "mode_select":
    try:
        hands = mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info("MediaPipe Hands 초기화 완료")
    except Exception as e:
        logger.error(f"MediaPipe Hands 초기화 실패: {e}")
        hands = None

# Face Mesh 초기화
mp_face = mp.solutions.face_mesh
face_mesh = None
use_iris = False

try:
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # 홍채 추적
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    use_iris = True
    logger.info("FaceMesh 초기화 완료 (iris 추적 지원)")
except Exception as e:
    logger.warning(f"FaceMesh refine 실패: {e} -> fallback으로 재시도")
    try:
        face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        use_iris = False
        logger.info("FaceMesh 초기화 완료 (기본 모드)")
    except Exception as e:
        logger.error(f"FaceMesh 초기화 완전 실패: {e}")
        face_mesh = None

# ===== MediaPipe 랜드마크 상수 =====
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_CORNERS = [33, 133]
LEFT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_CORNERS = [362, 263]
RIGHT_EYE_TOP_BOTTOM = [386, 374]

LEFT_EYE_MAIN = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_MAIN = [362, 385, 387, 263, 373, 380]

# ===== 설정값 =====
EAR_THRESHOLD = 0.25
BLINK_FRAMES = 2
DOUBLE_BLINK_TIME = 1.0
PROCESSING_INTERVAL = 10  # N프레임마다 처리

# ===== 상태 변수 =====
blink_counter = 0
total_blinks = 0
blink_times = []
last_click_time = 0
frame_count = 0
smoothing_factor = 0.3
prev_x, prev_y = screen_w // 2, screen_h // 2

# ===== 카메라 초기화 =====
cap = None
try:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise Exception("카메라를 열 수 없습니다")
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 카메라 테스트
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        raise Exception("카메라에서 프레임을 읽을 수 없습니다")
    
    logger.info("카메라 초기화 완료")
    
except Exception as e:
    logger.error(f"카메라 초기화 실패: {e}")
    if cap:
        cap.release()
    cap = None

logger.info(f"모드: {MODE}, 서버: {SERVER_URI}")
logger.info("'q' 키를 눌러 종료")

# ===== 도우미 함수 =====
def calculate_ear(landmarks, eye_points, frame_w, frame_h):
    """Eye Aspect Ratio 계산"""
    try:
        points = []
        for point_idx in eye_points:
            x = landmarks.landmark[point_idx].x * frame_w
            y = landmarks.landmark[point_idx].y * frame_h
            points.append([x, y])
        
        if len(points) != 6:
            return 0
        
        p1, p2, p3, p4, p5, p6 = points
        dist1 = np.linalg.norm(np.array(p2) - np.array(p6))
        dist2 = np.linalg.norm(np.array(p3) - np.array(p5))
        dist3 = np.linalg.norm(np.array(p1) - np.array(p4))
        
        if dist3 == 0:
            return 0
        return (dist1 + dist2) / (2.0 * dist3)
    except Exception as e:
        logger.debug(f"EAR 계산 오류: {e}")
        return 0

def estimate_iris_from_eye(landmarks, eye_corners, eye_tb, frame_w, frame_h):
    """홍채 위치 추정 (fallback용)"""
    try:
        left_corner = [landmarks.landmark[eye_corners[0]].x * frame_w,
                       landmarks.landmark[eye_corners[0]].y * frame_h]
        right_corner = [landmarks.landmark[eye_corners[1]].x * frame_w,
                        landmarks.landmark[eye_corners[1]].y * frame_h]
        top_point = [landmarks.landmark[eye_tb[0]].x * frame_w,
                     landmarks.landmark[eye_tb[0]].y * frame_h]
        bottom_point = [landmarks.landmark[eye_tb[1]].x * frame_w,
                        landmarks.landmark[eye_tb[1]].y * frame_h]
        
        center_x = (left_corner[0] + right_corner[0]) / 2
        center_y = (top_point[1] + bottom_point[1]) / 2
        return int(center_x), int(center_y)
    except Exception as e:
        logger.debug(f"홍채 위치 추정 오류: {e}")
        return None, None

def is_fist(hand_landmarks):
    """주먹 감지: 모든 손가락이 접혔는지 확인"""
    try:
        FIST_LANDMARKS = [8, 12, 16, 20]  # index/middle/ring/pinky tips
        folded_count = 0
        
        for tip_idx in FIST_LANDMARKS:
            tip = hand_landmarks.landmark[tip_idx]
            pip = hand_landmarks.landmark[tip_idx - 2]
            if tip.y > pip.y:  # 손가락 끝이 아래에 있으면 접힌 것
                folded_count += 1
        
        # 4개 손가락 중 3개 이상 접혔으면 주먹으로 판단
        return folded_count >= 3
    except Exception as e:
        logger.debug(f"주먹 감지 오류: {e}")
        return False

# ===== WebSocket 통신 =====
def send_message_sync(payload: dict):
    """동기 방식으로 메시지 전송 (별도 스레드에서 실행)"""
    try:
        # 새 이벤트 루프 생성해서 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _send():
            async with websockets.connect(SERVER_URI) as ws:
                await ws.send(json.dumps(payload))
        
        loop.run_until_complete(_send())
        loop.close()
        logger.info(f"메시지 전송 성공: {payload}")
        return True
    except Exception as e:
        logger.error(f"메시지 전송 실패: {e}")
        return False

# 스레드 풀 생성
executor = ThreadPoolExecutor(max_workers=2)

def send_message_async(payload: dict):
    """비동기로 메시지 전송"""
    future = executor.submit(send_message_sync, payload)
    return future

# ===== 메인 처리 루프 =====
def main():
    global frame_count, prev_x, prev_y, _running
    
    if not cap:
        logger.error("카메라가 초기화되지 않아 종료합니다")
        return
    
    if not face_mesh:
        logger.error("FaceMesh가 초기화되지 않아 종료합니다")
        return
    
    logger.info(f"아이트래킹 시작 - 모드: {MODE}")
    
    try:
        while _running and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("카메라 프레임을 읽을 수 없습니다")
                time.sleep(0.1)
                continue

            frame_count += 1
            
            # 성능 최적화: N프레임마다 처리
            if frame_count % PROCESSING_INTERVAL != 0:
                cv2.imshow("Eye Tracking Mouse", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ===== 손(주먹) 처리: mode_select에서만 =====
            if MODE == "mode_select" and hands is not None:
                try:
                    hand_results = hands.process(frame_rgb)
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            # 손 랜드마크 그리기
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                            )
                            
                            # 주먹 감지
                            if is_fist(hand_landmarks):
                                logger.info("주먹 감지! CHAT_ORDER_ON 전송 후 종료")
                                # 비동기 메시지 전송
                                send_message_async({
                                    "type": "CHAT_ORDER_ON", 
                                    "source": "worker"
                                })
                                _running = False
                                break
                        
                        if not _running:
                            break
                except Exception as e:
                    logger.error(f"손 처리 오류: {e}")

            # ===== 얼굴/눈 처리 =====
            try:
                face_results = face_mesh.process(frame_rgb)
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        try:
                            # 홍채 위치 획득
                            if use_iris:
                                left_coords = np.array([
                                    [int(face_landmarks.landmark[i].x * w), 
                                     int(face_landmarks.landmark[i].y * h)]
                                    for i in LEFT_IRIS
                                ])
                                if len(left_coords) > 0:
                                    lx, ly = np.mean(left_coords, axis=0).astype(int)
                                else:
                                    continue
                            else:
                                lx, ly = estimate_iris_from_eye(
                                    face_landmarks, LEFT_EYE_CORNERS, 
                                    LEFT_EYE_TOP_BOTTOM, w, h
                                )
                                if lx is None or ly is None:
                                    continue

                            # 화면 좌표 변환 (좌우 반전)
                            screen_x = np.interp(w - lx, [0, w], [0, screen_w])
                            screen_y = np.interp(ly, [0, h], [0, screen_h])

                            # 스무딩 적용
                            smooth_x = prev_x * (1 - smoothing_factor) + screen_x * smoothing_factor
                            smooth_y = prev_y * (1 - smoothing_factor) + screen_y * smoothing_factor

                            # 마우스 이동 (시각화)
                            try:
                                pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)
                            except Exception as e:
                                logger.debug(f"마우스 이동 오류: {e}")

                            prev_x, prev_y = smooth_x, smooth_y

                            # EAR 계산 (깜빡임 감지용 - 향후 확장 가능)
                            left_ear = calculate_ear(face_landmarks, LEFT_EYE_MAIN, w, h)
                            right_ear = calculate_ear(face_landmarks, RIGHT_EYE_MAIN, w, h)
                            avg_ear = (left_ear + right_ear) / 2.0

                            # 시각화
                            cv2.circle(frame, (lx, ly), 3, (0, 255, 255), -1)
                            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            cv2.putText(frame, f"Mode: {MODE}", (10, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        except Exception as e:
                            logger.debug(f"얼굴 랜드마크 처리 오류: {e}")
                            continue
            except Exception as e:
                logger.error(f"얼굴 처리 오류: {e}")

            # 화면 중앙 십자선 (캘리브레이션용)
            cv2.line(frame, (w//2-10, h//2), (w//2+10, h//2), (0, 0, 255), 1)
            cv2.line(frame, (w//2, h//2-10), (w//2, h//2+10), (0, 0, 255), 1)

            # 화면 표시
            cv2.imshow("Eye Tracking Mouse", frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("사용자가 'q'를 눌러 종료")
                break

    except Exception as e:
        logger.error(f"메인 루프 오류: {e}")
    finally:
        cleanup()

def cleanup():
    """리소스 정리"""
    global cap, executor
    
    logger.info("리소스 정리 중...")
    
    try:
        if cap:
            cap.release()
    except:
        pass
    
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    try:
        if executor:
            executor.shutdown(wait=True)
    except:
        pass
    
    logger.info("리소스 정리 완료")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("사용자 인터럽트로 종료")
    except Exception as e:
        logger.error(f"실행 중 오류: {e}")
    finally:
        logger.info("아이트래킹 종료")