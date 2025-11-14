#!/usr/bin/env python3
"""
주먹 감지 테스트 프로그램
실시간으로 카메라에서 주먹을 감지하고 시각화
"""

import cv2
import numpy as np
import mediapipe as mp
import time

# ============ 설정 ============
CAMERA_INDEX = 0
MIN_HAND_SIZE = 100  # 최소 손 크기 (픽셀)
FIST_HOLD_TIME = 2.0  # 주먹 유지 시간 (초)

# ============ MediaPipe 초기화 ============
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============ 주먹 감지 함수들 ============
def _lm_xy(hand_landmarks, idx, w, h):
    """랜드마크 XY 좌표 추출"""
    lm = hand_landmarks.landmark[idx]
    return np.array([lm.x * w, lm.y * h], dtype=float)


def is_finger_curled(hand_landmarks, tip_idx, pip_idx, wrist_idx, w, h):
    """손가락이 구부러졌는지 확인"""
    tip = _lm_xy(hand_landmarks, tip_idx, w, h)
    pip = _lm_xy(hand_landmarks, pip_idx, w, h)
    wrist = _lm_xy(hand_landmarks, wrist_idx, w, h)
    return np.linalg.norm(tip - wrist) < np.linalg.norm(pip - wrist)


def is_thumb_curled(hand_landmarks, w, h):
    """엄지가 구부러졌는지 확인"""
    wrist = _lm_xy(hand_landmarks, 0, w, h)
    tip = _lm_xy(hand_landmarks, 4, w, h)
    mcp = _lm_xy(hand_landmarks, 2, w, h)
    return np.linalg.norm(tip - wrist) < np.linalg.norm(mcp - wrist)


def is_fist(hand_landmarks, w, h, min_hand_size=100):
    """
    주먹 제스처 감지
    
    Returns:
        tuple: (is_fist: bool, hand_size: float, curled_count: int)
    """
    # 1. 손 크기 체크 (손목 ~ 중지 끝 거리)
    wrist = _lm_xy(hand_landmarks, 0, w, h)
    middle_tip = _lm_xy(hand_landmarks, 12, w, h)
    hand_size = np.linalg.norm(middle_tip - wrist)
    
    if hand_size < min_hand_size:
        return False, hand_size, 0
    
    # 2. 손가락 구부림 체크
    curled = 0
    curled += int(is_finger_curled(hand_landmarks, 8, 6, 0, w, h))   # 검지
    curled += int(is_finger_curled(hand_landmarks, 12, 10, 0, w, h)) # 중지
    curled += int(is_finger_curled(hand_landmarks, 16, 14, 0, w, h)) # 약지
    curled += int(is_finger_curled(hand_landmarks, 20, 18, 0, w, h)) # 소지
    curled += int(is_thumb_curled(hand_landmarks, w, h))              # 엄지
    
    return (curled >= 5), hand_size, curled


def draw_text_with_background(frame, text, position, font_scale=1.0, 
                               thickness=2, text_color=(255, 255, 255), 
                               bg_color=(0, 0, 0), padding=10):
    """배경이 있는 텍스트 그리기"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    # 배경 사각형
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1
    )
    # 텍스트
    cv2.putText(
        frame, text, (x, y),
        font, font_scale, text_color, thickness, cv2.LINE_AA
    )


# ============ 메인 루프 ============
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("=" * 60)
    print("🎥 주먹 감지 테스트 프로그램")
    print("=" * 60)
    print(f"📹 해상도: {w}x{h}")
    print(f"✊ 최소 손 크기: {MIN_HAND_SIZE} 픽셀")
    print(f"⏱️  유지 시간: {FIST_HOLD_TIME}초")
    print("\n[조작법]")
    print("  - Q: 종료")
    print("  - +/-: 최소 손 크기 조정")
    print("  - [/]: 유지 시간 조정")
    print("=" * 60)
    
    # 주먹 감지 상태
    fist_start_time = None
    fist_detected = False
    
    # 설정값 (동적 조정 가능)
    min_hand_size = MIN_HAND_SIZE
    hold_time = FIST_HOLD_TIME
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)
        
        # RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        current_fist_detected = False
        hand_info = []
        
        # 손 감지 및 그리기
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # 주먹 감지
                is_fist_result, hand_size, curled_count = is_fist(
                    hand_landmarks, w, h, min_hand_size=min_hand_size
                )
                
                hand_info.append({
                    'hand_size': hand_size,
                    'curled_count': curled_count,
                    'is_fist': is_fist_result
                })
                
                if is_fist_result:
                    current_fist_detected = True
                
                # 손목 위치에 정보 표시
                wrist = _lm_xy(hand_landmarks, 0, w, h)
                info_x = int(wrist[0]) + 20
                info_y = int(wrist[1])
                
                # 손 크기
                size_color = (0, 255, 0) if hand_size >= min_hand_size else (0, 0, 255)
                cv2.putText(frame, f"Size: {hand_size:.0f}px", 
                           (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, size_color, 2, cv2.LINE_AA)
                
                # 구부러진 손가락 수
                finger_color = (0, 255, 0) if curled_count >= 5 else (255, 255, 0)
                cv2.putText(frame, f"Fingers: {curled_count}/5", 
                           (info_x, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, finger_color, 2, cv2.LINE_AA)
        
        # 주먹 유지 시간 체크
        current_time = time.time()
        hold_duration = 0.0
        
        if current_fist_detected:
            if fist_start_time is None:
                fist_start_time = current_time
            else:
                hold_duration = current_time - fist_start_time
                if hold_duration >= hold_time and not fist_detected:
                    fist_detected = True
                    print(f"✅ 주먹 인식 완료! ({hold_duration:.1f}초)")
        else:
            if fist_start_time is not None:
                fist_start_time = None
            if fist_detected:
                fist_detected = False
        
        # ============ UI 오버레이 ============
        
        # 상단 설정 정보
        draw_text_with_background(
            frame,
            f"Min Hand Size: {min_hand_size}px  |  Hold Time: {hold_time:.1f}s",
            (10, 30),
            font_scale=0.7,
            thickness=2,
            bg_color=(50, 50, 50)
        )
        
        # 주먹 상태 표시
        if fist_detected:
            # 주먹 인식 완료
            draw_text_with_background(
                frame,
                "FIST DETECTED!",
                (w // 2 - 200, h // 2),
                font_scale=2.0,
                thickness=4,
                text_color=(0, 255, 0),
                bg_color=(0, 100, 0),
                padding=20
            )
        elif current_fist_detected and fist_start_time is not None:
            # 주먹 유지 중
            progress = (hold_duration / hold_time) * 100
            draw_text_with_background(
                frame,
                f"Holding... {hold_duration:.1f}s / {hold_time:.1f}s ({progress:.0f}%)",
                (w // 2 - 250, h // 2),
                font_scale=1.2,
                thickness=2,
                text_color=(0, 255, 255),
                bg_color=(100, 100, 0),
                padding=15
            )
            
            # 프로그레스 바
            bar_width = 400
            bar_height = 30
            bar_x = w // 2 - bar_width // 2
            bar_y = h // 2 + 50
            
            # 배경
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            # 진행
            progress_width = int(bar_width * (hold_duration / hold_time))
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + progress_width, bar_y + bar_height),
                         (0, 255, 255), -1)
            # 테두리
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 2)
        
        # 하단 도움말
        help_text = "Press: Q=Quit  |  +/-=Hand Size  |  [/]=Hold Time"
        draw_text_with_background(
            frame,
            help_text,
            (10, h - 20),
            font_scale=0.6,
            thickness=1,
            bg_color=(50, 50, 50)
        )
        
        # 프레임 표시
        cv2.imshow('Fist Detection Test', frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q 또는 ESC
            break
        elif key == ord('+') or key == ord('='):  # 손 크기 증가
            min_hand_size += 10
            print(f"최소 손 크기: {min_hand_size}px")
        elif key == ord('-') or key == ord('_'):  # 손 크기 감소
            min_hand_size = max(50, min_hand_size - 10)
            print(f"최소 손 크기: {min_hand_size}px")
        elif key == ord(']'):  # 유지 시간 증가
            hold_time += 0.5
            print(f"유지 시간: {hold_time:.1f}초")
        elif key == ord('['):  # 유지 시간 감소
            hold_time = max(0.5, hold_time - 0.5)
            print(f"유지 시간: {hold_time:.1f}초")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n프로그램 종료")


if __name__ == "__main__":
    main()