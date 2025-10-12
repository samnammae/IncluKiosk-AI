import cv2
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import detect
from linear_actuator.linear_actuator_controller import (
    init_motor,
    cleanup_motor,
    moveUp, 
    moveDown,
    exceed_max_height,
    exceed_min_height 
)

# ====== 설정 ======
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
MIN_DET_CONF = 0.5  # 얼굴 검출 임계값
PERSON_SCORE_TH = 0.4  # 사람 검출 임계값

DEADBAND_PCT = 0.06  # 중앙 허용 범위
EMA_ALPHA = 0.3  # 스무딩 계수
STABLE_FRAMES = 10  # 안정화 프레임 수

WITH_FACE = 100  # 얼굴 있을 때 모터 속도
WITHOUT_FACE = 500  # 얼굴 없을 때 모터 속도

AUTO_EXIT_STABLE_TIME = 3  # 3초간 안정화 유지하면 자동 종료
NO_DETECTION_TIMEOUT = 15   # 15초간 미감지 시 종료

# 모델 경로
FACE_MODEL = "models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
PERSON_MODEL = "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"

# ===================


def detect_faces_ssd(interpreter, frame_bgr, score_threshold=0.5):
    """SSD로 얼굴 감지"""
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    
    H, W = frame_bgr.shape[:2]
    if H <= 0 or W <= 0:
        return []
    
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        width, height = detect.input_size(interpreter)
        resized = cv2.resize(frame_rgb, (width, height))
        
        tensor = detect.input_tensor(interpreter)
        tensor.fill(0)
        tensor[:, :] = resized.copy()
        del tensor
        
        interpreter.invoke()
        objs = detect.get_output(interpreter, score_threshold, (1.0, 1.0))
        
        faces = []
        for obj in objs:
            bbox = obj.bbox
            xmin = max(0.0, min(1.0, bbox.xmin / width))
            ymin = max(0.0, min(1.0, bbox.ymin / height))
            xmax = max(0.0, min(1.0, bbox.xmax / width))
            ymax = max(0.0, min(1.0, bbox.ymax / height))
            faces.append((xmin, ymin, xmax, ymax, obj.score))
        
        return faces
    except Exception as e:
        print(f"[Face] 오류: {e}")
        return []


def detect_person_ssd(interpreter, frame_bgr, score_threshold=0.4):
    """SSD로 사람 감지 (COCO class_id=0)"""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    
    H, W = frame_bgr.shape[:2]
    if H <= 0 or W <= 0:
        return None
    
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        width, height = detect.input_size(interpreter)
        resized = cv2.resize(frame_rgb, (width, height))
        
        tensor = detect.input_tensor(interpreter)
        tensor.fill(0)
        tensor[:, :] = resized.copy()
        del tensor
        
        interpreter.invoke()
        objs = detect.get_output(interpreter, score_threshold, (1.0, 1.0))
        
        best = None
        best_area = -1.0
        
        for obj in objs:
            if obj.id != 0:  # class_id=0 is person in COCO
                continue
            
            bbox = obj.bbox
            xmin = max(0.0, min(1.0, bbox.xmin / width))
            ymin = max(0.0, min(1.0, bbox.ymin / height))
            xmax = max(0.0, min(1.0, bbox.xmax / width))
            ymax = max(0.0, min(1.0, bbox.ymax / height))
            
            area = (xmax - xmin) * (ymax - ymin)
            if area > best_area:
                best_area = area
                best = (xmin, ymin, xmax, ymax)
        
        return best
    except Exception as e:
        print(f"[Person] 오류: {e}")
        return None


def draw_detection(frame, faces, person, state, ema_y, target_y, deadband, stable_count):
    """화면에 검출 결과 및 상태 표시"""
    H, W = frame.shape[:2]
    
    # 타겟 라인 (중앙)
    target_line_y = int(target_y * H)
    cv2.line(frame, (0, target_line_y), (W, target_line_y), (255, 255, 0), 2)
    
    # 데드밴드 영역
    deadband_top = int((target_y - deadband) * H)
    deadband_bottom = int((target_y + deadband) * H)
    cv2.rectangle(frame, (0, deadband_top), (W, deadband_bottom), (0, 255, 255), 1)
    
    # 얼굴 바운딩 박스
    if faces:
        for xmin, ymin, xmax, ymax, score in faces:
            x1, y1 = int(xmin * W), int(ymin * H)
            x2, y2 = int(xmax * W), int(ymax * H)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Face {score*100:.0f}%', (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 얼굴 중심점
            center_x = int((xmin + xmax) * 0.5 * W)
            center_y = int((ymin + ymax) * 0.5 * H)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    
    # 사람 바운딩 박스 (얼굴 없을 때)
    elif person:
        xmin, ymin, xmax, ymax = person
        x1, y1 = int(xmin * W), int(ymin * H)
        x2, y2 = int(xmax * W), int(ymax * H)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, 'Person', (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # EMA 라인
    if ema_y is not None:
        ema_line_y = int(ema_y * H)
        cv2.line(frame, (0, ema_line_y), (W, ema_line_y), (0, 255, 0), 1)
    
    # 상태 텍스트
    status_color = (0, 255, 0) if state == "center" else (0, 165, 255)
    status_text = f"State: {state}"
    if state == "center":
        status_text += f" ({stable_count}/{STABLE_FRAMES})"
    
    cv2.putText(frame, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # 안내 메시지
    cv2.putText(frame, "Press ESC to quit", (10, H-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def main():
    print("=" * 50)
    print("높이 조절 테스트 프로그램")
    print("=" * 50)
    
    # 모터 초기화
    try:
        print("[1/4] 모터 초기화 중...")
        init_motor()
        print("✅ 모터 초기화 완료")
    except Exception as e:
        print(f"❌ 모터 초기화 실패: {e}")
        return
    
    # 카메라 초기화
    print("[2/4] 카메라 초기화 중...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"❌ 카메라 열기 실패 (인덱스: {CAM_INDEX})")
        cleanup_motor()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ 카메라 준비 완료: {actual_w}x{actual_h}")
    
    # 모델 로드
    print("[3/4] AI 모델 로드 중...")
    try:
        face_interpreter = tflite.Interpreter(
            model_path=FACE_MODEL,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        face_interpreter.allocate_tensors()
        print("✅ 얼굴 검출 모델 로드 완료 (EdgeTPU)")
        
        person_interpreter = tflite.Interpreter(
            model_path=PERSON_MODEL,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        person_interpreter.allocate_tensors()
        print("✅ 사람 검출 모델 로드 완료 (EdgeTPU)")
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        cap.release()
        cleanup_motor()
        return
    
    print("[4/4] 높이 조절 시작!")
    print("-" * 50)
    print("📷 화면에서 얼굴/사람 위치를 확인하세요")
    print("🎯 노란색 라인: 목표 중앙선")
    print("⚡ ESC 키: 종료")
    print("-" * 50)
    
    # 추적 변수
    target_y = 0.5
    deadband = DEADBAND_PCT
    ema_y = None
    stable_count = 0
    stable_start_time = None
    last_detection_time = time.time()
    
    # FPS 측정
    fps_start = time.time()
    fps_count = 0
    current_fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ 프레임 읽기 실패")
                time.sleep(0.1)
                continue
            
            # FPS 계산
            fps_count += 1
            if fps_count >= 30:
                fps_end = time.time()
                current_fps = fps_count / (fps_end - fps_start)
                fps_count = 0
                fps_start = time.time()
            
            # 얼굴 검출
            faces = detect_faces_ssd(face_interpreter, frame, MIN_DET_CONF)
            
            state = None
            detection_found = False
            person = None
            
            if faces:
                # 얼굴 검출 성공
                detection_found = True
                face = max(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]))
                xmin, ymin, xmax, ymax, score = face
                
                y_center = (ymin + ymax) * 0.5
                ema_y = y_center if ema_y is None else EMA_ALPHA * y_center + (1 - EMA_ALPHA) * ema_y
                diff = ema_y - target_y
                
                if abs(diff) <= deadband:
                    state = "center"
                    stable_count += 1
                elif diff < 0:
                    state = "up"
                    stable_count = 0
                    print("⬇️ 아래로 이동 (얼굴이 위쪽)")
                    moveUp(WITH_FACE)
                    
                    if exceed_max_height():
                        print("🚫 최대 높이 도달!")
                        break
                else:
                    state = "down"
                    stable_count = 0
                    print("⬆️ 위로 이동 (얼굴이 아래쪽)")
                    moveDown(WITH_FACE)
                    
                    if exceed_min_height():
                        print("🚫 최소 높이 도달!")
                        break
            
            else:
                # 얼굴 없음 → 사람 검출
                person = detect_person_ssd(person_interpreter, frame, PERSON_SCORE_TH)
                stable_count = 0
                
                if person is None:
                    state = "hint_down"
                    print("👤 사람 미감지 → 아래로 스캔")
                    moveDown(WITHOUT_FACE)
                    
                    if exceed_min_height():
                        print("🚫 최소 높이 도달!")
                        break
                else:
                    detection_found = True
                    x0, y0, x1, y1 = person
                    
                    if y0 <= 0.05:
                        state = "hint_up"
                        print("👤 사람이 위쪽 → 위로 스캔")
                        moveUp(WITHOUT_FACE)
                        
                        if exceed_max_height():
                            print("🚫 최대 높이 도달!")
                            break
                    elif y1 >= 0.95:
                        state = "hint_down"
                        print("👤 사람이 아래쪽 → 아래로 스캔")
                        moveDown(WITHOUT_FACE)
                        
                        if exceed_min_height():
                            print("🚫 최소 높이 도달!")
                            break
                    else:
                        y_center = (y0 + y1) * 0.5
                        state = "up" if y_center < target_y - deadband else "down"
            
            # 감지 시간 업데이트
            if detection_found:
                last_detection_time = time.time()
            
            # 타임아웃 체크
            if time.time() - last_detection_time > NO_DETECTION_TIMEOUT:
                print(f"⏱️ {NO_DETECTION_TIMEOUT}초간 미감지 → 종료")
                break
            
            # 자동 종료 체크
            if state == "center" and stable_count >= STABLE_FRAMES:
                if stable_start_time is None:
                    stable_start_time = time.time()
                    print("✅ 중앙 안정화 시작")
                elif time.time() - stable_start_time >= AUTO_EXIT_STABLE_TIME:
                    print(f"🎉 {AUTO_EXIT_STABLE_TIME}초간 안정화 완료!")
                    break
            else:
                stable_start_time = None
            
            # 화면 표시
            display_frame = draw_detection(frame, faces, person, state, ema_y, 
                                          target_y, deadband, stable_count)
            
            # FPS 표시
            cv2.putText(display_frame, f'FPS: {current_fps:.1f}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Height Adjustment Test', display_frame)
            
            # ESC 키로 종료
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("🛑 사용자 종료")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C로 중단됨")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n정리 중...")
        cap.release()
        cv2.destroyAllWindows()
        cleanup_motor()
        print("✅ 종료 완료")


if __name__ == "__main__":
    main()