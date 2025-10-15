"""얼굴 감지 데모 - 카메라로 실시간 얼굴 감지 테스트 (시각화 개선)"""
import cv2
import time
import tflite_runtime.interpreter as tflite
from . import config
from . import detection


def draw_guide_lines(image):
    """중앙 가이드 라인 그리기"""
    h, w = image.shape[:2]
    
    # 중앙 세로선
    cv2.line(image, (w//2, 0), (w//2, h), (255, 255, 0), 1)
    
    # 중앙 가로선
    cv2.line(image, (0, h//2), (w, h//2), (255, 255, 0), 1)
    
    # 목표 영역 (중앙에서 8% 위)
    target_offset = int(h * config.TARGET_OFFSET_PCT)
    target_y = h//2 - target_offset
    cv2.line(image, (0, target_y), (w, target_y), (0, 255, 255), 2)
    
    # 데드밴드 영역 (회색 박스)
    deadband = int(h * config.DEADBAND_PCT)
    cv2.rectangle(
        image,
        (w//2 - 50, target_y - deadband),
        (w//2 + 50, target_y + deadband),
        (128, 128, 128),
        1
    )


def draw_detections(image, faces, person=None):
    """
    감지된 얼굴과 사람을 이미지에 그리기
    faces: [(xmin, ymin, xmax, ymax, score), ...] 정규화된 좌표 (0~1)
    person: (xmin, ymin, xmax, ymax) 정규화된 좌표 또는 None
    """
    h, w = image.shape[:2]
    
    # 얼굴 바운딩 박스
    for i, face in enumerate(faces):
        xmin, ymin, xmax, ymax, score = face
        
        # 정규화된 좌표를 픽셀 좌표로 변환
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        
        # 바운딩 박스 (초록색, 두껍게)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 중심점
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(image, (center_x, center_y), 8, (0, 0, 255), -1)
        
        # 신뢰도 표시 (배경 추가로 가독성 향상)
        label = f'Face {i+1}: {score*100:.0f}%'
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # 배경 박스
        cv2.rectangle(image, (x1, y1-label_h-10), (x1+label_w+10, y1), (0, 255, 0), -1)
        
        # 텍스트
        cv2.putText(
            image,
            label,
            (x1+5, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
    
    # 사람 바운딩 박스 (얼굴이 없을 때)
    if person is not None and len(faces) == 0:
        xmin, ymin, xmax, ymax = person
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        
        # 바운딩 박스 (파란색, 점선 효과)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 128, 0), 2)
        
        # 라벨
        label = 'Person'
        cv2.putText(
            image,
            label,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 128, 0),
            2,
            cv2.LINE_AA
        )


def draw_info_panel(image, fps, face_count, person_detected):
    """정보 패널 그리기"""
    h, w = image.shape[:2]
    
    # 반투명 배경 패널
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # 텍스트 정보
    info_lines = [
        f'FPS: {fps:.1f}',
        f'Faces: {face_count}',
        f'Person: {"Yes" if person_detected else "No"}',
        f'Status: {"Tracking" if face_count > 0 else "Searching"}'
    ]
    
    y_offset = 35
    for line in info_lines:
        cv2.putText(
            image,
            line,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y_offset += 25


def main():
    """데모 메인 함수"""
    print("[Demo] 얼굴 감지 데모 시작")
    
    # 모델 로드
    print("[Demo] AI 모델 로드 중...")
    try:
        face_interpreter = tflite.Interpreter(
            config.FACE_MODEL,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        face_interpreter.allocate_tensors()
        print("[Demo] ✅ 얼굴 모델 로드 완료")
        
        person_interpreter = tflite.Interpreter(
            config.PERSON_MODEL,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        person_interpreter.allocate_tensors()
        print("[Demo] ✅ 사람 모델 로드 완료")
        
    except Exception as e:
        print(f"[Demo] ❌ 모델 로드 실패: {e}")
        return
    
    # 카메라 열기
    print("[Demo] 카메라 열기...")
    cap = cv2.VideoCapture(config.CAM_INDEX)
    
    if not cap.isOpened():
        print(f"[Demo] ❌ 카메라 열기 실패 (인덱스: {config.CAM_INDEX})")
        return
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[Demo] ✅ 카메라 열림 ({actual_w}x{actual_h})")
    print("[Demo] ESC 키를 눌러 종료")
    print("[Demo] 's' 키를 눌러 스크린샷 저장")
    
    # FPS 계산용
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[Demo] ⚠️ 프레임 읽기 실패")
            break

        # 원본 프레임 유지 (시각화용)
        display_frame = frame.copy()
        
        # 얼굴 감지
        faces = detection.detect_faces(
            face_interpreter,
            frame,
            score_threshold=config.MIN_DET_CONF,
            debug=False
        )
        
        # 사람 감지 (얼굴이 없을 때만)
        person = None
        if len(faces) == 0:
            person = detection.detect_person(
                person_interpreter,
                frame,
                score_threshold=config.PERSON_SCORE_TH,
                debug=False
            )
        
        # 가이드 라인 그리기
        draw_guide_lines(display_frame)
        
        # 감지 결과 그리기
        draw_detections(display_frame, faces, person)
        
        # FPS 계산
        frame_count += 1
        if frame_count >= 10:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
        
        # 정보 패널 그리기
        draw_info_panel(display_frame, fps, len(faces), person is not None)
        
        # 화면에 표시
        cv2.imshow('Face Detection Demo - Press ESC to quit', display_frame)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xff
        if key == 27:  # ESC
            print("[Demo] 종료 키 입력됨")
            break
        elif key == ord('s'):  # 스크린샷
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"[Demo] 📸 스크린샷 저장: {filename}")
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    print("[Demo] 종료")


if __name__ == '__main__':
    main()