"""얼굴 감지 데모 - 카메라로 실시간 얼굴 감지 테스트"""
import cv2
import tflite_runtime.interpreter as tflite
from . import config
from . import detection


def draw_detections(image, objs):
    """감지된 객체를 이미지에 그리기"""
    for obj in objs:
        bbox = obj.bbox
        
        # 바운딩 박스
        cv2.rectangle(
            image, 
            (bbox.xmin, bbox.ymin), 
            (bbox.xmax, bbox.ymax), 
            (0, 255, 0), 
            2
        )

        # 중심점
        center_x = bbox.xmin + ((bbox.xmax - bbox.xmin) // 2)
        center_y = bbox.ymin + ((bbox.ymax - bbox.ymin) // 2)
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 신뢰도 표시
        cv2.putText(
            image, 
            text=f'{obj.score*100:.0f}%',
            org=(bbox.xmin, bbox.ymin-5), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0), 
            thickness=1,
            lineType=cv2.LINE_AA
        )


def main():
    """데모 메인 함수"""
    print("[Demo] 얼굴 감지 데모 시작")
    
    # 모델 로드
    print("[Demo] 모델 로드 중...")
    try:
        interpreter = tflite.Interpreter(
            config.FACE_MODEL,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        interpreter.allocate_tensors()
        print("[Demo] ✅ 모델 로드 완료")
    except Exception as e:
        print(f"[Demo] ❌ 모델 로드 실패: {e}")
        return
    
    # 카메라 열기
    print("[Demo] 카메라 열기...")
    cap = cv2.VideoCapture(config.CAM_INDEX)
    
    if not cap.isOpened():
        print(f"[Demo] ❌ 카메라 열기 실패 (인덱스: {config.CAM_INDEX})")
        return
    
    print("[Demo] ✅ 카메라 열림")
    print("[Demo] ESC 키를 눌러 종료")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[Demo] ⚠️ 프레임 읽기 실패")
            break

        # 이미지 리사이즈 (처리 속도 향상)
        display_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
        
        # BGR -> RGB 변환
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # 모델 입력
        tensor = detection.input_tensor(interpreter)
        tensor.fill(0)
        tensor[:, :] = frame_rgb.copy()
        del tensor
        
        # 추론
        interpreter.invoke()
        
        # 결과 가져오기
        objs = detection.get_output(interpreter, 0.5, (1.0, 1.0))
        
        # 감지 결과 그리기
        if len(objs) > 0:
            draw_detections(display_frame, objs)

        # RGB -> BGR 변환 (OpenCV 디스플레이용)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        
        # 화면에 표시
        cv2.imshow('Face Detection Demo', display_frame)

        # ESC 키로 종료
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            print("[Demo] 종료 키 입력됨")
            break
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    print("[Demo] 종료")


if __name__ == '__main__':
    main()