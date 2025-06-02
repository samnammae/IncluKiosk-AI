import cv2
import numpy as np
import time
import pyautogui
from ultralytics import YOLO
import mediapipe as mp

# 화면 해상도
screen_w, screen_h = pyautogui.size()

# YOLO 모델
model = YOLO('yolov8n.pt')

# MediaPipe 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO로 사람 감지
    results = model.track(frame, persist=True, verbose=False, iou=0.3, conf=0.5)

    if results[0].boxes.id is not None:
        object_ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for obj_id, cls, box in zip(object_ids, classes, boxes):
            if int(cls) == 0:  # 사람 클래스
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[y1:y2, x1:x2]
                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                results_mediapipe = holistic.process(rgb_crop)

                if results_mediapipe.face_landmarks:
                    h_crop, w_crop, _ = face_crop.shape
                    try:
                        left_iris = results_mediapipe.face_landmarks.landmark[468]
                        right_iris = results_mediapipe.face_landmarks.landmark[473]

                        # 원본 프레임 기준 좌표로 변환
                        lx = x1 + int(left_iris.x * w_crop)
                        ly = y1 + int(left_iris.y * h_crop)
                        rx = x1 + int(right_iris.x * w_crop)
                        ry = y1 + int(right_iris.y * h_crop)

                        # 화면 좌표로 매핑
                        iris_x_global = (lx + rx) // 2
                        iris_y_global = (ly + ry) // 2
                        screen_x = np.interp(iris_x_global, [0, frame.shape[1]], [0, screen_w])
                        screen_y = np.interp(iris_y_global, [0, frame.shape[0]], [0, screen_h])
                        pyautogui.moveTo(screen_x, screen_y, duration=0.05)

                        # 디버깅용 시각화 (양쪽 눈에 점 찍기)
                        cv2.circle(frame, (lx, ly), 5, (0, 255, 0), -1)  # 왼쪽 초록
                        cv2.circle(frame, (rx, ry), 5, (255, 0, 0), -1)  # 오른쪽 파랑
                        cv2.circle(frame, (iris_x_global, iris_y_global), 3, (0, 255, 255), -1)  # 중간점 노랑

                    except IndexError:
                        pass

                # 전체 랜드마크 시각화 (손, 얼굴, 포즈)
                mp_drawing.draw_landmarks(face_crop, results_mediapipe.face_landmarks,
                                            mp_holistic.FACEMESH_TESSELATION,
                                            landmark_drawing_spec=None,
                                            connection_drawing_spec=mp_drawing_styles
                                            .get_default_face_mesh_tesselation_style())

                mp_drawing.draw_landmarks(face_crop, results_mediapipe.left_hand_landmarks,
                                            mp_holistic.HAND_CONNECTIONS)

                mp_drawing.draw_landmarks(face_crop, results_mediapipe.right_hand_landmarks,
                                            mp_holistic.HAND_CONNECTIONS)

                mp_drawing.draw_landmarks(face_crop, results_mediapipe.pose_landmarks,
                                            mp_holistic.POSE_CONNECTIONS,
                                            landmark_drawing_spec=mp_drawing_styles
                                            .get_default_pose_landmarks_style())

                # 업데이트된 얼굴 부분을 원본 프레임에 적용
                frame[y1:y2, x1:x2] = face_crop

    # YOLO 박스 시각화 후 화면에 출력
    annotated_frame = results[0].plot()
    combined = cv2.addWeighted(annotated_frame, 0.5, frame, 0.5, 0)
    cv2.imshow("Eye-tracking Mouse Control with Full Body Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
holistic.close()
cap.release()
cv2.destroyAllWindows()