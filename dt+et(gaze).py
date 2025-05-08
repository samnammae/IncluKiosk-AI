import cv2
import numpy as np
import time
import pyautogui
from ultralytics import YOLO
import mediapipe as mp
from gaze_tracking import GazeTracking  # gaze_tracking 추가

# 화면 해상도
screen_w, screen_h = pyautogui.size()

# YOLO 모델
model = YOLO('yolov8n.pt')

# MediaPipe 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# gaze_tracking 객체 초기화
gaze = GazeTracking()

# MediaPipe holistic 초기화
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

    # gaze_tracking 분석
    gaze.refresh(frame)

    # gaze_tracking 기반 상대 마우스 이동
    if gaze.is_right():
        pyautogui.moveRel(20, 0)
    elif gaze.is_left():
        pyautogui.moveRel(-20, 0)
    elif gaze.is_up():
        pyautogui.moveRel(0, -20)
    elif gaze.is_down():
        pyautogui.moveRel(0, 20)

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

                        lx = x1 + int(left_iris.x * w_crop)
                        ly = y1 + int(left_iris.y * h_crop)
                        rx = x1 + int(right_iris.x * w_crop)
                        ry = y1 + int(right_iris.y * h_crop)

                        iris_x_global = (lx + rx) // 2
                        iris_y_global = (ly + ry) // 2
                        screen_x = np.interp(iris_x_global, [0, frame.shape[1]], [0, screen_w])
                        screen_y = np.interp(iris_y_global, [0, frame.shape[0]], [0, screen_h])
                        # pyautogui.moveTo(screen_x, screen_y, duration=0.05)  # 절대 위치 이동은 사용하지 않음

                        # 시각화
                        cv2.circle(frame, (lx, ly), 5, (0, 255, 0), -1)
                        cv2.circle(frame, (rx, ry), 5, (255, 0, 0), -1)
                        cv2.circle(frame, (iris_x_global, iris_y_global), 3, (0, 255, 255), -1)

                    except IndexError:
                        pass

                # 전체 랜드마크 시각화
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

                frame[y1:y2, x1:x2] = face_crop

    annotated_frame = results[0].plot()
    combined = cv2.addWeighted(annotated_frame, 0.5, frame, 0.5, 0)
    cv2.imshow("Eye-tracking Mouse Control with Gaze", combined)

    # 눈동자의 방향을 출력한다
    if gaze.is_right():
        cv2.putText(frame, "Looking right", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)
    elif gaze.is_left():
        cv2.putText(frame, "Looking left", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)
    elif gaze.is_center():
        cv2.putText(frame, "Looking center", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)

    # 왼쪽눈과 오른쪽 눈의 좌표를 카메라 화면에 출력한다
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil: " + str(left_pupil), (10, 440), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (10, 470), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 2)

    # Esc 키를 누르면 프로그램 종료
    if cv2.waitKey(1) == 27:
        break

holistic.close()
cap.release()
cv2.destroyAllWindows()
