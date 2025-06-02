import cv2
import numpy as np

# YOLO 가중치 및 구성 파일 경로
config_path = "yolov4-tiny.cfg"
weights_path = "yolov4-tiny.weights"
classes_path = "coco.names"

# 클래스 이름 불러오기
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 네트워크 설정
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 출력 레이어 이름
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 카메라 열기
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("카메라 프레임을 불러올 수 없습니다!")
        break
    frame_count+=1
    if frame_count % 3 != 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # YOLO를 위한 입력 blob 생성
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (128, 128), swapRB=True, crop=False)
    # (416, 416): basic yolo input resolution
    # (320, 320) -> (256, 256) -> (128, 128) improve in speed
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 탐지 결과 처리
    class_ids = []
    confidences = []
    boxes = []

    height, width = frame.shape[:2]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 임계값
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 비최대 억제 (NMS)로 중복 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 결과 시각화
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
