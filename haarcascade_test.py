import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade  = cv2.CascadeClassifier('haarcascade_eye.xml')

# 직전 프레임의 눈 중심(절대 좌표)을 저장하여 단일 검출 시에도 방향 유지
prev_centers = {"L": None, "R": None}

def detect_eyes(gray, frame):
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5,
        minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE
    )
    # 얼굴이 여러 개면 첫 번째만 사용(필요시 가장 큰 얼굴 선택하도록 변경 가능)
    if len(faces) == 0:
        return []
    x, y, w, h = faces[0]
    face_gray  = gray[y:y+h, x:x+w]
    face_color = frame[y:y+h, x:x+w]

    eyes = eyeCascade.detectMultiScale(face_gray, 1.1, 3)
    dets = []
    for (ex, ey, ew, eh) in eyes:
        # 절대 좌표 중심 (화면 기준)
        cx = x + ex + ew//2
        cy = y + ey + eh//2
        roi = face_color[ey:ey+eh, ex:ex+ew]
        dets.append(((x+ex, y+ey, ew, eh), (cx, cy), roi))
    return dets

def assign_left_right(dets, prev_centers):
    """
    dets: [ ((abs_x, abs_y, w, h), (cx, cy), roi), ... ]
    return: dict {"L": roi_or_None, "R": roi_or_None}  + 업데이트된 prev_centers
    """
    out = {"L": None, "R": None}

    if len(dets) >= 2:
        # x 중심 기준으로 정렬 → 좌: 작은 x, 우: 큰 x
        dets_sorted = sorted(dets, key=lambda d: d[1][0])
        left  = dets_sorted[0]
        right = dets_sorted[1]
        out["L"] = left[2]
        out["R"] = right[2]
        prev_centers["L"] = left[1]
        prev_centers["R"] = right[1]

    elif len(dets) == 1:
        # 하나만 잡혔을 때는 이전 중심과 더 가까운 쪽에 배정
        (bbox, center, roi) = dets[0]
        cL, cR = prev_centers["L"], prev_centers["R"]

        # 이전 정보가 전혀 없으면 화면 중앙 기준으로 좌/우 추정
        if cL is None and cR is None:
            # center x가 화면 중앙보다 작으면 L, 크면 R
            # 화면 폭을 알기 위해 roi가 아니라 bbox 기반으로는 어렵다 → 그냥 x 좌표로 휴리스틱
            # 더 안정적으로 하려면 frame width를 main loop에서 넘겨 사용
            # 여기서는 일단 이전 정보가 없으면 '가까운 곳' 규칙 대신 x 비교는 생략, L에 넣고 시작
            out["L"] = roi
            out["R"] = None
            prev_centers["L"] = center
        else:
            # 두 이전 중심 중 가까운 곳 선택
            def dist(a, b):
                return np.hypot(a[0]-b[0], a[1]-b[1])

            dL = dist(center, cL) if cL is not None else float("inf")
            dR = dist(center, cR) if cR is not None else float("inf")

            if dL <= dR:
                out["L"] = roi
                out["R"] = None
                prev_centers["L"] = center
            else:
                out["L"] = None
                out["R"] = roi
                prev_centers["R"] = center

    # dets == 0: 아무것도 없음 → prev 유지, 출력은 None
    return out, prev_centers

def pad_or_resize(eye_img, size=(140, 90)):
    """눈 이미지를 일정 크기(size)로 리사이즈. 없으면 검은 캔버스."""
    w, h = size[0], size[1]
    if eye_img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    # 비율 유지 리사이즈 후 중앙 배치
    ih, iw = eye_img.shape[:2]
    scale = min(w/iw, h/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    resized = cv2.resize(eye_img, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = (w - nw)//2
    y0 = (h - nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def stack_both(left_eye, right_eye):
    L = pad_or_resize(left_eye)
    R = pad_or_resize(right_eye)
    # 가운데 구분선 추가(선택)
    sep = np.zeros((L.shape[0], 6, 3), dtype=np.uint8)
    return np.hstack([L, sep, R])

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detect_eyes(gray, frame)
    lr, prev_centers = assign_left_right(dets, prev_centers)

    both = stack_both(lr["L"], lr["R"])
    cv2.imshow("Both Eyes (L | R)", both)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
