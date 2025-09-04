import cv2
import numpy as np

# ── 1) 사용할 Cascade 교체 ─────────────────────────────────────
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyeCascade  = cv2.CascadeClassifier('haarcascade_eye.xml')

# 직전 프레임의 눈 중심(절대 좌표)을 저장하여 단일 검출 시에도 방향 유지
prev_centers = {"L": None, "R": None}

# 얼굴 상단 몇 %만 눈 검출에 사용할지
TOP_RATIO = 0.60   # 얼굴 ROI의 위쪽 60%만 사용

def detect_eyes(gray, frame):
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5,
        minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return []

    # 필요시 가장 큰 얼굴로 교체하고 싶다면 아래 두 줄로 faces를 정렬 후 faces[0] 사용
    # faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]

    face_gray  = gray[y:y+h, x:x+w]
    face_color = frame[y:y+h, x:x+w]

    # ── 2) 얼굴 상단 60%만 대상으로 눈 탐지 ─────────────────────
    top_h = int(h * TOP_RATIO)
    face_gray_top = face_gray[0:top_h, :].copy()

    # (선택) 조명 안정화를 위해 히스토그램 평활화
    face_gray_top = cv2.equalizeHist(face_gray_top)

    # 눈 최소 크기를 얼굴 폭/높이에 비례해 설정 (환경에 맞게 조절)
    min_eye_w = max(16, int(w * 0.12))
    min_eye_h = max(10, int(h * 0.08))

    eyes = eyeCascade.detectMultiScale(
        face_gray_top,
        scaleFactor=1.1,
        minNeighbors=5,                 # 오검출 많으면 6~7로 올려보세요
        minSize=(min_eye_w, min_eye_h)  # 너무 작은 것 배제
    )

    dets = []
    for (ex, ey, ew, eh) in eyes:
        # ── 3) 추가 필터: 눈의 세로 위치/비율/크기 간단 검증 ───────
        # (a) 상단 60% 내에서도 더 위쪽에 있는 것만(예: 70% 이하 중심)
        cy_rel = (ey + eh * 0.5) / h  # 얼굴 전체 높이 기준 상대값(0~1)
        # 여기서 h로 나누면 비율이 작게 나오므로 face_gray_top 기준으로 다시 계산:
        cy_top_rel = (ey + eh * 0.5) / float(top_h)  # 상단 ROI 높이 기준
        if cy_top_rel > 0.85:
            # 상단영역의 거의 하단부(입가 근처)에 있으면 스킵
            continue

        # (b) 눈 가로가 세로보다 커야 함(대략적인 규칙)
        if ew < eh:
            continue

        # (c) 얼굴 대비 너무 큰/작은 눈 배제 (환경 맞춰 조정)
        if ew > w * 0.6 or eh > h * 0.5:
            continue

        # ── 절대 좌표 중심/ROI 산출 ────────────────────────────
        # 주의: face_gray_top은 face_gray의 y=0부터 시작이므로,
        # 절대 좌표로 환산 시 ey는 그대로 사용(얼굴 ROI 상단 기준)
        cx = x + ex + ew//2
        cy = y + ey + eh//2
        roi = face_color[ey:ey+eh, ex:ex+ew]
        dets.append(((x+ex, y+ey, ew, eh), (cx, cy), roi))

    return dets

def assign_left_right(dets, prev_centers):
    out = {"L": None, "R": None}

    if len(dets) >= 2:
        # x 중심 기준 정렬 → 좌/우 고정
        dets_sorted = sorted(dets, key=lambda d: d[1][0])
        left, right = dets_sorted[0], dets_sorted[1]
        out["L"], out["R"] = left[2], right[2]
        prev_centers["L"], prev_centers["R"] = left[1], right[1]

    elif len(dets) == 1:
        (bbox, center, roi) = dets[0]
        cL, cR = prev_centers["L"], prev_centers["R"]

        if cL is None and cR is None:
            out["L"] = roi
            prev_centers["L"] = center
        else:
            def dist(a, b): return np.hypot(a[0]-b[0], a[1]-b[1])
            dL = dist(center, cL) if cL is not None else float("inf")
            dR = dist(center, cR) if cR is not None else float("inf")
            if dL <= dR:
                out["L"] = roi; prev_centers["L"] = center
            else:
                out["R"] = roi; prev_centers["R"] = center

    return out, prev_centers

def pad_or_resize(eye_img, size=(140, 90)):
    w, h = size
    if eye_img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    ih, iw = eye_img.shape[:2]
    scale = min(w/iw, h/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    resized = cv2.resize(eye_img, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = (w - nw)//2; y0 = (h - nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def stack_both(left_eye, right_eye):
    L = pad_or_resize(left_eye)
    R = pad_or_resize(right_eye)
    sep = np.zeros((L.shape[0], 6, 3), dtype=np.uint8)  # 구분선
    return np.hstack([L, sep, R])

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break

    # 거울처럼 보이도록 좌우반전(원하면 유지)
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
