import cv2
import numpy as np
import os

# =========================================================
# 전역 설정
# =========================================================
CLAHE_CLIP     = 2.0
CLAHE_TILE     = (8,8)
GAUSS_KSIZE    = (5,5)
DILATE_ITER    = 2
ELLIPSE_KERNEL = (5,5)
MIRRORED       = True

# Orlosky 변형 파라미터
DARKEST_IGNORE_FRAC = 0.05
DARKEST_STEP        = 4
DARKEST_WIN         = 12
DARKEST_INNER_STEP  = 3
LOCAL_BOX_FRAC      = 0.60

MIN_AREA_FRAC       = 0.002
MAX_ASPECT_RATIO    = 3.0

PREVIEW_SIZE = (160, 110)
SEP_W        = 8

# ===== 명시 경로(고정) =====
FACE_XML = r"C:\Users\dmxth\IncluKiosk\IncluKiosk-AI\haarcascade_frontalface_alt2.xml"
EYE_XML  = r"C:\Users\dmxth\IncluKiosk\IncluKiosk-AI\haarcascade_eye.xml"

# =========================================================
# 캐스케이드 로더 (경로 고정 + 폴백)
# =========================================================
def load_cascade_exact(user_path: str, fallback_names):
    if user_path:
        upath = os.path.abspath(user_path)
        if os.path.exists(upath):
            c = cv2.CascadeClassifier(upath)
            if not c.empty():
                print(f"[INFO] Loaded cascade (USER): {upath}")
                return c
            else:
                print(f"[WARN] Failed to load user cascade: {upath}")
        else:
            print(f"[WARN] User cascade not found: {upath}")

    for name in fallback_names:
        fpath = os.path.join(cv2.data.haarcascades, name)
        c = cv2.CascadeClassifier(fpath)
        if not c.empty():
            print(f"[INFO] Loaded cascade (CV2): {fpath}")
            return c

    print(f"[ERR] Could not load any cascade. Tried user_path={user_path} and {fallback_names}")
    return cv2.CascadeClassifier()  # empty

# ====== 실제 로드(단 1회) ======
faceCascade = load_cascade_exact(
    FACE_XML,
    ["haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_alt.xml", "haarcascade_frontalface_default.xml"]
)
eyeCascade = load_cascade_exact(
    EYE_XML,
    ["haarcascade_eye.xml", "haarcascade_eye_tree_eyeglasses.xml"]
)
if faceCascade.empty() or eyeCascade.empty():
    raise FileNotFoundError("Face/Eye cascade not loaded. Check XML paths or OpenCV install.")

prev_centers = {"L": None, "R": None}

# =========================================================
# 유틸
# =========================================================
def pad_or_resize_gray(img, size=(160, 110)):
    w, h = size
    canvas = np.zeros((h, w), dtype=np.uint8)
    if img is None or img.size == 0:
        return canvas
    ih, iw = img.shape[:2]
    scale = min(w/iw, h/ih)
    nw, nh = max(1, int(iw*scale)), max(1, int(ih*scale))
    resized = cv2.resize(img, (nw, nh))
    x0 = (w - nw)//2; y0 = (h - nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def stack_both_gray_with_centers(left_img, right_img, lc=None, rc=None, size=(160,110)):
    L = pad_or_resize_gray(left_img, size)
    R = pad_or_resize_gray(right_img, size)
    if lc is not None:
        cv2.circle(L, (L.shape[1]//2, L.shape[0]//2), 2, (128,), -1)
    if rc is not None:
        cv2.circle(R, (R.shape[1]//2, R.shape[0]//2), 2, (128,), -1)
    sep = np.zeros((L.shape[0], SEP_W), dtype=np.uint8)
    return np.hstack([L, sep, R])

def hconcat_safe(imgs, gap=6):
    """단일채널 이미지를 가로로 이어 붙이기(높이 맞춰 패딩)."""
    imgs = [img if img is not None else np.zeros_like(imgs[0]) for img in imgs if img is not None]
    if not imgs:
        return None
    h = max(im.shape[0] for im in imgs)
    outs = []
    for im in imgs:
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        pad = np.zeros((h, abs(h - im.shape[0])), dtype=np.uint8)
        if im.shape[0] < h:
            # 위쪽 패딩
            top = (h - im.shape[0]) // 2
            bottom = h - im.shape[0] - top
            im = cv2.copyMakeBorder(im, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        outs.append(im)
    if gap > 0:
        seps = [np.zeros((h, gap), dtype=np.uint8)] * (len(outs)-1)
        row = []
        for i, im in enumerate(outs):
            row.append(im)
            if i < len(outs)-1: row.append(seps[i])
        return np.hstack(row)
    return np.hstack(outs)

def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# =========================================================
# 얼굴/눈 검출
# =========================================================
def detect_eyes(gray, frame, top_ratio, scale_factor, min_neighbors, min_eye_w_frac, min_eye_h_frac):
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(100,100)
    )
    if len(faces) == 0:
        return [], None, None

    x, y, w, h = faces[0]
    face_gray  = gray[y:y+h, x:x+w]

    top_h = int(h * top_ratio)
    face_gray_top = face_gray[0:top_h, :].copy()
    face_gray_top = cv2.equalizeHist(face_gray_top)

    min_eye_w = max(12, int(w * min_eye_w_frac))
    min_eye_h = max( 8, int(h * min_eye_h_frac))

    eyes = eyeCascade.detectMultiScale(
        face_gray_top, scaleFactor=1.1, minNeighbors=4, minSize=(min_eye_w, min_eye_h)
    )

    dets = []
    for (ex, ey, ew, eh) in eyes:
        cy_top_rel = (ey + eh*0.5) / float(max(1, top_h))
        if cy_top_rel > 0.9:    # 너무 아래는 제외
            continue
        if ew < eh:             # 세로가 더 긴 케이스 제외
            continue
        if ew > w*0.65 or eh > h*0.55:
            continue

        cx_f = x + ex + ew//2
        cy_f = y + ey + eh//2
        roi = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
        dets.append(((x+ex, y+ey, ew, eh), (cx_f, cy_f), roi))

    # 디버그: 얼굴/상단 ROI/눈 박스
    dbg = frame.copy()
    cv2.rectangle(dbg, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.rectangle(dbg, (x, y), (x+w, y+top_h), (255,255,0), 2)
    for (bx, by, bw, bh), c, _ in dets:
        cv2.rectangle(dbg, (bx,by), (bx+bw, by+bh), (0,128,255), 2)
        cv2.circle(dbg, c, 3, (0,128,255), -1)
    cv2.putText(dbg, f"faces:{len(faces)} eyes:{len(dets)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

    return dets, dbg, (x, y, w, h, top_h)

def assign_left_right(dets, prev_centers):
    out = {"L": None, "R": None}
    if len(dets) >= 2:
        dets_sorted = sorted(dets, key=lambda d: d[1][0])
        left, right = dets_sorted[0], dets_sorted[1]
        out["L"], out["R"] = left[2], right[2]
        prev_centers["L"], prev_centers["R"] = left[1], right[1]
    elif len(dets) == 1:
        (bbox, center, roi) = dets[0]
        cL, cR = prev_centers["L"], prev_centers["R"]
        if cL is None and cR is None:
            out["L"] = roi; prev_centers["L"] = center
        else:
            def dist(a, b):
                if a is None or b is None: return 1e9
                return np.hypot(a[0]-b[0], a[1]-b[1])
            if dist(center, cL) <= dist(center, cR):
                out["L"] = roi; prev_centers["L"] = center
            else:
                out["R"] = roi; prev_centers["R"] = center
    return out, prev_centers

def fallback_eye_rois_from_face(frame, face_info):
    if face_info is None:
        return []
    x, y, w, h, top_h = face_info
    H, W, _ = frame.shape
    band_y0 = y + int(0.22 * h)
    band_h  = max(12, int(0.28 * h))
    pad_x   = int(0.08 * w)
    eye_w   = int(0.36 * w)
    lx0 = np.clip(x + pad_x, 0, W-1); lx1 = np.clip(lx0 + eye_w, 0, W)
    ly0 = np.clip(band_y0,   0, H-1); ly1 = np.clip(ly0 + band_h, 0, H)
    rx1 = np.clip(x + w - pad_x, 0, W); rx0 = np.clip(rx1 - eye_w, 0, W-1)
    ry0 = ly0; ry1 = ly1
    dets = []
    if lx1 - lx0 > 10 and ly1 - ly0 > 8:
        roiL = frame[ly0:ly1, lx0:lx1]
        dets.append(((lx0, ly0, lx1-lx0, ly1-ly0), (lx0+(lx1-lx0)//2, ly0+(ly1-ly0)//2), roiL))
    if rx1 - rx0 > 10 and ry1 - ry0 > 8:
        roiR = frame[ry0:ry1, rx0:rx1]
        dets.append(((rx0, ry0, rx1-rx0, ry1-ry0), (rx0+(rx1-rx0)//2, ry0+(ry1-ry0)//2), roiR))
    return dets

# =========================================================
# Orlosky 변형 파이프라인 + 디버그 스테이지
# =========================================================
def darkest_point_gray(g, ignore=4, step=4, win=12, inner_step=3):
    H, W = g.shape
    best_sum = 1e18
    best = (W//2, H//2)
    for y in range(ignore, H - ignore - win, step):
        for x in range(ignore, W - ignore - win, step):
            s = 0; cnt = 0
            for yy in range(0, win, inner_step):
                for xx in range(0, win, inner_step):
                    s += int(g[y+yy, x+xx]); cnt += 1
            if cnt and s < best_sum:
                best_sum = s
                best = (x + win//2, y + win//2)
    return best

def mask_square_gray(g, center, size):
    h, w = g.shape
    cx, cy = center
    s = max(8, int(size))
    x0 = max(0, cx - s//2); x1 = min(w, cx + s//2)
    y0 = max(0, cy - s//2); y1 = min(h, cy + s//2)
    out = np.zeros_like(g)
    out[y0:y1, x0:x1] = g[y0:y1, x0:x1]
    return out

def thr_and_score(g, thr, min_area=120, max_ratio=3.0, dilate_iter=2, want_debug=False):
    _, bin_inv = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ELLIPSE_KERNEL)
    dil = cv2.dilate(bin_inv, kernel, iterations=dilate_iter)

    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_score = 0.0; best_cnt = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or len(c) < 5:
            continue
        x,y,w,h = cv2.boundingRect(c)
        ratio = (max(w,h) / max(1,min(w,h)))
        if ratio > max_ratio:
            continue

        ellipse = cv2.fitEllipse(c)

        mask = np.zeros_like(dil)
        cv2.ellipse(mask, ellipse, 255, -1)
        ellipse_area = np.count_nonzero(mask)
        if ellipse_area == 0: 
            continue
        filled = np.count_nonzero((dil == 255) & (mask == 255))
        filled_ratio = filled / float(ellipse_area)

        border = np.zeros_like(dil)
        cv2.drawContours(border, [c], -1, 255, 1)
        border_tot = np.count_nonzero(border)
        thick = np.zeros_like(dil)
        cv2.ellipse(thick, ellipse, 255, 4)
        overlap = np.count_nonzero((border == 255) & (thick == 255))
        overlap_ratio = overlap / float(border_tot) if border_tot else 0.0

        score = filled_ratio * overlap_ratio * (area**0.5)
        if score > best_score:
            best_score = score; best = ellipse; best_cnt = c

    if want_debug:
        # 컨투어/타원 오버레이(BGR)
        vis = to_bgr(dil)
        if best_cnt is not None:
            cv2.drawContours(vis, [best_cnt], -1, (0,200,255), 1)
        if best is not None:
            cv2.ellipse(vis, best, (0,255,0), 2)
        return bin_inv, best, best_score, best_cnt, dil, vis

    return bin_inv, best, best_score, best_cnt

def process_eye_roi_debug(eye_bgr):
    """
    디버그용: 각 스테이지 결과를 dict로 반환
    returns:
      best_bin, center, best_score, stages(dict)
    """
    stages = {
        "gray": None, "clahe_blur": None, "seed_vis": None,
        "masked": None, "cand_bins": None, "best_dilated": None, "ellipse_vis": None
    }

    if eye_bgr is None or eye_bgr.size == 0:
        return None, None, 0.0, stages

    # 1) GRAY
    g = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    stages["gray"] = g.copy()

    # 2) CLAHE + BLUR
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    g2 = clahe.apply(g)
    g2 = cv2.GaussianBlur(g2, GAUSS_KSIZE, 0)
    stages["clahe_blur"] = g2.copy()

    h, w = g2.shape
    # 3) SEED & LOCAL MASK
    ignore = int(min(h,w) * DARKEST_IGNORE_FRAC)
    seed = darkest_point_gray(g2, ignore=ignore, step=DARKEST_STEP,
                              win=DARKEST_WIN, inner_step=DARKEST_INNER_STEP)
    seed_vis = to_bgr(g2)
    cv2.circle(seed_vis, (int(seed[0]), int(seed[1])), 3, (0,255,0), -1)
    stages["seed_vis"] = seed_vis

    box = LOCAL_BOX_FRAC * min(h, w)
    g_masked = mask_square_gray(g2, seed, box)
    stages["masked"] = g_masked.copy()

    # 4) MULTI-THRESH
    base = int(g2[seed[1], seed[0]])
    cand_T = [base + 5, base + 15, base + 25]
    cand_imgs = []
    best = (None, None, 0.0, None, None, None)  # bin, ellipse, score, cnt, dil, vis
    for T in cand_T:
        bin_inv, ellipse, score, cnt, dil, vis = thr_and_score(
            g_masked, T,
            min_area=int(MIN_AREA_FRAC * w * h),
            max_ratio=MAX_ASPECT_RATIO,
            dilate_iter=DILATE_ITER,
            want_debug=True
        )
        # 후보 이진 이미지 기록(팁: 각 이미지에 T값 워터마크)
        bin_disp = bin_inv.copy()
        cv2.putText(bin_disp, f"T={T}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
        cand_imgs.append(bin_disp)

        if score > best[2]:
            best = (bin_inv, ellipse, score, cnt, dil, vis)

    stages["cand_bins"] = hconcat_safe(cand_imgs, gap=6)

    best_bin, best_ellipse, best_score, _, best_dil, best_vis = best
    stages["best_dilated"] = best_dil.copy() if best_dil is not None else None
    stages["ellipse_vis"]  = best_vis.copy()  if best_vis is not None else None

    center = None
    if best_ellipse is not None:
        cx, cy = map(int, best_ellipse[0])
        center = (cx, cy)

    return best_bin, center, best_score, stages

# =========================================================
# 트랙바(슬라이더)
# =========================================================
def nothing(v): pass

def setup_trackbars():
    cv2.namedWindow("Frame (debug)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame (debug)", 960, 540)
    cv2.createTrackbar("scaleFactor x100", "Frame (debug)", 105, 150, nothing)   # 1.05 ~ 1.50
    cv2.createTrackbar("minNeighbors", "Frame (debug)", 5, 12, nothing)
    cv2.createTrackbar("TOP_RATIO %", "Frame (debug)", 60, 90, nothing)         # 60% 기본
    cv2.createTrackbar("minEyeW %faceW", "Frame (debug)", 12, 30, nothing)      # 12% 기본
    cv2.createTrackbar("minEyeH %faceH", "Frame (debug)", 8, 25, nothing)       # 8% 기본

def read_trackbars():
    sf = max(101, cv2.getTrackbarPos("scaleFactor x100", "Frame (debug)")) / 100.0
    mn = max(1, cv2.getTrackbarPos("minNeighbors", "Frame (debug)"))
    tr = max(40, cv2.getTrackbarPos("TOP_RATIO %", "Frame (debug)")) / 100.0
    mew = max(6, cv2.getTrackbarPos("minEyeW %faceW", "Frame (debug)")) / 100.0
    meh = max(4, cv2.getTrackbarPos("minEyeH %faceH", "Frame (debug)")) / 100.0
    return sf, mn, tr, mew, meh

# === NEW: 대시보드(단계별 패널) 만들기 유틸 ===
def ensure_bgr(img):
    if img is None:
        return None
    return img if (img.ndim == 3 and img.shape[2] == 3) else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def labelize(img, title, size=(320, 240)):
    """
    단일 이미지를 지정 크기로 리사이즈하고 상단에 반투명 라벨 바 + 타이틀을 올린다.
    img: gray 또는 BGR
    """
    if img is None or (hasattr(img, "size") and img.size == 0):
        img = np.zeros((size[1], size[0]), dtype=np.uint8)
    img = ensure_bgr(img)
    ih, iw = img.shape[:2]
    scale = min(size[0]/iw, size[1]/ih)
    nw, nh = max(1, int(iw*scale)), max(1, int(ih*scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    x0 = (size[0]-nw)//2; y0 = (size[1]-nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized

    # 라벨 바(반투명)
    bar_h = 24
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0,0), (size[0], bar_h), (40,40,40), -1)
    canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

    # 텍스트
    cv2.putText(canvas, title, (8,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
    return canvas

def make_pipeline_panel(l_stages, both_bin, box_size=(320,240)):
    """
    2x4 타일 패널 생성.
    l_stages: process_eye_roi_debug가 반환한 dict
    both_bin: 좌/우 이진 미리보기(gray)
    """
    tiles_row1 = [
        labelize(l_stages.get("gray"),        "L-GRAY",         box_size),
        labelize(l_stages.get("clahe_blur"),  "L-CLAHE+BLUR",   box_size),
        labelize(l_stages.get("seed_vis"),    "L-SEEDED",       box_size),
        labelize(l_stages.get("masked"),      "L-MASKED",       box_size),
    ]
    tiles_row2 = [
        labelize(l_stages.get("cand_bins"),   "L-CAND THR",     box_size),
        labelize(l_stages.get("best_dilated"),"L-BEST DILATED", box_size),
        labelize(l_stages.get("ellipse_vis"), "L-ELLIPSE",      box_size),
        labelize(both_bin,                    "Both Eyes (Binary L|R)", box_size),
    ]
    row1 = np.hstack(tiles_row1)
    row2 = np.hstack(tiles_row2)
    panel = np.vstack([row1, row2])
    return panel


# =========================================================
# 메인
# =========================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Camera open failed")
        return

    setup_trackbars()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERR] Camera read failed")
            break

        if MIRRORED:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 트랙바에서 파라미터 읽기
        scale_factor, min_neighbors, top_ratio, min_eye_w_frac, min_eye_h_frac = read_trackbars()

        dets, dbg, face_info = detect_eyes(
            gray, frame, 
            top_ratio=top_ratio,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_eye_w_frac=min_eye_w_frac,
            min_eye_h_frac=min_eye_h_frac
        )

        # 눈이 하나도 안 잡히면 얼굴 기반 폴백으로 좌/우 ROI 생성
        if len(dets) == 0:
            dets = fallback_eye_rois_from_face(frame, face_info)
            if dbg is None:
                dbg = frame.copy()
            cv2.putText(dbg, "Fallback eye ROI", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)

        lr, _ = assign_left_right(dets, prev_centers)

        # ===== 왼쪽 눈 파이프라인(스테이지별 디스플레이) =====
        left_bin,  left_center,  lscore, lstages  = process_eye_roi_debug(lr["L"])
        right_bin, right_center, rscore, _        = process_eye_roi_debug(lr["R"])

        both_bin = stack_both_gray_with_centers(left_bin, right_bin, left_center, right_center, PREVIEW_SIZE)
        cv2.imshow("Both Eyes (Binary L | R)", both_bin)

        # --- 파이프라인 스테이지 창들 (왼쪽 기준) ---
        panel = make_pipeline_panel(lstages, both_bin, box_size=(320, 240))
        cv2.imshow("Pipeline Panel", panel)
        
        # 디버그 오버레이
        if dbg is None:
            dbg = frame.copy()
            cv2.putText(dbg, "No face detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
        else:
            info = f"sf:{scale_factor:.2f} neigh:{min_neighbors} top:{int(top_ratio*100)}% minEyeW:{int(min_eye_w_frac*100)}% minEyeH:{int(min_eye_h_frac*100)}%"
            cv2.putText(dbg, info, (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Frame (debug)", dbg)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
