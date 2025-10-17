"""
Eye Tracking Worker
메인 실행 파일 - 시선 추적 메인 루프
"""
import cv2
import numpy as np
import time
import asyncio
import pyautogui
import keyboard
from collections import deque

from . import config
from .utils import setup_logging, dbg, print_boot_info
from .detection import init_mediapipe, mediapipe_face_detect, expand_and_clip_bbox, to_global_landmarks, is_fist
from .gaze_tracking import compute_coordinate_box, convert_gaze_to_screen_coordinates
from .calibration import CalibrationState, perform_eye_calibration, perform_screen_calibration, compute_gaze_vectors, perform_full_calibration
from .click_controller import ClickController, draw_click_feedback
from .mouse_control import MouseController, write_screen_position
from .websocket_handler import WebSocketHandler


class EyeTrackingWorker:
    """시선 추적 워커 클래스"""
    
    def __init__(self):
        # Setup
        setup_logging()
        print_boot_info()
        
        # Components
        self.ws_handler = WebSocketHandler()
        self.ws_handler.set_logger(dbg)
        self.mouse_controller = MouseController()
        self.click_controller = ClickController(
            prepare_time=config.CLICK_PREPARE_TIME,
            progress_time=config.CLICK_PROGRESS_TIME,
            click_time=config.CLICK_TIME,
            radius=config.CLICK_RADIUS,
            cooldown=config.CLICK_COOLDOWN
        )
        self.calib_state = CalibrationState()
        
        # MediaPipe
        self.face_detection, self.face_mesh, self.hands = init_mediapipe()
        
        # Camera
        self.cap = self._init_camera()
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # State
        self.frame_count = 0
        self.last_face_bbox = None
        self.last_face_time = 0.0
        self.last_head_center = None
        self.last_R_final = None
        self.last_nose_points_3d = None
        self.last_iris_3d_left = None
        self.last_iris_3d_right = None
        self.last_face_landmarks = None  # ✅ face_landmarks 저장용
        
        # Hand state
        self.hand_last_state = False
        self.last_fist_time = 0.0
        
        # FPS
        self.last_ts = None
        self.fps_ema = None
        
        # Gaze smoothing
        self.combined_gaze_directions = deque(maxlen=config.FILTER_LENGTH)
        
        # Reference matrix
        self.R_ref_nose = [None]
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _init_camera(self):
        """카메라 초기화"""
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
        dbg(f"[Camera] open={cap.isOpened()}")
        
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)
        
        return cap
    
    def _setup_callbacks(self):
        """WebSocket 콜백 설정"""
        def on_eye_calib_on():
            self.click_controller.set_enabled(False)
        
        def on_eye_order_on():
            self.mouse_controller.set_enabled(True)
            self.click_controller.set_enabled(True)
        
        def on_mouse_on():
            self.mouse_controller.set_enabled(True)
            self.click_controller.set_enabled(False)
        
        def on_stop_all():
            self.click_controller.set_enabled(False)
            self.mouse_controller.set_enabled(False)
        
        def on_touch_active():
            self.mouse_controller.set_touch_active(True)
        
        def on_touch_idle():
            self.mouse_controller.set_touch_active(False)
        
        self.ws_handler.on_eye_calib_on = on_eye_calib_on
        self.ws_handler.on_eye_order_on = on_eye_order_on
        self.ws_handler.on_mouse_on = on_mouse_on
        self.ws_handler.on_stop_all = on_stop_all
        self.ws_handler.on_touch_active = on_touch_active
        self.ws_handler.on_touch_idle = on_touch_idle
    
    def start(self):
        """워커 시작"""
        # Start WebSocket receiver
        self.ws_handler.start_receiver()
        
        # Start mouse controller
        self.mouse_controller.start()
        
        # Run main loop
        self.run()
    
    def run(self):
        """메인 루프"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("🟡[eye_worker WS] [ERROR] Failed to read frame!")
                break
            
            # FPS 계산
            now_f = time.perf_counter()
            if self.last_ts is not None:
                inst_fps = 1.0 / max(1e-6, (now_f - self.last_ts))
                self.fps_ema = inst_fps if self.fps_ema is None else (self.fps_ema * 0.85 + inst_fps * 0.15)
            self.last_ts = now_f
            
            self.frame_count += 1
            now = time.time()
            
            # Face Detection
            if self.frame_count % config.DETECT_EVERY == 0:
                dets = mediapipe_face_detect(self.face_detection, frame)
                if dets:
                    dets.sort(key=lambda x: x[4], reverse=True)
                    x0, y0, x1, y1, sc = dets[0]
                    self.last_face_bbox = (x0, y0, x1, y1)
                    self.last_face_time = now
            
            # ROI 결정
            roi = (0, 0, self.w, self.h)
            run_facemesh = False
            frame_rgb = None
            
            roi_valid = (self.last_face_bbox is not None) and (now - self.last_face_time <= config.FACE_TTL)
            
            if roi_valid:
                x0, y0, x1, y1 = expand_and_clip_bbox(self.last_face_bbox, config.ROI_MARGIN, self.w, self.h)
                roi = (x0, y0, x1, y1)
                roi_bgr = frame[y0:y1, x0:x1]
                frame_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                run_facemesh = (self.frame_count % config.FACEMESH_EVERY == 0)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            elif config.ALLOW_FALLBACK_FULLFRAME:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                run_facemesh = (self.frame_count % config.FACEMESH_EVERY == 0)
            
            # Hand detection
            if self.ws_handler.fist_enabled and (self.frame_count % config.HAND_EVERY == 0):
                self._process_hand_detection(frame, roi_valid, now)
            
            # FaceMesh 실행
            results = None
            if run_facemesh and frame_rgb is not None:
                results = self.face_mesh.process(frame_rgb)
            
            # 결과 처리
            if results and results.multi_face_landmarks:
                self._process_face_mesh(results, roi, frame, now)
            
            # Gaze tracking
            if self.last_head_center is not None and self.calib_state.is_calibrated():
                self._process_gaze_tracking(frame, now)
            
            # UI 표시
            self._draw_status(frame)
            
            cv2.imshow("Eye Tracking", frame)
            cv2.waitKey(1)
        
        # Cleanup
        self.cleanup()
    
    def _process_hand_detection(self, frame, roi_valid, now):
        """손 제스처 감지"""
        if roi_valid:
            hx0, hy0, hx1, hy1 = expand_and_clip_bbox(self.last_face_bbox, config.HAND_ROI_SCALE, self.w, self.h)
            hand_roi_bgr = frame[hy0:hy1, hx0:hx1]
            hands_rgb = cv2.cvtColor(hand_roi_bgr, cv2.COLOR_BGR2RGB)
            hand_w, hand_h = hx1 - hx0, hy1 - hy0
        else:
            hands_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_w, hand_h = self.w, self.h
        
        hand_results = self.hands.process(hands_rgb)
        
        curr_fist = False
        if hand_results.multi_hand_landmarks:
            for hlm in hand_results.multi_hand_landmarks:
                if is_fist(hlm, hand_w, hand_h):
                    curr_fist = True
        
        # 주먹 감지
        if curr_fist and (not self.hand_last_state) and (now - self.last_fist_time > config.FIST_COOLDOWN):
            self.last_fist_time = now
            print("🟡[eye_worker WS] [Hand] FIST TRIGGER -> Sending FIST_DETECTED to hub")
            asyncio.run(self.ws_handler.send_fist_detected())
        
        self.hand_last_state = curr_fist
    
    def _process_face_mesh(self, results, roi, frame, now):
        """얼굴 메시 처리"""
        raw_lms = results.multi_face_landmarks[0].landmark
        face_landmarks = to_global_landmarks(raw_lms, roi, self.w, self.h)
        
        head_center, R_final, nose_points_3d = compute_coordinate_box(
            face_landmarks, config.NOSE_INDICES, self.R_ref_nose, self.w, self.h
        )
        
        # Iris 3D 좌표
        l = face_landmarks[config.LEFT_IRIS_IDX]
        r = face_landmarks[config.RIGHT_IRIS_IDX]
        iris_3d_left = np.array([l.x * self.w, l.y * self.h, l.z * self.w], dtype=float)
        iris_3d_right = np.array([r.x * self.w, r.y * self.h, r.z * self.w], dtype=float)
        
        self.last_head_center = head_center.copy()
        self.last_R_final = R_final.copy()
        self.last_nose_points_3d = nose_points_3d.copy()
        self.last_iris_3d_left = iris_3d_left.copy()
        self.last_iris_3d_right = iris_3d_right.copy()
        self.last_face_landmarks = face_landmarks  # ✅ 저장
        
        # EYE_READY 전송
        if not self.ws_handler.sent_ready:
            asyncio.run(self.ws_handler.send_ready())
        
        # 캘리브레이션 자동 실행
        if self.ws_handler.eye_calib_requested and not self.calib_state.is_calibrated():
            print("🟡[eye_worker WS] [Calib] 🎯 얼굴 감지됨 → 자동 통합 캘리브레이션 시작")
            print("🟡[eye_worker WS] [Calib] 💡 화면 중앙을 보세요...")
            
            # ✅ 통합 캘리브레이션 (c + s 한 번에)
            perform_full_calibration(
                self.calib_state, head_center, R_final, nose_points_3d,
                iris_3d_left, iris_3d_right, face_landmarks, self.w, self.h
            )
        
            # ✅ 마우스 컨트롤러의 target도 중앙으로 초기화
            self.mouse_controller.set_target(config.CENTER_X, config.CENTER_Y)
            print(f"🟡[eye_worker WS] [Calibration] 마우스 컨트롤러 target 초기화: ({config.CENTER_X}, {config.CENTER_Y})")
        
            print("🟡[eye_worker WS] [Calibration] ✓ Complete (눈 위치 + 화면 중앙 보정)")
            
            asyncio.run(self.ws_handler.send_calib_complete())
            self.ws_handler.eye_calib_requested = False
    
    def _process_gaze_tracking(self, frame, now):
        """시선 추적 처리"""
        left_gaze_dir, right_gaze_dir, combined_gaze_dir = compute_gaze_vectors(
            self.calib_state,
            self.last_head_center,
            self.last_R_final,
            self.last_nose_points_3d,
            self.last_iris_3d_left,
            self.last_iris_3d_right
        )
        
        if combined_gaze_dir is None:
            return
        
        # Smoothing
        self.combined_gaze_directions.append(combined_gaze_dir)
        avg_combined_direction = np.mean(self.combined_gaze_directions, axis=0)
        avg_combined_direction /= np.linalg.norm(avg_combined_direction)
        
        # 화면 좌표 변환
        screen_x, screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
            avg_combined_direction,
            self.calib_state.offset_yaw,
            self.calib_state.offset_pitch
        )
        
        # 클릭 컨트롤러 업데이트
        current_time = time.time()
        
        if self.mouse_controller.touch_active or \
           (current_time - self.mouse_controller.last_touch_end < config.TOUCH_HOLDOFF):
            current_pos = None
        else:
            current_pos = (screen_x, screen_y)
        
        click_state = self.click_controller.update(current_pos, current_time)
        
        # 클릭 실행
        if click_state['should_click']:
            try:
                pyautogui.click(screen_x, screen_y)
                dbg(f"[Click] ✓ at ({screen_x}, {screen_y}) [count={self.click_controller.click_count}]")
                print(f"🟡[eye_worker WS] [Click] ✓ Executed at ({screen_x}, {screen_y})")
            except Exception as e:
                dbg(f"[Click] ✗ Error: {e}")
        
        # 시각화
        draw_click_feedback(frame, click_state)
        
        # 강제 마우스 ON
        if self.ws_handler.force_mouse_on:
            self.mouse_controller.set_enabled(True)
        
        # 마우스 이동
        if self.mouse_controller.enabled:
            if (not self.mouse_controller.touch_active) and \
               (time.time() - self.mouse_controller.last_touch_end >= config.TOUCH_HOLDOFF):
                self.mouse_controller.set_target(screen_x, screen_y)
        
        write_screen_position(screen_x, screen_y)
        cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})", (10, self.h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _draw_status(self, frame):
        """상태 표시"""
        status_text = []
        if self.fps_ema:
            status_text.append(f"FPS: {self.fps_ema:.1f}")
        status_text.append("Face: OK" if self.last_face_bbox else "Face: NONE")
        status_text.append("Mesh: OK" if self.last_head_center is not None else "Mesh: NONE")
        if self.calib_state.is_calibrated():
            status_text.append("Calib: OK")
        if not self.ws_handler.fist_enabled:
            status_text.append("Fist: OFF")
        if self.click_controller.enabled:
            status_text.append("Click: ON")
        else:
            status_text.append("Click: OFF")
        
        for i, text in enumerate(status_text):
            color = (0, 255, 0) if "OK" in text or "OFF" in text else (0, 0, 255)
            cv2.putText(frame, text, (10, 30 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def cleanup(self):
        """리소스 정리"""
        dbg("[Shutdown] releasing resources")
        self.mouse_controller.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        dbg("=== [BOOT] eye_tracking_worker.py end ===")


def main():
    """메인 함수"""
    worker = EyeTrackingWorker()
    worker.start()


if __name__ == "__main__":
    main()