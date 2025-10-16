"""
WebSocket Handler
WebSocket 통신 관리
"""
import asyncio
import websockets
import json
import threading
from . import config


class WebSocketHandler:
    """WebSocket 통신 핸들러"""
    
    def __init__(self):
        # Flags
        self.toggle_mouse_requested = False
        self.eye_calib_requested = False
        self.force_mouse_on = False
        self.fist_enabled = True
        
        # Ready flag
        self.sent_ready = False
        
        # 콜백 함수들
        self.on_eye_calib_on = None
        self.on_eye_order_on = None
        self.on_mouse_on = None
        self.on_stop_all = None
        self.on_touch_active = None
        self.on_touch_idle = None
        
        # 디버그 로거
        self.logger = None
    
    def set_logger(self, logger):
        """로거 설정"""
        self.logger = logger
    
    def _log(self, msg):
        """로그 출력"""
        if self.logger:
            self.logger(msg)
        else:
            print(msg)
    
    async def send_message(self, payload: dict):
        """메시지 전송"""
        try:
            async with websockets.connect(config.WS_URL) as ws:
                await ws.send(json.dumps(payload, ensure_ascii=False))
                await asyncio.sleep(0.05)
                self._log(f"[WS] sent {payload}")
        except Exception as e:
            self._log(f"[WS] send failed: {e}")
    
    async def send_ready(self):
        """EYE_READY 메시지 전송"""
        if not self.sent_ready:
            await self.send_message({"type": "EYE_READY"})
            self.sent_ready = True
    
    async def send_fist_detected(self):
        """FIST_DETECTED 메시지 전송"""
        await self.send_message({"type": "FIST_DETECTED"})
    
    async def send_calib_complete(self):
        """EYE_CALIB_COMPLETE 메시지 전송"""
        await self.send_message({"type": "EYE_CALIB_COMPLETE"})
    
    async def _receiver(self):
        """메시지 수신 루프"""
        while True:
            try:
                self._log("[WS] trying to connect...")
                async with websockets.connect(config.WS_URL) as ws:
                    self._log("[WS] connected")
                    
                    while True:
                        raw = await ws.recv()
                        self._log(f"[WS] recv raw={raw}")
                        
                        try:
                            data = json.loads(raw)
                            msg_type = data.get("type")
                        except Exception:
                            continue
                        
                        self._log(f"[WS] parsed type={msg_type}")
                        self._handle_message(msg_type, data)
                        
            except Exception as e:
                self._log(f"[WS] connect failed/disconnected: {e} (retry in 2s)")
                await asyncio.sleep(2)
    
    def _handle_message(self, msg_type, data):
        """메시지 처리"""
        if msg_type == "EYE_CALIB_ON":
            # ✅ 자동으로 full calibration 수행 (c + s 통합)
            self.eye_calib_requested = True
            self.fist_enabled = False
            self.force_mouse_on = False
            self._log("[WS] → eye_calib_requested=True, ALL features OFF")
            print("[WS] EYE_CALIB_ON → 통합 캘리브레이션 모드 (화면 중앙을 보세요)")
            
            if self.on_eye_calib_on:
                self.on_eye_calib_on()
        
        elif msg_type == "EYE_ORDER_ON":
            # 눈 주문: 마우스 + 클릭 ON, 주먹 OFF
            self.fist_enabled = False
            self.force_mouse_on = True
            self._log("[WS] → fist=False, mouse=True, click=True")
            print("[WS] EYE_ORDER_ON → 주먹OFF, 마우스ON, 클릭ON")
            
            if self.on_eye_order_on:
                self.on_eye_order_on()
        
        elif msg_type == "MOUSE_ON":
            self.fist_enabled = True
            self.force_mouse_on = True
            self._log("[WS] → fist=True, mouse=True, click=False")
            print("[WS] MOUSE_ON → 주먹ON, 마우스ON, 클릭OFF")
            
            if self.on_mouse_on:
                self.on_mouse_on()
        
        elif msg_type == "STOP_ALL":
            self.fist_enabled = False
            self.force_mouse_on = False
            self._log("[WS] → STOP_ALL: all features disabled")
            print("[WS] STOP_ALL → 모든 기능 비활성화")
            
            if self.on_stop_all:
                self.on_stop_all()
        
        elif msg_type == "TOUCH_ACTIVE":
            self._log("[WS] TOUCH_ACTIVE")
            if self.on_touch_active:
                self.on_touch_active()
        
        elif msg_type == "TOUCH_IDLE":
            self._log("[WS] TOUCH_IDLE")
            if self.on_touch_idle:
                self.on_touch_idle()
    
    def start_receiver(self):
        """수신 스레드 시작"""
        def runner():
            asyncio.run(self._receiver())
        
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        self._log("[WS] receiver thread starting...")