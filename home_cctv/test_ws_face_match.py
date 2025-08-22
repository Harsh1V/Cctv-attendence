import asyncio
import base64
import json
import cv2
import websockets
from datetime import datetime

WS_URL = "ws://127.0.0.1:8000/ws/face/"
CAMERA_SOURCE = 0

DETECT_W, DETECT_H = 320, 240
WIN_W, WIN_H = 960, 720
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICK = 2
COLOR_TEXT = (255, 255, 255)
COLOR_MATCH = (0, 255, 0)
COLOR_NOMATCH = (0, 0, 255)
COLOR_INFO = (0, 200, 255)
TEXT_NOFACE = "No Face Detected"
TEXT_UNKNOWN = "Unknown"
TIMEOUT_SEC = 8.0

def frame_to_base64(frame_bgr):
    ok, buffer = cv2.imencode('.jpg', frame_bgr)
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode('utf-8')

async def run_client():
    async with websockets.connect(WS_URL, max_size=8 * 1024 * 1024) as ws:
        print("✅ Connected:", WS_URL)

        cv2.namedWindow("WS Live (Client)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("WS Live (Client)", WIN_W, WIN_H)

        try:
            hello = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print("Server hello:", hello)
        except asyncio.TimeoutError:
            pass

        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break

                frame_count += 1
                small = cv2.resize(frame, (DETECT_W, DETECT_H))

                b64_img = frame_to_base64(small)
                if b64_img is None:
                    print("❌ JPEG encode failed")
                    break

                payload = {
                    "camera_id": "CCTV-01",
                    "timestamp": datetime.utcnow().isoformat(),
                    "image": b64_img,
                }

                await ws.send(json.dumps(payload))

                display = frame.copy()
                h, w = display.shape[:2]
                sx, sy = w / DETECT_W, h / DETECT_H
                overlay_text = None

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_SEC)
                    data = json.loads(raw)
                    print(f"Server response: {data}")  # Debugging

                    faces_list = data.get("faces")

                    if not faces_list or len(faces_list) == 0:
                        overlay_text = TEXT_NOFACE
                    else:
                        for face_result in faces_list:
                            status = face_result.get("status")
                            face_box = face_result.get("box")
                            
                            if face_box and len(face_box) == 4:
                                x1, y1, x2, y2 = map(int, face_box)
                                X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                                color = COLOR_MATCH if status == "match_found" else COLOR_NOMATCH
                                cv2.rectangle(display, (X1, Y1), (X2, Y2), color, 5)  # Thicker box
                                
                                if status == "match_found":
                                    name = face_result.get("matched_name", "Match")
                                    score = face_result.get("match_score", 0.0)
                                    label = f"{name} ({score:.2f})"
                                else:
                                    label = TEXT_UNKNOWN

                                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
                                cv2.putText(display, label, (X1, Y1 - 10),
                                            FONT, FONT_SCALE, color, FONT_THICK)
                            elif status == "error":
                                msg = face_result.get("note") or "Error"
                                overlay_text = f"Err: {msg[:32]}"
                            elif status == "no_face":
                                overlay_text = TEXT_NOFACE

                    if overlay_text:
                        cv2.putText(display, overlay_text, (20, 40), FONT, FONT_SCALE, COLOR_NOMATCH, FONT_THICK)

                except asyncio.TimeoutError:
                    overlay_text = "No response (timeout)"
                    cv2.putText(display, overlay_text, (20, 40), FONT, FONT_SCALE, COLOR_NOMATCH, FONT_THICK)

                cv2.imshow("WS Live (Client)", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("📴 Camera released / windows closed")

if __name__ == "__main__":
    import os
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_client())

