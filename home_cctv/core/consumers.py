import json
import base64
import io
import pickle

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cosine

from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

from facenet_pytorch import MTCNN, InceptionResnetV1

# =================== Config / Models ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETECT_W, DETECT_H = 320, 240
CONFIDENCE_THRESHOLD = 0.90
MATCH_THRESHOLD = 0.80

mtcnn = MTCNN(keep_all=True, device=device)
facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# =================== DB Helpers (async safe) ===================
@database_sync_to_async
def get_all_approved_faces():
    from core.models import ApprovedFace
    return list(ApprovedFace.objects.all())


# =================== Utils ===================
def decode_base64_image(b64_string: str):
    """RAW ya data-uri, dono base64 ko OpenCV BGR numpy me convert karta hai."""
    try:
        if b64_string.startswith("data:image"):
            b64_string = b64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("❌ Base64 decode failed:", e)
        return None


def generate_embedding(image_bgr: np.ndarray, face_box):
    """Face crop -> 160x160 -> FaceNet embedding (L2-normalized) -> np.ndarray."""
    try:
        x1, y1, x2, y2 = [int(v) for v in face_box]
        face_crop = image_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_tensor = preprocess(face_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = facenet_model(face_tensor)
        emb = emb / emb.norm(dim=1, keepdim=True)
        return emb[0].cpu().numpy()
    except Exception as e:
        print("🚨 Error in generate_embedding:", str(e))
        return None


def best_match_for_embedding(embedding: np.ndarray, approved_faces, threshold: float):
    """
    Compare embedding with all approved faces (may have multiple pose entries per person).
    Returns best match dict: {"name": ..., "score": ...} if found, else None.
    """
    best_entry = None
    best_dist = 1e9
    for entry in approved_faces:
        known = pickle.loads(entry.embedding)
        if isinstance(known, torch.Tensor):
            known = known.detach().cpu().numpy()
        dist = cosine(known, embedding)
        if dist < threshold and dist < best_dist:   # Also check score improves
            best_dist = dist
            best_entry = entry

    if best_entry is not None and best_dist < threshold:
        return {"name": best_entry.name, "score": float(round(1.0 - best_dist, 4))}
    return None



# =================== WebSocket Consumer ===================
class FaceStreamConsumer(AsyncWebsocketConsumer):
    mtcnn = mtcnn

    async def connect(self):
        await self.accept()
        await self.send(json.dumps({"message": "✅ WebSocket connected successfully."}))
        print("🔌 Client connected")

    async def disconnect(self, code):
        print("🔌 Client disconnected:", code)

    async def receive(self, text_data):
        """Client JSON: {camera_id, timestamp, image(base64)}"""
        try:
            data = json.loads(text_data)
        except Exception:
            await self.send(json.dumps({"status": "error", "note": "Invalid JSON"}))
            return

        camera_id = data.get("camera_id") or "CAM"
        timestamp = data.get("timestamp") or ""
        img_b64 = data.get("image")

        if not img_b64:
            await self.send(json.dumps({"status": "error", "note": "Missing 'image'"}))
            return

        frame_bgr = decode_base64_image(img_b64)
        if frame_bgr is None:
            await self.send(json.dumps({"status": "error", "note": "❌ Could not decode image"}))
            return

        resized = cv2.resize(frame_bgr, (DETECT_W, DETECT_H))

        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(image_rgb)

        results = []
        if boxes is not None and len(boxes) > 0:
            probs_list = probs.tolist() if probs is not None else [None] * len(boxes)
            for idx, (box, prob) in enumerate(zip(boxes, probs_list)):
                if prob is not None and prob >= CONFIDENCE_THRESHOLD:
                    face_box = [int(v) for v in box]
                    emb = generate_embedding(resized, face_box)
                    if emb is None:
                        results.append({
                            "status": "error",
                            "note": "❌ Embedding generation failed",
                            "box": face_box
                        })
                        continue
                    approved = await get_all_approved_faces()
                    match = best_match_for_embedding(emb, approved, MATCH_THRESHOLD)
                    if match:
                        result = {
                            "status": "match_found",
                            "matched_name": match["name"],
                            "match_score": float(match["score"]),
                            "box": face_box
                        }
                    else:
                        result = {
                            "status": "no_match",
                            "box": face_box
                        }
                    results.append(result)
        else:
            results.append({"status": "no_face"})

        response = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "faces": results  # list of all faces with match/no_match status
        }
        await self.send(json.dumps(response))
