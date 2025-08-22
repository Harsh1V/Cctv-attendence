import time
import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Django setup (agar django models ya ORM use kar rahe hain)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cctv_backend.settings')
import django
django.setup()

from core.face_utils import compare_embedding_with_known_faces

# Device configuration
device = torch.device('cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def generate_embedding(image, face_box, facenet_model, device):
    """
    image: Original BGR numpy image
    face_box: bounding box (x1, y1, x2, y2) format
    facenet_model: FaceNet model
    device: torch device
    Returns normalized embedding numpy array
    """
    try:
        x1, y1, x2, y2 = [int(coord) for coord in face_box]
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            print("🚨 Face crop empty, skipping embedding")
            return None

        # BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))

        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
        ])

        face_tensor = preprocess(face_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = facenet_model(face_tensor)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)

        return embedding[0].cpu().numpy()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def main():
    cap = cv2.VideoCapture(0)  # ya aapka CCTV stream URL
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    frame_count = 0
    skip_interval = 10
    quality_threshold = 0.90
    match_threshold = 0.6  # Tight threshold to reduce false positives

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        frame_count += 1

        # Resize frame for faster detection
        frame_resized = cv2.resize(frame, (320, 240))

        faces, probs = mtcnn.detect(frame_resized)

        if faces is not None and len(faces) > 0:
            frame_count = 0  # reset skip counter when face detected

            for i, box in enumerate(faces):
                confidence = probs[i]
                if confidence < quality_threshold:
                    continue

                embedding = generate_embedding(frame_resized, box, facenet_model, device)
                if embedding is None:
                    continue

                match_result = compare_embedding_with_known_faces(embedding, threshold=match_threshold)

                # Scale box coords to original frame size
                x1 = int(box[0] * (frame.shape[1] / 320))
                y1 = int(box[1] * (frame.shape[0] / 240))
                x2 = int(box[2] * (frame.shape[1] / 320))
                y2 = int(box[3] * (frame.shape[0] / 240))

                if match_result:
                    label = f"{match_result['name']} ({match_result['score']:.2f})"
                    color = (0, 255, 0)  # green for match
                else:
                    label = "Unknown"
                    color = (0, 0, 255)  # red for unknown

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # Skip frames without faces to save CPU
            if frame_count % skip_interval != 0:
                time.sleep(0.2)
                continue
            cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Live CCTV Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
