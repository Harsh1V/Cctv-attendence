import base64
import cv2
import numpy as np
from PIL import Image
import io
import pickle
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
from torchvision import transforms
from asgiref.sync import sync_to_async

def get_approved_face_model():
    from core.models import ApprovedFace
    return ApprovedFace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def decode_base64_image(base64_string):
    try:
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("❌ Base64 decode failed:", e)
        return None
    
def detect_face(image):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(image_rgb)
        print(f"Detected boxes: {boxes}")
        print(f"Detection probabilities: {probs}")
        if boxes is not None:
            return boxes
        return []
    except Exception as e:
        print(f"Face detection error: {e}")
        return []
    



def generate_embedding(image, face_box, facenet_model, device):
    """
    Generate L2 normalized face embedding from bounding box.
    """
    try:
        x1, y1, x2, y2 = [int(coord) for coord in face_box]
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            print("🚨 Face crop empty, invalid bounding box")
            return None

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        face_tensor = preprocess(face_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = facenet_model(face_tensor)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)
        return embedding[0].cpu().numpy()

    except Exception as e:
        print(f"Error in generate_embedding: {str(e)}")
        return None

def save_embedding(name, embedding, pose):
    from core.models import ApprovedFace
    if ApprovedFace.objects.filter(name=name, pose=pose).exists():
        print(f"⚠️ {name}({pose}) already exists.")
        return
    embedding_bytes = pickle.dumps(embedding)
    ApprovedFace.objects.create(name=name, pose=pose, embedding=embedding_bytes)
    print(f"✅ {name} with pose {pose} added.")


@sync_to_async
def compare_embeddings_with_known_faces_sync(embeddings, threshold=0.9):
    return compare_embeddings_with_known_faces(embeddings, threshold)

def compare_embeddings_with_known_faces(embeddings, threshold=0.9):
    """
    embeddings: List of embedding numpy arrays
    Returns list of dicts: { "name": ..., "score": ..., "box": ... } for matches, None if no match
    """
    ApprovedFace = get_approved_face_model()
    results = []
    try:
        known_faces = ApprovedFace.objects.all()

        for item in embeddings:
            emb = item["embedding"]
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            match_found = None
            for entry in known_faces:
                name = entry.name
                known_embedding = pickle.loads(entry.embedding)
                if isinstance(known_embedding, torch.Tensor):
                    known_embedding = known_embedding.detach().cpu().numpy()
                distance = cosine(known_embedding, emb)
                print(f"📏 Distance with {name}: {distance:.4f}")
                if distance < threshold:
                    print(f"✅ Match found: {name} (distance={distance:.4f})")
                    match_found = {"name": name, "score": float(round(1 - distance, 4)), "box": item["box"]}
                    break
            if match_found:
                results.append(match_found)
            else:
                results.append({"name": None, "score": None, "box": item["box"]})
        return results
    except Exception as e:
        print(f"🚨 Error in compare_embeddings_with_known_faces: {str(e)}")
        return []


