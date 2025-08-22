import os
import django

# Set DJANGO_SETTINGS_MODULE for your project
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cctv_backend.settings')
django.setup()

import torch
import cv2
import pickle
from facenet_pytorch import InceptionResnetV1
from core.face_utils import generate_embedding, detect_face, save_embedding  # Must have save_embedding(name, embedding, pose)
from core.models import ApprovedFace

device = torch.device("cpu")
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Path to known faces folders (folder per person)
KNOWN_FOLDER = "cctv_backend/known_faces"

POSE_LIST = ["front", "left", "right"]

# Loop through each person folder
for person_name in os.listdir(KNOWN_FOLDER):
    person_path = os.path.join(KNOWN_FOLDER, person_name)
    if not os.path.isdir(person_path):
        continue
    print(f"\nProcessing person: {person_name}")

    for pose in POSE_LIST:
        img_filename = f"{pose}.jpg"
        img_path = os.path.join(person_path, img_filename)
        if not os.path.exists(img_path):
            print(f"❌ Missing {pose}.jpg for {person_name}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Failed to load image: {img_path}")
            continue

        faces = detect_face(image)
        if len(faces) == 0:
            print(f"❌ No face found in: {img_filename}")
            continue

        try:
            embedding = generate_embedding(image, faces[0], facenet_model, device)
        except Exception as e:
            print(f"Embedding failed for: {img_filename} | Error: {str(e)}")
            continue

        try:
            save_embedding(person_name, embedding, pose)  # Save name + embedding + pose
            print(f"✅ {person_name} ({pose}) embedding saved to the database.")
        except Exception as e:
            print(f"Failed to save {person_name} ({pose}) to database. Error: {str(e)}")
            continue

print("\n✅ All known faces processed and multi-pose embeddings saved to the database!")
