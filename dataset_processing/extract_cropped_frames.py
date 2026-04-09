import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# Conmon Config 
FRAMES_DIR = '/content/frames/'
OUTPUT_DIR = '/content/dataset/images/'
ANNOTATION_CSV = '/content/annotation.csv'

NUM_SEGMENTS = 5
SAMPLES_PER_SEGMENT = 5

random.seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup MTCNN dengan device yang sesuai (GPU jika tersedia, CPU jika tidak)
# CPU cuman jaga-jaga aja walaupun gk perlu sih di konteks ini
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mtcnn = MTCNN(
    keep_all=True,
    device=device
)

df = pd.read_csv(ANNOTATION_CSV)
video_names = df['video_name'].tolist()

# testing cepat, jangan lupa dihapus
video_names = video_names[:50]

# Functions

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_landmark_alignment(landmarks):
    if landmarks is None:
        return float('-inf')

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[0]

    eye_distance = np.linalg.norm(left_eye - right_eye)
    mouth_width = np.linalg.norm(left_mouth - right_mouth)

    nose_center = (left_eye[1] + right_eye[1]) / 2
    nose_offset = abs(nose[1] - nose_center)

    symmetry_score = eye_distance / (mouth_width + 1e-6)

    return -nose_offset + symmetry_score

def detect_face(image):
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    return boxes, probs, landmarks

 # MAIN 

for video_name in tqdm(video_names):

    base = os.path.splitext(video_name)[0]
    video_out_dir = os.path.join(OUTPUT_DIR, base)
    os.makedirs(video_out_dir, exist_ok=True)

    # ambil semua frame milik video berdasarkan nama file
    frames = [
        f for f in os.listdir(FRAMES_DIR)
        if f.startswith(base)
    ]

    if len(frames) == 0:
        continue

    frames = sorted(frames)

    # split jadi 5 segment
    segments = np.array_split(frames, NUM_SEGMENTS)

    selected_frames = []

    for segment in segments:

        segment = list(segment)
        if len(segment) == 0:
            continue

        # random sample max 5
        sampled = random.sample(segment, min(SAMPLES_PER_SEGMENT, len(segment)))

        best_score = float('-inf')
        best_frame = None
        best_crop = None

        fallback_frames = []

        for frame_name in sampled:
            frame_path = os.path.join(FRAMES_DIR, frame_name)
            image = load_image(frame_path)

            if image is None:
                continue

            boxes, probs, landmarks = detect_face(image)

            if boxes is not None and landmarks is not None:
                sharpness = compute_sharpness(image)
                alignment = compute_landmark_alignment(landmarks)
                score = 0.8 * sharpness + 0.2 * alignment

                x1, y1, x2, y2 = boxes[0]
                h, w, _ = image.shape
                x1, y1 = int(max(0, x1)), int(max(0, y1))
                x2, y2 = int(min(w, x2)), int(min(h, y2))

                crop = image[y1:y2, x1:x2]

                if score > best_score:
                    best_score = score
                    best_frame = frame_name
                    best_crop = crop

            else:
                fallback_frames.append(frame_name)

        # Fallback jika semua sampled frames gagal deteksi wajah → coba random frame dari segment sampai dapat crop wajah terbaik
        if best_crop is None:
            # ambil random frame dari segment
            fallback_choice = random.choice(segment)
            frame_path = os.path.join(FRAMES_DIR, fallback_choice)
            image = load_image(frame_path)

            if image is None:
                continue

            boxes, _, _ = detect_face(image)

            if boxes is not None:
                x1, y1, x2, y2 = boxes[0]
                h, w, _ = image.shape
                x1, y1 = int(max(0, x1)), int(max(0, y1))
                x2, y2 = int(min(w, x2)), int(min(h, y2))
                best_crop = image[y1:y2, x1:x2]
            else:
                # kalau tetap gagal → pakai full image
                best_crop = image

        selected_frames.append(best_crop)

    # Simpan 5 frame terbaik 
    for i, img in enumerate(selected_frames):
        if img is None:
            continue

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(video_out_dir, f"frame_{i:02d}.jpg")
        cv2.imwrite(save_path, img_bgr)

print("DONE: 5 best frames per video saved.")