import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# =========================
# CONFIG
# =========================
FRAMES_DIR = '/content/frames/'
OUTPUT_DIR = '/content/dataset/images/'
ANNOTATION_CSV = '/content/annotation.csv'

NUM_SEGMENTS = 5
SAMPLES_PER_SEGMENT = 5
IMG_SIZE = 224
DEBUG = False

random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DEVICE & MODEL
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mtcnn = MTCNN(keep_all=True, device=device)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(ANNOTATION_CSV)
video_names = df['video_name'].tolist()

if DEBUG:
    video_names = video_names[:50]

# =========================
# FUNCTIONS
# =========================
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


# =========================
# MAIN
# =========================
for idx, video_name in enumerate(tqdm(video_names)):

    base = os.path.splitext(video_name)[0]
    video_out_dir = os.path.join(OUTPUT_DIR, base)
    os.makedirs(video_out_dir, exist_ok=True)

    frames = [f for f in os.listdir(FRAMES_DIR) if f.startswith(base)]

    if len(frames) == 0:
        continue

    frames = sorted(frames)
    segments = np.array_split(frames, NUM_SEGMENTS)

    selected_frames = []

    for segment in segments:

        segment = list(segment)
        if len(segment) == 0:
            continue

        sampled = random.sample(segment, min(SAMPLES_PER_SEGMENT, len(segment)))

        best_score = float('-inf')
        best_crop = None

        for frame_name in sampled:
            frame_path = os.path.join(FRAMES_DIR, frame_name)
            image = load_image(frame_path)

            if image is None:
                continue

            boxes, probs, landmarks = detect_face(image)

            if boxes is not None and landmarks is not None:
                x1, y1, x2, y2 = boxes[0]
                h, w, _ = image.shape

                x1, y1 = int(max(0, x1)), int(max(0, y1))
                x2, y2 = int(min(w, x2)), int(min(h, y2))

                crop = image[y1:y2, x1:x2]

                # =========================
                # 🔥 RESIZE FACE
                # =========================
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

                sharpness = compute_sharpness(crop)
                alignment = compute_landmark_alignment(landmarks)

                score = 0.8 * sharpness + 0.2 * alignment

                if score > best_score:
                    best_score = score
                    best_crop = crop

        # =========================
        # FALLBACK
        # =========================
        if best_crop is None:
            fallback_choice = random.choice(segment)
            frame_path = os.path.join(FRAMES_DIR, fallback_choice)
            image = load_image(frame_path)

            if image is not None:
                boxes, _, _ = detect_face(image)

                if boxes is not None:
                    x1, y1, x2, y2 = boxes[0]
                    h, w, _ = image.shape

                    x1, y1 = int(max(0, x1)), int(max(0, y1))
                    x2, y2 = int(min(w, x2)), int(min(h, y2))

                    best_crop = image[y1:y2, x1:x2]
                else:
                    # fallback full image
                    best_crop = image

                # 🔥 RESIZE fallback juga
                best_crop = cv2.resize(best_crop, (IMG_SIZE, IMG_SIZE))

        selected_frames.append(best_crop)

    # =========================
    # ENSURE FIXED LENGTH (NO BLACK FRAME)
    # =========================
    while len(selected_frames) < NUM_SEGMENTS:
        if len(selected_frames) > 0:
            # duplikat frame terakhir
            selected_frames.append(selected_frames[-1])
        else:
            # fallback: ambil frame random dari video asli
            fallback_frame = random.choice(frames)
            frame_path = os.path.join(FRAMES_DIR, fallback_frame)
            image = load_image(frame_path)

            if image is not None:
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                selected_frames.append(image)

    # =========================
    # SAVE RESULT
    # =========================
    for i, img in enumerate(selected_frames):
        if img is None:
            continue

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(video_out_dir, f"frame_{i:02d}.jpg")
        cv2.imwrite(save_path, img_bgr)

    # =========================
    # DELETE RAW FRAMES
    # =========================
    if not DEBUG:
        for frame_name in frames:
            frame_path = os.path.join(FRAMES_DIR, frame_name)
            try:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            except Exception as e:
                print(f"Gagal hapus {frame_path}: {e}")

    # =========================
    # 🔥 LOGGING PROGRESS
    # =========================
    if idx % 100 == 0:
        print(f"[INFO] Processed {idx}/{len(video_names)} videos")

print("✅ DONE: 5 best frames per video saved.")