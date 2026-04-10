import os
import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_DIR = '/content/drive/MyDrive/Dataset_TA/clean_dataset/'

INPUT_DIR = os.path.join(BASE_DIR, 'images/')              # hasil extract_frames
OUTPUT_DIR = os.path.join(BASE_DIR, 'images_cropped/')     # hasil final

ANNOTATION_CSV = '/content/annotation.csv'
IMG_SIZE = 224
DEBUG = False

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

# =========================
# MAIN
# =========================
for idx, video_name in enumerate(tqdm(video_names)):

    base = os.path.splitext(video_name)[0]

    input_video_dir = os.path.join(INPUT_DIR, base)
    output_video_dir = os.path.join(OUTPUT_DIR, base)

    if not os.path.exists(input_video_dir):
        continue

    os.makedirs(output_video_dir, exist_ok=True)

    frames = sorted(os.listdir(input_video_dir))

    for frame_name in frames:
        frame_path = os.path.join(input_video_dir, frame_name)
        image = load_image(frame_path)

        if image is None:
            continue

        boxes, probs, landmarks = mtcnn.detect(image)

        # =========================
        # FACE CROP
        # =========================
        if boxes is not None:
            x1, y1, x2, y2 = boxes[0]
            h, w, _ = image.shape

            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(w, x2)), int(min(h, y2))

            crop = image[y1:y2, x1:x2]
        else:
            # fallback: pakai full image
            crop = image

        # =========================
        # RESIZE
        # =========================
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

        # =========================
        # SAVE
        # =========================
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(output_video_dir, frame_name)
        cv2.imwrite(save_path, crop_bgr)

    # =========================
    # LOGGING
    # =========================
    if idx % 100 == 0:
        print(f"[INFO] Processed {idx}/{len(video_names)} videos")

print("✅ DONE: Cropping & resizing selesai.")