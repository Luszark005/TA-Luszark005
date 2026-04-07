import os
import cv2
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# ==============================
# DEVICE
# ==============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# ==============================
# INIT MTCNN
# ==============================
mtcnn = MTCNN(
    keep_all=False,
    device=device,
    post_process=True,
    image_size=224,
    margin=20
)

# ==============================
# PATH
# ==============================
FRAMES_DIR = '/content/frames/'
OUTPUT_DIR = '/content/dataset_images/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD ANNOTATION
# ==============================
df = pd.read_csv('/content/annotation.csv')
videos = df['video_name'].tolist()

# 🔥 TEST DULU
videos = videos[:50]

# ==============================
# MAIN LOOP
# ==============================
for video_name in tqdm(videos):
    base_name = os.path.splitext(video_name)[0]

    video_out_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(video_out_dir, exist_ok=True)

    # ambil semua frame video ini
    video_frames = [
        f for f in os.listdir(FRAMES_DIR)
        if f.startswith(base_name)
    ]

    if len(video_frames) == 0:
        continue

    for frame_file in video_frames:
        img_path = os.path.join(FRAMES_DIR, frame_file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        save_path = os.path.join(video_out_dir, frame_file)

        try:
            mtcnn(img_rgb, save_path=save_path)
        except:
            continue

print("✅ DONE: Cropped faces saved.")