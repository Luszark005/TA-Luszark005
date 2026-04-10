import pandas as pd
import cv2
import os
import numpy as np
from multiprocessing import Pool
from facenet_pytorch import MTCNN
import torch

# =========================
# CONFIG
# =========================
video_root = '/content/drive/MyDrive/Dataset_TA/'
phases = ['training', 'validation', 'test']

BASE_DIR = '/content/drive/MyDrive/Dataset_TA/clean_dataset/'
OUTPUT_DIR = os.path.join(BASE_DIR, 'images/')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ANNOTATION_CSV = 'annotation.csv'

NUM_SEGMENTS = 5
FRAME_STRIDE = 5
DEBUG = False

# =========================
# DEVICE & MODEL
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(ANNOTATION_CSV)
videos = df['video_name'].tolist()

if DEBUG:
    videos = videos[:50]

cpu_count = os.cpu_count() or 1
chunk_size = max(1, len(videos) // cpu_count)
video_chunks = [videos[i:i + chunk_size] for i in range(0, len(videos), chunk_size)]


# =========================
# HELPER
# =========================
def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# =========================
# MAIN FUNCTION
# =========================
def process_video_chunk(video_chunk):

    for video_file in video_chunk:
        try:
            if not str(video_file).endswith('.mp4'):
                video_file += '.mp4'

            base = os.path.splitext(video_file)[0]
            video_out_dir = os.path.join(OUTPUT_DIR, base)
            os.makedirs(video_out_dir, exist_ok=True)

            # cari video path
            video_path = None
            for phase in phases:
                temp_path = os.path.join(video_root, phase, video_file)
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break

            if video_path is None:
                continue

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # buffer segment
            segment_buffers = [[] for _ in range(NUM_SEGMENTS)]

            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # =========================
                # STRIDE SAMPLING
                # =========================
                if frame_idx % FRAME_STRIDE == 0:

                    # tentukan segment
                    segment_idx = int((frame_idx / total_frames) * NUM_SEGMENTS)
                    segment_idx = min(segment_idx, NUM_SEGMENTS - 1)

                    segment_buffers[segment_idx].append(frame)

                frame_idx += 1

            cap.release()

            # =========================
            # SELECT BEST FRAME PER SEGMENT
            # =========================
            selected_frames = []

            for segment in segment_buffers:

                if len(segment) == 0:
                    continue

                best_score = -1
                best_frame = None

                for frame in segment:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes, _, _ = mtcnn.detect(rgb)

                    if boxes is not None:
                        sharpness = compute_sharpness(frame)

                        if sharpness > best_score:
                            best_score = sharpness
                            best_frame = frame

                # fallback
                if best_frame is None:
                    best_frame = segment[0]

                selected_frames.append(best_frame)

            # =========================
            # ENSURE 5 FRAME
            # =========================
            while len(selected_frames) < NUM_SEGMENTS:
                selected_frames.append(selected_frames[-1])

            # =========================
            # SAVE
            # =========================
            for i, frame in enumerate(selected_frames):
                save_path = os.path.join(video_out_dir, f"frame_{i:02d}.jpg")
                cv2.imwrite(save_path, frame)

        except Exception as e:
            print(f"Error {video_file}: {e}")


# =========================
# RUN
# =========================
if __name__ == '__main__':
    with Pool(processes=cpu_count) as pool:
        pool.map(process_video_chunk, video_chunks)

    print("✅ DONE: Direct best-frame extraction selesai")