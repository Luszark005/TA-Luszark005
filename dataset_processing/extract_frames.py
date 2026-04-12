import pandas as pd
import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm

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
BATCH_SIZE = 1000
MTCNN_BATCH = 128   # 🔥 ini kunci performa (bisa 8–32 tergantung GPU)
DEBUG = False

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
videos = df['video_name'].tolist()

if DEBUG:
    videos = videos[:50]

# =========================
# HELPER
# =========================
def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


def split_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


# =========================
# 🔥 BATCH DETECTION FUNCTION
# =========================
def batch_detect(frames):

    results = []

    for i in range(0, len(frames), MTCNN_BATCH):
        batch = frames[i:i + MTCNN_BATCH]

        rgb_batch = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]

        boxes, probs, landmarks = mtcnn.detect(rgb_batch, landmarks=True)

        for j in range(len(batch)):
            results.append((
                boxes[j] if boxes is not None else None,
                probs[j] if probs is not None else None,
                landmarks[j] if landmarks is not None else None
            ))

    return results


# =========================
# MAIN PROCESS FUNCTION
# =========================
def process_video(video_file):

    try:
        if not str(video_file).endswith('.mp4'):
            video_file += '.mp4'

        base = os.path.splitext(video_file)[0]
        video_out_dir = os.path.join(OUTPUT_DIR, base)

        # 🔥 GRANULAR SKIP
        if os.path.exists(video_out_dir):
            existing = [f for f in os.listdir(video_out_dir) if f.endswith('.jpg')]
            if len(existing) >= NUM_SEGMENTS:
                return

        os.makedirs(video_out_dir, exist_ok=True)

        # cari video
        video_path = None
        for phase in phases:
            temp = os.path.join(video_root, phase, video_file)
            if os.path.exists(temp):
                video_path = temp
                break

        if video_path is None:
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        segment_buffers = [[] for _ in range(NUM_SEGMENTS)]
        frame_idx = 0

        # =========================
        # READ VIDEO
        # =========================
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STRIDE == 0:
                seg_idx = int((frame_idx / total_frames) * NUM_SEGMENTS)
                seg_idx = min(seg_idx, NUM_SEGMENTS - 1)
                segment_buffers[seg_idx].append(frame)

            frame_idx += 1

        cap.release()

        # =========================
        # SELECT BEST FRAME (BATCH)
        # =========================
        selected_frames = []

        for segment in segment_buffers:

            if len(segment) == 0:
                continue

            detections = batch_detect(segment)

            best_score = -float('inf')
            best_frame = None

            for frame, (boxes, probs, landmarks) in zip(segment, detections):

                if boxes is not None and landmarks is not None:

                    sharpness = compute_sharpness(frame)
                    alignment = compute_landmark_alignment(landmarks)

                    score = 0.8 * sharpness + 0.2 * alignment

                    if score > best_score:
                        best_score = score
                        best_frame = frame

            if best_frame is None:
                best_frame = segment[0]

            selected_frames.append(best_frame)

        if len(selected_frames) == 0:
            return

        while len(selected_frames) < NUM_SEGMENTS:
            selected_frames.append(selected_frames[-1])

        # =========================
        # SAVE
        # =========================
        for i, frame in enumerate(selected_frames):
            path = os.path.join(video_out_dir, f"frame_{i:02d}.jpg")
            cv2.imwrite(path, frame)

    except Exception as e:
        print(f"Error {video_file}: {e}")


# =========================
# RUN
# =========================
if __name__ == '__main__':

    batches = split_batches(videos, BATCH_SIZE)

    for i, batch in enumerate(batches):
        print(f"\n🚀 Processing batch {i} ({len(batch)} videos)\n")

        for video_file in tqdm(batch, desc=f"Batch {i}"):
            process_video(video_file)

        print(f"✅ Finished batch {i}")

    print("\n🎉 ALL DONE")