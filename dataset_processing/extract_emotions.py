import os
import pandas as pd
import numpy as np
import shutil
from deepface import DeepFace
from tqdm import tqdm

# ================= CONFIG =================
DATASET_CSV = '/content/annotation.csv'
IMAGES_DIR = '/content/dataset/images/'   # hasil dari extract_cropped_frames
FINAL_IMAGES_DIR = '/content/dataset/images_emotions' # folder baru untuk menyimpan frame yang sudah dipasangi emosi
EMOTIONS_OUT_DIR = '/content/dataset/emotions'

NUM_FRAMES = 5

os.makedirs(EMOTIONS_OUT_DIR, exist_ok=True)
os.makedirs(FINAL_IMAGES_DIR, exist_ok=True)

# ================= FUNCTION =================
def extract_emotion_from_frame(image_path):
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            align=False,
            enforce_detection=False,
            actions=['emotion'],
            silent=True
        )

        emotion_probs = result[0]['emotion']

        emotion_vector = np.array([
            emotion_probs['angry'],
            emotion_probs['disgust'],
            emotion_probs['fear'],
            emotion_probs['happy'],
            emotion_probs['sad'],
            emotion_probs['surprise'],
            emotion_probs['neutral']
        ]) / 100.0

        return emotion_vector.astype(np.float32)

    except:
        # fallback → neutral
        return np.array([0,0,0,0,0,0,1.0], dtype=np.float32)

# ================= MAIN =================
if __name__ == "__main__":

    df = pd.read_csv(DATASET_CSV)
    video_names = df['video_name'].tolist()

    # testing cepat (optional)
    #video_names = video_names[:50]

    print(f"🚀 Processing {len(video_names)} videos...")

    for video_name in tqdm(video_names):
        video_basename = os.path.splitext(video_name)[0]
        video_folder = os.path.join(IMAGES_DIR, video_basename)

        if not os.path.exists(video_folder):
            continue

        # folder output final (yang dipakai training)
        final_video_img_dir = os.path.join(FINAL_IMAGES_DIR, video_basename)
        os.makedirs(final_video_img_dir, exist_ok=True)

        emotion_sequence = []

        for i in range(NUM_FRAMES):
            frame_name = f"frame_{i:02d}.jpg"
            source_frame_path = os.path.join(video_folder, frame_name)
            target_frame_path = os.path.join(final_video_img_dir, frame_name)

            if os.path.exists(source_frame_path):
                # extract emotion
                emotion_vector = extract_emotion_from_frame(source_frame_path)
                emotion_sequence.append(emotion_vector)

                # copy ke folder final
                shutil.copy(source_frame_path, target_frame_path)

            else:
                # fallback kalau frame missing
                emotion_sequence.append(
                    np.array([0,0,0,0,0,0,1.0], dtype=np.float32)
                )

        # save emotion sequence (5,7)
        emotion_sequence_np = np.array(emotion_sequence)
        out_filepath = os.path.join(EMOTIONS_OUT_DIR, f"{video_basename}.npy")
        np.save(out_filepath, emotion_sequence_np)

    print("✅ DONE: Emotion extraction selesai")