import os
import pandas as pd
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

# --- KONFIGURASI ---
DATASET_CSV = '/content/dataset/annotation.csv'
IMAGES_DIR = '/content/dataset_images/'
EMOTIONS_OUT_DIR = '/content/dataset/emotions'
NUM_FRAMES = 8 

os.makedirs(EMOTIONS_OUT_DIR, exist_ok=True)

def extract_emotion_from_frame(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, align=False, enforce_detection=False, actions=['emotion'], silent=True)
        emotion_probs = result[0]['emotion']
        emotion_vector = np.array([
            emotion_probs['angry'], emotion_probs['disgust'], emotion_probs['fear'],
            emotion_probs['happy'], emotion_probs['sad'], emotion_probs['surprise'],
            emotion_probs['neutral']
        ]) / 100.0
        return emotion_vector.astype(np.float32)
    except:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

if __name__ == "__main__":
    df = pd.read_csv(DATASET_CSV)
    video_names = df['video_name'].tolist()

    for video_name in tqdm(video_names, desc="Mengekstrak 8 Emosi"):
        video_basename = os.path.splitext(video_name)[0]
        video_folder = os.path.join(IMAGES_DIR, video_basename)
        
        if not os.path.exists(video_folder):
            continue
            
        all_frames = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
        
        # LOGIKA STRIDE: Mengambil 8 frame dari total 16 yang ada
        # Jika ada 16 frame, stride = 2 (ambil index 0, 2, 4, 6, 8, 10, 12, 14)
        total_available = len(all_frames)
        stride = max(1, total_available // NUM_FRAMES)
        
        emotion_sequence = []
        for i in range(NUM_FRAMES):
            idx = i * stride
            # Pastikan tidak out of bounds
            if idx < total_available:
                frame_path = os.path.join(video_folder, all_frames[idx])
                emotion_vector = extract_emotion_from_frame(frame_path)
                emotion_sequence.append(emotion_vector)
            else:
                # Padding jika frame kurang dari 8
                emotion_sequence.append(np.array([0,0,0,0,0,0,1.0], dtype=np.float32))
            
        emotion_sequence_np = np.array(emotion_sequence) # Ukuran (8, 7)
        out_filepath = os.path.join(EMOTIONS_OUT_DIR, f"{video_basename}.npy")
        np.save(out_filepath, emotion_sequence_np)
