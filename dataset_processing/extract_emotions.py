import os
import pandas as pd
import numpy as np
import shutil
from deepface import DeepFace
from tqdm import tqdm

# --- KONFIGURASI PATH ---
DATASET_CSV = '/content/dataset/annotation.csv'
# Folder input (hasil dari extract_cropped_frames.py)
IMAGES_DIR = '/content/dataset_images/' 

# Folder output untuk file emosi (.npy)
EMOTIONS_OUT_DIR = '/content/dataset/emotions'
# Folder output untuk gambar final (yang akan dibaca oleh dataset.py)
FINAL_IMAGES_DIR = '/content/dataset/images'

NUM_FRAMES = 8 

# Membuat folder jika belum ada
os.makedirs(EMOTIONS_OUT_DIR, exist_ok=True)
os.makedirs(FINAL_IMAGES_DIR, exist_ok=True)

def extract_emotion_from_frame(image_path):
    """Mengekstrak vektor emosi 7-dimensi menggunakan DeepFace."""
    try:
        # Menganalisis emosi tanpa deteksi wajah ulang (karena sudah di-crop MTCNN)
        result = DeepFace.analyze(img_path=image_path, align=False, enforce_detection=False, actions=['emotion'], silent=True)
        emotion_probs = result[0]['emotion']
        emotion_vector = np.array([
            emotion_probs['angry'], emotion_probs['disgust'], emotion_probs['fear'],
            emotion_probs['happy'], emotion_probs['sad'], emotion_probs['surprise'],
            emotion_probs['neutral']
        ]) / 100.0
        return emotion_vector.astype(np.float32)
    except:
        # Fallback ke emosi Netral jika terjadi error
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

if __name__ == "__main__":
    if not os.path.exists(DATASET_CSV):
        print(f"❌ ERROR: File {DATASET_CSV} tidak ditemukan!")
    else:
        df = pd.read_csv(DATASET_CSV)
        video_names = df['video_name'].tolist()

        print(f"🚀 Memproses {len(video_names)} video (Ekstraksi Emosi & Penyelarasan Frame)...")

        for video_name in tqdm(video_names, desc="Progress"):
            video_basename = os.path.splitext(video_name)[0]
            video_folder = os.path.join(IMAGES_DIR, video_basename)
            
            # Lewati jika folder video tidak ditemukan
            if not os.path.exists(video_folder):
                continue
                
            # Ambil semua frame hasil crop dan urutkan
            all_frames = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
            total_available = len(all_frames)
            
            if total_available == 0:
                continue

            # Tentukan folder tujuan untuk gambar final video ini
            final_video_img_dir = os.path.join(FINAL_IMAGES_DIR, video_basename)
            os.makedirs(final_video_img_dir, exist_ok=True)
            
            # LOGIKA STRIDE: Mengambil 8 frame secara merata dari total yang ada
            stride = max(1, total_available // NUM_FRAMES)
            
            emotion_sequence = []
            for i in range(NUM_FRAMES):
                idx = i * stride
                
                # Nama file baru agar cocok dengan dataset.py
                target_frame_name = f"frame_{i:02d}.jpg"
                target_frame_path = os.path.join(final_video_img_dir, target_frame_name)

                if idx < total_available:
                    source_frame_path = os.path.join(video_folder, all_frames[idx])
                    
                    # 1. Ekstrak Emosi
                    emotion_vector = extract_emotion_from_frame(source_frame_path)
                    emotion_sequence.append(emotion_vector)
                    
                    # 2. PENYELARASAN: Salin dan Rename frame terpilih
                    shutil.copy(source_frame_path, target_frame_path)
                else:
                    # Padding jika frame kurang dari 8
                    emotion_sequence.append(np.array([0,0,0,0,0,0,1.0], dtype=np.float32))
                
            # Simpan urutan emosi ke file .npy
            emotion_sequence_np = np.array(emotion_sequence) # Ukuran (8, 7)
            out_filepath = os.path.join(EMOTIONS_OUT_DIR, f"{video_basename}.npy")
            np.save(out_filepath, emotion_sequence_np)

        print(f"✅ SUCCESS: Emosi disimpan di {EMOTIONS_OUT_DIR}")
        print(f"✅ SUCCESS: Gambar final diselaraskan di {FINAL_IMAGES_DIR}")