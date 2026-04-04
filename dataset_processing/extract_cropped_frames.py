import os
import cv2
import pandas as pd
from facenet_pytorch import MTCNN
import torch

# Gunakan GPU jika tersedia agar Cepat!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)

# --- KONFIGURASI ---
FRAMES_DIR = '/content/frames/'
OUTPUT_DIR = '/content/dataset_images/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv('annotation.csv')
videos = df['video_name'].tolist()

for video_name in videos:
    # Hilangkan ekstensi .mp4 jika ada untuk mencari base name
    base_name = video_name.replace('.mp4', '')
    
    # MENCARI SEMUA FRAME MILIK VIDEO INI
    # Contoh: mencari semua yang diawali '05l5bteT_qA.001_frame'
    video_frames = [f for f in os.listdir(FRAMES_DIR) if f.startswith(base_name) and f.endswith('.jpg')]
    
    if len(video_frames) == 0:
        print(f"Skip {base_name}: Tidak ditemukan file .jpg di folder frames.")
        continue

    print(f"Processing: {base_name} ({len(video_frames)} frames)")

    for frame_file in video_frames:
        img_path = os.path.join(FRAMES_DIR, frame_file)
        img = cv2.imread(img_path)
        
        if img is None: continue
        
        # Deteksi dan Crop Wajah
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # MTCNN akan me-return gambar yang sudah di-crop dan di-resize (misal 224)
        save_path = os.path.join(OUTPUT_DIR, frame_file)
        
        try:
            # mtcnn(image, save_path) langsung menyimpan hasil crop wajah
            mtcnn(img_rgb, save_path=save_path)
        except Exception as e:
            print(f"Gagal crop {frame_file}: {e}")
