import os
import cv2
import pandas as pd
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm

# 1. SETUP DEVICE & MTCNN
# Gunakan GPU jika tersedia agar proses deteksi wajah jauh lebih cepat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device, post_process=True, image_size=224)

# 2. KONFIGURASI PATH
# Pastikan folder frames berisi hasil dari extract_frames.py
FRAMES_DIR = '/content/frames/'
OUTPUT_DIR = '/content/dataset_images/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Membaca daftar video dari anotasi
if not os.path.exists('annotation.csv'):
    print("❌ ERROR: File annotation.csv tidak ditemukan!")
else:
    df = pd.read_csv('annotation.csv')
    videos = df['video_name'].tolist()

    print(f"🚀 Memulai Face Cropping untuk {len(videos)} video...")

    for video_name in tqdm(videos, desc="Processing Videos"):
        # Hilangkan ekstensi .mp4 untuk mendapatkan base name
        base_name = video_name.replace('.mp4', '')
        
        # --- PERBAIKAN: BUAT SUBFOLDER PER VIDEO ---
        # Ini penting agar extract_emotions.py bisa menemukan folder ini
        video_out_dir = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(video_out_dir, exist_ok=True)
        
        # Mencari semua frame yang diawali dengan nama video tersebut
        video_frames = [f for f in os.listdir(FRAMES_DIR) if f.startswith(base_name) and f.endswith('.jpg')]
        
        if len(video_frames) == 0:
            continue

        for frame_file in video_frames:
            img_path = os.path.join(FRAMES_DIR, frame_file)
            img = cv2.imread(img_path)
            
            if img is None: 
                continue
            
            # Konversi BGR ke RGB (kebutuhan MTCNN)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # --- PERBAIKAN: SIMPAN KE SUBFOLDER ---
            # Path penyimpanan sekarang masuk ke folder masing-masing video
            save_path = os.path.join(video_out_dir, frame_file)
            
            try:
                # MTCNN akan mendeteksi wajah, meng-crop, dan menyimpannya langsung
                mtcnn(img_rgb, save_path=save_path)
            except Exception as e:
                # Jika gagal (misal tidak ada wajah terdeteksi), file tidak akan tersimpan
                pass

    print(f"✅ SUCCESS: Hasil crop wajah disimpan di {OUTPUT_DIR}")