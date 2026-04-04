import pandas as pd
import cv2
import os
import torch
from facenet_pytorch import MTCNN
import numpy as np
from multiprocessing import Pool

# --- KONFIGURASI MULTI-FRAME ---
NUM_FRAMES = 8 # Jumlah frame yang ingin diekstrak per video (Sesuaikan dengan kebutuhanmu)
DATASET_CSV = 'dataset/annotation.csv'
FRAMES_DIR = 'frames'
OUTPUT_DIR = './dataset/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def retrieve_video_frames(video_name, all_images):
    video_basename = os.path.splitext(video_name)[0]
    # Ambil semua gambar yang namanya mengandung ID video ini
    video_images = [img for img in all_images if img.startswith(video_basename)]
    
    # PENTING: Urutkan frame berdasarkan angka agar urutan waktunya benar
    def extract_frame_num(filename):
        try:
            return int(filename.split('_frame')[-1].split('.jpg')[0])
        except ValueError:
            return 0
            
    video_images.sort(key=extract_frame_num)
    return video_images

def load_image(path):
    image = cv2.imread(os.path.join(FRAMES_DIR, path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Inisialisasi MTCNN
mtcnn = MTCNN(keep_all=True, device=torch.device("cpu"))

def process_video_chunk(video_chunk):
    # Baca folder frames sekali saja per chunk agar proses I/O lebih cepat
    all_images = os.listdir(FRAMES_DIR)
    
    for video_name in video_chunk:
        video_basename = os.path.splitext(video_name)[0]
        print(f"Processing: {video_basename}")
        
        frames = retrieve_video_frames(video_name, all_images)
        
        if len(frames) < NUM_FRAMES:
            print(f"Skip {video_basename}: Hanya memiliki {len(frames)} frames.")
            continue
            
        # 1. UNIFORM SAMPLING (Ambil N frame dengan jarak beraturan)
        step = len(frames) / NUM_FRAMES
        selected_indices = [int(i * step) for i in range(NUM_FRAMES)]
        selected_frames = [frames[i] for i in selected_indices]
        
        # 2. BUAT FOLDER KHUSUS UNTUK VIDEO INI
        # Output: ./dataset/images/NamaVideo/
        video_out_dir = os.path.join(OUTPUT_DIR, video_basename)
        os.makedirs(video_out_dir, exist_ok=True)
        
        # 3. DETEKSI & POTONG WAJAH UNTUK SETIAP FRAME YANG DIPILIH
        for i, img_path in enumerate(selected_frames):
            image = load_image(img_path)
            boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
            
            h, w, _ = image.shape
            
            if boxes is not None:
                # Ambil wajah utama (probabilitas tertinggi dari MTCNN)
                x1, y1, x2, y2 = boxes[0]
                x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
                cropped_face = image[y1:y2, x1:x2]
            else:
                # FALLBACK (PENYELAMAT): Jika MTCNN gagal mendeteksi wajah (misal karena blur/menoleh)
                # Kita terpaksa potong tengah (Center Crop) agar jumlah gambar tetap genap 8 frame.
                cy, cx = h // 2, w // 2
                size = min(h, w) // 2
                cropped_face = image[cy-size:cy+size, cx-size:cx+size]
                
            # Resize agar semua wajah ukurannya pasti sama (wajib untuk Transformer)
            cropped_face = cv2.resize(cropped_face, (224, 224))
            
            # Simpan dengan urutan nama: frame_00.jpg, frame_01.jpg, dst
            out_filename = os.path.join(video_out_dir, f"frame_{i:02d}.jpg")
            cv2.imwrite(out_filename, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    df = pd.read_csv(DATASET_CSV)
    videos = df['video_name'].unique().tolist()
    
    chunk_size = max(1, len(videos) // os.cpu_count())
    video_chunks = [videos[i:i + chunk_size] for i in range(0, len(videos), chunk_size)]
    
    with Pool() as pool:
        pool.map(process_video_chunk, video_chunks)