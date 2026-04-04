import pandas as pd
import cv2
import os
from multiprocessing import Pool

# --- UBAH ALAMAT VIDEO KE LOKAL COLAB ---
video_folder = '/content/dataset_raw/training/'

# Output folder kita buat absolut saja biar aman
output_folder = '/content/frames/'
os.makedirs(output_folder, exist_ok=True)

# 1. BACA FILE CSV DARI FOLDER YANG SAMA
df = pd.read_csv('annotation.csv')
videos = df['video_name'].tolist()

# Mencegah error jika jumlah video lebih sedikit dari jumlah CPU
cpu_count = os.cpu_count() or 1
chunk_size = max(1, len(videos) // cpu_count)
video_chunks = [videos[i:i + chunk_size] for i in range(0, len(videos), chunk_size)]

def process_video_chunk(video_chunk):
    for video_file in video_chunk:
        # Berjaga-jaga jika nama video di CSV tidak berakhiran .mp4
        if not str(video_file).endswith('.mp4'):
            video_file = str(video_file) + '.mp4'
            
        video_path = os.path.join(video_folder, video_file)
        
        # Cek apakah videonya benar-benar ada di folder
        if not os.path.exists(video_path):
            continue
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 25 # Fallback aman
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ambil 1 frame setiap 1 detik
            if frame_count % fps == 0:
                frame_filename = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_frame{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
        
        cap.release()

if __name__ == '__main__':
    print(f"🚀 Memulai ekstraksi {len(videos)} video menggunakan {cpu_count} CPU core...")
    print("⏳ Mohon tunggu, proses ini butuh waktu beberapa menit. Jangan tutup tab Colab-nya.")
    
    # Eksekusi Multiprocessing
    with Pool() as pool:
        pool.map(process_video_chunk, video_chunks)
        
    print("✅ SUCCESS: Semua frame berhasil diekstrak ke folder /content/frames/")