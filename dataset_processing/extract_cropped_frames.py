import pandas as pd
import cv2
import os
from multiprocessing import Pool
from tqdm import tqdm

# --- KONFIGURASI PATH ---
# Sesuaikan dengan struktur Drive kamu
base_video_path = '/content/drive/MyDrive/Dataset_TA/'
output_folder = '/content/frames/'
os.makedirs(output_folder, exist_ok=True)

# 1. BACA FILE CSV
df = pd.read_csv('annotation.csv')

def process_video(row):
    video_file = row['video_name']
    phase = row['phase'] # 'train', 'validation', atau 'test'
    
    if not str(video_file).endswith('.mp4'):
        video_file = str(video_file) + '.mp4'
            
    # Mencari video di subfolder yang sesuai (train/test/validation)
    video_path = os.path.join(base_video_path, phase, video_file)
    
    if not os.path.exists(video_path):
        return
            
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
            
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 25
    
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
    video_rows = df.to_dict('records')
    print(f"🚀 Memulai ekstraksi {len(video_rows)} video...")
    
    with Pool(os.cpu_count()) as pool:
        # Menggunakan tqdm untuk memantau progres
        list(tqdm(pool.imap(process_video, video_rows), total=len(video_rows)))
        
    print(f"✅ SUCCESS: Frame disimpan di {output_folder}")