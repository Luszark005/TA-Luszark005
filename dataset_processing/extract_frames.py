import pandas as pd
import cv2
import os
from multiprocessing import Pool

# --- UBAH ALAMAT VIDEO KE LOKAL COLAB ---
video_folder = '/content/dataset_raw/training/'

# Output folder kita buat absolut saja biar aman
output_folder = '/content/frames/'

os.makedirs(output_folder, exist_ok=True)

# File CSV ini akan dibaca langsung dari folder GitHub yang sudah kamu clone
df = pd.read_csv('./dataset/annotation.csv')

videos = df['video_name'].tolist()


chunk_size = len(videos) // os.cpu_count()
video_chunks = [videos[i:i + chunk_size] for i in range(0, len(videos), chunk_size)]

def process_video_chunk(video_chunk):
    for video_file in video_chunk:
        video_path = os.path.join(video_folder, video_file)
        
        if video_file.endswith('.mp4'):
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % int(fps) == 0:
                    frame_filename = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_frame{frame_count}.jpg")
                    cv2.imwrite(frame_filename, frame)
                
                frame_count += 1
            
            cap.release()

if __name__ == '__main__':
    with Pool() as pool:
        pool.map(process_video_chunk, video_chunks)
