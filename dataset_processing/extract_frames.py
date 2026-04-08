import pandas as pd
import cv2
import os
from multiprocessing import Pool

# Setup Config
video_root = '/content/drive/MyDrive/Dataset_TA/'
phases = ['training', 'validation', 'test']

output_folder = '/content/frames/'
os.makedirs(output_folder, exist_ok=True)

# Mendapatkan daftar video yang akan diproses
df = pd.read_csv('annotation.csv')
videos = df['video_name'].tolist()

# Keperluan Testing Pipeline, jangan lupa dihapus
videos = videos[:50]  # Hanya proses 50 video pertama untuk testing cepat

# Mencegah error jika jumlah video lebih sedikit dari jumlah CPU
cpu_count = os.cpu_count() or 1
chunk_size = max(1, len(videos) // cpu_count)
video_chunks = [videos[i:i + chunk_size] for i in range(0, len(videos), chunk_size)]


# Main function untuk memproses chunk video
def process_video_chunk(video_chunk):
    for video_file in video_chunk:
        try:
            # Memastikan nama video PUNYA ekstensi .mp4
            if not str(video_file).endswith('.mp4'):
                video_file = str(video_file) + '.mp4'

            # Cari video di semua phase
            video_path = None
            for phase in phases:
                temp_path = os.path.join(video_root, phase, video_file)
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break

            # Kalau tidak ditemukan
            if video_path is None:
                print(f"⚠️ Video tidak ditemukan: {video_file}")
                continue

            # Buka video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️ Gagal membuka video: {video_file}")
                continue

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 25  # fallback aman

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Ambil 1 frame per detik
                if frame_count % fps == 0:
                    frame_filename = os.path.join(
                        output_folder,
                        f"{os.path.splitext(video_file)[0]}_frame{frame_count}.jpg"
                    )
                    cv2.imwrite(frame_filename, frame)

                frame_count += 1

            cap.release()

        except Exception as e:
            print(f"Error pada video {video_file}: {e}") # debugging statement


# Run multiprocessing untuk ekstraksi frame
if __name__ == '__main__':
    print(f"🚀 Memulai ekstraksi {len(videos)} video menggunakan {cpu_count} CPU core...")
    print("⏳ Proses ini bisa memakan waktu cukup lama...")

    with Pool(processes=cpu_count) as pool:
        pool.map(process_video_chunk, video_chunks)

    print("SUCCESS: Semua frame berhasil diekstrak ke folder /content/frames/") # debugging statement