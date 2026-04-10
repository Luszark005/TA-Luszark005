import pandas as pd
import cv2
import os
from multiprocessing import Pool

# =========================
# CONFIG
# =========================
video_root = '/content/drive/MyDrive/Dataset_TA/'
phases = ['training', 'validation', 'test']

output_folder = '/content/frames/'
os.makedirs(output_folder, exist_ok=True)

ANNOTATION_CSV = 'annotation.csv'
DEBUG = False  # ubah ke True kalau mau testing cepat

# =========================
# LOAD VIDEO LIST
# =========================
df = pd.read_csv(ANNOTATION_CSV)
videos = df['video_name'].tolist()

if DEBUG:
    videos = videos[:50]

# =========================
# MULTIPROCESS SETUP
# =========================
cpu_count = os.cpu_count() or 1
chunk_size = max(1, len(videos) // cpu_count)
video_chunks = [videos[i:i + chunk_size] for i in range(0, len(videos), chunk_size)]


# =========================
# MAIN FUNCTION
# =========================
def process_video_chunk(video_chunk):
    for video_file in video_chunk:
        try:
            # Pastikan .mp4
            if not str(video_file).endswith('.mp4'):
                video_file = str(video_file) + '.mp4'

            base_name = os.path.splitext(video_file)[0]

            # Cari path video
            video_path = None
            for phase in phases:
                temp_path = os.path.join(video_root, phase, video_file)
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break

            if video_path is None:
                print(f"⚠️ Video tidak ditemukan: {video_file}")
                continue

            # Buka video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️ Gagal membuka video: {video_file}")
                continue

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # =========================
                # SAVE SEMUA FRAME
                # =========================
                frame_filename = os.path.join(
                    output_folder,
                    f"{base_name}_frame{frame_count:05d}.jpg"
                )

                cv2.imwrite(frame_filename, frame)

                frame_count += 1

            cap.release()

        except Exception as e:
            print(f"❌ Error pada video {video_file}: {e}")


# =========================
# RUN
# =========================
if __name__ == '__main__':
    print(f"🚀 Ekstraksi {len(videos)} video menggunakan {cpu_count} CPU core...")
    print("⏳ Proses bisa lama tergantung jumlah video...")

    with Pool(processes=cpu_count) as pool:
        pool.map(process_video_chunk, video_chunks)

    print("✅ DONE: Semua frame berhasil diekstrak.")