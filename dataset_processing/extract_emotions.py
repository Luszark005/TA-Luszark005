import os
import pandas as pd
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

# --- KONFIGURASI MULTI-FRAME ---
DATASET_CSV = './dataset/annotation.csv'
IMAGES_DIR = './dataset/images'       # Folder tempat hasil extract_cropped_frames disimpan
EMOTIONS_OUT_DIR = './dataset/emotions' # Folder baru khusus untuk menyimpan file .npy emosi

os.makedirs(EMOTIONS_OUT_DIR, exist_ok=True)

def extract_emotion_from_frame(image_path):
    try:
        # enforce_detection=False sangat penting agar tidak crash jika wajah agak miring/blur
        result = DeepFace.analyze(img_path=image_path, align=False, enforce_detection=False, actions=['emotion'])
        emotion_probs = result[0]['emotion']

        # Normalisasi ke 0.0 - 1.0 sesuai kode asli Aryan
        emotion_vector = np.array([
            emotion_probs['angry'],
            emotion_probs['disgust'],
            emotion_probs['fear'],
            emotion_probs['happy'],
            emotion_probs['sad'],
            emotion_probs['surprise'],
            emotion_probs['neutral']
        ]) / 100.0
        return emotion_vector.astype(np.float32)
        
    except Exception as e:
        # FALLBACK: Jika DeepFace gagal memproses gambar karena alasan apapun,
        # kita kembalikan array Netral murni untuk menjaga bentuk tensor tidak rusak.
        # (Marah:0, Jijik:0, Takut:0, Senang:0, Sedih:0, Kaget:0, Netral:1)
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

if __name__ == "__main__":
    df = pd.read_csv(DATASET_CSV)
    video_names = df['video_name'].tolist()

    # Pakai tqdm biar kamu bisa lihat progress bar-nya
    for video_name in tqdm(video_names, desc="Mengekstrak Emosi Multi-Frame"):
        video_basename = os.path.splitext(video_name)[0]
        video_folder = os.path.join(IMAGES_DIR, video_basename)
        
        # Pastikan foldernya ada (hasil ekstraksi sebelumnya)
        if not os.path.exists(video_folder):
            print(f"Folder {video_folder} tidak ditemukan. Dilewati.")
            continue
            
        # Ambil semua gambar frame (frame_00.jpg, frame_01.jpg, dst)
        frame_files = sorted(os.listdir(video_folder))
        
        emotion_sequence = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_folder, frame_file)
            emotion_vector = extract_emotion_from_frame(frame_path)
            emotion_sequence.append(emotion_vector)
            
        # Ubah list menjadi matriks Numpy 2D. 
        # Jika ada 8 frame, ukurannya jadi (8, 7)
        emotion_sequence_np = np.array(emotion_sequence)
        
        # Simpan matriks ke dalam file .npy
        out_filepath = os.path.join(EMOTIONS_OUT_DIR, f"{video_basename}.npy")
        np.save(out_filepath, emotion_sequence_np)
        
    print(f"Selesai! File matriks emosi berhasil disimpan di {EMOTIONS_OUT_DIR}")