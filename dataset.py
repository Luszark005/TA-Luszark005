import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 

class FirstImpressionsVideoDataset(Dataset):
    def __init__(self, phase, data_path, transform=None, num_frames=8):
        self.phase = phase
        self.data_path = data_path
        self.transform = transform
        self.num_frames = num_frames

        # Mastiin file anotasi tersedia di folder dataset
        # Sesuaikan aja kalau pakai Google Drive atau path lokal
        csv_path = os.path.join(data_path, 'annotation.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Annotation file not found at {csv_path}")
            
        df = pd.read_csv(csv_path)
        # Filter data berdasarkan fase (train/validation/test)
        self.data = df[df['phase'] == phase]

        self.video_names = self.data['video_name'].apply(lambda x: os.path.splitext(x)[0]).tolist()
        self.labels = self.data[['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']].values

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_basename = self.video_names[idx]
        # Folder images sekarang merujuk ke folder 'images' hasil penyelarasan
        frames_dir = os.path.join(self.data_path, 'images', video_basename)
        frames = []
        
        # Loaad Frames (linear index)
        # PERBAIKAN: Tidak perlu lagi menggunakan stride karena file sudah 
        # dinamai berurutan frame_00.jpg s/d frame_07.jpg oleh extract_emotions.py
        for i in range(self.num_frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:02d}.jpg")
            
            try:
                img = Image.open(frame_path).convert('RGB')
                # Resize tetap dilakukan untuk memastikan kompatibilitas tensor
                img = img.resize((224, 224)) 
            except FileNotFoundError:
                # Fallback jika gambar hilang: Hitam polos agar tidak crash saat training
                img = Image.new('RGB', (224, 224), (0, 0, 0))
            frames.append(img)

        # Mengubah list gambar menjadi tensor video [Batch x 3 x 224 x 224]
        frames_tensor = torch.stack([v2.functional.pil_to_tensor(img) for img in frames])

        # LOAD EMOTIONS (.npy) 
        # File .npy sekarang berisi urutan 8 emosi yang sudah sinkron dengan gambarnya
        emotion_path = os.path.join(self.data_path, 'emotions', f"{video_basename}.npy")
        try:
            emotions = np.load(emotion_path) 
            
            # Jika file .npy memiliki dimensi yang berbeda, lakukan penyesuaian otomatis
            if emotions.shape[0] > self.num_frames:
                emotions = emotions[:self.num_frames]
            elif emotions.shape[0] < self.num_frames:
                padding = np.tile(emotions[-1], (self.num_frames - emotions.shape[0], 1))
                emotions = np.vstack([emotions, padding])
        except FileNotFoundError:
            # Fallback emosi netral jika file .npy belum diekstrak
            emotions = np.zeros((self.num_frames, 7), dtype=np.float32)
            emotions[:, 6] = 1.0 
            
        emotions_tensor = torch.tensor(emotions, dtype=torch.float32)
        labels_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        # AUGMENTASI
        # Menggunakan transformasi torchvision v2 untuk efisiensi
        if self.transform is not None:
            frames_tensor = self.transform(frames_tensor)

        return frames_tensor, labels_tensor, emotions_tensor, idx

def get_dataloader(data_path='', batch_size=4, num_workers=4, num_frames=8):
    """Fungsi pembantu untuk membuat DataLoader train dan validation."""
    
    # Augmentasi gambar khusus untuk training
    train_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FirstImpressionsVideoDataset(phase='train', data_path=data_path, transform=train_transform, num_frames=num_frames)
    val_dataset = FirstImpressionsVideoDataset(phase='validation', data_path=data_path, transform=val_transform, num_frames=num_frames)

    # Menggunakan pin_memory=True untuk mempercepat transfer data ke GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_loader, val_loader