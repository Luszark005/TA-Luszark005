import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2  # Wajib untuk konsistensi Multi-Frame

class FirstImpressionsVideoDataset(Dataset):
    def __init__(self, phase, data_path, transform=None, num_frames=16):
        """
        Dataset Multi-Frame yang disesuaikan untuk Swin Transformer & Ada-DF.
        Target: Klasifikasi 4 Kelas (0-3) untuk 5 Trait Big Five.
        """
        self.phase = phase
        self.data_path = data_path
        self.transform = transform
        self.num_frames = num_frames

        # 1. Baca anotasi (Pastikan sudah dijalankan discretization.py sebelumnya)
        csv_path = os.path.join(data_path, 'annotation.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Annotation file not found at {csv_path}")
            
        df = pd.read_csv(csv_path)
        self.data = df[df['phase'] == phase]

        # Ambil nama video dan label Big Five (0-3)
        self.video_names = self.data['video_name'].apply(lambda x: os.path.splitext(x)[0]).tolist()
        self.labels = self.data[['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']].values

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_basename = self.video_names[idx]
        
        # --- A. LOAD 16 FRAMES ---
        frames_dir = os.path.join(self.data_path, 'images', video_basename)
        frames = []
        
        for i in range(self.num_frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:02d}.jpg")
            try:
                img = Image.open(frame_path).convert('RGB')
                # ✨ TAMBAHKAN BARIS INI: Paksa resize sebelum masuk ke list
                img = img.resize((224, 224)) 
            except FileNotFoundError:
                img = Image.new('RGB', (224, 224), (0, 0, 0))
            frames.append(img)

        # Sekarang torch.stack tidak akan protes karena semua ukurannya sudah (3, 224, 224)
        frames_tensor = torch.stack([v2.functional.pil_to_tensor(img) for img in frames])

        # --- B. LOAD EMOTIONS (.npy) ---
        emotion_path = os.path.join(self.data_path, 'emotions', f"{video_basename}.npy")
        try:
            emotions = np.load(emotion_path)  # Ekspektasi: [16, 7]
            # Jika jumlah frame di .npy beda, lakukan padding/truncating
            if emotions.shape[0] < self.num_frames:
                padding = np.tile(emotions[-1], (self.num_frames - emotions.shape[0], 1))
                emotions = np.vstack([emotions, padding])
            else:
                emotions = emotions[:self.num_frames]
        except FileNotFoundError:
            # Fallback: Emosi Netral (indeks ke-6) jika file .npy belum ada
            emotions = np.zeros((self.num_frames, 7), dtype=np.float32)
            emotions[:, 6] = 1.0 
            
        emotions_tensor = torch.tensor(emotions, dtype=torch.float32)
        
        # Label sebagai LongTensor untuk CrossEntropyLoss (Klasifikasi)
        labels_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        # --- C. AUGMENTASI KONSISTEN (V2) ---
        if self.transform is not None:
            # V2 menerapkan transform yang sama untuk semua T frame dalam satu panggil
            frames_tensor = self.transform(frames_tensor)

        return frames_tensor, labels_tensor, emotions_tensor, idx

def get_dataloader(data_path='', batch_size=4, num_workers=4, num_frames=16):
    """
    Menghasilkan DataLoader untuk Train dan Validation.
    Batch size 4 direkomendasikan untuk menghindari OOM saat menggunakan 16 Frame + Swin.
    """
    # Transformasi Training: Augmentasi Spasial identik per sequence
    train_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transformasi Validation: Tanpa augmentasi random
    val_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FirstImpressionsVideoDataset(
        phase='train', data_path=data_path, transform=train_transform, num_frames=num_frames
    )

    val_dataset = FirstImpressionsVideoDataset(
        phase='validation', data_path=data_path, transform=val_transform, num_frames=num_frames
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True, 
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        pin_memory=True
    )

    return train_loader, val_loader
