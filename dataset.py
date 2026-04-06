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

        csv_path = os.path.join(data_path, 'annotation.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Annotation file not found at {csv_path}")
            
        df = pd.read_csv(csv_path)
        self.data = df[df['phase'] == phase]

        self.video_names = self.data['video_name'].apply(lambda x: os.path.splitext(x)[0]).tolist()
        self.labels = self.data[['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']].values

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_basename = self.video_names[idx]
        frames_dir = os.path.join(self.data_path, 'images', video_basename)
        frames = []
        
        # --- A. LOAD FRAMES (DENGAN STRIDE) ---
        # Jika folder punya 16 file tapi num_frames=8, kita ambil setiap 2 frame (0, 2, 4...)
        # Kita asumsi folder berisi setidaknya 8 atau 16 frame
        for i in range(self.num_frames):
            stride = 2 if self.num_frames == 8 else 1 
            frame_idx = i * stride
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:02d}.jpg")
            
            try:
                img = Image.open(frame_path).convert('RGB')
                img = img.resize((224, 224)) # ✨ WAJIB: Biar tidak error ukuran tensor
            except FileNotFoundError:
                # Fallback jika gambar hilang: Hitam polos
                img = Image.new('RGB', (224, 224), (0, 0, 0))
            frames.append(img)

        # Ubah list gambar jadi satu tensor besar (Batch x C x H x W)
        frames_tensor = torch.stack([v2.functional.pil_to_tensor(img) for img in frames])

        # --- B. LOAD EMOTIONS (.npy) ---
        emotion_path = os.path.join(self.data_path, 'emotions', f"{video_basename}.npy")
        try:
            emotions = np.load(emotion_path) 
            # Logika fleksibel: Jika .npy isinya 16 tapi diminta 8, kita ambil 8 saja
            if emotions.shape[0] > self.num_frames:
                # Ambil dengan stride yang sama (0, 2, 4...) agar sinkron dengan gambar
                indices = np.arange(0, self.num_frames) * (emotions.shape[0] // self.num_frames)
                emotions = emotions[indices]
            elif emotions.shape[0] < self.num_frames:
                # Jika .npy lebih sedikit, lakukan padding
                padding = np.tile(emotions[-1], (self.num_frames - emotions.shape[0], 1))
                emotions = np.vstack([emotions, padding])
        except FileNotFoundError:
            # Fallback: Emosi Netral jika file belum ada
            emotions = np.zeros((self.num_frames, 7), dtype=np.float32)
            emotions[:, 6] = 1.0 
            
        emotions_tensor = torch.tensor(emotions, dtype=torch.float32)
        labels_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        # --- C. AUGMENTASI ---
        if self.transform is not None:
            frames_tensor = self.transform(frames_tensor)

        return frames_tensor, labels_tensor, emotions_tensor, idx

def get_dataloader(data_path='', batch_size=4, num_workers=4, num_frames=8):
    # (Transformasi tetap sama seperti kodemu sebelumnya)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_loader, val_loader
