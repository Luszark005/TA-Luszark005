import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2  # Wajib pakai v2 untuk Multi-Frame/Video

class FirstImpressionsVideoDataset(Dataset):
    def __init__(self, phase, data_path, transform=None, num_frames=8):
        """
        Dataset Multi-Frame untuk arsitektur Swin Transformer.
        """
        self.phase = phase
        self.data_path = data_path
        self.transform = transform
        self.num_frames = num_frames

        # 1. Baca anotasi utama
        df = pd.read_csv(os.path.join(data_path, 'annotation.csv'))
        self.data = df[df['phase'] == phase]

        # Ambil nama video (tanpa ekstensi .mp4)
        self.video_names = self.data['video_name'].apply(lambda x: os.path.splitext(x)[0]).tolist()

        # Load label Big Five
        self.labels = self.data[['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']].values

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_basename = self.video_names[idx]
        
        # --- A. LOAD N FRAMES (GAMBAR) ---
        frames_dir = os.path.join(self.data_path, 'images', video_basename)
        frames = []
        
        for i in range(self.num_frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:02d}.jpg")
            try:
                img = Image.open(frame_path).convert('RGB')
            except FileNotFoundError:
                # Fallback: Jika gambar hilang, buat gambar hitam kosong agar batch tidak error
                img = Image.new('RGB', (224, 224), (0, 0, 0))
            frames.append(img)

        # Ubah list gambar PIL menjadi satu Tensor Video [T, C, H, W]
        # T = Time/Frames, C = Channels, H = Height, W = Width
        frames_tensor = torch.stack([v2.functional.pil_to_tensor(img) for img in frames])

        # --- B. LOAD N EMOTIONS (.npy) ---
        emotion_path = os.path.join(self.data_path, 'emotions', f"{video_basename}.npy")
        try:
            emotions = np.load(emotion_path)  # Bentuk: [N_frames, 7]
        except FileNotFoundError:
            # Fallback: Jika emosi hilang, tembak Netral (indeks ke-6)
            emotions = np.zeros((self.num_frames, 7), dtype=np.float32)
            emotions[:, 6] = 1.0 
            
        emotions_tensor = torch.tensor(emotions, dtype=torch.float32)
        labels_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        # --- C. AUGMENTASI KONSISTEN ---
        # Transformasi v2 otomatis menerapkan efek yang sama ke dimensi T (Frames)
        if self.transform is not None:
            frames_tensor = self.transform(frames_tensor)

        return frames_tensor, labels_tensor, emotions_tensor, idx

def get_dataloader(data_path='', batch_size=4, num_workers=4, num_frames=8):
    """
    Returns the DataLoader for training and validation.
    Perhatikan batch_size default diturunkan menjadi 4 untuk mencegah GPU Out-of-Memory.
    """
    # Transformasi untuk Training (Dengan Augmentasi)
    train_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5), # 50% peluang video di-mirror
        v2.RandomRotation(degrees=10),  # Putar maksimal 10 derajat
        v2.ColorJitter(brightness=0.2, contrast=0.2), # Ubah pencahayaan
        v2.ToDtype(torch.float32, scale=True), # Wajib: Ubah dari 0-255 ke 0.0-1.0
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transformasi untuk Validation (Tanpa Augmentasi, murni asli)
    val_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Inisialisasi Dataset
    train_dataset = FirstImpressionsVideoDataset(
        phase='train', data_path=data_path, transform=train_transform, num_frames=num_frames
    )

    val_dataset = FirstImpressionsVideoDataset(
        phase='validation', data_path=data_path, transform=val_transform, num_frames=num_frames
    )

    # Inisialisasi DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True,       # BUG FIXED: Sebelumnya False, sekarang True agar data tidak dihafal
        drop_last=True,
        pin_memory=True     # Mempercepat transfer ke GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False,      # Validation wajib False
        pin_memory=True
    )

    return train_loader, val_loader