import copy
import timm
import torch
import torch.nn as nn

def create_model(num_classes=4, drop_rate=0.2, emotion=True):
    # Karena TA kamu menggunakan emosi, kita langsung arahkan ke model dengan fitur emosi
    return SwinMultiFrameEmotion(num_classes=num_classes, drop_rate=drop_rate)

class SwinMultiFrameEmotion(nn.Module):
    def __init__(self, num_classes=4, drop_rate=0.2):
        super(SwinMultiFrameEmotion, self).__init__()
        self.drop_rate = drop_rate

        # 1. BACKBONE: SWIN TRANSFORMER
        # Menggunakan versi 'tiny' agar muat di GPU, num_classes=0 agar hanya mengambil fitur (bukan prediksi)
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        swin_out_dim = self.backbone.num_features  # Biasanya 768 untuk Swin-Tiny

        # 2. PEMROSESAN EMOSI (Temporal Aggregation sudah dilakukan di forward)
        self.emotion_feature = nn.Sequential(
            nn.Linear(7, 128), 
            nn.ReLU(), 
            nn.BatchNorm1d(128)
        )
        
        # 3. DUAL-BRANCH FEATURE EXTRACTOR
        # Karena kita tidak bisa memotong layer Transformer seperti ResNet, 
        # kita buat cabang (branch) baru menggunakan Multilayer Perceptron (MLP)
        branch_dim = 512
        self.branch_1_feature = nn.Sequential(
            nn.Linear(swin_out_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(self.drop_rate)
        )
        self.branch_2_feature = copy.deepcopy(self.branch_1_feature)

        # 4. CLASSIFIERS & ATTENTION (Untuk 5 Sifat Big Five)
        fused_dim = branch_dim + 128 # 512 + 128 = 640

        self.branch_1_classifiers = nn.ModuleList(nn.Linear(fused_dim, num_classes) for _ in range(5))
        self.branch_2_classifiers = nn.ModuleList(nn.Linear(fused_dim, num_classes) for _ in range(5))
        
        self.alphas_1 = nn.ModuleList(nn.Sequential(nn.Linear(fused_dim, 1), nn.Sigmoid()) for _ in range(5))
        self.alphas_2 = nn.ModuleList(nn.Sequential(nn.Linear(fused_dim, 1), nn.Sigmoid()) for _ in range(5))

    def forward(self, frames, emotions):
        """
        frames: Tensor berdimensi [Batch, Frames, Channels, Height, Width]
        emotions: Tensor berdimensi [Batch, Frames, 7]
        """
        B, F, C, H, W = frames.shape

        # --- A. TEMPORAL RE-SHAPING UNTUK SWIN ---
        # Swin Transformer standar (2D) tidak mengerti dimensi 'Frames'.
        # Kita lebur Batch dan Frames: [Batch * Frames, C, H, W]
        frames_reshaped = frames.view(B * F, C, H, W)
        
        # Masukkan ke Swin Transformer
        features = self.backbone(frames_reshaped) # Output: [Batch * Frames, 768]
        
        # Kembalikan dimensinya ke bentuk video: [Batch, Frames, 768]
        features = features.view(B, F, -1)

        # --- B. TEMPORAL AGGREGATION (PENGGABUNGAN WAKTU) ---
        # Kita ambil rata-rata fitur dari seluruh frame dalam satu video (Mean Pooling)
        # Output menjadi: [Batch, 768]
        video_features = torch.mean(features, dim=1)

        # Lakukan hal yang sama untuk fitur emosi: [Batch, Frames, 7] -> [Batch, 7]
        video_emotions = torch.mean(emotions, dim=1)
        
        # Masukkan emosi yang sudah dirata-rata ke layer MLP Emosi
        emotion_features = self.emotion_feature(video_emotions) # Output: [Batch, 128]

        # --- C. DUAL-BRANCH FEATURE EXTRACTION ---
        feature_1 = self.branch_1_feature(video_features) # Output: [Batch, 512]
        feature_2 = self.branch_2_feature(video_features) # Output: [Batch, 512]

        # --- D. FUSI (PENGGABUNGAN) VISUAL + EMOSI ---
        feature_fused_1 = torch.cat([feature_1, emotion_features], dim=1) # Output: [Batch, 640]
        feature_fused_2 = torch.cat([feature_2, emotion_features], dim=1) # Output: [Batch, 640]

        outputs_1 = []
        outputs_2 = []
        attention_weights = []

        # --- E. KLASIFIKASI & ADAPTIVE FUSION ---
        for i in range(5):
            attention_weights_1 = self.alphas_1[i](feature_fused_1)
            attention_weights_2 = self.alphas_2[i](feature_fused_2)

            out_1 = attention_weights_1 * self.branch_1_classifiers[i](feature_fused_1)
            out_2 = attention_weights_2 * self.branch_2_classifiers[i](feature_fused_2)

            outputs_1.append(out_1)
            outputs_2.append(out_2)

            attention_weights.append((attention_weights_1 + attention_weights_2) / 2)
        
        return outputs_1, outputs_2, attention_weights