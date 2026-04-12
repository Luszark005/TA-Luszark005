import copy
import timm
import torch
import torch.nn as nn

def create_model(num_classes=4, drop_rate=0.2, emotion=True):
    """
    Factory function untuk membuat model.
    Default num_classes=4 (Sangat Rendah, Rendah, Tinggi, Sangat Tinggi).
    """
    return SwinMultiFrameEmotion(num_classes=num_classes, drop_rate=drop_rate)

class SwinMultiFrameEmotion(nn.Module):
    def __init__(self, num_classes=4, drop_rate=0.2):
        super(SwinMultiFrameEmotion, self).__init__()
        self.drop_rate = drop_rate

        # BACKBONE: SWIN TRANSFORMER (Visual Feature Extractor)
        # Menggunakan swin_tiny_patch4_window7_224. 
        # num_classes=0 agar bertindak sebagai feature extractor (output 768-dim)
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        swin_out_dim = self.backbone.num_features  # 768 untuk Swin-Tiny

        # EMOTION BRANCH (Integrasi Fitur Emosi)
        # Input: 7-dimensi vektor emosi dari DeepFace
        self.emotion_feature = nn.Sequential(
            nn.Linear(7, 128), 
            nn.ReLU(), # Bisa masuikin ke buku alasan kenapa pake ReLU
            nn.BatchNorm1d(128)
        )
        
        # DUAL-BRANCH MLP (Untuk Ada-DF)
        # Membuat dua jalur fitur mandiri sebelum klasifikasi
        branch_dim = 512
        self.branch_1_feature = nn.Sequential(
            nn.Linear(swin_out_dim, branch_dim),
            nn.GELU(), # Bisa masuikin ke buku alasan kenapa pake GELU
            nn.Dropout(self.drop_rate)
        )
        self.branch_2_feature = copy.deepcopy(self.branch_1_feature)

        # MULTI-OUTPUT CLASSIFIERS (Big Five OCEAN)
        # Gabungan visual (512) + emosi (128) = 640 dimensi
        fused_dim = branch_dim + 128 

        # Cabang 1 (Auxiliary/Tambahan) & Cabang 2 (Target/Prediksi Akhir)
        self.branch_1_classifiers = nn.ModuleList(nn.Linear(fused_dim, num_classes) for _ in range(5))
        self.branch_2_classifiers = nn.ModuleList(nn.Linear(fused_dim, num_classes) for _ in range(5))
        
        # Modul Perhatian (Attention Weight) untuk Adaptive Fusion
        self.alphas_1 = nn.ModuleList(nn.Sequential(nn.Linear(fused_dim, 1), nn.Sigmoid()) for _ in range(5))
        self.alphas_2 = nn.ModuleList(nn.Sequential(nn.Linear(fused_dim, 1), nn.Sigmoid()) for _ in range(5))

    def forward(self, frames, emotions):
        """
        Input:
          frames: [Batch, 16, 3, 224, 224]
          emotions: [Batch, 16, 7]
        """
        B, F, C, H, W = frames.shape

        # TEMPORAL PROCESSING (Mean Pooling)
        # Flatten Batch & Frames agar bisa masuk ke Swin 2D
        frames_reshaped = frames.view(B * F, C, H, W)
        
        # Extract Fitur Visual
        visual_features = self.backbone(frames_reshaped) # [B*F, 768]
        visual_features = visual_features.view(B, F, -1) # Kembali ke [B, 16, 768]

        # Temporal Aggregation (Mean Pooling)
        # Menggabungkan informasi dari 16 frame menjadi satu representasi video
        video_visual = torch.mean(visual_features, dim=1) # [B, 768]
        video_emotions = torch.mean(emotions, dim=1)      # [B, 7]

        # FEATURE ENHANCEMENT
        # Jalankan cabang emosi
        emotion_features = self.emotion_feature(video_emotions) # [B, 128]

        # Jalankan cabang Dual-Branch
        feat_1 = self.branch_1_feature(video_visual) # [B, 512]
        feat_2 = self.branch_2_feature(video_visual) # [B, 512]

        # ADAPTIVE DISTRIBUTION FUSION
        # Gabungkan visual dan emosi
        fused_1 = torch.cat([feat_1, emotion_features], dim=1) # [B, 640]
        fused_2 = torch.cat([feat_2, emotion_features], dim=1) # [B, 640]

        out_aux = []    # Hasil dari Cabang Tambahan
        out_target = [] # Hasil dari Cabang Target (Prediksi Utama)
        att_weights = []

        for i in range(5):
            # Hitung Bobot Atensi (w_aux dan w_tar)
            w1 = self.alphas_1[i](fused_1)
            w2 = self.alphas_2[i](fused_2)

            # Klasifikasi dengan pembobotan adaptif
            res_1 = w1 * self.branch_1_classifiers[i](fused_1)
            res_2 = w2 * self.branch_2_classifiers[i](fused_2)

            out_aux.append(res_1)
            out_target.append(res_2)
            
            # Rata-rata bobat atensi (w_avg) untuk perhitungan loss
            att_weights.append((w1 + w2) / 2)
        
        return out_aux, out_target, att_weights
