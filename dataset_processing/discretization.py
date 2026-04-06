"""
discretization.py
Fungsi: Mengonversi skor kepribadian kontinu (0.0 - 1.0) menjadi label kelas 
berdasarkan Batas Kuartil (Equal-Frequency Discretization) sesuai metodologi TA.
Output: Label 0, 1, 2, 3 (4 Kelas: Sangat Rendah, Rendah, Tinggi, Sangat Tinggi).
"""

import pandas as pd
import numpy as np
import os

def run_discretization(file_path='annotation.csv'):
    # 1. BACA FILE
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File {file_path} tidak ditemukan!")
        return

    df = pd.read_csv(file_path)
    traits = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

    print("🔄 Memulai proses diskretisasi kuartil (4 kelas)...")

    # 2. FUNGSI DISKRETISASI (KUARTIL)
    def discretize(labels):
        # Menggunakan bins [0, 0.25, 0.5, 0.75, 1.0] untuk membagi 4 bagian (Kuartil)
        bins = np.array([0, 0.25, 0.5, 0.75, 1.0])
        
        # Hitung nilai ambang batas berdasarkan distribusi data asli
        quantiles = np.quantile(labels, bins) 
        
        # Petakan label ke dalam bin (0, 1, 2, 3)
        # right=True memastikan nilai batas masuk ke bin sebelah kiri (sesuai standar TA)
        new_labels = np.digitize(labels, quantiles, right=True) - 1
        
        # Pastikan tidak ada label di luar rentang 0-3 (menangani edge case minimum)
        new_labels = np.clip(new_labels, 0, 3)
        return new_labels

    # 3. APLIKASIKAN PADA SETIAP TRAIT
    for trait in traits:
        label_data = df[trait].to_numpy(dtype=np.float32)
        df[trait] = discretize(label_data)

    # 4. SIMPAN KEMBALI
    df.to_csv(file_path, index=False)
    print(f"✅ SUCCESS: Diskretisasi selesai! File {file_path} diperbarui dengan label 0-3.")
    
    # Tampilkan distribusi singkat untuk verifikasi
    print("\n📊 Distribusi Kelas per Dimensi (Harusnya seimbang):")
    print(df[traits].apply(pd.Series.value_counts))

if __name__ == '__main__':
    run_discretization()
