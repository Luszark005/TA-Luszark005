"""
Discretisasi label kepribadian menjadi 5 kelas berdasarkan 
kuantil untuk memastikan distribusi kelas yang seimbang, 
sehingga model dapat belajar dengan lebih efektif tanpa bias 
terhadap kelas mayoritas.
"""

import pandas as pd
import numpy as np
# import torch (Kita tidak butuh torch di tahap ini, jadi dihapus saja biar ringan)

# 1. BACA FILE DARI FOLDER YANG SAMA
df = pd.read_csv('annotation.csv')

# Pelabelanan kepribadian dan dikonversi menjadi nilai numerik
label_extraversions = df['extraversion'].to_numpy(dtype=np.float32)
label_neuroticisms = df['neuroticism'].to_numpy(dtype=np.float32)
label_agreeablenesses = df['agreeableness'].to_numpy(dtype=np.float32)
label_conscientiousnesses = df['conscientiousness'].to_numpy(dtype=np.float32)
label_opennesses = df['openness'].to_numpy(dtype=np.float32)

# Discretisasi label kepribadian menjadi 5 kelas berdasarkan kuantil
def discretize(labels):
    bins = np.array([0, 0.25, 0.5, 0.75, 1.0])
    quantiles = np.quantile(labels, bins) 
    # print(quantiles) # Bisa di-comment agar output terminal tidak kepanjangan
    
    # 2. PERBAIKAN TYPO: np.digitize (tanpa spasi)
    labels = np.digitize(labels, quantiles, right=True) - 1
    # print(labels)
    return labels

# Mengaplikasikan discretisasi pada setiap label kepribadian
df['extraversion'] = discretize(label_extraversions)
df['neuroticism'] = discretize(label_neuroticisms)
df['agreeableness'] = discretize(label_agreeablenesses)
df['conscientiousness'] = discretize(label_conscientiousnesses)
df['openness'] = discretize(label_opennesses)

# 3. SIMPAN KEMBALI KE FOLDER YANG SAMA (Menimpa file sebelumnya)
df.to_csv('annotation.csv', index=False)
print("✅ SUCCESS: Discretisasi selesai! File annotation.csv telah diperbarui dengan label kelas 0-4.")