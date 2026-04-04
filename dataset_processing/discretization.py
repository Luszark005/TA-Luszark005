"""
Discretisasi label kepribadian menjadi 5 kelas berdasarkan 
kuantil untuk memastikan distribusi kelas yang seimbang, 
sehingga model dapat belajar dengan lebih efektif tanpa bias 
terhadap kelas mayoritas.
"""

import pandas as pd
import os
import numpy as np
import torch

df = pd.read_csv(os.path.join('../dataset', 'annotation.csv'))

# Pelabelanan kepribadian dan dikonversi menjadi nilai numerik
label_extraversions = df['extraversion'].to_numpy(dtype=np.float32)
label_neuroticisms = df['neuroticism'].to_numpy(dtype=np.float32)
label_agreeablenesses = df['agreeableness'].to_numpy(dtype=np.float32)
label_conscientiousnesses = df['conscientiousness'].to_numpy(dtype=np.float32)
label_opennesses = df['openness'].to_numpy(dtype=np.float32)


# Discretisasi label kepribadian menjadi 5 kelas berdasarkan kuantil
def discretize(labels):
    bins = np.array([0, 0.25, 0.5, 0.75, 1.0])
    quantiles = np.quantile(labels, bins) # Menghitung kuantil untuk menentukan batas kelas agar distribusi kelas seimbang
    print(quantiles)
    labels = np.digit ize(labels, quantiles, right=True) - 1
    print(labels)
    return labels

# Mengaplikasikan discretisasi pada setiap label kepribadian dan menyimpan hasilnya kembali ke DataFrame
df['extraversion'] = discretize(label_extraversions)
df['neuroticism'] = discretize(label_neuroticisms)
df['agreeableness'] = discretize(label_agreeablenesses)
df['conscientiousness'] = discretize(label_conscientiousnesses)
df['openness'] = discretize(label_opennesses)

df.to_csv(os.path.join('./dataset', 'annotation.csv'), index=False)