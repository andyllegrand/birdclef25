import pandas as pd
import numpy as np
import torch
import torchaudio
import os

csv_path = os.path.join('data', '2025', 'train.csv')
df = pd.read_csv(csv_path)

df = df.drop_duplicates(subset='filename')

def compute_power(filepath):
    try:
        wav, _ = torchaudio.load(filepath, normalize=True)
        power = torch.mean(wav ** 2).item()
        return power
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.nan

# Make full path to each audio file
audio_dir = os.path.join('data', '2025', 'train_audio')
df['filepath'] = df['filename'].apply(lambda x: os.path.join(audio_dir, x))

print("Computing power for each file (this may take a while)...")
df['power'] = df['filepath'].apply(compute_power)

# Drop rows where power returned NaN
df = df.dropna(subset=['power'])

# Remove the top 20% of files by power
threshold = np.percentile(df['power'], 80)
df_filtered = df[df['power'] <= threshold]

# Save cleaned CSV
output_path = os.path.join('data', '2025', 'train_cleaned.csv')
df_filtered.drop(columns=['filepath', 'power'], inplace=True)
df_filtered.to_csv(output_path, index=False)

print(f"Saved cleaned CSV as '{output_path}'.")
