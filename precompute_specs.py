import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
mel_spec_params = {
    "sample_rate": 32000,
    "n_mels": 128,
    "f_min": 20,
    "f_max": 16000,
    "n_fft": 1024,
    "hop_length": 500,
    "normalized": True,
    "center": True,
    "pad_mode": "constant",
    "norm": "slaney",
    "mel_scale": "slaney"
}
top_db = 80
train_duration = 10 * mel_spec_params["sample_rate"]  # 10 seconds
overlap_duration = 2.5 * mel_spec_params["sample_rate"]  # 2.5 seconds overlap
#overlap_duration = 0
step_duration = int(train_duration - overlap_duration)

# Paths
train_csv = 'data/2025/train.csv'
taxonomy_csv = 'data/2025/taxonomy.csv'
output_dir = 'data/precomputed_spectrograms'

# dataset and mel conversion
def normalize_melspec(X, eps=1e-6):
    """Normalize mel spectrogram"""
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)

    norm_min, norm_max = (
        Xstd.min(-1)[0].min(-1)[0],
        Xstd.max(-1)[0].max(-1)[0],
    )
    fix_ind = (norm_max - norm_min) > eps * torch.ones_like(
        (norm_max - norm_min)
    )
    V = torch.zeros_like(Xstd)
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(
            torch.min(V_fix, norm_max_fix),
            norm_min_fix,
        )
        V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
        V[fix_ind] = V_fix
    return V

def read_audio(path):
    """Read audio file (ogg format) and resample"""
    # Using torchaudio to read ogg files
    wav, org_sr = torchaudio.load(path, normalize=True)
    wav = torchaudio.functional.resample(
        wav, orig_freq=org_sr, new_freq=mel_spec_params["sample_rate"]
    )        
    return wav

def crop_audio(wav, start, duration_):
    """
    Crop or pad audio to specific duration starting from 'start'.
    
    Args:
        wav (Tensor): 2D tensor of shape (channels, samples)
        start (int): Starting point in samples
        duration_ (int): Number of samples in the output
    """
    total_length = wav.size(-1)
    
    # If start is beyond the end, start at 0 (or you could raise an error)
    if start >= total_length:
        raise ValueError("Start index is beyond the length of the audio signal.")

    # Calculate end index
    end = start + duration_

    # Extract segment
    segment = wav[:, start:end]

    # If the segment is shorter than the desired duration, repeat it
    if segment.size(-1) < duration_:
        num_repeats = int(np.ceil(duration_ / segment.size(-1)))
        segment = torch.cat([segment] * num_repeats, dim=1)

    # Ensure exact duration
    segment = segment[:, :duration_]
    return segment

def extract_relative_audio_path(audio_path):
  path_no_ext, _ = os.path.splitext(audio_path)
  parts = path_no_ext.split(os.sep)
  return os.path.join(parts[-2], parts[-1])

ds_df = pd.read_csv(train_csv)
taxonomy_df = pd.read_csv(taxonomy_csv)

# Map species id to label
taxonomy = sorted(set(taxonomy_df['primary_label'].values))
taxonomy = {species: i for i, species in enumerate(taxonomy)}
num_classes = len(taxonomy)

# Create mel spectrogram transformation objects
mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spec_params)
db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)

# Output
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'spectrograms'), exist_ok=True)
output = []
file_counts = []


# Loop through each row in the DataFrame
for index, row in tqdm(ds_df.iterrows(), total=len(ds_df)):
    # Get the audio file path and label
    audio_path = os.path.join('data/2025/train_audio', row['filename'])
    relative_path = extract_relative_audio_path(audio_path)
    labels = row['primary_label'].split(' ')
    
    # Read the audio file
    wav = read_audio(audio_path)
    
    start = 0
    count = 0
    while start < wav.size(1):
        # Crop the audio
        segment = crop_audio(wav, start, train_duration)
        
        # Create mel spectrogram
        mel_spectrogram = mel_transform(segment)
        mel_spectrogram = db_transform(mel_spectrogram)
        mel_spectrogram = normalize_melspec(mel_spectrogram)

        # Scale to 0-255 range for image-like processing
        mel_spectrogram = mel_spectrogram * 255
        
        # Convert to 3-channel image format (RGB)
        mel_spectrogram = mel_spectrogram.expand(3, -1, -1)
        
        # Save the spectrogram
        spectrogram_path = os.path.join(output_dir, 'spectrograms', f"{relative_path}/{count}.pt")
        os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)
        torch.save(mel_spectrogram, spectrogram_path)
        
        # update the labels
        primary_label = ds_df.iloc[index]['primary_label']
        label = taxonomy[primary_label]
        row = {
            'file_path': spectrogram_path,
            'label': label,
            'file_num': index
        }
        output.append(row)
        start += step_duration
        count+=1
    
    # add to count
    count_path = os.path.join(output_dir, 'spectrograms', f"{relative_path}")
    count_row = {
        'file_path': count_path,
        'count': count,
        'label': label
    }
    file_counts.append(count_row)

# Save the file labels and index 2 labels to a CSV file
labels_df = pd.DataFrame(output)
labels_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)

counts_df = pd.DataFrame(file_counts)
counts_df.to_csv(os.path.join(output_dir, 'counts.csv'), index=False)
