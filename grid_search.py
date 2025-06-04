import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
from transformers import AutoModel, AutoModelForImageClassification, AutoImageProcessor, AutoConfig
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, get_cosine_schedule_with_warmup
import torch.nn.functional as F
import torchaudio
import itertools

from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import random
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

NUM_CLASSES = 207
BASE_OUTPUT_DIR = '/home/andy/Desktop/BirdClef/customSED/gridsearch_out'

# Set seed
def seed_all(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class BirdTrainDatasetPrecomputed(Dataset):
    def __init__(self, counts_df, labels_df, data_path='data/precomputed_spectrograms/spectrograms', use_cutmix=True, use_masking=True, num_classes = 206, sample_random_ms = False):
        self.path = data_path
        self.use_cutmix = use_cutmix
        self.use_masking = use_masking
        self.num_classes = num_classes
        self.sample_random_ms = sample_random_ms
        self.labels_df_indexed = labels_df.set_index('file_path')
        self.labels_df = labels_df
        self.counts_df = counts_df

    def __len__(self):
        if self.sample_random_ms:
            return len(self.counts_df)
        return len(self.labels_df) 

    def __getitem__(self, idx):
        path, label = self.get_path_and_label(idx)
        spec = torch.load(path)

        if self.use_cutmix and random.random() < 0.5:
            mix_path, mix_label = self.get_path_and_label(-1)
            mix_spec = torch.load(mix_path)

            if self.use_masking:
                spec = self.xy_masking(spec)
                mix_spec = self.xy_masking(mix_spec)

            spec, label = self.horizontal_cutmix(spec, label, mix_spec, mix_label)

        else:
            if self.use_masking:
                spec = self.xy_masking(spec)
            label = F.one_hot(torch.tensor(label), self.num_classes).float()

        return {
            "pixel_values": spec,
            "labels": label,
            "file_name": str(path),
        }

    def get_path_and_label(self, idx = -1):
        if idx == -1:
            idx = random.randint(0, self.__len__() - 1)
        
        if self.sample_random_ms:
            dir_path = Path(self.counts_df.iloc[idx]['file_path'])
            count = self.counts_df.iloc[idx]['count']
            filename = random.randint(0, count - 1)
            path = dir_path / f"{filename}.pt"
            label = self.labels_df_indexed.loc[str(path)]['label']
            return path, label
        else:
            return self.labels_df.iloc[idx]['file_path'], self.labels_df.iloc[idx]['label']

    def xy_masking(self, spec, num_x_masks=2, num_y_masks=1, max_width=10, max_height=10):
        """
        Applies vertical (x) and horizontal (y) rectangular zero-masks to the spectrogram.
        """
        cloned = spec.clone()
        _, height, width = cloned.shape

        # Apply x-masks (vertical)
        for _ in range(num_x_masks):
            w = random.randint(1, max_width)
            x = random.randint(0, max(0, width - w))
            cloned[:, :, x:x+w] = 0.0

        # Apply y-masks (horizontal)
        for _ in range(num_y_masks):
            h = random.randint(1, max_height)
            y = random.randint(0, max(0, height - h))
            cloned[:, y:y+h, :] = 0.0

        return cloned

    def horizontal_cutmix(self, spec1, label1, spec2, label2, alpha=1.0):
        """
        Mix two spectrograms horizontally (along the time axis),
        and create soft labels using torch.nn.functional.one_hot.
        """
        _, h, w = spec1.shape
        cut_point = random.randint(int(0.3 * w), int(0.7 * w))
        lam = cut_point / w

        # Concatenate spectrograms along the time axis (width)
        new_spec = torch.cat((spec1[:, :, :cut_point], spec2[:, :, cut_point:]), dim=2)

        # Convert scalar labels to one-hot vectors
        label1_onehot = F.one_hot(torch.tensor(label1), num_classes=self.num_classes).float()
        label2_onehot = F.one_hot(torch.tensor(label2), num_classes=self.num_classes).float()

        # Mix the labels
        mixed_label = lam * label1_onehot + (1 - lam) * label2_onehot

        return new_spec, mixed_label

  
def get_datasets(counts_df, labels_df, sample_random = False, use_cutmix = False, use_masking = False):
    """
    divides by original files instead of snippet files to avoid data leakage
    """

    train_counts_df, train_val_df = train_test_split(
      counts_df, 
      test_size=0.2, 
      random_state=42, 
      stratify=counts_df['label']
    )

    # Filter labels_df using file_num
    train_labels_df = labels_df[labels_df['file_num'].isin(train_counts_df['file_num'])].copy()
    val_labels_df   = labels_df[labels_df['file_num'].isin(train_val_df['file_num'])].copy()
    
    # make the datasets
    train_ds = BirdTrainDatasetPrecomputed(train_counts_df, train_labels_df, num_classes=NUM_CLASSES, sample_random_ms=sample_random, use_cutmix=use_cutmix, use_masking=use_masking)
    val_ds = BirdTrainDatasetPrecomputed(train_val_df, val_labels_df, num_classes=NUM_CLASSES, sample_random_ms=sample_random, use_cutmix=False, use_masking=False)
    
    return train_ds, val_ds

# Define metrics computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    # Defensive check: if labels are soft (i.e., one-hot), convert them
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1
    }

class SoftLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # soft labels (probabilities)
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (batch_size, num_classes)

        # Use log_softmax and soft cross entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(labels * log_probs).sum(dim=-1).mean()

        return (loss, outputs) if return_outputs else loss

def train_model(train_counts_df, train_labels_df, val_counts_df, val_labels_df, model_name, output_dir, lr, dropout, batch_size, wd, rs):
    print(f"Training {model_name} | LR: {lr}, Dropout: {dropout}, Batch: {batch_size}, Weight Decay: {wd}, Random Sample: {rs}")

    ds_train = BirdTrainDatasetPrecomputed(train_counts_df, train_labels_df, num_classes=NUM_CLASSES, sample_random_ms=rs, use_cutmix=True, use_masking=True)
    ds_val = BirdTrainDatasetPrecomputed(val_counts_df, val_labels_df, num_classes=NUM_CLASSES, sample_random_ms=False, use_cutmix=False, use_masking=False)

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = NUM_CLASSES
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout

    model = AutoModelForImageClassification.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=20,
        weight_decay=wd,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
        warmup_steps=500,
        lr_scheduler_type="linear",
        disable_tqdm=True
    )

    trainer = SoftLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()


# set seed
seed_all(42)

# datasets
counts_df = pd.read_csv("/home/andy/Desktop/BirdClef/customSED/data/precomputed_spectrograms/counts.csv")
labels_df = pd.read_csv("/home/andy/Desktop/BirdClef/customSED/data/precomputed_spectrograms/labels.csv")

train_counts_df, val_counts_df = train_test_split(
  counts_df, 
  test_size=0.2, 
  random_state=42, 
  stratify=counts_df['label']
)

# Filter labels_df using file_num
train_labels_df = labels_df[labels_df['file_num'].isin(train_counts_df['file_num'])].copy()
val_labels_df   = labels_df[labels_df['file_num'].isin(val_counts_df['file_num'])].copy()

# --- Grid Search Setup ---
model_names = ["google/efficientnet-b2", "facebook/regnet-y-008"]
lrs = [1e-4, 5e-4]
dropouts = [0.2, 0.3]
batch_sizes = [32, 64]
weight_decays = [1e-4, 5e-5]
random_samples = [True, False]

# gen all combinations then shuffle
all_combinations = list(itertools.product(model_names, lrs, dropouts, batch_sizes, weight_decays, random_samples))
random.shuffle(all_combinations)
print(all_combinations)

start_seed = 100

for i, (model_name, lr, dropout, batch_size, weight_decay, random_sample) in enumerate(all_combinations[:100]):
    try:
      out_dir = os.path.join(
          BASE_OUTPUT_DIR,
          f"{model_name.split('/')[-1]}_lr{lr}_drop{dropout}_bs{batch_size}_wd{weight_decay}_rs{random_sample}"
      )
      os.makedirs(out_dir, exist_ok=True)
      train_model(train_counts_df, train_labels_df, val_counts_df, val_labels_df, model_name, out_dir, lr, dropout, batch_size, wd=weight_decay, rs = random_sample)
    except Exception as e:
        print(f"Error training {model_name} with LR: {lr}, Dropout: {dropout}, Batch: {batch_size}, Seed: {start_seed}, Weight Decay: {weight_decay}, Random Sample: {random_sample} - {e}")