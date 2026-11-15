import os
import time
import math
import random, json, copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

import torchvision
from torchvision.transforms import functional as tvF

#Transformers
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer

from tqdm import tqdm

# Reproducibility
SEED = 1090
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic runs are slower; enable if you need exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("device:", DEVICE)

def print_counts_subset(ds, ds_name):

    print(f"Total number of images in {ds_name} dataset is: {len(ds)}, consisting of...")

    base_ds = ds.dataset
    targets = [base_ds.targets[i] for i in ds.indices]

    for name, i in base_ds.class_to_idx.items():
        count = sum(1 for t in targets if t == i)
        print(f'\t* {name}: {count}')
    print("")

class SpecAugmentBands:
    """
    Apply simple spectrogram-specific augmentation:
    - frequency masking: masks horizontal bands
    - time masking: masks vertical bands

    Expects a tensor of shape (C, H, W).
    """
    def __init__(
        self,
        freq_mask_param=16,
        time_mask_param=24,
        num_freq_masks=2,
        num_time_masks=2,
        p=0.5
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x

        _, H, W = x.shape

        # Frequency masking: mask along height
        for _ in range(self.num_freq_masks):
            mask_h = random.randint(0, self.freq_mask_param)
            if mask_h > 0 and mask_h < H:
                top = random.randint(0, H - mask_h)
                x[:, top:top + mask_h, :] = 0

        # Time masking: mask along width
        for _ in range(self.num_time_masks):
            mask_w = random.randint(0, self.time_mask_param)
            if mask_w > 0 and mask_w < W:
                left = random.randint(0, W - mask_w)
                x[:, :, left:left + mask_w] = 0

        return x


vit_transform_val = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

vit_transform_train = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.02), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    SpecAugmentBands(
        freq_mask_param=16,
        time_mask_param=24,
        num_freq_masks=2,
        num_time_masks=2,
        p=0.6
    ),
    #transforms.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

class RandomVerticalChunkDataset(Dataset):
    """
    For training:
    - takes a Subset of an ImageFolder
    - returns one random vertical chunk from each original image
    - then applies the transform
    """
    def __init__(self, subset, transform=None, chunk_width=384):
        self.subset = subset
        self.transform = transform
        self.chunk_width = chunk_width

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]   # image is still PIL because base ImageFolder has no transform
        width, height = image.size

        crop_width = min(self.chunk_width, width)

        if width > crop_width:
            left = torch.randint(0, width - crop_width + 1, (1,)).item()
        else:
            left = 0

        image = image.crop((left, 0, left + crop_width, height))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class MultiVerticalChunkDataset(Dataset):
    """
    For validation:
    - takes a Subset of an ImageFolder
    - returns multiple deterministic vertical chunks from each original image
    - output shape for image is (num_chunks, C, H, W)
    """
    def __init__(self, subset, transform=None, chunk_width=384, num_chunks=4):
        self.subset = subset
        self.transform = transform
        self.chunk_width = chunk_width
        self.num_chunks = num_chunks

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        width, height = image.size

        crop_width = min(self.chunk_width, width)

        if width <= crop_width or self.num_chunks == 1:
            left_positions = [0]
        else:
            max_left = width - crop_width
            left_positions = torch.linspace(0, max_left, steps=self.num_chunks).round().int().tolist()

        chunks = []
        for left in left_positions:
            chunk = image.crop((left, 0, left + crop_width, height))
            if self.transform is not None:
                chunk = self.transform(chunk)
            chunks.append(chunk)

        chunks = torch.stack(chunks, dim=0)   # (num_chunks, C, H, W)
        return chunks, label


def compute_metrics(pred):
    preds = pred.predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

if __name__ == '__main__':
    set_seed(SEED)
    
    # merged_ds = datasets.ImageFolder('dataset_merged', transforms.ToTensor()) #Uncomment for basic transform
    merged_ds = datasets.ImageFolder('dataset_merged') #Uncomment for resnet transform
    
    #train-val split
    seed = torch.Generator().manual_seed(109)
    train_size = int(0.9 * len(merged_ds))
    val_size = len(merged_ds) - train_size

    labels = merged_ds.classes
    label2id = merged_ds.class_to_idx
    id2label = {idx: label for label, idx in label2id.items()}
    
    train_subset, val_subset = random_split(merged_ds, [train_size, val_size], generator=seed)

    train_dataset = RandomVerticalChunkDataset(
    train_subset,
    transform=vit_transform_train,
    chunk_width=700
    )
    
    val_dataset = MultiVerticalChunkDataset(
        val_subset,
        transform=vit_transform_val,
        chunk_width=700,
        num_chunks=5
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    
    model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-384",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
    ).to(DEVICE)

    save_dir = 'models/vit'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    patience = 5
    epochs_without_improvement = 0
    epoch = 0

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    training_hist = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    while epochs_without_improvement < patience:
        epoch += 1

        # -----------------
        # Training
        # -----------------
        model.train()

        running_train_loss = 0.0
        running_train_correct = 0
        running_train_total = 0

        for inputs, target_labels in train_loader:
            inputs = inputs.to(DEVICE)
            target_labels = target_labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, target_labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            running_train_correct += (preds == target_labels).sum().item()
            running_train_total += target_labels.size(0)

        epoch_train_loss = running_train_loss / running_train_total
        epoch_train_acc = running_train_correct / running_train_total

        # -----------------
        # Validation
        # -----------------
        model.eval()

        running_val_loss = 0.0
        running_val_correct = 0
        running_val_total = 0

        with torch.no_grad():
            for inputs, target_labels in val_loader:
                # inputs: (B, N, C, H, W)
                inputs = inputs.to(DEVICE)
                target_labels = target_labels.to(DEVICE)

                batch_size, num_chunks, C, H, W = inputs.shape

                inputs = inputs.view(batch_size * num_chunks, C, H, W)

                outputs = model(pixel_values=inputs).logits
                # outputs: (B * N, num_classes)

                outputs = outputs.view(batch_size, num_chunks, -1)
                outputs = outputs.mean(dim=1)
                # outputs: (B, num_classes)

                loss = criterion(outputs, target_labels)

                running_val_loss += loss.item() * batch_size

                preds = outputs.argmax(dim=1)
                running_val_correct += (preds == target_labels).sum().item()
                running_val_total += batch_size

        epoch_val_loss = running_val_loss / running_val_total
        epoch_val_acc = running_val_correct / running_val_total

        scheduler.step(epoch_val_loss)

        training_hist["train_loss"].append(epoch_train_loss)
        training_hist["train_acc"].append(epoch_train_acc)
        training_hist["val_loss"].append(epoch_val_loss)
        training_hist["val_acc"].append(epoch_val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch} | "
            f"LR: {current_lr:.6g} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

        # -----------------
        # Early stopping
        # -----------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(
                best_model_wts,
                os.path.join(save_dir, "best_model_vit-384.pt")
            )

            with open(os.path.join(save_dir, "training_hist_vit-384.json"), "w") as f:
                json.dump(training_hist, f, indent=4)

        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for "
                f"{epochs_without_improvement}/{patience} epoch(s)."
            )

    model.load_state_dict(best_model_wts)

    with open(os.path.join(save_dir, "training_hist_vit-384.json"), "w") as f:
        json.dump(training_hist, f, indent=4)