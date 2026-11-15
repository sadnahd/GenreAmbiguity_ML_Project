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
from sklearn.metrics import silhouette_score

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


class RepeatedRandomVariableWidthChunkDataset(Dataset):
    def __init__(
        self,
        subset,
        transform=None,
        min_chunk_width=280,
        max_chunk_width=420,
        chunks_per_image=2
    ):
        self.subset = subset
        self.transform = transform
        self.min_chunk_width = min_chunk_width
        self.max_chunk_width = max_chunk_width
        self.chunks_per_image = chunks_per_image

    def __len__(self):
        return len(self.subset) * self.chunks_per_image

    def __getitem__(self, idx):
        real_idx = idx // self.chunks_per_image

        image, label = self.subset[real_idx]
        width, height = image.size

        crop_width = random.randint(self.min_chunk_width, self.max_chunk_width)
        crop_width = min(crop_width, width)

        if width > crop_width:
            left = random.randint(0, width - crop_width)
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
    def __init__(self, subset, transform=None, chunk_width=224, num_chunks=4):
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

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all layers first
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace classifier
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        #Two outputs now: logits and log sigma

        self.logits_head = nn.Linear(512, 7)
        self.log_sigma_head = nn.Linear(512, 7)

        # Unfreeze the last two residual stages
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Unfreeze the classifier
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
        for param in self.logits_head.parameters():
            param.requires_grad = True
        
        for param in self.log_sigma_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)

        logits = self.logits_head(features)
        log_sigma = self.log_sigma_head(features).clamp(-5, 3)
        sigma = torch.exp(log_sigma)

        return logits, sigma

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class HeteroscedasticClassificationLoss(nn.Module):
    """
    Implements Kendall & Gal classification loss, eqs. 10-12.

    logits: (B, C)
    sigma:  (B, C)
    targets: (B,)
    """
    def __init__(self, num_samples=20, label_smoothing=0.0):
        super().__init__()
        self.num_samples = num_samples
        self.label_smoothing = label_smoothing

    def forward(self, logits, sigma, targets):
        B, C = logits.shape
        T = self.num_samples

        eps = torch.randn(T, B, C, device=logits.device, dtype=logits.dtype)

        # x_hat_{i,t} = f_i + sigma_i * epsilon_t
        noisy_logits = logits.unsqueeze(0) + sigma.unsqueeze(0) * eps

        log_probs = F.log_softmax(noisy_logits, dim=-1)  # (T, B, C)

        if self.label_smoothing > 0:
            with torch.no_grad():
                target_dist = torch.zeros(B, C, device=logits.device)
                target_dist.fill_(self.label_smoothing / (C - 1))
                target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

            sample_log_likelihood = (log_probs * target_dist.unsqueeze(0)).sum(dim=-1)
        else:
            sample_log_likelihood = log_probs[:, torch.arange(B, device=logits.device), targets]

        # - log mean_t exp(log p_t)
        loss = -torch.logsumexp(sample_log_likelihood, dim=0) + torch.log(
            torch.tensor(T, device=logits.device, dtype=logits.dtype)
        )

        return loss.mean()

resnet_transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

resnet_transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
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
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    set_seed(SEED)
    
    # merged_ds = datasets.ImageFolder('dataset_merged', transforms.ToTensor()) #Uncomment for basic transform
    merged_ds = datasets.ImageFolder('dataset_merged') #Uncomment for resnet transform
    
    #train-val split
    seed = torch.Generator().manual_seed(109)
    train_size = int(0.9 * len(merged_ds))
    val_size = len(merged_ds) - train_size
    
    train_subset, val_subset = random_split(merged_ds, [train_size, val_size], generator=seed)

    train_dataset = RepeatedRandomVariableWidthChunkDataset(
    train_subset,
    transform=resnet_transform_train,
    min_chunk_width=400,
    max_chunk_width=700,
    chunks_per_image=2
    )
    
    val_dataset = MultiVerticalChunkDataset(
        val_subset,
        transform=resnet_transform_val,
        chunk_width=600,
        num_chunks=5
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    
    model = CustomResNet().to(DEVICE)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    

    optimizer = torch.optim.AdamW([
    {"params": model.resnet.layer3.parameters(), "lr": 1e-5, "weight_decay": 1e-2},
    {"params": model.resnet.layer4.parameters(), "lr": 3e-5, "weight_decay": 1e-2},
    {"params": model.resnet.fc.parameters(),     "lr": 1e-4, "weight_decay": 1e-2},
    ])
    
    criterion = HeteroscedasticClassificationLoss(num_samples=20, label_smoothing=0.03)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )
    
    save_dir = 'models/resnet_uncert_aware'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    patience = 10
    epochs_without_improvement = 0
    epoch = 0
    
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    
    training_hist = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc_det": [],
        "val_acc_noisy": [],
    }
    
    while epochs_without_improvement < patience:
        epoch += 1
    
        # -----------------
        # Training
        # -----------------
        model.train()
        model.freeze_bn()
        
        running_train_loss = 0.0
        running_train_correct = 0
        running_train_total = 0
    
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
    
            optimizer.zero_grad()
    
            logits, sigma = model(inputs)
            loss = criterion(logits, sigma, labels)

            loss.backward()
            optimizer.step()
    
            running_train_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            running_train_correct += (preds == labels).sum().item()
            running_train_total += labels.size(0)
    
        epoch_train_loss = running_train_loss / running_train_total
        epoch_train_acc = running_train_correct / running_train_total
    
        # -----------------
        # Validation
        # -----------------
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        running_val_total = 0
        running_val_noisy_correct = 0
        running_val_noisy_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # inputs: (B, N, C, H, W)
                labels = labels.to(DEVICE)
                inputs = inputs.to(DEVICE)
        
                batch_size, num_chunks, C, H, W = inputs.shape
        
                inputs = inputs.view(batch_size * num_chunks, C, H, W)
                logits, sigma = model(inputs)
        
                #Average outputs
                logits = logits.view(batch_size, num_chunks, -1).mean(dim=1)
                sigma = sigma.view(batch_size, num_chunks, -1).mean(dim=1)

                loss = criterion(logits, sigma, labels)

                running_val_loss += loss.item() * batch_size

                #Logit-only val acc

                preds = logits.argmax(dim=1)

                running_val_correct += (preds == labels).sum().item()
                running_val_total += batch_size

                # noisy / uncertainty-aware validation accuracy
                T = 50
                B, C = logits.shape

                eps = torch.randn(T, B, C, device=logits.device, dtype=logits.dtype)
                noisy_logits = logits.unsqueeze(0) + sigma.unsqueeze(0) * eps

                noisy_probs = F.softmax(noisy_logits, dim=-1).mean(dim=0)
                noisy_preds = noisy_probs.argmax(dim=1)

                running_val_noisy_correct += (noisy_preds == labels).sum().item()
                running_val_noisy_total += batch_size
    
        epoch_val_loss = running_val_loss / running_val_total
        epoch_val_acc = running_val_correct / running_val_total
        epoch_val_noisy_acc = running_val_noisy_correct / running_val_noisy_total

        scheduler.step(epoch_val_loss)
    
        # -----------------
        # Save history
        # -----------------
        training_hist["train_loss"].append(epoch_train_loss)
        training_hist["train_acc"].append(epoch_train_acc)
        training_hist["val_loss"].append(epoch_val_loss)
        training_hist["val_acc_det"].append(epoch_val_acc)
        training_hist["val_acc_noisy"].append(epoch_val_noisy_acc)
    
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch} | "
            f"LR: {current_lr:.6g} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc (logit-only): {epoch_val_acc:.4f}"
            f" | Val Acc (Uncertainty-aware): {epoch_val_noisy_acc:.4f}"
        )
    
        # -----------------
        # Early stopping on validation loss
        # -----------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            best_model_wts = copy.deepcopy(model.state_dict())
    
            torch.save(best_model_wts, os.path.join(save_dir, "best_model_resnet_uncert.pt"))
    
            with open(os.path.join(save_dir, "training_hist_resnet_uncert.json"), "w") as f:
                json.dump(training_hist, f, indent=4)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epoch(s).")
    
    # Load best model weights at the end
    model.load_state_dict(best_model_wts)
    
    # Final save of history after training ends
    with open(os.path.join(save_dir, "training_hist_resnet_uncert.json"), "w") as f:
        json.dump(training_hist, f, indent=4)