import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Sequence

# ============================
# 0. Repro & matmul precision
# ============================
def set_seed(seed=1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(1337)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ============================
# 1. Dataset
# ============================
class HyperspectralDataset(Dataset):
    """
    Loads patches/labels via memory-mapped .npy, selects bands (K),
    returns torch tensors shaped (K, 50, 50).
    """
    def __init__(self, patches_path, labels_path, bands_idx: Sequence[int]):
        self.patches_mm = np.load(patches_path, mmap_mode='r')  # (N, 50, 50, B)
        self.labels_mm  = np.load(labels_path,  mmap_mode='r')  # (N,)
        self.patches = self.patches_mm[..., bands_idx]          # (N, 50, 50, K)
        # store as (N, K, 50, 50)
        self.patches = np.transpose(self.patches, (0, 3, 1, 2))
        self.labels = self.labels_mm  # keep separate handle

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx]).float()  # (K, 50, 50)
        label = int(self.labels[idx])
        return patch, label

class SubsetWithAugment(Dataset):
    """
    Wrap a base dataset with a subset of indices and optional train-time augmentations.
    Augmentations are kept gentle for hyperspectral data.
    """
    def __init__(self, base_ds: HyperspectralDataset, indices: Sequence[int], augment: bool = False):
        self.base = base_ds
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def _augment(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: (K, 50, 50)
        # Random flips
        if torch.rand(1).item() < 0.5:
            patch = torch.flip(patch, dims=[-1])   # horizontal
        if torch.rand(1).item() < 0.5:
            patch = torch.flip(patch, dims=[-2])   # vertical
        # Mild Gaussian noise (safe scale)
        if torch.rand(1).item() < 0.3:
            patch = patch + torch.randn_like(patch) * 0.01
        return patch

    def __getitem__(self, i):
        patch, label = self.base[self.indices[i]]
        if self.augment:
            patch = self._augment(patch)
        return patch, label

# ============================
# 2. AgniNet (3D CNN)
# ============================
class AgniNet(nn.Module):
    def __init__(self, num_bands, num_classes=3):
        super(AgniNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (N, 1, K, 50, 50)
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================
# 3. Utilities for imbalance
# ============================
def compute_class_weights(train_labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    counts = np.bincount(train_labels.astype(int), minlength=num_classes).astype(float)
    # Inverse-frequency weights normalized like: N / (C * count_c)
    weights = (len(train_labels) / (num_classes * counts))
    return weights

def make_weighted_sampler_for_subset(base_ds: HyperspectralDataset, subset_indices: Sequence[int],
                                     class_weights: np.ndarray) -> WeightedRandomSampler:
    # map each index -> class -> per-sample weight
    subset_labels = base_ds.labels[subset_indices].astype(int)
    sample_weights = class_weights[subset_labels]
    # WeightedRandomSampler expects a 1D torch Tensor of weights, length == len(subset)
    return WeightedRandomSampler(weights=torch.as_tensor(sample_weights, dtype=torch.double),
                                 num_samples=len(subset_indices),
                                 replacement=True)

# ============================
# 4. Training Function (AMP + grad clip + scheduler)
# ============================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50,
                max_norm=1.0, use_amp=True, print_gpu_mem=False):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for patches, labels in train_loader:
            patches = patches.to(device, non_blocking=True)       # (N, K, 50, 50)
            labels  = labels.to(device, non_blocking=True)        # (N,)
            inputs  = patches.unsqueeze(1)                        # (N, 1, K, 50, 50)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validation (AMP safe)
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for patches, labels in val_loader:
                patches = patches.to(device, non_blocking=True)
                labels  = labels.to(device, non_blocking=True)
                outputs = model(patches.unsqueeze(1))
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        acc = accuracy_score(val_true, val_preds)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

        if print_gpu_mem and device.type == 'cuda' and (epoch == 0 or (epoch+1) % 10 == 0):
            print("\n[GPU Memory Usage]")
            print(torch.cuda.memory_summary(device=device))

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "agni_best.pth")

    print(f"✅ Training complete. Best Val Acc: {best_acc:.4f}")

# ============================
# 5. Evaluation & Metrics
# ============================
def evaluate_model(model, loader, device, class_names, use_amp=True):
    model.eval()
    preds, true = [], []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for patches, labels in loader:
            patches = patches.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            outputs = model(patches.unsqueeze(1))
            predictions = torch.argmax(outputs, dim=1)
            preds.extend(predictions.cpu().numpy())
            true.extend(labels.cpu().numpy())

    acc = accuracy_score(true, preds)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(true, preds, target_names=class_names, zero_division=0))

    cm = sns.heatmap(
        confusion_matrix(true, preds),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix - AgniNet")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Precision, Recall, F1
    p, r, f1, _ = precision_recall_fscore_support(true, preds, labels=list(range(len(class_names))), zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(true, preds, average="macro", zero_division=0)
    return acc, p, r, f1, p_macro, r_macro, f1_macro

# ============================
# 6. Main Script
# ============================
if __name__ == "__main__":
    PATCHES = "/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn/vnir_patches_clean.npy"
    LABELS  = "/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn/vnir_labels_clean.npy"
    SELECTED_BANDS = np.load("/home/n51x164/SPIE-2025/BurnSSL-DRL/outputs/consensus_top_bands_drl.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOP_K_LIST = [10, 20, 30, 40, 50]
    results = []
    class_names = ["Soil", "Grass", "Tree"]

    # DataLoader settings
    NUM_WORKERS = min(8, os.cpu_count() or 4)
    PIN = (device.type == "cuda")

    for k in TOP_K_LIST:
        print(f"\n🔹 Training AgniNet with Top-{k} DRL Bands...\n")
        selected_bands_k = SELECTED_BANDS[:k]

        # Base dataset (shared memory maps)
        base_dataset = HyperspectralDataset(PATCHES, LABELS, selected_bands_k)
        n = len(base_dataset)
        print(n)

        # Reproducible split
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        test_size = n - train_size - val_size
        g = torch.Generator().manual_seed(1337)
        train_subset, val_subset, test_subset = random_split(base_dataset, [train_size, val_size, test_size], generator=g)

        # Wrap with per-split augment flags (train only)
        train_set = SubsetWithAugment(base_dataset, train_subset.indices, augment=True)
        val_set   = SubsetWithAugment(base_dataset, val_subset.indices,   augment=False)
        test_set  = SubsetWithAugment(base_dataset, test_subset.indices,  augment=False)

        # ==== Class weights (train split only) ====
        train_labels_np = base_dataset.labels[train_subset.indices].astype(int)
        class_weights_np = compute_class_weights(train_labels_np, num_classes=len(class_names))
        class_weights_t  = torch.tensor(class_weights_np, dtype=torch.float, device=device)

        # ==== Weighted sampler for training ====
        sampler = make_weighted_sampler_for_subset(base_dataset, train_subset.indices, class_weights_np)

        # Batch size auto-tune for higher K
        batch_size = 16 if k > 30 else 32

        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=PIN, persistent_workers=PIN)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=PIN, persistent_workers=PIN)
        test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=PIN, persistent_workers=PIN)

        # Model, loss (weighted), optimizer
        model = AgniNet(num_bands=k, num_classes=len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)  # imbalance-aware
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Train & Evaluate
        train_model(model, train_loader, val_loader, criterion, optimizer, device,
                    epochs=50, max_norm=1.0, use_amp=True, print_gpu_mem=False)

        model.load_state_dict(torch.load("agni_best.pth", map_location=device))

        acc, p, r, f1, p_macro, r_macro, f1_macro = evaluate_model(model, test_loader, device, class_names, use_amp=True)

        results.append({
            "Top-K": k,
            "Accuracy": acc,
            "Precision_Soil": p[0], "Recall_Soil": r[0], "F1_Soil": f1[0],
            "Precision_Grass": p[1], "Recall_Grass": r[1], "F1_Grass": f1[1],
            "Precision_Tree": p[2], "Recall_Tree": r[2], "F1_Tree": f1[2],
            "Precision_Macro": p_macro, "Recall_Macro": r_macro, "F1_Macro": f1_macro
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("agni_drl_classification_balanced_results.csv", index=False)
    print("\n✅ Results saved to agni_drl_classification_results.csv")
    print(df)
