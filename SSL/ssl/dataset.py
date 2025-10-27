import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class PatchDataset(Dataset):
    def __init__(self, patch_dir, augment=True):
        self.files = sorted([os.path.join(patch_dir, f) for f in os.listdir(patch_dir) if f.endswith('.npy')])
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patch = np.load(self.files[idx])
        if self.augment:
            patch1 = self.apply_augmentations(patch)
            patch2 = self.apply_augmentations(patch)
            return torch.tensor(patch1.copy(), dtype=torch.float32), torch.tensor(patch2.copy(), dtype=torch.float32)
        else:
            return torch.tensor(patch.copy(), dtype=torch.float32)

    def apply_augmentations(self, patch):
        # spectral dropout
        if random.random() < 0.3:
            drop_idx = np.random.choice(patch.shape[2], size=10, replace=False)
            patch[:, :, drop_idx] = 0

        # spatial flips
        if random.random() < 0.5:
            patch = np.flip(patch, axis=0)
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1)

        # random crop to 45x45
        # if random.random() < 0.5:
        #     x = random.randint(0, 5)
        #     y = random.randint(0, 5)
        #     patch = patch[x:x + 45, y:y + 45, :]

        return patch
