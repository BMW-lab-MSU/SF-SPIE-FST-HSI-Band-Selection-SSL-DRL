import numpy as np
import os

PATCH_DIR = "/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1"
OUTPUT_PATH = os.path.join(PATCH_DIR, "patch_means.npy")

means = []

patch_files = sorted([f for f in os.listdir(PATCH_DIR) if f.startswith("patch_") and f.endswith(".npy")])

for idx, file in enumerate(patch_files):
    patch_path = os.path.join(PATCH_DIR, file)
    patch = np.load(patch_path)

    # ✅ Validate shape: must be (50, 50, 165)
    if patch.ndim == 3 and patch.shape == (50, 50, 165):
        means.append(patch.mean(axis=(0, 1)))  # -> (165,)
    else:
        print(f"⚠️ Skipping invalid patch: {file}, shape: {patch.shape}")

    if idx % 5000 == 0:
        print(f"Processed {idx}/{len(patch_files)} patches")

means = np.stack(means)  # (N, 165)
np.save(OUTPUT_PATH, means)
print(f"✅ Saved mean spectra: {means.shape} to {OUTPUT_PATH}")
