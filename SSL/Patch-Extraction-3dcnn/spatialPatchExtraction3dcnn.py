import spectral as sp
import numpy as np
from PIL import Image
import os

def extract_patches_direct_write(hdr_path, mask_path, save_dir, patch_size=50, stride=25):
    """
    OOM-safe extraction: directly writes patches into memmap files (no merge step needed).
    """
    # 1. Load hyperspectral raw file (lazy)
    hsi = sp.open_image(hdr_path)
    H, W, B = hsi.shape
    print(f"✅ Hyperspectral raw shape: {H}, {W}, {B}")

    # 2. Load segmentation mask
    from PIL import Image
    mask_img = Image.open(mask_path).convert("L")
    labels = np.array(mask_img, dtype=np.uint8)
    print(f"✅ Label mask shape: {labels.shape}")

    os.makedirs(save_dir, exist_ok=True)

    # 3. First pass: count total patches
    total_patches = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            if labels[i + patch_size // 2, j + patch_size // 2] in [0, 1, 2]:
                total_patches += 1
    print(f"🔢 Total patches to extract: {total_patches}")

    # 4. Preallocate memmap files (writes directly to disk)
    patches_file = os.path.join(save_dir, "vnir_patches.npy")
    labels_file = os.path.join(save_dir, "vnir_segmented_patch_labels.npy")
    patches_memmap = np.memmap(patches_file, dtype='float32', mode='w+', shape=(total_patches, patch_size, patch_size, B))
    labels_memmap = np.memmap(labels_file, dtype='uint8', mode='w+', shape=(total_patches,))

    # 5. Second pass: write patches incrementally
    idx = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            center_label = labels[i + patch_size // 2, j + patch_size // 2]
            if center_label in [0, 1, 2]:
                patch = hsi[i:i+patch_size, j:j+patch_size, :]  # Lazy read
                patches_memmap[idx] = patch
                labels_memmap[idx] = center_label
                idx += 1

    # 6. Flush memmap to finalize files
    patches_memmap.flush()
    labels_memmap.flush()

    print(f"✅ Saved patches: {patches_file}")
    print(f"✅ Saved labels: {labels_file}")
    print(f"✅ Final shapes: {patches_memmap.shape}, {labels_memmap.shape}")

# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    extract_patches_direct_write(
        hdr_path="/home/n51x164/SPIE-2025/BurnSSL-DRL/data/raw/VNIR.hdr",
        mask_path="/home/n51x164/SPIE-2025/BurnSSL-DRL/data/segmentation_mask.png",
        save_dir="/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn",
        patch_size=50,
        stride=25
    )
