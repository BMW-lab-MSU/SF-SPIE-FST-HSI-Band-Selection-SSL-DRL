import spectral as sp
import numpy as np
from PIL import Image
import os

def extract_patches_chunked(hdr_path, mask_path, save_dir, patch_size=50, stride=25, chunk_size=20000):
    """
    Extracts patches from raw VNIR and saves in chunks to avoid OOM, then merges them properly.
    """
    hsi = sp.open_image(hdr_path)  # Lazy load hyperspectral image
    H, W, B = hsi.shape
    print(f"✅ Hyperspectral raw shape: {H}, {W}, {B}")

    mask_img = Image.open(mask_path).convert("L")
    labels = np.array(mask_img, dtype=np.uint8)
    print(f"✅ Label mask shape: {labels.shape}")

    os.makedirs(save_dir, exist_ok=True)

    patches, patch_labels = [], []
    chunk_idx, total_patches = 0, 0

    # Sliding window
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            center_label = labels[i + patch_size // 2, j + patch_size // 2]
            if center_label in [0, 1, 2]:  
                patch = hsi[i:i+patch_size, j:j+patch_size, :]
                patches.append(patch)
                patch_labels.append(center_label)
                total_patches += 1

                # Save chunks periodically
                if len(patches) >= chunk_size:
                    np.save(os.path.join(save_dir, f"vnir_patches_part{chunk_idx}.npy"), np.array(patches, dtype=np.float32))
                    np.save(os.path.join(save_dir, f"vnir_labels_part{chunk_idx}.npy"), np.array(patch_labels, dtype=np.uint8))
                    print(f"💾 Saved chunk {chunk_idx} with {len(patches)} patches")
                    patches, patch_labels = [], []
                    chunk_idx += 1

    # Save final chunk
    if patches:
        np.save(os.path.join(save_dir, f"vnir_patches_part{chunk_idx}.npy"), np.array(patches, dtype=np.float32))
        np.save(os.path.join(save_dir, f"vnir_labels_part{chunk_idx}.npy"), np.array(patch_labels, dtype=np.uint8))
        print(f"💾 Saved final chunk {chunk_idx} with {len(patches)} patches")

    print(f"✅ Total patches extracted: {total_patches}")
    merge_chunks(save_dir)

def merge_chunks(save_dir):
    """
    Merges saved chunks into final npy arrays without full memory load.
    """
    patch_files = sorted([f for f in os.listdir(save_dir) if f.startswith("vnir_patches_part")])
    label_files = sorted([f for f in os.listdir(save_dir) if f.startswith("vnir_labels_part")])

    # First, determine total shape from chunks
    sample_patch = np.load(os.path.join(save_dir, patch_files[0]), mmap_mode='r')
    patch_size, _, bands = sample_patch.shape[1:]
    total_patches = sum(np.load(os.path.join(save_dir, f), mmap_mode='r').shape[0] for f in patch_files)
    print(f"🔢 Merging {total_patches} patches...")

    # Create final memmap files
    patches_final = np.memmap(os.path.join(save_dir, "vnir_patches.npy"), dtype='float32', mode='w+', shape=(total_patches, patch_size, patch_size, bands))
    labels_final = np.memmap(os.path.join(save_dir, "vnir_segmented_patch_labels.npy"), dtype='uint8', mode='w+', shape=(total_patches,))

    # Stream copy chunks into final memmap
    idx = 0
    for pf, lf in zip(patch_files, label_files):
        p_chunk = np.load(os.path.join(save_dir, pf), mmap_mode='r')
        l_chunk = np.load(os.path.join(save_dir, lf), mmap_mode='r')
        patches_final[idx:idx+len(p_chunk)] = p_chunk
        labels_final[idx:idx+len(l_chunk)] = l_chunk
        idx += len(p_chunk)
        print(f"✅ Merged chunk {pf}")

    patches_final.flush()
    labels_final.flush()
    print(f"✅ Final patches saved: {patches_final.shape}")
    print(f"✅ Final labels saved: {labels_final.shape}")

# Run
extract_patches_chunked(
    hdr_path="/home/n51x164/SPIE-2025/BurnSSL-DRL/data/raw/VNIR.hdr",
    mask_path="/home/n51x164/SPIE-2025/BurnSSL-DRL/data/segmentation_mask.png",
    save_dir="/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn",
    patch_size=50,
    stride=25,
    chunk_size=20000  # Adjust chunk size based on node RAM
)
