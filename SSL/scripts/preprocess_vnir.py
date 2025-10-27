# Script: preprocess_vnir.py
# Description: Load and normalize the VNIR hyperspectral cube

import spectral
import numpy as np
import os


def normalize_cube(cube):
    for i in range(cube.shape[2]):
        band = cube[:, :, i]
        cube[:, :, i] = (band - np.min(band)) / (np.max(band) - np.min(band) + 1e-6)
    return cube


def main():
    hdr_path = "/home/n51x164/SPIE-2025/BurnSSL-DRL/data/raw/VNIR.hdr"  # Adjust as needed
    print(f"Loading HDR from: {hdr_path}")
    print(f"File exists: {os.path.exists(hdr_path)}")
    img = spectral.open_image(hdr_path)
    cube = img.load().astype(np.float32)
    normalized_cube = normalize_cube(cube)
    os.makedirs("/home/n51x164/SPIE-2025/BurnSSL-DRL/data/processed", exist_ok=True)
    np.save("/home/n51x164/SPIE-2025/BurnSSL-DRL/data/processed/VNIR_normalized.npy", normalized_cube)
    print("✅ VNIR cube normalized and saved.")


if __name__ == "__main__":
    main()
