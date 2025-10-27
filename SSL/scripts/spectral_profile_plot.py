# Script: spectral_profile_plot.py
# Description: Generate a smoothed mean ± std spectral profile plot

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


def main():
    cube = np.load("/home/n51x164/SPIE-2025/BurnSSL-DRL/data/processed/VNIR_normalized.npy")
    H, W, B = cube.shape
    mean_spectrum = np.mean(cube.reshape(-1, B), axis=0)
    std_spectrum = np.std(cube.reshape(-1, B), axis=0)

    mean_smooth = savgol_filter(mean_spectrum, 11, 2)
    std_smooth = savgol_filter(std_spectrum, 11, 2)

    plt.figure(figsize=(12, 5))
    plt.plot(range(B), mean_smooth, label="Mean Spectrum")
    plt.fill_between(range(B), mean_smooth - std_smooth, mean_smooth + std_smooth,
                     color='gray', alpha=0.3, label="±1 STD")
    plt.title("VNIR Spectral Profile (Mean ± STD with SG Smoothing)")
    plt.xlabel("Band Index")
    plt.ylabel("Normalized Reflectance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("/home/n51x164/SPIE-2025/BurnSSL-DRL/plots", exist_ok=True)
    plt.savefig("/home/n51x164/SPIE-2025/BurnSSL-DRL/plots/spectral_profile.png")
    plt.close()
    print("✅ Spectral profile plot saved to plots/spectral_profile.png")


if __name__ == "__main__":
    main()
