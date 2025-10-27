# Script: extract_patches.py
# Description: Extract 50x50 spatial patches with stride 25 from cleaned VNIR cube

import numpy as np
import os


class PatchExtractor:
    def __init__(self, cube_path, patch_size=50, stride=25, output_dir="/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1"):
        self.cube_path = cube_path
        self.patch_size = patch_size
        self.stride = stride
        self.output_dir = output_dir
        self.cube = None
        self.patches = []

    def load_cube(self):
        self.cube = np.load(self.cube_path)
        print(f"✅ Loaded cube with shape: {self.cube.shape}")

    def extract_patches(self):
        H, W, B = self.cube.shape
        p = self.patch_size
        s = self.stride
        patch_id = 0
        os.makedirs(self.output_dir, exist_ok=True)

        for i in range(0, H - p + 1, s):
            for j in range(0, W - p + 1, s):
                patch = self.cube[i:i + p, j:j + p, :]
                patch_file = os.path.join(self.output_dir, f"patch_{patch_id:05d}.npy")
                np.save(patch_file, patch)
                patch_id += 1

        print(f"✅ Extracted {patch_id} patches to {self.output_dir}")


def main():
    extractor = PatchExtractor(
        cube_path="/home/n51x164/SPIE-2025/BurnSSL-DRL/data/processed/VNIR_cleaned.npy",
        patch_size=50,
        stride=25,
        output_dir="/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1"
    )
    extractor.load_cube()
    extractor.extract_patches()


if __name__ == "__main__":
    main()
