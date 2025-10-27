# # Script: remove_noisy_bands.py
# # Description: Remove bands within known noisy wavelength ranges
#
# from spectral import envi
# import numpy as np
# import os
#
#
# def remove_noisy_bands(cube, wavelengths, noisy_ranges=[(1340, 1450), (1800, 2000)]):
#     bands_to_keep = [i for i, wl in enumerate(wavelengths)
#                      if all(not (low <= wl <= high) for (low, high) in noisy_ranges)]
#     cleaned_cube = cube[:, :, bands_to_keep]
#     return cleaned_cube, bands_to_keep
#
#
# def main():
#     hdr_path = "../data/raw/VNIR.hdr"
#     cube_path = "../data/processed/VNIR_normalized.npy"
#     output_dir = "../data/processed"
#
#     hdr = envi.read_envi_header(hdr_path)
#     wavelengths = [float(w) for w in hdr['wavelength']]
#     cube = np.load(cube_path)
#
#     cleaned_cube, kept_indices = remove_noisy_bands(cube, wavelengths)
#     os.makedirs(output_dir, exist_ok=True)
#     np.save(f"{output_dir}/VNIR_cleaned.npy", cleaned_cube)
#     np.save(f"{output_dir}/kept_band_indices.npy", kept_indices)
#     print(f"✅ Saved cleaned cube and kept indices. Total bands kept: {len(kept_indices)}")
#
#
# if __name__ == "__main__":
#     main()

# Script: remove_noisy_bands.py
# Description: Remove noisy bands using predefined index ranges

from spectral import envi
import numpy as np
import os


class BandCleaner:
    def __init__(self, hdr_path, cube_path, output_dir, bad_band_ranges=None):
        self.hdr_path = hdr_path
        self.cube_path = cube_path
        self.output_dir = output_dir
        self.bad_band_ranges = bad_band_ranges if bad_band_ranges else [
            (0, 40),  # Low signal-to-noise
            (135, 150),  # Water absorption
            (180, 200),  # Water absorption
            (240, 273)  # Edge noise
        ]
        self.cube = None
        self.cleaned_cube = None
        self.kept_indices = []

    def load_cube_and_metadata(self):
        self.cube = np.load(self.cube_path)
        hdr = envi.read_envi_header(self.hdr_path)
        self.total_bands = self.cube.shape[2]
        print(f"✅ Loaded cube with shape: {self.cube.shape}")

    def get_kept_band_indices(self):
        remove_indices = set()
        for start, end in self.bad_band_ranges:
            remove_indices.update(range(start, min(end, self.total_bands)))
        self.kept_indices = [i for i in range(self.total_bands) if i not in remove_indices]

    def filter_bands(self):
        self.cleaned_cube = self.cube[:, :, self.kept_indices]

    def save_outputs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        np.save(os.path.join(self.output_dir, "VNIR_cleaned.npy"), self.cleaned_cube)
        np.save(os.path.join(self.output_dir, "kept_band_indices.npy"), self.kept_indices)
        print(f"✅ Saved cleaned cube and band indices. Bands kept: {len(self.kept_indices)}")


def main():
    cleaner = BandCleaner(
        hdr_path="/home/n51x164/SPIE-2025/BurnSSL-DRL/data/raw/VNIR.hdr",
        cube_path="/home/n51x164/SPIE-2025/BurnSSL-DRL/data/processed/VNIR_normalized.npy",
        output_dir="/home/n51x164/SPIE-2025/BurnSSL-DRL/data/processed"
    )
    cleaner.load_cube_and_metadata()
    cleaner.get_kept_band_indices()
    cleaner.filter_bands()
    cleaner.save_outputs()


if __name__ == "__main__":
    main()
