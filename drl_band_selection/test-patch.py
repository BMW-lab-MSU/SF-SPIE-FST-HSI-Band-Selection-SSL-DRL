import numpy as np

patches = np.load("/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn/vnir_patches_clean.npy", mmap_mode='r')
labels = np.load("/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn/vnir_labels_clean.npy", mmap_mode='r')

print("Patches:", patches.shape, patches.dtype)  # Expect (141764, 50, 50, 273), float32
print("Labels:", labels.shape, labels.dtype)    # Expect (141764,), uint8
print("Unique labels:", np.unique(labels))      # Expect [0, 1, 2]

# Patches: (141764, 50, 50, 273) float32
# Labels: (141764,) uint8
# Unique labels: [0 1 2]



# import numpy as np
# import os
#
# # Paths
# patches_mem_path = "/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn/vnir_patches.npy"
# labels_mem_path = "/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn/vnir_segmented_patch_labels.npy"
# save_dir = "/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn"
#
# # Known shapes and dtypes
# num_patches = 141764
# patch_shape = (50, 50, 273)
#
# # Open memmap
# patches_mem = np.memmap(patches_mem_path, dtype='float32', mode='r', shape=(num_patches,) + patch_shape)
# labels_mem = np.memmap(labels_mem_path, dtype='uint8', mode='r', shape=(num_patches,))
#
# # Prepare final .npy writers
# clean_patches_path = os.path.join(save_dir, "vnir_patches_clean.npy")
# clean_labels_path = os.path.join(save_dir, "vnir_labels_clean.npy")
#
# # Save patches in chunks
# chunk_size = 5000  # adjust based on memory
# print("Saving clean patches in chunks...")
#
# # Open npy file for writing manually
# with open(clean_patches_path, 'wb') as f_patches:
#     np.lib.format.write_array_header_1_0(f_patches, dict(
#         descr=np.dtype('float32').str,
#         fortran_order=False,
#         shape=(num_patches,) + patch_shape
#     ))
#
#     for i in range(0, num_patches, chunk_size):
#         end = min(i + chunk_size, num_patches)
#         np.lib.format.write_array(f_patches, patches_mem[i:end], version=(1, 0))
#
# print("✅ Patches saved cleanly!")
#
# # Save labels (small enough to fit in memory safely)
# np.save(clean_labels_path, np.array(labels_mem, dtype='uint8'))
# print("✅ Labels saved cleanly!")
#
# print("🎯 Final clean files:")
# print(f"Patches: {clean_patches_path}")
# print(f"Labels: {clean_labels_path}")

# Output
# Saving clean patches in chunks...
# ✅ Patches saved cleanly!
# ✅ Labels saved cleanly!
# 🎯 Final clean files:
# Patches: /home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn/vnir_patches_clean.npy
# Labels: /home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn/vnir_labels_clean.npy


