import numpy as np
import os

patch_dir = "/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1"
patch_files = sorted([f for f in os.listdir(patch_dir) if f.endswith('.npy')])
shapes = {}

for file in patch_files:
    patch = np.load(os.path.join(patch_dir, file))
    if patch.shape not in shapes:
        shapes[patch.shape] = []
    shapes[patch.shape].append(file)

for shape, files in shapes.items():
    print(f"Shape: {shape}, Count: {len(files)}")

if len(shapes) > 1:
    print("\n❌ Found inconsistent shapes!")
else:
    print("\n✅ All patches have the same shape.")

# Shape: (50, 50, 165), Count: 141764
# Shape: (141764,), Count: 1
#
# ❌ Found inconsistent shapes!