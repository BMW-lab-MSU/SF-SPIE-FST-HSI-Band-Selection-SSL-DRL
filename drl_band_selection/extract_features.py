import torch
import numpy as np
import os
from SimCLR3DCNN import SimCLR3DCNN


def main():
    PATCH_DIR = "/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1"
    SAVE_PATH = "/home/n51x164/SPIE-2025/BurnSSL-DRL/features/features.npy"
    BATCH_SIZE = 64  # adjust based on memory
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
    model = SimCLR3DCNN().encoder.to(DEVICE)
    model.load_state_dict(torch.load("/home/n51x164/SPIE-2025/BurnSSL-DRL/outputs/ssl_encoder.pth"))
    model.eval()

    patch_files = sorted(f for f in os.listdir(PATCH_DIR) if f.endswith(".npy"))
    features = []
    batch = []

    for idx, pf in enumerate(patch_files):
        patch = np.load(os.path.join(PATCH_DIR, pf))

        if patch.ndim != 3 or patch.shape[0] != 50 or patch.shape[1] != 50:
            print(f"⚠️ Skipping patch {pf} with shape {patch.shape}")
            continue

        patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 50, 50, B)
        batch.append(patch)

        if len(batch) == BATCH_SIZE or idx == len(patch_files) - 1:
            batch_tensor = torch.cat(batch, dim=0).to(DEVICE)  # (B, 1, 50, 50, B)
            with torch.no_grad():
                batch_feats = model(batch_tensor).cpu().numpy()  # (B, 32)
                features.append(batch_feats)
            batch = []

    features = np.concatenate(features, axis=0)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.save(SAVE_PATH, features)
    print(f"✅ Saved features to {SAVE_PATH} with shape {features.shape}")


if __name__ == "__main__":
    main()

#     ⚠️ Skipping patch patch_labels.npy with shape (141764,)
# ✅ Saved features to /home/n51x164/SPIE-2025/BurnSSL-DRL/features/features.npy with shape (141760, 32, 1, 1, 1)
