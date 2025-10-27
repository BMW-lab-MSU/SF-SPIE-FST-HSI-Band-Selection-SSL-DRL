import torch
from torch.utils.data import DataLoader
from SimCLR3DCNN import SimCLR3DCNN
from dataset import PatchDataset
from contrastive_loss import contrastive_loss
import os


# epochs=10, batch_size=32,
def train_ssl(patch_dir, epochs=10, batch_size=32, lr=1e-3, save_path="/home/n51x164/SPIE-2025/BurnSSL-DRL/outputs/ssl_encoder.pth"):
    dataset = PatchDataset(patch_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # model = SimCLR3DCNN().cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR3DCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2 in dataloader:
            # x1, x2 = x1.cuda(), x2.cuda()
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = contrastive_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.encoder.state_dict(), save_path)
    print(f"✅ Saved encoder to {save_path}")


if __name__ == "__main__":
    train_ssl(patch_dir="/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1")
