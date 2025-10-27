import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 1. Dataset
# ============================
class HyperspectralDataset(Dataset):
    def __init__(self, patches_path, labels_path, bands_idx):
        self.patches = np.load(patches_path)  # (N, 50, 50, 165)
        self.labels = np.load(labels_path)
        self.patches = self.patches[..., bands_idx]  # Select DRL bands
        self.patches = np.transpose(self.patches, (0, 3, 1, 2))  # (N, K, 50, 50)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = torch.tensor(self.patches[idx], dtype=torch.float32).unsqueeze(0)  # (1, K, 50, 50)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return patch, label

# ============================
# 2. AgniNet (3D CNN)
# ============================
class AgniNet(nn.Module):
    def __init__(self, num_bands, num_classes=3):
        super(AgniNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ============================
# 3. Training Function
# ============================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                outputs = model(patches)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        acc = accuracy_score(val_true, val_preds)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "agni_best.pth")
    print(f"✅ Training complete. Best Val Acc: {best_acc:.4f}")

# ============================
# 4. Evaluation & Metrics
# ============================
def evaluate_model(model, loader, device, class_names):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for patches, labels in loader:
            patches, labels = patches.to(device), labels.to(device)
            outputs = model(patches)
            predictions = torch.argmax(outputs, dim=1)
            preds.extend(predictions.cpu().numpy())
            true.extend(labels.cpu().numpy())

    acc = accuracy_score(true, preds)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(true, preds, target_names=class_names))

    cm = sns.heatmap(
        confusion_matrix(true, preds),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix - AgniNet")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Precision, Recall, F1 (per class + macro)
    p, r, f1, _ = precision_recall_fscore_support(true, preds, labels=[0,1,2], zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(true, preds, average="macro")
    return acc, p, r, f1, p_macro, r_macro, f1_macro

# ============================
# 5. Main Script
# ============================
if __name__ == "__main__":

    PATCHES = "/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn/vnir_patches.npy"
    LABELS = "/home/n51x164/SPIE-2025/BurnSSL-DRL/data/spatialpatches3dcnn/vnir_segmented_patch_labels.npy"
    SELECTED_BANDS = np.load("/home/n51x164/SPIE-2025/BurnSSL-DRL/outputs/consensus_top_bands_drl.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOP_K_LIST = [10, 20, 30, 40, 50]
    results = []

    for k in TOP_K_LIST:
        print(f"\n🔹 Training AgniNet with Top-{k} DRL Bands...\n")
        selected_bands_k = SELECTED_BANDS[:k]

        dataset = HyperspectralDataset(PATCHES, LABELS, selected_bands_k)
        n = len(dataset)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        test_size = n - train_size - val_size
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        test_loader = DataLoader(test_set, batch_size=32)

        model = AgniNet(num_bands=k, num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50)
        model.load_state_dict(torch.load("agni_best.pth"))

        acc, p, r, f1, p_macro, r_macro, f1_macro = evaluate_model(model, test_loader, device, ["Tree", "Grass", "Soil"])

        results.append({
            "Top-K": k,
            "Accuracy": acc,
            "Precision_Tree": p[0], "Recall_Tree": r[0], "F1_Tree": f1[0],
            "Precision_Grass": p[1], "Recall_Grass": r[1], "F1_Grass": f1[1],
            "Precision_Soil": p[2], "Recall_Soil": r[2], "F1_Soil": f1[2],
            "Precision_Macro": p_macro, "Recall_Macro": r_macro, "F1_Macro": f1_macro
        })

    df = pd.DataFrame(results)
    df.to_csv("agni_drl_classification_results.csv", index=False)
    print("\n✅ Results saved to agni_drl_classification_results.csv")
    print(df)
