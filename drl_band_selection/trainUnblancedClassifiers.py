import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
PATCH_MEANS_PATH = "/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1/patch_means.npy"
LABELS_PATH = "/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1/patch_labels.npy"
BANDS_PATH = "/home/n51x164/SPIE-2025/BurnSSL-DRL/outputs/consensus_top_bands_drl.npy"

# === Load data ===
features = np.load(PATCH_MEANS_PATH)  # Shape: (N, 165)
labels = np.load(LABELS_PATH)
selected_bands = np.load(BANDS_PATH)

print(f"✅ Loaded features: {features.shape}, labels: {labels.shape}, selected bands: {selected_bands.shape}")

# Align labels length if mismatch
if labels.shape[0] > features.shape[0]:
    labels = labels[:features.shape[0]]

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

# === Classifier Training Function ===
def train_and_evaluate(X_tr, X_te, y_tr, y_te, classifier_name):
    if classifier_name == "SVM":
        clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    elif classifier_name == "RF":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Unknown classifier")

    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_te, preds, average=None)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_te, preds, average="macro")
    return acc, precision, recall, f1, precision_macro, recall_macro, f1_macro, preds

# === Evaluation Loop ===
results = []
TOP_K_LIST = [10, 20, 30, 40, 50]

for k in TOP_K_LIST:
    top_k_bands = selected_bands[:k]
    X_train_k = X_train[:, top_k_bands]
    X_test_k = X_test[:, top_k_bands]

    for clf_name in ["SVM", "RF", "KNN"]:
        acc, precision, recall, f1, p_macro, r_macro, f1_macro, preds = train_and_evaluate(
            X_train_k, X_test_k, y_train, y_test, clf_name
        )

        # Save per-class metrics
        for cls_idx in range(len(np.unique(labels))):
            results.append({
                "Top-K": k,
                "Classifier": clf_name,
                "Class": f"Class {cls_idx}",
                "Accuracy": acc,
                "Precision": precision[cls_idx],
                "Recall": recall[cls_idx],
                "F1": f1[cls_idx],
                "Precision (Macro)": p_macro,
                "Recall (Macro)": r_macro,
                "F1 (Macro)": f1_macro
            })

        # === Confusion Matrix ===
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
        plt.title(f"Confusion Matrix: {clf_name} (Top-{k} Bands)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"conf_matrix_{clf_name}_top{k}.png", dpi=300)
        plt.close()

# === Save metrics to CSV ===
df_results = pd.DataFrame(results)
df_results.to_csv("classification_metrics_drl_patch_means.csv", index=False)
print("✅ Saved classification metrics to classification_metrics_drl_patch_means.csv")
