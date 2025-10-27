import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

# === Paths ===
PATCH_MEANS_PATH = "/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1/patch_means.npy"
LABELS_PATH = "/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1/patch_labels.npy"
BANDS_PATH = "/home/n51x164/SPIE-2025/BurnSSL-DRL/outputs/consensus_top_bands_drl.npy"

# === Load Data ===
features = np.load(PATCH_MEANS_PATH)  # (N, 165)
labels = np.load(LABELS_PATH)
selected_bands = np.load(BANDS_PATH)

if labels.shape[0] > features.shape[0]:
    labels = labels[:features.shape[0]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

# === Metric Calculation Function ===
def evaluate_classifier(X_tr, y_tr, X_te, y_te, classifier_name, use_class_weight=True):
    if classifier_name == "SVM":
        clf = SVC(kernel="rbf", C=10, gamma="scale",
                  class_weight="balanced" if use_class_weight else None, random_state=42)
    elif classifier_name == "RF":
        clf = RandomForestClassifier(n_estimators=100,
                                     class_weight="balanced" if use_class_weight else None,
                                     random_state=42)
    elif classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Unknown classifier")

    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_te, preds, average="macro")
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_te, preds, average="weighted")
    return acc, p_macro, r_macro, f1_macro, p_weighted, r_weighted, f1_weighted

# === Evaluation: Baseline vs SMOTE ===
TOP_K_LIST = [10, 20, 30, 40, 50]
baseline_results, smote_results = [], []

smote = SMOTE(random_state=42)

for k in TOP_K_LIST:
    top_k_bands = selected_bands[:k]
    X_train_k = X_train[:, top_k_bands]
    X_test_k = X_test[:, top_k_bands]

    # === Baseline (No SMOTE) ===
    for clf_name in ["SVM", "RF", "KNN"]:
        acc, p_macro, r_macro, f1_macro, p_weighted, r_weighted, f1_weighted = evaluate_classifier(
            X_train_k, y_train, X_test_k, y_test, clf_name
        )
        baseline_results.append([k, clf_name, acc, p_macro, r_macro, f1_macro, p_weighted, r_weighted, f1_weighted])

    # === SMOTE Balanced ===
    X_train_bal, y_train_bal = smote.fit_resample(X_train_k, y_train)
    for clf_name in ["SVM", "RF", "KNN"]:
        acc, p_macro, r_macro, f1_macro, p_weighted, r_weighted, f1_weighted = evaluate_classifier(
            X_train_bal, y_train_bal, X_test_k, y_test, clf_name
        )
        smote_results.append([k, clf_name, acc, p_macro, r_macro, f1_macro, p_weighted, r_weighted, f1_weighted])

# === Convert to DataFrames ===
baseline_df = pd.DataFrame(baseline_results, columns=["Top-K", "Classifier", "Acc", "P_Macro", "R_Macro", "F1_Macro", "P_Weighted", "R_Weighted", "F1_Weighted"])
smote_df = pd.DataFrame(smote_results, columns=["Top-K", "Classifier", "Acc", "P_Macro", "R_Macro", "F1_Macro", "P_Weighted", "R_Weighted", "F1_Weighted"])

# === Merge for Side-by-Side Comparison ===
comparison_df = baseline_df.merge(smote_df, on=["Top-K", "Classifier"], suffixes=("_Baseline", "_SMOTE"))

# === Save CSV ===
comparison_df.to_csv("baseline_vs_smote_comparison.csv", index=False)
print("✅ Saved comparison results to baseline_vs_smote_comparison.csv")

# === Generate LaTeX Table ===
latex_table = comparison_df[[
    "Top-K", "Classifier",
    "F1_Macro_Baseline", "F1_Macro_SMOTE",
    "F1_Weighted_Baseline", "F1_Weighted_SMOTE"
]]

print("\n=== LaTeX Table: Baseline vs SMOTE ===\n")
print(latex_table.to_latex(index=False, float_format="%.4f"))
