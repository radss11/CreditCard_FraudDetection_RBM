
import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

DATA_DIR = "data/processed"
MODEL_PATH = "artifacts/rbm_model.pt"
OUTPUT_DIR = "outputs/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Evaluation device:", device)


# --------------------------------------------------
# 1. Define the same RBM architecture used in training
# --------------------------------------------------

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()

        self.W = nn.Parameter(torch.zeros(n_visible, n_hidden))
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))

    def hidden_probabilities(self, visible):
        return torch.sigmoid(visible @ self.W + self.hidden_bias)

    def visible_probabilities(self, hidden):
        return torch.sigmoid(hidden @ self.W.T + self.visible_bias)

    def reconstruction_error(self, visible):
        hidden_prob = self.hidden_probabilities(visible)
        reconstructed = self.visible_probabilities(hidden_prob)
        return ((visible - reconstructed) ** 2).mean(dim=1)


# --------------------------------------------------
# 2. Load model and validation data
# --------------------------------------------------

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = RBM(
    n_visible=checkpoint["n_visible"],
    n_hidden=checkpoint["n_hidden"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

print("Validation shape:", X_val.shape)
print("Validation fraud count:", int(y_val.sum()))


# --------------------------------------------------
# 3. Calculate reconstruction-error scores
# --------------------------------------------------

scores = []

with torch.no_grad():
    for start in range(0, len(X_val), 1024):
        batch = torch.tensor(
            X_val[start:start + 1024],
            dtype=torch.float32,
            device=device
        )

        batch_scores = model.reconstruction_error(batch)
        scores.extend(batch_scores.cpu().numpy())

scores = np.asarray(scores)

print("\nScore statistics:")
print("Minimum:", scores.min())
print("Maximum:", scores.max())
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())

print("\nMean score for normal transactions:",
      scores[y_val == 0].mean())

print("Mean score for fraudulent transactions:",
      scores[y_val == 1].mean())


# --------------------------------------------------
# 4. Evaluate score ranking
# --------------------------------------------------

if len(np.unique(y_val)) < 2:
    raise ValueError("Validation data must contain both classes.")

pr_auc = average_precision_score(y_val, scores)
roc_auc = roc_auc_score(y_val, scores)

print("\nValidation ranking metrics")
print("Average precision / PR-AUC:", round(pr_auc, 5))
print("ROC-AUC:", round(roc_auc, 5))


# --------------------------------------------------
# 5. Select threshold using validation data only
# --------------------------------------------------

precision, recall, thresholds = precision_recall_curve(
    y_val, scores
)

# precision and recall have one more element than thresholds.
f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

if len(thresholds) == 0:
    raise ValueError("Could not determine a validation threshold.")

best_index = int(np.argmax(f1_values))
best_threshold = float(thresholds[best_index])

val_predictions = (scores >= best_threshold).astype(int)

print("\nSelected validation threshold:", best_threshold)
print("Validation precision:",
      round(precision_score(y_val, val_predictions, zero_division=0), 5))
print("Validation recall:",
      round(recall_score(y_val, val_predictions, zero_division=0), 5))
print("Validation F1:",
      round(f1_score(y_val, val_predictions, zero_division=0), 5))

print("\nValidation classification report:")
print(classification_report(
    y_val,
    val_predictions,
    target_names=["Normal", "Fraud"],
    zero_division=0
))


# --------------------------------------------------
# 6. Visualize score distributions
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    scores[y_val == 0],
    bins=60,
    alpha=0.65,
    density=True,
    label="Normal",
)

plt.hist(
    scores[y_val == 1],
    bins=60,
    alpha=0.65,
    density=True,
    label="Fraud",
)

plt.axvline(
    best_threshold,
    color="red",
    linestyle="--",
    label="Selected threshold",
)

plt.xlabel("Reconstruction error")
plt.ylabel("Density")
plt.title("RBM Anomaly Scores on Validation Data")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "validation_score_distribution.png"),
    dpi=150
)
plt.close()


# --------------------------------------------------
# 7. Save threshold and validation metrics
# --------------------------------------------------

results = {
    "threshold": best_threshold,
    "validation_average_precision": float(pr_auc),
    "validation_roc_auc": float(roc_auc),
    "validation_precision": float(
        precision_score(y_val, val_predictions, zero_division=0)
    ),
    "validation_recall": float(
        recall_score(y_val, val_predictions, zero_division=0)
    ),
    "validation_f1": float(
        f1_score(y_val, val_predictions, zero_division=0)
    ),
}

with open(
    os.path.join(OUTPUT_DIR, "validation_results.json"),
    "w"
) as file:
    json.dump(results, file, indent=4)

np.save(
    os.path.join(OUTPUT_DIR, "validation_scores.npy"),
    scores
)

print("\nSaved validation results to:", OUTPUT_DIR)
