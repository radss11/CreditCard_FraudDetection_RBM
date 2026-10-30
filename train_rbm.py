import os
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = "data/processed"
MODEL_DIR = "artifacts"
OUTPUT_DIR = "outputs/training"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_UNITS = 64
CD_STEPS = 1

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Training device:", device)

# --------------------------------------------------
# 2. Load training and validation data
# --------------------------------------------------

X_train = np.load(os.path.join(DATA_DIR, "X_train_normal.npy")).astype(np.float32)

# Use only normal validation examples to monitor reconstruction.
# This does not use fraud labels to train the RBM.
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy")).astype(np.float32)
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy")).astype(np.int64).reshape(-1)

if len(X_val) != len(y_val):
    raise ValueError("X_val.npy and y_val.npy have different row counts.")

X_val_normal = X_val[y_val == 0]
if len(X_val_normal) == 0:
    raise ValueError("No normal rows found in the validation set.")

print("Normal training data shape:", X_train.shape)
print("Normal validation data shape:", X_val_normal.shape)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val_normal)),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)

# --------------------------------------------------
# 3. Define the RBM
# --------------------------------------------------

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))

    def hidden_probabilities(self, visible):
        return torch.sigmoid(visible @ self.W + self.hidden_bias)

    def sample_hidden(self, visible):
        probabilities = self.hidden_probabilities(visible)
        return probabilities, torch.bernoulli(probabilities)

    def visible_probabilities(self, hidden):
        return torch.sigmoid(hidden @ self.W.T + self.visible_bias)

    def sample_visible(self, hidden):
        probabilities = self.visible_probabilities(hidden)
        return probabilities, torch.bernoulli(probabilities)

    def contrastive_divergence(self, visible, k=1):
        positive_hidden_prob = self.hidden_probabilities(visible)
        hidden = torch.bernoulli(positive_hidden_prob)

        for _ in range(k):
            _, visible_sample = self.sample_visible(hidden)
            hidden_prob, hidden = self.sample_hidden(visible_sample)

        return positive_hidden_prob, visible, visible_sample, hidden_prob

    def reconstruction_error(self, visible):
        hidden_prob = self.hidden_probabilities(visible)
        reconstructed = self.visible_probabilities(hidden_prob)
        return torch.mean((visible - reconstructed) ** 2, dim=1)


# --------------------------------------------------
# 4. Initialize model
# --------------------------------------------------

n_visible = X_train.shape[1]
model = RBM(n_visible=n_visible, n_hidden=HIDDEN_UNITS).to(device)

print("Visible units:", n_visible)
print("Hidden units:", HIDDEN_UNITS)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("CD steps:", CD_STEPS)

# --------------------------------------------------
# 5. Train and monitor validation reconstruction error
# --------------------------------------------------

history_path = os.path.join(OUTPUT_DIR, "rbm_training_history.csv")
best_model_path = os.path.join(MODEL_DIR, "rbm_model.pt")
last_model_path = os.path.join(MODEL_DIR, "rbm_model_last.pt")

best_val_error = float("inf")
best_epoch = 0
best_state = None

with open(history_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "train_reconstruction_error", "val_normal_reconstruction_error"]
    )
    writer.writeheader()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_error_sum = 0.0
        train_samples = 0

        for (batch,) in train_loader:
            batch = batch.to(device)

            positive_hidden, v0, vk, negative_hidden = model.contrastive_divergence(
                batch, k=CD_STEPS
            )

            current_batch_size = batch.size(0)

            # Contrastive Divergence parameter updates.
            with torch.no_grad():
                model.W.add_(
                    LEARNING_RATE
                    * (v0.T @ positive_hidden - vk.T @ negative_hidden)
                    / current_batch_size
                )
                model.visible_bias.add_(
                    LEARNING_RATE * (v0 - vk).mean(dim=0)
                )
                model.hidden_bias.add_(
                    LEARNING_RATE * (positive_hidden - negative_hidden).mean(dim=0)
                )

            batch_error = model.reconstruction_error(batch).mean()
            train_error_sum += batch_error.item() * current_batch_size
            train_samples += current_batch_size

        train_error = train_error_sum / max(train_samples, 1)

        model.eval()
        val_error_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                errors = model.reconstruction_error(batch)
                val_error_sum += errors.sum().item()
                val_samples += batch.size(0)

        val_error = val_error_sum / max(val_samples, 1)

        writer.writerow({
            "epoch": epoch,
            "train_reconstruction_error": train_error,
            "val_normal_reconstruction_error": val_error
        })
        csv_file.flush()

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train error: {train_error:.6f} | "
            f"Normal validation error: {val_error:.6f}"
        )

        # Select the checkpoint with the lowest normal-validation reconstruction error.
        if np.isfinite(val_error) and val_error < best_val_error:
            best_val_error = val_error
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            checkpoint = {
                "model_state_dict": best_state,
                "n_visible": n_visible,
                "n_hidden": HIDDEN_UNITS,
                "cd_steps": CD_STEPS,
                "epochs": epoch,
                "feature_range": [0.0, 1.0],
                "best_val_normal_reconstruction_error": best_val_error,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "seed": SEED
            }
            torch.save(checkpoint, best_model_path)

# Save the final epoch separately for comparison, without replacing the best checkpoint.
last_checkpoint = {
    "model_state_dict": {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    },
    "n_visible": n_visible,
    "n_hidden": HIDDEN_UNITS,
    "cd_steps": CD_STEPS,
    "epochs": EPOCHS,
    "feature_range": [0.0, 1.0],
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "seed": SEED
}
torch.save(last_checkpoint, last_model_path)

print("\nTraining completed.")
print(f"Best epoch: {best_epoch}")
print(f"Best normal validation reconstruction error: {best_val_error:.6f}")
print("Best checkpoint:", best_model_path)
print("Final-epoch checkpoint:", last_model_path)
print("Training history:", history_path)
print("\nThe Streamlit app can continue loading artifacts/rbm_model.pt.")
