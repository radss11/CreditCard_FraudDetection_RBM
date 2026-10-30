
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = "data/creditcard.csv"
OUTPUT_DIR = "data/processed"
FIGURE_DIR = "outputs/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("FRAUDLENS AI - DATA EXPLORATION")
print("=" * 55)

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:", df.isnull().sum().sum())
print("\nDuplicate rows:", df.duplicated().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

# --------------------------------------------------
# 2. Visualize class imbalance
# --------------------------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Class")
plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/class_distribution.png", dpi=150)
plt.close()

# --------------------------------------------------
# 3. Explore transaction amounts
# --------------------------------------------------

plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=60,
    stat="count",
    element="step",
    common_norm=False
)
plt.title("Transaction Amount Distribution")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/amount_distribution.png", dpi=150)
plt.close()

print("\nAmount statistics:")
print(df.groupby("Class")["Amount"].describe())

# --------------------------------------------------
# 4. Prepare features and labels
# --------------------------------------------------

# Class is the target label, NOT an RBM input.
X = df.drop(columns=["Class"])
y = df["Class"].astype(np.int64)

feature_columns = X.columns.tolist()

# Split into 70% train, 15% validation, 15% test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nSplit sizes:")
print("Training:", X_train.shape, y_train.value_counts().to_dict())
print("Validation:", X_val.shape, y_val.value_counts().to_dict())
print("Test:", X_test.shape, y_test.value_counts().to_dict())

# --------------------------------------------------
# 5. Train-only preprocessing
# --------------------------------------------------

# The RBM will learn normal transaction patterns only.
normal_train = X_train.loc[y_train == 0]

# Scale all features to [0, 1] for a Bernoulli-style RBM.
# Fit on normal training data only to avoid information leakage.
scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
scaler.fit(normal_train)

X_train_normal = scaler.transform(normal_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# --------------------------------------------------
# 6. Save prepared data and preprocessing artifacts
# --------------------------------------------------

np.save(f"{OUTPUT_DIR}/X_train_normal.npy", X_train_normal)
np.save(f"{OUTPUT_DIR}/X_val.npy", X_val_scaled)
np.save(f"{OUTPUT_DIR}/y_val.npy", y_val.to_numpy())

np.save(f"{OUTPUT_DIR}/X_test.npy", X_test_scaled)
np.save(f"{OUTPUT_DIR}/y_test.npy", y_test.to_numpy())

joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
joblib.dump(feature_columns, f"{OUTPUT_DIR}/feature_columns.pkl")

print("\nPreprocessing completed successfully!")
print("Normal training samples:", X_train_normal.shape)
print("Validation samples:", X_val_scaled.shape)
print("Test samples:", X_test_scaled.shape)

print("\nSaved artifacts in:", OUTPUT_DIR)
print("Saved EDA plots in:", FIGURE_DIR)
