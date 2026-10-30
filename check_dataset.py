import pandas as pd
from pathlib import Path

file_path = Path("data/creditcard.csv")

if not file_path.exists():
    print("Dataset not found. Check the data folder.")
else:
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nClass distribution:")
    print(df["Class"].value_counts())