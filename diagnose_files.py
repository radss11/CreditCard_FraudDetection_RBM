
from pathlib import Path
import pickle
import torch

base = Path(__file__).resolve().parent

files = {
    "RBM checkpoint": base / "artifacts" / "rbm_model.pt",
    "Scaler": base / "data" / "processed" / "scaler.pkl",
    "Feature columns": base / "data" / "processed" / "feature_columns.pkl",
}

for name, path in files.items():
    print(f"\n--- {name} ---")
    print("Path:", path)
    print("Exists:", path.exists())

    if not path.exists():
        continue

    try:
        if name == "RBM checkpoint":
            obj = torch.load(
                path,
                map_location="cpu",
                weights_only=False
            )
            print("LOAD OK")
            print("Keys:", list(obj.keys()))
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            print("LOAD OK")
            print("Type:", type(obj))
            if name == "Feature columns":
                print("Columns:", list(obj)[:5], "...")
    except Exception as e:
        print("LOAD FAILED:", repr(e))
