import argparse
from pathlib import Path
import pandas as pd
from src.preprocessing import prepare_xy
from src.modeling import train_and_evaluate, save_pipeline

def main():
    parser = argparse.ArgumentParser(description="Train Project Sentinel asteroid classifier.")
    parser.add_argument("--data", default="data/neo.csv", help="Path to NEO CSV dataset")
    parser.add_argument("--output", default="models/asteroid_pipeline.joblib", help="Output model path")
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}. See data/README.md.")
    X, y = prepare_xy(pd.read_csv(data_path))
    model, details = train_and_evaluate(X, y)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_pipeline(model, details, output)
    print(f"Selected model: {details['selected_model']}")
    print("Held-out test metrics:")
    for name, value in details["metrics"].items():
        print(f"  {name}: {value:.4f}")
    print(f"Saved pipeline: {output}")

if __name__ == "__main__":
    main()
