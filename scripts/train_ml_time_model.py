from __future__ import annotations

from pathlib import Path

from fl_sim.ml_training import train_ml_time_model


def main() -> None:
    data_path = Path("data/training_time_dataset.csv")
    out_path = Path("data/ml_time_model.npz")

    params = train_ml_time_model(data_path, out_path=out_path)
    print(f"Trained ML time model and saved parameters to: {out_path}")
    print("Weights:", params.w)
    print("Bias:", params.b)


if __name__ == "__main__":
    main()

