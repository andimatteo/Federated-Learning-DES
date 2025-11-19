from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .config import ModelConfig
from .hardware import HostHardwareInfo, collect_host_hardware


FEATURE_NAMES: List[str] = [
    "cpu_cores",
    "cpu_freq_ghz",
    "mem_available_gb",
    "accelerator_tflops",
    "log_flops_per_sample",
    "log_model_size_bytes",
    "log_samples_per_client",
    "log_batch_size",
    "log_local_epochs",
]


def build_feature_vector(hw: HostHardwareInfo, model_cfg: ModelConfig) -> np.ndarray:
    """
    Build the feature vector used by the ML time model from hardware + model metadata.
    """
    if model_cfg.samples_per_client <= 0:
        raise ValueError("samples_per_client must be positive for ML time model")
    if model_cfg.flops_per_sample <= 0:
        raise ValueError("flops_per_sample must be positive for ML time model")
    if model_cfg.model_size_bytes <= 0:
        raise ValueError("model_size_bytes must be positive for ML time model")
    if model_cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive for ML time model")
    if model_cfg.local_epochs <= 0:
        raise ValueError("local_epochs must be positive for ML time model")

    return np.array(
        [
            float(hw.cpu_cores),
            float(hw.cpu_freq_ghz),
            float(hw.mem_available_gb),
            float(hw.accelerator_tflops),
            float(np.log10(model_cfg.flops_per_sample)),
            float(np.log10(model_cfg.model_size_bytes)),
            float(np.log10(model_cfg.samples_per_client)),
            float(np.log10(model_cfg.batch_size)),
            float(np.log10(model_cfg.local_epochs)),
        ],
        dtype=float,
    )


@dataclass
class MLTimeModelParams:
    # Either linear (w, b) or small MLP (W1, b1, W2, b2).
    model_type: str
    w: np.ndarray | None
    b: float | None
    W1: np.ndarray | None
    b1: np.ndarray | None
    W2: np.ndarray | None
    b2: np.ndarray | None
    mean: np.ndarray  # shape (D,)
    std: np.ndarray  # shape (D,)
    feature_names: List[str]
    label_scale: str | None = None  # e.g. "seconds" or "log10_seconds"


def _train_mlp_regressor(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 16,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train a small 1-hidden-layer MLP regressor:
      y ≈ W2(ReLU(X W1 + b1)) + b2
    using simple mini-batch gradient descent.
    """
    n, d = X.shape
    rng = np.random.default_rng(123)

    W1 = 0.01 * rng.standard_normal((d, hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=float)
    W2 = 0.01 * rng.standard_normal((hidden_dim, 1))
    b2 = np.zeros((1,), dtype=float)

    for _ in range(epochs):
        idx = rng.permutation(n)
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = X_shuffled[start:end]
            yb = y_shuffled[start:end].reshape(-1, 1)

            # Forward
            z1 = Xb @ W1 + b1
            h1 = np.maximum(z1, 0.0)
            y_pred = h1 @ W2 + b2

            # Loss and gradient (MSE)
            diff = y_pred - yb
            B = Xb.shape[0]
            grad_y_pred = 2.0 * diff / float(B)

            grad_W2 = h1.T @ grad_y_pred + l2 * W2
            grad_b2 = grad_y_pred.sum(axis=0)

            dh1 = grad_y_pred @ W2.T
            dz1 = dh1 * (z1 > 0.0)

            grad_W1 = Xb.T @ dz1 + l2 * W1
            grad_b1 = dz1.sum(axis=0)

            W1 -= lr * grad_W1
            b1 -= lr * grad_b1
            W2 -= lr * grad_W2
            b2 -= lr * grad_b2

    return W1, b1, W2, b2


def train_ml_time_model(
    csv_path: str | Path,
    out_path: str | Path = "data/ml_time_model.npz",
    l2: float = 1e-3,
) -> MLTimeModelParams:
    """
    Train a small MLP-based ML model to approximate training time from the
    emulated dataset (real measurements).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("Empty dataset for ML time model")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    host_hw = collect_host_hardware()

    for row in rows:
        def _get_float(key: str, default: float) -> float:
            v = row.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _get_int(key: str, default: int) -> int:
            v = row.get(key)
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        cpu_cores = _get_int("cpu_cores", host_hw.cpu_cores)
        cpu_freq_ghz = _get_float("cpu_freq_ghz", host_hw.cpu_freq_ghz)
        mem_available_gb = _get_float("mem_available_gb", host_hw.mem_available_gb)
        accelerator_tflops = _get_float("accelerator_tflops", host_hw.accelerator_tflops)

        hw = HostHardwareInfo(
            cpu_cores=cpu_cores,
            cpu_freq_ghz=cpu_freq_ghz,
            mem_total_gb=float(mem_available_gb) * 1.5,
            mem_available_gb=mem_available_gb,
            accelerator_tflops=accelerator_tflops,
            has_gpu=accelerator_tflops > 0.0,
        )
        samples_per_client = _get_int("samples_per_client", 100)
        batch_size = _get_int("batch_size", 128)
        local_epochs = _get_int("local_epochs", 1)

        model_cfg = ModelConfig(
            kind="dummy_logreg",
            input_dim=10,
            num_classes=3,
            samples_per_client=samples_per_client,
            test_samples=500,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=0.1,
            eval_every_rounds=1,
            flops_per_sample=_get_float("flops_per_sample", 1e6),
            model_size_bytes=_get_int("model_size_bytes", 10_000_000),
        )

        try:
            y_val = float(row.get("training_time_seconds"))
        except (TypeError, ValueError):
            # Skip malformed rows.
            continue

        x = build_feature_vector(hw, model_cfg)
        X_list.append(x)
        y_list.append(y_val)

    X = np.vstack(X_list)
    y_seconds = np.asarray(y_list, dtype=float)

    # Work in log10-space to handle the wide dynamic range of training times.
    y = np.log10(y_seconds + 1e-6)

    # Standardise features for numerical stability.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    X_std = (X - mean) / std

    W1, b1, W2, b2 = _train_mlp_regressor(X_std, y, hidden_dim=16, epochs=200, batch_size=64, lr=1e-3, l2=l2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        model_type="mlp_v1",
        label_scale="log10_seconds",
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        mean=mean,
        std=std,
        feature_names=np.array(FEATURE_NAMES, dtype=object),
    )

    params = MLTimeModelParams(
        model_type="mlp_v1",
        w=None,
        b=None,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        mean=mean,
        std=std,
        feature_names=list(FEATURE_NAMES),
        label_scale="log10_seconds",
    )
    return params
