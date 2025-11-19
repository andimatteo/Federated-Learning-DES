"""
Self-contained script to generate a CSV dataset of training times
for different model families (logreg, CNN-like, Transformer-like)
and hardware-like configurations.

This file is intentionally independent of the rest of the repository:
you can copy it to another machine and run it there to collect
additional samples. Requirements:

  - Python 3.9+
  - numpy
  - torch

Usage examples:

  python generate_training_time_dataset_torch.py \\
      --output training_time_dataset.csv \\
      --n-samples 2000

  # Append more samples (different seed) to reach >= 10000 rows:
  python generate_training_time_dataset_torch.py \\
      --output training_time_dataset.csv \\
      --n-samples 2000 --seed 2 --append
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def _read_cpu_info() -> Tuple[int, float]:
    cores = os.cpu_count() or 1
    max_mhz = 0.0
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "cpu MHz" in line:
                    parts = line.split(":")
                    if len(parts) == 2:
                        try:
                            mhz = float(parts[1])
                            if mhz > max_mhz:
                                max_mhz = mhz
                        except ValueError:
                            continue
    except OSError:
        pass
    if max_mhz <= 0:
        max_mhz = 2500.0
    return cores, max_mhz / 1000.0


def _read_mem_available_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    kb = float(parts[1])
                    return kb / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


def _build_logreg_model(input_dim: int, num_classes: int) -> nn.Module:
    return nn.Linear(input_dim, num_classes)


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, c1: int, c2: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SmallTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, num_classes: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = self.enc(x)
        x = x.mean(dim=1)
        return self.fc(x)


def _sample_configuration(rng: np.random.Generator, host_cores: int) -> Dict[str, object]:
    """
    Sample one configuration (model + data + hardware-like settings).
    """
    model_family = rng.choice(["logreg", "cnn_like", "transformer_like"], p=[0.4, 0.3, 0.3])

    # Effective cores via thread limitation.
    possible_threads = [1, 2, 4, 8, 16]
    threads_used = min(host_cores, rng.choice(possible_threads))

    # Device type (for now, CPU only by default).
    device_type = "cpu"

    # Choose model and data sizes.
    if model_family == "logreg":
        input_dim = int(np.exp(rng.uniform(np.log(32), np.log(4096))))
        num_classes = int(rng.choice([2, 10, 100]))
        param_count = input_dim * num_classes
        batch_size = int(rng.choice([64, 128, 256, 512]))
        samples_per_client = int(np.exp(rng.uniform(np.log(100), np.log(20000))))
        seq_len = None
        img_size = None
    elif model_family == "cnn_like":
        input_dim = None
        num_classes = int(rng.choice([10, 100]))
        c1 = int(rng.choice([16, 32, 64]))
        c2 = int(rng.choice([32, 64, 128, 256]))
        img_size = int(rng.choice([32, 64, 128]))
        # Rough param count estimate; we will recompute after model creation.
        param_count = None
        batch_size = int(rng.choice([16, 32, 64]))
        samples_per_client = int(np.exp(rng.uniform(np.log(100), np.log(20000))))
        seq_len = None
    else:  # transformer_like
        input_dim = None
        num_classes = int(rng.choice([10, 100]))
        d_model = int(rng.choice([64, 128, 256, 512]))
        nhead = int(rng.choice([2, 4, 8]))
        num_layers = int(rng.choice([1, 2, 3, 4]))
        dim_ff = int(rng.choice([4 * d_model, 8 * d_model]))
        seq_len = int(rng.choice([32, 64, 128]))
        batch_size = int(rng.choice([8, 16, 32]))
        samples_per_client = int(np.exp(rng.uniform(np.log(100), np.log(10000))))
        img_size = None

    local_epochs = int(rng.integers(1, 6))  # 1–5 epochs

    cfg: Dict[str, object] = {
        "model_family": model_family,
        "threads_used": threads_used,
        "device_type": device_type,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "samples_per_client": samples_per_client,
        "local_epochs": local_epochs,
        "img_size": img_size,
        "seq_len": seq_len,
    }

    if model_family == "cnn_like":
        cfg["c1"] = c1
        cfg["c2"] = c2
    elif model_family == "transformer_like":
        cfg["d_model"] = d_model
        cfg["nhead"] = nhead
        cfg["num_layers"] = num_layers
        cfg["dim_ff"] = dim_ff

    return cfg


def _build_model_and_data(cfg: Dict[str, object], device: torch.device) -> Tuple[nn.Module, torch.Tensor, torch.Tensor, int]:
    family = cfg["model_family"]
    num_classes = int(cfg["num_classes"])
    batch_size = int(cfg["batch_size"])

    if family == "logreg":
        input_dim = int(cfg["input_dim"])
        model = _build_logreg_model(input_dim, num_classes).to(device)
        x = torch.randn(batch_size, input_dim, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
    elif family == "cnn_like":
        c1 = int(cfg["c1"])
        c2 = int(cfg["c2"])
        img_size = int(cfg["img_size"])
        model = SmallCNN(3, c1, c2, num_classes).to(device)
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
    else:  # transformer_like
        d_model = int(cfg["d_model"])
        nhead = int(cfg["nhead"])
        num_layers = int(cfg["num_layers"])
        dim_ff = int(cfg["dim_ff"])
        seq_len = int(cfg["seq_len"])
        model = SmallTransformer(d_model, nhead, num_layers, dim_ff, num_classes).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)

    param_count = sum(p.numel() for p in model.parameters())
    return model, x, y, param_count


def _estimate_training_time_seconds(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    samples_per_client: int,
    local_epochs: int,
    threads_used: int,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    torch.set_num_threads(max(1, threads_used))
    # Some small randomisation of learning rate.
    base_lr = 0.01
    lr = float(base_lr * math.exp(rng.normal(0.0, 0.3)))

    if device.type == "cpu":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Warm-up steps.
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    # Measure time-per-step on a fixed number of iterations.
    num_steps_measure = 5
    start = time.perf_counter()
    for _ in range(num_steps_measure):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    end = time.perf_counter()

    time_per_step = (end - start) / num_steps_measure

    batch_size = x.shape[0]
    steps_per_epoch = math.ceil(samples_per_client / batch_size)
    total_steps = steps_per_epoch * local_epochs

    return float(time_per_step * total_steps)


def generate_dataset(
    n_samples: int = 1000,
    seed: int = 123,
    output_path: str | Path = "training_time_dataset.csv",
    append: bool = False,
    device_type: str = "cpu",
) -> Path:
    """
    Generate a dataset by measuring real training times on this host using PyTorch.

    Columns:
      - cpu_cores:          effective cores (threads) used by training
      - cpu_freq_ghz:       host CPU frequency (approximate)
      - mem_available_gb:   memory available during measurement
      - accelerator_tflops: 0 if CPU, heuristic value if GPU
      - flops_per_sample:   proxy based on model family and parameter count
      - model_size_bytes:   approx parameter size (param_count * 4)
      - samples_per_client: number of samples for local training
      - model_family:       'logreg' / 'cnn_like' / 'transformer_like'
      - param_count:        number of parameters in the model
      - batch_size:         batch size used in measurement
      - local_epochs:       local epochs used for estimating total time
      - device_type:        'cpu' or 'cuda'
      - training_time_seconds: estimated total training time
    """
    rng = np.random.default_rng(seed)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    host_cores, cpu_freq_ghz = _read_cpu_info()

    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        # Very rough guess for TFLOPs; for learning it is enough that it is > 0.
        accelerator_tflops = 10.0
    else:
        device = torch.device("cpu")
        accelerator_tflops = 0.0

    fieldnames = [
        "cpu_cores",
        "cpu_freq_ghz",
        "mem_available_gb",
        "accelerator_tflops",
        "flops_per_sample",
        "model_size_bytes",
        "samples_per_client",
        "model_family",
        "input_dim",
        "img_size",
        "seq_len",
        "param_count",
        "batch_size",
        "local_epochs",
        "device_type",
        "training_time_seconds",
    ]

    write_header = True
    if append and out_path.exists():
        mode = "a"
        write_header = False
    else:
        mode = "w"

    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for _ in range(n_samples):
            cfg = _sample_configuration(rng, host_cores)
            cfg["device_type"] = device.type

            model, x, y, param_count = _build_model_and_data(cfg, device)
            cfg["param_count"] = int(param_count)

            # Approximate model size and flops_per_sample as features.
            model_size_bytes = float(param_count * 4)  # float32
            if cfg["model_family"] == "logreg":
                flops_per_sample = float(param_count * 2)
            elif cfg["model_family"] == "cnn_like":
                flops_per_sample = float(param_count * 10)
            else:  # transformer_like
                flops_per_sample = float(param_count * 20)

            mem_available_gb = _read_mem_available_gb()

            training_time_seconds = _estimate_training_time_seconds(
                model=model,
                x=x,
                y=y,
                samples_per_client=int(cfg["samples_per_client"]),
                local_epochs=int(cfg["local_epochs"]),
                threads_used=int(cfg["threads_used"]),
                device=device,
                rng=rng,
            )

            row = {
                "cpu_cores": int(cfg["threads_used"]),
                "cpu_freq_ghz": float(cpu_freq_ghz),
                "mem_available_gb": float(mem_available_gb),
                "accelerator_tflops": float(accelerator_tflops),
                "flops_per_sample": float(flops_per_sample),
                "model_size_bytes": float(model_size_bytes),
                "samples_per_client": int(cfg["samples_per_client"]),
                "model_family": str(cfg["model_family"]),
                "input_dim": int(cfg.get("input_dim") or 0),
                "img_size": int(cfg.get("img_size") or 0),
                "seq_len": int(cfg.get("seq_len") or 0),
                "param_count": int(cfg["param_count"]),
                "batch_size": int(cfg["batch_size"]),
                "local_epochs": int(cfg["local_epochs"]),
                "device_type": device.type,
                "training_time_seconds": float(training_time_seconds),
            }
            writer.writerow(row)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure training times for different model families using PyTorch.",
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of configurations.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="training_time_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for measurement (default: cpu).",
    )
    args = parser.parse_args()

    out_path = generate_dataset(
        n_samples=args.n_samples,
        seed=args.seed,
        output_path=args.output,
        append=args.append,
        device_type=args.device,
    )
    print(f"Generated training-time dataset at: {out_path}")


if __name__ == "__main__":
    main()
