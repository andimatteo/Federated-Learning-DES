from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Event, Process
from pathlib import Path
from typing import Dict

import numpy as np

from fl_sim.config import ModelConfig
from fl_sim.hardware import collect_host_hardware
from fl_sim.model import DummyLogisticRegressionModel


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


def _cpu_stress_worker(stop_event: Event) -> None:
    x = 0.0
    while not stop_event.is_set():
        # Simple numerical loop to keep the core busy.
        x = x * 1.0000001 + 1.0


class CPULoadManager:
    def __init__(self, workers: int) -> None:
        self.workers = max(0, int(workers))
        self.stop_event: Event | None = None
        self.processes: list[Process] = []

    def __enter__(self) -> "CPULoadManager":
        if self.workers <= 0:
            return self
        self.stop_event = Event()
        for _ in range(self.workers):
            p = Process(target=_cpu_stress_worker, args=(self.stop_event,))
            p.daemon = True
            p.start()
            self.processes.append(p)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.stop_event is None:
            return
        self.stop_event.set()
        for p in self.processes:
            p.join(timeout=1.0)


def _sample_config(rng: np.random.Generator, cpu_cores_actual: int) -> Dict[str, float]:
    """
    Sample a configuration of:
      - effective cpu_cores (via background workers)
      - flops_per_sample
      - model_size_bytes
      - samples_per_client
      - cpu and memory load levels
      - model_family: 'logreg', 'cnn_like', 'transformer_like'
    """
    # How many cores to occupy with background load.
    max_bg_workers = max(0, cpu_cores_actual - 1)
    bg_workers = int(rng.integers(0, max_bg_workers + 1)) if max_bg_workers > 0 else 0
    effective_cores = max(1, cpu_cores_actual - bg_workers)

    # Memory load as fraction of total memory.
    mem_load_level = float(rng.uniform(0.0, 0.7))  # up to 70% of total

    # Model family: approximate different architectures.
    model_family = rng.choice(["logreg", "cnn_like", "transformer_like"], p=[0.4, 0.3, 0.3])

    if model_family == "logreg":
        flops_per_sample = float(np.exp(rng.uniform(np.log(5e4), np.log(5e6))))
        model_size_bytes = float(np.exp(rng.uniform(np.log(5e4), np.log(5e6))))
        samples_per_client = int(np.exp(rng.uniform(np.log(50), np.log(2000))))
    elif model_family == "cnn_like":
        flops_per_sample = float(np.exp(rng.uniform(np.log(5e6), np.log(5e8))))
        model_size_bytes = float(np.exp(rng.uniform(np.log(5e6), np.log(5e8))))
        samples_per_client = int(np.exp(rng.uniform(np.log(50), np.log(5000))))
    else:  # transformer_like
        flops_per_sample = float(np.exp(rng.uniform(np.log(1e7), np.log(1e9))))
        model_size_bytes = float(np.exp(rng.uniform(np.log(1e7), np.log(1e9))))
        samples_per_client = int(np.exp(rng.uniform(np.log(50), np.log(5000))))

    return {
        "cpu_cores": effective_cores,
        "bg_workers": bg_workers,
        "mem_load_level": mem_load_level,
        "flops_per_sample": flops_per_sample,
        "model_size_bytes": model_size_bytes,
        "samples_per_client": samples_per_client,
        "model_family": model_family,
    }


def _allocate_memory(mem_total_gb: float, mem_load_level: float) -> bytes | None:
    """
    Allocate a memory block to emulate pressure.
    """
    if mem_load_level <= 0.0 or mem_total_gb <= 0.0:
        return None
    mem_total_bytes = mem_total_gb * (1024.0**3)
    # Do not use more than 1 GB to avoid stressing the environment too much.
    target_bytes = int(min(mem_load_level * mem_total_bytes, 1_000_000_000))
    if target_bytes <= 0:
        return None
    try:
        return bytearray(target_bytes)
    except MemoryError:
        return None


def _build_model_config(cfg: Dict[str, float]) -> ModelConfig:
    # Derive input_dim and num_classes roughly consistent with model_size_bytes,
    # and scaled according to the model family to emulate larger architectures.
    model_size_bytes = max(float(cfg["model_size_bytes"]), 1.0)
    family = cfg.get("model_family", "logreg")

    if family == "logreg":
        input_dim = int(np.clip(np.exp(np.log(10.0) + np.random.rand() * (np.log(200.0 / 10.0))), 10, 200))
    elif family == "cnn_like":
        input_dim = int(np.clip(np.exp(np.log(200.0) + np.random.rand() * (np.log(2000.0 / 200.0))), 200, 2000))
    else:  # transformer_like
        input_dim = int(np.clip(np.exp(np.log(512.0) + np.random.rand() * (np.log(4096.0 / 512.0))), 512, 4096))

    num_classes = max(2, int(model_size_bytes / (8.0 * input_dim)))
    num_classes = int(np.clip(num_classes, 2, 1000))

    return ModelConfig(
        kind="dummy_logreg",
        input_dim=input_dim,
        num_classes=num_classes,
        samples_per_client=int(cfg["samples_per_client"]),
        test_samples=500,
        local_epochs=1,
        learning_rate=0.05,
        eval_every_rounds=1,
        flops_per_sample=float(cfg["flops_per_sample"]),
        model_size_bytes=int(model_size_bytes),
    )


def generate_dataset(
    n_samples: int = 2000,
    seed: int = 123,
    output_path: str | Path = "data/training_time_dataset.csv",
    append: bool = False,
) -> Path:
    """
    Generate a labeled dataset by *actually running* local training under
    different emulated hardware loads.

    For each sample:
      - choose a background CPU and memory load,
      - configure a dummy model (flops_per_sample, model_size_bytes, samples_per_client),
      - run one local training epoch and measure wall-clock training time.

    The label is the real measured training time, not un modello analitico.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hw_host = collect_host_hardware()

    fieldnames = [
        "cpu_cores",           # effective cores available to training
        "cpu_freq_ghz",        # host frequency (GHz)
        "mem_available_gb",    # measured under load
        "accelerator_tflops",  # host accelerators
        "flops_per_sample",
        "model_size_bytes",
        "samples_per_client",
        "bg_workers",
        "mem_load_level",
        "model_family",
        "training_time_seconds",
    ]

    # Decide mode (write vs append) and whether to write header.
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
            cfg = _sample_config(rng, hw_host.cpu_cores)
            bg_workers = int(cfg["bg_workers"])
            mem_load_level = float(cfg["mem_load_level"])

            mem_block = _allocate_memory(hw_host.mem_total_gb, mem_load_level)

            with CPULoadManager(bg_workers):
                # Leave some time for the load to start.
                time.sleep(0.02)

                mem_available_gb = _read_mem_available_gb()

                model_cfg = _build_model_config(cfg)
                model = DummyLogisticRegressionModel(model_cfg, rng)

                t0 = time.perf_counter()
                model.local_train("c0")
                elapsed = time.perf_counter() - t0

            # Release memory after training.
            del mem_block

            row = {
                "cpu_cores": int(cfg["cpu_cores"]),
                "cpu_freq_ghz": float(hw_host.cpu_freq_ghz),
                "mem_available_gb": float(mem_available_gb),
                "accelerator_tflops": float(hw_host.accelerator_tflops),
                "flops_per_sample": float(cfg["flops_per_sample"]),
                "model_size_bytes": float(cfg["model_size_bytes"]),
                "samples_per_client": int(cfg["samples_per_client"]),
                "bg_workers": bg_workers,
                "mem_load_level": mem_load_level,
                "model_family": cfg["model_family"],
                "training_time_seconds": float(elapsed),
            }
            writer.writerow(row)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Emulate hardware loads and measure REAL training times for different model sizes.\n"
            "Use --append and multiple runs to build large datasets (>= 10000 rows)."
        )
    )
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/training_time_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting.",
    )
    args = parser.parse_args()

    out_path = generate_dataset(
        n_samples=args.n_samples,
        seed=args.seed,
        output_path=args.output,
        append=args.append,
    )
    print(f"Generated REAL training-time dataset at: {out_path}")


if __name__ == "__main__":
    main()
