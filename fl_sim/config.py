from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping
import json
import tomllib


@dataclass
class SimulationConfig:
    rounds: int = 10
    num_clients: int = 10
    clients_per_round: int | None = None
    participation_rate: float = 1.0
    seed: int = 0


@dataclass
class TimeModelConfig:
    kind: str = "lognormal"  # lognormal, exponential, normal, constant, trace
    mean: float = 1.0
    sigma: float = 0.5
    rate: float | None = None  # for exponential
    constant: float | None = None
    file: str | None = None  # for trace
    column: str = "value"  # column name in CSV when using trace
    # For bandwidth-based communication time model: optional global defaults.
    default_bandwidth_mean_mbps: float | None = None
    default_bandwidth_jitter: float | None = None


@dataclass
class ModelConfig:
    # Whether to actually instantiate and train the FL model.
    enabled: bool = True
    # If True, use wall-clock time of dummy training inside the simulator as training_time
    # (overrides statistical/ML time models when the model is enabled).
    measure_training_time: bool = False
    # Kind of FL model to use ("dummy_logreg" for now).
    kind: str = "dummy_logreg"
    # Synthetic dataset / model shape.
    input_dim: int = 10
    num_classes: int = 2
    samples_per_client: int = 100
    test_samples: int = 500
    # Local training hyperparameters.
    local_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.1
    # How often to evaluate the global model (in rounds).
    eval_every_rounds: int = 1
    # Meta-information for time modelling (approximate).
    flops_per_sample: float = 1e6
    model_size_bytes: int = 10_000_000


@dataclass
class OutputConfig:
    directory: str = "output"
    write_traces_csv: bool = True
    write_summary_json: bool = True
    write_histograms_json: bool = False
    per_client_histograms: bool = False
    histogram_bins: int = 20


@dataclass
class Config:
    simulation: SimulationConfig
    training_time: TimeModelConfig
    communication_time: TimeModelConfig
    output: OutputConfig
    model: ModelConfig | None = None
    # Optional per-device overrides (indexed by client_id as string).
    devices: Dict[str, "DeviceConfig"] | None = None


@dataclass
class DeviceConfig:
    """
    Optional per-device hardware and network configuration.

    Any field left as None falls back to host defaults or global settings.
    """

    # Hardware
    cpu_cores: int | None = None
    cpu_freq_ghz: float | None = None
    mem_total_gb: float | None = None
    mem_available_gb: float | None = None
    accelerator_tflops: float | None = None
    has_gpu: bool | None = None

    # Network (uplink bandwidth for communication time).
    bandwidth_mean_mbps: float | None = None
    bandwidth_jitter: float | None = None
    # Optional override for local batch size used in ML time model features.
    batch_size: int | None = None


def _load_raw_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    data: Dict[str, Any]
    if suffix == ".toml":
        with path.open("rb") as f:
            data = tomllib.load(f)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix} (use .toml or .json)")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a table/object")
    return data


def _dict_to_simulation_config(raw: Dict[str, Any]) -> SimulationConfig:
    sim = raw.get("simulation", {}) or {}
    return SimulationConfig(
        rounds=int(sim.get("rounds", 10)),
        num_clients=int(sim.get("num_clients", 10)),
        clients_per_round=(
            int(sim["clients_per_round"]) if sim.get("clients_per_round") is not None else None
        ),
        participation_rate=float(sim.get("participation_rate", 1.0)),
        seed=int(sim.get("seed", 0)),
    )


def _dict_to_time_model_config(raw: Dict[str, Any], key: str) -> TimeModelConfig:
    tm = raw.get(key, {}) or {}
    return TimeModelConfig(
        kind=str(tm.get("kind", "lognormal")),
        mean=float(tm.get("mean", 1.0)),
        sigma=float(tm.get("sigma", 0.5)),
        rate=(float(tm["rate"]) if tm.get("rate") is not None else None),
        constant=(float(tm["constant"]) if tm.get("constant") is not None else None),
        file=(str(tm["file"]) if tm.get("file") is not None else None),
        column=str(tm.get("column", "value")),
        default_bandwidth_mean_mbps=(
            float(tm["default_bandwidth_mean_mbps"])
            if tm.get("default_bandwidth_mean_mbps") is not None
            else None
        ),
        default_bandwidth_jitter=(
            float(tm["default_bandwidth_jitter"])
            if tm.get("default_bandwidth_jitter") is not None
            else None
        ),
    )


def _dict_to_output_config(raw: Dict[str, Any]) -> OutputConfig:
    out = raw.get("output", {}) or {}
    return OutputConfig(
        directory=str(out.get("directory", "output")),
        write_traces_csv=bool(out.get("write_traces_csv", True)),
        write_summary_json=bool(out.get("write_summary_json", True)),
        write_histograms_json=bool(out.get("write_histograms_json", False)),
        per_client_histograms=bool(out.get("per_client_histograms", False)),
        histogram_bins=int(out.get("histogram_bins", 20)),
    )


def _dict_to_model_config(raw: Dict[str, Any]) -> ModelConfig | None:
    model = raw.get("model")
    if not model:
        return None
    m = model or {}
    return ModelConfig(
        enabled=bool(m.get("enabled", True)),
        measure_training_time=bool(m.get("measure_training_time", False)),
        kind=str(m.get("kind", "dummy_logreg")),
        input_dim=int(m.get("input_dim", 10)),
        num_classes=int(m.get("num_classes", 2)),
        samples_per_client=int(m.get("samples_per_client", 100)),
        test_samples=int(m.get("test_samples", 500)),
        local_epochs=int(m.get("local_epochs", 1)),
        batch_size=int(m.get("batch_size", 128)),
        learning_rate=float(m.get("learning_rate", 0.1)),
        eval_every_rounds=int(m.get("eval_every_rounds", 1)),
        flops_per_sample=float(m.get("flops_per_sample", 1e6)),
        model_size_bytes=int(m.get("model_size_bytes", 10_000_000)),
    )


def _dict_to_devices_config(raw: Dict[str, Any]) -> Dict[str, DeviceConfig] | None:
    devices_raw = raw.get("devices")
    if not devices_raw or not isinstance(devices_raw, Mapping):
        return None

    devices: Dict[str, DeviceConfig] = {}
    for key, dev in devices_raw.items():
        if not isinstance(dev, Mapping):
            continue
        devices[str(key)] = DeviceConfig(
            cpu_cores=(int(dev["cpu_cores"]) if dev.get("cpu_cores") is not None else None),
            cpu_freq_ghz=(
                float(dev["cpu_freq_ghz"]) if dev.get("cpu_freq_ghz") is not None else None
            ),
            mem_total_gb=(
                float(dev["mem_total_gb"]) if dev.get("mem_total_gb") is not None else None
            ),
            mem_available_gb=(
                float(dev["mem_available_gb"])
                if dev.get("mem_available_gb") is not None
                else None
            ),
            accelerator_tflops=(
                float(dev["accelerator_tflops"])
                if dev.get("accelerator_tflops") is not None
                else None
            ),
            has_gpu=(
                bool(dev["has_gpu"]) if dev.get("has_gpu") is not None else None
            ),
            batch_size=(int(dev["batch_size"]) if dev.get("batch_size") is not None else None),
            bandwidth_mean_mbps=(
                float(dev["bandwidth_mean_mbps"])
                if dev.get("bandwidth_mean_mbps") is not None
                else None
            ),
            bandwidth_jitter=(
                float(dev["bandwidth_jitter"]) if dev.get("bandwidth_jitter") is not None else None
            ),
        )
    return devices or None


def load_config(path: str | Path) -> Config:
    """
    Load a simulation configuration from a TOML or JSON file.
    """
    p = Path(path)
    raw = _load_raw_config(p)
    sim_cfg = _dict_to_simulation_config(raw)
    training_cfg = _dict_to_time_model_config(raw, "training_time")
    comm_cfg = _dict_to_time_model_config(raw, "communication_time")
    out_cfg = _dict_to_output_config(raw)
    model_cfg = _dict_to_model_config(raw)
    devices_cfg = _dict_to_devices_config(raw)
    return Config(
        simulation=sim_cfg,
        training_time=training_cfg,
        communication_time=comm_cfg,
        output=out_cfg,
        model=model_cfg,
        devices=devices_cfg,
    )
