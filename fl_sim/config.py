from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
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
    return Config(
        simulation=sim_cfg,
        training_time=training_cfg,
        communication_time=comm_cfg,
        output=out_cfg,
    )


