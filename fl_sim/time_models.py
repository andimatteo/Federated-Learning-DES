from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List

import csv
import numpy as np

from .config import TimeModelConfig


class TimeModel(ABC):
    """
    Abstract base class for models that generate training / communication times.
    """

    @abstractmethod
    def sample(self, client_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Sample a time value for each client id.
        Returns an array of shape (len(client_ids),).
        """


class DistributionTimeModel(TimeModel):
    """
    Time model based on a parametric statistical distribution.
    """

    def __init__(self, cfg: TimeModelConfig) -> None:
        self.cfg = cfg

    def sample(self, client_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        size = len(client_ids)
        kind = self.cfg.kind.lower()

        if kind == "lognormal":
            mean = float(self.cfg.mean)
            sigma = float(self.cfg.sigma)
            return rng.lognormal(mean=mean, sigma=sigma, size=size)

        if kind == "exponential":
            rate = float(self.cfg.rate) if self.cfg.rate is not None else 1.0 / max(
                self.cfg.mean, 1e-9
            )
            return rng.exponential(scale=1.0 / rate, size=size)

        if kind == "normal":
            mean = float(self.cfg.mean)
            sigma = float(self.cfg.sigma)
            return np.clip(rng.normal(loc=mean, scale=sigma, size=size), a_min=0.0, a_max=None)

        if kind == "constant":
            value = float(self.cfg.constant if self.cfg.constant is not None else self.cfg.mean)
            return np.full(shape=size, fill_value=value, dtype=float)

        raise ValueError(f"Unsupported distribution kind: {self.cfg.kind}")


class TraceTimeModel(TimeModel):
    """
    Time model that replays values from a trace file.

    The trace is a CSV file with at least one numeric column (default: 'value').
    Optionally it can contain a 'client_id' column to specify per-client traces.
    """

    def __init__(self, cfg: TimeModelConfig) -> None:
        if cfg.file is None:
            raise ValueError("TraceTimeModel requires 'file' in TimeModelConfig")
        self.cfg = cfg
        self._global_values: np.ndarray | None = None
        self._per_client_values: Dict[str, np.ndarray] | None = None
        self._load_trace(Path(cfg.file))

    def _load_trace(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        values: List[float] = []
        per_client: Dict[str, List[float]] = {}
        has_client_id = False

        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if self.cfg.column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Trace file {path} missing column '{self.cfg.column}'. "
                    f"Available columns: {reader.fieldnames}"
                )
            has_client_id = "client_id" in (reader.fieldnames or [])

            for row in reader:
                try:
                    v = float(row[self.cfg.column])
                except ValueError:
                    continue
                values.append(v)
                if has_client_id:
                    cid = str(row["client_id"])
                    per_client.setdefault(cid, []).append(v)

        if not values:
            raise ValueError(f"Trace file {path} has no valid rows")

        self._global_values = np.asarray(values, dtype=float)

        if has_client_id:
            self._per_client_values = {k: np.asarray(v, dtype=float) for k, v in per_client.items()}
        else:
            self._per_client_values = None

    def sample(self, client_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        assert self._global_values is not None
        size = len(client_ids)

        if self._per_client_values is None:
            # Use a global pool of times
            idx = rng.integers(0, len(self._global_values), size=size)
            return self._global_values[idx]

        # Per-client traces when available
        out = np.empty(shape=size, dtype=float)
        for i, cid in enumerate(client_ids):
            arr = self._per_client_values.get(str(cid))
            if arr is None or len(arr) == 0:
                # Fallback to global distribution
                j = rng.integers(0, len(self._global_values))
                out[i] = self._global_values[j]
            else:
                j = rng.integers(0, len(arr))
                out[i] = arr[j]
        return out


def build_time_model(cfg: TimeModelConfig) -> TimeModel:
    kind = cfg.kind.lower()
    if kind == "trace":
        return TraceTimeModel(cfg)
    return DistributionTimeModel(cfg)


