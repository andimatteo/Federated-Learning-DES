from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import csv
import numpy as np

from .config import TimeModelConfig, SimulationConfig, ModelConfig
from .hardware import HostHardwareInfo, DeviceProfile
from .ml_training import build_feature_vector, MLTimeModelParams, FEATURE_NAMES


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
    # Kept for backward compatibility: when no hardware/context is needed.
    kind = cfg.kind.lower()
    if kind == "trace":
        return TraceTimeModel(cfg)
    if kind in {"lognormal", "exponential", "normal", "constant"}:
        return DistributionTimeModel(cfg)
    raise ValueError(f"Unsupported time model kind without context: {cfg.kind}")


@dataclass
class TimeModelBuildContext:
    """
    Additional information for hardware-aware time models.
    """

    sim_cfg: SimulationConfig
    model_cfg: Optional[ModelConfig]
    hardware: HostHardwareInfo
    role: str  # "training" or "communication"
    # Optional per-client profiles (hardware + bandwidth).
    device_profiles: Optional[Dict[str, DeviceProfile]] = None
    # Optional per-client batch sizes for ML time model.
    device_batch_sizes: Optional[Dict[str, int]] = None


class HardwareCalibratedTimeModel(TimeModel):
    """
    Time model that derives a base time from:
      - host hardware (CPU cores/frequency, memory, accelerators),
      - FL model meta (flops per sample, model size),
      - and then adds stochastic variation (lognormal noise).

    This gives a more "plausible" time than a pure abstract distribution,
    while still remaining cheap enough for large-scale simulations.
    """

    def __init__(self, cfg: TimeModelConfig, context: TimeModelBuildContext) -> None:
        self.cfg = cfg
        self.context = context
        role_lower = context.role.lower()
        if role_lower not in {"training", "communication"}:
            raise ValueError(f"Invalid role for HardwareCalibratedTimeModel: {context.role}")
        self.role = role_lower

    def _base_time_for_training(self, hw: HostHardwareInfo) -> float:
        model_cfg = self.context.model_cfg

        # Fallback: if no model is configured, just use a constant or mean.
        if model_cfg is None:
            if self.cfg.constant is not None:
                return max(float(self.cfg.constant), 1e-4)
            return max(float(self.cfg.mean), 1e-4)

        # Approximate operation count: samples * epochs * flops_per_sample.
        samples = max(model_cfg.samples_per_client, 1)
        epochs = max(model_cfg.local_epochs, 1)
        flops_per_sample = max(model_cfg.flops_per_sample, 1e5)
        total_flops = float(samples * epochs) * flops_per_sample

        # Approximate compute capability:
        #   CPU: assume ~16 GFLOPs per core * GHz
        cpu_gflops = hw.cpu_cores * hw.cpu_freq_ghz * 16.0
        #   Accelerators: use provided TFLOPs.
        accel_gflops = hw.accelerator_tflops * 1000.0
        effective_gflops = max(cpu_gflops + accel_gflops, 1.0)

        base_time = total_flops / (effective_gflops * 1e9)  # seconds

        # Memory pressure: if working set is large vs available RAM, slow down.
        working_bytes = samples * model_cfg.input_dim * 4.0  # float32
        ram_avail_bytes = hw.mem_available_gb * (1024.0**3)
        if ram_avail_bytes > 0:
            ratio = working_bytes / ram_avail_bytes
            if ratio > 0.5:
                # Beyond 50% of available RAM, increase time.
                base_time *= 1.0 + (ratio - 0.5) * 2.0

        return max(base_time, 1e-4)

    def sample(self, client_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        size = len(client_ids)
        times = np.empty(size, dtype=float)

        if self.role == "training":
            for i, cid in enumerate(client_ids):
                # Prefer per-device hardware profile when available.
                if self.context.device_profiles and str(cid) in self.context.device_profiles:
                    hw = self.context.device_profiles[str(cid)].hardware
                else:
                    hw = self.context.hardware
                base = self._base_time_for_training(hw)
                times[i] = base
        else:
            # For communication, this model is no longer used; see BandwidthTimeModel.
            # Kept for backward compatibility as a simple constant.
            model_cfg = self.context.model_cfg
            if model_cfg is None:
                base = max(float(self.cfg.constant or self.cfg.mean), 1e-4)
            else:
                payload_bytes = float(max(model_cfg.model_size_bytes, 1))
                hw = self.context.hardware
                base_mbps = 10.0 * (2.0 if hw.has_gpu else 1.0)
                bandwidth_bps = base_mbps * 1e6 / 8.0
                base = max(payload_bytes / max(bandwidth_bps, 1e5), 1e-4)
            times.fill(base)

        # Per-client heterogeneity (lognormal noise).
        if self.cfg.sigma > 0.0 or self.cfg.mean != 0.0:
            noise = rng.lognormal(mean=self.cfg.mean, sigma=self.cfg.sigma, size=size)
            times *= noise

        return times


def per_client_factors(client_ids: np.ndarray) -> np.ndarray:
    """
    Per-client multiplicative factor to emulate heterogeneous devices.
    Stable across runs thanks to hashing the client id.
    """
    factors = np.empty(len(client_ids), dtype=float)
    for i, cid in enumerate(client_ids):
        h = hash(str(cid)) & 0xFFFFFFFF
        # Small lognormal variation per client (mean 1, about 30% std).
        rng = np.random.default_rng(h)
        factors[i] = rng.lognormal(mean=0.0, sigma=0.3)
    return factors


class BandwidthTimeModel(TimeModel):
    """
    Time model for communication based on per-device bandwidth.

    For each device:
      time ≈ payload_bytes / bandwidth,
    where bandwidth is a lognormal random variable around a configured mean.
    """

    def __init__(self, cfg: TimeModelConfig, context: TimeModelBuildContext) -> None:
        self.cfg = cfg
        self.context = context
        if context.model_cfg is None:
            raise ValueError("BandwidthTimeModel requires a model configuration")

    def sample(self, client_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        size = len(client_ids)
        model_cfg = self.context.model_cfg
        payload_bytes = float(max(model_cfg.model_size_bytes, 1))

        times = np.empty(size, dtype=float)

        # Global defaults for bandwidth when per-device values are not specified.
        global_mean_mbps = (
            float(self.cfg.default_bandwidth_mean_mbps)
            if self.cfg.default_bandwidth_mean_mbps is not None
            else 10.0
        )
        global_jitter = (
            float(self.cfg.default_bandwidth_jitter)
            if self.cfg.default_bandwidth_jitter is not None
            else 0.4
        )
        for i, cid in enumerate(client_ids):
            cid_str = str(cid)
            if self.context.device_profiles and cid_str in self.context.device_profiles:
                prof = self.context.device_profiles[cid_str]
                mean_mbps = prof.bandwidth_mean_mbps
                jitter = prof.bandwidth_jitter
            else:
                # Use global defaults when per-device values are not present.
                mean_mbps = global_mean_mbps
                jitter = global_jitter

            mean_mbps = max(mean_mbps, 0.1)
            jitter = max(jitter, 0.01)

            # Lognormal bandwidth process; mu chosen so that exp(mu) ≈ mean when jitter is small.
            mu = np.log(mean_mbps)
            sigma = jitter
            bw_mbps = float(rng.lognormal(mean=mu, sigma=sigma))

            bw_bps = bw_mbps * 1e6 / 8.0
            t = payload_bytes / max(bw_bps, 1e3)
            times[i] = max(t, 1e-6)

        return times

class MLTimeModel(TimeModel):
    """
    Time model based on a pre-trained ML regressor, trained offline on
    emulated hardware/model configurations.
    """

    def __init__(
        self,
        cfg: TimeModelConfig,
        context: TimeModelBuildContext,
        model_path: str | Path = "data/ml_time_model.npz",
    ) -> None:
        if context.model_cfg is None:
            raise ValueError("MLTimeModel requires a model configuration")
        self.cfg = cfg
        self.context = context
        self.model_cfg = context.model_cfg
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ML time model parameters not found at {self.model_path}. "
                f"Train it first with scripts/train_ml_time_model.py"
            )

        data = np.load(self.model_path, allow_pickle=True)
        feature_names = [str(x) for x in data["feature_names"].tolist()]

        if "model_type" in data and data["model_type"].item() == "mlp_v1":
            label_scale = (
                str(data["label_scale"].item()) if "label_scale" in data else "seconds"
            )
            self.params = MLTimeModelParams(
                model_type="mlp_v1",
                w=None,
                b=None,
                W1=data["W1"],
                b1=data["b1"],
                W2=data["W2"],
                b2=float(data["b2"]),
                mean=data["mean"],
                std=data["std"],
                feature_names=feature_names,
                label_scale=label_scale,
            )
        else:
            # Backward compatibility: linear model.
            self.params = MLTimeModelParams(
                model_type="linear",
                w=data["w"],
                b=float(data["b"]),
                W1=None,
                b1=None,
                W2=None,
                b2=None,
                mean=data["mean"],
                std=data["std"],
                feature_names=feature_names,
                label_scale="seconds",
            )

        # Sanity check: feature order.
        if list(self.params.feature_names) != FEATURE_NAMES:
            raise ValueError(
                f"ML time model feature mismatch.\n"
                f"Expected: {FEATURE_NAMES}\n"
                f"Found in file: {self.params.feature_names}"
            )

    def sample(self, client_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        size = len(client_ids)
        times = np.empty(size, dtype=float)

        for i, cid in enumerate(client_ids):
            if (
                self.context.device_profiles is not None
                and str(cid) in self.context.device_profiles
            ):
                hw = self.context.device_profiles[str(cid)].hardware
            else:
                hw = self.context.hardware
            # Allow per-device batch size override when available.
            if (
                self.context.device_batch_sizes is not None
                and str(cid) in self.context.device_batch_sizes
                and self.model_cfg is not None
            ):
                bs = self.context.device_batch_sizes[str(cid)]
                local_cfg = ModelConfig(
                    kind=self.model_cfg.kind,
                    input_dim=self.model_cfg.input_dim,
                    num_classes=self.model_cfg.num_classes,
                    samples_per_client=self.model_cfg.samples_per_client,
                    test_samples=self.model_cfg.test_samples,
                    local_epochs=self.model_cfg.local_epochs,
                    batch_size=bs,
                    learning_rate=self.model_cfg.learning_rate,
                    eval_every_rounds=self.model_cfg.eval_every_rounds,
                    flops_per_sample=self.model_cfg.flops_per_sample,
                    model_size_bytes=self.model_cfg.model_size_bytes,
                )
            else:
                local_cfg = self.model_cfg

            x = build_feature_vector(hw, local_cfg)
            x_std = (x - self.params.mean) / self.params.std
            if self.params.model_type == "mlp_v1":
                assert self.params.W1 is not None and self.params.W2 is not None
                z1 = x_std @ self.params.W1 + self.params.b1
                h1 = np.maximum(z1, 0.0)
                pred = float(h1 @ self.params.W2 + self.params.b2)
            else:
                assert self.params.w is not None and self.params.b is not None
                pred = float(self.params.w @ x_std + self.params.b)

            if self.params.label_scale == "log10_seconds":
                base = 10.0**pred
            else:
                base = pred

            times[i] = max(base, 1e-6)

        # Optional extra stochastic noise on top of ML prediction.
        if self.cfg.sigma > 0.0 or self.cfg.mean != 0.0:
            noise = rng.lognormal(mean=self.cfg.mean, sigma=self.cfg.sigma, size=size)
            times *= noise

        return times


def build_time_model_with_context(cfg: TimeModelConfig, context: TimeModelBuildContext) -> TimeModel:
    """
    Build a time model that may depend on hardware and FL model metadata.
    """
    kind = cfg.kind.lower()
    if kind == "trace":
        return TraceTimeModel(cfg)
    if kind in {"lognormal", "exponential", "normal", "constant"}:
        return DistributionTimeModel(cfg)
    if kind == "hardware":
        # For training this remains HardwareCalibratedTimeModel; for communication, prefer bandwidth model.
        if context.role == "communication":
            return BandwidthTimeModel(cfg, context)
        return HardwareCalibratedTimeModel(cfg, context)
    if kind == "bandwidth":
        return BandwidthTimeModel(cfg, context)
    if kind == "ml":
        return MLTimeModel(cfg, context)
    raise ValueError(f"Unsupported time model kind: {cfg.kind}")
