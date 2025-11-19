from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import csv
import json
import time

import numpy as np

from .config import Config
from .selection import build_selector
from .time_models import (
    build_time_model_with_context,
    DistributionTimeModel,
    TimeModelBuildContext,
)
from .metrics import summarize_per_client, histogram_per_client
from .types import ClientRoundRecord
from .hardware import collect_host_hardware, HostHardwareInfo, DeviceProfile
from .model import build_model, DummyLogisticRegressionModel


class FederatedSimulator:
    """
    Simple discrete-event style simulator for synchronous FL rounds.

    Each round:
      - select a subset of clients
      - sample training and communication times
      - advance the global time to the completion of the slowest client
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.simulation.seed)
        self.all_client_ids = np.array(
            [str(i) for i in range(cfg.simulation.num_clients)],
            dtype=object,
        )
        self.selector = build_selector(cfg.simulation)

        # Optional real (or synthetic) FL model to compute accuracy/F1.
        self.model: Optional[DummyLogisticRegressionModel] = build_model(
            cfg.model,
            self.rng,
        )
        self.measure_training_time: bool = bool(
            cfg.model and cfg.model.enabled and cfg.model.measure_training_time
        )

        # Host hardware info for hardware-aware time models.
        self.hardware: HostHardwareInfo = collect_host_hardware()

        # Per-device profiles (hardware + bandwidth), with sane defaults.
        self.device_profiles: Dict[str, DeviceProfile] = {}
        self.device_batch_sizes: Dict[str, int] = {}
        default_bw_mean = 10.0
        default_bw_jitter = 0.4
        global_batch_size = int(cfg.model.batch_size) if cfg.model and cfg.model.batch_size > 0 else 128
        for cid in self.all_client_ids:
            cid_str = str(cid)
            dev_cfg = (cfg.devices or {}).get(cid_str) if cfg.devices else None

            # Determine has_gpu for this device.
            if dev_cfg and dev_cfg.has_gpu is not None:
                dev_has_gpu = bool(dev_cfg.has_gpu)
            else:
                # Fallback: infer from accelerator_tflops or host.
                acc_tflops = (
                    dev_cfg.accelerator_tflops
                    if dev_cfg and dev_cfg.accelerator_tflops is not None
                    else self.hardware.accelerator_tflops
                )
                dev_has_gpu = acc_tflops > 0.0

            hw = HostHardwareInfo(
                cpu_cores=int(
                    dev_cfg.cpu_cores if dev_cfg and dev_cfg.cpu_cores is not None else self.hardware.cpu_cores
                ),
                cpu_freq_ghz=float(
                    dev_cfg.cpu_freq_ghz
                    if dev_cfg and dev_cfg.cpu_freq_ghz is not None
                    else self.hardware.cpu_freq_ghz
                ),
                mem_total_gb=float(
                    dev_cfg.mem_total_gb
                    if dev_cfg and dev_cfg.mem_total_gb is not None
                    else self.hardware.mem_total_gb
                ),
                mem_available_gb=float(
                    dev_cfg.mem_available_gb
                    if dev_cfg and dev_cfg.mem_available_gb is not None
                    else self.hardware.mem_available_gb
                ),
                accelerator_tflops=float(
                    dev_cfg.accelerator_tflops
                    if dev_cfg and dev_cfg.accelerator_tflops is not None
                    else self.hardware.accelerator_tflops
                ),
                has_gpu=dev_has_gpu,
            )
            bw_mean = (
                float(dev_cfg.bandwidth_mean_mbps)
                if dev_cfg and dev_cfg.bandwidth_mean_mbps is not None
                else default_bw_mean
            )
            bw_jitter = (
                float(dev_cfg.bandwidth_jitter)
                if dev_cfg and dev_cfg.bandwidth_jitter is not None
                else default_bw_jitter
            )
            self.device_profiles[cid_str] = DeviceProfile(
                hardware=hw,
                bandwidth_mean_mbps=bw_mean,
                bandwidth_jitter=bw_jitter,
            )
            # Batch size override per device (used by ML time model features).
            if dev_cfg and dev_cfg.batch_size is not None and dev_cfg.batch_size > 0:
                self.device_batch_sizes[cid_str] = int(dev_cfg.batch_size)
            else:
                self.device_batch_sizes[cid_str] = global_batch_size

        # Build time models; if kind does not require context, we fall back to pure
        # distribution-based models.
        try:
            self.training_model = build_time_model_with_context(
                cfg.training_time,
                TimeModelBuildContext(
                    sim_cfg=cfg.simulation,
                    model_cfg=cfg.model,
                    hardware=self.hardware,
                    device_profiles=self.device_profiles,
                    device_batch_sizes=self.device_batch_sizes,
                    role="training",
                ),
            )
        except ValueError:
            # Use purely statistical model.
            self.training_model = DistributionTimeModel(cfg.training_time)

        try:
            self.communication_model = build_time_model_with_context(
                cfg.communication_time,
                TimeModelBuildContext(
                    sim_cfg=cfg.simulation,
                    model_cfg=cfg.model,
                    hardware=self.hardware,
                    device_profiles=self.device_profiles,
                    device_batch_sizes=self.device_batch_sizes,
                    role="communication",
                ),
            )
        except ValueError:
            self.communication_model = DistributionTimeModel(cfg.communication_time)

        self.records: List[ClientRoundRecord] = []
        self.current_time: float = 0.0
        self.model_metrics: List[Dict[str, float]] = []

    def run(self) -> None:
        """
        Run the full simulation.
        """
        for r in range(self.cfg.simulation.rounds):
            self._run_round(r)

    def _run_round(self, round_index: int) -> None:
        client_ids = self.selector.select(self.all_client_ids, self.rng)
        if len(client_ids) == 0:
            return

        # Compute training times, either via model-based time model or via
        # real wall-clock measurement of dummy training (when enabled).
        train_times: np.ndarray
        client_updates: List[tuple[np.ndarray, np.ndarray]] = []

        if self.model is not None and self.measure_training_time:
            train_times = np.empty(len(client_ids), dtype=float)
            for i, cid in enumerate(client_ids):
                t0 = time.perf_counter()
                w, b = self.model.local_train(str(cid))
                elapsed = time.perf_counter() - t0
                train_times[i] = elapsed
                client_updates.append((w, b))
            # Aggregate updates once per round.
            self.model.aggregate(client_updates)
            # Evaluate every eval_every_rounds.
            eval_every = max(self.cfg.model.eval_every_rounds if self.cfg.model else 1, 1)
            if (round_index + 1) % eval_every == 0:
                metrics = self.model.evaluate()
                metrics_with_round = {"round": round_index + 1, **metrics}
                self.model_metrics.append(metrics_with_round)
        else:
            # Use statistical/ML time model for training times.
            train_times = self.training_model.sample(client_ids, self.rng)
            # Optionally also update the learning model (for accuracy/F1 only).
            if self.model is not None:
                for cid in client_ids:
                    w, b = self.model.local_train(str(cid))
                    client_updates.append((w, b))
                self.model.aggregate(client_updates)
                eval_every = max(self.cfg.model.eval_every_rounds if self.cfg.model else 1, 1)
                if (round_index + 1) % eval_every == 0:
                    metrics = self.model.evaluate()
                    metrics_with_round = {"round": round_index + 1, **metrics}
                    self.model_metrics.append(metrics_with_round)

        comm_times = self.communication_model.sample(client_ids, self.rng)

        total_times = train_times + comm_times
        round_duration = float(np.max(total_times))

        # Global time jumps directly to the end of the round.
        self.current_time += round_duration

        for cid, t_train, t_comm, t_total in zip(
            client_ids, train_times, comm_times, total_times, strict=True
        ):
            self.records.append(
                ClientRoundRecord(
                    round_index=round_index,
                    client_id=str(cid),
                    training_time=float(t_train),
                    communication_time=float(t_comm),
                    total_time=float(t_total),
                )
            )

        # Optional: update the learning model and record metrics.
        if self.model is not None:
            client_updates = []
            for cid in client_ids:
                w, b = self.model.local_train(str(cid))
                client_updates.append((w, b))
            self.model.aggregate(client_updates)
            # Evaluate every eval_every_rounds.
            eval_every = max(self.cfg.model.eval_every_rounds if self.cfg.model else 1, 1)
            if (round_index + 1) % eval_every == 0:
                metrics = self.model.evaluate()
                metrics_with_round = {"round": round_index + 1, **metrics}
                self.model_metrics.append(metrics_with_round)

    # ---- Export utilities -------------------------------------------------

    def export(self, base_dir: str | Path | None = None) -> None:
        """
        Export trace and statistics according to the output config.
        """
        out_cfg = self.cfg.output
        base = Path(base_dir) if base_dir is not None else Path(out_cfg.directory)
        base.mkdir(parents=True, exist_ok=True)

        if out_cfg.write_traces_csv:
            self._export_traces_csv(base / "client_times.csv")

        if out_cfg.write_summary_json:
            self._export_summary_json(base / "summary.json")

        if out_cfg.write_histograms_json:
            self._export_histograms_json(base / "histograms.json")

    def _export_traces_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["round", "client_id", "training_time", "communication_time", "total_time"]
            )
            for r in self.records:
                writer.writerow(
                    [
                        r.round_index,
                        r.client_id,
                        f"{r.training_time:.9f}",
                        f"{r.communication_time:.9f}",
                        f"{r.total_time:.9f}",
                    ]
                )

    def _export_summary_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = summarize_per_client(self.records)
        model_metrics = self.model_metrics if self.model_metrics else None
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "simulation_rounds": self.cfg.simulation.rounds,
                    "num_clients": self.cfg.simulation.num_clients,
                    "per_client": summary,
                    "model_metrics": model_metrics,
                },
                f,
                indent=2,
            )

    def _export_histograms_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        hist = histogram_per_client(self.records, bins=self.cfg.output.histogram_bins)
        with path.open("w", encoding="utf-8") as f:
            json.dump(hist, f)
