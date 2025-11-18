from __future__ import annotations

from pathlib import Path
from typing import List
import csv
import json

import numpy as np

from .config import Config
from .selection import build_selector
from .time_models import build_time_model
from .metrics import summarize_per_client, histogram_per_client
from .types import ClientRoundRecord


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
        self.training_model = build_time_model(cfg.training_time)
        self.communication_model = build_time_model(cfg.communication_time)

        self.records: List[ClientRoundRecord] = []
        self.current_time: float = 0.0

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

        train_times = self.training_model.sample(client_ids, self.rng)
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
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "simulation_rounds": self.cfg.simulation.rounds,
                    "num_clients": self.cfg.simulation.num_clients,
                    "per_client": summary,
                },
                f,
                indent=2,
            )

    def _export_histograms_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        hist = histogram_per_client(self.records, bins=self.cfg.output.histogram_bins)
        with path.open("w", encoding="utf-8") as f:
            json.dump(hist, f)

