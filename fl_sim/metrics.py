from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .types import ClientRoundRecord


def summarize_per_client(
    records: Iterable[ClientRoundRecord],
) -> Dict[str, Dict[str, float]]:
    """
    Compute simple statistics per client: mean, std, percentiles.
    """
    by_client: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for r in records:
        by_client[r.client_id].append((r.training_time, r.communication_time, r.total_time))

    summary: Dict[str, Dict[str, float]] = {}
    for cid, triplets in by_client.items():
        arr = np.asarray(triplets, dtype=float)  # shape (n, 3)
        train = arr[:, 0]
        comm = arr[:, 1]
        total = arr[:, 2]

        summary[cid] = {
            "rounds": float(len(arr)),
            "train_mean": float(train.mean()),
            "train_std": float(train.std(ddof=0)),
            "comm_mean": float(comm.mean()),
            "comm_std": float(comm.std(ddof=0)),
            "total_mean": float(total.mean()),
            "total_p50": float(np.percentile(total, 50)),
            "total_p90": float(np.percentile(total, 90)),
            "total_p99": float(np.percentile(total, 99)),
        }
    return summary


def histogram_per_client(
    records: Iterable[ClientRoundRecord],
    bins: int = 20,
) -> Dict[str, Dict[str, object]]:
    """
    Build histograms per client for training and communication times.

    Returns a dict:
      {client_id: { "bin_edges": [...], "train_counts": [...], "comm_counts": [...] } }
    """
    by_client_train: Dict[str, List[float]] = defaultdict(list)
    by_client_comm: Dict[str, List[float]] = defaultdict(list)

    for r in records:
        by_client_train[r.client_id].append(r.training_time)
        by_client_comm[r.client_id].append(r.communication_time)

    out: Dict[str, Dict[str, object]] = {}
    for cid, train_vals in by_client_train.items():
        train_arr = np.asarray(train_vals, dtype=float)
        comm_arr = np.asarray(by_client_comm[cid], dtype=float)

        # Use joint range for consistent stacked bars
        tmin = min(train_arr.min(initial=0.0), comm_arr.min(initial=0.0))
        tmax = max(train_arr.max(initial=0.0), comm_arr.max(initial=0.0))
        if tmax <= tmin:
            tmax = tmin + 1.0

        hist_train, bin_edges = np.histogram(train_arr, bins=bins, range=(tmin, tmax))
        hist_comm, _ = np.histogram(comm_arr, bins=bin_edges)

        out[cid] = {
            "bin_edges": bin_edges.tolist(),
            "train_counts": hist_train.astype(int).tolist(),
            "comm_counts": hist_comm.astype(int).tolist(),
        }
    return out

