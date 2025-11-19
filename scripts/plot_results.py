from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_client_times(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "round": int(row["round"]),
                    "client_id": row["client_id"],
                    "training_time": float(row["training_time"]),
                    "communication_time": float(row["communication_time"]),
                    "total_time": float(row["total_time"]),
                }
            )
    return rows


def load_summary(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_global_histograms(rows, out_dir: Path) -> Path:
    train = [r["training_time"] for r in rows]
    comm = [r["communication_time"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(train, bins=40, alpha=0.6, label="training")
    ax.hist(comm, bins=40, alpha=0.6, label="communication")
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.set_title("Global distribution of training and communication times")
    ax.legend()
    out_path = out_dir / "plot_hist_training_comm_all.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_avg_per_client(summary: Dict, out_dir: Path) -> Path:
    per_client = summary.get("per_client", {})
    if not per_client:
        raise ValueError("summary.json has no 'per_client' section")

    # Build list of (client_id, train_mean, comm_mean) and sort by total mean ascending.
    entries = []
    for cid, stats in per_client.items():
        t_mean = float(stats.get("train_mean", 0.0))
        c_mean = float(stats.get("comm_mean", 0.0))
        total = t_mean + c_mean
        entries.append((cid, t_mean, c_mean, total))

    entries.sort(key=lambda x: x[3])  # sort by total mean time

    client_ids = [cid for cid, _, _, _ in entries]
    train_means = [t for _, t, _, _ in entries]
    comm_means = [c for _, _, c, _ in entries]

    x = range(len(client_ids))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, train_means, label="train", color="C0")
    ax.bar(
        x,
        comm_means,
        bottom=train_means,
        label="communication",
        color="C3",
    )
    ax.set_xlabel("client (sorted by total mean time)")
    ax.set_ylabel("mean time per round")
    ax.set_title("Average training + communication time per client (sorted)")
    ax.set_xticks(x)
    ax.set_xticklabels(client_ids, rotation=90, fontsize=6)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "plot_avg_time_per_client.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_model_metrics(summary: Dict, out_dir: Path) -> Path | None:
    metrics = summary.get("model_metrics")
    if not metrics:
        return None

    rounds = [m["round"] for m in metrics]
    acc = [m.get("accuracy", 0.0) for m in metrics]
    f1 = [m.get("macro_f1", 0.0) for m in metrics]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, acc, label="accuracy", marker="o")
    ax.plot(rounds, f1, label="macro F1", marker="x")
    ax.set_xlabel("round")
    ax.set_ylabel("metric")
    ax.set_title("Model accuracy and macro-F1 over rounds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "plot_model_metrics.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot histograms and metrics from simulation outputs.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="output",
        help="Directory containing client_times.csv and summary.json (default: output).",
    )
    args = parser.parse_args()

    out_dir = Path(args.dir)
    csv_path = out_dir / "client_times.csv"
    summary_path = out_dir / "summary.json"

    if not csv_path.exists():
        raise SystemExit(f"Missing CSV results file: {csv_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")

    rows = load_client_times(csv_path)
    summary = load_summary(summary_path)

    hist_path = plot_global_histograms(rows, out_dir)
    avg_path = plot_avg_per_client(summary, out_dir)
    metrics_path = plot_model_metrics(summary, out_dir)

    print("Generated plots:")
    print(f" - {hist_path}")
    print(f" - {avg_path}")
    if metrics_path:
        print(f" - {metrics_path}")


if __name__ == "__main__":
    main()
