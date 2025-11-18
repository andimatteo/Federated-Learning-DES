from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

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
    client_ids: List[str] = sorted(per_client.keys(), key=lambda x: int(x))
    train_means = [per_client[c]["train_mean"] for c in client_ids]
    comm_means = [per_client[c]["comm_mean"] for c in client_ids]

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
    ax.set_xlabel("client id")
    ax.set_ylabel("mean time per round")
    ax.set_title("Average training + communication time per client")
    ax.set_xticks(x)
    ax.set_xticklabels(client_ids, rotation=90, fontsize=6)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "plot_avg_time_per_client.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    out_dir = Path("output")
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

    print("Generated plots:")
    print(f" - {hist_path}")
    print(f" - {avg_path}")


if __name__ == "__main__":
    main()


