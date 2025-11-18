from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .simulation import FederatedSimulator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DES simulator for federated learning (training + communication times)."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to TOML or JSON configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Optional override for output directory.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    sim = FederatedSimulator(cfg)
    sim.run()
    sim.export(base_dir=args.output_dir)


if __name__ == "__main__":
    main()


