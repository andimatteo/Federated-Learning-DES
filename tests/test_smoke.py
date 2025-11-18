from pathlib import Path

from fl_sim.config import load_config
from fl_sim.simulation import FederatedSimulator


def test_smoke_example_config(tmp_path: Path) -> None:
    cfg = load_config("configs/example.toml")
    sim = FederatedSimulator(cfg)
    sim.run()

    out_dir = tmp_path / "sim_output"
    sim.export(base_dir=out_dir)

    assert (out_dir / "client_times.csv").exists()
    assert (out_dir / "summary.json").exists()


