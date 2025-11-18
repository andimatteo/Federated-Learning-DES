from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from .config import SimulationConfig


class ClientSelector(ABC):
    """
    Strategy for selecting participating clients each round.
    """

    @abstractmethod
    def select(self, all_client_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Return an array of selected client ids (subset of all_client_ids).
        """


class AllClientsSelector(ClientSelector):
    def select(self, all_client_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return all_client_ids


class RandomSubsetSelector(ClientSelector):
    def __init__(self, sim_cfg: SimulationConfig) -> None:
        self.sim_cfg = sim_cfg

    def select(self, all_client_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(all_client_ids)
        if n == 0:
            return all_client_ids

        if self.sim_cfg.clients_per_round is not None:
            k = max(1, min(self.sim_cfg.clients_per_round, n))
        else:
            rate = float(self.sim_cfg.participation_rate)
            k = max(1, min(int(round(rate * n)), n))

        idx = rng.choice(n, size=k, replace=False)
        return all_client_ids[idx]


def build_selector(sim_cfg: SimulationConfig) -> ClientSelector:
    """
    Build a simple selection strategy from simulation config.

    For now we expose:
      - random subset (default)
      - all clients (when participation_rate >= 0.999 and clients_per_round is None)
    """
    if sim_cfg.clients_per_round is None and sim_cfg.participation_rate >= 0.999:
        return AllClientsSelector()
    return RandomSubsetSelector(sim_cfg)


