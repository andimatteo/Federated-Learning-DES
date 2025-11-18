from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClientRoundRecord:
    round_index: int
    client_id: str
    training_time: float
    communication_time: float
    total_time: float


