from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .config import ModelConfig


@dataclass
class ClientDataset:
    x: np.ndarray  # shape (n, d)
    y: np.ndarray  # shape (n,)


class DummyLogisticRegressionModel:
    """
    Very simple logistic-regression-like model with FedAvg-style aggregation.

    - Linear layer + softmax (cross-entropy loss).
    - SGD updates on each client; parameters are averaged on server.
    - Synthetic dataset (Gaussian features, random labels).
    """

    def __init__(self, cfg: ModelConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        d = cfg.input_dim
        c = cfg.num_classes
        # Initialize weights and bias small.
        self.w = 0.01 * rng.standard_normal(size=(d, c))
        self.b = np.zeros((c,), dtype=float)

        # Prepare synthetic per-client datasets and a global test set.
        self.client_data: Dict[str, ClientDataset] = {}
        self.test_data: ClientDataset = self._make_dataset(cfg.test_samples)

    def _make_dataset(self, n: int) -> ClientDataset:
        d = self.cfg.input_dim
        c = self.cfg.num_classes
        x = self.rng.standard_normal(size=(n, d))
        logits = self.rng.standard_normal(size=(n, c))
        y = np.argmax(logits, axis=1)
        return ClientDataset(x=x, y=y)

    def ensure_client_dataset(self, client_id: str) -> ClientDataset:
        if client_id not in self.client_data:
            ds = self._make_dataset(self.cfg.samples_per_client)
            self.client_data[client_id] = ds
        return self.client_data[client_id]

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z_max = np.max(z, axis=1, keepdims=True)
        e = np.exp(z - z_max)
        return e / np.sum(e, axis=1, keepdims=True)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def _loss_and_grad(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        logits = self._forward(x)
        probs = self._softmax(logits)
        n = x.shape[0]
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), y] = 1.0

        # Cross-entropy loss
        eps = 1e-9
        loss = -np.mean(np.log(probs[np.arange(n), y] + eps))

        # Gradients
        grad_logits = (probs - one_hot) / n
        grad_w = x.T @ grad_logits
        grad_b = np.sum(grad_logits, axis=0)
        return loss, grad_w, grad_b

    def local_train(self, client_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform local training on a single client and return updated parameters.
        """
        ds = self.ensure_client_dataset(client_id)
        x, y = ds.x, ds.y
        w = self.w.copy()
        b = self.b.copy()

        lr = self.cfg.learning_rate
        epochs = max(self.cfg.local_epochs, 1)

        for _ in range(epochs):
            loss, grad_w, grad_b = self._loss_and_grad(x, y)
            w -= lr * grad_w
            b -= lr * grad_b

        return w, b

    def aggregate(self, client_params: Iterable[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        FedAvg-style aggregation: simple average of weights and biases.
        """
        ws: List[np.ndarray] = []
        bs: List[np.ndarray] = []
        for w, b in client_params:
            ws.append(w)
            bs.append(b)
        if not ws:
            return
        self.w = np.mean(ws, axis=0)
        self.b = np.mean(bs, axis=0)

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the global model on the synthetic test set.

        Returns metrics including accuracy and macro-F1.
        """
        x = self.test_data.x
        y_true = self.test_data.y

        logits = self._forward(x)
        y_pred = np.argmax(logits, axis=1)

        accuracy = float(np.mean(y_pred == y_true))

        # Macro F1
        num_classes = self.cfg.num_classes
        f1s: List[float] = []
        eps = 1e-9
        for cls in range(num_classes):
            tp = float(np.sum((y_true == cls) & (y_pred == cls)))
            fp = float(np.sum((y_true != cls) & (y_pred == cls)))
            fn = float(np.sum((y_true == cls) & (y_pred != cls)))
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s))

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }


def build_model(cfg: ModelConfig | None, rng: np.random.Generator) -> DummyLogisticRegressionModel | None:
    """
    Factory for FL models.

    Currently supports:
      - kind = 'dummy_logreg'
    """
    if cfg is None or not getattr(cfg, "enabled", True):
        return None
    if cfg.kind == "dummy_logreg":
        return DummyLogisticRegressionModel(cfg, rng)
    raise ValueError(f"Unsupported model kind: {cfg.kind}")

