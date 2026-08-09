"""
Trainers sequenciais para RUL.

- SequenceMLPTrainer: achata a janela (T×F) e treina MLP (sempre disponível).
- LSTMTrainer: LSTM PyTorch se torch instalado; senão, erro explícito.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class SequenceModelResult:
    name: str
    algorithm: str
    y_pred: np.ndarray
    loss_curve: list[float] | None
    n_epochs: int
    predictor: Any


class SequenceMLPTrainer:
    """Baseline sequencial sem torch: flatten (T, F) → MLPRegressor."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        max_iter: int = 200,
        seed: int = 42,
    ) -> None:
        self._hidden = hidden_layer_sizes
        self._max_iter = max_iter
        self._seed = seed

    def train_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> SequenceModelResult:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        n_tr, t, f = X_train.shape
        Xtr = X_train.reshape(n_tr, t * f)
        Xte = X_test.reshape(X_test.shape[0], t * f)

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=self._hidden,
                        max_iter=self._max_iter,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=12,
                        random_state=self._seed,
                        learning_rate_init=1e-3,
                    ),
                ),
            ]
        )
        pipe.fit(Xtr, y_train)
        y_pred = np.clip(pipe.predict(Xte), 0, None)

        loss_curve = None
        n_epochs = 0
        try:
            mlp = pipe.named_steps["mlp"]
            loss_curve = list(mlp.loss_curve_) if hasattr(mlp, "loss_curve_") else None
            n_epochs = int(getattr(mlp, "n_iter_", 0) or 0)
        except Exception:
            pass

        return SequenceModelResult(
            name="sequence_mlp",
            algorithm=f"SeqMLP{list(self._hidden)} T={t}",
            y_pred=y_pred,
            loss_curve=loss_curve,
            n_epochs=n_epochs,
            predictor=pipe,
        )


if HAS_TORCH:

    class _LSTMNet(nn.Module):
        def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 1, dropout: float = 0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)


class LSTMTrainer:
    """LSTM com PyTorch (opcional)."""

    def __init__(
        self,
        hidden: int = 64,
        n_layers: int = 1,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self._hidden = hidden
        self._n_layers = n_layers
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._seed = seed

    def train_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> SequenceModelResult:
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch não está instalado. Use Sequence MLP ou instale torch "
                "(requirements/ml.txt). No Streamlit Cloud, prefira Sequence MLP."
            )

        torch.manual_seed(self._seed)
        device = torch.device("cpu")
        n_features = X_train.shape[-1]

        model = _LSTMNet(n_features, hidden=self._hidden, n_layers=self._n_layers).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()

        xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(xt, yt),
            batch_size=min(self._batch_size, max(1, len(xt))),
            shuffle=True,
        )

        loss_curve: list[float] = []
        model.train()
        for _ in range(self._epochs):
            total, n = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(xb)
                n += len(xb)
            loss_curve.append(total / max(n, 1))

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        y_pred = np.clip(y_pred, 0, None)

        return SequenceModelResult(
            name="lstm",
            algorithm=f"LSTM(h={self._hidden},L={self._n_layers})",
            y_pred=y_pred,
            loss_curve=loss_curve,
            n_epochs=self._epochs,
            predictor=model,
        )
