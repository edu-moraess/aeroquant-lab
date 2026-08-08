from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from aeroquant.ml.domain.entities import TrainedModel


class ModelTrainer(Protocol):
    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        ...

    def predict(self, model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_interval(
        self, model: TrainedModel, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Retorna (lower, upper) para 90% de confiança, ou None se o
        trainer não suportar quantificação de incerteza."""
        ...