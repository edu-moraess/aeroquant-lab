"""
TrainAndCompareModels — o use case central da Fase 6. Treina cada modelo
candidato no split de treino (por unidade), avalia no split de teste
(unidades nunca vistas), e monta uma tabela comparativa de métricas.

Comparação contra o baseline da Fase 5 é feita SEPARADAMENTE (ver
scripts/demo_ml_comparison.py) porque o baseline opera em modo streaming
(usa histórico incremental por unidade), diferente dos modelos aqui, que
são batch (preveem RUL a partir de um snapshot de features). Misturar os
dois modos no mesmo use case criaria uma interface artificial — mais
honesto manter os dois fluxos de avaliação separados e comparar só o
resultado final (RMSE/MAE/NASA score), que é o que realmente importa.
"""
from __future__ import annotations

import pandas as pd

from aeroquant.ml.application.ports import ModelTrainer
from aeroquant.ml.domain.entities import ComparisonResult
from aeroquant.ml.infrastructure.evaluation.metrics import evaluate


class TrainAndCompareModels:
    def __init__(self, trainers: dict[str, ModelTrainer]) -> None:
        self._trainers = trainers

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "rul",
        baseline_name: str = "linear_extrapolation (Fase 5)",
    ) -> ComparisonResult:
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]

        result = ComparisonResult(baseline_name=baseline_name)
        for name, trainer in self._trainers.items():
            model = trainer.train(X_train, y_train)
            y_pred = trainer.predict(model, X_test)
            interval = trainer.predict_interval(model, X_test)
            lower, upper = interval if interval is not None else (None, None)
            result.results[name] = evaluate(y_test.to_numpy(), y_pred, lower, upper)

        return result