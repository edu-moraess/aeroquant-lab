"""
UpdateDigitalTwin — o caso de uso central da Fase 5. Recebe UMA leitura por
vez (simulando streaming/tempo real), atualiza o baseline online da frota,
recalcula Health Index e RUL, detecta anomalia por desvio do HI em relação
à própria tendência recente, e persiste o snapshot.
"""
from __future__ import annotations

import numpy as np

from aeroquant.digital_twin.application.ports import (
    DigitalTwinRepository,
    FleetBaselineTracker,
    HealthIndexEstimator,
    RULEstimator,
)
from aeroquant.digital_twin.domain.entities import DigitalTwinSnapshot


class UpdateDigitalTwin:
    def __init__(
        self,
        baseline_tracker: FleetBaselineTracker,
        hi_estimator: HealthIndexEstimator,
        rul_estimator: RULEstimator,
        repository: DigitalTwinRepository,
        anomaly_z_threshold: float = 3.0,
        anomaly_min_history: int = 6,
        healthy_window_cycles: int = 20,
    ) -> None:
        self._baseline = baseline_tracker
        self._hi_estimator = hi_estimator
        self._rul_estimator = rul_estimator
        self._repository = repository
        self._anomaly_z_threshold = anomaly_z_threshold
        self._anomaly_min_history = anomaly_min_history
        # BUG REAL ENCONTRADO E CORRIGIDO NESTA SESSÃO: atualizar o baseline
        # com TODAS as leituras (inclusive já degradadas) faz a "média
        # saudável" perseguir o próprio sinal de degradação — o z-score (e
        # portanto o HI) volta a cair perto do fim de vida, exatamente onde
        # deveria estar mais alto. Corrigido travando a atualização do
        # baseline a uma janela inicial de vida assumida-saudável, coerente
        # com a mesma premissa já usada no cap do rótulo de RUL (Fase 3).
        self._healthy_window_cycles = healthy_window_cycles

    def ingest(
        self,
        unit_id: str,
        cycle: int,
        operating_condition: int,
        sensor_values: dict[str, float],
        failure_threshold: float = 1.0,
    ) -> DigitalTwinSnapshot:
        # 1) Aprendizado online, mas só durante a janela inicial assumida-
        #    saudável: fora dela, o baseline fica congelado (deixa de ser
        #    contaminado pela própria degradação que está sendo medida).
        if cycle <= self._healthy_window_cycles:
            self._baseline.update(operating_condition, sensor_values)
        stats = self._baseline.stats(operating_condition)

        health_index, hi_uncertainty = self._hi_estimator.estimate(sensor_values, stats)

        state = self._repository.load(unit_id)
        is_anomaly, reason = self._detect_anomaly(state.history, health_index)

        rul = self._rul_estimator.estimate(state.history + [], failure_threshold)
        # RUL usa o histórico ANTES deste ponto para extrapolar; o ponto atual
        # entra no próximo ingest. Se quisermos incluir o ponto atual na regressão,
        # fazemos isso explicitamente após montar o snapshot (ver abaixo).

        snapshot = DigitalTwinSnapshot(
            cycle=cycle,
            health_index=health_index,
            health_index_uncertainty=hi_uncertainty,
            rul=rul,
            is_anomaly=is_anomaly,
            anomaly_reason=reason,
        )
        # Recalcula RUL agora incluindo o snapshot atual (dado mais recente
        # disponível) — reordenado para simplicidade e clareza de teste.
        rul_with_current = self._rul_estimator.estimate(state.history + [snapshot], failure_threshold)
        snapshot = DigitalTwinSnapshot(
            cycle=cycle,
            health_index=health_index,
            health_index_uncertainty=hi_uncertainty,
            rul=rul_with_current,
            is_anomaly=is_anomaly,
            anomaly_reason=reason,
        )

        state.append(snapshot)
        self._repository.save(state)
        return snapshot

    def _detect_anomaly(self, history: list[DigitalTwinSnapshot], new_hi: float) -> tuple[bool, str | None]:
        if len(history) < self._anomaly_min_history:
            return False, None

        recent = history[-self._anomaly_min_history :]
        increments = np.diff([s.health_index for s in recent] + [new_hi])
        rolling_std = float(np.std(increments[:-1])) or 1e-6
        rolling_mean = float(np.mean(increments[:-1]))
        last_increment = increments[-1]

        z = (last_increment - rolling_mean) / rolling_std
        if abs(z) > self._anomaly_z_threshold:
            return True, f"salto de HI com z-score {z:.2f} (limiar {self._anomaly_z_threshold})"
        return False, None
