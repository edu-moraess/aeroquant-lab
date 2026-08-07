"""
StochasticSensorGenerator — implementação concreta do port DataGenerator.

Modelos estocásticos usados (e por quê):

1. Degradação: processo Gamma (incrementos i.i.d. Gamma(shape, scale),
   cumulativos, normalizados para atingir 1.0 no ciclo de falha). É o
   modelo padrão em literatura de confiabilidade para processos de
   desgaste monotônico (ex.: fadiga, corrosão) — ao contrário de um
   passeio aleatório gaussiano, o processo Gamma é sempre não-decrescente,
   o que é fisicamente correto para "quanto dano acumulado" (dano não
   se recupera espontaneamente).

2. Deriva de sensor (drift): tendência linear lenta, independente da
   saúde do componente — representa deriva de calibração do sensor,
   não degradação do componente monitorado. Modelada separadamente de
   propósito (é um dos pontos que a Fase 1 identificou como precisando
   ser distinguível para quantificar o gap sintético-real por fonte de
   erro, não só o erro agregado).

3. Ruído de medição: gaussiano branco, i.i.d. por leitura — modelo padrão
   para ruído de instrumentação quando não há razão para assumir
   correlação temporal no próprio ruído (a correlação temporal do sinal
   vem da degradação e da deriva, não do ruído).

4. Falha abrupta: processo de Bernoulli por ciclo (aproximação discreta
   de um processo de Poisson) — na primeira ocorrência, aplica um salto
   permanente. Justificativa: falhas abruptas (ex.: FOD — Foreign Object
   Damage) são eventos raros e memoryless em primeira aproximação.

5. Falha intermitente: pulsos i.i.d. Bernoulli por ciclo, cada um
   independente e não-permanente — modela mau contato / soltura
   intermitente de sensor, diferente da falha abrupta permanente.
"""
from __future__ import annotations

import numpy as np

from aeroquant.sensor_data.domain.entities import SensorReading, Unit
from aeroquant.sensor_data.domain.value_objects import DegradationParams, SensorSchema


class StochasticSensorGenerator:
    def generate_unit(
        self, unit: Unit, schema: SensorSchema, params: DegradationParams
    ) -> list[SensorReading]:
        rng = np.random.default_rng(params.seed)
        n_cycles = unit.max_cycles

        health = self._gamma_degradation_path(n_cycles, params, rng)
        operating_conditions = rng.integers(0, schema.n_operating_conditions, size=n_cycles)

        # Falha abrupta: primeiro ciclo em que o evento Bernoulli ocorre (se ocorrer)
        abrupt_trigger = self._first_bernoulli_hit(n_cycles, params.abrupt_fault_rate, rng)
        # Falha intermitente: máscara booleana, independente por ciclo
        intermittent_mask = rng.random(n_cycles) < params.intermittent_fault_prob

        readings: list[SensorReading] = []
        for t in range(n_cycles):
            cycle = t + 1  # convenção C-MAPSS: ciclos começam em 1
            values: dict[str, float] = {}
            for spec in schema.sensors:
                # Escala do efeito de degradação em função do RANGE VÁLIDO do
                # sensor (valid_max - baseline), não do baseline em si. Usar o
                # baseline como escala causava saturação prematura no teto do
                # range (ex.: sensor_4 achatava aos ~15% da vida útil) sempre
                # que baseline * coupling superava a folga real até valid_max.
                headroom = spec.valid_max - spec.baseline
                drift = spec.baseline * params.drift_rate * t
                degradation_effect = spec.degradation_coupling * headroom * health[t]
                noise = rng.normal(0.0, params.noise_std * abs(spec.baseline))

                abrupt_effect = 0.0
                if abrupt_trigger is not None and t >= abrupt_trigger:
                    abrupt_effect = spec.degradation_coupling * headroom * params.abrupt_fault_magnitude

                intermittent_effect = 0.0
                if intermittent_mask[t]:
                    intermittent_effect = spec.degradation_coupling * headroom * params.intermittent_fault_magnitude

                value = spec.baseline + degradation_effect + drift + noise + abrupt_effect + intermittent_effect
                values[spec.name] = float(np.clip(value, spec.valid_min, spec.valid_max))

            readings.append(
                SensorReading(
                    unit_id=unit.unit_id,
                    cycle=cycle,
                    operating_condition=int(operating_conditions[t]),
                    values=values,
                )
            )
        return readings

    @staticmethod
    def _gamma_degradation_path(
        n_cycles: int, params: DegradationParams, rng: np.random.Generator
    ) -> np.ndarray:
        increments = rng.gamma(shape=params.degradation_shape, scale=params.degradation_scale, size=n_cycles)
        path = np.cumsum(increments)
        if path[-1] == 0:
            return np.zeros(n_cycles)
        return path / path[-1]  # normalizado: health(0) ~ 0, health(max_cycles) = 1.0

    @staticmethod
    def _first_bernoulli_hit(n_cycles: int, rate: float, rng: np.random.Generator) -> int | None:
        if rate <= 0:
            return None
        hits = np.where(rng.random(n_cycles) < rate)[0]
        return int(hits[0]) if len(hits) > 0 else None
