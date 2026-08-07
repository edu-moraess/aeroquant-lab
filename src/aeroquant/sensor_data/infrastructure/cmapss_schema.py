"""
Schema de referência compatível com o C-MAPSS (21 sensores + 3 operational
settings, tratados aqui como parte do schema geral com 6 "condições
operacionais" discretas, como no FD002/FD004).

Baselines e coeficientes de acoplamento com degradação são heurísticos,
calibrados para produzir séries plausíveis — eles SERÃO recalibrados
estatisticamente contra o C-MAPSS real assim que os arquivos forem
enviados (ver Fase 2, H2: o gap sintético-real é justamente o que
queremos medir e reduzir).
"""
from __future__ import annotations

from aeroquant.sensor_data.domain.value_objects import SensorSchema, SensorSpec

# (nome, baseline, faixa_min, faixa_max, acoplamento_com_degradação)
_CMAPSS_LIKE_SENSORS: tuple[tuple[str, float, float, float, float], ...] = (
    ("sensor_1", 518.67, 500.0, 540.0, 0.02),
    ("sensor_2", 642.0, 620.0, 660.0, 0.35),
    ("sensor_3", 1590.0, 1500.0, 1650.0, 0.40),
    ("sensor_4", 1400.0, 1300.0, 1470.0, 0.45),
    ("sensor_5", 14.62, 14.0, 15.2, 0.05),
    ("sensor_6", 21.6, 20.5, 22.0, 0.10),
    ("sensor_7", 553.0, 500.0, 560.0, 0.30),
    ("sensor_8", 2388.0, 2380.0, 2400.0, 0.15),
    ("sensor_9", 9050.0, 9000.0, 9250.0, 0.35),
    ("sensor_10", 1.30, 1.25, 1.35, 0.08),
    ("sensor_11", 47.3, 45.0, 48.5, 0.30),
    ("sensor_12", 522.0, 500.0, 530.0, 0.25),
    ("sensor_13", 2388.0, 2380.0, 2400.0, 0.15),
    ("sensor_14", 8130.0, 8050.0, 8300.0, 0.35),
    ("sensor_15", 8.42, 8.0, 8.7, 0.20),
    ("sensor_16", 0.03, 0.02, 0.04, 0.05),
    ("sensor_17", 393.0, 380.0, 400.0, 0.25),
    ("sensor_18", 2388.0, 2380.0, 2400.0, 0.02),
    ("sensor_19", 100.0, 98.0, 102.0, 0.02),
    ("sensor_20", 38.9, 37.0, 40.0, 0.30),
    ("sensor_21", 23.4, 22.0, 24.5, 0.30),
)


def build_cmapss_like_schema(n_operating_conditions: int = 1) -> SensorSchema:
    """
    LIMITAÇÃO CONHECIDA (documentada, não escondida): o C-MAPSS real
    (subconjuntos FD002/FD004) tem 6 regimes operacionais que DE FATO
    deslocam a distribuição dos sensores. O StochasticSensorGenerator
    atual NÃO modela isso — `operating_condition` é gerado, mas nenhum
    sensor muda de comportamento em função dele. Usar
    n_operating_conditions=6 aqui criaria uma ilusão de suporte
    multi-regime: o Digital Twin acabaria calculando baseline/z-score
    separado por "regime" sem que exista sinal real diferenciando os
    regimes — o efeito prático, medido nesta sessão, foi baseline
    instável por sub-amostragem (poucos pontos por regime), gerando
    Health Index oscilando artificialmente. Por isso o default aqui é 1
    (comportamento condition-agnostic, fiel ao que o gerador realmente
    simula) até que a Fase 4 receba um upgrade para deslocar baseline de
    sensor por operating_condition de verdade — nesse momento, subir
    para 6 volta a fazer sentido.
    """

    specs = tuple(
        SensorSpec(
            name=name,
            unit="raw",
            baseline=baseline,
            valid_min=vmin,
            valid_max=vmax,
            degradation_coupling=coupling,
        )
        for name, baseline, vmin, vmax, coupling in _CMAPSS_LIKE_SENSORS
    )
    return SensorSchema(sensors=specs, n_operating_conditions=n_operating_conditions)
