"""
API mínima (Presentation Context) expondo os use cases já implementados e
testados em sensor_data/ e digital_twin/.

AVISO: fastapi/uvicorn/pydantic NÃO estão instalados neste container (sem
acesso à rede) — este arquivo foi escrito com cuidado sintático mas NÃO foi
executado nem testado. Antes de confiar nele, rode:

    pip install -r requirements/api.txt
    PYTHONPATH=src uvicorn aeroquant.api.main:app --reload

e valide manualmente os endpoints abaixo.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from aeroquant.digital_twin.application.use_cases import UpdateDigitalTwin
from aeroquant.digital_twin.infrastructure.estimators.linear_extrapolation_rul import (
    LinearExtrapolationRULEstimator,
)
from aeroquant.digital_twin.infrastructure.estimators.threshold_calibration import (
    calibrate_failure_threshold,
)
from aeroquant.digital_twin.infrastructure.estimators.welford_fleet_baseline import OnlineFleetBaseline
from aeroquant.digital_twin.infrastructure.estimators.zscore_health_index import ZScoreHealthIndexEstimator
from aeroquant.digital_twin.infrastructure.repositories.in_memory_repository import (
    InMemoryDigitalTwinRepository,
)
from aeroquant.sensor_data.application.use_cases import GenerateSyntheticFleet
from aeroquant.sensor_data.domain.value_objects import DegradationParams
from aeroquant.sensor_data.infrastructure.cmapss_schema import build_cmapss_like_schema
from aeroquant.sensor_data.infrastructure.generators.stochastic_generator import (
    StochasticSensorGenerator,
)
from aeroquant.sensor_data.infrastructure.repositories.csv_repository import CSVSensorRepository

app = FastAPI(
    title="AeroQuant Lab API",
    description="Sensor Data + Digital Twin Context — Fases 1-5. Ver /docs para OpenAPI interativo.",
    version="0.3.0",
)

_SCHEMA = build_cmapss_like_schema()

# Composition root: instâncias compartilhadas do Digital Twin (estado em
# memória — troque por um repositório persistente antes de usar em produção).
_baseline = OnlineFleetBaseline()
_hi_estimator = ZScoreHealthIndexEstimator(_SCHEMA, coupling_threshold=0.2)
_failure_threshold = calibrate_failure_threshold(_SCHEMA, OnlineFleetBaseline(), _hi_estimator)
_digital_twin = UpdateDigitalTwin(
    baseline_tracker=_baseline,
    hi_estimator=_hi_estimator,
    rul_estimator=LinearExtrapolationRULEstimator(),
    repository=InMemoryDigitalTwinRepository(),
)


class GenerateFleetRequest(BaseModel):
    n_units: int = 30
    lifetime_mean: int = 180
    lifetime_std: int = 35
    seed: int | None = None


class IngestReadingRequest(BaseModel):
    unit_id: str
    cycle: int
    operating_condition: int = 0
    sensor_values: dict[str, float]


@app.post("/fleet/generate")
def generate_fleet(req: GenerateFleetRequest) -> dict:
    """Gera uma frota sintética e retorna estatísticas (não persiste em disco)."""
    generator = StochasticSensorGenerator()
    repo = CSVSensorRepository("/tmp/aeroquant_api_fleet.csv")
    use_case = GenerateSyntheticFleet(generator, repo)
    result = use_case.run(
        _SCHEMA, DegradationParams(), req.n_units, req.lifetime_mean, req.lifetime_std, seed=req.seed
    )
    return {
        "n_units": result.n_units,
        "n_readings": result.n_readings,
        "lifetime_mean": result.lifetime_mean,
        "lifetime_std": result.lifetime_std,
    }


@app.post("/digital-twin/ingest")
def ingest_reading(req: IngestReadingRequest) -> dict:
    """Alimenta uma leitura no Digital Twin e retorna o snapshot atualizado."""
    snapshot = _digital_twin.ingest(
        req.unit_id, req.cycle, req.operating_condition, req.sensor_values, _failure_threshold
    )
    return {
        "cycle": snapshot.cycle,
        "health_index": snapshot.health_index,
        "rul_point": snapshot.rul.point,
        "rul_lower": snapshot.rul.lower,
        "rul_upper": snapshot.rul.upper,
        "is_anomaly": snapshot.is_anomaly,
        "anomaly_reason": snapshot.anomaly_reason,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
