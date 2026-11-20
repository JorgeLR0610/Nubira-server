import logging
import os
from datetime import date
from threading import BoundedSemaphore
from zipfile import is_zipfile

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from config import BEST_MODEL_PATH, CONTEXT_PATH

logger = logging.getLogger(__name__)
inference_slots = BoundedSemaphore(max(1, int(os.environ.get("NUBIRA_INFERENCE_CONCURRENCY", "2"))))


class ClimaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latitude: float = Field(ge=-90, le=90, strict=True)
    longitude: float = Field(ge=-180, le=180, strict=True)
    day: int = Field(ge=1, le=31, strict=True)
    month: int = Field(ge=1, le=12, strict=True)
    year: int = Field(ge=1, le=9999, strict=True)

    def future_date(self) -> date:
        try:
            target = date(self.year, self.month, self.day)
        except ValueError as exc:
            raise HTTPException(422, "La fecha no es válida.") from exc
        if target <= date.today():
            raise HTTPException(422, "Elige una fecha posterior a hoy.")
        return target


class TemperatureThresholds(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    calor_extremo_desde: float
    frio_extremo_hasta: float


class Prediction(BaseModel):
    latitud: float = Field(ge=-90, le=90)
    longitud: float = Field(ge=-180, le=180)
    fecha_objetivo: date
    dias_hacia_el_futuro: int = Field(gt=0)
    probabilidad_lluvia_pct: float = Field(ge=0, le=100)
    probabilidad_viento_fuerte_pct: float = Field(ge=0, le=100)
    probabilidad_temperaturas_adversas_pct: float = Field(ge=0, le=100)
    referencia_umbral_adversa_local_c: TemperatureThresholds


PROBABILITIES = {
    "probabilidad_lluvia_pct": "Lluvia",
    "probabilidad_viento_fuerte_pct": "Viento fuerte",
    "probabilidad_temperaturas_adversas_pct": "Temperatura adversa",
}


def artifacts_ready() -> bool:
    try:
        return all(path.stat().st_size > 1024 and is_zipfile(path) for path in (BEST_MODEL_PATH, CONTEXT_PATH))
    except OSError:
        return False


def evaluate(latitude: float, longitude: float, target_date: date) -> Prediction:
    if not artifacts_ready():
        raise HTTPException(503, "El servicio de evaluación no está disponible.")
    if not inference_slots.acquire(blocking=False):
        raise HTTPException(503, "El servicio está ocupado. Intenta nuevamente.", headers={"Retry-After": "3"})
    try:
        from predict import predict_daily_risk_probabilities

        return Prediction.model_validate(predict_daily_risk_probabilities(latitude, longitude, target_date.isoformat()))
    except Exception as exc:
        logger.exception("La inferencia no pudo completarse.")
        raise HTTPException(503, "El servicio de evaluación no está disponible.") from exc
    finally:
        inference_slots.release()
