from datetime import date
from pathlib import Path
import sys

# Asegurar que la raíz del proyecto esté en sys.path para importar predict
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from predict import predict_daily_risk_probabilities


def predict_by_date(year: int, month: int, day: int, lat: float = 28.1901, lon: float = -105.4701, **kwargs) -> dict:
    """
    Predice probabilidades climáticas utilizando el modelo de deep learning pre-entrenado.
    Recibe el año, mes, día y coordenadas opcionales.
    Retorna el pronóstico probabilístico diario completo generado por el modelo.
    """
    target_date_str = f"{year:04d}-{month:02d}-{day:02d}"
    return predict_daily_risk_probabilities(lat=lat, lon=lon, target_date_str=target_date_str)


def predict_weather(lat: float, lon: float, target_date_str: str) -> dict:
    """
    Función de acceso directo a la predicción probabilística de riesgo climático.
    """
    return predict_daily_risk_probabilities(lat=lat, lon=lon, target_date_str=target_date_str)
